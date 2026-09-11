import csv
import io
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from expense_extract.types import ExpenseLine
from launcher.expense_review import ExpenseReview
from launcher.errors import Conflict


@pytest.fixture
def review(tmp_path):
    images = [tmp_path / f"receipt-{i}.png" for i in range(3)]
    for i, path in enumerate(images):
        path.write_bytes(f"image {i}".encode())
    store = ExpenseReview(tmp_path / "review.json")
    store.begin(images)
    for i, path in enumerate(images):
        store.add(ExpenseLine(path.name, "Demo Hotel", "2026-09-05", Decimal("426"),
                              "Lodging", "OCR original", currency="EUR" if i < 2 else "USD"))
    store.finish("complete")
    return store


def save(store, index=0, status="approved", note="", **edits):
    snap = store.snapshot()
    item = snap["items"][index]
    return store.update(snap["batch_id"], item["id"], snap["revision"],
                        {**item["fields"], **edits}, status, note)


def test_approval_duplicates_exclusion_and_separate_currency_totals(review):
    assert review.snapshot()["totals"] == {}
    with pytest.raises(ValueError, match="duplicate"):
        save(review)
    save(review, 1, status="excluded")
    save(review)
    snap = save(review, 2, amount="-18.00")
    assert snap["totals"] == {"EUR": "426.00", "USD": "-18.00"}
    exported = list(csv.DictReader(io.StringIO(review.export(snap["batch_id"], snap["revision"]).lstrip("\ufeff"))))
    assert len(exported) == 2 and exported[1]["amount"] == "-18.00"
    restored = ExpenseReview(review.path).snapshot()
    assert restored["totals"] == snap["totals"]
    assert restored["items"][2]["original"]["amount"] == "426"


@pytest.mark.parametrize("edits", [{"amount": "NaN"}, {"amount": "1,234"},
    {"amount": "1e3"}, {"amount": "1.001"}, {"currency": "JPY", "amount": "1.50"},
    {"currency": "XYZ"}, {"date": "2026-02-30"}, {"vendor": ""}, {"category": "Bad"}])
def test_invalid_approval_does_not_modify_saved_review(review, edits):
    before = review.snapshot()
    with pytest.raises(ValueError):
        save(review, note="Separate expense checked", **edits)
    assert review.snapshot() == before


def test_new_duplicate_reopens_prior_approval(review):
    save(review, 1, status="excluded")
    save(review)
    snap = save(review, 1, status="pending")
    assert snap["items"][0]["status"] == "pending"
    assert not snap["totals"]


def test_stale_revision_and_changed_source_are_rejected(review):
    snap = review.snapshot()
    save(review, status="pending", vendor="Edited")
    with pytest.raises(Conflict):
        review.update(snap["batch_id"], snap["items"][0]["id"], snap["revision"], {}, "pending", "")
    with pytest.raises(Conflict):
        review.export("old-batch", snap["revision"])
    (review.path.parent / "receipt-0.png").write_bytes(b"changed")
    with pytest.raises(Conflict, match="source image changed"):
        save(review)


def test_csv_neutralizes_formulas_and_retains_numeric_refunds(review):
    snap = save(review, 2, vendor="=HYPERLINK(bad)", amount="-18", note="@formula")
    content = review.export(snap["batch_id"], snap["revision"])
    assert "'=HYPERLINK(bad)" in content and "'@formula" in content and ",-18.00," in content


def test_exact_image_duplicates_even_when_fields_differ(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"same"); b.write_bytes(b"same")
    store = ExpenseReview(tmp_path / "review.json")
    store.begin([a, b])
    assert store.snapshot()["items"][0]["duplicates"] == ["b.png"]
    assert ExpenseReview(store.path).snapshot()["phase"] == "interrupted"
    with pytest.raises(Conflict):
        save(store, status="pending")


def test_review_api_contract_and_image_access(review, monkeypatch):
    from launcher.app import app, expense_extract_runner
    monkeypatch.setattr(expense_extract_runner, "review", review)
    client = TestClient(app)
    snap = client.get("/api/expense-extract/review").json()
    item = snap["items"][2]
    assert "path" not in item and "hash" not in item
    assert client.get(item["image_url"]).content == b"image 2"
    assert client.get(item["image_url"].replace(item["id"], "unknown")).status_code == 400
    res = client.post(f'/api/expense-extract/review/{item["id"]}', json={
        "batch_id": snap["batch_id"], "revision": snap["revision"], "fields": item["fields"], "status": "approved"})
    assert res.status_code == 200
    export = client.get(f'/api/expense-extract/review/{snap["batch_id"]}/export', params={"revision": res.json()["revision"]})
    assert export.status_code == 200 and "text/csv" in export.headers["content-type"]
    assert "reviewed-expenses.csv" in export.headers["content-disposition"]


def test_runner_persists_results_without_a_websocket(tmp_path, monkeypatch):
    import asyncio
    from launcher.expense_extract_runner import ExpenseExtractRunner
    from launcher import expense_extract_runner as module
    from pantherlake_ai_core.engine import Engine
    image = tmp_path / "receipt.png"
    image.write_bytes(b"receipt")
    runner = ExpenseExtractRunner()
    runner.review = ExpenseReview(tmp_path / "review.json")
    def fake_run(**kwargs):
        line = ExpenseLine(image.name, "Demo", "2026-09-05", Decimal("12.50"), "Meals", "original OCR", currency="EUR")
        kwargs["on_structured"](line)
        return [line]
    monkeypatch.setattr(module.pipeline, "run", fake_run)
    async def exercise():
        runner.start(loop=asyncio.get_running_loop(), queue=asyncio.Queue(), folder=str(tmp_path),
                     ocr_engine=Engine.PORTABLE, ocr_device="cpu", llm_engine=Engine.PORTABLE, llm_device="cpu")
        await asyncio.to_thread(runner._thread.join, 5)
        assert not runner.running
    asyncio.run(exercise())
    snapshot = ExpenseReview(runner.review.path).snapshot()
    assert snapshot["phase"] == "complete"
    assert snapshot["items"][0]["fields"]["amount"] == "12.50"
    assert snapshot["items"][0]["status"] == "pending"
