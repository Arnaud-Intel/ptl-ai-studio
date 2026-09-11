from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pantherlake_ai_core.demo_samples import SAMPLE_ROOT, load_samples
from launcher.demo_assets import asset, enrich_sample, guide
from launcher.registry import REGISTRY
from launcher.app import app
from doc_qa.documents import load_documents
from html_creator.folder_input import MAX_DOCUMENT_CHARS, read_documents


def test_every_available_brick_has_an_actionable_guide():
    for demo in REGISTRY:
        if demo.status == "available":
            content = guide(demo.id)
            assert content and content["title"] and len(content["text"]) > 80, demo.id


def test_all_catalog_assets_exist_and_are_served_locally():
    catalog = json.loads((SAMPLE_ROOT / "catalog.json").read_text(encoding="utf-8"))
    client = TestClient(app)
    for demo_id in catalog:
        for sample in load_samples(demo_id):
            payload = enrich_sample(asdict(sample))
            for item in payload["assets"]:
                assert item["url"].startswith("/demo-assets/")
                assert client.get(item["url"]).status_code == 200, item
            if sample.folder:
                assert Path(sample.folder).is_dir()
    assert client.get("/demo-assets/../VERSION").status_code == 404


def test_demo_links_cannot_escape_sample_root():
    with pytest.raises(ValueError):
        asset("../VERSION")


def test_expense_fixture_totals_and_pack_counts():
    truth = json.loads((SAMPLE_ROOT / "expenses/expected-results.json").read_text(encoding="utf-8"))
    totals = {}
    for receipt in truth["receipts"]:
        assert sum(Decimal(value) for _, value in receipt["items"]) == Decimal(receipt["amount"])
        currency = receipt["currency"]
        totals[currency] = totals.get(currency, Decimal(0)) + Decimal(receipt["amount"])
    assert totals == {key: Decimal(value) for key, value in truth["currency_totals"].items()}
    assert [len(list(Path(s.folder).glob("*.png"))) for s in load_samples("expense-extract")] == [3, 9, 3]
    quick = {p.name for p in (SAMPLE_ROOT / "expenses/quick-start").glob("*.png")}
    assert sum(Decimal(r["amount"]) for r in truth["receipts"] if r["file"] in quick) == Decimal("215.10")
    assert len(truth["review_cases"]) == 3


def test_document_pack_is_fictional_and_fits_html_context():
    folder = SAMPLE_ROOT / "meridian-rollout-2026"
    docs = load_documents(folder)
    assert len(docs) == 6
    assert all("FICTIONAL DEMO DATA" in text for _, text in docs)
    assert len(read_documents(str(folder))) < MAX_DOCUMENT_CHARS
    assert not any("expected-results" in name for name, _ in docs)


def test_review_samples_switch_to_diff_mode_and_ocr_has_images():
    for sample in load_samples("code-review-assist"):
        assert sample.source == "diff_text" and sample.diff_text.startswith("diff --git")
    for sample in load_samples("screen-ocr"):
        assert sample.source == "sample"
        assert enrich_sample(asdict(sample))["image_url"].endswith(".png")
