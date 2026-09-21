"""The model inventory stays in step with the bricks that load the models,
and the prefetcher reports what a show needs to know (BACKLOG R10)."""
from __future__ import annotations

import threading
import time
from fnmatch import fnmatch

import pytest
from launcher.registry import REGISTRY
from pantherlake_ai_core import models
from pantherlake_ai_core.prefetch import Prefetcher, humanize_bytes, humanize_seconds


def _spec(repo_id):
    return next(spec for spec in models.MODELS if spec.repo_id == repo_id)


def _covers(spec, filename):
    return not spec.patterns or any(fnmatch(filename, pattern) for pattern in spec.patterns)


def test_every_demo_named_in_the_inventory_exists():
    known = {demo.id for demo in REGISTRY}
    for spec in models.MODELS:
        assert set(spec.demos) <= known, f"{spec.key} names a demo that isn't in the registry"
        assert spec.demos, f"{spec.key} belongs to no demo"


def test_keys_are_unique_and_the_lookup_matches():
    assert len(models.BY_KEY) == len(models.MODELS)
    assert all(models.BY_KEY[spec.key] is spec for spec in models.MODELS)


def test_the_openvino_models_the_bricks_name_are_all_listed():
    from code_review_assist import session as code_review
    from doc_qa import embedder_openvino, llm_openvino
    from live_translation import transcriber_openvino
    from object_detection import detector_openvino
    from screen_ocr import extractor_openvino

    listed = {spec.repo_id for spec in models.MODELS}
    assert transcriber_openvino._DEFAULT_REPO_TEMPLATE.format(size="base") in listed
    assert llm_openvino._DEFAULT_REPO in listed
    assert embedder_openvino._DEFAULT_REPO in listed
    assert extractor_openvino._DEFAULT_REPO in listed
    assert code_review._DEFAULT_OPENVINO_REPO in listed
    assert code_review._DEFAULT_PORTABLE_REPO in listed
    assert detector_openvino._DEFAULT_REPO in listed
    # ...and the files that repo actually loads, not the whole repository.
    for name in detector_openvino._DEFAULT_FILES:
        assert _covers(_spec(detector_openvino._DEFAULT_REPO), name)


def test_the_portable_models_the_bricks_name_are_all_listed():
    from doc_qa import embedder_portable, llm_portable
    from object_detection import detector_portable
    from webcam_effects import matte

    listed = {spec.repo_id for spec in models.MODELS}
    for module, repo_attr, file_attr in (
        (llm_portable, "_DEFAULT_REPO", "_DEFAULT_FILENAME"),
        (embedder_portable, "_DEFAULT_REPO", "_DEFAULT_FILENAME"),
        (detector_portable, "_DEFAULT_REPO", "_DEFAULT_FILENAME"),
        (matte, "DEFAULT_REPO", "DEFAULT_FILENAME"),
    ):
        repo = getattr(module, repo_attr)
        assert repo in listed, f"{repo} is loaded by a brick but missing from the inventory"
        assert _covers(_spec(repo), getattr(module, file_attr))


def test_the_voice_cloning_files_are_covered():
    from voice_clone_studio import chatterbox_model

    chatterbox = _spec(chatterbox_model.REPO_ID)
    graphs = chatterbox_model._graph_files()
    for path in [*graphs.values(), *(f"{p}_data" for p in graphs.values()), "tokenizer.json"]:
        assert _covers(chatterbox, path), f"{path} would not be prefetched"

    torch = pytest.importorskip("torch")  # voice_model imports it at module level
    assert torch
    from voice_clone_studio import voice_model

    openvoice = _spec(voice_model.REPO_ID)
    for path in voice_model._CHECKPOINT_FILES.values():
        assert _covers(openvoice, path), f"{path} would not be prefetched"


# --- the prefetcher ------------------------------------------------------------------


@pytest.fixture
def fake_hub(monkeypatch):
    """A machine where nothing is cached and every model is 100 bytes."""
    state = {"cached": set(), "bytes": {}, "downloaded": []}
    monkeypatch.setattr(models, "is_cached", lambda spec: spec.key in state["cached"])
    monkeypatch.setattr(models, "cached_bytes", lambda spec: state["bytes"].get(spec.key, 0))
    # None for the two models a library fetches itself, as the real one does.
    monkeypatch.setattr(models, "remote_size", lambda spec: 100 if spec.repo_id else None)

    def download(spec, on_progress=None):
        state["downloaded"].append(spec.key)
        state["cached"].add(spec.key)
        state["bytes"][spec.key] = 100
        if on_progress:
            on_progress(100)  # the client reports absolute bytes for this model

    monkeypatch.setattr(models, "download", download)
    return state


def _settled(prefetcher, timeout=10.0):
    deadline = time.monotonic() + timeout
    while prefetcher.running and time.monotonic() < deadline:
        time.sleep(0.05)
    return prefetcher.status()


def test_status_says_what_is_missing_and_what_it_weighs(fake_hub):
    prefetcher = Prefetcher()
    prefetcher.sizes_in_background()
    for _ in range(40):
        if prefetcher.status()["sizes_known"]:
            break
        time.sleep(0.05)
    status = prefetcher.status()
    assert status["ready_count"] == 0
    assert status["total_count"] == len(models.MODELS)
    hub_models = [spec for spec in models.MODELS if spec.repo_id]
    assert status["missing_bytes"] == 100 * len(hub_models)
    assert all(entry["state"] == "missing" for entry in status["models"])


def test_a_run_downloads_everything_missing_and_ends_ready(fake_hub):
    prefetcher = Prefetcher()
    started = prefetcher.start()
    assert started == [spec.key for spec in models.MODELS]
    status = _settled(prefetcher)
    assert fake_hub["downloaded"] == started
    assert status["ready_count"] == len(models.MODELS)
    assert status["running"] is False and status["errors"] == {}


def test_one_model_failing_does_not_stop_the_others(fake_hub, monkeypatch):
    real_download = models.download

    def download(spec, on_progress=None):
        if spec.key == "vlm-7b-ov":
            raise RuntimeError("connection reset")
        real_download(spec, on_progress)

    monkeypatch.setattr(models, "download", download)
    prefetcher = Prefetcher()
    prefetcher.start()
    status = _settled(prefetcher)
    assert status["errors"] == {"vlm-7b-ov": "connection reset"}
    assert any(entry["state"] == "failed" for entry in status["models"])
    assert status["ready_count"] == len(models.MODELS) - 1


def test_stopping_leaves_the_rest_untouched(fake_hub, monkeypatch):
    started = threading.Event()
    real_download = models.download

    def download(spec, on_progress=None):
        started.set()
        time.sleep(0.2)
        real_download(spec, on_progress)

    monkeypatch.setattr(models, "download", download)
    prefetcher = Prefetcher()
    prefetcher.start()
    started.wait(5)
    prefetcher.stop()
    status = _settled(prefetcher)
    assert status["stopped"] is True
    assert len(fake_hub["downloaded"]) < len(models.MODELS)


def test_a_second_run_is_refused_while_one_is_going(fake_hub, monkeypatch):
    monkeypatch.setattr(models, "download", lambda spec, on_progress=None: time.sleep(0.3))
    prefetcher = Prefetcher()
    prefetcher.start()
    with pytest.raises(RuntimeError, match="already running"):
        prefetcher.start()
    prefetcher.stop()
    _settled(prefetcher)


def test_unknown_keys_and_nothing_to_do_are_told_apart(fake_hub):
    prefetcher = Prefetcher()
    with pytest.raises(ValueError, match="Unknown model"):
        prefetcher.start(["not-a-model"])
    fake_hub["cached"].update(spec.key for spec in models.MODELS)
    with pytest.raises(RuntimeError, match="already downloaded"):
        prefetcher.start()


def test_sizes_and_durations_read_as_english():
    assert humanize_bytes(None) == "unknown"
    assert humanize_bytes(512) == "512 B"
    assert humanize_bytes(16_900_000_000) == "15.7 GB"
    assert humanize_seconds(None) == "--"
    assert humanize_seconds(45) == "45 s"
    assert humanize_seconds(3 * 60 + 5) == "3 min 05 s"
    assert humanize_seconds(3700) == "1 h 01 min"


# --- the launcher's routes -----------------------------------------------------------


def test_routes_report_and_refuse_sensibly(fake_hub, monkeypatch):
    from fastapi.testclient import TestClient
    from launcher import app as launcher_app
    from launcher import model_routes

    client = TestClient(launcher_app.app)
    body = client.get("/api/models").json()
    assert len(body["models"]) == len(models.MODELS) and body["running"] is False

    assert client.post("/api/models/prefetch", json={"keys": ["nope"]}).status_code == 400
    monkeypatch.setattr(model_routes.prefetcher, "start", lambda keys: ["yolo11s-ov"])
    accepted = client.post("/api/models/prefetch", json={"keys": ["yolo11s-ov"]})
    assert accepted.status_code == 202 and accepted.json() == {"started": ["yolo11s-ov"]}

    def busy(keys):
        raise RuntimeError("A download is already running.")

    monkeypatch.setattr(model_routes.prefetcher, "start", busy)
    assert client.post("/api/models/prefetch", json={}).status_code == 409
    assert client.post("/api/models/prefetch/stop").json() == {"stopping": True}


def test_bytes_on_disk_are_counted_in_either_cache_layout(tmp_path, monkeypatch):
    """Hub 1.x writes into snapshots/ and has no blobs/; older hubs do the
    reverse and link snapshots/ at blobs/. Counting one layout only reported
    zero on the other, which left the progress bar at 0% throughout."""
    monkeypatch.setattr(models, "_hub_cache", lambda: tmp_path)
    spec = models.BY_KEY["detr-portable"]
    folder = tmp_path / "models--Xenova--detr-resnet-50"
    (folder / "snapshots" / "abc" / "onnx").mkdir(parents=True)
    (folder / "snapshots" / "abc" / "onnx" / "model.onnx").write_bytes(b"x" * 2048)
    assert models.cached_bytes(spec) == 2048

    blobs = folder / "blobs"
    blobs.mkdir()
    (blobs / "sha").write_bytes(b"y" * 1024)
    (blobs / "sha.incomplete").write_bytes(b"z" * 512)  # a download in flight counts too
    assert models.cached_bytes(spec) == 1536


def test_progress_follows_what_the_client_reports(fake_hub, monkeypatch):
    """The Xet backend assembles a file elsewhere and only puts it in place at
    the end, so progress has to come from the client's own byte reports --
    watching the cache folder would sit at zero for the whole download."""
    monkeypatch.setattr(models, "cached_bytes", lambda spec: 0)  # nothing visible on disk yet
    seen = []

    def download(spec, on_progress=None):
        for position in (25, 50, 75, 100):  # absolute, as the client reports it
            on_progress(position)
        seen.append(spec.key)
        fake_hub["cached"].add(spec.key)

    monkeypatch.setattr(models, "download", download)
    prefetcher = Prefetcher()
    prefetcher.start(["yolo11s-ov"])
    _settled(prefetcher)
    assert seen == ["yolo11s-ov"]
    assert prefetcher._run.completed_bytes == 100
