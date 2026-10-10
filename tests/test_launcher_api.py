"""The launcher's API contract, with every hardware probe monkeypatched so
this runs on a machine (or a CI runner) with no microphone, camera, or
OpenVINO device -- and no model is ever loaded."""
from __future__ import annotations

import threading
import time

import pytest
from code_review_assist.types import ReviewResult
from fastapi.testclient import TestClient
from object_detection import pipeline as object_detection_pipeline
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import GenerationStats

from launcher import app as launcher_app
from launcher.errors import Conflict
from launcher.registry import REGISTRY

FAKE_OPENVINO_DEVICES = ["CPU", "GPU.0", "NPU"]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher_app.events, "LOG_FILE", tmp_path / "events.log")
    monkeypatch.setattr(launcher_app.audio, "list_microphones", lambda: ["Test Mic"])
    monkeypatch.setattr(launcher_app.audio, "list_speakers", lambda: ["Test Speakers"])
    monkeypatch.setattr(launcher_app.video, "list_cameras", lambda: [0])
    monkeypatch.setattr(launcher_app.video, "list_screens", lambda: [{"index": 1, "width": 1920, "height": 1080}])
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: list(FAKE_OPENVINO_DEVICES))
    monkeypatch.setattr(launcher_app, "list_gpu_devices", lambda: [])
    # No lifespan (no telemetry poller thread): none of these routes need it.
    return TestClient(launcher_app.app)


# --- pure helpers -----------------------------------------------------------------


def test_hex_to_bgr():
    assert launcher_app._hex_to_bgr("#0068B5") == (181, 104, 0)
    assert launcher_app._hex_to_bgr("ff0000") == (0, 0, 255)


@pytest.mark.parametrize("bad", ["", "#12345", "#gggggg", "blue"])
def test_hex_to_bgr_rejects_anything_but_rrggbb(bad):
    with pytest.raises(ValueError):
        launcher_app._hex_to_bgr(bad)


def test_resolve_applies_the_cli_rule(monkeypatch):
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: FAKE_OPENVINO_DEVICES)
    monkeypatch.setattr(launcher_app, "resolve_engine", lambda explicit: Engine(explicit) if explicit else Engine.OPENVINO)
    monkeypatch.setattr(launcher_app, "preferred_large_model_device", lambda: "GPU.1")
    monkeypatch.setattr(launcher_app, "preferred_device", lambda: "GPU.0")
    # Nothing chosen is a real chip now, not "AUTO" (see test_side_panel.py).
    assert launcher_app.resolve(None, None) == (Engine.OPENVINO, "GPU.0")
    assert launcher_app.resolve("portable", None) == (Engine.PORTABLE, "cpu")
    assert launcher_app.resolve("openvino", "NPU") == (Engine.OPENVINO, "NPU")
    assert launcher_app.resolve(None, None, large_model=True) == (Engine.OPENVINO, "GPU.1")
    assert launcher_app.resolve("portable", None, large_model=True) == (Engine.PORTABLE, "cpu")


def test_a_brick_set_to_a_lost_npu_runs_where_its_models_moved_to(client, monkeypatch):
    from pantherlake_ai_core import npu

    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: FAKE_OPENVINO_DEVICES)
    monkeypatch.setattr(launcher_app, "resolve_engine", lambda explicit: Engine.OPENVINO)
    monkeypatch.setattr(npu, "fallback_device", lambda: "GPU.0")
    npu._reset_for_tests()
    try:
        assert launcher_app.resolve("openvino", "NPU") == (Engine.OPENVINO, "NPU")
        assert client.get("/api/telemetry").json()["npu_lost"] is None
        with pytest.raises(npu.NpuLost):
            with npu.guard("NPU"):  # Windows resets the chip under some brick
                raise RuntimeError("L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST")
        # From then on nothing is started on it, whatever the page still has selected...
        assert launcher_app.resolve("openvino", "NPU") == (Engine.OPENVINO, "GPU.0")
        assert launcher_app.resolve("openvino", "CPU") == (Engine.OPENVINO, "CPU")
        # ...and the hardware panel is told, so it can say so under the NPU.
        lost = client.get("/api/telemetry").json()["npu_lost"]
        assert lost["moved_to"] == "GPU.0" and lost["at"] > 0
    finally:
        npu._reset_for_tests()


@pytest.mark.parametrize(
    "exc, status",
    [
        (Conflict("busy"), 409),
        (ValueError("bad"), 400),
        (FileNotFoundError("missing"), 400),
        (RuntimeError("the model blew up"), 500),
        (KeyError("x"), 500),
    ],
)
def test_error_policy(exc, status):
    response = launcher_app.error_response(exc)
    assert response.status_code == status
    assert b'"error"' in response.body


# --- routes ------------------------------------------------------------------------


def test_demos_is_the_registry(client):
    demos = client.get("/api/demos").json()
    assert [d["id"] for d in demos] == [d.id for d in REGISTRY]
    assert {"id", "name", "status", "devices", "samples", "large_model"} <= set(demos[0])


def test_every_available_demo_has_a_devices_route(client):
    for demo in REGISTRY:
        response = client.get(f"/api/{demo.id}/devices")
        if demo.status != "available":
            assert response.status_code == 404, demo.id
            continue
        assert response.status_code == 200, demo.id
        payload = response.json()
        assert payload["openvino_devices"] == FAKE_OPENVINO_DEVICES, demo.id
        for kind in demo.devices:
            assert kind in payload, (demo.id, kind)
        assert ("samples" in payload) == bool(demo.samples), demo.id


def test_unknown_demo_is_a_404(client):
    assert client.get("/api/not-a-brick/devices").status_code == 404


def test_unavailable_device_rejected_before_work(client):
    response = client.post("/api/doc-qa/ingest", json={"folder": "x", "engine": "openvino", "compute_device": "GPU.99"})
    assert response.status_code == 400 and "unavailable" in response.json()["error"]


def test_webcam_gpu_is_disabled_in_controls_and_api(client):
    devices = client.get("/api/webcam-effects/devices").json()
    assert {"AUTO", "GPU.0"} <= devices["openvino_unsupported"].keys()
    for device in ("AUTO", "GPU.0"):
        response = client.post("/api/webcam-effects/start", json={"engine": "openvino", "compute_device": device})
        assert response.status_code == 400 and "temporarily unavailable" in response.json()["error"]


def test_status_version_and_logs(client):
    assert isinstance(client.get("/api/status").json(), dict)
    assert client.get("/api/version").json()["version"]
    assert isinstance(client.get("/api/logs").json(), list)


def test_an_unknown_engine_is_a_400(client):
    response = client.post("/api/doc-qa/ingest", json={"folder": "x", "engine": "bogus"})
    assert response.status_code == 400
    assert "bogus" in response.json()["error"]


def test_a_malformed_body_is_a_400_in_the_same_shape(client):
    response = client.post("/api/doc-qa/ask", json={"question": 1, "top_k": "x"})
    assert response.status_code == 400
    assert response.json()["error"].startswith("invalid request")


def test_doing_things_out_of_order_is_a_409(client):
    assert client.post("/api/doc-qa/ask", json={"question": "hi"}).status_code == 409
    assert client.post("/api/meeting-notes/generate").status_code == 409
    assert client.post("/api/voice-clone-studio/synthesize", json={"text": "hi"}).status_code == 409


def test_a_bad_color_is_a_400(client):
    response = client.post("/api/webcam-effects/effect", json={"effect": "blur", "color": "nope"})
    assert response.status_code == 400


def test_video_streams_404_when_the_brick_is_not_running(client):
    for path in ("/api/object-detection/stream", "/api/webcam-effects/stream", "/api/smart-city-monitor/stream?feed=feed-1"):
        assert client.get(path).status_code == 404, path


class FakeStreamRunner:
    """Just enough of a runner for the start/stop routes: no thread, no camera."""

    error = None

    def __init__(self) -> None:
        self.running = False
        self.calls: list[dict] = []

    def start(self, **kwargs) -> None:
        if self.running:
            raise Conflict("object-detection is already running")
        self.calls.append(kwargs)
        self.running = True

    def stop(self) -> None:
        self.running = False

    def latest_jpeg(self):
        return None

    def latest_detections(self) -> list:
        return []


def test_a_brick_too_busy_to_stop_says_stopping_until_it_actually_does(client, monkeypatch):
    """The real runner, with a pipeline that ignores the stop event -- the
    shape of a brick inside a model load or one long inference. Until it
    returns, the launcher has to say "stopping" rather than go on
    reporting the brick's normal running message."""
    release = threading.Event()

    def stuck_pipeline(**kwargs):
        release.wait(timeout=10)

    # The runner calls pipeline.run() on the shared module object, so
    # patching it here is what its thread will actually execute.
    monkeypatch.setattr(object_detection_pipeline, "run", stuck_pipeline)

    try:
        assert client.post(
            "/api/object-detection/start",
            json={"source": "screen", "engine": "portable", "compute_device": "cpu"},
        ).status_code == 200

        assert client.post("/api/object-detection/stop").status_code == 200
        assert client.get("/api/status").json()["object-detection"]["phase"] == "stopping"
        # It genuinely is still running, so a Start now is refused -- and
        # says which kind of busy it is, since this one clears on its own.
        refused = client.post(
            "/api/object-detection/start",
            json={"source": "screen", "engine": "portable", "compute_device": "cpu"},
        )
        assert refused.status_code == 409
        assert "still stopping" in refused.json()["error"]
    finally:
        release.set()

    deadline = time.time() + 5
    while time.time() < deadline and "object-detection" in client.get("/api/status").json():
        time.sleep(0.05)
    assert "object-detection" not in client.get("/api/status").json()
    assert not launcher_app.object_detection_runner.running


def test_starting_twice_is_a_409_and_stop_resets(client, monkeypatch):
    fake = FakeStreamRunner()
    monkeypatch.setattr(launcher_app, "object_detection_runner", fake)
    body = {"source": "screen", "engine": "portable", "compute_device": "cpu"}

    assert client.post("/api/object-detection/start", json=body).json() == {"status": "started"}
    assert fake.calls[0]["engine"] is Engine.PORTABLE
    assert fake.calls[0]["compute_device"] == "cpu"

    again = client.post("/api/object-detection/start", json=body)
    assert again.status_code == 409
    assert "already running" in again.json()["error"]

    assert client.post("/api/object-detection/stop").json() == {"status": "stopped"}
    assert client.post("/api/object-detection/start", json=body).status_code == 200


def test_a_feed_with_no_source_is_rejected_by_name(client):
    """A blank feed is the user's mistake, not a server error -- and the
    message has to say *which* feed, since there can be several."""
    res = client.post(
        "/api/smart-city-monitor/start",
        json={"feeds": [{"path": "clip.mp4"}, {"path": "   "}]},
    )
    assert res.status_code == 400
    assert "feed 2" in res.json()["error"].lower()


def test_an_unknown_per_feed_engine_is_a_400(client):
    res = client.post(
        "/api/smart-city-monitor/start",
        json={"feeds": [{"path": "clip.mp4", "engine": "not-an-engine"}]},
    )
    assert res.status_code == 400


def test_an_upload_without_a_filename_is_a_400(client):
    res = client.post("/api/smart-city-monitor/upload", files={"file": ("", b"x")})
    assert res.status_code in (400, 422)


def test_the_version_reported_is_the_one_running_not_the_one_on_disk(client, monkeypatch):
    """A launcher left running while the repo moves on under it used to
    report whatever VERSION said, so the page claimed to be the newest code
    while serving the oldest -- and the static files, read per request,
    made the UI agree with the claim. It now reports what it started with,
    and says a restart is pending."""
    from launcher import app as app_module

    monkeypatch.setattr(app_module, "RUNNING_VERSION", "0.1.0")
    monkeypatch.setattr(app_module, "read_version_file", lambda: "9.9.9")
    body = client.get("/api/version").json()
    assert body["version"] == "0.1.0"
    assert body["on_disk"] == "9.9.9"
    assert body["restart_needed"] is True


def test_no_restart_is_claimed_when_the_process_matches_the_disk(client, monkeypatch):
    from launcher import app as app_module

    monkeypatch.setattr(app_module, "RUNNING_VERSION", "1.2.3")
    monkeypatch.setattr(app_module, "read_version_file", lambda: "1.2.3")
    body = client.get("/api/version").json()
    assert body["version"] == "1.2.3"
    assert body["restart_needed"] is False


def test_a_taken_port_is_reported_as_already_running(monkeypatch):
    """Starting a second copy used to open a browser tab -- at the copy
    already running -- and only then die on the bind, so a restart looked
    like it had worked while serving the old code."""
    import socket as socket_mod

    from launcher import app as app_module

    with socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        # a backlog big enough for both probes: nothing accepts them here,
        # so with listen(1) the second connect would be refused and the
        # test, not the code, would be what failed
        taken.listen(8)
        port = taken.getsockname()[1]
        assert app_module.is_already_serving("127.0.0.1", port) is True
        # 0.0.0.0 is asked about over the loopback it would actually serve
        assert app_module.is_already_serving("0.0.0.0", port) is True

    # and once nothing is listening there, it is free again
    assert app_module.is_already_serving("127.0.0.1", port) is False


def test_voice_clone_defaults_to_the_better_cloning_model(client):
    body = client.get("/api/voice-clone-studio/status").json()
    assert set(body) >= {"enrolled", "model", "supports_styles"}


def test_an_unknown_voice_model_is_a_400(client):
    res = client.post("/api/voice-clone-studio/enroll-record", json={"seconds": 1, "model": "bogus"})
    assert res.status_code == 400
    assert "bogus" in res.json()["error"]


def test_large_model_badges_state_a_measured_fact():
    """R20: no card claims a discrete GPU is required -- a large model says
    what it takes instead, and where that was measured."""
    badged = {d.id: d.large_model for d in REGISTRY if d.large_model}
    assert {"code-review-assist", "html-creator", "screen-ocr"} <= set(badged)
    for model in badged.values():
        assert model.label and model.gpu_memory_gb > 0 and model.measured_on
    assert not any(hasattr(d, "requires_dgpu") for d in REGISTRY)


def test_code_review_reports_how_fast_the_answer_came(client, monkeypatch):
    stats = GenerationStats(device="GPU.0", tokens=300, seconds=8.1, tokens_per_second=38.2, first_token_seconds=0.4)
    monkeypatch.setattr(
        launcher_app.code_review_assist_runner,
        "review",
        lambda **kwargs: ReviewResult("feat: x", "- none", 10, False, stats=stats),
    )
    body = client.post(
        "/api/code-review-assist/review", json={"source": "diff_text", "diff_text": "x", "engine": "portable"}
    ).json()
    assert body["stats"] == {
        "device": "GPU.0", "tokens": 300, "seconds": 8.1, "tokens_per_second": 38.2, "first_token_seconds": 0.4,
        "energy": None, "cancelled": False,
    }


def test_telemetry_reports_power_or_says_it_cannot(client):
    """R18: the dock's power gauge needs to know whether there is a reading
    at all -- never a zero standing in for "no counters"."""
    power = client.get("/api/telemetry").json()["power"]
    assert "available" in power and "battery" in power


def _available_update():
    return launcher_app.updates.UpdateStatus(
        local="0.2.45", latest="0.2.47", update_available=True, can_upgrade=True, checked_at=1.0
    )


def test_update_status_offers_the_prompt_once(client, monkeypatch):
    monkeypatch.setattr(launcher_app.updates, "_status", _available_update())
    monkeypatch.setattr(launcher_app.updates, "_prompt_pending", True)
    monkeypatch.setattr(launcher_app.events, "status_snapshot", lambda: {})
    body = client.get("/api/update").json()
    assert (body["prompt"], body["latest"], body["running_demos"]) == (True, "0.2.47", [])
    client.post("/api/update/prompted")
    assert client.get("/api/update").json()["prompt"] is False


def test_html_creator_passes_pictures_through_and_returns_the_readable_page(client, monkeypatch):
    from html_creator.types import HtmlResult

    asked = {}

    def generate(**kwargs):
        asked.update(kwargs)
        return HtmlResult(
            html='<img src="data:image/svg+xml;base64,AAAA">', mode="landing_page", source_char_count=10,
            source_truncated=False, fence_stripped=False, html_truncated=False,
            pictures_offered=2, pictures_used=["hero.svg"], picture_notes=[], html_source='<img src="hero.svg">',
        )

    monkeypatch.setattr(launcher_app.html_creator_runner, "generate", generate)
    body = client.post(
        "/api/html-creator/generate", json={"prompt": "a page", "pictures": "C:/kit", "engine": "portable"}
    ).json()
    assert asked["pictures"] == "C:/kit"
    assert asked["repeatable"] is False and body["repeatable"] is False  # the switch is off unless asked
    assert (body["pictures_offered"], body["pictures_used"]) == (2, ["hero.svg"])
    client.post("/api/html-creator/generate", json={"prompt": "a page", "repeatable": True, "engine": "portable"})
    assert asked["repeatable"] is True
    assert body["html_source"] == '<img src="hero.svg">' and body["html"].startswith('<img src="data:')


def test_the_page_stamps_every_script_and_stylesheet_it_loads(client):
    """An unstamped one stays in the browser's cache across an update."""
    import re

    page = client.get("/").text
    loaded = re.findall(r'(?:src|href)="(/static/[^"]+\.(?:js|css)[^"]*)"', page)
    assert len(loaded) >= 3  # the stylesheet, app.js and expense-review.js at least
    assert all(re.search(r"\?v=\d+$", url) for url in loaded), loaded


def test_the_changelog_says_which_version_is_running(client, monkeypatch):
    listed = {"available": True, "reason": None, "versions": [{"version": "0.2.47", "changes": []}], "repo_url": "x"}
    monkeypatch.setattr(launcher_app.updates, "history", lambda: listed)
    body = client.get("/api/changelog").json()
    assert body["versions"] == listed["versions"]
    assert body["running"] == launcher_app.RUNNING_VERSION and "on_disk" in body


def test_upgrade_is_refused_while_a_demo_runs(client, monkeypatch):
    monkeypatch.setattr(launcher_app.updates, "refresh", _available_update)
    monkeypatch.setattr(launcher_app.activity, "snapshot", lambda: [{"demo_id": "live-translation"}])
    response = client.post("/api/update/upgrade")
    assert response.status_code == 409 and "Live Speech Translation" in response.json()["error"]


def test_upgrade_hands_off_to_the_helper(client, monkeypatch):
    monkeypatch.setattr(launcher_app.updates, "refresh", _available_update)
    monkeypatch.setattr(launcher_app.updates, "launcher_pids", lambda: [1])
    monkeypatch.setattr(launcher_app.events, "status_snapshot", lambda: {})
    monkeypatch.setattr(launcher_app, "_upgrade_requested", threading.Event())
    spawned = []
    monkeypatch.setattr(launcher_app.updates, "_spawn_detached", spawned.append)
    response = client.post("/api/update/upgrade")
    assert response.status_code == 202 and response.json()["to"] == "0.2.47"
    assert len(spawned) == 1 and launcher_app._upgrade_requested.is_set()


# --- the language model a brick answers with (BACKLOG R36) ------------------------


def test_the_four_bricks_with_a_small_language_model_say_which_ones_they_can_use(client, monkeypatch):
    from doc_qa import language_models

    # The larger model is on this laptop in its NPU build only.
    monkeypatch.setattr(language_models, "on_disk", lambda key, engine, device: key == "1.5b" or device == "NPU")
    for demo, default in (("doc-qa", {"npu": "8b", "other": "1.5b"}), ("expense-extract", {"npu": "1.5b", "other": "1.5b"}),
                          ("voice-assistant", {"npu": "1.5b", "other": "1.5b"}), ("video-commentary", {"npu": "1.5b", "other": "1.5b"})):
        offered = client.get(f"/api/{demo}/devices").json()["language_models"]
        assert [model["key"] for model in offered["models"]] == ["1.5b", "8b"]
        assert offered["models"][1]["on_disk"] == {"npu": True, "other": False} and offered["models"][1]["name"] == "Qwen3 8B"
        # What "Auto" means, chip by chip: Document Q&A takes the larger model where it is there, the others do not.
        assert offered["default"] == default and offered["portable"] == "1.5b"
    assert "language_models" not in client.get("/api/object-detection/devices").json()


def test_a_language_model_that_does_not_exist_is_refused_before_anything_starts(client, monkeypatch):
    started = []
    monkeypatch.setattr(launcher_app.voice_assistant_runner, "start", lambda **kwargs: started.append(kwargs))
    monkeypatch.setattr(launcher_app.expense_extract_runner, "start", lambda **kwargs: started.append(kwargs))
    monkeypatch.setattr(launcher_app.app.state, "voice_assistant_queue", object(), raising=False)  # the lifespan's, not started here

    refused = client.post("/api/voice-assistant/start", json={"engine": "openvino", "compute_device": "NPU", "llm_model": "gpt-9"})
    assert refused.status_code == 400 and "Unknown language model 'gpt-9'" in refused.json()["error"]
    refused = client.post("/api/expense-extract/start", json={"folder": "x", "llm_engine": "portable", "llm_model": "8b"})
    assert refused.status_code == 400 and "needs the OpenVINO engine" in refused.json()["error"]
    assert client.post("/api/doc-qa/ask", json={"question": "Who?", "model": "gpt-9"}).status_code == 400
    assert started == []

    assert client.post("/api/voice-assistant/start", json={"engine": "openvino", "compute_device": "NPU", "llm_model": "8b"}).status_code == 200
    assert client.post("/api/voice-assistant/start", json={"engine": "openvino", "compute_device": "NPU"}).status_code == 200
    assert [call["llm_model"] for call in started] == ["8b", None]  # nothing said: the brick's own choice


def test_document_qa_changes_its_model_between_two_questions_without_reading_the_folder_again(client, monkeypatch):
    from launcher import doc_qa_runner

    class Session:
        made = []

        def __init__(self, engine, *, device, model=None, on_downloading=None):
            self.model, self.folder = model or "8b", None
            self.store = type("Store", (), {"size": 3, "sources": ["a.md"]})()
            self.swaps = []
            Session.made.append(self)

        def ingest(self, folder, force=False):
            self.folder = folder
            return 3

        def use_model(self, model):
            self.swaps.append(model)
            self.model = model
            return True

        def ask(self, question, top_k=4, control=None):
            return doc_qa_runner.Answer(text=f"answered by {self.model}")

    monkeypatch.setattr(doc_qa_runner, "DocQASession", Session)
    monkeypatch.setattr(doc_qa_runner, "default_model", lambda engine, device: "8b")
    monkeypatch.setattr(launcher_app, "doc_qa_runner", doc_qa_runner.DocQARunner())

    read = client.post("/api/doc-qa/ingest", json={"folder": "docs", "engine": "openvino", "compute_device": "NPU"})
    assert read.status_code == 200 and read.json()["model"] == "8b"  # nobody chose: the brick's own choice, said back
    assert client.post("/api/doc-qa/ask", json={"question": "Who?"}).json()["text"] == "answered by 8b"
    answer = client.post("/api/doc-qa/ask", json={"question": "Who?", "model": "1.5b"}).json()
    assert answer["text"] == "answered by 1.5b" and answer["model"] == "1.5b"
    assert len(Session.made) == 1 and Session.made[0].swaps == ["1.5b"]  # the same session: the index is the one it had
    assert client.get("/api/doc-qa/status").json()["model"] == "1.5b"
