"""The launcher's API contract, with every hardware probe monkeypatched so
this runs on a machine (or a CI runner) with no microphone, camera, or
OpenVINO device -- and no model is ever loaded."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pantherlake_ai_core.engine import Engine

from launcher import app as launcher_app
from launcher.errors import Conflict
from launcher.registry import REGISTRY

FAKE_OPENVINO_DEVICES = ["CPU", "GPU.0", "NPU"]


@pytest.fixture
def client(monkeypatch):
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
    monkeypatch.setattr(launcher_app, "resolve_engine", lambda explicit: Engine(explicit) if explicit else Engine.OPENVINO)
    monkeypatch.setattr(launcher_app, "preferred_large_model_device", lambda: "GPU.1")
    assert launcher_app.resolve(None, None) == (Engine.OPENVINO, "AUTO")
    assert launcher_app.resolve("portable", None) == (Engine.PORTABLE, "cpu")
    assert launcher_app.resolve("openvino", "NPU") == (Engine.OPENVINO, "NPU")
    assert launcher_app.resolve(None, None, large_model=True) == (Engine.OPENVINO, "GPU.1")
    assert launcher_app.resolve("portable", None, large_model=True) == (Engine.PORTABLE, "cpu")


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
    assert {"id", "name", "status", "devices", "samples", "requires_dgpu"} <= set(demos[0])


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
