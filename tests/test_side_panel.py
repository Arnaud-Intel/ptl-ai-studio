"""What the side panel is built on: a real chip for every brick (never
"AUTO"), each brick's own number, and one way to stop or unload any of them."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient
from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core.engine import Engine, GpuDevice
from pantherlake_ai_core.types import TranslationResult

from launcher import activity, loaded, metrics
from launcher import app as launcher_app
from launcher.errors import Conflict

IGPU = GpuDevice("GPU.0", "Intel(R) Arc(TM) B390 GPU (iGPU)", None)
DGPU = GpuDevice("GPU.1", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", None)


@pytest.fixture(autouse=True)
def clean_state():
    yield
    with metrics._lock:
        metrics._metrics.clear()
    with activity._lock:
        activity._active.clear()


# --- a real chip, never "AUTO" ---------------------------------------------------------


def test_auto_means_the_integrated_gpu_then_the_cpu(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [IGPU, DGPU])
    assert engine_mod.preferred_device() == "GPU.0"  # not the discrete card: small models lose on the trip
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [])
    assert engine_mod.preferred_device() == "CPU"


@pytest.fixture
def machine(monkeypatch):
    """A laptop with an iGPU, a dGPU and an NPU, as the launcher sees it."""
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "GPU.1", "NPU"])
    monkeypatch.setattr(launcher_app, "resolve_engine", lambda explicit: Engine(explicit) if explicit else Engine.OPENVINO)
    monkeypatch.setattr(launcher_app, "preferred_device", lambda: "GPU.0")
    monkeypatch.setattr(launcher_app, "preferred_large_model_device", lambda: "GPU.1")
    monkeypatch.setattr(launcher_app, "preferred_realtime_vision_device", lambda: "GPU.0")


@pytest.mark.parametrize("asked", [None, "", "AUTO", "auto"])
def test_a_device_left_to_the_app_resolves_to_a_chip(machine, asked):
    """The bug: a brick on "AUTO" reported "AUTO" and appeared under no chip."""
    assert launcher_app.resolve(None, asked) == (Engine.OPENVINO, "GPU.0")
    assert launcher_app.resolve(None, asked, large_model=True) == (Engine.OPENVINO, "GPU.1")
    assert launcher_app.resolve(None, asked, realtime_vision=True) == (Engine.OPENVINO, "GPU.0")


def test_auto_on_a_machine_without_a_gpu_is_the_cpu(machine, monkeypatch):
    # Both older rules answer "AUTO" when there's no GPU; that must not leak out.
    monkeypatch.setattr(launcher_app, "preferred_large_model_device", lambda: "AUTO")
    monkeypatch.setattr(launcher_app, "preferred_realtime_vision_device", lambda: "AUTO")
    assert launcher_app.resolve(None, "AUTO", large_model=True) == (Engine.OPENVINO, "CPU")
    assert launcher_app.resolve(None, None, realtime_vision=True) == (Engine.OPENVINO, "CPU")


def test_an_explicit_choice_is_kept_and_still_validated(machine):
    assert launcher_app.resolve("openvino", "npu") == (Engine.OPENVINO, "NPU")
    assert launcher_app.resolve("portable", None) == (Engine.PORTABLE, "cpu")
    with pytest.raises(ValueError, match="unavailable"):
        launcher_app.resolve("openvino", "GPU.7")
    with pytest.raises(ValueError, match="Portable engines"):
        launcher_app.resolve("portable", "NPU")


# --- each brick's own number -----------------------------------------------------------


def test_a_frame_rate_follows_the_last_couple_of_seconds():
    now = [0.0]
    meter = metrics.RateMeter(window=2.0, clock=lambda: now[0])
    assert meter.tick() == 0.0  # one frame is not a rate
    for _ in range(60):
        now[0] += 1 / 30
        rate = meter.tick()
    assert rate == pytest.approx(30, rel=0.02)
    for _ in range(20):  # the source slows to 10 fps: the number follows within the window
        now[0] += 1 / 10
        rate = meter.tick()
    assert rate == pytest.approx(10, rel=0.05)


def test_a_live_number_goes_with_its_stage_and_a_sticky_one_stays():
    activity.set_active("object-detection", engine="openvino", device="GPU.0")
    metrics.report("object-detection", 31.4, "fps")
    metrics.report("code-review-assist", 44.4, "tok/s", sticky=True)
    activity.clear_active("object-detection")
    activity.clear_active("code-review-assist")
    left = {(m["demo_id"], m["unit"]) for m in metrics.snapshot()}
    assert left == {("code-review-assist", "tok/s")}
    metrics.clear("code-review-assist")  # unloading takes the sticky one too
    assert metrics.snapshot() == []


def test_stages_keep_their_own_numbers_and_say_which_stage():
    activity.set_active("expense-extract", engine="openvino", device="NPU", stage="ocr", stage_label="OCR")
    activity.set_active("expense-extract", engine="openvino", device="GPU.0", stage="llm", stage_label="Structuring")
    metrics.report("expense-extract", 6.0, "receipts/min", stage="ocr")
    metrics.report("expense-extract", 4.5, "receipts/min", stage="llm")
    assert {a["stage"]: a["device"] for a in activity.snapshot()} == {"ocr": "NPU", "llm": "GPU.0"}
    assert {m["stage"]: m["value"] for m in metrics.snapshot()} == {"ocr": 6.0, "llm": 4.5}


def test_speech_is_timed_per_utterance(monkeypatch):
    """Times real time needs how long the audio was and how long it took."""
    from live_translation import pipeline

    class _Translator:
        def translate(self, segment):
            return TranslationResult(text="bonjour", detected_language="fr", language_probability=1.0)

    monkeypatch.setattr(pipeline, "create_translator", lambda **kwargs: _Translator())
    monkeypatch.setattr(pipeline.audio, "stream_blocks", lambda *a, **k: iter(()))
    monkeypatch.setattr(pipeline, "segment_stream", lambda blocks, config: iter([[0.0] * (pipeline.audio.SAMPLE_RATE * 2)]))
    results = []
    pipeline.run(
        source="mic", audio_device=None, engine=Engine.OPENVINO, model_size="base",
        compute_device="NPU", on_result=results.append,
    )
    assert results[0].audio_seconds == 2.0 and results[0].processing_seconds >= 0


# --- stopping and unloading -------------------------------------------------------------


class _OneShotRunner:
    def __init__(self, device="GPU.0"):
        self._session = object()
        self._engine = "openvino"
        self._device = device
        self._lock = threading.Lock()


def test_an_idle_brick_holding_a_model_is_listed_and_can_be_unloaded():
    runner = _OneShotRunner()
    metrics.report("code-review-assist", 44.4, "tok/s", sticky=True)
    assert loaded.info(runner) == {"engine": "openvino", "device": "GPU.0"}
    assert loaded.unload(runner, "code-review-assist") is True
    assert loaded.info(runner) is None and metrics.snapshot() == []
    assert loaded.unload(runner, "code-review-assist") is False  # nothing left: not an error


def test_unloading_mid_request_is_refused_not_forced():
    runner = _OneShotRunner()
    with runner._lock:
        with pytest.raises(Conflict, match="middle of a request"):
            loaded.unload(runner, "doc-qa")
    assert runner._session is not None


def test_the_session_knows_where_it_really_loaded():
    runner = _OneShotRunner(device="GPU")
    runner._session = type("S", (), {"device": "GPU.1"})()
    assert loaded.info(runner)["device"] == "GPU.1"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher_app.events, "LOG_FILE", tmp_path / "events.log")
    return TestClient(launcher_app.app)


def test_telemetry_carries_what_the_panel_draws(client, monkeypatch):
    monkeypatch.setitem(launcher_app._UNLOADABLE, "code-review-assist", _OneShotRunner())
    activity.set_active("live-translation", engine="openvino", device="NPU")
    metrics.report("live-translation", 12.5, "x real time")
    body = client.get("/api/telemetry").json()
    assert body["active"][0] | {"stage": "default"} == body["active"][0]
    assert body["metrics"][0]["unit"] == "x real time"
    assert {"demo_id": "code-review-assist", "engine": "openvino", "device": "GPU.0"} in body["loaded"]


def test_one_route_stops_a_loop_and_unloads_a_one_shot(client, monkeypatch):
    stopped = []
    monkeypatch.setitem(launcher_app._STOPPABLE, "live-translation", type("R", (), {"stop": lambda self: stopped.append(1)})())
    assert client.post("/api/bricks/live-translation/stop").json() == {"status": "stopped"} and stopped == [1]

    runner = _OneShotRunner()
    monkeypatch.setitem(launcher_app._UNLOADABLE, "screen-ocr", runner)
    assert client.post("/api/bricks/screen-ocr/stop").json() == {"status": "unloaded"}
    assert client.post("/api/bricks/screen-ocr/stop").json() == {"status": "idle"}

    busy = _OneShotRunner()
    monkeypatch.setitem(launcher_app._UNLOADABLE, "doc-qa", busy)
    with busy._lock:
        assert client.post("/api/bricks/doc-qa/stop").status_code == 409
    assert client.post("/api/bricks/not-a-brick/stop").status_code == 404


def test_every_available_brick_can_be_stopped_or_unloaded():
    from launcher.registry import REGISTRY

    reachable = set(launcher_app._STOPPABLE) | set(launcher_app._UNLOADABLE)
    assert {demo.id for demo in REGISTRY if demo.status == "available"} == reachable
