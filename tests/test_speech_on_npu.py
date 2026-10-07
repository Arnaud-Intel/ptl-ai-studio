"""Speech goes to the NPU when nobody chooses a chip, with the Whisper size
that translates rather than the one that gets the gist."""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from live_translation import transcriber_openvino
from live_translation.transcriber_openvino import OpenVINOTranslator
from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core import npu
from pantherlake_ai_core.engine import Engine

from launcher import app as launcher_app


@pytest.fixture
def xps(monkeypatch):
    """A machine like the XPS 14: a CPU, an integrated GPU and an NPU in working order."""
    monkeypatch.setattr(npu, "lost", lambda: None)
    monkeypatch.setattr(engine_mod, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    monkeypatch.setattr(engine_mod, "preferred_device", lambda: "GPU.0")
    monkeypatch.setattr(launcher_app, "resolve_engine", lambda engine: Engine(engine) if engine else Engine.OPENVINO)
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    monkeypatch.setattr(launcher_app, "preferred_device", lambda: "GPU.0")


def test_the_npu_is_picked_for_speech_when_there_is_one_in_working_order(xps, monkeypatch):
    assert engine_mod.preferred_npu_device() == "NPU" == engine_mod.default_speech_device(Engine.OPENVINO)
    assert engine_mod.preferred_npu_device(["CPU", "GPU.0"]) == "GPU.0"  # no NPU: the chip any brick gets
    assert engine_mod.default_speech_device(Engine.PORTABLE) == "cpu"
    monkeypatch.setattr(npu, "lost", lambda: "the device was removed")  # reset by Windows earlier this session
    assert engine_mod.preferred_npu_device() == "GPU.0"


def test_the_launcher_resolves_auto_to_the_npu_only_where_asked(xps, monkeypatch):
    assert launcher_app.resolve("openvino", None, prefer_npu=True) == (Engine.OPENVINO, "NPU")
    assert launcher_app.resolve("openvino", "AUTO", prefer_npu=True) == (Engine.OPENVINO, "NPU")
    assert launcher_app.resolve("openvino", "GPU.0", prefer_npu=True) == (Engine.OPENVINO, "GPU.0")  # a choice is a choice
    assert launcher_app.resolve("openvino", None) == (Engine.OPENVINO, "GPU.0")  # every other brick: unchanged

    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0"])
    assert launcher_app.resolve("openvino", None, prefer_npu=True) == (Engine.OPENVINO, "GPU.0")
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    monkeypatch.setattr(npu, "lost", lambda: "the device was removed")
    assert launcher_app.resolve("openvino", None, prefer_npu=True) == (Engine.OPENVINO, "GPU.0")


def test_live_translation_starts_on_the_npu_with_the_medium_model(xps, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher_app.events, "LOG_FILE", tmp_path / "events.log")
    monkeypatch.setattr(launcher_app.app.state, "live_translation_queue", None, raising=False)
    started: list[dict] = []
    monkeypatch.setattr(launcher_app.live_translation_runner, "start", lambda **kwargs: started.append(kwargs))
    client = TestClient(launcher_app.app)

    assert client.post("/api/live-translation/start", json={}).status_code == 200
    assert client.post("/api/live-translation/start", json={"compute_device": "GPU.0", "model_size": "base"}).status_code == 200
    assert client.post("/api/live-translation/start", json={"engine": "portable"}).status_code == 200
    assert [(call["compute_device"], call["model_size"]) for call in started] == [
        ("NPU", "medium"), ("GPU.0", "base"), ("cpu", "small"),
    ]


def test_the_page_is_told_what_auto_means_for_the_speech_bricks(xps, monkeypatch):
    monkeypatch.setattr(launcher_app.audio, "list_microphones", lambda: ["Test Mic"])
    monkeypatch.setattr(launcher_app.audio, "list_speakers", lambda: ["Test Speakers"])
    client = TestClient(launcher_app.app)
    for demo_id in ("live-translation", "meeting-notes"):
        assert client.get(f"/api/{demo_id}/devices").json()["auto_device"] == "NPU"
    assert "auto_device" not in client.get("/api/doc-qa/devices").json()  # "Auto" stays "the app picks" elsewhere


def test_the_small_model_is_offered_on_openvino(monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(transcriber_openvino, "resolve_snapshot", lambda repo, **kwargs: asked.append(repo) or "dir")
    for size in ("small", "medium"):
        assert transcriber_openvino._resolve_model_dir(size, None) == "dir"
    assert asked == ["OpenVINO/whisper-small-fp16-ov", "OpenVINO/whisper-medium-fp16-ov"]


# --- large-v3 on the NPU: right words, wrong language label -----------------------------


def _translator(tmp_path, device: str, mel_bins: int, language: str | None) -> OpenVINOTranslator:
    (tmp_path / "config.json").write_text(json.dumps({"num_mel_bins": mel_bins}), encoding="utf-8")

    class _Result(str):
        languages = ["vi"]

    class _Pipeline:
        def __init__(self, model_dir, device, **config):
            pass

        def generate(self, samples, **options):
            result = _Result("The UN hopes to finalise a fund.")
            # What the NPU answered for French ("vi"), unless it was told the language.
            result.languages = [options["language"].strip("<|>")] if "language" in options else ["vi"]
            return result

    translator = OpenVINOTranslator.__new__(OpenVINOTranslator)  # no model: the label it reports is under test
    translator._pipeline_cls = _Pipeline
    translator._model_dir = str(tmp_path)
    translator.task = "translate"
    translator.language = language
    translator._load(device)
    return translator


@pytest.mark.parametrize(
    ("device", "mel_bins", "language", "label"),
    [
        ("NPU", 128, None, "auto"),  # large-v3 on the NPU: the label it gives is not to be believed
        ("NPU", 128, "fr", "fr"),  # ...unless the language was fixed, which is what it then reports
        ("GPU.0", 128, None, "vi"),  # on a GPU it is right, so whatever it says is shown
        ("NPU", 80, None, "vi"),  # every smaller size is right on the NPU too
    ],
)
def test_a_language_label_known_to_be_wrong_is_not_shown(monkeypatch, tmp_path, device, mel_bins, language, label):
    monkeypatch.setattr(transcriber_openvino, "ov_config_for", lambda device: {})
    monkeypatch.setattr(npu, "lost", lambda: None)
    heard = _translator(tmp_path, device, mel_bins, language).translate(np.zeros(1600, dtype=np.float32))
    assert heard.detected_language == label and heard.text.startswith("The UN")
