"""Live translation and meeting notes can be told which language is being
spoken instead of detecting it for every utterance (live_translation.languages)."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from live_translation.languages import SPOKEN_LANGUAGES, spoken_language
from live_translation.transcriber_openvino import OpenVINOTranslator
from pantherlake_ai_core.engine import Engine

from launcher import app as launcher_app


def test_nothing_chosen_means_detect_and_an_unknown_language_is_refused():
    assert spoken_language(None) is None and spoken_language("") is None and spoken_language("auto") is None
    assert spoken_language("FR") == "fr" and SPOKEN_LANGUAGES["fr"] == "French"
    with pytest.raises(ValueError, match="Unknown spoken language"):
        spoken_language("klingon")


class _Heard:
    languages = ["fr"]

    def __str__(self) -> str:
        return "bonjour"


class _Listening:
    """Stands in for the OpenVINO speech pipeline: remembers what it was asked."""

    def __init__(self, model_dir, device, **config):
        self.options: dict = {}

    def generate(self, samples, **options):
        self.options = options
        return _Heard()


@pytest.mark.parametrize(
    ("language", "expected"),
    [(None, {"task": "translate"}), ("fr", {"task": "translate", "language": "<|fr|>"})],
)
def test_the_speech_model_is_told_the_language_only_when_one_was_chosen(monkeypatch, language, expected):
    monkeypatch.setattr("live_translation.transcriber_openvino.ov_config_for", lambda device: {})
    translator = OpenVINOTranslator.__new__(OpenVINOTranslator)  # no model: the call it makes is under test
    translator._pipeline_cls = _Listening
    translator._model_dir = "model"
    translator.task = "translate"
    translator.language = language
    translator._load("GPU.0")
    result = translator.translate(np.zeros(1600, dtype=np.float32))
    assert translator.pipeline.options == expected
    assert result.detected_language == "fr"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher_app.events, "LOG_FILE", tmp_path / "events.log")
    monkeypatch.setattr(launcher_app.audio, "list_microphones", lambda: ["Test Mic"])
    monkeypatch.setattr(launcher_app.audio, "list_speakers", lambda: ["Test Speakers"])
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    monkeypatch.setattr(launcher_app, "resolve", lambda engine, device, **kwargs: (Engine.OPENVINO, "NPU"))
    # No lifespan here, so the queues a session reports through are stand-ins.
    monkeypatch.setattr(launcher_app.app.state, "live_translation_queue", None, raising=False)
    monkeypatch.setattr(launcher_app.app.state, "meeting_notes_queue", None, raising=False)
    return TestClient(launcher_app.app)


@pytest.mark.parametrize(
    ("route", "runner", "argument"),
    [
        ("/api/live-translation/start", "live_translation_runner", "language"),
        ("/api/meeting-notes/start", "meeting_notes_runner", "spoken_language"),
    ],
)
def test_the_chosen_language_reaches_the_session(client, monkeypatch, route, runner, argument):
    started: list[dict] = []
    monkeypatch.setattr(getattr(launcher_app, runner), "start", lambda **kwargs: started.append(kwargs))

    assert client.post(route, json={"language": "fr"}).status_code == 200
    assert client.post(route, json={"language": "auto"}).status_code == 200
    assert client.post(route, json={}).status_code == 200
    assert [call[argument] for call in started] == ["fr", None, None]

    refused = client.post(route, json={"language": "klingon"})
    assert refused.status_code == 400 and "Unknown spoken language" in refused.json()["error"]
    assert len(started) == 3  # nothing was started for a language the model was never offered


@pytest.mark.parametrize("demo_id", ["live-translation", "meeting-notes"])
def test_the_page_is_given_the_languages_to_offer(client, demo_id):
    offered = client.get(f"/api/{demo_id}/devices").json()["spoken_languages"]
    assert {"code": "fr", "name": "French"} in offered and len(offered) == len(SPOKEN_LANGUAGES)
