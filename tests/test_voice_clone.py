"""voice-clone-studio's wrapper around the vendored OpenVoice extractor.

The vendored code states its input preconditions as bare `assert`s. Those
are bad-input conditions, so `enroll` re-raises them as ValueError -- which
the launcher answers with a 400 rather than a 500, and which survives
`python -O` (asserts don't).
"""
from __future__ import annotations

import pytest
from voice_clone_studio import voice_model


def test_a_clip_with_no_speech_is_a_value_error(monkeypatch):
    def refuse(path, converter, **kwargs):
        raise AssertionError("no speech detected in the reference clip -- try a clearer recording")

    monkeypatch.setattr(voice_model.se_extractor, "get_se", refuse)
    with pytest.raises(ValueError, match="no speech detected"):
        voice_model.enroll(converter=object(), reference_audio_path="quiet.wav")


def test_a_bare_assert_still_gets_a_usable_message(monkeypatch):
    def refuse(path, converter, **kwargs):
        raise AssertionError()

    monkeypatch.setattr(voice_model.se_extractor, "get_se", refuse)
    with pytest.raises(ValueError, match="can't be used"):
        voice_model.enroll(converter=object(), reference_audio_path="quiet.wav")


def test_a_good_clip_passes_the_embedding_through(monkeypatch):
    monkeypatch.setattr(voice_model.se_extractor, "get_se", lambda path, converter, **kw: ("embedding", None))
    assert voice_model.enroll(converter=object(), reference_audio_path="speech.wav") == "embedding"


# --- the NPU, which these models end the program on ---------------------------------------
#
# Found on 2026-10-09 and confirmed the next day through both bricks: handed
# to the NPU's compiler, the speaker model does not fail -- it ends the
# process ("LLVM ERROR", exit code 127), launcher and all. So nothing here
# may ever try: the NPU is refused by name, before a model is loaded.


def test_the_npu_is_refused_by_name_and_auto_is_the_cpu():
    with pytest.raises(ValueError, match="does not compile for the NPU"):
        voice_model.validate_device("NPU")
    with pytest.raises(ValueError, match="does not compile for the NPU"):
        voice_model.compile_device("npu")
    # Left to the app, the chip it is fastest on -- never OpenVINO's own AUTO, which could pick the NPU.
    assert voice_model.compile_device("AUTO") == voice_model.compile_device("") == voice_model.TTS_DEVICE == "CPU"
    assert voice_model.compile_device("GPU.0") == "GPU.0" and voice_model.compile_device("CPU") == "CPU"


def test_the_voice_is_never_compiled_for_the_npu_not_even_to_see_if_it_works(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_openvino(name, *args, **kwargs):
        assert name != "openvino", "the NPU must be refused before OpenVINO is even imported"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_openvino)
    with pytest.raises(ValueError, match="does not compile for the NPU"):
        voice_model.accelerate_tts_with_openvino(object(), device="NPU")


def test_a_clone_asked_for_on_the_npu_is_refused_before_any_model_is_loaded(monkeypatch):
    from pantherlake_ai_core.engine import Engine
    from voice_clone_studio import engine_factory

    monkeypatch.setattr(voice_model, "load_models", lambda *args, **kwargs: pytest.fail("a model was loaded for the NPU"))
    with pytest.raises(ValueError, match="does not compile for the NPU"):
        engine_factory.create_cloner(Engine.OPENVINO, model=engine_factory.OPENVOICE, device="NPU")


def test_the_launcher_greys_the_npu_out_and_refuses_it_for_a_cloned_voice(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from launcher import app as launcher_app
    from pantherlake_ai_core.engine import Engine

    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    monkeypatch.setattr(launcher_app, "_DEVICE_SOURCES", {**launcher_app._DEVICE_SOURCES, "microphones": lambda: [], "speakers": lambda: []})
    monkeypatch.setattr(launcher_app, "resolve_engine", lambda engine: Engine(engine) if engine else Engine.OPENVINO)
    web = TestClient(launcher_app.app)
    told = web.get("/api/voice-clone-studio/devices").json()
    assert list(told["openvino_unsupported"]) == ["NPU"] and "does not compile for the NPU" in told["openvino_unsupported"]["NPU"]
    assert told["auto_device"] == "CPU"  # what "Auto" means here, so the menu can say it
    # Left to the app: the CPU. A GPU: allowed, it works. The NPU: refused, never tried.
    assert launcher_app._voice_clone_engine("openvoice", "openvino", None) == (Engine.OPENVINO, "CPU")
    assert launcher_app._voice_clone_engine("openvoice", "openvino", "AUTO") == (Engine.OPENVINO, "CPU")
    with pytest.raises(ValueError, match="does not compile for the NPU"):
        launcher_app._voice_clone_engine("openvoice", "openvino", "NPU")
    enrolled = []
    monkeypatch.setattr(launcher_app.voice_clone_studio_runner, "enroll", lambda **kwargs: enrolled.append(kwargs))
    clip = tmp_path / "me.wav"
    clip.write_bytes(b"RIFF")
    with clip.open("rb") as handle:
        refused = web.post("/api/voice-clone-studio/enroll-upload", files={"file": ("me.wav", handle, "audio/wav")},
                           data={"model": "openvoice", "engine": "openvino", "compute_device": "NPU"})
    assert refused.status_code == 400 and "does not compile for the NPU" in refused.json()["error"] and enrolled == []


def test_the_voice_assistant_listens_and_answers_on_the_npu_and_speaks_from_the_cpu(monkeypatch):
    from pantherlake_ai_core.engine import Engine
    from voice_assistant import session

    asked = {}
    monkeypatch.setattr(session, "WakeWordDetector", lambda **kwargs: object())
    monkeypatch.setattr(session, "create_translator", lambda engine, size, device, **kwargs: asked.setdefault("listens", device))
    monkeypatch.setattr(session, "create_llm", lambda engine, device, **kwargs: asked.setdefault("answers", device))
    monkeypatch.setattr(session.vc_voice_model, "load_tts_only", lambda: "voice")
    monkeypatch.setattr(session.vc_voice_model, "accelerate_tts_with_openvino", lambda tts, device: asked.setdefault("speaks", device))
    assistant = session.VoiceAssistantSession(Engine.OPENVINO, whisper_model_size="base", device="NPU")
    assert asked == {"listens": "NPU", "answers": "NPU", "speaks": "CPU"} and assistant.voice_device == "CPU"
    # The same on a GPU: there the voice would be compiled again for every sentence of a new length.
    asked.clear()
    session.VoiceAssistantSession(Engine.OPENVINO, whisper_model_size="base", device="GPU.0")
    assert asked == {"listens": "GPU.0", "answers": "GPU.0", "speaks": "CPU"}


def test_the_hardware_panel_shows_the_assistants_voice_on_the_chip_it_is_made_on(monkeypatch):
    import threading
    import time

    from fastapi.testclient import TestClient
    from launcher import app as launcher_app
    from pantherlake_ai_core.engine import Engine
    from voice_assistant import session

    running = threading.Event()

    def run(*, on_ready, stop_event, **settings):
        on_ready()
        running.set()
        stop_event.wait(5)

    monkeypatch.setattr(session, "run", run)
    monkeypatch.setattr(launcher_app, "resolve", lambda engine, device, **kwargs: (Engine.OPENVINO, device or "NPU"))

    def rows(web):
        return {entry["stage"]: entry["device"] for entry in web.get("/api/telemetry").json()["active"]
                if entry["demo_id"] == "voice-assistant"}

    with TestClient(launcher_app.app) as web:
        assert web.post("/api/voice-assistant/start", json={"engine": "openvino", "compute_device": "NPU"}).status_code == 200
        assert running.wait(3)
        assert rows(web) == {"default": "NPU", "voice": "CPU"}  # two rows: where it listens and answers, where it speaks
        web.post("/api/voice-assistant/stop")
        deadline = time.time() + 3
        while rows(web) and time.time() < deadline:
            time.sleep(0.02)
        assert rows(web) == {}
        # Told not to speak, or already on the CPU: one row.
        for body in ({"engine": "openvino", "compute_device": "NPU", "speak_replies": False}, {"engine": "openvino", "compute_device": "CPU"}):
            running.clear()
            assert web.post("/api/voice-assistant/start", json=body).status_code == 200
            assert running.wait(3) and list(rows(web)) == ["default"]
            web.post("/api/voice-assistant/stop")
            deadline = time.time() + 3
            while rows(web) and time.time() < deadline:
                time.sleep(0.02)
