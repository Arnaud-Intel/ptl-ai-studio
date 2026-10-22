"""The Voice Assistant listens while its button is held; a wake word is a
choice, and nothing of it is loaded until it is made (BACKLOG R17: the wake
words are models for non-commercial use)."""
from __future__ import annotations

import threading

import numpy as np
import pytest
from pantherlake_ai_core import audio
from pantherlake_ai_core.engine import Engine

BLOCK = int(audio.SAMPLE_RATE * 0.03)


def _microphone(talk: threading.Event, held: range, total: int, value: float = 0.5):
    """`total` blocks of sound, numbered in their first sample, with the
    button down for the blocks in `held` -- as a person holding it would."""
    for index in range(total):
        if index == held.start:
            talk.set()
        elif index == held.stop:
            talk.clear()
        block = np.full(BLOCK, value, dtype=np.float32)
        block[0] = index
        yield block


def _numbers(sound: np.ndarray) -> list[int]:
    return [int(sound[start]) for start in range(0, len(sound), BLOCK)]


def test_a_question_is_what_is_said_while_the_button_is_held():
    from voice_assistant import session

    talk, started, missed = threading.Event(), [], []
    questions = session._while_held(_microphone(talk, range(40, 90), 120), talk, lambda: started.append(1), missed.append)
    heard = next(questions)
    # The third of a second before the press comes with it, so that a word
    # begun as the button goes down is whole; nothing older does, and it ends
    # with the block in which the button came up.
    assert _numbers(heard) == list(range(30, 91))
    assert started == [1] and missed == []
    assert next(questions, "the microphone was closed") == "the microphone was closed"


def test_a_click_is_not_a_question_and_a_question_has_a_longest_length(monkeypatch):
    from voice_assistant import session

    talk, missed = threading.Event(), []
    questions = session._while_held(_microphone(talk, range(20, 25), 60), talk, lambda: None, missed.append)
    assert next(questions) is None and missed == [session.TOO_SHORT]

    monkeypatch.setattr(session, "_LONGEST_SECONDS", 1.5)
    talk = threading.Event()
    questions = session._while_held(_microphone(talk, range(10, 500), 500), talk, lambda: None, missed.append)
    assert len(next(questions)) == pytest.approx((1.5 + session._LEAD_SECONDS) * audio.SAMPLE_RATE, abs=2 * BLOCK)


class _Heard:
    def __init__(self, text: str):
        self.text = text


def _assistant(monkeypatch, *, hears: str = "what is the capital of France"):
    """The session with its models stood in for, and no loudspeaker."""
    from voice_assistant import session

    said: list[str] = []

    class Words:
        last_stats = None

        def answer(self, system, question, max_tokens=200):
            return f"You asked: {question}"

    class Ears:
        def translate(self, sound):
            return _Heard(hears)

    class Voice:
        def tts(self, text, *args, **kwargs):
            said.append(text)
            return np.zeros(10, dtype=np.float32)

    monkeypatch.setattr(session, "create_translator", lambda *args, **kwargs: Ears())
    monkeypatch.setattr(session, "create_llm", lambda *args, **kwargs: Words())
    monkeypatch.setattr(session.vc_voice_model, "load_tts_only", lambda: Voice())
    monkeypatch.setattr(session.vc_voice_model, "accelerate_tts_with_openvino", lambda tts, device: None)
    monkeypatch.setattr(session.audio, "play", lambda sound, rate: None)
    return session, said


def test_with_no_wake_word_none_is_loaded_and_the_button_asks(monkeypatch):
    session, said = _assistant(monkeypatch)

    def no_wake_word(**kwargs):
        raise AssertionError("the wake word's models were loaded without one being asked for")

    monkeypatch.setattr(session, "WakeWordDetector", no_wake_word)
    talk = threading.Event()
    monkeypatch.setattr(session.audio, "stream_blocks", lambda source, device, stop_event=None: _microphone(talk, range(20, 70), 100))
    seen: list[tuple] = []
    session.run(
        audio_device=None, engine=Engine.OPENVINO, whisper_model_size="base", compute_device="NPU", talk=talk,
        on_wake=lambda: seen.append(("listening",)), on_heard=lambda text: seen.append(("heard", text)),
        on_reply=lambda text: seen.append(("reply", text)), on_idle=lambda: seen.append(("idle",)),
    )
    assert seen == [("listening",), ("heard", "what is the capital of France"),
                    ("reply", "You asked: what is the capital of France"), ("idle",)]
    assert said == ["You asked: what is the capital of France"]
    assert not talk.is_set()  # a press made during the answer is not kept for the next question

    with pytest.raises(ValueError, match="talk"):
        session.run(audio_device=None, engine=Engine.OPENVINO, whisper_model_size="base", compute_device="NPU")


def test_a_hold_in_which_nothing_was_said_is_told_and_the_assistant_listens_again(monkeypatch):
    session, said = _assistant(monkeypatch, hears="  ")
    talk = threading.Event()
    monkeypatch.setattr(session.audio, "stream_blocks", lambda source, device, stop_event=None: _microphone(talk, range(20, 70), 100))
    seen: list = []
    session.run(audio_device=None, engine=Engine.PORTABLE, whisper_model_size="small", compute_device="CPU", talk=talk,
                on_missed=seen.append, on_idle=lambda: seen.append("idle"))
    assert seen == [session.NOTHING_HEARD, "idle"] and said == []


def test_the_launcher_starts_on_the_button_and_takes_it_only_when_it_can_listen(monkeypatch):
    import time

    from fastapi.testclient import TestClient
    from launcher import app as launcher_app
    from launcher import voice_assistant_runner as runner_module
    from voice_assistant import session

    asked: dict = {}
    release = threading.Event()

    def run(*, on_ready, on_wake, on_idle, stop_event, talk, wake_word, **settings):
        asked.update(wake_word=wake_word, talk=talk)
        on_ready()
        while not stop_event.is_set():
            if talk.is_set():
                on_wake()  # a question is being heard and answered...
                release.wait(5)
                talk.clear()
                on_idle()  # ...and the assistant listens again
            time.sleep(0.01)

    monkeypatch.setattr(session, "run", run)
    monkeypatch.setattr(launcher_app, "resolve", lambda engine, device, **kwargs: (Engine.OPENVINO, device or "NPU"))
    runner = launcher_app.voice_assistant_runner

    def settled(want: bool):
        deadline = time.monotonic() + 5
        while runner.state()["listening"] is not want and time.monotonic() < deadline:
            time.sleep(0.01)
        return runner.state()

    with TestClient(launcher_app.app) as web:
        assert web.post("/api/voice-assistant/talk", json={"down": True}).json()["taken"] is False  # nothing runs
        started = web.post("/api/voice-assistant/start", json={"engine": "openvino", "compute_device": "NPU"})
        try:
            assert started.status_code == 200 and started.json()["trigger"] == runner_module.BUTTON
            assert settled(True) == {"running": True, "trigger": "button", "listening": True}
            assert asked["wake_word"] is None
            assert web.get("/api/voice-assistant/state").json()["listening"] is True

            assert web.post("/api/voice-assistant/talk", json={"down": True}).json()["taken"] is True
            assert settled(False)["listening"] is False
            # While the answer is written and said, a new press is not a question.
            assert web.post("/api/voice-assistant/talk", json={"down": True}).json()["taken"] is False
            release.set()
            assert settled(True)["listening"] is True and not asked["talk"].is_set()
        finally:
            release.set()
            web.post("/api/voice-assistant/stop")
    assert runner.state() == {"running": False, "trigger": "button", "listening": False}


def test_a_wake_word_is_still_there_for_who_asks_for_it(monkeypatch):
    session, _said = _assistant(monkeypatch)
    loaded = []
    monkeypatch.setattr(session, "WakeWordDetector", lambda **kwargs: loaded.append(kwargs) or object())
    assistant = session.VoiceAssistantSession(Engine.OPENVINO, whisper_model_size="base", device="NPU", wake_word="hey_jarvis")
    assert loaded == [{"wake_word": "hey_jarvis", "threshold": 0.5}] and assistant.wake_detector is not None
    assert session.VoiceAssistantSession(Engine.OPENVINO, whisper_model_size="base", device="NPU").wake_detector is None
