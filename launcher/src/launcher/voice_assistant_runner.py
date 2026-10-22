"""Runs the voice-assistant brick's listen/think/speak loop on a background
thread and forwards each event into an asyncio queue the web UI drains over a
WebSocket -- same shape as LiveTranslationRunner. A single demo instance runs
at a time.

A question is what is said while the panel's button is held (`talk`), unless
the run was started with a wake word.
"""
from __future__ import annotations

import asyncio
import threading

from pantherlake_ai_core.engine import Engine
from voice_assistant import session

from . import activity, events, metrics, worker

_DEMO_ID = "voice-assistant"
_VOICE = "voice"
_VOICE_DEVICE = "CPU"  # voice_clone_studio.voice_model.TTS_DEVICE, without loading the brick to ask

BUTTON = "button"
WAKE_WORD = "wake_word"


class VoiceAssistantRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._talk = threading.Event()
        self.error: str | None = None
        self.trigger = BUTTON
        # Ready for a question: the models are loaded and none is being heard
        # or answered. What the panel's button goes by.
        self.listening = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def state(self) -> dict:
        running = self.running
        return {"running": running, "trigger": self.trigger, "listening": running and self.listening}

    def talk(self, down: bool) -> bool:
        """The panel's button going down or coming up. False when it means
        nothing: no run, a run on a wake word, or an answer in the making."""
        if not self.running or self.trigger != BUTTON or (down and not self.listening):
            return False
        if down:
            self._talk.set()
        else:
            self._talk.clear()
        return True

    def start(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        audio_device: str | None,
        engine: Engine,
        whisper_model_size: str,
        compute_device: str,
        wake_word: str | None,
        wake_threshold: float,
        speak_replies: bool,
        llm_model: str | None = None,
    ) -> None:
        worker.refuse_if_busy(_DEMO_ID, self._thread, self._stop_event)

        self.error = None
        self._stop_event = threading.Event()
        stop_event = self._stop_event
        self.trigger = trigger = WAKE_WORD if wake_word else BUTTON
        self.listening = False
        self._talk.clear()
        waiting = "Listening for the wake word..." if wake_word else "Hold the button and ask."

        def emit(message: dict) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(message), loop)

        def on_ready() -> None:
            self.listening = True
            events.set_phase(_DEMO_ID, "running", waiting)
            emit({"type": "ready", "trigger": trigger})

        def on_wake() -> None:
            # From here to on_idle a question is heard, answered and the
            # answer said: a new press of the button is not a new question.
            self.listening = False
            emit({"type": "wake", "trigger": trigger})

        def on_idle() -> None:
            self.listening = True
            emit({"type": "idle"})

        # The voice is made on the CPU whatever chip listens and answers
        # (voice_model.TTS_DEVICE): when that is another chip, the hardware
        # panel shows the voice on its own row, under the chip it is on.
        voice_elsewhere = speak_replies and compute_device.upper() != _VOICE_DEVICE

        def target() -> None:
            activity.set_active(_DEMO_ID, engine=engine.value, device=compute_device)
            if voice_elsewhere:
                activity.set_active(_DEMO_ID, engine=engine.value, device=_VOICE_DEVICE, stage=_VOICE, stage_label="Voice")
            # Several models load before the mic even opens -- say so, rather
            # than claiming to be listening from the very first millisecond.
            events.set_phase(
                _DEMO_ID, "loading",
                f"Loading models (engine={engine.value}, device={compute_device}"
                + (f"; the voice on the {_VOICE_DEVICE}" if voice_elsewhere else "") + ")...",
            )
            try:
                session.run(
                    audio_device=audio_device,
                    engine=engine,
                    whisper_model_size=whisper_model_size,
                    compute_device=compute_device,
                    wake_word=wake_word,
                    wake_threshold=wake_threshold,
                    talk=self._talk,
                    llm_model=llm_model,
                    on_wake=on_wake,
                    on_missed=lambda reason: emit({"type": "missed", "reason": reason}),
                    on_heard=lambda text: emit({"type": "heard", "text": text}),
                    on_reply=lambda text: emit({"type": "reply", "text": text}),
                    on_idle=on_idle,
                    on_ready=on_ready,
                    on_stats=lambda stats: metrics.report(_DEMO_ID, stats.tokens_per_second, "tok/s", detail="last reply"),
                    speak_replies=speak_replies,
                    stop_event=stop_event,
                )
            except Exception as exc:  # surfaced to the UI, not silently dropped
                self.error = str(exc)
                events.set_phase(_DEMO_ID, "error", str(exc))
                emit({"type": "error", "message": str(exc)})
            else:
                events.clear_phase(_DEMO_ID)
            finally:
                self.listening = False
                self._talk.clear()
                activity.clear_active(_DEMO_ID)
                activity.clear_active(_DEMO_ID, stage=_VOICE)
                emit({"type": "stopped"})

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not worker.request_stop(_DEMO_ID, self._thread, self._stop_event):
            return
        self._thread = None
