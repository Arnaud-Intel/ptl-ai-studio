"""Runs the voice-assistant brick's wake/listen/think/speak loop on a
background thread and forwards each event into an asyncio queue the web UI
drains over a WebSocket -- same shape as LiveTranslationRunner. A single
demo instance runs at a time.
"""
from __future__ import annotations

import asyncio
import threading

from pantherlake_ai_core.engine import Engine
from voice_assistant import session

from . import activity, events
from .errors import Conflict

_DEMO_ID = "voice-assistant"


class VoiceAssistantRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        audio_device: str | None,
        engine: Engine,
        whisper_model_size: str,
        compute_device: str,
        wake_word: str,
        wake_threshold: float,
        speak_replies: bool,
    ) -> None:
        if self.running:
            raise Conflict("voice-assistant is already running")

        self.error = None
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def emit(message: dict) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(message), loop)

        def on_ready() -> None:
            events.set_phase(_DEMO_ID, "running", "Listening for the wake word...")
            emit({"type": "ready"})

        def target() -> None:
            activity.set_active(_DEMO_ID, engine=engine.value, device=compute_device)
            # Four models load before the mic even opens -- say so, rather
            # than claiming to be listening from the very first millisecond.
            events.set_phase(_DEMO_ID, "loading", f"Loading models (engine={engine.value}, device={compute_device})...")
            try:
                session.run(
                    audio_device=audio_device,
                    engine=engine,
                    whisper_model_size=whisper_model_size,
                    compute_device=compute_device,
                    wake_word=wake_word,
                    wake_threshold=wake_threshold,
                    on_wake=lambda: emit({"type": "wake"}),
                    on_heard=lambda text: emit({"type": "heard", "text": text}),
                    on_reply=lambda text: emit({"type": "reply", "text": text}),
                    on_ready=on_ready,
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
                activity.clear_active(_DEMO_ID)
                emit({"type": "stopped"})

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        thread = self._thread
        if thread is not None:
            # Wait for the loop to actually exit, so `running` only turns
            # false once it has -- otherwise a quick Stop -> Start overlaps two
            # threads on the same mic/queue. A thread still inside a long
            # model load keeps `running` true until it gets out.
            thread.join(timeout=3.0)
            if thread.is_alive():
                return
        self._thread = None
