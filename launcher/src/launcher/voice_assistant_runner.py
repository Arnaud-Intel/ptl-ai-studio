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

from . import activity, events, metrics, worker

_DEMO_ID = "voice-assistant"
_VOICE = "voice"
_VOICE_DEVICE = "CPU"  # voice_clone_studio.voice_model.TTS_DEVICE, without loading the brick to ask


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
        llm_model: str | None = None,
    ) -> None:
        worker.refuse_if_busy(_DEMO_ID, self._thread, self._stop_event)

        self.error = None
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def emit(message: dict) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(message), loop)

        def on_ready() -> None:
            events.set_phase(_DEMO_ID, "running", "Listening for the wake word...")
            emit({"type": "ready"})

        # The voice is made on the CPU whatever chip listens and answers
        # (voice_model.TTS_DEVICE): when that is another chip, the hardware
        # panel shows the voice on its own row, under the chip it is on.
        voice_elsewhere = speak_replies and compute_device.upper() != _VOICE_DEVICE

        def target() -> None:
            activity.set_active(_DEMO_ID, engine=engine.value, device=compute_device)
            if voice_elsewhere:
                activity.set_active(_DEMO_ID, engine=engine.value, device=_VOICE_DEVICE, stage=_VOICE, stage_label="Voice")
            # Four models load before the mic even opens -- say so, rather
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
                    llm_model=llm_model,
                    on_wake=lambda: emit({"type": "wake"}),
                    on_heard=lambda text: emit({"type": "heard", "text": text}),
                    on_reply=lambda text: emit({"type": "reply", "text": text}),
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
                activity.clear_active(_DEMO_ID)
                activity.clear_active(_DEMO_ID, stage=_VOICE)
                emit({"type": "stopped"})

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not worker.request_stop(_DEMO_ID, self._thread, self._stop_event):
            return
        self._thread = None
