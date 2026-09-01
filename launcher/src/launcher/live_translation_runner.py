"""Runs the live-translation brick's capture/translate loop on a background
thread and forwards each result into an asyncio queue the web UI drains
over a WebSocket. A single demo instance runs at a time.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict

from live_translation import pipeline
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import TranslationResult

from . import activity, events

_DEMO_ID = "live-translation"


class LiveTranslationRunner:
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
        source: str,
        audio_device: str | None,
        engine: Engine,
        model_size: str,
        compute_device: str,
    ) -> None:
        if self.running:
            raise RuntimeError("live-translation is already running")

        self.error = None
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def on_result(result: TranslationResult) -> None:
            asyncio.run_coroutine_threadsafe(queue.put({"type": "result", **asdict(result)}), loop)

        def on_ready() -> None:
            events.set_phase(_DEMO_ID, "running", "Listening and translating...")

        def target() -> None:
            activity.set_active(_DEMO_ID, engine=engine.value, device=compute_device)
            events.set_phase(_DEMO_ID, "loading", f"Loading model (engine={engine.value}, device={compute_device})...")
            try:
                pipeline.run(
                    source=source,
                    audio_device=audio_device,
                    engine=engine,
                    model_size=model_size,
                    compute_device=compute_device,
                    on_result=on_result,
                    on_ready=on_ready,
                    stop_event=stop_event,
                )
            except Exception as exc:  # surfaced to the UI, not silently dropped
                self.error = str(exc)
                events.set_phase(_DEMO_ID, "error", str(exc))
                asyncio.run_coroutine_threadsafe(queue.put({"type": "error", "message": str(exc)}), loop)
            else:
                events.clear_phase(_DEMO_ID)
            finally:
                activity.clear_active(_DEMO_ID)
                asyncio.run_coroutine_threadsafe(queue.put({"type": "stopped"}), loop)

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        self._thread = None
