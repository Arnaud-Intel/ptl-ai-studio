"""Runs the live-translation brick's capture/translate loop on a background
thread and forwards each result into an asyncio queue the web UI drains
over a WebSocket. A single demo instance runs at a time.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import threading
import time
from dataclasses import asdict

from live_translation import pipeline
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import TranslationResult

from . import activity, energy, events, metrics, worker

_DEMO_ID = "live-translation"


def _new_transcript_id() -> int:
    # The clock, so that an id is never reused: a page left open across a
    # launcher restart must not take the new transcript for the old one.
    return int(time.time() * 1000)


class LiveTranslationRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self.error: str | None = None
        # Everything heard since the transcript was last cleared. The page
        # used to be the only place it existed, where a reload lost it and
        # nothing else could use it; kept here, Meeting Notes can be handed
        # the lot for a summary. It carries on across Stop and Start on
        # purpose: changing the spoken language mid-meeting takes a restart,
        # and that must not drop the first half of the meeting.
        self._transcript: list[dict] = []
        self._transcript_id = _new_transcript_id()
        self._transcript_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def transcript(self) -> dict:
        """`lines`, oldest first -- each `seq` (1, 2, ...), `timestamp`,
        `text`, `detected_language` -- and the transcript's `id`, which
        changes when it is cleared (and is larger each time, launcher
        restarts included): the two together name a line for good."""
        with self._transcript_lock:
            return {"id": self._transcript_id, "lines": [dict(line) for line in self._transcript]}

    def clear_transcript(self) -> int:
        """Start a new, empty transcript; returns its id. Fine mid-session:
        the lines that follow open the new one."""
        with self._transcript_lock:
            self._transcript = []
            self._transcript_id = max(_new_transcript_id(), self._transcript_id + 1)
            return self._transcript_id

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
        language: str | None = None,
    ) -> None:
        worker.refuse_if_busy(_DEMO_ID, self._thread, self._stop_event)

        self.error = None
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        # Each line carries the energy spent since the previous one (or since
        # the model was ready): listening, voice detection and translation
        # together, above the idle baseline (BACKLOG R18).
        window = [None]

        def on_result(result: TranslationResult) -> None:
            if result.audio_seconds and result.processing_seconds:
                metrics.report(_DEMO_ID, result.audio_seconds / result.processing_seconds, "x real time")
            with self._transcript_lock:
                line = {
                    "seq": len(self._transcript) + 1,
                    "timestamp": dt.datetime.now().strftime("%H:%M:%S"),
                    "text": result.text,
                    "detected_language": result.detected_language,
                }
                self._transcript.append(line)
                transcript_id = self._transcript_id
            # Which line of which transcript this is: the page draws the
            # transcript from transcript() after a reload, and must be able
            # to tell a line it already has from a new one.
            message = {
                "type": "result",
                **asdict(result),
                "transcript": transcript_id,
                "seq": line["seq"],
                "timestamp": line["timestamp"],
            }
            now = energy.mark()
            if window[0] is not None and now is not None:
                message["energy"] = energy.between(window[0], now, _DEMO_ID)
            window[0] = now
            asyncio.run_coroutine_threadsafe(queue.put(message), loop)

        def on_ready() -> None:
            events.set_phase(_DEMO_ID, "running", "Listening and translating...")
            window[0] = energy.mark()

        def on_downloading() -> None:
            events.set_phase(_DEMO_ID, "loading", f"Downloading model (first run only, engine={engine.value})...")

        def on_recovering(exc: Exception) -> None:
            # Logged in full: this is the only trace a rare driver fault
            # leaves once the session has carried on past it.
            cause = " ".join(str(exc).split())
            events.set_phase(
                _DEMO_ID, "loading", f"An utterance failed on {compute_device}; reloading the model and retrying ({cause})"
            )

        def on_device(device: str) -> None:
            # The model moved mid-session (the NPU was reset under it): the
            # hardware panel has to show the chip doing the work now.
            activity.set_active(_DEMO_ID, engine=engine.value, device=device)
            events.set_phase(
                _DEMO_ID, "running", f"{compute_device} stopped responding and was taken out of use; carrying on on {device}."
            )

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
                    language=language,
                    on_result=on_result,
                    on_ready=on_ready,
                    on_downloading=on_downloading,
                    on_recovering=on_recovering,
                    on_device=on_device,
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
        if not worker.request_stop(_DEMO_ID, self._thread, self._stop_event):
            return
        self._thread = None
