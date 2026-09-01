"""Runs the meeting-notes brick's transcription loop on a background
thread (same shape as live_translation_runner.py) and exposes on-demand
notes generation (same shape as doc_qa_runner.py) on top of the same
session. This brick is a stream and a request/response layered together,
because that's genuinely what "live transcript, notes on demand" is.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict

from meeting_notes.session import MeetingSession
from meeting_notes.types import MeetingNotes, TranscriptLine
from pantherlake_ai_core.engine import Engine

from . import activity, events

_DEMO_ID = "meeting-notes"


class MeetingNotesRunner:
    def __init__(self) -> None:
        self._session: MeetingSession | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._engine: Engine | None = None
        self._compute_device: str | None = None
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
        compute_device: str,
        whisper_model_size: str,
    ) -> None:
        if self.running:
            raise RuntimeError("meeting-notes is already running")

        self.error = None
        self._engine = engine
        self._compute_device = compute_device
        self._session = MeetingSession(engine, compute_device=compute_device, whisper_model_size=whisper_model_size)
        self._stop_event = threading.Event()
        stop_event = self._stop_event
        session = self._session

        def on_line(line: TranscriptLine) -> None:
            asyncio.run_coroutine_threadsafe(queue.put({"type": "line", **asdict(line)}), loop)

        def on_ready() -> None:
            events.set_phase(_DEMO_ID, "running", "Transcribing...")

        def on_downloading() -> None:
            events.set_phase(_DEMO_ID, "loading", f"Downloading model (first run only, engine={engine.value})...")

        def target() -> None:
            activity.set_active(_DEMO_ID, engine=engine.value, device=compute_device)
            events.set_phase(_DEMO_ID, "loading", f"Loading model (engine={engine.value}, device={compute_device})...")
            try:
                session.transcribe(
                    source=source,
                    audio_device=audio_device,
                    on_line=on_line,
                    on_ready=on_ready,
                    on_downloading=on_downloading,
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

    def generate_notes(self) -> MeetingNotes:
        """Blocking -- call via run_in_threadpool. Works while still
        transcribing (notes reflect everything captured so far) or after
        stopping (the session and its transcript outlive the thread)."""
        if self._session is None:
            raise RuntimeError("Start capturing audio first.")
        if self._engine is not None:
            activity.set_active(_DEMO_ID, engine=self._engine.value, device=self._compute_device)

        def on_ready() -> None:
            events.set_phase(f"{_DEMO_ID}:notes", "running", "Generating notes...")

        def on_downloading() -> None:
            events.set_phase(f"{_DEMO_ID}:notes", "loading", "Downloading notes model (first run only)...")

        events.set_phase(f"{_DEMO_ID}:notes", "loading", "Preparing notes model...")
        try:
            notes = self._session.generate_notes(on_ready=on_ready, on_downloading=on_downloading)
        except Exception as exc:
            events.set_phase(f"{_DEMO_ID}:notes", "error", str(exc))
            raise
        finally:
            if not self.running:
                activity.clear_active(_DEMO_ID)
        events.clear_phase(f"{_DEMO_ID}:notes")
        return notes
