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

from . import activity, events, worker
from .errors import Conflict

_DEMO_ID = "meeting-notes"


class MeetingNotesRunner:
    def __init__(self) -> None:
        self._session: MeetingSession | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._engine: Engine | None = None
        self._compute_device: str | None = None
        self.error: str | None = None
        # Guards the session/thread state (start vs. stop vs. a notes
        # request racing each other); notes generation itself runs outside
        # it so Stop stays responsive during a long LLM call.
        self._state_lock = threading.Lock()
        # Serializes generate_notes calls: the session builds its LLM lazily
        # on the first one, and two at once would build it twice.
        self._notes_lock = threading.Lock()

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
        with self._state_lock:
            worker.refuse_if_busy(_DEMO_ID, self._thread, self._stop_event)

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
        with self._state_lock:
            if not worker.request_stop(_DEMO_ID, self._thread, self._stop_event):
                return
            self._thread = None

    def generate_notes(self) -> MeetingNotes:
        """Blocking -- call via run_in_threadpool. Works while still
        transcribing (notes reflect everything captured so far) or after
        stopping (the session and its transcript outlive the thread)."""
        with self._state_lock:
            session = self._session
            engine = self._engine
            device = self._compute_device
        if session is None or engine is None:
            raise Conflict("Start capturing audio first.")

        def on_ready() -> None:
            events.set_phase(_DEMO_ID, "running", "Generating notes...", stage="notes")

        def on_downloading() -> None:
            events.set_phase(_DEMO_ID, "loading", "Downloading notes model (first run only)...", stage="notes")

        # The notes stage gets its own activity entry, so it never clears
        # the transcription thread's while that is still running.
        activity.set_active(_DEMO_ID, engine=engine.value, device=device, stage="notes", stage_label="notes")
        events.set_phase(_DEMO_ID, "loading", "Preparing notes model...", stage="notes")
        with self._notes_lock:
            try:
                notes = session.generate_notes(on_ready=on_ready, on_downloading=on_downloading)
            except Exception as exc:
                events.set_phase(_DEMO_ID, "error", str(exc), stage="notes")
                raise
            finally:
                activity.clear_active(_DEMO_ID, stage="notes")
        events.clear_phase(_DEMO_ID, stage="notes")
        return notes
