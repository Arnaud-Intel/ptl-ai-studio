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

from . import activity, events, generation, metrics, worker
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

    @property
    def engine(self) -> Engine | None:
        """The engine of the meeting in hand, None before the first one."""
        return self._engine

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
        spoken_language: str | None = None,
        notes_device: str | None = None,
    ) -> None:
        """`compute_device` transcribes; `notes_device` (the same one if
        not given) writes the notes."""
        with self._state_lock:
            worker.refuse_if_busy(_DEMO_ID, self._thread, self._stop_event)

            self.error = None
            self._engine = engine
            self._compute_device = compute_device
            self._session = MeetingSession(
                engine,
                compute_device=compute_device,
                whisper_model_size=whisper_model_size,
                spoken_language=spoken_language,
                notes_device=notes_device,
            )
            self._stop_event = threading.Event()
            stop_event = self._stop_event
            session = self._session

            def on_line(line: TranscriptLine) -> None:
                if line.realtime_factor:
                    metrics.report(_DEMO_ID, line.realtime_factor, "x real time")
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

    def adopt(
        self, lines: list[dict], *, engine: Engine, notes_device: str, whisper_model_size: str
    ) -> list[TranscriptLine]:
        """Take a transcript captured elsewhere (live translation's: dicts
        with `timestamp`, `text`, `detected_language`) as this brick's
        meeting, ready for generate_notes() on `engine`/`notes_device`.
        Returns the lines taken.

        One recording, two uses: without this, a summary of what live
        translation heard meant running this brick beside it -- a second
        speech model listening to the same microphone for the same words.
        Refused while this brick is transcribing a meeting of its own,
        which the new transcript would replace."""
        with self._state_lock:
            if self.running:
                raise Conflict(
                    "Meeting Notes is transcribing a meeting of its own: stop it before summarising another transcript."
                )
            transcript = [
                TranscriptLine(
                    timestamp=str(line.get("timestamp") or ""),
                    text=str(line["text"]).strip(),
                    detected_language=str(line.get("detected_language") or "auto"),
                )
                for line in lines
                if str(line.get("text") or "").strip()
            ]
            if not transcript:
                raise Conflict("Nothing has been transcribed yet, so there is nothing to summarise.")
            self.error = None
            session = self._session
            if session is not None and session.engine == engine:
                session.replace_transcript(transcript)
                session.write_notes_on(notes_device)  # the same chip as before keeps the model loaded
            else:
                # Nothing is transcribed here, so the session's own device is only a name.
                self._session = MeetingSession(
                    engine, compute_device=notes_device, whisper_model_size=whisper_model_size, transcript=transcript
                )
            self._engine = engine
            self._compute_device = notes_device
            return transcript

    def stop(self) -> None:
        with self._state_lock:
            if not worker.request_stop(_DEMO_ID, self._thread, self._stop_event):
                return
            self._thread = None

    def generate_notes(self, notes_device: str | None = None) -> MeetingNotes:
        """Blocking -- call via run_in_threadpool. Works while still
        transcribing (notes reflect everything captured so far) or after
        stopping (the session and its transcript outlive the thread).
        `notes_device`, if given, is the chip to write them on from now on:
        the one thing about a meeting that can change while it runs."""
        with self._state_lock:
            session = self._session
            engine = self._engine
        if session is None or engine is None:
            raise Conflict("Start capturing audio first.")

        def on_ready() -> None:
            events.set_phase(_DEMO_ID, "running", "Generating notes...", stage="notes")

        def on_downloading() -> None:
            events.set_phase(_DEMO_ID, "loading", "Downloading notes model (first run only)...", stage="notes")

        def on_progress(step: int, steps: int) -> None:
            # A long meeting is summarised part by part; say where it is.
            message = f"Long meeting: summarising part {step} of {steps - 1}..." if step < steps else "Merging the parts into one set of notes..."
            events.set_phase(_DEMO_ID, "running", message, stage="notes")

        # The notes stage gets its own activity entry, so it never clears
        # the transcription thread's while that is still running.
        live = generation.get(_DEMO_ID, "notes")  # its own stage: stopping it leaves the transcription running
        with self._notes_lock:
            if notes_device:
                session.write_notes_on(notes_device)
            device = session.notes_device
            activity.set_active(_DEMO_ID, engine=engine.value, device=device, stage="notes", stage_label="notes")
            events.set_phase(_DEMO_ID, "loading", f"Preparing notes model on {device}...", stage="notes")
            try:
                notes = session.generate_notes(
                    on_ready=on_ready, on_downloading=on_downloading, on_progress=on_progress, control=live.begin()
                )
            except Exception as exc:
                events.set_phase(_DEMO_ID, "error", str(exc), stage="notes")
                raise
            finally:
                live.end()
                activity.clear_active(_DEMO_ID, stage="notes")
        events.clear_phase(_DEMO_ID, stage="notes")
        if notes.stats is not None:
            # Sticky: the notes stage is over, but its speed is still worth showing.
            metrics.report(_DEMO_ID, notes.stats.tokens_per_second, "tok/s", stage="notes", sticky=True, detail="notes")
        return notes
