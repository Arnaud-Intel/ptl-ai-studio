"""Combines live-translation's transcription pipeline with doc-qa's local
LLM to produce running meeting notes and action items.

This brick deliberately has no transcriber or LLM code of its own: it
imports `live_translation.pipeline` for the capture->segment->transcribe
loop and `doc_qa.engine_factory` for the LLM, the same way the launcher
composes bricks rather than re-implementing them. A third Whisper wrapper
or llama.cpp wrapper would just be a bug generator with extra steps.
"""
from __future__ import annotations

import datetime as dt
import threading
from typing import Callable

from doc_qa.engine_factory import create_llm
from live_translation import pipeline as live_translation_pipeline
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import TranslationResult

from .types import MeetingNotes, TranscriptLine

_MIN_WORDS_FOR_NOTES = 25

_NOTES_SYSTEM_PROMPT = (
    "You are an assistant that writes concise meeting notes from a raw "
    "speech transcript. Given the transcript so far, produce:\n"
    "1. A short running summary (2-6 bullet points) of what's been discussed.\n"
    "2. An 'Action items' section: one bullet per concrete commitment, task, "
    "or deadline anyone in the transcript stated -- including first-person "
    "commitments like 'I need to write the tests by Friday' or 'I'll fix "
    "that by Wednesday', and follow-ups like 'let's meet again Monday'. "
    "Treat any sentence naming a task plus a timeframe, or promising to do "
    "something, as an action item -- don't restrict this to items explicitly "
    "labeled as tasks. If you list at least one real action item, do not "
    "also write 'None identified' -- only write that line if the list would "
    "otherwise be completely empty.\n"
    "Only use what's actually in the transcript. Don't invent details, "
    "attendees, or decisions that weren't said."
)


class MeetingSession:
    """One live meeting: accumulates a transcript as audio comes in, and
    can generate notes from everything accumulated so far, any time."""

    def __init__(self, engine: Engine, *, compute_device: str, whisper_model_size: str):
        self.engine = engine
        self.compute_device = compute_device
        self.whisper_model_size = whisper_model_size
        self._transcript: list[TranscriptLine] = []
        self._lock = threading.Lock()
        self._llm = None  # built lazily -- no reason to load it if notes are never requested

    def transcribe(
        self,
        *,
        source: str,
        audio_device: str | None,
        on_line: Callable[[TranscriptLine], None],
        on_ready: Callable[[], None] | None = None,
        on_downloading: Callable[[], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Blocks the calling thread until stop_event is set (or forever if
        none given), appending each transcribed utterance to the running
        transcript and forwarding it to `on_line`. `on_ready`, if given,
        fires once the whisper model is loaded and capture is about to
        start -- __init__ doesn't load it (the LLM is also lazy, built on
        first `generate_notes()` call), so this call is where that actually
        happens. `on_downloading`, if given, fires before that load has to
        fetch the model from the network rather than just reading local disk."""

        def handle_result(result: TranslationResult) -> None:
            line = TranscriptLine(
                timestamp=dt.datetime.now().strftime("%H:%M:%S"),
                text=result.text,
                detected_language=result.detected_language,
            )
            with self._lock:
                self._transcript.append(line)
            on_line(line)

        live_translation_pipeline.run(
            source=source,
            audio_device=audio_device,
            engine=self.engine,
            model_size=self.whisper_model_size,
            compute_device=self.compute_device,
            on_result=handle_result,
            on_ready=on_ready,
            on_downloading=on_downloading,
            stop_event=stop_event,
        )

    def transcript_text(self) -> str:
        with self._lock:
            return "\n".join(f"[{line.timestamp}] {line.text}" for line in self._transcript)

    def generate_notes(
        self,
        max_tokens: int = 600,
        *,
        on_ready: Callable[[], None] | None = None,
        on_downloading: Callable[[], None] | None = None,
    ) -> MeetingNotes:
        """`on_ready`/`on_downloading`, if given, mirror `transcribe`'s: the
        notes LLM is also lazy (built on first call, reused after), so a
        caller that wants to distinguish "building the notes LLM" from
        "actually generating notes" needs the same seam here."""
        with self._lock:
            line_count = len(self._transcript)
        transcript_text = self.transcript_text()
        if not transcript_text.strip():
            raise RuntimeError("No transcript yet -- start capturing audio first.")

        # Small local LLMs are not reliably groundable on very thin input:
        # tested with a 2-line off-topic transcript, the model filled the
        # gap by inventing named attendees and action items out of nothing
        # rather than admitting there wasn't enough to summarize. A prompt
        # instruction alone didn't stop it, so this is a hard, deterministic
        # gate instead of a probabilistic mitigation -- no LLM call at all
        # (and therefore no chance to hallucinate) until there's enough
        # transcript to actually ground a summary in.
        word_count = len(transcript_text.split())
        if word_count < _MIN_WORDS_FOR_NOTES:
            raise RuntimeError(
                f"Not enough transcript yet to generate reliable notes ({word_count} words so far, "
                f"need at least {_MIN_WORDS_FOR_NOTES}) -- small local models tend to invent content "
                "rather than admit there's too little to summarize, so this waits for more to be said."
            )

        if self._llm is None:
            self._llm = create_llm(self.engine, device=self.compute_device, on_downloading=on_downloading)
        if on_ready is not None:
            on_ready()

        notes_text = self._llm.answer(_NOTES_SYSTEM_PROMPT, transcript_text, max_tokens=max_tokens)
        return MeetingNotes(text=notes_text, transcript_line_count=line_count)
