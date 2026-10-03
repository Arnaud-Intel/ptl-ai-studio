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
import re
import threading
from typing import Callable

from doc_qa.engine_factory import TEMPLATE_TOKENS, create_llm
from live_translation import pipeline as live_translation_pipeline
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import TranslationResult, combine_stats

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

# A meeting longer than the model's window is summarised part by part, then
# the parts are merged. The window is small on the NPU (it compiles the LLM
# for a fixed prompt length) and finite everywhere, so this is the only way
# an hour-long meeting gets notes at all.
# The single-call prompt, not a thinner copy of it: a shorter one tried
# first had the 1.5B model file every commitment under the summary and
# answer "None in this part" for action items -- the guidance above about
# what counts as an action item is what makes it list them.
_PART_SYSTEM_PROMPT = (
    "The transcript below is ONE PART of a longer meeting; write notes on this part only. "
    + _NOTES_SYSTEM_PROMPT
    + " Write each action item as 'Name: task (deadline)', with the name the transcript gives."
)
_MERGE_SYSTEM_PROMPT = (
    "You are combining notes written separately on consecutive parts of ONE "
    "meeting into a single set of meeting notes. Produce:\n"
    "1. A short running summary (2-6 bullet points) of the whole meeting.\n"
    "2. An 'Action items' section with every action item from the parts, "
    "merging duplicates. Keep each item's owner name and deadline exactly as "
    "the parts give them -- an action item without its owner is useless. "
    "Only write 'None identified' if no part had any.\n"
    "Only use what the part notes say. Don't invent details."
)
_PART_MAX_TOKENS = 350
# Per-item token counts add up to slightly less than the joined text's
# count; leave room so a piece measured as fitting still fits.
_SPLIT_HEADROOM = 0.95


_NONE_LINE = re.compile(r"^[\s\-*•]*none (identified|in this part)\.?[\s*]*$", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^\s*([-*•]|\d+\.)\s*\S")


def without_contradicting_none(text: str) -> str:
    """Drop a 'None identified' line when the notes do list action items.

    The prompt says not to write it then; the 1.5B model writes it anyway,
    typically at the end of a long list, where it reads as a contradiction
    on screen. Only the action-items section is looked at, and a genuinely
    empty list keeps its 'None identified'."""
    lines = text.splitlines()
    heading = next((i for i, line in enumerate(lines) if "action item" in line.lower()), None)
    if heading is None:
        return text
    after = lines[heading + 1:]
    if not any(_LIST_ITEM.match(line) and not _NONE_LINE.match(line) for line in after):
        return text
    return "\n".join(lines[: heading + 1] + [line for line in after if not _NONE_LINE.match(line)]).rstrip()


def _halve_until_fits(text: str, count_tokens: Callable[[str], int], budget: int) -> list[str]:
    if count_tokens(text) <= budget:
        return [text]
    words = text.split()
    if len(words) < 2:
        return [text]  # can't cut further; the model will say it's too long
    middle = len(words) // 2
    return _halve_until_fits(" ".join(words[:middle]), count_tokens, budget) + _halve_until_fits(
        " ".join(words[middle:]), count_tokens, budget
    )


def split_to_fit(
    items: list[str], count_tokens: Callable[[str], int], budget: int, separator: str = "\n"
) -> list[str]:
    """Consecutive items joined into as few pieces as fit `budget` tokens
    each, in order. An item too big on its own is cut by words -- a 14-second
    utterance never is, but a pasted wall of text could be."""
    limit = int(budget * _SPLIT_HEADROOM)
    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for item in items:
        for piece in _halve_until_fits(item, count_tokens, limit):
            tokens = count_tokens(piece) + 1
            if current and used + tokens > limit:
                pieces.append(separator.join(current))
                current, used = [], 0
            current.append(piece)
            used += tokens
    if current:
        pieces.append(separator.join(current))
    return pieces


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
                realtime_factor=(
                    round(result.audio_seconds / result.processing_seconds, 1)
                    if result.audio_seconds and result.processing_seconds
                    else None
                ),
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
        return "\n".join(self._transcript_lines())

    def _transcript_lines(self) -> list[str]:
        with self._lock:
            return [f"[{line.timestamp}] {line.text}" for line in self._transcript]

    def generate_notes(
        self,
        max_tokens: int = 600,
        *,
        on_ready: Callable[[], None] | None = None,
        on_downloading: Callable[[], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> MeetingNotes:
        """`on_ready`/`on_downloading`, if given, mirror `transcribe`'s: the
        notes LLM is also lazy (built on first call, reused after), so a
        caller that wants to distinguish "building the notes LLM" from
        "actually generating notes" needs the same seam here.
        `on_progress(step, steps)`, if given, fires before each model call
        when a long meeting has to be summarised in parts."""
        lines = self._transcript_lines()
        line_count = len(lines)
        transcript_text = "\n".join(lines)
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

        llm = self._llm
        budget = llm.prompt_budget(max_tokens) - llm.count_tokens(_NOTES_SYSTEM_PROMPT) - TEMPLATE_TOKENS
        if llm.count_tokens(transcript_text) <= budget:
            notes_text = llm.answer(_NOTES_SYSTEM_PROMPT, transcript_text, max_tokens=max_tokens)
            return MeetingNotes(
                text=without_contradicting_none(notes_text),
                transcript_line_count=line_count,
                stats=getattr(llm, "last_stats", None),
            )
        return self._notes_in_parts(llm, lines, line_count, max_tokens, on_progress)

    def _notes_in_parts(self, llm, lines, line_count, max_tokens, on_progress) -> MeetingNotes:
        part_budget = llm.prompt_budget(_PART_MAX_TOKENS) - llm.count_tokens(_PART_SYSTEM_PROMPT) - TEMPLATE_TOKENS
        parts = split_to_fit(lines, llm.count_tokens, part_budget)
        steps = len(parts) + 1  # each part, then the merge
        partial = []
        stats = []  # one entry per model call, combined at the end

        def ask(system_prompt: str, text: str, tokens: int) -> str:
            answer = llm.answer(system_prompt, text, max_tokens=tokens)
            stats.append(getattr(llm, "last_stats", None))
            return answer

        for index, part in enumerate(parts, 1):
            if on_progress is not None:
                on_progress(index, steps)
            partial.append(ask(_PART_SYSTEM_PROMPT, part, _PART_MAX_TOKENS))

        merge_budget = llm.prompt_budget(max_tokens) - llm.count_tokens(_MERGE_SYSTEM_PROMPT) - TEMPLATE_TOKENS
        notes = [f"Part {i} of {len(partial)}:\n{text}" for i, text in enumerate(partial, 1)]
        groups = split_to_fit(notes, llm.count_tokens, merge_budget, separator="\n\n")
        while len(groups) > 1:
            # Too many parts to merge at once: merge them in groups first. A
            # group of one means a single part's notes fill the window --
            # merging can't make progress then, so say so instead of looping.
            # (More groups than notes is the same dead end: one part's notes
            # were bigger than the window and got cut up.)
            if len(groups) >= len(notes):
                raise RuntimeError("This meeting is too long to condense into one set of notes on this device.")
            notes = [ask(_MERGE_SYSTEM_PROMPT, group, _PART_MAX_TOKENS) for group in groups]
            groups = split_to_fit(notes, llm.count_tokens, merge_budget, separator="\n\n")
        if on_progress is not None:
            on_progress(steps, steps)
        notes_text = ask(_MERGE_SYSTEM_PROMPT, groups[0], max_tokens)
        return MeetingNotes(
            text=without_contradicting_none(notes_text),
            transcript_line_count=line_count,
            parts=len(parts),
            stats=combine_stats([s for s in stats if s is not None]),
        )
