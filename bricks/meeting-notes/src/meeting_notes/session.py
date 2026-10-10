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

from doc_qa import language_models
from doc_qa.engine_factory import TEMPLATE_TOKENS, create_llm
from live_translation import pipeline as live_translation_pipeline
from pantherlake_ai_core import npu
from pantherlake_ai_core.engine import Engine, list_openvino_devices
from pantherlake_ai_core.types import GenerationControl, TranslationResult, combine_stats, say, stopped

from .types import MeetingNotes, TranscriptLine

_MIN_WORDS_FOR_NOTES = 25

# The notes model on the OpenVINO engine: Qwen3-8B. doc-qa's default
# (Qwen2.5-1.5B) is too small for this job. On four test meetings with 20
# stated tasks (2026-10-07, XPS 14, integrated GPU, one or two runs a cell,
# the draw being seeded) it listed 6, answered "None identified" to a
# meeting with four, and made up deadlines. Qwen3-8B listed 17 and 18 and
# gave every named task to the right person, at 23 tokens/s -- 6 to 10 s a
# meeting. Qwen3-4B was faster (39 tokens/s) and listed 16, but gave two
# tasks to the wrong person in every run. Qwen3-Coder-30B, the model the
# coding bricks load, was no better at this (14-17) for 17 GB.
#
# The NPU needs a build quantised for it: the standard one does not compile
# there ("Can't convert 44 Bit to Byte") and the 4B answers garbage. The
# channel-wise build runs at 19 tokens/s and is weaker -- 11 of 20 -- but
# still ahead of the 1.5B on the same chip (8, with invented owners).
# Nothing else published does better there (tried the same evening): the
# int8 builds of Qwen3-4B and -8B compile in 66 and 93 s and then wrote no
# answer in 13 and 8 minutes; Phi-3.5-mini channel-wise writes 31 tokens/s
# and invents owners and dates ("John", "Friday, October 26th");
# Mistral-7B-v0.3 channel-wise lists 13 at 20 tokens/s and gives tasks
# deadlines nobody set ("by the end of the current week"). Asking for
# the two sections in two calls found 14 but listed decisions as tasks and
# none of the three tasks of the one real transcript. The channel-wise 8B
# was quantised with no calibration data (its openvino_config.json says
# `"dataset": null`, the standard build's says wikitext2): a calibrated
# channel-wise build is what would close the gap, and nobody publishes one.
# The portable engine keeps doc-qa's small default.
# Which build goes on which chip is said in doc-qa's language_models, for
# every brick that uses this model.
NOTES_MODEL = language_models.CAREFUL.key


def notes_model_repo(engine: Engine, device: str) -> str | None:
    """The model that writes the notes on `device`; None for the LLM
    backend's own default."""
    if engine != Engine.OPENVINO:
        return None
    return language_models.repo_for(NOTES_MODEL, engine, device)


def default_notes_device(engine: Engine, compute_device: str, devices: list[str] | None = None) -> str:
    """The chip the notes are written on when nobody chose one: the NPU if
    the machine has one in working order, else `compute_device`. `devices`
    is the machine's OpenVINO devices, when the caller already has the list.

    Writing notes is the kind of work the NPU is there for -- a few seconds
    now and then -- and it leaves the GPUs to whatever else is on screen.
    The notes it writes are weaker than a GPU's (see above), which is why
    the choice stays visible and can be changed."""
    if engine != Engine.OPENVINO or npu.lost():
        return compute_device
    available = list_openvino_devices() if devices is None else devices
    return "NPU" if any(npu.is_npu(device) for device in available) else compute_device


# What the instructions are for, each line of them earned on the test
# meetings above:
# - The transcript has no speaker names, and the models gave a task to
#   whoever was mentioned last. "Unassigned:" is the honest owner when nobody
#   was named -- which is every task of a meeting dictated by one person.
# - No personal name in the examples: one that was there ("Maria, can you
#   send it?") came back as the owner of every task in a meeting with no
#   Maria in it.
# - Plain text, because the page and the terminal show the notes as written:
#   asterisks and # headings arrived as asterisks and # signs.
# - The summary first: with the action items first the 8B model copied
#   transcript lines out instead of writing notes.
_NOTES_SYSTEM_PROMPT = (
    "You write meeting notes from a raw speech transcript. The transcript is "
    "what a speech recogniser heard: one line per utterance, and no speaker "
    "names. A line was said by whoever was talking at that moment, which is "
    "often not the last person mentioned.\n\n"
    "Write two sections in plain text. No Markdown: no asterisks and no # "
    'headings; start each list line with "- ".\n\n'
    "Summary\n"
    "2 to 6 lines on what was discussed and decided.\n\n"
    "Action items\n"
    "One line per task, commitment or follow-up stated in the meeting, with "
    "its deadline if one was said. This covers first-person commitments "
    '("I\'ll fix that by Wednesday", "I need to write the tests by Friday"), '
    "a request that the person accepted, a task stated for the group "
    '("we will need to estimate the costs"), and the next meeting if one was '
    "agreed. List a commitment here even when the summary mentions it too.\n"
    "Rules for this section:\n"
    "- Start a line with a person's name only when the transcript makes "
    "clear that this person will do it: they were asked by name just before, "
    "they said their own name, or someone named them as the one doing it. "
    'Otherwise start the line with "Unassigned:". Never guess who.\n'
    "- Give days, dates and times exactly as they were said. Never add a "
    "deadline, a date or a month that was not said.\n"
    "- Something explicitly not promised, put off without a date, or needing "
    "nothing done is not an action item.\n"
    '- Write "None identified" only if there is no action item at all.\n\n'
    "Use only what is in the transcript. Do not invent details, attendees or "
    "decisions."
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
    + " Write each action item as 'Name: task (deadline)', or 'Unassigned: task (deadline)' when the"
    " transcript does not make clear who."
)
_MERGE_SYSTEM_PROMPT = (
    "You are combining notes written separately on consecutive parts of ONE "
    "meeting into a single set of meeting notes. Write two sections in plain "
    'text. No Markdown: no asterisks and no # headings; start each list line with "- ".\n\n'
    "Summary\n"
    "2 to 6 lines on the whole meeting.\n\n"
    "Action items\n"
    "Every action item from the parts, merging duplicates. Keep each item's "
    "owner (a name, or 'Unassigned') and its deadline exactly as the parts "
    "give them. Only write 'None identified' if no part had any.\n\n"
    "Use only what the part notes say. Do not invent details."
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


_SECTION_HEADING = re.compile(r"^\s*(?:[-*•#]+\s*)?(?:\d+\.\s*)?\**(summary|action items)\**\s*:?\s*\**\s*$", re.IGNORECASE)


def with_plain_headings(text: str) -> str:
    """The two section headings as plain lines of their own. The
    instructions ask for exactly that, and the NPU's model still writes
    "- Summary" and "- Action items" as if they were items of a list -- on a
    page that shows the notes as written."""
    lines = []
    for line in text.splitlines():
        heading = _SECTION_HEADING.match(line)
        lines.append(heading.group(1).capitalize() if heading else line)
    return "\n".join(lines)


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

    def __init__(
        self,
        engine: Engine,
        *,
        compute_device: str,
        whisper_model_size: str,
        spoken_language: str | None = None,
        transcript: list[TranscriptLine] | None = None,
        notes_device: str | None = None,
    ):
        """`transcript`, if given, is a meeting already transcribed somewhere
        else (a live translation session, say): notes can be generated from
        it straight away, with nothing captured here.

        `notes_device` is the chip the notes are written on, when it is not
        the one that transcribes (`compute_device`): see default_notes_device()."""
        self.engine = engine
        self.compute_device = compute_device
        self.notes_device = notes_device or compute_device
        self.whisper_model_size = whisper_model_size
        self.spoken_language = spoken_language  # None: detected for each utterance
        self._transcript: list[TranscriptLine] = list(transcript or [])
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
            language=self.spoken_language,
            on_result=handle_result,
            on_ready=on_ready,
            on_downloading=on_downloading,
            stop_event=stop_event,
        )

    def replace_transcript(self, transcript: list[TranscriptLine]) -> None:
        """Swap in another meeting's transcript, keeping the notes model this
        session has loaded -- loading it again for every summary would be
        the slow part, and on the NPU a load is the risky one."""
        with self._lock:
            self._transcript = list(transcript)

    def write_notes_on(self, device: str) -> None:
        """Write the notes on `device` from now on. The notes model is loaded
        for one chip, so a different one means loading it again on the next
        generate_notes(); the same one keeps what is loaded."""
        if device == self.notes_device:
            return
        self.notes_device = device
        self._llm = None

    def transcript_text(self) -> str:
        return "\n".join(self._transcript_lines())

    def _transcript_lines(self) -> list[str]:
        with self._lock:
            return [f"[{line.timestamp}] {line.text}" for line in self._transcript]

    def _spoken_lines(self) -> list[str]:
        """What the notes model reads: the words, without each line's time.
        The notes never quote a time of day; given them, the models turned
        them into dates nobody said ("by today (10/07/2024)"), and on the
        NPU's fixed window they are tokens taken from the meeting itself."""
        with self._lock:
            return [line.text for line in self._transcript]

    def generate_notes(
        self,
        max_tokens: int = 600,
        *,
        on_ready: Callable[[], None] | None = None,
        on_downloading: Callable[[], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        control: GenerationControl | None = None,
    ) -> MeetingNotes:
        """`on_ready`/`on_downloading`, if given, mirror `transcribe`'s: the
        notes LLM is also lazy (built on first call, reused after), so a
        caller that wants to distinguish "building the notes LLM" from
        "actually generating notes" needs the same seam here.
        `on_progress(step, steps)`, if given, fires before each model call
        when a long meeting has to be summarised in parts."""
        lines = self._spoken_lines()
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
            self._llm = create_llm(
                self.engine,
                device=self.notes_device,
                model_repo=notes_model_repo(self.engine, self.notes_device),
                on_downloading=on_downloading,
            )
        if on_ready is not None:
            on_ready()

        llm = self._llm
        budget = llm.prompt_budget(max_tokens) - llm.count_tokens(_NOTES_SYSTEM_PROMPT) - TEMPLATE_TOKENS
        if llm.count_tokens(transcript_text) <= budget:
            notes_text = llm.answer(_NOTES_SYSTEM_PROMPT, transcript_text, max_tokens=max_tokens, control=control)
            return MeetingNotes(
                text=without_contradicting_none(with_plain_headings(notes_text)),
                transcript_line_count=line_count,
                stats=getattr(llm, "last_stats", None),
                cancelled=stopped(control),
            )
        return self._notes_in_parts(llm, lines, line_count, max_tokens, on_progress, control)

    def _notes_in_parts(self, llm, lines, line_count, max_tokens, on_progress, control=None) -> MeetingNotes:
        part_budget = llm.prompt_budget(_PART_MAX_TOKENS) - llm.count_tokens(_PART_SYSTEM_PROMPT) - TEMPLATE_TOKENS
        parts = split_to_fit(lines, llm.count_tokens, part_budget)
        steps = len(parts) + 1  # each part, then the merge
        partial = []
        stats = []  # one entry per model call, combined at the end

        def ask(system_prompt: str, text: str, tokens: int) -> str:
            answer = llm.answer(system_prompt, text, max_tokens=tokens, control=control)
            stats.append(getattr(llm, "last_stats", None))
            return answer

        def stopped_notes() -> MeetingNotes:
            """Asked to stop part-way: the notes on the parts done so far,
            marked as not being the finished notes."""
            return MeetingNotes(
                text="\n\n".join(f"Part {i} of {len(parts)}:\n{text}" for i, text in enumerate(partial, 1)),
                transcript_line_count=line_count,
                parts=len(parts),
                stats=combine_stats([s for s in stats if s is not None]),
                cancelled=True,
            )

        # Each part's notes go to `control` under a heading as they are
        # written: someone watching a long meeting being summarised sees the
        # work, and the finished notes replace it at the end.
        for index, part in enumerate(parts, 1):
            if stopped(control):
                return stopped_notes()
            if on_progress is not None:
                on_progress(index, steps)
            say(control, f"\n\n-- Part {index} of {len(parts)} --\n\n")
            partial.append(ask(_PART_SYSTEM_PROMPT, part, _PART_MAX_TOKENS))
        if stopped(control):
            return stopped_notes()

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
            notes = []
            for group in groups:
                if stopped(control):
                    return stopped_notes()
                notes.append(ask(_MERGE_SYSTEM_PROMPT, group, _PART_MAX_TOKENS))
            groups = split_to_fit(notes, llm.count_tokens, merge_budget, separator="\n\n")
        if on_progress is not None:
            on_progress(steps, steps)
        say(control, "\n\n-- All parts together --\n\n")
        notes_text = ask(_MERGE_SYSTEM_PROMPT, groups[0], max_tokens)
        if stopped(control):
            return stopped_notes()
        return MeetingNotes(
            text=without_contradicting_none(with_plain_headings(notes_text)),
            transcript_line_count=line_count,
            parts=len(parts),
            stats=combine_stats([s for s in stats if s is not None]),
        )
