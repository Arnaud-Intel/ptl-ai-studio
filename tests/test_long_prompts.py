"""Prompts longer than a model's window: the NPU compiles its LLM for a fixed
prompt length, so a long meeting or a few long passages used to fail with a
runtime assertion ("Stateful LLM pipeline on NPU may only process prompts
... up to 1024 tokens. 1599 is passed")."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from doc_qa.engine_factory import PromptTooLong
from doc_qa.llm_openvino import NPU_MAX_PROMPT_LEN, NPU_MIN_RESPONSE_LEN, pipeline_config
from doc_qa.pipeline import fit_excerpts
from meeting_notes.session import (
    _MERGE_SYSTEM_PROMPT,
    _NOTES_SYSTEM_PROMPT,
    _PART_SYSTEM_PROMPT,
    MeetingSession,
    split_to_fit,
)
from meeting_notes.types import TranscriptLine
from pantherlake_ai_core.engine import Engine


def _words(text: str) -> int:
    return len(text.split())


# What each kind of call spends before any transcript: its instructions plus
# the template margin. Windows below are sized from these, so a reworded
# prompt can't quietly turn a test into a different scenario.
NOTES = _words(_NOTES_SYSTEM_PROMPT) + 32
PART = _words(_PART_SYSTEM_PROMPT) + 32
MERGE = _words(_MERGE_SYSTEM_PROMPT) + 32


class _WordLLM:
    """Counts a token per word, with a small fixed window -- and refuses
    anything over it, exactly as the NPU does."""

    def __init__(self, window: int = 150, reply_words: int = 8):
        self.window = window
        self.reply_words = reply_words
        self.calls: list[tuple[str, str, int]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def prompt_budget(self, max_tokens: int) -> int:
        return self.window

    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, control=None) -> str:
        needed = self.count_tokens(system_prompt) + self.count_tokens(user_prompt) + 32
        if needed > self.window:
            raise PromptTooLong(needed, self.window, "NPU")
        self.calls.append((system_prompt, user_prompt, max_tokens))
        return " ".join(["Maria:"] + ["task"] * (self.reply_words - 1))


def _session(llm, lines: int, words_per_line: int = 10) -> MeetingSession:
    session = MeetingSession(Engine.OPENVINO, compute_device="NPU", whisper_model_size="base")
    session._llm = llm
    text = " ".join(["budget"] * words_per_line)
    session._transcript = [TranscriptLine(f"10:{i // 60:02d}:{i % 60:02d}", text, "en") for i in range(lines)]
    return session


# --- the NPU is configured for real prompts -------------------------------------------


def test_the_npu_gets_a_real_prompt_window_and_room_for_the_answer():
    config = pipeline_config("NPU")
    assert config["MAX_PROMPT_LEN"] == NPU_MAX_PROMPT_LEN > 1024  # OpenVINO's default was the failure
    assert config["MIN_RESPONSE_LEN"] == NPU_MIN_RESPONSE_LEN >= 600  # meeting notes ask for up to 600
    assert "CACHE_DIR" in config


@pytest.mark.parametrize("device", ["GPU.0", "CPU", "AUTO"])
def test_other_devices_get_no_npu_window(device):
    assert "MAX_PROMPT_LEN" not in pipeline_config(device)


def test_too_long_is_said_plainly():
    message = str(PromptTooLong(1599, 1024, "NPU"))
    assert "1599" in message and "1024" in message and "NPU" in message and "Jenkins" not in message


# --- meeting notes --------------------------------------------------------------------


def test_a_meeting_that_fits_is_one_call():
    llm = _WordLLM(window=NOTES + 100)  # five 11-word lines fit
    notes = _session(llm, lines=5).generate_notes()
    assert len(llm.calls) == 1 and notes.parts == 1


def test_a_long_meeting_is_summarised_in_parts_then_merged():
    llm = _WordLLM(window=PART + 120)  # about nine lines a part
    progress = []
    notes = _session(llm, lines=60).generate_notes(on_progress=lambda step, steps: progress.append((step, steps)))
    assert notes.parts > 1
    assert notes.transcript_line_count == 60
    # Every call fitted the window (the fake raises otherwise), the parts
    # covered the transcript in order, and the last call was the merge.
    part_calls = [c for c in llm.calls if "ONE PART" in c[0]]
    assert len(part_calls) == notes.parts
    assert "combining notes" in llm.calls[-1][0]
    assert progress[-1][0] == progress[-1][1] and [s for s, _ in progress] == sorted(s for s, _ in progress)


def test_so_many_parts_they_are_merged_in_rounds():
    llm = _WordLLM(window=PART + 50)  # three lines a part: hundreds of parts
    notes = _session(llm, lines=600).generate_notes()
    merges = [c for c in llm.calls if "combining notes" in c[0]]
    assert notes.parts > 5 and len(merges) > 1


def test_a_window_too_small_to_merge_says_so_instead_of_looping():
    # Each part's notes are bigger than the merge window: merging can only
    # cut them up, never shrink them, so it must stop rather than loop.
    llm = _WordLLM(window=PART + 50, reply_words=400)
    with pytest.raises(RuntimeError, match="too long to condense"):
        _session(llm, lines=60).generate_notes()


def test_the_thin_transcript_gate_still_comes_first():
    with pytest.raises(RuntimeError, match="Not enough transcript"):
        _session(_WordLLM(), lines=1, words_per_line=5).generate_notes()


def test_split_keeps_order_and_fits():
    count = lambda text: len(text.split())
    items = [f"line{i} " + "w " * 9 for i in range(30)]
    pieces = split_to_fit(items, count, budget=40)
    assert all(count(piece) <= 40 for piece in pieces)
    assert " ".join(pieces).split()[0] == "line0" and "line29" in pieces[-1]


def test_one_huge_line_is_cut_by_words():
    count = lambda text: len(text.split())
    pieces = split_to_fit(["word " * 500], count, budget=100)
    assert len(pieces) > 4 and all(count(piece) <= 100 for piece in pieces)


# --- document Q&A ---------------------------------------------------------------------


def _hit(source: str, words: int):
    return SimpleNamespace(chunk=SimpleNamespace(source=source, text="passage " * words), score=1.0)


def test_qa_drops_the_weakest_excerpts_that_do_not_fit():
    retrieved = [_hit("best.md", 40), _hit("second.md", 40), _hit("third.md", 40), _hit("fourth.md", 40)]
    # 180 - 40 (instructions) - 32 (template) leaves ~108: two 40-word passages, not three.
    used = fit_excerpts(_WordLLM(window=180), retrieved, "What is the budget?", max_tokens=100)
    assert [r.chunk.source for r in used] == ["best.md", "second.md"]


def test_qa_keeps_every_excerpt_when_they_fit():
    retrieved = [_hit("a.md", 20), _hit("b.md", 20)]
    assert len(fit_excerpts(_WordLLM(window=1000), retrieved, "Question?", max_tokens=100)) == 2


def test_qa_always_keeps_the_best_excerpt():
    used = fit_excerpts(_WordLLM(window=50), [_hit("huge.md", 400), _hit("b.md", 5)], "Q?", max_tokens=100)
    assert [r.chunk.source for r in used] == ["huge.md"]


def test_a_stray_none_identified_after_real_items_is_dropped():
    from meeting_notes.session import without_contradicting_none

    # As the NPU model wrote it on a long meeting, 2026-09-25.
    notes = (
        "### Summary\n- Budget update\n\n**Action Items**\n\n"
        "- **Maria**:\n  - Send revised budget by Friday.\n- **Ahmed**:\n  - Fix login bug by Thursday.\n\n"
        "- **None Identified**"
    )
    cleaned = without_contradicting_none(notes)
    assert "None Identified" not in cleaned and "Fix login bug by Thursday" in cleaned


def test_a_genuinely_empty_action_list_keeps_its_none():
    notes = "Summary\n- We chatted about the weather.\n\nAction items\n- None identified."
    from meeting_notes.session import without_contradicting_none

    assert without_contradicting_none(notes) == notes
