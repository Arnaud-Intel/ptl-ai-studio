"""Meeting notes are written by a model sized for the job, picked for the
chip, and told what a speech transcript is and is not."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from doc_qa.llm_openvino import OpenVINOLLM
from meeting_notes import session as notes
from meeting_notes.session import MeetingSession
from meeting_notes.types import TranscriptLine
from pantherlake_ai_core.engine import Engine


def test_the_notes_model_depends_on_the_chip():
    on_gpu = notes.notes_model_repo(Engine.OPENVINO, "GPU.0")
    assert on_gpu == notes.notes_model_repo(Engine.OPENVINO, "CPU") == notes.notes_model_repo(Engine.OPENVINO, "AUTO")
    # The NPU needs a build quantised for it: the other one answers garbage there.
    assert notes.notes_model_repo(Engine.OPENVINO, "NPU") not in (on_gpu, None)
    assert notes.notes_model_repo(Engine.PORTABLE, "cpu") is None  # the backend's own default


def test_nobody_choosing_the_notes_go_to_the_npu_when_there_is_one(monkeypatch):
    monkeypatch.setattr(notes.npu, "lost", lambda: None)
    assert notes.default_notes_device(Engine.OPENVINO, "GPU.0", ["CPU", "GPU.0", "NPU"]) == "NPU"
    assert notes.default_notes_device(Engine.OPENVINO, "GPU.0", ["CPU", "GPU.0"]) == "GPU.0"  # no NPU: where it transcribes
    assert notes.default_notes_device(Engine.PORTABLE, "cpu", ["CPU", "GPU.0", "NPU"]) == "cpu"

    monkeypatch.setattr(notes, "list_openvino_devices", lambda: ["CPU", "NPU"])  # asked without a list: it looks
    assert notes.default_notes_device(Engine.OPENVINO, "CPU") == "NPU"
    monkeypatch.setattr(notes.npu, "lost", lambda: "the device was removed")  # reset by Windows earlier this session
    assert notes.default_notes_device(Engine.OPENVINO, "CPU") == "CPU"


class _Notes:
    last_stats = None

    def __init__(self):
        self.read: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def prompt_budget(self, max_tokens: int) -> int:
        return 4000

    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, control=None) -> str:
        self.read.append(user_prompt)
        return "Summary\n- Shipping\n\nAction items\n- Unassigned: estimate the costs"


def _meeting(device: str) -> MeetingSession:
    lines = [
        TranscriptLine("10:41:56", "we will need to figure out the unit cost of each card and estimate the transport costs", "fr"),
        TranscriptLine("10:42:13", "and finally we will have to determine which clients will be priority", "fr"),
    ]
    return MeetingSession(Engine.OPENVINO, compute_device=device, whisper_model_size="base", transcript=lines)


@pytest.mark.parametrize("device", ["GPU.0", "NPU"])
def test_the_session_loads_that_model_and_gives_it_the_words_only(monkeypatch, device):
    asked: list[dict] = []
    llm = _Notes()

    def create_llm(engine, **kwargs):
        asked.append(kwargs)
        return llm

    monkeypatch.setattr(notes, "create_llm", create_llm)
    session = _meeting(device)
    written = session.generate_notes()

    assert asked[0]["model_repo"] == notes.notes_model_repo(Engine.OPENVINO, device) and asked[0]["device"] == device
    # No "[10:41:56]" in front of each line: the models made dates out of them.
    assert llm.read == [
        "we will need to figure out the unit cost of each card and estimate the transport costs\n"
        "and finally we will have to determine which clients will be priority"
    ]
    assert written.transcript_line_count == 2
    assert session.transcript_text().startswith("[10:41:56] we will need")  # a person reading it still gets the times


def test_the_notes_can_be_written_on_another_chip_than_the_one_that_transcribes(monkeypatch):
    asked: list[dict] = []
    monkeypatch.setattr(notes, "create_llm", lambda engine, **kwargs: asked.append(kwargs) or _Notes())
    lines = [TranscriptLine("10:00:00", " ".join(["we will need to estimate the transport costs"] * 5), "en")]
    session = MeetingSession(
        Engine.OPENVINO, compute_device="GPU.0", whisper_model_size="base", transcript=lines, notes_device="NPU"
    )
    session.generate_notes()
    session.write_notes_on("NPU")  # the same chip: nothing to load again
    session.generate_notes()
    assert [(call["device"], call["model_repo"]) for call in asked] == [("NPU", notes.notes_model_repo(Engine.OPENVINO, "NPU"))]

    session.write_notes_on("GPU.0")  # another chip is another build of the model
    session.generate_notes()
    assert (asked[-1]["device"], asked[-1]["model_repo"]) == ("GPU.0", notes.notes_model_repo(Engine.OPENVINO, "GPU.0"))
    assert len(asked) == 2 and session.compute_device == "GPU.0"


def test_too_few_words_is_counted_in_words_not_in_timestamps():
    # Twelve short lines used to pass as 25 "words" on their timestamps alone.
    session = MeetingSession(
        Engine.OPENVINO, compute_device="GPU.0", whisper_model_size="base",
        transcript=[TranscriptLine(f"10:00:{second:02d}", "yes", "en") for second in range(20)],
    )
    with pytest.raises(RuntimeError, match="20 words so far"):
        session.generate_notes()


def test_headings_written_as_list_items_or_markdown_come_out_plain():
    written = "- Summary\n- We talked about costs.\n\n**Action items:**\n- Unassigned: estimate the costs\n- Summary of costs to Dana"
    assert notes.with_plain_headings(written) == (
        "Summary\n- We talked about costs.\n\nAction items\n- Unassigned: estimate the costs\n- Summary of costs to Dana"
    )
    assert notes.with_plain_headings("### 2. Action Items\n- None identified") == "Action items\n- None identified"


def test_the_instructions_say_what_a_transcript_cannot_tell():
    for prompt in (notes._NOTES_SYSTEM_PROMPT, notes._PART_SYSTEM_PROMPT, notes._MERGE_SYSTEM_PROMPT):
        assert "Unassigned" in prompt  # nobody is named in a transcript; the notes must be able to say so
    assert "no speaker names" in notes._NOTES_SYSTEM_PROMPT and "No Markdown" in notes._NOTES_SYSTEM_PROMPT
    # A name in an example came back as the owner of every task (2026-10-07).
    assert not any(name in notes._NOTES_SYSTEM_PROMPT for name in ("Maria", "John", "Alice", "Bob"))


# --- a model that reasons aloud is told not to ------------------------------------------


class _History(list):
    def set_extra_context(self, context):
        self.context = context


def _llm(chat_template: str | None) -> tuple[OpenVINOLLM, list]:
    seen: list = []

    class _Pipeline:
        def __init__(self, model_dir, device, **config):
            pass

        def get_tokenizer(self):
            tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(input_ids=SimpleNamespace(shape=(1, len(text.split())))))
            if chat_template is not None:
                tokenizer.chat_template = chat_template
            return tokenizer

        def generate(self, history, **options):
            seen.append(history)
            return SimpleNamespace(texts=["the answer"], perf_metrics=None)

    llm = OpenVINOLLM.__new__(OpenVINOLLM)  # no model: what it sends is under test
    llm._ov_genai = SimpleNamespace(ChatHistory=_History, LLMPipeline=_Pipeline)
    llm._model_dir, llm._context, llm.last_stats = "model", 32768, None
    llm._load("GPU.0")
    return llm, seen


def test_a_model_with_a_thinking_switch_has_it_turned_off(monkeypatch):
    monkeypatch.setattr("doc_qa.llm_openvino.pipeline_config", lambda device: {})
    llm, seen = _llm("{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>{%- endif %}")
    assert llm.answer("system", "question") == "the answer"
    assert seen[0].context == {"enable_thinking": False}


@pytest.mark.parametrize("chat_template", ["{{ messages }}", "", None])
def test_a_model_without_one_is_sent_what_it_always_was(monkeypatch, chat_template):
    # Today's models, and the fixed pages HTML Creator depends on, must not change.
    monkeypatch.setattr("doc_qa.llm_openvino.pipeline_config", lambda device: {})
    llm, seen = _llm(chat_template)
    assert llm.answer("system", "question") == "the answer"
    assert not hasattr(seen[0], "context")
