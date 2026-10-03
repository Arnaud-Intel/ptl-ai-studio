"""Streaming the language models (BACKLOG R32): text as it is written, a
live rate, and stopping an answer part-way -- with no model loaded."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from code_review_assist import session as code_review_session
from code_review_assist.session import CodeReviewSession
from doc_qa.llm_openvino import OpenVINOLLM
from doc_qa.llm_portable import PortableLLM
from fastapi.testclient import TestClient
from meeting_notes.session import MeetingSession, _PART_SYSTEM_PROMPT
from meeting_notes.types import TranscriptLine
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import (
    GenerationControl,
    GenerationStats,
    combine_stats,
    openvino_streamer,
    say,
    stopped,
)

from launcher import activity, generation, metrics
from launcher import app as launcher_app


@pytest.fixture(autouse=True)
def clean_state():
    yield
    with generation._lock:
        generation._live.clear()
    with metrics._lock:
        metrics._metrics.clear()
    with activity._lock:
        activity._active.clear()


def _collector(stop_after: int | None = None):
    """A control that keeps what it is given, and asks to stop after N pieces."""
    seen: list[str] = []
    control = GenerationControl(
        on_text=seen.append,
        should_stop=lambda: stop_after is not None and len(seen) >= stop_after,
    )
    return control, seen


# --- the control object ----------------------------------------------------------------


def test_no_control_means_nothing_to_do():
    say(None, "heading")  # must not raise
    assert stopped(None) is False
    assert stopped(GenerationControl()) is False


class _TextStreamer:
    """openvino_genai.TextStreamer's contract: it is given tokens, and calls
    back only when one completes some text. Here a token is its own text,
    and an empty one completes nothing."""

    def __init__(self, tokenizer, callback):
        self._callback = callback

    def write(self, token):
        return self._callback(token) if token else "running"

    def end(self):
        pass


_GENAI = SimpleNamespace(
    ChatHistory=list,
    StreamerBase=object,
    TextStreamer=_TextStreamer,
    StreamingStatus=SimpleNamespace(RUNNING="running", CANCEL="cancel"),
)


def test_the_openvino_streamer_passes_text_on_and_cancels_when_asked():
    control, seen = _collector(stop_after=2)
    streamer, was_cancelled = openvino_streamer(control, _GENAI, tokenizer=None)
    assert streamer.write("Hello") == "running" and was_cancelled() is False
    assert streamer.write(" world") == "cancel" and was_cancelled() is True
    assert seen == ["Hello", " world"]


def test_the_openvino_streamer_counts_tokens_not_pieces_of_text():
    counted: list[int] = []
    text: list[str] = []
    control = GenerationControl(on_text=text.append, on_tokens=counted.append)
    streamer, _ = openvino_streamer(control, _GENAI, tokenizer=None)
    for token in ["<div", "", "", ">", ["a", "b"]]:  # two complete no text; the last is two at once
        streamer.write(token)
    assert sum(counted) == 6 and len(text) == 3


def test_combined_stats_remember_that_a_part_was_stopped():
    done = GenerationStats("GPU", tokens=10, seconds=1.0, tokens_per_second=10.0)
    cut = GenerationStats("GPU", tokens=5, seconds=0.5, tokens_per_second=10.0, cancelled=True)
    assert combine_stats([done, cut]).cancelled is True
    assert combine_stats([done, done]).cancelled is False


# --- the two backends, with stand-in runtimes -------------------------------------------


class _PerfMetrics:
    def get_num_generated_tokens(self):
        return 3

    def get_generate_duration(self):
        return SimpleNamespace(mean=300.0)

    def get_throughput(self):
        return SimpleNamespace(mean=10.0)

    def get_ttft(self):
        return SimpleNamespace(mean=90.0)


class _FakePipeline:
    """openvino_genai.LLMPipeline's streaming contract: call the streamer per
    piece, stop when it answers CANCEL, return what was written."""

    def __init__(self):
        self.streamer_used = None

    def generate(self, history, max_new_tokens, temperature, streamer=None):
        self.streamer_used = streamer
        written = []
        for piece in ["The", " budget", " is", " approved", "."]:
            written.append(piece)
            if streamer is not None and streamer.write(piece) == "cancel":
                break
        return SimpleNamespace(texts=["".join(written)], perf_metrics=_PerfMetrics())


def _openvino_llm():
    llm = OpenVINOLLM.__new__(OpenVINOLLM)  # no model: only answer()'s own logic is under test
    llm._ov_genai = _GENAI
    llm.pipeline = _FakePipeline()
    llm.device = "GPU"
    llm._on_npu = False
    llm._context = 32768
    llm._tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(input_ids=SimpleNamespace(shape=(1, len(text.split())))))
    llm.last_stats = None
    return llm


def test_openvino_without_a_control_installs_no_streamer():
    llm = _openvino_llm()
    assert llm.answer("system", "question") == "The budget is approved."
    assert llm.pipeline.streamer_used is None  # a call nobody is watching runs as it always did
    assert llm.last_stats.cancelled is False


def test_openvino_streams_and_stops_part_way():
    llm = _openvino_llm()
    control, seen = _collector(stop_after=2)
    assert llm.answer("system", "question", control=control) == "The budget"
    assert seen == ["The", " budget"]
    assert llm.last_stats.cancelled is True


def _portable_llm(pieces):
    llm = PortableLLM.__new__(PortableLLM)
    calls = []

    def create_chat_completion(messages, max_tokens, temperature, stream=False):
        calls.append(stream)
        if stream:
            return iter({"choices": [{"delta": {"content": piece}}]} for piece in pieces)
        return {"choices": [{"message": {"content": "".join(pieces)}}], "usage": {"completion_tokens": len(pieces)}}

    llm.model = SimpleNamespace(create_chat_completion=create_chat_completion, tokenize=lambda data, add_bos, special: [0] * 3)
    llm.n_ctx = 4096
    llm.last_stats = None
    return llm, calls


def test_llama_streams_only_when_asked_and_stops_part_way():
    llm, calls = _portable_llm(["Yes", ",", " by", " Friday", "."])
    assert llm.answer("system", "question") == "Yes, by Friday."
    control, seen = _collector(stop_after=3)
    counted: list[int] = []
    control.on_tokens = counted.append
    assert llm.answer("system", "question", control=control) == "Yes, by"
    assert calls == [False, True]
    assert seen == ["Yes", ",", " by"] and llm.last_stats.cancelled is True and llm.last_stats.tokens == 3
    assert sum(counted) == 3


def test_openvino_draws_by_default_and_takes_its_first_choice_when_asked():
    llm = _openvino_llm()
    asked = []

    class Recording:
        def generate(self, history, max_new_tokens, **choice):
            asked.append(choice)
            return SimpleNamespace(texts=["ok"], perf_metrics=_PerfMetrics())

    llm.pipeline = Recording()
    llm.answer("system", "question")
    llm.answer("system", "question", sample=False)
    assert asked == [{"temperature": 0.2}, {"do_sample": False}]


def test_llama_takes_its_first_choice_at_temperature_zero():
    llm, _ = _portable_llm(["Yes", "."])
    temperatures = []
    complete = llm.model.create_chat_completion

    def recording(messages, max_tokens, temperature, stream=False):
        temperatures.append(temperature)
        return complete(messages, max_tokens, temperature, stream)

    llm.model.create_chat_completion = recording
    llm.answer("system", "question")
    llm.answer("system", "question", sample=False)
    llm.answer("system", "question", control=GenerationControl(), sample=False)  # streamed
    assert temperatures == [0.2, 0.0, 0.0]


# --- bricks ------------------------------------------------------------------------------


class _StreamingLLM:
    """Writes five words per answer through the control, honouring a stop."""

    def __init__(self):
        self.calls: list[str] = []
        self.last_stats = None

    def count_tokens(self, text):
        return len(text.split())

    def prompt_budget(self, max_tokens):
        return 100_000

    def answer(self, system_prompt, user_prompt, max_tokens=512, control=None):
        self.calls.append(system_prompt)
        written = []
        for word in ["one", "two", "three", "four", "five"]:
            written.append(word)
            say(control, word + " ")
            if stopped(control):
                break
        self.last_stats = GenerationStats("GPU", tokens=len(written), seconds=0.5, tokens_per_second=10.0, cancelled=stopped(control))
        return " ".join(written)


def test_code_review_streams_both_answers_under_their_headings(monkeypatch):
    llm = _StreamingLLM()
    monkeypatch.setattr(code_review_session, "create_llm", lambda *a, **k: llm)
    control, seen = _collector()
    result = CodeReviewSession(Engine.OPENVINO, compute_device="GPU").review(diff_text="diff --git a/x b/x\n+1\n", control=control)
    text = "".join(seen)
    assert text.index("Commit message") < text.index("Review notes")
    assert result.cancelled is False and len(llm.calls) == 2


def test_stopping_during_the_first_answer_never_starts_the_second(monkeypatch):
    llm = _StreamingLLM()
    monkeypatch.setattr(code_review_session, "create_llm", lambda *a, **k: llm)
    control, _ = _collector(stop_after=3)  # the heading, then two words
    result = CodeReviewSession(Engine.OPENVINO, compute_device="GPU").review(diff_text="diff --git a/x b/x\n+1\n", control=control)
    assert len(llm.calls) == 1
    assert result.cancelled is True and result.commit_message == "one two" and result.review_notes == ""


def _meeting(llm, lines=40):
    session = MeetingSession(Engine.OPENVINO, compute_device="NPU", whisper_model_size="base")
    session._llm = llm
    session._transcript = [TranscriptLine(f"10:00:{i:02d}", "we agreed the budget is due on friday", "en") for i in range(lines)]
    return session


def test_a_long_meeting_stopped_part_way_keeps_the_parts_done(monkeypatch):
    llm = _StreamingLLM()
    monkeypatch.setattr(llm, "prompt_budget", lambda max_tokens: len(_PART_SYSTEM_PROMPT.split()) + 32 + 120)
    pieces: list[str] = []
    # Part 1 is a heading and five words; the stop lands two words into part 2.
    control = GenerationControl(on_text=pieces.append, should_stop=lambda: len(pieces) >= 9)
    notes = _meeting(llm).generate_notes(control=control)
    assert notes.cancelled is True and notes.parts > 2
    assert notes.text.startswith("Part 1 of") and "Part 2 of" in notes.text  # what was done is kept
    assert "-- Part 1 of" in "".join(pieces)  # and the page saw it being written
    assert len(llm.calls) == 2  # nothing was started after the stop


def test_a_short_meeting_streams_its_one_answer():
    llm = _StreamingLLM()
    control, seen = _collector()
    notes = _meeting(llm, lines=5).generate_notes(control=control)
    assert notes.cancelled is False and "".join(seen).strip() == "one two three four five"


# --- the launcher ----------------------------------------------------------------------


def test_an_answer_in_flight_holds_its_text_and_reports_a_live_rate():
    import time

    live = generation.get("code-review-assist")
    control = live.begin()
    words = [f"word{i} " for i in range(generation._RATE_MIN_TOKENS)]
    for word in words[:-1]:
        control.on_tokens(1)
        control.on_text(word)
        time.sleep(0.02)  # tokens arrive over time; Windows' clock ticks every ~15 ms
    assert metrics.snapshot() == []  # a couple of tokens are not a rate yet
    control.on_tokens(1)
    control.on_text(words[-1])
    assert generation.snapshot("code-review-assist") == {"active": True, "text": "".join(words), "cancelled": False}
    assert [(m["unit"], m["sticky"]) for m in metrics.snapshot()] == [("tok/s", False)]
    assert control.should_stop() is False
    assert generation.cancel("code-review-assist") is True and control.should_stop() is True
    live.end()
    assert generation.snapshot("code-review-assist")["cancelled"] is True
    assert generation.cancel("code-review-assist") is False  # nothing in flight any more


def test_a_heading_joins_the_text_but_not_the_rate():
    import time

    live = generation.get("code-review-assist")
    control = live.begin()
    for _ in range(generation._RATE_MIN_TOKENS - 1):
        control.on_tokens(1)
        control.on_text("x")
        time.sleep(0.02)
    say(control, "\n\nReview notes\n\n")  # the second answer starts: its rate starts with it
    control.on_tokens(1)
    control.on_text("y")
    assert generation.snapshot("code-review-assist")["text"].endswith("Review notes\n\ny")
    assert metrics.snapshot() == []


def test_a_new_answer_starts_from_nothing():
    live = generation.get("doc-qa")
    live.begin().on_text("old answer")
    live.cancel()
    live.end()
    control = live.begin()
    assert generation.snapshot("doc-qa") == {"active": True, "text": "", "cancelled": False}
    assert control.should_stop() is False


def test_one_stage_can_be_stopped_without_the_others():
    notes = generation.get("meeting-notes", "notes")
    notes.begin()
    assert generation.cancel("meeting-notes", "ocr") is False
    assert generation.cancel("meeting-notes", "notes") is True


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher_app.events, "LOG_FILE", tmp_path / "events.log")
    return TestClient(launcher_app.app)


class _Runner:
    def __init__(self):
        import threading

        self._session, self._engine, self._device, self._lock = object(), "openvino", "GPU", threading.Lock()


def test_the_close_button_stops_an_answer_first_and_unloads_second(client, monkeypatch):
    runner = _Runner()
    monkeypatch.setitem(launcher_app._UNLOADABLE, "code-review-assist", runner)
    live = generation.get("code-review-assist")
    live.begin().on_text("Commit message")
    activity.set_active("code-review-assist", engine="openvino", device="GPU")

    telemetry = client.get("/api/telemetry").json()
    assert telemetry["active"][0]["can_stop"] is True  # it used to be disabled mid-answer
    assert client.get("/api/bricks/code-review-assist/partial").json()["text"] == "Commit message"
    assert client.post("/api/bricks/code-review-assist/stop").json() == {"status": "cancelling"}
    assert runner._session is not None  # the model is still loaded

    live.end()
    assert client.post("/api/bricks/code-review-assist/stop").json() == {"status": "unloaded"}


def test_meeting_notes_summary_stops_without_ending_the_transcription(client, monkeypatch):
    stopped_runs = []
    monkeypatch.setitem(launcher_app._STOPPABLE, "meeting-notes", type("R", (), {"stop": lambda self: stopped_runs.append(1)})())
    generation.get("meeting-notes", "notes").begin()
    assert client.post("/api/bricks/meeting-notes/stop?stage=notes").json() == {"status": "cancelling"}
    assert stopped_runs == []
    assert client.post("/api/bricks/meeting-notes/stop").json() == {"status": "stopped"}
    assert stopped_runs == [1]


def test_nothing_being_written_is_an_empty_answer(client):
    assert client.get("/api/bricks/doc-qa/partial").json() == {"active": False, "text": "", "cancelled": False}
