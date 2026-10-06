"""The NPU can be reset by Windows under a running model. These tests pin
the two things the app does about it (pantherlake_ai_core.npu): bricks take
turns on the chip, and once the driver reports the loss nothing touches the
NPU again -- the speech and language models carry on on another chip."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from doc_qa.llm_openvino import OpenVINOLLM
from live_translation.transcriber_openvino import OpenVINOTranslator
from pantherlake_ai_core import npu

# What OpenVINO raised in the launcher on 2026-10-05 and 2026-10-06.
LOST = (
    "Exception from src\\inference\\src\\cpp\\infer_request.cpp:246: Exception from "
    "src\\plugins\\intel_npu\\src\\utils\\src\\zero\\zero_wrappers.cpp:382: L0 zeFenceHostSynchronize result: "
    "ZE_RESULT_ERROR_DEVICE_LOST, code 0x70000001 - device hung, reset, was removed, or driver update occurred"
)


@pytest.fixture(autouse=True)
def a_working_npu(monkeypatch):
    npu._reset_for_tests()
    monkeypatch.setattr(npu, "fallback_device", lambda: "GPU.0")
    yield
    npu._reset_for_tests()


def _until(condition) -> None:
    """Wait for another thread to get somewhere, and fail rather than hang if it never does."""
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "the other thread never got there"
        time.sleep(0.001)


def _in_thread(target) -> threading.Thread:
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


# --- turns ------------------------------------------------------------------------------


def test_two_bricks_never_hold_the_npu_together_and_go_in_the_order_they_asked():
    order: list[str] = []
    inside = threading.Event()
    release_first = threading.Event()
    queued = [threading.Event(), threading.Event()]

    def first():
        with npu.guard("NPU"):
            inside.set()
            release_first.wait(5)
            order.append("first")

    def later(name: str, mine: threading.Event):
        mine.set()
        with npu.guard("NPU"):
            order.append(name)

    one = _in_thread(first)
    assert inside.wait(5)
    two = _in_thread(lambda: later("second", queued[0]))
    assert queued[0].wait(5)
    _until(npu._turns.someone_waiting)  # "second" is in line before "third" asks
    three = _in_thread(lambda: later("third", queued[1]))
    assert queued[1].wait(5)
    _until(lambda: len(npu._turns._waiting) == 2)
    assert order == []  # nobody got in while the first call held the chip
    release_first.set()
    for thread in (one, two, three):
        thread.join(5)
    assert order == ["first", "second", "third"]


def test_other_chips_do_not_wait_for_the_npu():
    holding = threading.Event()
    done = threading.Event()

    def hold():
        with npu.guard("NPU"):
            holding.set()
            done.wait(5)

    thread = _in_thread(hold)
    assert holding.wait(5)
    with npu.guard("GPU.0"), npu.guard("CPU"), npu.guard(None):
        pass  # returns at once: only the NPU has turns
    done.set()
    thread.join(5)


def test_a_long_call_steps_aside_for_a_brick_that_is_waiting():
    order: list[str] = []
    started = threading.Event()
    other_waiting = threading.Event()

    def long_answer():
        with npu.guard("NPU"):
            order.append("token 1")
            started.set()
            other_waiting.wait(5)
            _until(npu._turns.someone_waiting)
            npu.breathe()  # between two tokens
            order.append("token 2")

    def utterance():
        started.wait(5)
        other_waiting.set()
        with npu.guard("NPU"):
            order.append("utterance")

    threads = [_in_thread(long_answer), _in_thread(utterance)]
    for thread in threads:
        thread.join(5)
    assert order == ["token 1", "utterance", "token 2"]


def test_breathe_is_harmless_with_nobody_waiting_or_outside_a_call():
    npu.breathe()
    with npu.guard("NPU"):
        npu.breathe()
    with npu.guard("NPU"):  # the turn was given back
        pass


def test_a_call_inside_a_call_on_the_same_thread_does_not_wait_for_itself():
    with npu.guard("NPU"):
        with npu.guard("NPU"):
            pass
    with npu.guard("NPU"):
        pass


# --- the loss ----------------------------------------------------------------------------


def test_the_drivers_report_closes_the_npu_for_the_rest_of_the_process():
    assert npu.lost() is None
    with pytest.raises(npu.NpuLost, match="restarted"):
        with npu.guard("NPU"):
            raise RuntimeError(LOST)
    assert "DEVICE_LOST" in npu.lost() and npu.lost_at() is not None

    touched = []
    with pytest.raises(npu.NpuLost):
        with npu.guard("NPU"):
            touched.append("the lost chip")  # a reload, a retry, the next receipt
    assert touched == []
    with npu.guard("GPU.0"):
        touched.append("another chip")
    assert touched == ["another chip"]


def test_any_other_error_is_passed_on_and_the_npu_stays_in_use():
    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        with npu.guard("NPU"):
            raise RuntimeError("L0 zeCommandQueueExecuteCommandLists result: ZE_RESULT_ERROR_INVALID_ARGUMENT")
    assert npu.lost() is None
    with npu.guard("NPU"):
        pass


def test_a_call_that_was_waiting_its_turn_is_turned_away_when_the_npu_goes():
    holding = threading.Event()
    outcome: list[str] = []

    def failing():
        try:
            with npu.guard("NPU"):
                holding.set()
                _until(npu._turns.someone_waiting)
                raise RuntimeError(LOST)
        except npu.NpuLost:
            outcome.append("first lost it")

    def waiting():
        holding.wait(5)
        try:
            with npu.guard("NPU"):
                outcome.append("second reached the chip")
        except npu.NpuLost:
            outcome.append("second turned away")

    threads = [_in_thread(failing), _in_thread(waiting)]
    for thread in threads:
        thread.join(5)
    assert sorted(outcome) == ["first lost it", "second turned away"]


# --- the two models that carry on elsewhere -----------------------------------------------


class _Pipeline:
    """Stands in for an openvino_genai pipeline on one chip. On the NPU it
    fails the way the driver does once `lost` is set."""

    made: list["_Pipeline"] = []

    def __init__(self, model_dir, device, **config):
        self.device = device
        self.config = config
        self.calls = 0
        _Pipeline.made.append(self)

    def get_tokenizer(self):
        return SimpleNamespace(encode=lambda text: SimpleNamespace(input_ids=SimpleNamespace(shape=(1, len(text.split())))))

    def generate(self, *args, **kwargs):
        self.calls += 1
        streamer = kwargs.get("streamer")
        if self.device == "NPU":
            if streamer is not None:
                streamer.write("Half")  # part of an answer was already on screen
            raise RuntimeError(LOST)
        if streamer is not None:
            streamer.write("Whole")
        perf = SimpleNamespace(
            get_num_generated_tokens=lambda: 1,
            get_generate_duration=lambda: SimpleNamespace(mean=100.0),
            get_throughput=lambda: SimpleNamespace(mean=10.0),
            get_ttft=lambda: SimpleNamespace(mean=50.0),
        )
        return SimpleNamespace(texts=[f"answered on {self.device}"], perf_metrics=perf)


class _TextStreamer:
    def __init__(self, tokenizer, callback):
        self._callback = callback

    def write(self, token):
        return self._callback(token)

    def end(self):
        pass


_GENAI = SimpleNamespace(
    ChatHistory=list,
    StreamerBase=object,
    TextStreamer=_TextStreamer,
    StreamingStatus=SimpleNamespace(RUNNING="running", CANCEL="cancel"),
    LLMPipeline=_Pipeline,
)


def _llm_on(device: str) -> OpenVINOLLM:
    _Pipeline.made.clear()
    llm = OpenVINOLLM.__new__(OpenVINOLLM)  # no model: the class's own logic is under test
    llm._ov_genai = _GENAI
    llm._model_dir = "model"
    llm._context = 32768
    llm.last_stats = None
    llm._load(device)
    return llm


def test_a_language_model_that_loses_the_npu_answers_on_another_chip(monkeypatch):
    monkeypatch.setattr("doc_qa.llm_openvino.pipeline_config", lambda device: {})
    llm = _llm_on("NPU")
    lost_pipeline = llm.pipeline
    assert llm.answer("system", "question") == "answered on GPU.0"
    assert llm.device == "GPU.0" and llm.last_stats.device == "GPU.0"
    assert lost_pipeline.calls == 1  # asked once, never again
    assert lost_pipeline in npu._retired  # and never released: freeing it is one more request to the driver
    assert llm.prompt_budget(512) == 32768 - 512  # no longer the NPU's fixed window

    assert llm.answer("system", "another") == "answered on GPU.0"
    assert [p.device for p in _Pipeline.made] == ["NPU", "GPU.0"]  # one move, then it stays


def test_a_language_model_asked_for_on_a_lost_npu_starts_elsewhere(monkeypatch):
    monkeypatch.setattr("doc_qa.llm_openvino.pipeline_config", lambda device: {})
    monkeypatch.setattr("doc_qa.llm_openvino.resolve_snapshot", lambda repo, **kwargs: "model")
    monkeypatch.setattr("doc_qa.llm_openvino._context_length", lambda model_dir: 32768)
    monkeypatch.setitem(__import__("sys").modules, "openvino_genai", _GENAI)
    with pytest.raises(npu.NpuLost):
        with npu.guard("NPU"):
            raise RuntimeError(LOST)
    _Pipeline.made.clear()
    llm = OpenVINOLLM(device="NPU")  # meeting notes writing its summary after the transcription lost the chip
    assert llm.device == "GPU.0" and [p.device for p in _Pipeline.made] == ["GPU.0"]


def test_on_the_npu_an_answer_nobody_is_watching_still_steps_aside_between_tokens(monkeypatch):
    monkeypatch.setattr("doc_qa.llm_openvino.pipeline_config", lambda device: {})
    stepped: list[bool] = []
    monkeypatch.setattr(npu, "breathe", lambda: stepped.append(True))
    llm = _llm_on("NPU")
    llm.answer("system", "question")
    assert stepped == [True]  # the one token the NPU pipeline wrote before it was lost


class _Heard:
    def __init__(self, text: str) -> None:
        self.text = text
        self.languages = ["fr"]

    def __str__(self) -> str:
        return self.text


class _Speech(_Pipeline):
    def generate(self, samples, task):
        self.calls += 1
        if self.device == "NPU":
            raise RuntimeError(LOST)
        return _Heard(f"heard on {self.device}")


def test_a_speech_model_that_loses_the_npu_redoes_the_utterance_on_another_chip(monkeypatch):
    import numpy as np

    monkeypatch.setattr("live_translation.transcriber_openvino.ov_config_for", lambda device: {})
    _Pipeline.made.clear()
    translator = OpenVINOTranslator.__new__(OpenVINOTranslator)
    translator._pipeline_cls = _Speech
    translator._model_dir = "model"
    translator.task = "translate"
    translator._load("NPU")
    lost_pipeline = translator.pipeline

    result = translator.translate(np.zeros(16000, dtype=np.float32))
    assert result.text == "heard on GPU.0" and result.detected_language == "fr"
    assert translator.device == "GPU.0"
    assert lost_pipeline.calls == 1 and lost_pipeline in npu._retired
