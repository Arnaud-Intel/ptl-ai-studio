"""The shared start/stop bookkeeping behind every streaming brick.

A stop event is cooperative: the worker only sees it between steps, so a
thread inside a model load or one long inference keeps running for a while
after Stop. These tests pin down what the launcher reports during that
stretch -- the thing that used to be wrong (it went on claiming the brick
was doing its normal work).
"""
from __future__ import annotations

import threading

import pytest

from launcher import events, worker
from launcher.errors import Conflict

DEMO = "test-brick"


@pytest.fixture(autouse=True)
def isolated_events(monkeypatch, tmp_path):
    """Keep phases (and the persisted log) out of the real launcher state."""
    monkeypatch.setattr(events, "_status", {})
    monkeypatch.setattr(events, "LOG_FILE", tmp_path / "events.log")


@pytest.fixture
def busy_thread():
    """A worker that ignores the stop event until the test lets it go --
    i.e. one stuck inside a call it can't be interrupted from."""
    release = threading.Event()
    thread = threading.Thread(target=lambda: release.wait(timeout=10), daemon=True)
    thread.start()
    yield thread
    release.set()
    thread.join(timeout=5)


def phases() -> dict[str, str]:
    return {key: entry["phase"] for key, entry in events.status_snapshot().items()}


# --- request_stop -------------------------------------------------------------


def test_a_worker_that_stops_promptly_reports_success_and_sets_no_phase():
    stop_event = threading.Event()
    thread = threading.Thread(target=lambda: stop_event.wait(timeout=5), daemon=True)
    thread.start()
    assert worker.request_stop(DEMO, thread, stop_event) is True
    assert phases() == {}
    assert stop_event.is_set()


def test_nothing_to_stop_is_success():
    assert worker.request_stop(DEMO, None, None) is True
    assert phases() == {}


def test_a_worker_still_in_a_call_says_so_instead_of_its_normal_message(busy_thread):
    stop_event = threading.Event()
    assert worker.request_stop(DEMO, busy_thread, stop_event, grace=0.05) is False
    assert phases() == {DEMO: "stopping"}
    assert events.status_snapshot()[DEMO]["message"] == worker.STOPPING_MESSAGE
    # The stop was still requested -- it just hasn't landed yet.
    assert stop_event.is_set()


def test_a_multi_stage_brick_marks_every_stage(busy_thread):
    assert worker.request_stop(DEMO, busy_thread, threading.Event(), stages=("ocr", "llm"), grace=0.05) is False
    assert phases() == {f"{DEMO}:ocr": "stopping", f"{DEMO}:llm": "stopping"}


# --- refuse_if_busy / is_stopping ------------------------------------------------


def test_an_idle_brick_can_start():
    worker.refuse_if_busy(DEMO, None, None)  # does not raise


def test_a_finished_thread_does_not_block_a_new_run():
    thread = threading.Thread(target=lambda: None, daemon=True)
    thread.start()
    thread.join(timeout=5)
    worker.refuse_if_busy(DEMO, thread, threading.Event())  # does not raise


def test_a_running_brick_refuses_a_second_start(busy_thread):
    with pytest.raises(Conflict, match="already running"):
        worker.refuse_if_busy(DEMO, busy_thread, threading.Event())


def test_a_brick_that_is_still_stopping_says_which_it_is(busy_thread):
    # The distinction matters: "already running" clears only when the user
    # presses Stop, "still stopping" clears on its own.
    stop_event = threading.Event()
    stop_event.set()
    with pytest.raises(Conflict, match="still stopping"):
        worker.refuse_if_busy(DEMO, busy_thread, stop_event)


def test_is_stopping(busy_thread):
    assert worker.is_stopping(None, None) is False
    assert worker.is_stopping(busy_thread, None) is False
    assert worker.is_stopping(busy_thread, threading.Event()) is False
    set_event = threading.Event()
    set_event.set()
    assert worker.is_stopping(busy_thread, set_event) is True
