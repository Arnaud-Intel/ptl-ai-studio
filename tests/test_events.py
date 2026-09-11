from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json

import pytest

from launcher import events


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(events, "LOG_FILE", tmp_path / "events.log")
    monkeypatch.setattr(events, "_history_file", None)
    monkeypatch.setattr(events, "_recent", deque(maxlen=200))
    monkeypatch.setattr(events, "_status", {})


def restart():
    events._history_file = None
    events._recent.clear()
    events._status.clear()


def test_history_survives_restart_without_resurrecting_workers():
    events.set_phase("demo", "error", "model unavailable")
    restart()
    assert events.recent_events()[0]["message"] == "model unavailable"
    assert events.status_snapshot() == {}
    assert events.recent_events(0) == events.recent_events(-1) == []


def test_partial_and_malformed_lines_do_not_hide_new_events():
    events.set_phase("demo", "error", "first")
    with events.LOG_FILE.open("ab") as stream:
        stream.write(b'[]\n{"broken":')
    restart()
    events.set_phase("demo", "error", "second")
    restart()
    assert [e["message"] for e in events.recent_events()] == ["first", "second"]


def test_rotation_is_bounded_and_restores_recent_order(monkeypatch):
    monkeypatch.setattr(events, "MAX_LOG_BYTES", 400)
    for i in range(40):
        events.set_phase("demo", "error", str(i))
    assert len(list(events.LOG_FILE.parent.glob("events.log*"))) <= 4
    assert all(p.stat().st_size <= 400 for p in events.LOG_FILE.parent.glob("events.log*"))
    restart()
    messages = [int(e["message"]) for e in events.recent_events()]
    assert messages == sorted(messages) and messages[-1] == 39


def test_concurrent_events_remain_complete_and_ordered():
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: events.set_phase("demo", "error", str(i)), range(100)))
    disk = [json.loads(line) for line in events.LOG_FILE.read_text().splitlines()]
    assert len(disk) == 100
    assert disk == events.recent_events(200)


def test_logging_failure_does_not_break_worker(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setattr(events, "LOG_FILE", blocker / "events.log")
    events.set_phase("demo", "error", "still visible")
    assert events.recent_events()[0]["message"] == "still visible"
