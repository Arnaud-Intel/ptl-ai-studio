"""The suite beside a launcher at work: nothing here ends up in its log.

This file asks for no fixture on purpose. What keeps its event out of
`logs/events.log` is `conftest.py`, as for every test that does not think of
it."""
from __future__ import annotations

import uuid
from pathlib import Path

import launcher
from launcher import events


def test_an_event_set_by_a_test_is_not_in_the_live_activity_log():
    live = Path(launcher.__file__).resolve().parents[3] / "logs" / "events.log"
    marker = f"written by a test {uuid.uuid4().hex}"
    events.set_phase("a-test", "running", marker)
    assert events.LOG_FILE != live
    assert marker in events.LOG_FILE.read_text(encoding="utf-8")  # it was written, and somewhere else
    assert not live.exists() or marker not in live.read_text(encoding="utf-8", errors="replace")
    events.clear_phase("a-test")
