"""What every test gets, asked for or not."""
from __future__ import annotations

import pytest

from launcher import events


@pytest.fixture(autouse=True)
def _activity_log_of_its_own(monkeypatch, tmp_path):
    """No test writes into `logs/events.log`. That file is what a launcher
    running beside the tests is writing, what a review of the app reads and
    what `/api/logs` serves: on 2026-10-13 a run of this suite put twenty-nine
    events into it that never happened -- a voice assistant loading on the
    NPU, a detector on a GPU the machine did not have that day. A test that
    wants the file somewhere in particular still says so itself."""
    monkeypatch.setattr(events, "LOG_FILE", tmp_path / "events.log")
