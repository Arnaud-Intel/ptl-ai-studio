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


@pytest.fixture(autouse=True)
def _a_laptop_with_memory_to_spare(monkeypatch):
    """Every test plays a 64 GB laptop, whatever runs it: the 30B coding
    model is greyed out under 32 GB (pantherlake_ai_core.machine), and
    GitHub's test machine has 16 -- on 2026-10-22 that turned ten tests
    about compiling, downloading and building a page into tests of a small
    laptop. A test about a smaller one says so itself (test_memory_fit.py)."""
    from pantherlake_ai_core import machine

    monkeypatch.setenv(machine.MEMORY_SWITCH, "64")
    # And 75% of it for the integrated GPU, which a 64 GB laptop left at
    # Intel's default (57%, 36 GB) would also have enough of: what a test
    # sees does not depend on the slider of the laptop that runs it.
    monkeypatch.setenv(machine.GPU_SHARE_SWITCH, "48")
