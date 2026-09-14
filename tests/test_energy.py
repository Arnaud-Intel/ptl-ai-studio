"""Energy per result: the arithmetic against the idle baseline, and what the
launcher says when it can't know (BACKLOG R18). No real counter is read."""
from __future__ import annotations

from collections import deque

import pytest
from pantherlake_ai_core.power import EnergySample

from launcher import energy


@pytest.fixture(autouse=True)
def quiet_machine(monkeypatch):
    monkeypatch.setattr(energy, "_idle", deque(maxlen=120))
    monkeypatch.setattr(energy.activity, "snapshot", lambda: [])


START = EnergySample(0.0, {"package": 1000.0})
END = EnergySample(2.0, {"package": 1040.0})


def test_energy_above_idle_subtracts_the_baseline_over_the_window():
    for watts in (12.0, 11.0, 13.0, 12.0, 30.0):
        energy.note_idle(watts)
    assert energy.idle_watts() == 12.0  # the median: one spike doesn't move it
    assert energy.between(START, END, "live-translation") == {
        "joules": 40.0, "seconds": 2.0, "above_idle_joules": 16.0, "shared_with": [],
    }


def test_before_a_baseline_is_learned_only_the_total_is_reported():
    energy.note_idle(12.0)
    assert energy.idle_watts() is None
    assert energy.between(START, END, "live-translation")["above_idle_joules"] is None


def test_other_demos_running_at_the_same_time_are_named(monkeypatch):
    monkeypatch.setattr(
        energy.activity,
        "snapshot",
        lambda: [{"demo_id": "live-translation"}, {"demo_id": "smart-city-monitor"}, {"demo_id": "smart-city-monitor"}],
    )
    assert energy.between(START, END, "live-translation")["shared_with"] == ["smart-city-monitor"]


def test_above_idle_never_goes_negative():
    for _ in range(5):
        energy.note_idle(25.0)
    assert energy.between(START, END, "screen-ocr")["above_idle_joules"] == 0.0


def test_without_counters_there_is_no_energy_line():
    assert energy.since(None, "screen-ocr") is None
