"""The energy meter's arithmetic and its honesty about missing hardware
(BACKLOG R18) -- nothing here reads a real counter."""
from __future__ import annotations

from types import SimpleNamespace

import psutil
from pantherlake_ai_core import power
from pantherlake_ai_core.power import EnergySample, watts_between


def test_watts_are_each_rails_energy_over_the_elapsed_time():
    a = EnergySample(10.0, {"package": 100.0, "cores": 60.0})
    b = EnergySample(12.0, {"package": 136.0, "cores": 80.0, "graphics": 5.0})
    # A rail missing from the first sample has no delta to report.
    assert watts_between(a, b) == {"package": 18.0, "cores": 10.0}
    assert watts_between(b, b) == {}


def test_no_counters_means_no_readings_not_zeros(monkeypatch):
    monkeypatch.setattr(power, "_IS_WINDOWS", False)
    meter = power.EnergyMeter()
    assert meter.available is False and meter.read() is None


def test_battery_on_battery(monkeypatch):
    monkeypatch.setattr(psutil, "sensors_battery", lambda: SimpleNamespace(percent=87.4, secsleft=11400, power_plugged=False))
    assert power.battery() == {"percent": 87, "plugged": False, "seconds_left": 11400}


def test_battery_plugged_in_has_no_time_left(monkeypatch):
    monkeypatch.setattr(
        psutil, "sensors_battery",
        lambda: SimpleNamespace(percent=100, secsleft=psutil.POWER_TIME_UNLIMITED, power_plugged=True),
    )
    assert power.battery() == {"percent": 100, "plugged": True, "seconds_left": None}


def test_no_battery(monkeypatch):
    monkeypatch.setattr(psutil, "sensors_battery", lambda: None)
    assert power.battery() is None
