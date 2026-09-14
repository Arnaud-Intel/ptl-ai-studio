"""What a result cost in energy: the processor package's energy over the
window it took, and how much of it was above what the machine burns
idling (BACKLOG R18).

One meter for the whole launcher. The telemetry poller samples it once a
second for the watts gauge and, whenever no demo is running, teaches it
the idle baseline; a runner marks where a result's work starts and asks
what it cost when the result is ready. Package energy covers the CPU, the
iGPU and the NPU alike -- the NPU has no rail of its own -- so when other
demos run in the same window the cost is shared, and the answer says so
rather than pretending it belongs to one demo.
"""
from __future__ import annotations

import statistics
import threading
from collections import deque

from pantherlake_ai_core import power

from . import activity

meter = power.EnergyMeter()

# Package watts, one sample per second while nothing runs. The median, not
# the mean: a browser repaint or a background scan shouldn't move the
# baseline the way a sustained load would. Two minutes of history, so it
# follows the machine settling (screen brightness, fans) without jumping.
_idle: deque[float] = deque(maxlen=120)
_idle_lock = threading.Lock()
_MIN_IDLE_SAMPLES = 5


def note_idle(package_watts: float) -> None:
    with _idle_lock:
        _idle.append(package_watts)


def idle_watts() -> float | None:
    """The idle baseline, or None until a few quiet seconds have been seen."""
    with _idle_lock:
        if len(_idle) < _MIN_IDLE_SAMPLES:
            return None
        return statistics.median(_idle)


def mark() -> power.EnergySample | None:
    """Where a result's window starts; None on a machine without counters."""
    return meter.read()


def since(start: power.EnergySample | None, demo_id: str) -> dict | None:
    """What `demo_id` has spent since `start` -- see between()."""
    if start is None:
        return None
    end = meter.read()
    return between(start, end, demo_id) if end is not None else None


def between(start: power.EnergySample, end: power.EnergySample, demo_id: str) -> dict:
    """Package joules over the window, its length, the part above the idle
    baseline (None until one is known -- a total is still honest, a guessed
    baseline isn't), and which other demos were running at the end of it."""
    seconds = end.at - start.at
    joules = end.joules["package"] - start.joules["package"]
    idle = idle_watts()
    others = sorted({a["demo_id"] for a in activity.snapshot() if a["demo_id"] != demo_id})
    return {
        "joules": round(joules, 2),
        "seconds": round(seconds, 3),
        "above_idle_joules": round(max(joules - idle * seconds, 0.0), 2) if idle is not None else None,
        "shared_with": others,
    }
