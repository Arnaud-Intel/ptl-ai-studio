"""What each brick is achieving right now, in its own unit -- tokens per
second for a language model, frames per second for video, times real time
for speech -- so the side panel can put a number next to every running brick.

One unit for everything would be simpler and wrong: four of the bricks never
produce a token. The runners report from what they already see (a frame
arriving, an utterance translated, an answer's stats); the bricks don't know
this module exists.

Two lifetimes:

- live: belongs to a running stage, and goes when that stage's activity
  entry is cleared (`activity.clear_active` does it), so a stopped brick
  can't leave a number behind.
- sticky: the last result of a brick that answers and goes quiet. Kept while
  its model stays loaded, so the panel can say "last: 44 tok/s".
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

_lock = threading.Lock()
_metrics: dict[tuple[str, str], dict] = {}


def report(
    demo_id: str,
    value: float,
    unit: str,
    *,
    stage: str = "default",
    sticky: bool = False,
    detail: str | None = None,
) -> None:
    entry = {
        "demo_id": demo_id,
        "stage": stage,
        "value": round(float(value), 2),
        "unit": unit,
        "sticky": sticky,
        "detail": detail,
        "at": time.time(),
    }
    with _lock:
        _metrics[(demo_id, stage)] = entry


def clear_live(demo_id: str, stage: str = "default") -> None:
    """Drop a stage's number when the stage stops -- unless it is a sticky
    one, which outlives the request that produced it."""
    with _lock:
        entry = _metrics.get((demo_id, stage))
        if entry is not None and not entry["sticky"]:
            del _metrics[(demo_id, stage)]


def clear(demo_id: str) -> None:
    """Everything a brick reported, sticky included: its model is gone."""
    with _lock:
        for key in [key for key in _metrics if key[0] == demo_id]:
            del _metrics[key]


def snapshot() -> list[dict]:
    with _lock:
        return [dict(entry) for entry in _metrics.values()]


class RateMeter:
    """Events per second over the last couple of seconds: a frame rate that
    follows what is happening now, not an average since the brick started."""

    def __init__(self, window: float = 2.0, clock: Callable[[], float] = time.monotonic):
        self._window = window
        self._clock = clock
        self._times: deque[float] = deque()

    def tick(self) -> float:
        now = self._clock()
        self._times.append(now)
        while self._times and now - self._times[0] > self._window:
            self._times.popleft()
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0
