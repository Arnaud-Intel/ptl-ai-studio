"""Living with an NPU that Windows can take away mid-demo.

What happens (XPS 14, NPU driver 32.0.100.5540; seen in the launcher on
2026-10-05 and 2026-10-06, reproduced outside it on 2026-10-06): while one
model is running on the NPU, a second one starts on it and, a few seconds
in, every request on the chip fails with

    L0 zeFenceHostSynchronize result: ZE_RESULT_ERROR_DEVICE_LOST ...
    device hung, reset, was removed, or driver update occurred

Windows has reset the NPU. That much is an error like any other. What made
it a crash is what came next: live translation reloaded its model and tried
again, the expense extractor moved on to its next receipt, and 20-30 s
later the NPU driver itself ended the process from inside
`npu_level_zero_umd.dll` (0xC0000409 one day, 0xC0000005 the next) -- the
whole launcher gone, with nothing Python can catch. The same driver
version and the same fault offset are in an open OpenVINO report
(openvinotoolkit/openvino#38403), so the cause is upstream.

It is not ours to fix, and it depends on what else is going on. Two models
alone on the NPU lost it twice in some twenty runs, each time within
seconds of the second model's first request; five at once ran five
minutes clean. The launcher's own scenario -- the 30B model writing a page
on the integrated GPU, the vision model loading there, speech and the
small language model on the NPU -- lost it in two runs out of three. So
this module does two things:

- **Turns** (`guard`). One call into the NPU at a time, across every
  brick, first come first served. With turns, that same scenario ran four
  times without a loss. Four runs against three is an indication, not
  proof. A long answer does not hold the others up: it steps aside
  between tokens (`breathe`).

- **No second chance on a lost chip** (`guard`, `NpuLost`, `retire`).
  The first time the driver reports the loss, the NPU is out of use for
  the rest of the process: `guard` refuses before touching it, so nothing
  can reload onto it, retry on it or free memory on it. Models that can
  carry on elsewhere move to `fallback_device()` (the speech and language
  models do); the rest stop with `NpuLost`, whose text is written for the
  person at the keyboard. Checked against a real loss on 2026-10-06, turns
  switched off to let it happen: both models moved to the GPU, the run
  went on to its end and the process left normally. Restarting the app
  brings the NPU back.
"""
from __future__ import annotations

import collections
import threading
import time
from contextlib import contextmanager
from typing import Iterator

# The driver's own words for it, as OpenVINO passes them through.
_LOSS_MARKS = ("ZE_RESULT_ERROR_DEVICE_LOST", "device hung, reset, was removed")

LOST_MESSAGE = (
    "Windows reset the NPU while it was in use (a fault in the NPU driver), so the NPU is out of use "
    "until the app is restarted. Everything else still works: start this demo again and it will run on "
    "another chip."
)


class NpuLost(RuntimeError):
    """The NPU was reset under this process and must not be used again."""


class _Turns:
    """A first-come-first-served lock. Python's own makes no promise about
    who gets it next, and a model that steps aside between tokens has to
    know the one waiting really goes first."""

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._waiting: collections.deque[threading.Event] = collections.deque()
        self._held = False

    def acquire(self) -> None:
        with self._mutex:
            if not self._held:
                self._held = True
                return
            mine = threading.Event()
            self._waiting.append(mine)
        mine.wait()  # release() hands the lock straight to the head of the line

    def release(self) -> None:
        with self._mutex:
            if self._waiting:
                self._waiting.popleft().set()
            else:
                self._held = False

    def someone_waiting(self) -> bool:
        return bool(self._waiting)


_turns = _Turns()
_holding = threading.local()  # how deep this thread is inside guard(), so nesting is harmless
_state_lock = threading.Lock()
_lost_reason: str | None = None
_lost_at: float | None = None
_retired: list[object] = []


def is_npu(device: str | None) -> bool:
    return bool(device) and device.upper().startswith("NPU")


def lost() -> str | None:
    """The driver's message if the NPU has been lost in this process, else None."""
    return _lost_reason


def lost_at() -> float | None:
    """When it was lost (`time.time()`), for telling the user."""
    return _lost_at


def says_lost(exc: BaseException) -> bool:
    text = str(exc)
    return any(mark in text for mark in _LOSS_MARKS)


def _mark_lost(exc: BaseException) -> None:
    global _lost_reason, _lost_at
    with _state_lock:
        if _lost_reason is None:
            _lost_reason = " ".join(str(exc).split())
            _lost_at = time.time()


@contextmanager
def guard(device: str | None) -> Iterator[None]:
    """Wrap every call that reaches the NPU: loading a model onto it and
    each inference. Does nothing for any other device.

    Waits its turn, so two bricks never have NPU requests in flight
    together. Refuses with `NpuLost` once the chip has been lost, and turns
    the driver's own report of the loss into `NpuLost` the first time it
    appears."""
    if not is_npu(device):
        yield
        return
    if _lost_reason is not None:
        raise NpuLost(LOST_MESSAGE)
    depth = getattr(_holding, "depth", 0)
    if depth == 0:
        _turns.acquire()
        if _lost_reason is not None:  # lost while this call was waiting for its turn
            _turns.release()
            raise NpuLost(LOST_MESSAGE)
    _holding.depth = depth + 1
    try:
        yield
    except NpuLost:
        raise
    except RuntimeError as exc:
        if says_lost(exc):
            _mark_lost(exc)
            raise NpuLost(LOST_MESSAGE) from exc
        raise
    finally:
        _holding.depth = depth
        if depth == 0:
            _turns.release()


def breathe() -> None:
    """Call between two steps of a long NPU call (a language model's
    tokens): if another brick is waiting for the NPU, let it go first. A
    no-op outside `guard` and when nobody is waiting."""
    if getattr(_holding, "depth", 0) == 1 and _turns.someone_waiting():
        _turns.release()
        _turns.acquire()


def retire(*models: object) -> None:
    """Keep models that live on a lost NPU from ever being released.
    Releasing one frees its memory on the chip, which is one more request
    to a driver that ends the process on the next thing it is asked. They
    cost their memory until the app closes; a loss is rare."""
    with _state_lock:
        _retired.extend(models)


def fallback_device() -> str:
    """Where a model that was on the NPU carries on: the integrated GPU if
    there is one, else the CPU."""
    from .engine import preferred_device

    picked = preferred_device()
    return "CPU" if is_npu(picked) or picked.upper() == "AUTO" else picked


def _reset_for_tests() -> None:
    global _lost_reason, _lost_at, _turns
    with _state_lock:
        _lost_reason = None
        _lost_at = None
        _retired.clear()
    _turns = _Turns()
    _holding.depth = 0
