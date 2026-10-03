"""An answer while it is being written: the text so far, how fast it is
coming, and whether it has been asked to stop (BACKLOG R32).

A runner calls `get(demo_id).begin()` and hands the returned control to its
brick; every piece of text the model writes lands here, and every token is
counted. Three things read it: the page, which asks for the text a few times
a second and shows it growing; the hardware panel, through a live
tokens-per-second figure; and the stop route, which sets the flag the model
checks after every piece.

Polled rather than pushed, on purpose: a missed ask loses nothing (the next
one has more text), and it adds no delivery mechanism to keep alive.
"""
from __future__ import annotations

import threading

from pantherlake_ai_core.types import GenerationControl

from . import metrics

# The live rate follows the last second and a half: long enough to be
# steady, short enough to show a model slowing down.
_RATE_WINDOW = 1.5
# ...and waits for a few tokens before it says anything: two tokens that
# happen to land together are not a rate.
_RATE_MIN_TOKENS = 8


class LiveGeneration:
    def __init__(self, demo_id: str, stage: str = "default"):
        self.demo_id = demo_id
        self.stage = stage
        self._lock = threading.Lock()
        self._pieces: list[str] = []
        self._active = False
        self._was_cancelled = False
        self._cancel = threading.Event()
        self._new_rate()

    def begin(self) -> GenerationControl:
        """Start following a new answer. Returns what the brick passes to
        the model: where each piece goes, and how to ask whether to stop."""
        with self._lock:
            self._pieces = []
            self._active = True
            self._was_cancelled = False
        self._cancel.clear()
        self._new_rate()
        return GenerationControl(
            on_text=self._on_text,
            should_stop=self._cancel.is_set,
            on_heading=self._on_heading,
            on_tokens=self._on_tokens,
        )

    def _new_rate(self) -> None:
        self._rate = metrics.RateMeter(window=_RATE_WINDOW)
        self._counted = 0

    def _on_text(self, piece: str) -> None:
        with self._lock:
            self._pieces.append(piece)

    def _on_tokens(self, count: int) -> None:
        rate = 0.0
        for _ in range(count):
            rate = self._rate.tick()
        self._counted += count
        if rate and self._counted >= _RATE_MIN_TOKENS:
            metrics.report(self.demo_id, rate, "tok/s", stage=self.stage)

    def _on_heading(self, text: str) -> None:
        """A line the brick wrote, not the model: it joins the text, and the
        rate starts afresh -- the model reads its next prompt before writing
        again, and that wait is not slow writing."""
        with self._lock:
            self._pieces.append(text)
        self._new_rate()

    def end(self) -> None:
        with self._lock:
            self._active = False
            self._was_cancelled = self._cancel.is_set()

    def cancel(self) -> bool:
        """Ask the answer in flight to stop. False if there isn't one."""
        with self._lock:
            active = self._active
        if active:
            self._cancel.set()
        return active

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "active": self._active,
                "text": "".join(self._pieces),
                "cancelled": self._cancel.is_set() if self._active else self._was_cancelled,
            }


_lock = threading.Lock()
_live: dict[tuple[str, str], LiveGeneration] = {}


def get(demo_id: str, stage: str = "default") -> LiveGeneration:
    with _lock:
        return _live.setdefault((demo_id, stage), LiveGeneration(demo_id, stage))


def snapshot(demo_id: str, stage: str = "default") -> dict:
    with _lock:
        live = _live.get((demo_id, stage))
    return live.snapshot() if live else {"active": False, "text": "", "cancelled": False}


def cancel(demo_id: str, stage: str | None = None) -> bool:
    """Stop the answer a brick is writing -- one stage's, or any of them.
    True if something was in flight."""
    with _lock:
        targets = [live for (demo, live_stage), live in _live.items() if demo == demo_id and stage in (None, live_stage)]
    return any([live.cancel() for live in targets])  # a list, so every one is asked


def in_flight(demo_id: str) -> bool:
    with _lock:
        targets = [live for (demo, _), live in _live.items() if demo == demo_id]
    return any(live.active for live in targets)
