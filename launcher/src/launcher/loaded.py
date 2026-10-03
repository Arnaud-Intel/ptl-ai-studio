"""Bricks that answer one request at a time keep their model in memory
between requests. That is what makes the second answer fast, and also what
quietly holds 17 GB after a code review nobody is looking at any more
(BACKLOG R11). The side panel lists them under the chip they sit on, and its
close button calls `unload`.

The five runners this covers share a shape -- `_session`, `_engine`,
`_device`, `_lock` -- so the rule lives here once instead of five times.
"""
from __future__ import annotations

import gc

from . import events, metrics
from .errors import Conflict


def info(runner) -> dict | None:
    """Engine and device of the model a runner is holding, or None."""
    session = getattr(runner, "_session", None)
    if session is None:
        return None
    # The session knows where it really loaded (screen-ocr's does); the
    # runner only knows what was asked for.
    return {"engine": runner._engine, "device": getattr(session, "device", None) or runner._device}


def unload(runner, demo_id: str) -> bool:
    """Drop the runner's model. True if there was one. Refuses while a
    request is in flight rather than pulling the model out from under it."""
    if not runner._lock.acquire(blocking=False):
        raise Conflict("It is in the middle of a request -- try again when that has finished.")
    try:
        had_model = runner._session is not None
        runner._session = None
        runner._engine = None
        runner._device = None
        # voice-clone keeps two facts about the session it no longer has.
        if hasattr(runner, "_enrolled"):
            runner._enrolled = False
        if hasattr(runner, "_model"):
            runner._model = None
    finally:
        runner._lock.release()
    gc.collect()  # the runtime frees the model's memory with its last reference
    metrics.clear(demo_id)
    events.clear_phase(demo_id)
    return had_model
