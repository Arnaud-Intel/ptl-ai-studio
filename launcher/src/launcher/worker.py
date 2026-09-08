"""The background-thread bookkeeping every streaming brick's runner shares.

A runner owns one worker thread and one `threading.Event` it watches. Both
things that can be said about that pair -- "you can't start now, and here
is why" and "stop, and tell the truth about how that went" -- are the same
in all eight runners, so they live here rather than in eight copies.

The subtlety both functions exist for: a stop event is *cooperative*. The
worker only sees it between steps, so a thread inside a model load or a
single long inference (screen-ocr's 7B vision-language model can take a
minute on one frame) keeps running for a while after Stop is pressed. That
is fine -- it does stop, and nothing leaks -- but the UI must not go on
claiming the brick is doing its normal work in the meantime.
"""
from __future__ import annotations

import threading
from typing import Sequence

from . import events
from .errors import Conflict

# Long enough that an ordinary stop (every loop checks the event at least
# every ~0.1s) finishes inside it and the UI never flickers "Stopping...";
# short enough that a genuinely stuck-in-a-call brick says so promptly
# instead of holding the request open.
_GRACE_SECONDS = 1.0

STOPPING_MESSAGE = "Stopping -- finishing the current step..."


def is_stopping(thread: threading.Thread | None, stop_event: threading.Event | None) -> bool:
    """True while a stop has been asked for but the worker hasn't come back."""
    return (
        thread is not None
        and thread.is_alive()
        and stop_event is not None
        and stop_event.is_set()
    )


def refuse_if_busy(demo_id: str, thread: threading.Thread | None, stop_event: threading.Event | None) -> None:
    """Guard a runner's start(). Raises Conflict if a run is in progress,
    distinguishing "already running" from "still winding down" -- starting
    a second thread on the same camera/mic/index is exactly what the wait
    in request_stop() protects against, and the caller deserves to know
    which of the two it hit, since only one of them clears on its own."""
    if thread is None or not thread.is_alive():
        return
    if is_stopping(thread, stop_event):
        raise Conflict(f"{demo_id} is still stopping -- try again in a moment")
    raise Conflict(f"{demo_id} is already running")


def request_stop(
    demo_id: str,
    thread: threading.Thread | None,
    stop_event: threading.Event | None,
    *,
    stages: Sequence[str | None] = (None,),
    grace: float = _GRACE_SECONDS,
) -> bool:
    """Ask a runner's worker to stop, and wait a moment for it to land.

    Returns True if it finished -- the caller can then clear its state.
    Returns False if it is still inside a call it can't be interrupted
    from, having first set each stage's phase to "stopping", so the UI
    says what is actually happening instead of the brick's normal running
    message. The worker clears that phase itself when it finally exits.
    """
    if stop_event is not None:
        stop_event.set()
    if thread is None:
        return True
    thread.join(timeout=grace)
    if not thread.is_alive():
        return True
    for stage in stages:
        events.set_phase(demo_id, "stopping", STOPPING_MESSAGE, stage=stage)
    return False
