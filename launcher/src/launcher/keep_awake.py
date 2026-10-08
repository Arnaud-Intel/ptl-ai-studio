"""Keeps the machine awake while it runs with nobody at it.

An Auto Demo on a stand is a laptop nobody touches for hours, and Windows
answers that by dimming the screen, then sleeping. A program can ask it not
to, for as long as it has a reason: the same request a video player makes.
It is a request of the thread that makes it, and ends with that thread or
when it is taken back -- no setting of the machine is changed, and nothing
is left behind if the launcher is killed. A lock forced by company policy
is not ours to hold off: that one has to be lifted for the stand.
"""
from __future__ import annotations

import sys

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def hold() -> bool:
    """Ask that the system and its display stay on. Call it on the thread
    that does the work, and `release()` on that same thread. False where
    there is nothing to ask (not Windows) or the request was refused."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:
        return False


def release() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    except Exception:
        pass
