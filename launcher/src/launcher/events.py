"""Per-demo lifecycle status (for a UI "what's happening" indicator) plus a
persisted event log (for reviewing what happened later).

Mirrors activity.py's shape (lock-guarded, in-memory) but tracks a
different, parallel concern: activity.py says *which device* a demo is
driving right now (for telemetry-gauge labeling); this module says *what
phase* a demo is in (loading a model, actively running, or failed) and
keeps a short history of those transitions.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path

# launcher/src/launcher/events.py -> repo root is 3 levels up.
LOG_FILE = Path(__file__).resolve().parents[3] / "logs" / "events.log"

_lock = threading.Lock()
_status: dict[str, dict] = {}
_recent: deque[dict] = deque(maxlen=200)


def set_phase(demo_id: str, phase: str, message: str = "") -> None:
    """phase is "loading", "running", or "error". Errors are NOT cleared by
    clear_phase -- they stay visible until the next loading/running call
    overwrites them, so a failed run doesn't silently look idle again."""
    entry = {"demo_id": demo_id, "phase": phase, "message": message, "at": time.time()}
    with _lock:
        _status[demo_id] = entry
        _recent.append(entry)
    _append_to_file(entry)


def clear_phase(demo_id: str) -> None:
    with _lock:
        _status.pop(demo_id, None)


def status_snapshot() -> dict[str, dict]:
    with _lock:
        return dict(_status)


def recent_events(limit: int = 100) -> list[dict]:
    with _lock:
        events = list(_recent)
    return events[-limit:]


def _append_to_file(entry: dict) -> None:
    # Best-effort: a logging failure must never break the actual request.
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
