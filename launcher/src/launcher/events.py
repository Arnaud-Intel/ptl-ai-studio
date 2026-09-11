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
import logging
import math
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

# launcher/src/launcher/events.py -> repo root is 3 levels up.
LOG_FILE = Path(__file__).resolve().parents[3] / "logs" / "events.log"

_lock = threading.Lock()
_status: dict[str, dict] = {}
_recent: deque[dict] = deque(maxlen=200)
MAX_LOG_BYTES = 1_048_576
LOG_BACKUPS = 3
_history_file: Path | None = None


def _load_history() -> None:
    """Called under _lock. Restore history, never resurrect active workers."""
    global _history_file
    if _history_file == LOG_FILE:
        return
    _recent.clear()
    for path in [Path(f"{LOG_FILE}.{i}") for i in range(LOG_BACKUPS, 0, -1)] + [LOG_FILE]:
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                start = max(0, stream.tell() - MAX_LOG_BYTES)
                stream.seek(start)
                if start:
                    stream.readline(MAX_LOG_BYTES)
                data = stream.read(MAX_LOG_BYTES)
            for line in data.splitlines():
                try:
                    item = json.loads(line)
                    if (isinstance(item, dict) and isinstance(item.get("demo_id"), str)
                            and isinstance(item.get("phase"), str) and isinstance(item.get("message"), str)
                            and isinstance(item.get("at"), (int, float)) and math.isfinite(item["at"])):
                        _recent.append(item)
                except (ValueError, UnicodeError):
                    continue
        except OSError:
            pass
    _history_file = LOG_FILE


def _key(demo_id: str, stage: str | None) -> str:
    return f"{demo_id}:{stage}" if stage else demo_id


def set_phase(demo_id: str, phase: str, message: str = "", *, stage: str | None = None) -> None:
    """phase is "loading", "running", "stopping", or "error". Errors are NOT
    cleared by clear_phase -- they stay visible until the next
    loading/running call overwrites them, so a failed run doesn't silently
    look idle again.

    "stopping" is set by worker.request_stop() when a stop was asked for but
    the worker is still inside a call it can't be interrupted from; the
    worker clears it itself on the way out.

    `stage`, for a demo running several things at once (expense-extract's
    OCR and LLM stages, smart-city-monitor's feeds), tracks each one's
    phase separately, keyed "demo_id:stage" -- the same compound key
    /api/status has always exposed, mirroring activity.set_active's stage.
    """
    key = _key(demo_id, stage)
    entry = {"demo_id": key[:200], "phase": phase, "message": message[:4000], "at": time.time()}
    with _lock:
        _load_history()
        _status[key] = entry
        _recent.append(entry)
        _append_to_file(entry)


def clear_phase(demo_id: str, *, stage: str | None = None) -> None:
    with _lock:
        _status.pop(_key(demo_id, stage), None)


def status_snapshot() -> dict[str, dict]:
    with _lock:
        return dict(_status)


def recent_events(limit: int = 100) -> list[dict]:
    with _lock:
        _load_history()
        events = list(_recent)
    return events[-min(limit, 200):] if limit > 0 else []


def _append_to_file(entry: dict) -> None:
    # Best-effort: a logging failure must never break the actual request.
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if LOG_FILE.exists() and LOG_FILE.stat().st_size:
            with LOG_FILE.open("rb+") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    f.write(b"\n")
        handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
        try:
            handler.emit(logging.LogRecord("activity", logging.INFO, "", 0, json.dumps(entry), (), None))
        finally:
            handler.close()
    except OSError:
        pass
