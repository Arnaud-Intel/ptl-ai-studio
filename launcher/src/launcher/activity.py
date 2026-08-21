"""Tracks which demo -- and, for a demo with more than one concurrently
active stage, which stage of it -- is currently driving which
engine/device, purely so the telemetry panel can annotate a utilization
reading with *why* -- e.g. "NPU 42% -- Expense Report Extractor (OCR)"
instead of a bare number.

Keyed by (demo_id, stage) rather than just demo_id: every demo before
`expense-extract` only ever has one thing running at a time, so `stage`
defaults to a fixed name and those callers can ignore it entirely.
`expense-extract` is the first demo that genuinely runs two stages on two
different devices at once (OCR pinned to one device, an LLM structuring
stage pinned to another, pipelined across a batch of receipts) -- both
need to show up on the telemetry strip simultaneously, which a single
entry per demo_id couldn't represent.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_active: dict[tuple[str, str], dict[str, str]] = {}


def set_active(
    demo_id: str, *, engine: str, device: str, stage: str = "default", stage_label: str | None = None
) -> None:
    with _lock:
        _active[(demo_id, stage)] = {
            "demo_id": demo_id,
            "engine": engine,
            "device": device,
            "stage_label": stage_label,
        }


def clear_active(demo_id: str, stage: str = "default") -> None:
    with _lock:
        _active.pop((demo_id, stage), None)


def snapshot() -> list[dict[str, str]]:
    with _lock:
        return list(_active.values())
