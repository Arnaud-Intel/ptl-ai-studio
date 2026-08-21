"""Shared result type for this brick."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExpenseLine:
    source_file: str
    vendor: str
    date: str
    amount: float | None
    category: str
    raw_text: str
    error: str | None = None
