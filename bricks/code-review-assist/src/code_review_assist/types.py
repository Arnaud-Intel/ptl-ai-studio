"""Shared result types for the code-review-assist brick."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReviewResult:
    commit_message: str
    review_notes: str
    diff_char_count: int
    diff_truncated: bool
