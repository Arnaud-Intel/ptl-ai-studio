"""Shared result types for the html-creator brick."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HtmlResult:
    html: str
    mode: str
    source_char_count: int
    source_truncated: bool
    fence_stripped: bool
    html_truncated: bool
