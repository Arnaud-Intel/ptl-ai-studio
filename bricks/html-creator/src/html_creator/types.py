"""Shared result types for the html-creator brick."""
from __future__ import annotations

from dataclasses import dataclass

from pantherlake_ai_core.types import GenerationStats


@dataclass
class HtmlResult:
    html: str
    mode: str
    source_char_count: int
    source_truncated: bool
    fence_stripped: bool
    html_truncated: bool
    stats: GenerationStats | None = None
