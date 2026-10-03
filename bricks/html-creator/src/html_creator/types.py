"""Shared result types for the html-creator brick."""
from __future__ import annotations

from dataclasses import dataclass, field

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
    # Pictures (see pictures.py): how many the model was offered, which it
    # placed, and why any were left out. `html` has them embedded;
    # `html_source` is the page as the model wrote it, file names and all
    # (None when there was nothing to embed).
    pictures_offered: int = 0
    pictures_used: list[str] = field(default_factory=list)
    picture_notes: list[str] = field(default_factory=list)
    html_source: str | None = None
    # Written on a freshly loaded model, without sampling: the same prompt
    # gives this page again.
    repeatable: bool = False
