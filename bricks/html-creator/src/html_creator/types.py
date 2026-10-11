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


@dataclass
class Revised:
    """One part of a page written again (see revise.py)."""

    # The page as it is now: the new part where the old one was, and the
    # rules and the script that came with it. The page as it was, when
    # `problem` says the change could not be made.
    source: str
    number: int  # the part's number in the page it was pointed at in
    comment: str
    before: str = ""  # the part as it was
    after: str = ""  # and as it is
    css: str = ""  # the style rules added for it
    script: str = ""  # the script added for it
    plan: str = ""  # what the model said it would change, in a line
    problem: str = ""  # why the change was not made, if it was not
    notes: list[str] = field(default_factory=list)
    attempts: int = 1
    seconds: float = 0.0
    stats: GenerationStats | None = None
