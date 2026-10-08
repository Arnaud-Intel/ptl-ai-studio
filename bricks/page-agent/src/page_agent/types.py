"""Shared result types for the page-agent brick."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pantherlake_ai_core.types import GenerationStats

from .plan import PagePlan


@dataclass
class Assignment:
    """Which chip does which part of the work."""

    planner: str
    images: str
    page: str
    # The pictures are drawn while the page is written, on two different
    # chips; otherwise one chip does them in turn, each model unloaded
    # before the other is loaded.
    together: bool


@dataclass
class DrawnPicture:
    name: str
    path: Path
    width: int
    height: int
    prompt: str
    seconds: float


@dataclass
class Check:
    """One thing the conductor verified about the finished page."""

    name: str
    passed: bool
    detail: str = ""
    # Failed in a way a second writing of the page could fix and is worth
    # the minute and a half it takes.
    retry: bool = False


@dataclass
class PageResult:
    html: str
    plan: PagePlan
    assignment: Assignment
    pictures: list[DrawnPicture] = field(default_factory=list)
    pictures_used: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    # How many times the page was written: 2 means the first one failed a
    # check that a second try could fix.
    attempts: int = 1
    # Seconds per step ("plan", "images", "page", "total"); with two GPUs the
    # pictures and the page overlap, so the steps add up to more than the total.
    seconds: dict[str, float] = field(default_factory=dict)
    planner_stats: GenerationStats | None = None
    stats: GenerationStats | None = None  # the page model's
    # The page as the model wrote it, with file names where `html` has the
    # pictures themselves.
    html_source: str | None = None
    cancelled: bool = False
