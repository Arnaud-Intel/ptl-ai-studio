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
    # The denoising steps it took, and what they took by the runtime's own
    # count (None where that cannot be read): `seconds` also holds reading
    # the description, decoding the picture and writing the file.
    steps: int = 0
    denoise_seconds: float | None = None


@dataclass
class PictureStats:
    """How fast a build's pictures were drawn, in an image model's own units
    -- it never produces a token. `seconds` is the drawing alone, loading
    the model left out. `steps_per_second` is the rate of the denoising
    steps, the figure image models are compared by; it is taken from the
    runtime's own counters and is None where they could not be read."""

    device: str
    pictures: int
    seconds: float
    megapixels: float
    steps: int
    images_per_minute: float
    steps_per_second: float | None = None

    @classmethod
    def of(cls, pictures: list[DrawnPicture], device: str) -> PictureStats | None:
        seconds = sum(picture.seconds for picture in pictures)
        if not pictures or seconds <= 0:
            return None
        steps = sum(picture.steps for picture in pictures)
        timed = [picture for picture in pictures if picture.denoise_seconds]
        denoising = sum(picture.denoise_seconds for picture in timed)
        return cls(
            device=device,
            pictures=len(pictures),
            seconds=round(seconds, 2),
            megapixels=round(sum(picture.width * picture.height for picture in pictures) / 1e6, 2),
            steps=steps,
            images_per_minute=round(60 * len(pictures) / seconds, 2),
            steps_per_second=round(sum(picture.steps for picture in timed) / denoising, 2) if denoising > 0 else None,
        )


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
    picture_stats: PictureStats | None = None
    # Seconds this build spent loading each step's model ("plan", "images",
    # "page"): what the first build costs that the next ones do not. A model
    # already in memory is not in it.
    loads: dict[str, float] = field(default_factory=dict)
    # The page as the model wrote it, with file names where `html` has the
    # pictures themselves.
    html_source: str | None = None
    cancelled: bool = False
