"""Shared result types for the smart-city-monitor brick."""
from __future__ import annotations

from dataclasses import dataclass, field

from pantherlake_ai_core.engine import Engine


@dataclass
class TrackedDetection:
    """A `object_detection.types.Detection` plus a persistent track id.

    Carries the same `label`/`confidence`/`box` fields as `Detection` (a
    superset, not a different shape) so it can be drawn/consumed anywhere
    a `Detection`-like object is expected.
    """

    track_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    is_new: bool  # True only on the frame this track was first created -- the "count" event


@dataclass
class FeedSpec:
    """One feed and everything about how it should be run.

    Engine and model live here, not just on the run as a whole, so two
    feeds can genuinely use different backends at once -- the point of the
    brick generalized one step further than "same model, different chip".
    Both default to the run's choice when left unset.
    """

    feed_id: str
    path: str
    compute_device: str
    name: str = ""  # display name (e.g. the file's basename); defaults to feed_id if unset
    engine: Engine | None = None  # None -> the engine the run was started with
    model_path: str | None = None  # None -> that engine's built-in model


@dataclass
class CountSnapshot:
    """A point-in-time read of every feed's counts, for the launcher's
    dashboard and the CLI's periodic summary line."""

    combined_last_60s: dict[str, int] = field(default_factory=dict)
    combined_total: dict[str, int] = field(default_factory=dict)
    per_feed_last_60s: dict[str, dict[str, int]] = field(default_factory=dict)
    per_feed_total: dict[str, dict[str, int]] = field(default_factory=dict)
    active_feeds: list[str] = field(default_factory=list)
