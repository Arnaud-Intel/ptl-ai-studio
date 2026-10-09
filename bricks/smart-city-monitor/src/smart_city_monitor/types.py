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
    counting: str = "street"  # what is counted in it: a key of COUNTING


# What a feed is watched for: the detector's labels worth a box and a count
# there, and the name each is counted under. Anything else it sees is dropped
# before tracking, so neither the picture nor the counts fill up with chairs.
#
# The labels are ones both engines' vocabularies spell the same way (DETR's
# COCO-91 and YOLO11s's COCO-80), which is why they are single words.
#
# A kind of counting, not a kind of model: it is the same detector on a
# street, a production line and a pasture. It names everyday things; it does
# not judge them -- a dented bottle is a bottle. And the same person is a
# pedestrian in one place and a worker in another, which is why this is a
# choice per feed and not one longer list.
COUNTING: dict[str, dict[str, str]] = {
    "street": {
        "person": "Pedestrians",
        "bicycle": "Bicycles",
        "car": "Cars",
        "motorcycle": "Motorcycles",
        "bus": "Buses",
        "truck": "Trucks",
    },
    "line": {
        "bottle": "Bottles",
        "cup": "Cups",
        "apple": "Apples",
        "orange": "Oranges",
        "banana": "Bananas",
        "broccoli": "Broccoli",
        "carrot": "Carrots",
        "donut": "Donuts",
        "person": "Workers",
    },
    "herd": {
        "cow": "Cattle",
        "sheep": "Sheep",
        "horse": "Horses",
        "dog": "Dogs",
        "bird": "Birds",
        "person": "People",
    },
}
COUNTING_NAMES: dict[str, str] = {"street": "Street traffic", "line": "Production line", "herd": "Herd"}


def labels_counted(counting: str) -> dict[str, str]:
    try:
        return COUNTING[counting]
    except KeyError:
        raise ValueError(f"Unknown kind of counting '{counting}': one of {', '.join(COUNTING)}.") from None


@dataclass
class CountSnapshot:
    """A point-in-time read of every feed's counts, for the launcher's
    dashboard and the CLI's periodic summary line."""

    combined_last_60s: dict[str, int] = field(default_factory=dict)
    combined_total: dict[str, int] = field(default_factory=dict)
    per_feed_last_60s: dict[str, dict[str, int]] = field(default_factory=dict)
    per_feed_total: dict[str, dict[str, int]] = field(default_factory=dict)
    active_feeds: list[str] = field(default_factory=list)
    count_basis: str = "new_tracks"
    accuracy_warning: str = "Experimental track counts, not unique objects. Lost tracks can be counted again."
