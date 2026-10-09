"""The videos the commentator can be pointed at straight away: the studio's
own sample videos (core's `sample_videos`), fetched once from where their
authors published them and played from this machine."""
from __future__ import annotations

from dataclasses import dataclass

from pantherlake_ai_core import sample_videos


@dataclass
class Sample:
    name: str
    description: str
    path: str  # the video file, on this machine
    # The sample videos it plays, by key: fetched the first time it is
    # started if "Prepare models" has not fetched them already.
    videos: tuple[str, ...] = ()
    default: bool = False  # what the panel opens with


def _sample(video: sample_videos.SampleVideo, default: bool = False) -> Sample:
    return Sample(
        name=video.name,
        description=f"{video.description} {video.licence}, {video.credit}.",
        path=str(video.path), videos=(video.key,), default=default,
    )


# The herd first: the picture with the most to say about -- animals, a rider,
# a dog, a road, a sky -- and none of it a face in a crowd.
SAMPLES: list[Sample] = [
    _sample(sample_videos.CATTLE_DRIVE, default=True),
    _sample(sample_videos.CAPPING_LINE),
    _sample(sample_videos.TORONTO),
    _sample(sample_videos.TYUMEN),
    _sample(sample_videos.TOKYO),
]
