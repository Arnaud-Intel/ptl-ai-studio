"""Videos the detector can be pointed at straight away: the studio's own
sample videos (core's `sample_videos`), fetched once from where their
authors published them and played from this machine."""
from __future__ import annotations

from dataclasses import dataclass

from pantherlake_ai_core import sample_videos


@dataclass
class Sample:
    name: str
    description: str
    path: str  # the video file, on this machine
    kind: str = "file"
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


# Toronto first: the one with the most in it at this detector's size --
# thirteen boxes a frame (people, cars, traffic lights), against five in
# Tokyo, where the crowd is seen from too far above to be told apart.
SAMPLES: list[Sample] = [
    _sample(sample_videos.TORONTO, default=True),
    _sample(sample_videos.TYUMEN),
    _sample(sample_videos.INTEL_STREET),
    _sample(sample_videos.TOKYO),
]
