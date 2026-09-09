"""The curated live city cameras behind the "pick a feed" control.

These are public 24/7 street cameras -- real pedestrians, real traffic,
at real scale, which a synthetic clip can't stand in for. They are also
the honest version of this demo: counting people at a Tokyo crossing is
what a smart-city monitor is actually for.

Two caveats worth knowing, both stated in the UI as well:

- They come over the internet. The *video* is a network stream; the
  detection still runs entirely on local silicon, and nothing about the
  frames is sent anywhere.
- They are somebody else's cameras. A stream can be renamed, rate-limited
  or taken down without notice, which is why a local file stays the
  option that always works.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveFeed:
    name: str
    description: str
    url: str


@dataclass
class Sample:
    name: str
    description: str
    feeds: str  # what goes in the feeds box: one source per line


SHINJUKU = LiveFeed(
    name="Shinjuku Kabukicho",
    description="Tokyo's busiest nightlife street, live. Dense pedestrian traffic, close to the camera.",
    url="https://www.youtube.com/watch?v=DjdUEyjx8GM",
)
SHIBUYA = LiveFeed(
    name="Shibuya Scramble Crossing",
    description="The scramble crossing from above -- heavy on buses and cars, lighter on countable people.",
    url="https://www.youtube.com/watch?v=dfVK7ld38Ys",
)
ABBEY_ROAD = LiveFeed(
    name="Abbey Road Crossing",
    description="The London zebra crossing, at street level: a steady mix of pedestrians and cars.",
    url="https://www.youtube.com/watch?v=zMCea32gpmg",
)
MELBOURNE = LiveFeed(
    name="Melbourne city traffic",
    description="A busy Melbourne street -- the most vehicle-heavy of these, good for the class breakdown.",
    url="https://www.youtube.com/watch?v=gEbrHdFRgpQ",
)

LIVE_FEEDS: list[LiveFeed] = [SHINJUKU, SHIBUYA, ABBEY_ROAD, MELBOURNE]

SAMPLES: list[Sample] = [
    Sample(name=f.name, description=f.description, feeds=f.url) for f in LIVE_FEEDS
] + [
    Sample(
        name="Two cities, two chips",
        description="Tokyo on the NPU and Melbourne on the GPU at once -- watch both gauges light up.",
        feeds=f"{SHINJUKU.url}|NPU\n{MELBOURNE.url}|GPU",
    ),
]


def label_for(source: str) -> str | None:
    """The human name of a curated feed, or None for anything else.

    Every YouTube URL looks the same once reduced to a hostname, so
    running both Tokyo cameras at once would otherwise give two feeds
    called "www.youtube.com" -- and the per-feed counts are the whole
    point of running two.
    """
    return next((f.name for f in LIVE_FEEDS if f.url == source), None)
