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
    # Which picker section it sits in. Not decoration: YouTube can refuse a
    # whole network at once ("confirm you're not a bot", seen 2026-09-11 on
    # every camera here), and when it does, the operator needs to see at a
    # glance which cameras don't go through YouTube at all.
    group: str = "YouTube"


@dataclass
class Sample:
    name: str
    description: str
    feeds: str  # what goes in the feeds box: one source per line
    group: str = "YouTube"


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

DUBLIN = LiveFeed(
    name="Dublin, Ireland",
    description="A busy Dublin street corner -- a steady mix of people on foot and passing cars.",
    url="https://www.youtube.com/watch?v=3nyPER2kzqk",
)
VENICE = LiveFeed(
    name="Venice, Italy",
    description="A canal and bridge from a hotel window. Scenic rather than countable -- people are small and far off, so the numbers stay low.",
    url="https://www.youtube.com/watch?v=mt7uE-n0YPI",
)
AMSTERDAM = LiveFeed(
    name="Amsterdam, Netherlands",
    description="Damrak at Beursplein, a main city-centre street -- heavy on traffic.",
    url="https://www.youtube.com/watch?v=43qH0tDA6lM",
)
TOKYO = LiveFeed(
    name="Tokyo, Japan",
    description="A Shinjuku crossing seen from further back than the Kabukicho camera -- mostly vehicles.",
    url="https://www.youtube.com/watch?v=6dp-bvQ7RWo",
)

# London traffic cameras from TfL's open-data JamCams ("Powered by TfL Open
# Data"): no YouTube, no account, and daylight during European show hours.
# Each is a ~10-second clip TfL replaces every few minutes, played once
# per revision (video.stream_refreshing_clip) -- so the picture pauses between
# clips instead of replaying one and counting the same cars again. At 352x288
# they count vehicles well and people poorly -- TfL downsamples the footage for GDPR on purpose; measured per sampled frame on
# 2026-09-11: Piccadilly/St James's 7.8 vehicles, Tower Bridge 6.1, Westminster
# Bridge 4.6, Piccadilly Circus 1.5 vehicles and 1.5 people.
_JAMCAM = "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/{}.mp4"
_JAMCAM_NOTE = "TfL JamCam: a 10-second clip refreshed every few minutes and downsampled by TfL for privacy, so it counts vehicles far better than people."

PICCADILLY_ST_JAMES = LiveFeed(
    name="Piccadilly / St James's St, London",
    description=f"Buses and black cabs on Piccadilly. {_JAMCAM_NOTE}",
    url=_JAMCAM.format("00001.06592"),
    group="Other",
)
TOWER_BRIDGE = LiveFeed(
    name="Tower Bridge approach, London",
    description=f"Traffic queuing for Tower Bridge. {_JAMCAM_NOTE}",
    url=_JAMCAM.format("00001.03500"),
    group="Other",
)
WESTMINSTER_BRIDGE = LiveFeed(
    name="Westminster Bridge, London",
    description=f"The approach to Westminster Bridge. {_JAMCAM_NOTE}",
    url=_JAMCAM.format("00001.04502"),
    group="Other",
)
PICCADILLY_CIRCUS = LiveFeed(
    name="Piccadilly Circus, London",
    description=f"Iconic, but sparse at this resolution. {_JAMCAM_NOTE}",
    url=_JAMCAM.format("00001.07450"),
    group="Other",
)

LIVE_FEEDS: list[LiveFeed] = [
    SHINJUKU,
    SHIBUYA,
    ABBEY_ROAD,
    MELBOURNE,
    DUBLIN,
    VENICE,
    AMSTERDAM,
    TOKYO,
    PICCADILLY_ST_JAMES,
    TOWER_BRIDGE,
    WESTMINSTER_BRIDGE,
    PICCADILLY_CIRCUS,
]

SAMPLES: list[Sample] = [
    Sample(name=f.name, description=f.description, feeds=f.url, group=f.group) for f in LIVE_FEEDS
] + [
    Sample(
        name="Two cities, two chips",
        description="Tokyo on the NPU and Melbourne on the GPU at once -- watch both gauges light up.",
        feeds=f"{SHINJUKU.url}|NPU\n{MELBOURNE.url}|GPU",
    ),
    Sample(
        name="London, two chips",
        description="Two TfL cameras at once, one on the NPU and one on the GPU -- works even when YouTube blocks the network.",
        feeds=f"{PICCADILLY_ST_JAMES.url}|NPU\n{TOWER_BRIDGE.url}|GPU",
        group="Other",
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
