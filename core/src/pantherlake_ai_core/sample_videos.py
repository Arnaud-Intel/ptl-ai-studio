"""The videos the city monitor plays from disk -- streets, a factory line, a
herd: which they are, where they come from, and fetching them.

A video is tens of megabytes: too much to keep in the repository, where every
clone would carry it for good. Each one is fetched instead from where its
author published it, into `sample-data/videos/` (ignored by git), by the
same step that fetches the models -- "Prepare models" in the app,
`panther-lake-prefetch`, the first-launch helper -- or the first time a demo
asks for it.

Each entry says under which licence its video may be passed on and whom to
credit; `sample-data/videos/README.md` says the same beside the files.

A file is used only if it is, byte for byte, the one that was looked at when
it was chosen (size and SHA-256). Two of the six come from Wikimedia
Commons as its own 1080p version of the upload, which Commons is free to
encode again one day: the checksum would then stop matching, and the remedy
is a second address in `urls` -- they are tried in order -- for a copy that
does not change.
"""
from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .demo_samples import SAMPLE_ROOT

VIDEO_DIR = SAMPLE_ROOT / "videos"

# Wikimedia asks every client to say who it is; an anonymous one is refused.
_AGENT = "PantherLakeAIStudio (sample videos; https://github.com/Arnaud-Intel/ptl-ai-studio)"
_CHUNK = 256 * 1024


@dataclass(frozen=True)
class SampleVideo:
    key: str
    name: str  # as a feed is named on screen
    description: str
    filename: str
    urls: tuple[str, ...]  # tried in order
    size_bytes: int
    sha256: str
    licence: str
    credit: str
    source_page: str
    # What there is to count in it, as the city monitor names its kinds of
    # counting: "street", "line" (a production line) or "herd".
    counts: str = "street"

    @property
    def path(self) -> Path:
        return VIDEO_DIR / self.filename


TORONTO = SampleVideo(
    key="toronto-crossing",
    name="Yonge-Dundas crossing, Toronto",
    description="A scramble crossing at street level: pedestrians in every direction, cars, bicycles. 57 seconds, 1080p.",
    filename="toronto-yonge-dundas-crossing.webm",
    urls=("https://upload.wikimedia.org/wikipedia/commons/transcoded/d/d2/DiagonalCrosswalkYongeDundas.webm/"
          "DiagonalCrosswalkYongeDundas.webm.1080p.vp9.webm",),
    size_bytes=36_657_572,
    sha256="f719af0f68cb775e304dcc7d3f66ca4f78735e780eb36bc0e9dafb7ab1b3aa20",
    licence="CC0 1.0",
    credit="Raysonho @ Open Grid Scheduler / Grid Engine, via Wikimedia Commons",
    source_page="https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm",
)
TYUMEN = SampleVideo(
    key="tyumen-crossing",
    name="Respubliki-Ordzhonikidze crossing, Tyumen",
    description="A city crossing at street level in winter sun: a steady stream of cars, pedestrians waiting and crossing. 31 seconds, 1080p.",
    filename="tyumen-crossing.webm",
    # The upload itself, not a version made of it: this one cannot change.
    urls=("https://upload.wikimedia.org/wikipedia/commons/8/8f/"
          "Kruci%C4%9Do_de_stratoj_Respubliko_kaj_Or%C4%9Donikidze_%28Tjumeno%29.webm",),
    size_bytes=10_360_605,
    sha256="a9aa8b656a5cc0881f653cc4ca6ee21bc8060dcda0a96dc1a0cc5769b66aedea",
    licence="CC BY-SA 4.0",
    credit="RG72, via Wikimedia Commons",
    source_page="https://commons.wikimedia.org/wiki/File:Kruci%C4%9Do_de_stratoj_Respubliko_kaj_Or%C4%9Donikidze_(Tjumeno).webm",
)
# Kept for what it shows of the detector's limits, not played by default: of
# several hundred people seen from far above it boxes a handful (see below).
SHIBUYA = SampleVideo(
    key="shibuya-crossing",
    name="Shibuya Crossing, Tokyo",
    description="The scramble crossing from above: a few hundred people at once, of whom the detector boxes a handful. 59 seconds, 1080p.",
    filename="tokyo-shibuya-crossing.webm",
    urls=("https://upload.wikimedia.org/wikipedia/commons/transcoded/5/53/Shibuya_Crossing%2C_Tokyo%2C_Japan_%28video%29.webm/"
          "Shibuya_Crossing%2C_Tokyo%2C_Japan_%28video%29.webm.1080p.vp9.webm",),
    size_bytes=37_432_911,
    sha256="93889a246ca9892f80f2587d681684e22a3fb9c5aee3bf42bb277994395deea7",
    licence="CC BY-SA 4.0",
    credit="Basile Morin, via Wikimedia Commons",
    source_page="https://commons.wikimedia.org/wiki/File:Shibuya_Crossing,_Tokyo,_Japan_(video).webm",
)
INTEL_STREET = SampleVideo(
    key="intel-street",
    name="Street corner (Intel sample)",
    description="Intel's own OpenVINO sample clip: people, bicycles and cars on a quiet street. 54 seconds, 768x432.",
    filename="intel-person-bicycle-car.mp4",
    # A commit, not a branch: the file at this address cannot change.
    urls=("https://github.com/intel-iot-devkit/sample-videos/raw/57978890822836f2b4743852f04f62fc511757e4/"
          "person-bicycle-car-detection.mp4",),
    size_bytes=6_031_199,
    sha256="452b11b7e0efbd019f1d9570d0c790e90416ad4ad29eec6003872d08443140ef",
    licence="CC BY 4.0",
    credit="Intel Corporation, intel-iot-devkit/sample-videos",
    source_page="https://github.com/intel-iot-devkit/sample-videos",
)

# Not streets: the same detector counting other things. Chosen by the same
# measurement from four that were tried (2026-10-09): bottles a median of 3
# boxes on screen and nothing mistaken; cattle a median of 8, with the rider,
# the horses and the dog named too. Fruit on a belt (empty 60% of the time,
# each item counted several times) and sheep on a slope (half of them called
# cows) were not kept.
CAPPING_LINE = SampleVideo(
    key="bottle-capping-line",
    name="Bottle capping line",
    description="Glass bottles moving through the capping machine of a distillery, a few in view at a time. 21 seconds, 1080p.",
    filename="bottle-capping-line.webm",
    urls=("https://upload.wikimedia.org/wikipedia/commons/a/ae/Capping_machine_in_action.webm",),
    size_bytes=7_081_982,
    sha256="eb8fc5e99ef515dfbceef74e59d533b696a4d735607c32295519a567b9ab2088",
    licence="CC BY 3.0",
    credit="Work With Sounds / La Fonderie, via Wikimedia Commons",
    source_page="https://commons.wikimedia.org/wiki/File:Capping_machine_in_action.webm",
    counts="line",
)
CATTLE_DRIVE = SampleVideo(
    key="cattle-drive",
    name="Cattle on the road",
    description="A herd driven along a gravel road past the camera, with a rider and a dog. 28 seconds, 720p.",
    filename="cattle-drive.webm",
    urls=("https://upload.wikimedia.org/wikipedia/commons/1/1a/Moving_cows_to_the_summer_range_%2842877666722%29.webm",),
    size_bytes=20_678_398,
    sha256="acc17d3aa29d3511f9c2ba47604e6f14bcbf882ddb0e62ad37054f75fa8c8c1f",
    licence="Public domain",
    credit="Bureau of Land Management Oregon and Washington, via Wikimedia Commons",
    source_page="https://commons.wikimedia.org/wiki/File:Moving_cows_to_the_summer_range_(42877666722).webm",
    counts="herd",
)

# In the order they are offered. The first two are what the city monitor
# opens on and what the Auto Demo plays, chosen by measurement (2026-10-09,
# the brick's own detector and tracker over one pass of each clip):
#
#                boxes on screen   frames with none   crossing the picture
#   Toronto        median 11            0%               122 a minute
#   Tyumen         median  9            0%                80 a minute
#   (Ljubljana)    median  7            1%                32 a minute   <- looked at, not kept
#   Shibuya        median  3           28%                27 a minute
#   Intel sample   median  0           58%                 --
VIDEOS: tuple[SampleVideo, ...] = (TORONTO, TYUMEN, SHIBUYA, INTEL_STREET, CAPPING_LINE, CATTLE_DRIVE)
BY_KEY = {video.key: video for video in VIDEOS}


class VideoUnavailable(RuntimeError):
    """A sample video could not be fetched, or what came is not the file."""


def present(video: SampleVideo) -> bool:
    """On disk and whole. The size, not the checksum: this is asked at every
    look of a dialog, and the checksum was checked when the file arrived."""
    try:
        return video.path.stat().st_size == video.size_bytes
    except OSError:
        return False


def all_present() -> bool:
    return all(present(video) for video in VIDEOS)


def total_bytes() -> int:
    return sum(video.size_bytes for video in VIDEOS)


def bytes_on_disk() -> int:
    """What is there, a download in flight included: what makes a progress
    bar move."""
    if not VIDEO_DIR.is_dir():
        return 0
    names = {video.filename for video in VIDEOS} | {video.filename + ".part" for video in VIDEOS}
    return sum(path.stat().st_size for path in VIDEO_DIR.iterdir() if path.name in names and path.is_file())


def for_path(path: str | Path) -> SampleVideo | None:
    """The sample video that lives at `path`, if it is one of them."""
    try:
        wanted = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return None
    return next((video for video in VIDEOS if video.path.resolve() == wanted), None)


def _open(url: str, start: int):
    headers = {"User-Agent": _AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def _fetch(video: SampleVideo, url: str, on_progress: Callable[[int], None] | None, stopped: Callable[[], bool]) -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    part = video.path.with_name(video.filename + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have >= video.size_bytes:  # more than the whole file is not a head start
        part.unlink()
        have = 0
    with _open(url, have) as response:
        # 206 is the rest of the file; 200 is all of it again, from a server
        # that does not resume.
        resumed = have > 0 and getattr(response, "status", 200) == 206
        if not resumed:
            have = 0
        with part.open("ab" if resumed else "wb") as out:
            while True:
                if stopped():
                    raise VideoUnavailable(f"{video.name}: stopped; what was fetched is kept and the rest resumes next time.")
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                have += len(chunk)
                if on_progress:
                    on_progress(have)
    digest = hashlib.sha256()
    with part.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if have != video.size_bytes or digest.hexdigest() != video.sha256:
        part.unlink(missing_ok=True)
        raise VideoUnavailable(
            f"{video.name}: what {url.split('/')[2]} sent is not the file that was chosen "
            f"({have:,} bytes for {video.size_bytes:,}, or another checksum). It has changed at the source."
        )
    part.replace(video.path)


def download(
    video: SampleVideo,
    on_progress: Callable[[int], None] | None = None,
    stopped: Callable[[], bool] = lambda: False,
) -> Path:
    """Fetch one video unless it is already there. Each address is tried in
    turn; a download cut short resumes where it stopped."""
    if present(video):
        return video.path
    problems = []
    for url in video.urls:
        try:
            _fetch(video, url, on_progress, stopped)
            return video.path
        except VideoUnavailable as exc:
            if stopped():
                raise
            problems.append(str(exc))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            problems.append(f"{video.name}: {url.split('/')[2]} could not be reached ({exc}).")
    raise VideoUnavailable(" ".join(problems) or f"{video.name}: nowhere to fetch it from.")


def download_all(on_progress: Callable[[int], None] | None = None) -> None:
    """Every video still missing. `on_progress` gets the bytes fetched by
    this call, all videos together."""
    done = 0
    for video in VIDEOS:
        if present(video):
            continue
        download(video, (lambda position, base=done: on_progress(base + position)) if on_progress else None)
        done += video.size_bytes
