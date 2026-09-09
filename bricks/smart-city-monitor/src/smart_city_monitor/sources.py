"""Where a feed's frames come from.

A feed is one of three things, and the difference matters to how it is
read, not to anything downstream -- so it is resolved here and the
pipeline just asks for frames:

- a **local video file**, paced to its own frame rate and looped, so a
  short clip can stand in for a continuous camera;
- a **live stream URL** (RTSP from a camera on your network, HTTP MJPEG,
  an HLS `.m3u8`), read as fast as it arrives and reopened if it drops;
- a **YouTube live URL**, which is a web page rather than a stream, so it
  is resolved to its underlying HLS URL first.

Note what does and doesn't leave the machine: an internet feed obviously
arrives over the network, but the detection still runs entirely on local
silicon. Nothing about the video is sent anywhere.
"""
from __future__ import annotations

import threading
from pathlib import Path
from urllib.parse import urlparse

from pantherlake_ai_core import video

from . import samples

_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}

# Detection resizes its input anyway, so a bigger stream costs bandwidth
# and decode time for no extra accuracy. 720p is plenty to count people
# and cars at street scale.
MAX_STREAM_HEIGHT = 720

# A resolved YouTube CDN link is time-limited, so a long run re-resolves
# rather than retrying a URL that has since expired.
_REOPEN_BACKOFF_SECONDS = 2.0


def is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https", "rtsp", "rtmp", "udp", "tcp")


def is_youtube(source: str) -> bool:
    return is_url(source) and urlparse(source).hostname in _YOUTUBE_HOSTS


def display_name(source: str) -> str:
    """A short label for a feed, for the viewer's picker and the per-feed
    counts.

    One of the curated live cameras gets its real name; any other stream
    gets its host, since the last path segment of a URL is usually a
    query string or an opaque id; a file gets its filename.
    """
    curated = samples.label_for(source)
    if curated:
        return curated
    if is_url(source):
        return urlparse(source).hostname or source
    return Path(source).name


def _extract_formats(url: str) -> list[dict]:
    """Ask yt-dlp what streams sit behind a YouTube live page.

    Isolated from the choosing below so the choice can be tested without
    a network round trip -- and because this is the part that breaks when
    YouTube changes its page, so the error belongs here.
    """
    try:
        import yt_dlp
    except ImportError:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "Reading a YouTube live stream needs yt-dlp. Install this brick's "
            "dependencies (`uv sync`), or use a direct stream URL instead."
        ) from None

    options = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise RuntimeError(
            f"Couldn't read the YouTube page for {url}: {exc}. If this used to work, "
            "YouTube has probably changed something -- try `uv sync --upgrade-package yt-dlp`."
        ) from exc
    return info.get("formats", []) if info else []


def resolve_youtube_stream(url: str, *, max_height: int = MAX_STREAM_HEIGHT) -> str:
    """The direct HLS URL behind a YouTube live page."""
    streams = [
        f
        for f in _extract_formats(url)
        if f.get("protocol") in ("m3u8", "m3u8_native") and f.get("vcodec") != "none" and f.get("url")
    ]
    if not streams:
        raise RuntimeError(f"No playable video stream found at {url} -- is it actually live?")

    # The largest that fits the cap, not the first under it: the list is
    # not sorted, and a 360p stream when 720p is there is a worse demo
    # for free.
    capped = [f for f in streams if (f.get("height") or 0) <= max_height]
    best = max(capped or streams, key=lambda f: (f.get("height") or 0))
    return best["url"]


def open_frames(source: str, *, loop: bool = True, stop_event: threading.Event | None = None):
    """Yield BGR frames from whatever kind of source this is."""
    if not is_url(source):
        yield from video.stream_video_file_frames(source, loop=loop, stop_event=stop_event)
        return

    if not is_youtube(source):
        yield from video.stream_live_frames(source, stop_event=stop_event)
        return

    # Reconnect by re-resolving: the CDN link expires, the page link doesn't.
    # A stream that opened and later dropped comes back here; one that never
    # opened raises out of stream_live_frames and fails the feed properly.
    while stop_event is None or not stop_event.is_set():
        yield from video.stream_live_frames(
            resolve_youtube_stream(source), stop_event=stop_event, reconnect=False
        )
        if video.sleep_unless_stopped(stop_event, _REOPEN_BACKOFF_SECONDS):
            return
