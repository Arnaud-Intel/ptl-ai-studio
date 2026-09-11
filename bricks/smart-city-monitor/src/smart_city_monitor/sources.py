"""Where a feed's frames come from.

A feed is one of four things, and the difference matters to how it is
read, not to anything downstream -- so it is resolved here and the
pipeline just asks for frames:

- a **local video file**, paced to its own frame rate and looped, so a
  short clip can stand in for a continuous camera;
- a **live stream URL** (RTSP from a camera on your network, HTTP MJPEG,
  an HLS `.m3u8`), read as fast as it arrives and reopened if it drops;
- a **clip URL** (an .mp4 over HTTP that is replaced in place, like a
  TfL JamCam), played once per revision rather than replayed;
- a **YouTube live URL**, which is a web page rather than a stream, so it
  is resolved to its underlying HLS URL first.

Note what does and doesn't leave the machine: an internet feed obviously
arrives over the network, but the detection still runs entirely on local
silicon. Nothing about the video is sent anywhere.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import PureWindowsPath
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


# A finite media file served over HTTP, as opposed to a stream playlist. TfL's
# JamCams are the case in point: an .mp4 per camera, overwritten in place.
_CLIP_EXTENSIONS = (".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi")


def is_clip_url(source: str) -> bool:
    """An http(s) URL to a whole video file -- which ends, and is replaced
    rather than streamed -- not a live stream and not a YouTube page."""
    parsed = urlparse(source)
    return (
        parsed.scheme in ("http", "https")
        and not is_youtube(source)
        and parsed.path.lower().endswith(_CLIP_EXTENSIONS)
    )


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
    # PureWindowsPath, not Path: it treats both "\\" and "/" as separators
    # on every platform, so a Windows path typed into the launcher is split
    # correctly even when the launcher itself is running on Linux (which a
    # plain PosixPath would leave whole).
    return PureWindowsPath(source).name


# Opt-in: a Netscape-format cookies.txt exported from a signed-in browser, for
# when YouTube refuses the network with a bot check. Read from the environment
# rather than from anything the UI can see or send, so a login never passes
# through the launcher's API. Unset -> anonymous, exactly as before.
#
# Why a file and not yt-dlp's cookies-from-browser: on Windows, Chrome and Edge
# now encrypt cookies with an app-bound key yt-dlp cannot decrypt (its cookie
# code handles Chromium's v10/v11 formats only), and a running browser locks
# its cookie database besides. An extension that exports from inside the
# browser sidesteps both.
YOUTUBE_COOKIES_ENV = "PTL_YOUTUBE_COOKIES"


def _cookies_path() -> str:
    return os.environ.get(YOUTUBE_COOKIES_ENV, "").strip()


def _cookies_hint() -> str:
    """The last sentence of a bot-check message: what a signed-in session
    could do, or -- if one is already configured -- why it didn't."""
    if _cookies_path():
        return (
            f" The signed-in session in {YOUTUBE_COOKIES_ENV} didn't get past it either -- "
            "its cookies have probably expired; export them again (see the smart-city README)."
        )
    return (
        f" With a signed-in YouTube account you can set {YOUTUBE_COOKIES_ENV} to a "
        "cookies.txt exported from it (see the smart-city README)."
    )


def _ytdlp_options() -> dict:
    """Options for resolving a YouTube page: quiet, uncoloured, and with the
    operator's exported cookies if -- and only if -- they opted in."""
    # no_color: yt-dlp colours its errors for a terminal, and those escape
    # codes were landing verbatim in the UI's status line.
    options = {"quiet": True, "no_warnings": True, "skip_download": True, "no_color": True}
    cookies = _cookies_path()
    if cookies:
        if not os.path.isfile(cookies):
            raise RuntimeError(
                f"{YOUTUBE_COOKIES_ENV} points at a file that doesn't exist ({cookies}). "
                "Export a cookies.txt from a signed-in browser to that path, or unset it."
            )
        options["cookiefile"] = cookies
    return options


# YouTube's anti-bot wall, as yt-dlp words it. On 2026-09-11 it hit every
# curated camera at once from this network: the newest yt-dlp didn't get past
# it, and neither did any of its alternative player clients -- they either met
# the same wall or came back with no video at all. So the usual "upgrade
# yt-dlp" advice is wrong for this failure, and repeating it wastes the time
# of whoever is standing in front of the demo.
_BOT_CHECK_MARKERS = ("not a bot", "Sign in to confirm")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _explain_youtube_failure(url: str, message: str) -> str:
    """An operator-facing reason for a YouTube page that wouldn't resolve."""
    message = _ANSI.sub("", message).strip()  # belt and braces with no_color
    if any(marker in message for marker in _BOT_CHECK_MARKERS):
        return (
            f"YouTube is asking this network to sign in to prove it isn't a bot, so {url} "
            "can't be opened right now. Updating yt-dlp won't help -- YouTube is blocking "
            "the connection, the parser isn't out of date. Use a direct camera stream "
            "(RTSP, or an HLS .m3u8 URL) or a local video file instead." + _cookies_hint()
        )
    return (
        f"Couldn't read the YouTube page for {url}: {message}. If this used to work, "
        "YouTube has probably changed something -- try `uv sync --upgrade-package yt-dlp`."
    )


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

    options = _ytdlp_options()
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise RuntimeError(_explain_youtube_failure(url, str(exc))) from exc
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

    if is_clip_url(source):
        # Played once per revision -- replaying an unchanged clip would count
        # the same objects again every pass. See video.stream_refreshing_clip.
        yield from video.stream_refreshing_clip(source, stop_event=stop_event)
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
