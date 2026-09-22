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
import shutil
import threading
import time
from pathlib import PureWindowsPath
from urllib.parse import urlparse

from pantherlake_ai_core import video

from . import samples
from . import youtube

_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}

# Detection resizes its input anyway, so a bigger stream costs bandwidth
# and decode time for no extra accuracy. 720p is plenty to count people
# and cars at street scale.
MAX_STREAM_HEIGHT = 720

# A resolved YouTube CDN link is time-limited, so a long run re-resolves
# rather than retrying a URL that has since expired.
_REOPEN_BACKOFF_SECONDS = 5.0
_MAX_FAILURES = 5
_STABLE_SECONDS = 60.0


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
            "cookies may be expired, or the account/network may be restricted. Re-exporting is not a guaranteed fix."
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
    options = {"quiet": True, "no_warnings": False, "skip_download": True, "no_color": True,
               "noplaylist": True, "socket_timeout": 10, "retries": 0, "extractor_retries": 0}
    if shutil.which('deno'):
        options['js_runtimes'] = {'deno': {}}
    elif shutil.which('node'):
        options['js_runtimes'] = {'node': {}}
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


def _explain_youtube_failure(url: str, message: str) -> str:
    """An operator-facing reason for a YouTube page that wouldn't resolve."""
    error = youtube.classify(message)
    return str(error) + (_cookies_hint() if error.blocked else '')


def _extract_formats(url: str, stop_event=None) -> list[dict]:
    """Ask yt-dlp what streams sit behind a YouTube live page.

    Isolated from the choosing below so the choice can be tested without
    a network round trip -- and because this is the part that breaks when
    YouTube changes its page, so the error belongs here.
    """
    try:
        return youtube.resolver.resolve(url, stop_event)
    except youtube.SourceError as exc:
        if exc.blocked:
            raise youtube.SourceError(str(exc) + _cookies_hint(), blocked=True) from None
        raise


def resolve_youtube_stream(url: str, *, max_height: int = MAX_STREAM_HEIGHT, stop_event=None) -> str:
    """The direct HLS URL behind a YouTube live page."""
    streams = [
        f
        for f in (_extract_formats(url) if stop_event is None else _extract_formats(url, stop_event))
        if f.get("protocol") in ("m3u8", "m3u8_native") and f.get("vcodec") != "none" and f.get("url")
    ]
    if not streams:
        raise youtube.SourceError('No playable HLS video stream -- is it actually live? Check Deno and yt-dlp[default], or choose Other > London, two chips.')

    # The largest that fits the cap, not the first under it: the list is
    # not sorted, and a 360p stream when 720p is there is a worse demo
    # for free.
    capped = [f for f in streams if (f.get("height") or 0) <= max_height]
    best = (max(capped, key=lambda f: (f.get('height') or 0)) if capped
            else min(streams, key=lambda f: (f.get('height') or 0)))
    return best["url"]


def open_frames(source: str, *, loop: bool = True, stop_event: threading.Event | None = None, on_status=None):
    """Yield BGR frames from whatever kind of source this is."""
    if not is_url(source):
        yield from video.stream_video_file_frames(source, loop=loop, stop_event=stop_event)
        return

    if is_clip_url(source):
        # Played once per revision -- replaying an unchanged clip would count
        # the same objects again every pass. See video.stream_refreshing_clip.
        yield from video.stream_refreshing_clip(source, stop_event=stop_event, on_status=on_status)
        return

    # Both direct streams and YouTube get a finite recovery budget. A brief
    # successful open does not reset it: only a full minute of frames does.
    failures = 0
    status = on_status or (lambda phase, message: None)
    while stop_event is None or not stop_event.is_set():
        started = None
        try:
            status('loading', 'Connecting to camera...' if not is_youtube(source) else 'Waiting for paced YouTube connection...')
            url = resolve_youtube_stream(source, stop_event=stop_event) if is_youtube(source) else source
            for frame in video.stream_live_frames(url, stop_event=stop_event, reconnect=False):
                if started is None:
                    started = time.monotonic()
                yield frame
            if stop_event is not None and stop_event.is_set():
                return
            reason = 'Camera stopped sending frames.'
        except youtube.Cancelled:
            return
        except youtube.SourceError as exc:
            if not exc.retryable:
                raise
            reason = str(exc)
        except (RuntimeError, OSError) as exc:
            reason = youtube.safe_message(exc)
        if started is not None and time.monotonic() - started >= _STABLE_SECONDS:
            failures = 0
        failures += 1
        if failures >= _MAX_FAILURES:
            raise RuntimeError(f'Camera failed after {_MAX_FAILURES} unstable connections. {reason} Choose another camera or a local video, then restart.')
        if is_youtube(source):
            youtube.resolver.invalidate(source, stop_event)
        delay = min(60, _REOPEN_BACKOFF_SECONDS * 2 ** (failures - 1))
        status('loading', f'Reconnecting in {delay:g}s ({failures}/{_MAX_FAILURES - 1}). {reason}')
        if video.sleep_unless_stopped(stop_event, delay):
            return
