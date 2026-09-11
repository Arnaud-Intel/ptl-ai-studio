"""Video frame capture from a webcam, the screen, or a local video file.

Mirrors audio.py's shape: list what's available, then stream BGR uint8
frames (OpenCV's native order) until a stop_event is set.
"""
from __future__ import annotations

import os
import platform
import threading
import time

import numpy as np

_IS_WINDOWS = platform.system() == "Windows"
_DEFAULT_FILE_FPS = 25.0  # fallback when a file doesn't report CAP_PROP_FPS


_CAMERA_CACHE_SECONDS = 30.0
_camera_cache: tuple[float, list[int]] | None = None
_camera_cache_lock = threading.Lock()


def list_cameras(max_index: int = 4) -> list[int]:
    """Probe camera indices 0..max_index-1, return the ones that open.

    Deliberately small: opening/closing a camera device is slow (~0.5s
    each on Windows/DirectShow), and most machines have at most one or two.
    The result is cached for a short while, since the launcher asks for it
    every time a panel opens and cameras don't come and go that often.
    """
    global _camera_cache
    with _camera_cache_lock:
        if _camera_cache is not None and time.monotonic() - _camera_cache[0] < _CAMERA_CACHE_SECONDS:
            return list(_camera_cache[1])

    import cv2

    # Probing indices that don't exist makes OpenCV print a warning each
    # time; that's expected here, not something to show the user.
    logging_api = getattr(cv2, "utils", None) and getattr(cv2.utils, "logging", None)
    previous_level = None
    if logging_api is not None:
        try:
            previous_level = logging_api.getLogLevel()
            logging_api.setLogLevel(logging_api.LOG_LEVEL_SILENT)
        except AttributeError:
            logging_api = None

    backend = cv2.CAP_DSHOW if _IS_WINDOWS else cv2.CAP_ANY
    available = []
    try:
        for index in range(max_index):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                available.append(index)
            cap.release()
    finally:
        if logging_api is not None and previous_level is not None:
            logging_api.setLogLevel(previous_level)

    with _camera_cache_lock:
        _camera_cache = (time.monotonic(), list(available))
    return available


def list_screens() -> list[dict]:
    """Return each capturable screen/monitor as {index, width, height}.

    Index 0 (all monitors combined) is skipped -- callers want one concrete
    screen, matching what a person would pick from a dropdown.
    """
    import mss

    with mss.mss() as sct:
        return [
            {"index": i, "width": m["width"], "height": m["height"]}
            for i, m in enumerate(sct.monitors)
            if i > 0
        ]


def capture_camera_frame(index: int = 0) -> np.ndarray:
    """Grab a single BGR uint8 frame from a webcam and release it -- for a
    one-shot capture (e.g. OCR), not a continuous stream."""
    import cv2

    backend = cv2.CAP_DSHOW if _IS_WINDOWS else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}.")
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read a frame from camera {index}.")
        return frame
    finally:
        cap.release()


def capture_screen_frame(monitor: int = 1) -> np.ndarray:
    """Grab a single BGR uint8 frame of one screen -- for a one-shot
    capture (e.g. OCR), not a continuous stream."""
    import mss

    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor <= 0 or monitor >= len(monitors):
            raise RuntimeError(f"No screen at index {monitor}. Available: 1..{len(monitors) - 1}")
        shot = sct.grab(monitors[monitor])
        return np.asarray(shot)[:, :, :3]


def stream_camera_frames(index: int = 0, stop_event: threading.Event | None = None):
    """Yield BGR uint8 frames from a webcam until stop_event is set."""
    import cv2

    backend = cv2.CAP_DSHOW if _IS_WINDOWS else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}.")
    try:
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def stream_screen_frames(monitor: int = 1, stop_event: threading.Event | None = None):
    """Yield BGR uint8 frames of one screen until stop_event is set."""
    import mss

    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor <= 0 or monitor >= len(monitors):
            raise RuntimeError(f"No screen at index {monitor}. Available: 1..{len(monitors) - 1}")
        region = monitors[monitor]
        # Cap at ~30 fps: mss can grab far faster than any consumer here
        # needs, and an unpaced loop pins a core between inferences.
        frame_interval = 1.0 / 30.0
        next_frame_at = time.monotonic()
        while stop_event is None or not stop_event.is_set():
            now = time.monotonic()
            if next_frame_at > now:
                time.sleep(next_frame_at - now)
            next_frame_at = max(next_frame_at, now) + frame_interval
            shot = sct.grab(region)
            # mss gives BGRA; drop alpha to match the BGR frames camera capture yields.
            yield np.asarray(shot)[:, :, :3]


def stream_video_file_frames(path: str, *, loop: bool = True, stop_event: threading.Event | None = None):
    """Yield BGR uint8 frames from a local video file, paced to the file's
    own frame rate so playback simulates a real-time feed -- a demo
    reading counts/rates off this stream (e.g. "N per minute") needs one
    processed second to correspond to one second of footage, not to
    however fast the CPU/GPU can chew through frames.

    If `loop`, restarts from frame 0 at EOF so a short clip can stand in
    for a continuous camera feed; otherwise the generator ends at EOF.
    """
    import cv2

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Not a file: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / fps if fps and fps > 0 else 1.0 / _DEFAULT_FILE_FPS
        next_frame_at = time.monotonic()
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if not loop:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                next_frame_at = time.monotonic()
                continue
            now = time.monotonic()
            if next_frame_at > now:
                time.sleep(next_frame_at - now)
            next_frame_at = max(next_frame_at, now) + frame_interval
            yield frame
    finally:
        cap.release()


# A live stream that drops mid-read is normal, not fatal: wifi blips, the
# CDN rotates a segment, the publisher restarts. Reconnect rather than
# ending the feed, backing off a little so a genuinely dead URL doesn't
# spin.
_RECONNECT_BACKOFF_SECONDS = 2.0
_RECONNECT_ATTEMPTS = 5


def sleep_unless_stopped(stop_event: threading.Event | None, seconds: float) -> bool:
    """Sleep for `seconds`, interruptibly if there is an event to
    interrupt it. Returns True if we were asked to stop (so the caller can
    `return` straight away), False if the wait simply elapsed."""
    if stop_event is None:
        time.sleep(seconds)
        return False
    return stop_event.wait(seconds)


def stream_live_frames(
    url: str,
    *,
    stop_event: threading.Event | None = None,
    reconnect: bool = True,
):
    """Yield BGR uint8 frames from a live stream URL (RTSP, HTTP MJPEG,
    HLS `.m3u8`, anything else FFmpeg can open).

    Unlike `stream_video_file_frames` this does **not** pace the frames:
    a live stream already arrives in real time, so sleeping between reads
    would only build a growing delay behind the present moment. And there
    is nothing to loop -- at a drop it reopens the URL instead of seeking
    back to the start, which is meaningless for a feed with no beginning.
    """
    import cv2

    attempts = 0
    while stop_event is None or not stop_event.is_set():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            attempts += 1
            if not reconnect or attempts >= _RECONNECT_ATTEMPTS:
                raise RuntimeError(f"Could not open stream: {url}")
            if sleep_unless_stopped(stop_event, _RECONNECT_BACKOFF_SECONDS):
                return
            continue

        attempts = 0  # a successful open resets the budget
        try:
            while stop_event is None or not stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break  # dropped -- fall out and reopen
                yield frame
        finally:
            cap.release()

        if not reconnect:
            return
        if sleep_unless_stopped(stop_event, _RECONNECT_BACKOFF_SECONDS):
            return


# A clip URL the publisher overwrites in place -- e.g. a traffic authority's
# camera that posts a ten-second clip every few minutes. It is not a stream:
# it ends. Replaying it until the next one lands would count the same cars on
# every pass (measured on TfL's JamCams: one clip stays up ~5-7 minutes, so
# ~30 replays of the same ten seconds). So each revision plays exactly once,
# at its own frame rate, and then the reader waits for the resource to change.
_CLIP_POLL_SECONDS = 15.0
_CLIP_FALLBACK_WAIT = 300.0  # the server won't say whether it changed: wait a full refresh
_CLIP_HEAD_TIMEOUT = 10.0
_HTTP_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0) PantherLakeAIStudio"


def http_clip_version(url: str) -> str | None:
    """What identifies this revision of a clip -- ETag, else Last-Modified,
    else Content-Length, whichever the server offers. None if it won't say
    (or can't be reached), which the caller treats as "unknown", never as
    "changed"."""
    import urllib.request

    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _HTTP_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_CLIP_HEAD_TIMEOUT) as response:
            headers = response.headers
            return headers.get("ETag") or headers.get("Last-Modified") or headers.get("Content-Length")
    except Exception:
        return None


def _play_clip_once(url: str, stop_event: threading.Event | None = None):
    """One pass over a clip URL, paced to its frame rate the same way
    `stream_video_file_frames` paces a local file -- an HTTP clip decodes as
    fast as the CPU allows otherwise, and ten seconds of footage would
    flash past in one, taking "per minute" with it."""
    import cv2

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open clip: {url}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / fps if fps and fps > 0 else 1.0 / _DEFAULT_FILE_FPS
        next_frame_at = time.monotonic()
        while stop_event is None or not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                return
            now = time.monotonic()
            if next_frame_at > now:
                time.sleep(next_frame_at - now)
            next_frame_at = max(next_frame_at, now) + frame_interval
            yield frame
    finally:
        cap.release()


def stream_refreshing_clip(
    url: str,
    *,
    stop_event: threading.Event | None = None,
    poll_seconds: float = _CLIP_POLL_SECONDS,
    _version=http_clip_version,
    _play=_play_clip_once,
):
    """Yield frames from a clip URL that is replaced in place, playing each
    revision of it exactly once.

    Between revisions nothing is yielded: the picture holds its last frame
    and per-minute counts decay honestly, instead of climbing on a replay.
    The first play is allowed to fail loudly (a bad URL should fail the
    feed); a later one that fails is treated as a blip and retried at the
    next poll, since the clip was demonstrably there a few minutes ago.
    """
    played = False
    last_version: object = None  # what we last played; a sentinel if unknown
    unknown = object()
    while stop_event is None or not stop_event.is_set():
        current = _version(url)
        if played and current is not None and current == last_version:
            if sleep_unless_stopped(stop_event, poll_seconds):
                return
            continue
        if played and current is None:
            # Can't tell whether it changed. Replaying on a hunch is the
            # over-count this reader exists to prevent, so wait it out.
            if sleep_unless_stopped(stop_event, _CLIP_FALLBACK_WAIT):
                return
        try:
            yield from _play(url, stop_event)
        except RuntimeError:
            if not played:
                raise
            if sleep_unless_stopped(stop_event, poll_seconds):
                return
            continue
        played = True
        last_version = current if current is not None else unknown
