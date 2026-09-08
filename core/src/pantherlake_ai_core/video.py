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
