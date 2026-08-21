"""Video frame capture from a webcam or the screen.

Mirrors audio.py's shape: list what's available, then stream BGR uint8
frames (OpenCV's native order) until a stop_event is set.
"""
from __future__ import annotations

import platform
import threading

import numpy as np

_IS_WINDOWS = platform.system() == "Windows"


def list_cameras(max_index: int = 4) -> list[int]:
    """Probe camera indices 0..max_index-1, return the ones that open.

    Deliberately small: opening/closing a camera device is slow, and most
    machines have at most one or two.
    """
    import cv2

    backend = cv2.CAP_DSHOW if _IS_WINDOWS else cv2.CAP_ANY
    available = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            available.append(index)
        cap.release()
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
        while stop_event is None or not stop_event.is_set():
            shot = sct.grab(region)
            # mss gives BGRA; drop alpha to match the BGR frames camera capture yields.
            yield np.asarray(shot)[:, :, :3]
