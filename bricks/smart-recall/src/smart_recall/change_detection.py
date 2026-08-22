"""Cheap "did the screen actually change" check, so a static desktop
doesn't get OCR'd and indexed over and over on every capture tick.
"""
from __future__ import annotations

import cv2
import numpy as np

_SAMPLE_SIZE = (160, 90)


def frame_changed(previous: np.ndarray | None, current: np.ndarray, threshold: float = 0.02) -> bool:
    """Downsamples both frames to grayscale thumbnails and compares mean
    absolute pixel difference, normalized to [0, 1]. `previous=None`
    (first capture) always counts as changed."""
    if previous is None:
        return True
    prev_small = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), _SAMPLE_SIZE)
    curr_small = cv2.resize(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), _SAMPLE_SIZE)
    diff = np.abs(prev_small.astype(np.int16) - curr_small.astype(np.int16)).mean() / 255.0
    return diff > threshold
