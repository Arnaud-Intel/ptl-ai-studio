"""Draws detection boxes onto a frame. Kept separate from the detectors so
both the CLI (--show) and the launcher (video stream) share one look."""
from __future__ import annotations

import cv2
import numpy as np

from .types import Detection

_BOX_COLOR = (181, 104, 0)  # BGR: Intel-blue-ish, matches the studio's accent
_TEXT_COLOR = (255, 255, 255)
# The frame height the sizes below were chosen at: a webcam's 480 lines. A
# taller picture gets them in proportion -- a two-pixel box on a 1080-line
# frame is a hairline, and its label unreadable, once a page shows that
# frame 600 wide (the city monitor's drawing does the same).
_DRAWN_FOR_HEIGHT = 540


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()
    scale = max(1.0, frame.shape[0] / _DRAWN_FOR_HEIGHT)
    line, font, stroke, pad = max(2, round(2 * scale)), 0.5 * scale, max(1, round(scale)), round(3 * scale)
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, line)

        label = f"{det.label} {det.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, stroke)
        label_y1 = max(0, y1 - text_h - baseline - pad - 1)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 2 * pad, y1), _BOX_COLOR, -1)
        cv2.putText(
            annotated, label, (x1 + pad, y1 - baseline - pad + 1),
            cv2.FONT_HERSHEY_SIMPLEX, font, _TEXT_COLOR, stroke, cv2.LINE_AA,
        )
    return annotated
