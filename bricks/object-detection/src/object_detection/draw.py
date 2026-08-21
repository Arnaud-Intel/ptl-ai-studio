"""Draws detection boxes onto a frame. Kept separate from the detectors so
both the CLI (--show) and the launcher (video stream) share one look."""
from __future__ import annotations

import cv2
import numpy as np

from .types import Detection

_BOX_COLOR = (181, 104, 0)  # BGR: Intel-blue-ish, matches the studio's accent
_TEXT_COLOR = (255, 255, 255)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, 2)

        label = f"{det.label} {det.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 6, y1), _BOX_COLOR, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA,
        )
    return annotated
