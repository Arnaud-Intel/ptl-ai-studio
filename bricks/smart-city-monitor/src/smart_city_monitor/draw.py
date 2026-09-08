"""Draws tracked-detection boxes onto a frame. A brick-local copy of
object_detection.draw's shape, not a reuse of it, since we also want to
show each box's track id -- visual proof the tracker is holding a stable
identity across frames, not just relabeling every detection -- and flag
the exact frame a new track (a "count" event) was created.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import TrackedDetection

_BOX_COLOR = (181, 104, 0)  # BGR: Intel-blue-ish, matches object-detection's and the studio's accent
_NEW_TRACK_COLOR = (74, 222, 128)  # BGR: matches the launcher's --success green, for a one-frame "counted" flash
_TEXT_COLOR = (255, 255, 255)


def draw_tracks(frame: np.ndarray, tracks: list[TrackedDetection]) -> np.ndarray:
    annotated = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.box
        color = _NEW_TRACK_COLOR if track.is_new else _BOX_COLOR
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{track.label} #{track.track_id} {track.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA,
        )
    return annotated
