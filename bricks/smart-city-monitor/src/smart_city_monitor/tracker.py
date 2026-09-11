"""A small IoU-based greedy multi-object tracker.

Not a claim to solve real multi-object-tracking/re-identification (no
appearance embedding, no motion model) -- just enough to stop massively
over-counting an object that sits in frame across many detections, by
giving it one persistent id instead of a new one every frame. One
instance per video feed: track identity must never cross feeds, since
each feed is a physically different camera/video.

Known, disclosed limitation (see the brick's README): an object that
leaves the frame and re-enters (or is fully occluded for longer than
`_MAX_MISS_SECONDS`) gets a new id and is counted again.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from object_detection.types import Detection

from .types import TrackedDetection

_IOU_MATCH_THRESHOLD = 0.3
_MAX_MISS_SECONDS = 1.0  # a track not matched for this long is dropped


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    label: str
    box: tuple[int, int, int, int]
    last_seen: float


@dataclass
class Tracker:
    _next_id: int = field(default=1, init=False)
    _tracks: dict[int, _Track] = field(default_factory=dict, init=False)

    def update(self, detections: list[Detection], now: float) -> list[TrackedDetection]:
        """`now` is the caller's own clock (`time.monotonic()`), passed in
        rather than read here so the tracker's behavior is deterministic
        and testable."""
        # Prune stale tracks *before* matching -- otherwise a track that
        # should already be expired can still win this frame's IoU match
        # (e.g. a new object appearing in the same spot an old one
        # vacated over a second ago would wrongly inherit its id).
        stale = [tid for tid, t in self._tracks.items() if now - t.last_seen > _MAX_MISS_SECONDS]
        for tid in stale:
            del self._tracks[tid]

        by_label: dict[str, list[int]] = {}
        for track_id, track in self._tracks.items():
            by_label.setdefault(track.label, []).append(track_id)

        results: list[TrackedDetection] = []
        # Strongest pair first, across the entire frame. A lower-confidence
        # overlap must not steal a track before its exact match is considered.
        pairs = []
        for index, det in enumerate(detections):
            for track_id in by_label.get(det.label, []):
                score = _iou(self._tracks[track_id].box, det.box)
                if score >= _IOU_MATCH_THRESHOLD:
                    pairs.append((-score, track_id, index))
        matches: dict[int, int] = {}
        matched_track_ids: set[int] = set()
        for _, track_id, index in sorted(pairs):
            if index not in matches and track_id not in matched_track_ids:
                matches[index] = track_id
                matched_track_ids.add(track_id)
        for index, det in enumerate(detections):
            best_id = matches.get(index)
            if best_id is not None:
                track = self._tracks[best_id]
                track.box = det.box
                track.last_seen = now
                matched_track_ids.add(best_id)
                results.append(
                    TrackedDetection(
                        track_id=best_id, label=det.label, confidence=det.confidence, box=det.box, is_new=False
                    )
                )
            else:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = _Track(track_id=new_id, label=det.label, box=det.box, last_seen=now)
                matched_track_ids.add(new_id)
                results.append(
                    TrackedDetection(
                        track_id=new_id, label=det.label, confidence=det.confidence, box=det.box, is_new=True
                    )
                )

        return results
