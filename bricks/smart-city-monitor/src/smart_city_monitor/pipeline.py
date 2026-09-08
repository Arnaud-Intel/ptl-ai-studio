"""Multi-feed capture -> detect -> track -> count loop, shared by the CLI
and the launcher. Composes object-detection's detector directly (no new
model code here) and adds what object-detection doesn't have: a tracker
per feed so objects are counted once, and N feeds running concurrently,
each independently pinnable to its own compute device.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Callable

import numpy as np
from object_detection.engine_factory import create_detector
from pantherlake_ai_core import video
from pantherlake_ai_core.engine import Engine

from .tracker import Tracker
from .types import CountSnapshot, FeedSpec, TrackedDetection

# COCO labels relevant to a street/CCTV scene, present with matching
# spellings in both object-detection engines' vocabularies (DETR's
# COCO-91 and YOLO11n's COCO-80) -- mapped to the display name shown in
# counts. Detections for anything else are dropped before tracking, so
# neither the drawn boxes nor the counts are cluttered with e.g. "chair".
RELEVANT_LABELS: dict[str, str] = {
    "person": "Pedestrians",
    "bicycle": "Bicycles",
    "car": "Cars",
    "motorcycle": "Motorcycles",
    "bus": "Buses",
    "truck": "Trucks",
}


class FeedCounters:
    """One feed's per-label trailing-60-second counts and running totals."""

    def __init__(self) -> None:
        self._last_60s: dict[str, deque[float]] = defaultdict(deque)
        self._total: dict[str, int] = defaultdict(int)

    def record(self, tracks: list[TrackedDetection], now: float) -> None:
        for t in tracks:
            if not t.is_new:
                continue
            label = RELEVANT_LABELS[t.label]  # tracks are already filtered to RELEVANT_LABELS
            self._last_60s[label].append(now)
            self._total[label] += 1

    def snapshot(self, now: float) -> tuple[dict[str, int], dict[str, int]]:
        last_60s: dict[str, int] = {}
        for label, timestamps in self._last_60s.items():
            while timestamps and now - timestamps[0] > 60:
                timestamps.popleft()
            last_60s[label] = len(timestamps)
        return last_60s, dict(self._total)


class _SharedState:
    """Lock-guarded aggregation of every feed's latest counts into one
    combined snapshot -- read by on_counts after every processed frame,
    from whichever feed thread just processed one."""

    def __init__(self, feeds: list[FeedSpec]) -> None:
        self._lock = threading.Lock()
        self._active_feeds = [f.feed_id for f in feeds]
        self._per_feed_last_60s: dict[str, dict[str, int]] = {f.feed_id: {} for f in feeds}
        self._per_feed_total: dict[str, dict[str, int]] = {f.feed_id: {} for f in feeds}

    def update_and_snapshot(self, feed_id: str, last_60s: dict[str, int], total: dict[str, int]) -> CountSnapshot:
        with self._lock:
            self._per_feed_last_60s[feed_id] = last_60s
            self._per_feed_total[feed_id] = total

            combined_last_60s: dict[str, int] = defaultdict(int)
            combined_total: dict[str, int] = defaultdict(int)
            for fid in self._active_feeds:
                for label, count in self._per_feed_last_60s[fid].items():
                    combined_last_60s[label] += count
                for label, count in self._per_feed_total[fid].items():
                    combined_total[label] += count

            return CountSnapshot(
                combined_last_60s=dict(combined_last_60s),
                combined_total=dict(combined_total),
                per_feed_last_60s={fid: dict(v) for fid, v in self._per_feed_last_60s.items()},
                per_feed_total={fid: dict(v) for fid, v in self._per_feed_total.items()},
                active_feeds=list(self._active_feeds),
            )


def _group_by_device(feeds: list[FeedSpec]) -> dict[str, list[FeedSpec]]:
    groups: dict[str, list[FeedSpec]] = defaultdict(list)
    for feed in feeds:
        groups[feed.compute_device].append(feed)
    return dict(groups)


def run(
    *,
    feeds: list[FeedSpec],
    engine: Engine,
    model_path: str | None = None,
    loop: bool = True,
    on_ready: Callable[[str], None] | None = None,
    on_frame: Callable[[str, np.ndarray, list[TrackedDetection]], None],
    on_counts: Callable[[CountSnapshot], None],
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread until every feed's video ends (only
    possible with `loop=False`) or `stop_event` is set. Runs N feeds
    concurrently, one detector per distinct `compute_device` among them
    (shared by every feed pinned to that device, guarded by one lock, so
    a device's model loads exactly once) -- feeds on different devices
    therefore run, and load, genuinely in parallel.

    `on_ready(feed_id)` fires once per feed, as soon as that feed's own
    device has finished loading its detector -- the real loading->running
    boundary, independent of how long other devices take.
    """
    if not feeds:
        raise ValueError("No feeds given.")
    for feed in feeds:
        if not feed.name:
            feed.name = os.path.basename(feed.path)

    state = _SharedState(feeds)
    errors: dict[str, Exception] = {}
    errors_lock = threading.Lock()

    def record_error(key: str, exc: Exception) -> None:
        with errors_lock:
            errors.setdefault(key, exc)

    def device_worker(device: str, device_feeds: list[FeedSpec]) -> None:
        try:
            detector = create_detector(engine, device=device, model_path=model_path)
        except Exception as exc:
            record_error(device, exc)
            return
        detect_lock = threading.Lock()

        def feed_worker(feed: FeedSpec) -> None:
            tracker = Tracker()
            counters = FeedCounters()
            try:
                for frame in video.stream_video_file_frames(feed.path, loop=loop, stop_event=stop_event):
                    now = time.monotonic()
                    with detect_lock:
                        detections = detector.detect(frame)
                    relevant = [d for d in detections if d.label in RELEVANT_LABELS]
                    tracks = tracker.update(relevant, now)
                    counters.record(tracks, now)
                    on_frame(feed.feed_id, frame, tracks)
                    last_60s, total = counters.snapshot(now)
                    on_counts(state.update_and_snapshot(feed.feed_id, last_60s, total))
            except Exception as exc:
                record_error(feed.feed_id, exc)

        feed_threads = []
        for feed in device_feeds:
            if on_ready is not None:
                on_ready(feed.feed_id)
            t = threading.Thread(target=feed_worker, args=(feed,), daemon=True)
            feed_threads.append(t)
            t.start()
        for t in feed_threads:
            t.join()

    feeds_by_device = _group_by_device(feeds)
    device_threads = [
        threading.Thread(target=device_worker, args=(device, device_feeds), daemon=True)
        for device, device_feeds in feeds_by_device.items()
    ]
    for t in device_threads:
        t.start()
    for t in device_threads:
        t.join()

    if errors:
        # Threads don't propagate exceptions to the caller on their own --
        # surface at least the first one rather than silently going dark.
        key, exc = next(iter(errors.items()))
        raise RuntimeError(f"'{key}' failed: {exc}") from exc
