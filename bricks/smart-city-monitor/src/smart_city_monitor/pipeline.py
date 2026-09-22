"""Multi-feed capture -> detect -> track -> count loop, shared by the CLI
and the launcher. Composes object-detection's detector directly (no new
model code here) and adds what object-detection doesn't have: a tracker
per feed so objects are counted once, and N feeds running concurrently,
each independently pinnable to its own compute device.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable

import numpy as np
from object_detection.engine_factory import create_detector
from pantherlake_ai_core.engine import Engine

from . import sources
from .tracker import Tracker
from .types import CountSnapshot, FeedSpec, TrackedDetection

# COCO labels relevant to a street/CCTV scene, present with matching
# spellings in both object-detection engines' vocabularies (DETR's
# COCO-91 and YOLO11s's COCO-80) -- mapped to the display name shown in
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
    combined snapshot. A heartbeat ages counts even when frames stop."""

    def __init__(self, feeds: list[FeedSpec]) -> None:
        self._lock = threading.Lock()
        self._active_feeds = [f.feed_id for f in feeds]
        self._per_feed_last_60s: dict[str, dict[str, int]] = {f.feed_id: {} for f in feeds}
        self._per_feed_total: dict[str, dict[str, int]] = {f.feed_id: {} for f in feeds}
        self._counters = {f.feed_id: FeedCounters() for f in feeds}

    def record(self, feed_id, tracks, now):
        with self._lock:
            self._counters[feed_id].record(tracks, now)

    def finish(self, feed_id):
        with self._lock:
            if feed_id in self._active_feeds:
                self._active_feeds.remove(feed_id)

    def snapshot(self, now):
        with self._lock:
            for fid, counters in self._counters.items():
                self._per_feed_last_60s[fid], self._per_feed_total[fid] = counters.snapshot(now)
            combined_minute, combined_total = defaultdict(int), defaultdict(int)
            for fid in self._counters:
                for label, count in self._per_feed_last_60s[fid].items():
                    combined_minute[label] += count
                for label, count in self._per_feed_total[fid].items():
                    combined_total[label] += count
            return CountSnapshot(
                combined_last_60s=dict(combined_minute), combined_total=dict(combined_total),
                per_feed_last_60s={fid: dict(v) for fid, v in self._per_feed_last_60s.items()},
                per_feed_total={fid: dict(v) for fid, v in self._per_feed_total.items()},
                active_feeds=list(self._active_feeds),
            )

# What makes two feeds able to share one loaded detector. Device alone
# isn't enough now that a feed picks its own engine and model: two feeds on
# the NPU running different models need two detectors.
DetectorKey = tuple[Engine, str, str | None]


def _group_by_detector(feeds: list[FeedSpec], default_engine: Engine) -> dict[DetectorKey, list[FeedSpec]]:
    groups: dict[DetectorKey, list[FeedSpec]] = defaultdict(list)
    for feed in feeds:
        groups[(feed.engine or default_engine, feed.compute_device, feed.model_path)].append(feed)
    return dict(groups)


def run(
    *,
    feeds: list[FeedSpec],
    engine: Engine,
    model_path: str | None = None,
    loop: bool = True,
    on_ready: Callable[[str], None] | None = None,
    on_feed_error: Callable[[str, str], None] | None = None,
    on_feed_status: Callable[[str, str, str], None] | None = None,
    on_frame: Callable[[str, np.ndarray, list[TrackedDetection]], None],
    on_counts: Callable[[CountSnapshot], None],
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread until every feed's video ends (only
    possible with `loop=False`) or `stop_event` is set. Runs N feeds
    concurrently, one detector per distinct (engine, device, model) among
    them (shared by every feed asking for the same three, guarded by one
    lock, so each combination loads exactly once) -- feeds on different
    devices therefore run, and load, genuinely in parallel.

    `engine` and `model_path` are the run's defaults; a feed that names its
    own `engine`/`model_path` overrides them, so one run can mix backends.

    `on_ready(feed_id)` fires once per feed, as its first frame is
    processed -- i.e. once that feed's own device has finished loading its
    detector *and* the file opened, independent of how long other devices
    take. `on_feed_error(feed_id, message)` fires the moment a feed fails
    (its device couldn't load, its file couldn't be read, ...); the other
    feeds keep running. The call itself only raises if no feed ever got
    going at all.
    """
    if not feeds:
        raise ValueError("No feeds given.")
    for feed in feeds:
        if not feed.name:
            feed.name = sources.display_name(feed.path)

    state = _SharedState(feeds)
    errors: dict[str, Exception] = {}
    ready_feeds: set[str] = set()
    bookkeeping_lock = threading.Lock()

    def fail(feed_id: str, exc: Exception) -> None:
        state.finish(feed_id)
        with bookkeeping_lock:
            errors.setdefault(feed_id, exc)
        if on_feed_error is not None:
            on_feed_error(feed_id, str(exc))

    def device_worker(key: DetectorKey, device_feeds: list[FeedSpec]) -> None:
        feed_engine, device, feed_model = key
        try:
            detector = create_detector(feed_engine, device=device, model_path=feed_model or model_path)
        except Exception as exc:
            for feed in device_feeds:
                fail(feed.feed_id, exc)
            return
        detect_lock = threading.Lock()

        def feed_worker(feed: FeedSpec) -> None:
            tracker = Tracker()
            ready = False
            failed = False

            def source_status(phase, message):
                nonlocal ready
                if phase == 'loading':
                    ready = False
                if on_feed_status is not None:
                    on_feed_status(feed.feed_id, phase, message)

            try:
                for frame in sources.open_frames(feed.path, loop=loop, stop_event=stop_event, on_status=source_status):
                    if stop_event is not None and stop_event.is_set():
                        break
                    now = time.monotonic()
                    with detect_lock:
                        detections = detector.detect(frame)
                    if not ready:
                        ready = True
                        with bookkeeping_lock:
                            ready_feeds.add(feed.feed_id)
                        if on_ready is not None:
                            on_ready(feed.feed_id)
                    relevant = [d for d in detections if d.label in RELEVANT_LABELS]
                    tracks = tracker.update(relevant, now)
                    state.record(feed.feed_id, tracks, now)
                    on_frame(feed.feed_id, frame, tracks)
            except Exception as exc:
                failed = True
                fail(feed.feed_id, exc)
            finally:
                state.finish(feed.feed_id)
                if not failed and not (stop_event is not None and stop_event.is_set()) and on_feed_status is not None:
                    on_feed_status(feed.feed_id, 'idle', 'Video ended.')

        feed_threads = [threading.Thread(target=feed_worker, args=(feed,), daemon=True) for feed in device_feeds]
        for t in feed_threads:
            t.start()
        for t in feed_threads:
            t.join()

    feeds_by_detector = _group_by_detector(feeds, engine)
    device_threads = [
        threading.Thread(target=device_worker, args=(key, device_feeds), daemon=True)
        for key, device_feeds in feeds_by_detector.items()
    ]
    for t in device_threads:
        t.start()
    while any(t.is_alive() for t in device_threads):
        on_counts(state.snapshot(time.monotonic()))
        for t in device_threads:
            t.join(timeout=0.5 / len(device_threads))
    on_counts(state.snapshot(time.monotonic()))

    if errors and not ready_feeds and not (stop_event is not None and stop_event.is_set()):
        # Threads don't propagate exceptions to the caller on their own --
        # if nothing ever ran, surface the first failure rather than
        # returning as if the run had simply finished.
        key, exc = next(iter(errors.items()))
        raise RuntimeError(f"'{key}' failed: {exc}") from exc
