"""Runs the smart-city-monitor brick's multi-feed capture/track/count loop
on a background thread. Same "latest frame wins" shape as
ObjectDetectionRunner, generalized to N feeds (one JPEG buffer per feed
id) plus a live counts snapshot. The interesting part: each feed gets its
own `activity.py` stage, keyed by feed id -- so N feeds pinned to N
different devices all show up on the telemetry gauges simultaneously,
correctly attributed, the same way expense-extract's OCR/LLM stages do.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from pantherlake_ai_core.engine import Engine
from smart_city_monitor import pipeline
from smart_city_monitor.draw import draw_tracks
from smart_city_monitor.types import CountSnapshot, FeedSpec, TrackedDetection

from . import activity, events, worker

_DEMO_ID = "smart-city-monitor"
_JPEG_QUALITY = 80


class SmartCityMonitorRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._frame_lock = threading.Lock()
        self._latest_jpeg: dict[str, bytes] = {}
        self._latest_snapshot: CountSnapshot | None = None
        self._feeds: list[FeedSpec] = []
        self.error: str | None = None
        self._health: dict[str, dict] = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        feeds: list[FeedSpec],
        engine: Engine,
        loop: bool,
    ) -> None:
        worker.refuse_if_busy(_DEMO_ID, self._thread, self._stop_event)

        self.error = None
        with self._frame_lock:
            self._latest_jpeg = {}
            self._latest_snapshot = None
            self._health = {f.feed_id: {'phase': 'loading', 'message': 'Loading model...', 'last_frame': None} for f in feeds}
        self._feeds = feeds
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def on_frame(feed_id: str, frame: np.ndarray, tracks: list[TrackedDetection]) -> None:
            annotated = draw_tracks(frame, tracks)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if not ok:
                return
            with self._frame_lock:
                self._latest_jpeg[feed_id] = buf.tobytes()
                self._health[feed_id]['last_frame'] = time.monotonic()

        def on_counts(snapshot: CountSnapshot) -> None:
            with self._frame_lock:
                self._latest_snapshot = snapshot

        def on_ready(feed_id: str) -> None:
            on_feed_status(feed_id, 'running', 'Monitoring...')

        def on_feed_status(feed_id: str, phase: str, message: str) -> None:
            if stop_event.is_set():
                return
            with self._frame_lock:
                old = self._health[feed_id]
                if old['phase'] == phase and old['message'] == message:
                    return
                self._health[feed_id].update(phase=phase, message=message)
            feed = next(f for f in feeds if f.feed_id == feed_id)
            if phase == 'running':
                activity.set_active(_DEMO_ID, engine=(feed.engine or engine).value, device=feed.compute_device,
                                    stage=feed_id, stage_label=f"Feed {feed_id.removeprefix('feed-')}")
            else:
                activity.clear_active(_DEMO_ID, stage=feed_id)
            events.set_phase(_DEMO_ID, phase, message, stage=feed_id)

        def on_feed_error(feed_id: str, message: str) -> None:
            # One feed failing (bad device id, unreadable file) must be visible
            # right away on that feed's tile -- the others keep running.
            on_feed_status(feed_id, 'error', message)
            activity.clear_active(_DEMO_ID, stage=feed_id)

        def target() -> None:
            for feed in feeds:
                # A feed may run a different engine from the rest, so report
                # its own rather than the run's -- the telemetry gauges and
                # the status line both name what that feed is actually using.
                feed_engine = (feed.engine or engine).value
                activity.set_active(
                    _DEMO_ID, engine=feed_engine, device=feed.compute_device,
                    stage=feed.feed_id, stage_label=f"Feed {feed.feed_id.removeprefix('feed-')}",
                )
                events.set_phase(
                    _DEMO_ID, "loading",
                    f"Loading model (engine={feed_engine}, device={feed.compute_device})...",
                    stage=feed.feed_id,
                )
            try:
                pipeline.run(
                    feeds=feeds,
                    engine=engine,
                    loop=loop,
                    on_ready=on_ready,
                    on_feed_error=on_feed_error,
                    on_feed_status=on_feed_status,
                    on_frame=on_frame,
                    on_counts=on_counts,
                    stop_event=stop_event,
                )
            except Exception as exc:  # surfaced to the UI, not silently dropped
                self.error = str(exc)
                for feed in feeds:
                    if self._health[feed.feed_id]['phase'] != 'error':
                        on_feed_status(feed.feed_id, 'error', str(exc))
            else:
                for feed in feeds:
                    if self._health[feed.feed_id]['phase'] != 'error' or stop_event.is_set():
                        events.clear_phase(_DEMO_ID, stage=feed.feed_id)
                        with self._frame_lock:
                            self._health[feed.feed_id].update(phase='idle', message='Stopped.')
                failed = [h['message'] for h in self._health.values() if h['phase'] == 'error']
                if failed and len(failed) == len(feeds) and not stop_event.is_set():
                    self.error = 'All feeds stopped. ' + failed[0]
            finally:
                for feed in feeds:
                    activity.clear_active(_DEMO_ID, stage=feed.feed_id)

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # One "stopping" phase per feed, since that is how this brick reports
        # every other phase -- each feed has its own tile in the UI.
        stages = [feed.feed_id for feed in self._feeds] or [None]
        if not worker.request_stop(_DEMO_ID, self._thread, self._stop_event, stages=stages):
            return
        self._thread = None
        self._feeds = []
        with self._frame_lock:
            self._latest_jpeg = {}
            self._latest_snapshot = None

    def latest_jpeg(self, feed_id: str) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg.get(feed_id)

    def latest_snapshot(self) -> CountSnapshot | None:
        with self._frame_lock:
            return self._latest_snapshot

    def health(self) -> dict:
        with self._frame_lock:
            now = time.monotonic()
            result = {}
            for fid, entry in self._health.items():
                age = None if entry['last_frame'] is None else max(0, now - entry['last_frame'])
                phase, message = entry['phase'], entry['message']
                if phase == 'running' and age is not None and age > 15:
                    phase, message = 'waiting', 'No recent frames; showing the last received image. Refreshed clips pause between updates.'
                result[fid] = {'phase': phase, 'message': message, 'frame_age_seconds': None if age is None else round(age, 1)}
            return result

    def feeds(self) -> list[FeedSpec]:
        return list(self._feeds)
