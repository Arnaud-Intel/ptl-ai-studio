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

import cv2
import numpy as np
from pantherlake_ai_core.engine import Engine
from smart_city_monitor import pipeline
from smart_city_monitor.draw import draw_tracks
from smart_city_monitor.types import CountSnapshot, FeedSpec, TrackedDetection

from . import activity, events

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
        if self.running:
            raise RuntimeError("smart-city-monitor is already running")

        self.error = None
        with self._frame_lock:
            self._latest_jpeg = {}
            self._latest_snapshot = None
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

        def on_counts(snapshot: CountSnapshot) -> None:
            with self._frame_lock:
                self._latest_snapshot = snapshot

        def on_ready(feed_id: str) -> None:
            events.set_phase(f"{_DEMO_ID}:{feed_id}", "running", "Monitoring...")

        def on_feed_error(feed_id: str, message: str) -> None:
            # One feed failing (bad device id, unreadable file) must be visible
            # right away on that feed's tile -- the others keep running.
            events.set_phase(f"{_DEMO_ID}:{feed_id}", "error", message)
            activity.clear_active(_DEMO_ID, stage=feed_id)

        def target() -> None:
            for feed in feeds:
                activity.set_active(
                    _DEMO_ID, engine=engine.value, device=feed.compute_device,
                    stage=feed.feed_id, stage_label=f"Feed {feed.feed_id.removeprefix('feed-')}",
                )
                events.set_phase(
                    f"{_DEMO_ID}:{feed.feed_id}", "loading",
                    f"Loading model (engine={engine.value}, device={feed.compute_device})...",
                )
            try:
                pipeline.run(
                    feeds=feeds,
                    engine=engine,
                    loop=loop,
                    on_ready=on_ready,
                    on_feed_error=on_feed_error,
                    on_frame=on_frame,
                    on_counts=on_counts,
                    stop_event=stop_event,
                )
            except Exception as exc:  # surfaced to the UI, not silently dropped
                self.error = str(exc)
                for feed in feeds:
                    events.set_phase(f"{_DEMO_ID}:{feed.feed_id}", "error", str(exc))
            else:
                for feed in feeds:
                    events.clear_phase(f"{_DEMO_ID}:{feed.feed_id}")
            finally:
                for feed in feeds:
                    activity.clear_active(_DEMO_ID, stage=feed.feed_id)

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        thread = self._thread
        if thread is not None:
            # Wait for every feed thread to actually exit, so `running` only
            # turns false once they have -- otherwise a quick Stop -> Start
            # overlaps two runs on the same files/devices. Feeds still inside
            # a long model load keep `running` true until they get out.
            thread.join(timeout=3.0)
            if thread.is_alive():
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

    def feeds(self) -> list[FeedSpec]:
        return list(self._feeds)
