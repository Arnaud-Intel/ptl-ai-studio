"""Runs the object-detection brick's capture/detect loop on a background
thread. Unlike live-translation's queue (every result matters), video only
needs the *latest* annotated frame -- older ones are stale the instant a
new one exists -- so this just keeps one JPEG buffer overwritten in place,
which an MJPEG stream reads from at its own pace.
"""
from __future__ import annotations

import threading
from dataclasses import asdict

import cv2
import numpy as np
from object_detection import pipeline
from object_detection.draw import draw_detections
from object_detection.types import Detection
from pantherlake_ai_core.engine import Engine

from . import activity

_DEMO_ID = "object-detection"
_JPEG_QUALITY = 80


class ObjectDetectionRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_detections: list[dict] = []
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        source: str,
        camera_index: int,
        screen_index: int,
        engine: Engine,
        compute_device: str,
    ) -> None:
        if self.running:
            raise RuntimeError("object-detection is already running")

        self.error = None
        with self._frame_lock:
            self._latest_jpeg = None
            self._latest_detections = []
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def on_frame(frame: np.ndarray, detections: list[Detection]) -> None:
            annotated = draw_detections(frame, detections)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if not ok:
                return
            with self._frame_lock:
                self._latest_jpeg = buf.tobytes()
                self._latest_detections = [asdict(d) for d in detections]

        def target() -> None:
            activity.set_active(_DEMO_ID, engine=engine.value, device=compute_device)
            try:
                pipeline.run(
                    source=source,
                    camera_index=camera_index,
                    screen_index=screen_index,
                    engine=engine,
                    compute_device=compute_device,
                    on_frame=on_frame,
                    stop_event=stop_event,
                )
            except Exception as exc:  # surfaced to the UI, not silently dropped
                self.error = str(exc)
            finally:
                activity.clear_active(_DEMO_ID)

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        self._thread = None
        with self._frame_lock:
            self._latest_jpeg = None
            self._latest_detections = []

    def latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def latest_detections(self) -> list[dict]:
        with self._frame_lock:
            return list(self._latest_detections)
