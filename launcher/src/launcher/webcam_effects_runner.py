"""Runs the webcam-effects brick's capture/segment loop on a background
thread (same latest-frame-buffer shape as object_detection_runner.py).

Unlike object-detection, the effect (blur vs. replace, and the replace
color) isn't part of the capture pipeline -- `webcam_effects.pipeline`
only produces (frame, mask), and this runner applies the effect itself
when encoding each frame. That means `set_effect()` can change the look
live, with no restart: the segmentation model keeps running exactly as
before, only the cheap per-frame blend changes.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np
from pantherlake_ai_core.engine import Engine
from webcam_effects import matte, pipeline

from . import activity

_DEMO_ID = "webcam-effects"
_JPEG_QUALITY = 80


class WebcamEffectsRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._person_coverage: float = 0.0
        self._effect = "blur"
        self._color: tuple[int, int, int] = (181, 104, 0)  # BGR -- Intel blue
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_effect(self, effect: str, color: tuple[int, int, int] | None = None) -> None:
        with self._frame_lock:
            self._effect = effect
            if color is not None:
                self._color = color

    def start(
        self,
        *,
        camera_index: int,
        engine: Engine,
        compute_device: str,
        effect: str = "blur",
        color: tuple[int, int, int] = (181, 104, 0),
    ) -> None:
        if self.running:
            raise RuntimeError("webcam-effects is already running")

        self.error = None
        with self._frame_lock:
            self._latest_jpeg = None
            self._person_coverage = 0.0
            self._effect = effect
            self._color = color
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def on_frame(frame: np.ndarray, mask: np.ndarray) -> None:
            with self._frame_lock:
                current_effect = self._effect
                current_color = self._color

            if current_effect == "replace":
                annotated = matte.apply_replace(frame, mask, current_color)
            else:
                annotated = matte.apply_blur(frame, mask)

            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if not ok:
                return
            with self._frame_lock:
                self._latest_jpeg = buf.tobytes()
                self._person_coverage = matte.person_coverage(mask)

        def target() -> None:
            activity.set_active(_DEMO_ID, engine=engine.value, device=compute_device)
            try:
                pipeline.run(
                    camera_index=camera_index,
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
            self._person_coverage = 0.0

    def latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def latest_stats(self) -> dict:
        with self._frame_lock:
            return {"person_coverage": self._person_coverage}
