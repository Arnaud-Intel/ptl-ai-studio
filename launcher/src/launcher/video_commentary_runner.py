"""Runs the video-commentary brick on a background thread: the video as an
MJPEG stream (only the newest frame matters, as for object detection), and
the comments as a short list the page asks for.

Two models on two chips, so two rows in the hardware panel: the vision
model under the chip that sees ("vision"), the language model under the one
that gives the line its mood ("mood"), each with how fast its last line
went. The mood can be changed while the video plays.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict

import cv2
import numpy as np
from pantherlake_ai_core.engine import Engine
from video_commentary import moods, pipeline

from . import activity, events, metrics, worker

_DEMO_ID = "video-commentary"
_ENGINE = Engine.OPENVINO.value
_JPEG_QUALITY = 80
# The picture as the page shows it: wide enough for a panel, small enough
# that encoding thirty of them a second is not the busiest thing here.
_STREAM_WIDTH = 960
_KEPT = 30  # comments the page can still ask for
VISION, MOOD = "vision", "mood"
_STAGES = (VISION, MOOD)
_LABELS = {VISION: "Sees", MOOD: "Says"}


class VideoCommentaryRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._comments: deque[dict] = deque(maxlen=_KEPT)
        self._count = 0
        self._mood = moods.DEFAULT
        self._devices: dict[str, str] = {}
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def mood(self) -> str:
        return self._mood

    def set_mood(self, key: str) -> str:
        """The voice of the next comment on. An unknown one is a ValueError."""
        self._mood = moods.get(key).key
        return self._mood

    def start(
        self, *, source: str, path: str, camera_index: int, screen_index: int, loop: bool,
        vision_device: str, mood_device: str, mood: str, every: float,
    ) -> None:
        worker.refuse_if_busy(_DEMO_ID, self._thread, self._stop_event)
        self.set_mood(mood)
        if source == "file" and not path.strip():
            raise ValueError("Choose a video file, or one of the samples.")
        self.error = None
        with self._lock:
            self._latest_jpeg = None
            self._comments.clear()
            self._count = 0
        self._devices = {VISION: vision_device, MOOD: mood_device}
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def on_frame(frame: np.ndarray) -> None:
            if frame.shape[1] > _STREAM_WIDTH:
                height = int(frame.shape[0] * _STREAM_WIDTH / frame.shape[1])
                frame = cv2.resize(frame, (_STREAM_WIDTH, height), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

        def on_comment(comment: pipeline.Comment) -> None:
            with self._lock:
                self._count += 1
                self._comments.append({"number": self._count, **asdict(comment)})

        def on_work(stage: str, working: bool, stats) -> None:
            if stats is not None and getattr(stats, "tokens_per_second", 0):
                # The figure of the line just written: it stays on the chip's
                # row between two lines, which is most of the time.
                metrics.report(_DEMO_ID, stats.tokens_per_second, "tok/s", stage=stage, sticky=True)

        def on_ready() -> None:
            events.set_phase(_DEMO_ID, "running", "Watching and commenting...")

        def on_downloading() -> None:
            events.set_phase(_DEMO_ID, "loading", "Downloading a model (first run only)...")

        def target() -> None:
            for stage in _STAGES:
                activity.set_active(_DEMO_ID, engine=_ENGINE, device=self._devices[stage], stage=stage, stage_label=_LABELS[stage])
            events.set_phase(
                _DEMO_ID, "loading", f"Loading the vision model on {vision_device} and the language model on {mood_device}..."
            )
            try:
                pipeline.run(
                    source=source, path=path.strip(), camera_index=camera_index, screen_index=screen_index, loop=loop,
                    vision_device=vision_device, mood_device=mood_device, mood=lambda: self._mood, every=every,
                    on_frame=on_frame, on_comment=on_comment, on_work=on_work, on_ready=on_ready,
                    on_downloading=on_downloading, stop_event=stop_event,
                )
            except Exception as exc:  # surfaced to the UI, not silently dropped
                self.error = str(exc)
                events.set_phase(_DEMO_ID, "error", str(exc))
            else:
                events.clear_phase(_DEMO_ID)
            finally:
                for stage in _STAGES:
                    activity.clear_active(_DEMO_ID, stage=stage)
                metrics.clear(_DEMO_ID)

        self._thread = threading.Thread(target=target, daemon=True, name="video-commentary")
        self._thread.start()

    def stop(self) -> None:
        if not worker.request_stop(_DEMO_ID, self._thread, self._stop_event):
            return
        self._thread = None
        with self._lock:
            self._latest_jpeg = None

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def comments(self, after: int = 0) -> list[dict]:
        """The comments numbered above `after`, oldest first."""
        with self._lock:
            return [comment for comment in self._comments if comment["number"] > after]

    def state(self) -> dict:
        return {
            "running": self.running, "mood": self._mood, "error": self.error,
            "devices": dict(self._devices) if self.running else {},
        }
