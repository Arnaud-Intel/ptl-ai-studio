"""Capture -> detect loop, shared by the CLI and any UI front-end (e.g. the
launcher) so this logic lives in one place. Deliberately does not draw
boxes or encode frames -- that's a presentation concern each consumer
handles differently (a CLI window vs. an MJPEG stream).
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from pantherlake_ai_core import video
from pantherlake_ai_core.engine import Engine

from .engine_factory import create_detector
from .types import Detection


def run(
    *,
    source: str,
    camera_index: int,
    screen_index: int,
    engine: Engine,
    compute_device: str,
    model_path: str | None = None,
    on_frame: Callable[[np.ndarray, list[Detection]], None],
    on_ready: Callable[[], None] | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread, calling `on_frame(frame, detections)` for
    each captured frame, until `stop_event` is set (or forever if none is
    given). `on_ready`, if given, fires once the model is loaded and capture
    is about to start -- the real "loading -> running" boundary (a first-run
    download plus compile can take a while)."""
    detector = create_detector(engine, device=compute_device, model_path=model_path)
    if on_ready is not None:
        on_ready()

    if source == "webcam":
        frames = video.stream_camera_frames(camera_index, stop_event=stop_event)
    elif source == "screen":
        frames = video.stream_screen_frames(screen_index, stop_event=stop_event)
    else:
        raise ValueError(f"Unknown source '{source}', expected 'webcam' or 'screen'.")

    for frame in frames:
        detections = detector.detect(frame)
        on_frame(frame, detections)
