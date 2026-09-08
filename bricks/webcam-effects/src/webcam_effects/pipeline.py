"""Capture -> segment loop, shared by the CLI and any UI front-end (e.g.
the launcher).

Deliberately doesn't apply the blur/replace effect itself -- re-rendering
an effect from (frame, mask) is cheap, so each consumer applies it
independently. That's what lets the launcher change the effect or its
color live while capture keeps running, instead of needing a restart for
something that isn't actually a capture/model concern.
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from pantherlake_ai_core import video
from pantherlake_ai_core.engine import Engine

from .engine_factory import create_segmenter


def run(
    *,
    camera_index: int,
    engine: Engine,
    compute_device: str,
    model_path: str | None = None,
    on_frame: Callable[[np.ndarray, np.ndarray], None],
    on_ready: Callable[[], None] | None = None,
    on_downloading: Callable[[], None] | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread, calling `on_frame(frame, mask)` for each
    captured frame, until `stop_event` is set (or forever if none given).
    `on_ready`, if given, fires once the model is loaded and the camera is
    about to be opened -- the real "loading -> running" boundary;
    `on_downloading` fires before that load has to fetch the model first."""
    segmenter = create_segmenter(engine, device=compute_device, model_path=model_path, on_downloading=on_downloading)
    if on_ready is not None:
        on_ready()
    frames = video.stream_camera_frames(camera_index, stop_event=stop_event)
    for frame in frames:
        mask = segmenter.segment(frame)
        on_frame(frame, mask)
