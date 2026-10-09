"""Capture -> detect loop, shared by the CLI and any UI front-end (e.g. the
launcher) so this logic lives in one place. Deliberately does not draw
boxes or encode frames -- that's a presentation concern each consumer
handles differently (a CLI window vs. an MJPEG stream).

Three sources: a webcam, a screen, and a video file -- the studio's sample
videos among them, so that the brick has something to show on a machine
whose camera is covered and whose screen holds a spreadsheet.

Checked on 2026-10-10 (the proofing pass the Auto Demo's "Seeing and
answering" waited for), YOLO11s on the XPS 14's integrated GPU:

- A frame is handed to the detector, and to whoever draws it, no wider
  than `PICTURE_WIDTH`. The model looks at 640 pixels whatever it is
  given, so nothing is found on a 2880-pixel screen that is not found on
  the same screen at 1280 -- and getting there cost 20 ms of the frame's
  83, with another 21 to draw and encode it whole. The screen went from
  12 frames a second to 19. What is left is the capture itself: 41 ms to
  grab 2880x1800 pixels, whatever is done with them afterwards.
- Through the launcher: a video file at its own rate (24 to 30 frames a
  second) on the integrated GPU, the NPU and the CPU alike; the webcam at
  30; DETR on the portable engine at 4, finding three times the people.
- The source is checked before a model is loaded for it. A wrong one used
  to cost the model's load, and was then reported from a thread nobody
  was waiting on: the launcher had already answered "started".
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Iterator

import cv2
import numpy as np
from pantherlake_ai_core import video
from pantherlake_ai_core.engine import Engine

from .engine_factory import create_detector
from .types import Detection

SOURCES = ("webcam", "screen", "file")
PICTURE_WIDTH = 1280


def shown(frame: np.ndarray) -> np.ndarray:
    """The frame at the size it is looked at and drawn on."""
    if frame.shape[1] <= PICTURE_WIDTH:
        return frame
    height = int(frame.shape[0] * PICTURE_WIDTH / frame.shape[1])
    return cv2.resize(frame, (PICTURE_WIDTH, height), interpolation=cv2.INTER_AREA)


def frames_from(
    source: str, *, camera_index: int = 0, screen_index: int = 1, path: str = "", loop: bool = True,
    stop_event: threading.Event | None = None,
) -> Iterator[np.ndarray]:
    """The source's frames. What can be known to be wrong without opening
    anything is said here and now, not at the first frame."""
    if source == "webcam":
        return video.stream_camera_frames(camera_index, stop_event=stop_event)
    if source == "screen":
        return video.stream_screen_frames(screen_index, stop_event=stop_event)
    if source == "file":
        if not path:
            raise ValueError("A video file is needed: give its path.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Not a file: {path}")
        return video.stream_video_file_frames(path, loop=loop, stop_event=stop_event)
    raise ValueError(f"Unknown source '{source}': one of {', '.join(SOURCES)}.")


def run(
    *,
    source: str,
    camera_index: int = 0,
    screen_index: int = 1,
    path: str = "",
    loop: bool = True,
    engine: Engine,
    compute_device: str,
    model_path: str | None = None,
    on_frame: Callable[[np.ndarray, list[Detection]], None],
    on_ready: Callable[[], None] | None = None,
    on_downloading: Callable[[], None] | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread, calling `on_frame(frame, detections)` for
    each captured frame, until `stop_event` is set, or the video file ends
    when it is not played in a loop (or forever if neither). The frame is
    the one the boxes belong to: no wider than `PICTURE_WIDTH`.

    `on_ready`, if given, fires once the model is loaded and capture is
    about to start -- the real "loading -> running" boundary (a first-run
    download plus compile can take a while); `on_downloading` fires before
    that load has to fetch the model first."""
    frames = frames_from(
        source, camera_index=camera_index, screen_index=screen_index, path=path, loop=loop, stop_event=stop_event
    )
    detector = create_detector(engine, device=compute_device, model_path=model_path, on_downloading=on_downloading)
    if on_ready is not None:
        on_ready()

    for frame in frames:
        picture = shown(frame)
        on_frame(picture, detector.detect(picture))
