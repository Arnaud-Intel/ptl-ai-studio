"""Picks a segmentation backend (engine) so the rest of the brick doesn't
need to know which one it's talking to -- both expose
`.segment(frame) -> mask` (a float32 (H, W) array in [0, 1]).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from pantherlake_ai_core.engine import Engine


class Segmenter(Protocol):
    def segment(self, frame: np.ndarray) -> np.ndarray: ...


def create_segmenter(engine: Engine, *, device: str = "AUTO", model_path: str | None = None) -> Segmenter:
    if engine == Engine.PORTABLE:
        from .segmenter_portable import PortableSegmenter

        return PortableSegmenter(model_path=model_path)

    if engine == Engine.OPENVINO:
        from .segmenter_openvino import OpenVINOSegmenter

        return OpenVINOSegmenter(device=device, model_path=model_path)

    raise ValueError(f"Unknown engine '{engine}'.")
