"""Picks a detection backend (engine) so the rest of the brick doesn't need
to know which one it's talking to -- both expose `.detect(frame) -> list[Detection]`.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from pantherlake_ai_core.engine import Engine

from .types import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def create_detector(engine: Engine, *, device: str = "AUTO", model_path: str | None = None) -> Detector:
    if engine == Engine.PORTABLE:
        from .detector_portable import PortableDetector

        return PortableDetector(model_path=model_path)

    if engine == Engine.OPENVINO:
        from .detector_openvino import OpenVINODetector

        return OpenVINODetector(device=device, model_dir=model_path)

    raise ValueError(f"Unknown engine '{engine}'.")
