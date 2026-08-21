"""Picks a text-extraction backend (engine) so the rest of the brick
doesn't need to know which one it's talking to -- both expose
`.extract(image, translate=False) -> ExtractionResult`.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from pantherlake_ai_core.engine import Engine

from .types import ExtractionResult


class Extractor(Protocol):
    def extract(self, image: np.ndarray, translate: bool = False) -> ExtractionResult: ...


def create_extractor(engine: Engine, *, device: str = "AUTO", model_path: str | None = None) -> Extractor:
    if engine == Engine.PORTABLE:
        from .extractor_portable import PortableExtractor

        return PortableExtractor()

    if engine == Engine.OPENVINO:
        from .extractor_openvino import OpenVINOExtractor

        return OpenVINOExtractor(device=device, model_dir=model_path)

    raise ValueError(f"Unknown engine '{engine}'.")
