"""Picks a text-extraction backend (engine) so the rest of the brick
doesn't need to know which one it's talking to -- both expose
`.extract(image, translate=False) -> ExtractionResult`.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from pantherlake_ai_core.engine import Engine

from .types import ExtractionResult


class Extractor(Protocol):
    def extract(self, image: np.ndarray, translate: bool = False) -> ExtractionResult: ...


def create_extractor(
    engine: Engine,
    *,
    device: str = "AUTO",
    model_path: str | None = None,
    on_downloading: Callable[[], None] | None = None,
) -> Extractor:
    """`on_downloading`, if given, fires before an openvino model that isn't
    already cached locally starts downloading (portable's RapidOCR download
    isn't covered)."""
    if engine == Engine.PORTABLE:
        from .extractor_portable import PortableExtractor

        return PortableExtractor()

    if engine == Engine.OPENVINO:
        from .extractor_openvino import OpenVINOExtractor

        return OpenVINOExtractor(device=device, model_dir=model_path, on_downloading=on_downloading)

    raise ValueError(f"Unknown engine '{engine}'.")
