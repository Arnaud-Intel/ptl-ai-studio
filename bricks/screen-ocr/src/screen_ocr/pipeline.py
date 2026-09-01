"""One-shot text extraction: capture or load an image, run it through the
selected engine. Shared by the CLI and the launcher.

Unlike live-translation's continuous loop or object-detection's video
feed, OCR is naturally a discrete action (grab one screenshot/photo, get
its text) -- so there's no capture loop here, just a session that holds
one loaded extractor and can be called as many times as you like.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from pantherlake_ai_core.engine import Engine

from .engine_factory import create_extractor
from .types import ExtractionResult


class OcrSession:
    def __init__(
        self,
        engine: Engine,
        *,
        device: str = "AUTO",
        model_path: str | None = None,
        on_downloading: Callable[[], None] | None = None,
    ):
        self.engine = engine
        self.extractor = create_extractor(engine, device=device, model_path=model_path, on_downloading=on_downloading)

    def extract(self, image: np.ndarray, translate: bool = False) -> ExtractionResult:
        return self.extractor.extract(image, translate=translate)
