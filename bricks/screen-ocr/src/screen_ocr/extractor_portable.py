"""Portable text-extraction backend: RapidOCR (PaddleOCR detection +
recognition models) via ONNX Runtime, CPU.

Dedicated OCR models, not a language model -- fast, and returns real
per-region boxes + confidence, but can't translate (there's no language
model attached). See extractor_openvino.py for the engine that can.
"""
from __future__ import annotations

import numpy as np

from .types import ExtractionResult, TextRegion


class PortableExtractor:
    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self.engine = RapidOCR()

    def extract(self, image: np.ndarray, translate: bool = False) -> ExtractionResult:
        if translate:
            raise ValueError(
                "Translation requires the openvino engine (it uses a vision-language model; "
                "the portable engine's dedicated OCR models have no language model attached)."
            )

        result = self.engine(image)
        regions: list[TextRegion] = []
        if result.txts:
            for text, score, box in zip(result.txts, result.scores, result.boxes):
                xs = [point[0] for point in box]
                ys = [point[1] for point in box]
                regions.append(
                    TextRegion(
                        text=text,
                        confidence=float(score),
                        box=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    )
                )

        full_text = "\n".join(r.text for r in regions)
        return ExtractionResult(text=full_text, regions=regions)
