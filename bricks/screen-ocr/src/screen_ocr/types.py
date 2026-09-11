"""Shared result types for the screen-ocr brick."""
from __future__ import annotations

from dataclasses import dataclass, field

from pantherlake_ai_core.types import GenerationStats


@dataclass
class TextRegion:
    text: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords in the original image


@dataclass
class ExtractionResult:
    text: str
    regions: list[TextRegion] = field(default_factory=list)  # empty when the engine doesn't localize text (e.g. a VLM)
    translated_text: str | None = None
    stats: GenerationStats | None = None  # the vision-language model's speed; None for dedicated OCR models
