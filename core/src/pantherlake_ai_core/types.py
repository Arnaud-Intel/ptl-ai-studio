"""Shared result types used across demo bricks."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranslationResult:
    text: str
    detected_language: str
    language_probability: float
