"""Shared result types for the object-detection brick."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords in the original frame
