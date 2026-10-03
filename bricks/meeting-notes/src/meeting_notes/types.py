"""Shared result types for the meeting-notes brick."""
from __future__ import annotations

from dataclasses import dataclass

from pantherlake_ai_core.types import GenerationStats


@dataclass
class TranscriptLine:
    timestamp: str
    text: str
    detected_language: str
    # Seconds of speech per second of processing, when the capture loop timed it.
    realtime_factor: float | None = None


@dataclass
class MeetingNotes:
    text: str
    transcript_line_count: int
    # 1 when the whole transcript fit the model at once; more when a long
    # meeting was summarised part by part and the parts merged.
    parts: int = 1
    stats: GenerationStats | None = None  # every model call for these notes, together
    cancelled: bool = False  # stopped part-way: these are not the finished notes
