"""Shared result types for the meeting-notes brick."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptLine:
    timestamp: str
    text: str
    detected_language: str


@dataclass
class MeetingNotes:
    text: str
    transcript_line_count: int
    # 1 when the whole transcript fit the model at once; more when a long
    # meeting was summarised part by part and the parts merged.
    parts: int = 1
