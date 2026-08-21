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
