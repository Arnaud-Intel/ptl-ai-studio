"""Shared result types for the doc-qa brick."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    source: str  # file path, relative to the ingested folder
    chunk_index: int


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
