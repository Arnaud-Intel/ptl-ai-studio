"""In-memory cosine-similarity vector store for indexed document chunks,
with on-disk caching so re-running against the same folder/engine/model
doesn't re-embed everything from scratch.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .types import Chunk, RetrievedChunk


def cache_dir_for(folder: Path, model_key: str) -> Path:
    """A stable cache location keyed by the ingested folder + which
    embedding model produced the vectors (different models => incompatible
    vector spaces, so they must never share a cache entry)."""
    key = f"{folder.resolve()}::{model_key}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return Path.home() / ".cache" / "pantherlake-ai-studio" / "doc-qa" / digest


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None  # (n, d), L2-normalized rows

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        """The files the index was built from, in the order they were read."""
        return list(dict.fromkeys(chunk.source for chunk in self.chunks))

    @staticmethod
    def _normalize(vectors: list[list[float]]) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def build(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.chunks = chunks
        self._vectors = self._normalize(vectors)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Appends to the existing index instead of replacing it -- for
        incremental/streaming ingestion (e.g. smart-recall indexing screen
        captures continuously) as opposed to build()'s one-shot full
        rebuild."""
        normalized = self._normalize(vectors)
        self._vectors = normalized if self._vectors is None else np.concatenate([self._vectors, normalized], axis=0)
        self.chunks.extend(chunks)

    def search(self, query_vector: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        if self._vectors is None or not self.chunks:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        scores = self._vectors @ query
        top_indices = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in top_indices]

    def save(self, cache_dir: Path, *, fingerprint: str | None = None) -> None:
        """`fingerprint`: what the index was built from, for `load` to
        check before it is used again."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "vectors.npy", self._vectors)
        meta = [asdict(c) for c in self.chunks]
        (cache_dir / "chunks.json").write_text(json.dumps(meta), encoding="utf-8")
        if fingerprint is not None:
            (cache_dir / "built-from.json").write_text(json.dumps({"fingerprint": fingerprint}), encoding="utf-8")

    @classmethod
    def load(cls, cache_dir: Path, *, fingerprint: str | None = None) -> "VectorStore | None":
        """With a `fingerprint`, only an index built from exactly that is
        loaded: one built from anything else, or from nobody knows what
        (saved before fingerprints were kept), is not there."""
        vectors_path = cache_dir / "vectors.npy"
        chunks_path = cache_dir / "chunks.json"
        if not vectors_path.exists() or not chunks_path.exists():
            return None
        if fingerprint is not None:
            try:
                built_from = json.loads((cache_dir / "built-from.json").read_text(encoding="utf-8")).get("fingerprint")
            except (OSError, ValueError, AttributeError):
                built_from = None
            if built_from != fingerprint:
                return None
        store = cls()
        store._vectors = np.load(vectors_path)
        meta = json.loads(chunks_path.read_text(encoding="utf-8"))
        store.chunks = [Chunk(**m) for m in meta]
        return store
