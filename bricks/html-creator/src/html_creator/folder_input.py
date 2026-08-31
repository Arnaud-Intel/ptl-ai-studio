"""Reads a folder of documents into one text blob for summarization.

Concatenates every supported file into a single pass rather than
summarizing per-document and composing the summaries -- simpler, and
consistent with code-review-assist's diff handling (cap input size, tell
the caller if truncation happened). Per-document-then-compose would be
more robust for a large folder (more signal survives truncation) at the
cost of one LLM call per document; worth revisiting if truncation turns
out to bite in practice, not built speculatively now.
"""
from __future__ import annotations

from pathlib import Path

MAX_DOCUMENT_CHARS = 20_000


def read_documents(folder: str) -> str:
    from doc_qa.documents import load_documents

    resolved = Path(folder).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Not a folder: {resolved}")

    docs = load_documents(resolved)
    return "\n\n".join(f"--- {path} ---\n{text}" for path, text in docs)


def truncate_documents(text: str, max_chars: int = MAX_DOCUMENT_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True
