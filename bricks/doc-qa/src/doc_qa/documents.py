"""Load and chunk local documents (.txt, .md, .pdf) for indexing."""
from __future__ import annotations

from pathlib import Path

from .types import Chunk

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(folder: Path) -> list[tuple[str, str]]:
    """Return (path relative to folder, full text) for every supported file."""
    docs: list[tuple[str, str]] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_text_file(path)
        text = text.strip()
        if text:
            docs.append((str(path.relative_to(folder)).replace("\\", "/"), text))
    return docs


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Character-based sliding-window chunking.

    Deliberately simple (no tokenizer dependency): a fixed character window
    with overlap is good enough for retrieval at this scale, and keeps
    chunking independent of whichever embedding model/engine is selected.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def build_chunks(folder: Path, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in load_documents(folder):
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(Chunk(text=piece, source=source, chunk_index=i))
    return chunks
