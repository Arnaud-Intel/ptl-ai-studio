"""Ingest a folder of documents and answer questions about them -- shared
by the CLI and any UI front-end (e.g. the launcher) so this logic lives in
one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from pantherlake_ai_core.engine import Engine

from . import documents
from .engine_factory import TEMPLATE_TOKENS, create_embedder, create_llm
from .store import VectorStore, cache_dir_for
from .types import Answer

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the "
    "provided excerpts from the user's own documents. If the excerpts "
    "don't contain the answer, say you don't know rather than guessing. "
    "Refer to which excerpt(s) you used by their [number]."
)


class DocQASession:
    """Holds one loaded embedder + LLM + index. Ingest once, ask many times."""

    def __init__(
        self,
        engine: Engine,
        *,
        device: str = "AUTO",
        model_dir: str | None = None,
        on_downloading: Callable[[], None] | None = None,
    ):
        self.engine = engine
        self.embedder = create_embedder(engine, device=device, model_dir=model_dir, on_downloading=on_downloading)
        self.llm = create_llm(engine, device=device, model_dir=model_dir, on_downloading=on_downloading)
        self.store = VectorStore()
        self.folder: Path | None = None

    def ingest(
        self,
        folder: str | Path,
        *,
        chunk_size: int = 900,
        overlap: int = 150,
        force: bool = False,
    ) -> int:
        """Build (or load a cached) index for `folder`. Returns the chunk count."""
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Not a folder: {folder}")

        model_key = f"{self.engine.value}:{chunk_size}:{overlap}"
        cache_dir = cache_dir_for(folder, model_key)

        if not force:
            cached = VectorStore.load(cache_dir)
            if cached is not None:
                self.store = cached
                self.folder = folder
                return self.store.size

        chunks = documents.build_chunks(folder, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            suffixes = ", ".join(sorted(documents.SUPPORTED_SUFFIXES))
            raise ValueError(f"No supported documents ({suffixes}) found under {folder}")

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.build(chunks, vectors)
        self.store.save(cache_dir)
        self.folder = folder
        return self.store.size

    def ask(self, question: str, *, top_k: int = 4, max_tokens: int = 512, control=None) -> Answer:
        if self.store.size == 0:
            raise RuntimeError("No documents ingested yet -- call ingest() first.")

        query_vector = self.embedder.embed_query(question)
        retrieved = self.store.search(query_vector, top_k=top_k)
        used = fit_excerpts(self.llm, retrieved, question, max_tokens)
        text = self.llm.answer(_SYSTEM_PROMPT, _user_prompt(used, question), max_tokens=max_tokens, control=control)
        # Only what the model actually saw: citing an excerpt it never read
        # would make the [number] references point at the wrong passage.
        return Answer(text=text, sources=used, stats=getattr(self.llm, "last_stats", None))


def _user_prompt(retrieved: list, question: str) -> str:
    excerpts = "\n\n".join(
        f"[{i + 1}] (from {r.chunk.source})\n{r.chunk.text}" for i, r in enumerate(retrieved)
    )
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def fit_excerpts(llm, retrieved: list, question: str, max_tokens: int) -> list:
    """The best-ranked excerpts that fit the model's window on this device.

    Dropping the weakest match beats refusing the question: the NPU runs an
    LLM compiled for a fixed prompt length, and a few long passages can pass
    it. The best one is always kept -- if even that doesn't fit, the model
    says so plainly (PromptTooLong)."""
    budget = llm.prompt_budget(max_tokens) - llm.count_tokens(_SYSTEM_PROMPT) - TEMPLATE_TOKENS
    used = list(retrieved)
    while len(used) > 1 and llm.count_tokens(_user_prompt(used, question)) > budget:
        used.pop()
    return used
