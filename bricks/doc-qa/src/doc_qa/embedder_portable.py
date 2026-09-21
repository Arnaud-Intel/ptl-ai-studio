"""Portable embedding backend: a GGUF embedding model via llama.cpp (CPU).

nomic-embed-text expects its inputs prefixed with a task instruction --
"search_document: " when embedding chunks to index, "search_query: " when
embedding a question -- so retrieval quality depends on using the right one.
"""
from __future__ import annotations

from pantherlake_ai_core.model_cache import resolve_gguf

_DEFAULT_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
_DEFAULT_FILENAME = "*Q4_K_M.gguf"


class PortableEmbedder:
    def __init__(self, repo_id: str = _DEFAULT_REPO, filename: str = _DEFAULT_FILENAME, n_ctx: int = 2048):
        from llama_cpp import Llama

        # See llm_portable: from_pretrained needs the network even for a
        # model that is already cached.
        self.model = Llama(
            model_path=resolve_gguf(repo_id, filename),
            embedding=True,
            n_ctx=n_ctx,
            verbose=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.model.embed(f"search_document: {t}", normalize=True) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed(f"search_query: {text}", normalize=True)
