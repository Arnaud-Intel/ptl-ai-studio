"""Picks an embedder + LLM backend (engine) so the rest of the brick
doesn't need to know which one it's talking to.
"""
from __future__ import annotations

from typing import Callable, Protocol

from pantherlake_ai_core.engine import Engine


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LLM(Protocol):
    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str: ...


def create_embedder(
    engine: Engine,
    *,
    device: str = "AUTO",
    model_dir: str | None = None,
    on_downloading: Callable[[], None] | None = None,
) -> Embedder:
    """`on_downloading`, if given, fires before an openvino model that isn't
    already cached locally starts downloading (portable's GGUF download
    isn't covered)."""
    if engine == Engine.PORTABLE:
        from .embedder_portable import PortableEmbedder

        return PortableEmbedder()

    if engine == Engine.OPENVINO:
        from .embedder_openvino import OpenVINOEmbedder

        return OpenVINOEmbedder(device=device, model_dir=model_dir, on_downloading=on_downloading)

    raise ValueError(f"Unknown engine '{engine}'.")


def create_llm(
    engine: Engine,
    *,
    device: str = "AUTO",
    model_dir: str | None = None,
    model_repo: str | None = None,
    n_ctx: int | None = None,
    on_downloading: Callable[[], None] | None = None,
) -> LLM:
    """`on_downloading`, if given, fires before an openvino model that isn't
    already cached locally starts downloading (portable's GGUF download
    isn't covered)."""
    if engine == Engine.PORTABLE:
        from .llm_portable import PortableLLM

        kwargs = {}
        if model_repo:
            kwargs["repo_id"] = model_repo
        if n_ctx:
            kwargs["n_ctx"] = n_ctx
        return PortableLLM(**kwargs)

    if engine == Engine.OPENVINO:
        from .llm_openvino import OpenVINOLLM

        return OpenVINOLLM(device=device, model_dir=model_dir, model_repo=model_repo, on_downloading=on_downloading)

    raise ValueError(f"Unknown engine '{engine}'.")
