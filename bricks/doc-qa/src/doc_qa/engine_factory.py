"""Picks an embedder + LLM backend (engine) so the rest of the brick
doesn't need to know which one it's talking to.
"""
from __future__ import annotations

from typing import Protocol

from pantherlake_ai_core.engine import Engine


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LLM(Protocol):
    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str: ...


def create_embedder(engine: Engine, *, device: str = "AUTO", model_dir: str | None = None) -> Embedder:
    if engine == Engine.PORTABLE:
        from .embedder_portable import PortableEmbedder

        return PortableEmbedder()

    if engine == Engine.OPENVINO:
        from .embedder_openvino import OpenVINOEmbedder

        return OpenVINOEmbedder(device=device, model_dir=model_dir)

    raise ValueError(f"Unknown engine '{engine}'.")


def create_llm(engine: Engine, *, device: str = "AUTO", model_dir: str | None = None) -> LLM:
    if engine == Engine.PORTABLE:
        from .llm_portable import PortableLLM

        return PortableLLM()

    if engine == Engine.OPENVINO:
        from .llm_openvino import OpenVINOLLM

        return OpenVINOLLM(device=device, model_dir=model_dir)

    raise ValueError(f"Unknown engine '{engine}'.")
