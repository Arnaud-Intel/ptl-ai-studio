"""OpenVINO embedding backend: runs an embedding model on Intel CPU/iGPU/NPU
via `openvino_genai.TextEmbeddingPipeline`.

Requires this brick's `openvino` extra. Downloads Intel's pre-converted
Qwen3 embedding model from Hugging Face by default; pass `model_dir` to use
one you converted yourself.
"""
from __future__ import annotations

from typing import Callable

from pantherlake_ai_core.engine import ov_config_for
from pantherlake_ai_core.model_cache import resolve_snapshot

_DEFAULT_REPO = "OpenVINO/Qwen3-Embedding-0.6B-int8-ov"


class OpenVINOEmbedder:
    def __init__(self, device: str = "AUTO", model_dir: str | None = None, on_downloading: Callable[[], None] | None = None):
        import openvino_genai as ov_genai

        resolved_dir = resolve_snapshot(_DEFAULT_REPO, local_dir=model_dir, on_downloading=on_downloading)
        self.pipeline = ov_genai.TextEmbeddingPipeline(resolved_dir, device, **ov_config_for(device))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.pipeline.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.pipeline.embed_query(text)
