"""OpenVINO embedding backend: runs an embedding model on Intel CPU/iGPU/NPU
via `openvino_genai.TextEmbeddingPipeline`.

Requires this brick's `openvino` extra. Downloads Intel's pre-converted
Qwen3 embedding model from Hugging Face by default; pass `model_dir` to use
one you converted yourself.
"""
from __future__ import annotations

from typing import Callable

_DEFAULT_REPO = "OpenVINO/Qwen3-Embedding-0.6B-int8-ov"


def _resolve_model_dir(model_dir: str | None, on_downloading: Callable[[], None] | None = None) -> str:
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download
    from pantherlake_ai_core.model_cache import is_repo_cached

    if on_downloading is not None and not is_repo_cached(_DEFAULT_REPO):
        on_downloading()
    return snapshot_download(_DEFAULT_REPO)


class OpenVINOEmbedder:
    def __init__(self, device: str = "AUTO", model_dir: str | None = None, on_downloading: Callable[[], None] | None = None):
        import openvino_genai as ov_genai

        resolved_dir = _resolve_model_dir(model_dir, on_downloading)
        ov_config = {"CACHE_DIR": "ov_cache"} if device == "NPU" or "GPU" in device else {}
        self.pipeline = ov_genai.TextEmbeddingPipeline(resolved_dir, device, **ov_config)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.pipeline.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.pipeline.embed_query(text)
