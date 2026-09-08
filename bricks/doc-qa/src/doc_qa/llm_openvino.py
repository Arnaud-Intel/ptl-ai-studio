"""OpenVINO LLM backend: runs a chat model on Intel CPU/iGPU/NPU via
`openvino_genai.LLMPipeline`.

Requires this brick's `openvino` extra. Downloads Intel's pre-converted
Qwen2.5 chat model from Hugging Face by default; pass `model_dir` to use
one you converted yourself, or `model_repo` to download a different HF
repo instead of this class's default (e.g. a brick composing this one that
needs a different model for its task -- see meeting-notes/code-review-assist).
"""
from __future__ import annotations

from typing import Callable

from pantherlake_ai_core.engine import ov_config_for
from pantherlake_ai_core.model_cache import resolve_snapshot

_DEFAULT_REPO = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"


class OpenVINOLLM:
    def __init__(
        self,
        device: str = "AUTO",
        model_dir: str | None = None,
        model_repo: str | None = None,
        on_downloading: Callable[[], None] | None = None,
    ):
        import openvino_genai as ov_genai

        self._ov_genai = ov_genai
        resolved_dir = resolve_snapshot(model_repo or _DEFAULT_REPO, local_dir=model_dir, on_downloading=on_downloading)
        self.pipeline = ov_genai.LLMPipeline(resolved_dir, device, **ov_config_for(device))

    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        history = self._ov_genai.ChatHistory(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        result = self.pipeline.generate(history, max_new_tokens=max_tokens, temperature=0.2)
        return result.texts[0].strip()
