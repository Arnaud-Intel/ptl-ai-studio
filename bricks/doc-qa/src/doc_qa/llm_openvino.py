"""OpenVINO LLM backend: runs a chat model on Intel CPU/iGPU/NPU via
`openvino_genai.LLMPipeline`.

Requires this brick's `openvino` extra. Downloads Intel's pre-converted
Qwen2.5 chat model from Hugging Face by default; pass `model_dir` to use
one you converted yourself, or `model_repo` to download a different HF
repo instead of this class's default (e.g. a brick composing this one that
needs a different model for its task -- see meeting-notes/code-review-assist).
"""
from __future__ import annotations

_DEFAULT_REPO = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"


def _resolve_model_dir(model_dir: str | None, model_repo: str | None = None) -> str:
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download

    return snapshot_download(model_repo or _DEFAULT_REPO)


class OpenVINOLLM:
    def __init__(self, device: str = "AUTO", model_dir: str | None = None, model_repo: str | None = None):
        import openvino_genai as ov_genai

        self._ov_genai = ov_genai
        resolved_dir = _resolve_model_dir(model_dir, model_repo)
        ov_config = {"CACHE_DIR": "ov_cache"} if device == "NPU" or "GPU" in device else {}
        self.pipeline = ov_genai.LLMPipeline(resolved_dir, device, **ov_config)

    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        history = self._ov_genai.ChatHistory(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        result = self.pipeline.generate(history, max_new_tokens=max_tokens, temperature=0.2)
        return result.texts[0].strip()
