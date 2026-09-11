"""Portable LLM backend: a GGUF chat model via llama.cpp (CPU)."""
from __future__ import annotations

import time

from pantherlake_ai_core.types import GenerationStats

_DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
_DEFAULT_FILENAME = "*q4_k_m.gguf"


class PortableLLM:
    def __init__(
        self,
        repo_id: str = _DEFAULT_REPO,
        filename: str = _DEFAULT_FILENAME,
        n_ctx: int = 4096,
    ):
        from llama_cpp import Llama

        self.model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            verbose=False,
        )
        self.last_stats: GenerationStats | None = None

    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        started = time.perf_counter()
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        # Wall clock, prompt processing included -- llama.cpp's reply has the
        # token count but not the timings -- so this reads a little lower
        # than the OpenVINO engine's decode rate would for the same model.
        seconds = time.perf_counter() - started
        tokens = int((response.get("usage") or {}).get("completion_tokens") or 0)
        self.last_stats = (
            GenerationStats("cpu", tokens=tokens, seconds=round(seconds, 3), tokens_per_second=round(tokens / seconds, 1))
            if tokens and seconds > 0
            else None
        )
        return response["choices"][0]["message"]["content"].strip()
