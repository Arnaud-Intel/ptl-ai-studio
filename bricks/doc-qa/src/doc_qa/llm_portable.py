"""Portable LLM backend: a GGUF chat model via llama.cpp (CPU)."""
from __future__ import annotations

import time

from pantherlake_ai_core.model_cache import resolve_gguf
from pantherlake_ai_core.types import GenerationStats

from .engine_factory import TEMPLATE_TOKENS, PromptTooLong

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

        # Not Llama.from_pretrained: it resolves the filename pattern by
        # listing the repo over the network, so a model already on disk
        # still fails offline (see model_cache.resolve_gguf).
        self.model = Llama(
            model_path=resolve_gguf(repo_id, filename),
            n_ctx=n_ctx,
            verbose=False,
        )
        self.n_ctx = n_ctx  # prompt and answer together
        self.last_stats: GenerationStats | None = None

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenize(text.encode("utf-8"), add_bos=False, special=True))

    def prompt_budget(self, max_tokens: int) -> int:
        return self.n_ctx - max_tokens

    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        needed = self.count_tokens(system_prompt) + self.count_tokens(user_prompt) + TEMPLATE_TOKENS
        budget = self.prompt_budget(max_tokens)
        if needed > budget:
            raise PromptTooLong(needed, budget, "cpu")
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
