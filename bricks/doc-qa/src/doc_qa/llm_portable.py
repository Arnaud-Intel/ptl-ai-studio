"""Portable LLM backend: a GGUF chat model via llama.cpp (CPU)."""
from __future__ import annotations

import time

from pantherlake_ai_core.model_cache import resolve_gguf
from pantherlake_ai_core.types import GenerationControl, GenerationStats

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

    def answer(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 512, control: GenerationControl | None = None
    ) -> str:
        """`control`, if given, receives each piece of the answer as it is
        written and can stop it part-way (see GenerationControl)."""
        needed = self.count_tokens(system_prompt) + self.count_tokens(user_prompt) + TEMPLATE_TOKENS
        budget = self.prompt_budget(max_tokens)
        if needed > budget:
            raise PromptTooLong(needed, budget, "cpu")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        started = time.perf_counter()
        if control is not None:
            return self._answer_streaming(messages, max_tokens, control, started)
        response = self.model.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.2)
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

    def _answer_streaming(self, messages: list, max_tokens: int, control: GenerationControl, started: float) -> str:
        pieces: list[str] = []
        first: float | None = None
        cancelled = False
        stream = self.model.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.2, stream=True)
        for chunk in stream:
            piece = chunk["choices"][0]["delta"].get("content")
            if not piece:
                continue
            first = first or time.perf_counter()
            pieces.append(piece)
            if control.on_tokens is not None:
                control.on_tokens(1)  # llama.cpp streams a token at a time
            if control.on_text is not None:
                control.on_text(piece)
            if control.should_stop is not None and control.should_stop():
                cancelled = True
                break  # abandoning the iterator is how llama.cpp is told to stop
        seconds = time.perf_counter() - started
        tokens = len(pieces)  # a stream carries no usage block; a piece is a token, near enough
        self.last_stats = (
            GenerationStats(
                "cpu",
                tokens=tokens,
                seconds=round(seconds, 3),
                tokens_per_second=round(tokens / seconds, 1),
                first_token_seconds=round(first - started, 3) if first else None,
                cancelled=cancelled,
            )
            if tokens and seconds > 0
            else None
        )
        return "".join(pieces).strip()
