"""Portable LLM backend: a GGUF chat model via llama.cpp (CPU)."""
from __future__ import annotations

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

    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response["choices"][0]["message"]["content"].strip()
