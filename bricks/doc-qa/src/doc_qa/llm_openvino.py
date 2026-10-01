"""OpenVINO LLM backend: runs a chat model on Intel CPU/iGPU/NPU via
`openvino_genai.LLMPipeline`.

Requires this brick's `openvino` extra. Downloads Intel's pre-converted
Qwen2.5 chat model from Hugging Face by default; pass `model_dir` to use
one you converted yourself, or `model_repo` to download a different HF
repo instead of this class's default (e.g. a brick composing this one that
needs a different model for its task -- see meeting-notes/code-review-assist).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pantherlake_ai_core.engine import ov_config_for
from pantherlake_ai_core.model_cache import resolve_snapshot
from pantherlake_ai_core.types import GenerationStats

from .engine_factory import TEMPLATE_TOKENS, PromptTooLong

_DEFAULT_REPO = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"

# The NPU runs an LLM from a graph compiled for a fixed window: at most
# MAX_PROMPT_LEN tokens of prompt, plus MIN_RESPONSE_LEN guaranteed for the
# answer. OpenVINO's defaults are 1024 and 128 -- a ten-minute meeting
# transcript overflows the first ("1599 is passed", 2026-09-25), and 128
# would cut meeting notes, which ask for up to 600 tokens, off mid-list.
#
# Measured on the XPS 14's NPU, Qwen2.5-1.5B, a 1,576-token transcript:
#   MAX_PROMPT_LEN 2048: first compile 44 s, 0.85 s to first token, 54 tokens/s
#   MAX_PROMPT_LEN 4096: first compile 38 s, 0.87 s to first token, 52 tokens/s
#   MAX_PROMPT_LEN 8192: first compile 131 s, 0.91 s to first token, 52 tokens/s
# Every one loads from the compile cache in about 2 s afterwards. 4096 runs
# as fast as the others and compiles in a third of 8192's time; a meeting
# longer than that is summarised in parts (meeting_notes.session).
NPU_MAX_PROMPT_LEN = 4096
NPU_MIN_RESPONSE_LEN = 1024
_DEFAULT_CONTEXT = 32768  # when a model's config.json doesn't say


def pipeline_config(device: str) -> dict:
    """LLMPipeline properties for `device`: the shared compile cache, and on
    the NPU the prompt and response window (the defaults are far too small)."""
    config = dict(ov_config_for(device))
    if device.upper().startswith("NPU"):
        config.update(MAX_PROMPT_LEN=NPU_MAX_PROMPT_LEN, MIN_RESPONSE_LEN=NPU_MIN_RESPONSE_LEN)
    return config


def _context_length(model_dir: str) -> int:
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        return int(config.get("max_position_embeddings") or _DEFAULT_CONTEXT)
    except (OSError, ValueError):
        return _DEFAULT_CONTEXT


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
        self.pipeline = ov_genai.LLMPipeline(resolved_dir, device, **pipeline_config(device))
        self._tokenizer = self.pipeline.get_tokenizer()
        self._on_npu = device.upper().startswith("NPU")
        self._context = _context_length(resolved_dir)
        self.device = device
        self.last_stats: GenerationStats | None = None

    def count_tokens(self, text: str) -> int:
        return int(self._tokenizer.encode(text).input_ids.shape[-1])

    def prompt_budget(self, max_tokens: int) -> int:
        """Prompt tokens (system + user, template included) that fit while
        leaving `max_tokens` for the answer."""
        if self._on_npu:
            return NPU_MAX_PROMPT_LEN  # MIN_RESPONSE_LEN keeps the answer's room apart
        return self._context - max_tokens

    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        needed = self.count_tokens(system_prompt) + self.count_tokens(user_prompt) + TEMPLATE_TOKENS
        budget = self.prompt_budget(max_tokens)
        if needed > budget:
            raise PromptTooLong(needed, budget, self.device)
        history = self._ov_genai.ChatHistory(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        result = self.pipeline.generate(history, max_new_tokens=max_tokens, temperature=0.2)
        self.last_stats = GenerationStats.from_openvino(result, self.device)
        return result.texts[0].strip()
