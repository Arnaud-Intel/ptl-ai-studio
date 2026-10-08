"""Picks an embedder + LLM backend (engine) so the rest of the brick
doesn't need to know which one it's talking to.
"""
from __future__ import annotations

from typing import Callable, Protocol

from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import GenerationControl, GenerationStats


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


# Tokens a chat template wraps around the system and user messages -- a
# margin, so a prompt measured as fitting still fits once templated.
TEMPLATE_TOKENS = 32


class PromptTooLong(ValueError):
    """A prompt bigger than the model's window on this device, said plainly
    rather than as the runtime's assertion text (on the NPU that read "Check
    'data->input_ids.get_size() <= m_max_prompt_len' failed at C:\\Jenkins...")."""

    def __init__(self, needed: int, budget: int, device: str):
        self.needed, self.budget, self.device = needed, budget, device
        super().__init__(
            f"This request needs about {needed} tokens of prompt, but the model on {device} "
            f"takes at most {budget}. Shorten the input, or run it on another device."
        )


class LLM(Protocol):
    # The last answer's speed (None before the first, or if unknown) --
    # what the bricks composing this one show the audience.
    last_stats: GenerationStats | None

    # `control`, if given, sees the answer as it is written and can stop it.
    # `begin`, if given, is how the answer starts: the model carries on from
    # those words instead of choosing its own first ones (OpenVINO engine;
    # the portable one takes it and leaves the model free). `temperature`,
    # if given, replaces the 0.2 every brick draws at.
    def answer(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 512, control: GenerationControl | None = None,
        sample: bool = True, begin: str | None = None, temperature: float | None = None,
    ) -> str: ...
    # How long a prompt may be, in the model's own tokens: what lets a caller
    # split or trim its input *before* the model refuses it.
    def count_tokens(self, text: str) -> int: ...
    def prompt_budget(self, max_tokens: int) -> int: ...


def create_embedder(
    engine: Engine,
    *,
    device: str = "AUTO",
    model_dir: str | None = None,
    on_downloading: Callable[[], None] | None = None,
) -> Embedder:
    """`on_downloading`, if given, fires before an openvino model that isn't
    already cached locally starts downloading (portable's GGUF download
    isn't covered)."""
    if engine == Engine.PORTABLE:
        from .embedder_portable import PortableEmbedder

        return PortableEmbedder()

    if engine == Engine.OPENVINO:
        from .embedder_openvino import OpenVINOEmbedder

        return OpenVINOEmbedder(device=device, model_dir=model_dir, on_downloading=on_downloading)

    raise ValueError(f"Unknown engine '{engine}'.")


def create_llm(
    engine: Engine,
    *,
    device: str = "AUTO",
    model_dir: str | None = None,
    model_repo: str | None = None,
    n_ctx: int | None = None,
    on_downloading: Callable[[], None] | None = None,
) -> LLM:
    """`on_downloading`, if given, fires before an openvino model that isn't
    already cached locally starts downloading (portable's GGUF download
    isn't covered)."""
    if engine == Engine.PORTABLE:
        from .llm_portable import PortableLLM

        kwargs = {}
        if model_repo:
            kwargs["repo_id"] = model_repo
        if n_ctx:
            kwargs["n_ctx"] = n_ctx
        return PortableLLM(**kwargs)

    if engine == Engine.OPENVINO:
        from .llm_openvino import OpenVINOLLM

        return OpenVINOLLM(device=device, model_dir=model_dir, model_repo=model_repo, on_downloading=on_downloading)

    raise ValueError(f"Unknown engine '{engine}'.")
