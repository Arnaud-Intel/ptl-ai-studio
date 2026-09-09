"""Picks a voice-cloning backend so the rest of the brick doesn't need to
know which one it's talking to. Every cloner exposes `.enroll(path) ->
voice` and `.synthesize(text, voice, ...) -> (audio, sample_rate)`.

Two axes now, not one:

- **model** -- *which* model. `chatterbox` (default) clones a real voice
  properly; `openvoice` is the older tone-color path, kept because it is
  the one that runs on the NPU and iGPU.
- **engine** -- *what executes it*. Only OpenVoice has two; Chatterbox is
  CPU-only, for the reasons in `chatterbox_model`'s docstring.
"""
from __future__ import annotations

from typing import Callable, Protocol

from pantherlake_ai_core.engine import Engine

CHATTERBOX = "chatterbox"
OPENVOICE = "openvoice"
MODELS = (CHATTERBOX, OPENVOICE)
DEFAULT_MODEL = CHATTERBOX

# Which engines each model can actually run on, so the UI and the CLI can
# both refuse an impossible pairing up front with a reason, rather than
# failing somewhere inside a compile.
MODEL_ENGINES = {
    CHATTERBOX: (Engine.PORTABLE,),
    OPENVOICE: (Engine.PORTABLE, Engine.OPENVINO),
}


class Cloner(Protocol):
    supports_styles: bool

    def enroll(self, reference_audio_path: str): ...
    def synthesize(self, text: str, target_se): ...


def create_cloner(
    engine: Engine,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "CPU",
    model_path: str | None = None,
    on_downloading: Callable[[], None] | None = None,
) -> Cloner:
    """`on_downloading`, if given, fires before checkpoints that aren't
    cached locally yet get fetched -- and not at all when they're there."""
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choices: {', '.join(MODELS)}")
    if engine not in MODEL_ENGINES[model]:
        allowed = ", ".join(e.value for e in MODEL_ENGINES[model])
        raise ValueError(
            f"The '{model}' model doesn't run on the '{engine.value}' engine (it supports: {allowed}). "
            f"Use --model openvoice for an engine choice, or --engine portable for this model."
        )

    if model == CHATTERBOX:
        from .cloner_chatterbox import ChatterboxCloner

        return ChatterboxCloner(model_path=model_path, on_downloading=on_downloading)

    if engine == Engine.PORTABLE:
        from .cloner_portable import PortableCloner

        return PortableCloner(model_path=model_path, on_downloading=on_downloading)

    from .cloner_openvino import OpenVINOCloner

    return OpenVINOCloner(device=device, model_path=model_path, on_downloading=on_downloading)
