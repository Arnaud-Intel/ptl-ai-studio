"""Picks a translation backend (engine) so the CLI doesn't need to know
which one it's talking to -- both expose `.translate(audio) -> TranslationResult | None`.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import TranslationResult


class Translator(Protocol):
    def translate(self, audio: np.ndarray) -> TranslationResult | None: ...


def create_translator(
    engine: Engine,
    model_size: str,
    device: str,
    compute_type: str = "auto",
    ov_model_dir: str | None = None,
    task: str = "translate",
) -> Translator:
    """`task="translate"` (the default) always outputs English text,
    whatever language is spoken. Pass `task="transcribe"` for same-language
    speech-to-text instead -- e.g. voice-assistant reuses this factory that
    way, since a voice assistant should hear you in the language you spoke,
    not have it silently translated."""
    if engine == Engine.PORTABLE:
        from .transcriber_portable import PortableTranslator

        return PortableTranslator(model_size=model_size, device=device, compute_type=compute_type, task=task)

    if engine == Engine.OPENVINO:
        from .transcriber_openvino import OpenVINOTranslator

        return OpenVINOTranslator(model_size=model_size, device=device, model_dir=ov_model_dir, task=task)

    raise ValueError(f"Unknown engine '{engine}'.")
