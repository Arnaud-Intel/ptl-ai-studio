"""Portable translation backend: faster-whisper (CTranslate2), CPU or CUDA.

Whisper's built-in "translate" task transcribes speech in whatever language
it detects and directly outputs English text (no separate MT step needed).
This backend runs anywhere but cannot target Intel's NPU or iGPU directly
-- see transcriber_openvino.py for that.
"""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel
from pantherlake_ai_core.types import TranslationResult


class PortableTranslator:
    """Loads a local Whisper model once and translates audio chunks to English text."""

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "auto", task: str = "translate"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.task = task

    def translate(self, audio: np.ndarray) -> TranslationResult | None:
        segments, info = self.model.transcribe(
            audio,
            task=self.task,
            beam_size=1,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            return None
        return TranslationResult(
            text=text,
            detected_language=info.language,
            language_probability=info.language_probability,
        )
