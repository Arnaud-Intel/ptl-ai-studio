"""OpenVINO text-extraction backend: a vision-language model reads the
image, via `openvino_genai.VLMPipeline`, targeting Intel CPU/iGPU/NPU.

Unlike the portable engine's dedicated OCR models, this is a full language
model looking at the image -- slower and much heavier to download, but it
can translate the text it reads in the same pass (no separate MT step),
and it's the only engine here that can actually target the NPU.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from pantherlake_ai_core.engine import ov_config_for
from pantherlake_ai_core.model_cache import resolve_snapshot
from pantherlake_ai_core.types import GenerationStats

# openvino itself is imported where it's used, not here: this brick's
# `openvino` extra is optional, and importing this module must not require
# it. resolve_device() below is plain Python that callers (and the tests)
# need whether or not the extra is installed.

from .types import ExtractionResult

_DEFAULT_REPO = "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov"

_EXTRACT_PROMPT = (
    "Read all text visible in this image, exactly as it appears. Output "
    "only the text you see, preserving line breaks where natural. If "
    "there is no legible text, output exactly: No text detected."
)
_TRANSLATE_PROMPT = (
    "Read all text visible in this image, then translate it to English. "
    "Output only the English translation, nothing else. If there is no "
    "legible text, output exactly: No text detected."
)


def resolve_device(device: str) -> str:
    """Turn "AUTO" into a device this model can actually run on.

    OpenVINO's AUTO plugin loads this VLM happily and then fails on every
    single generate() with "Exception from src/core/src/shape_util.cpp:66:
    Accessing out-of-range dimension". Reproduced on openvino-genai
    2026.3.0 with an image that succeeds byte-for-byte on GPU.0 and GPU.1,
    so it is AUTO itself, not the image, the size, or the tensor rank.
    AUTO is the launcher's and the CLI's default device, which made the
    OpenVINO engine of this brick fail out of the box.

    A GPU is preferred -- core.engine.preferred_large_model_device's pick:
    the discrete one if the machine has one, else the integrated one, since
    this 7B model's ~6 GB sit fine in an iGPU's shared memory -- else CPU.
    The NPU is never chosen automatically: this model's NPU compile fails
    on this hardware, which is documented in this brick's README.
    """
    if device.upper() != "AUTO":
        return device
    from pantherlake_ai_core.engine import preferred_large_model_device

    pick = preferred_large_model_device()
    return "CPU" if pick == "AUTO" else pick


class OpenVINOExtractor:
    def __init__(self, device: str = "AUTO", model_dir: str | None = None, on_downloading: Callable[[], None] | None = None):
        import openvino_genai as ov_genai

        resolved_dir = resolve_snapshot(_DEFAULT_REPO, local_dir=model_dir, on_downloading=on_downloading)
        # The device actually used, which a caller can read back to report
        # honestly (the launcher labels its telemetry gauge from it).
        self.device = resolve_device(device)
        self.pipeline = ov_genai.VLMPipeline(resolved_dir, self.device, **ov_config_for(self.device))

    def extract(self, image: np.ndarray, translate: bool = False) -> ExtractionResult:
        import openvino as ov

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = ov.Tensor(np.ascontiguousarray(rgb))

        prompt = _TRANSLATE_PROMPT if translate else _EXTRACT_PROMPT
        result = self.pipeline.generate(prompt, images=[tensor], max_new_tokens=512)
        text = result.texts[0].strip()
        stats = GenerationStats.from_openvino(result, self.device)

        if translate:
            return ExtractionResult(text="", regions=[], translated_text=text, stats=stats)
        return ExtractionResult(text=text, regions=[], stats=stats)
