"""OpenVINO text-extraction backend: a vision-language model reads the
image, via `openvino_genai.VLMPipeline`, targeting Intel CPU/iGPU/NPU.

Unlike the portable engine's dedicated OCR models, this is a full language
model looking at the image -- slower and much heavier to download, but it
can translate the text it reads in the same pass (no separate MT step),
and it's the only engine here that can actually target the NPU.
"""
from __future__ import annotations

import cv2
import numpy as np
import openvino as ov

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


def _resolve_model_dir(model_dir: str | None) -> str:
    if model_dir:
        return model_dir
    from huggingface_hub import snapshot_download

    return snapshot_download(_DEFAULT_REPO)


class OpenVINOExtractor:
    def __init__(self, device: str = "AUTO", model_dir: str | None = None):
        import openvino_genai as ov_genai

        resolved_dir = _resolve_model_dir(model_dir)
        ov_config = {"CACHE_DIR": "ov_cache"} if device == "NPU" or "GPU" in device else {}
        self.pipeline = ov_genai.VLMPipeline(resolved_dir, device, **ov_config)

    def extract(self, image: np.ndarray, translate: bool = False) -> ExtractionResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = ov.Tensor(np.ascontiguousarray(rgb))

        prompt = _TRANSLATE_PROMPT if translate else _EXTRACT_PROMPT
        result = self.pipeline.generate(prompt, images=[tensor], max_new_tokens=512)
        text = result.texts[0].strip()

        if translate:
            return ExtractionResult(text="", regions=[], translated_text=text)
        return ExtractionResult(text=text, regions=[])
