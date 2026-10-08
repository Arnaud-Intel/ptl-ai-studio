"""The image model: draws the pictures a plan asks for, through
`openvino_genai.Text2ImagePipeline`. The only model in this brick that no
other brick already wraps -- there is no portable engine for it.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from pantherlake_ai_core import npu
from pantherlake_ai_core.engine import ov_config_for
from pantherlake_ai_core.model_cache import resolve_snapshot

# FLUX.1-schnell: a picture in four steps instead of twenty-five or more,
# which is what makes three pictures for one page affordable, and Apache-2.0.
# Measured on the XPS 14 (2026-10-08):
#   integrated GPU: loads in 27 s; 1024x576 in 6.5-8.5 s, 768x512 in 4.5-5.3 s;
#                   about 13 GB of shared memory while loaded
#   Arc Pro B60:    loads in 41 s; 1024x576 in 3.3 s, 768x512 in 2.3 s once warm
# 9.2 GB on disk. The lighter candidate, LCM Dreamshaper v7 (2.3 GB, loads in
# 13 s, 0.3-0.8 s a picture after 6 s for the first of each size), draws
# soft, painterly pictures and follows the description loosely -- asked for
# croissants on a counter it drew a room with bread-like shapes; FLUX drew
# the croissants, and a "flat vector illustration" came out flat and vector.
DEFAULT_REPO = "OpenVINO/FLUX.1-schnell-int4-ov"
_STEPS = 4
_JPEG_QUALITY = 88


class ImageMaker:
    def __init__(
        self,
        device: str = "GPU",
        model_dir: str | None = None,
        model_repo: str | None = None,
        on_downloading: Callable[[], None] | None = None,
    ):
        if npu.is_npu(device):
            raise ValueError("The image model does not run on the NPU: choose a GPU, or the CPU.")
        import openvino_genai as ov_genai

        self._model_dir = resolve_snapshot(model_repo or DEFAULT_REPO, local_dir=model_dir, on_downloading=on_downloading)
        self.device = device
        self.pipeline = ov_genai.Text2ImagePipeline(self._model_dir, device, **ov_config_for(device))

    def draw(self, prompt: str, width: int, height: int, path: Path, seed: int = 0) -> float:
        """Draw `prompt` at `width` x `height` into `path`. Returns the
        seconds it took. A `.jpg` path is written as a JPEG: the pictures end
        up inside the page, where three PNGs weighed 5 MB and three JPEGs
        weigh a tenth of that. `seed` picks the picture among the many a
        description allows: the same description and seed give the same
        picture again, so two pictures of one page never share a seed."""
        import numpy as np
        from PIL import Image

        started = time.perf_counter()
        tensor = self.pipeline.generate(
            prompt, width=width, height=height, num_inference_steps=_STEPS, rng_seed=seed
        )
        image = Image.fromarray(np.array(tensor.data[0], dtype=np.uint8))
        if path.suffix.lower() in (".jpg", ".jpeg"):
            image.save(path, quality=_JPEG_QUALITY, optimize=True)
        else:
            image.save(path)
        return time.perf_counter() - started
