"""The image model: draws the pictures a plan asks for, through
`openvino_genai.Text2ImagePipeline`. The only model in this brick that no
other brick already wraps -- there is no portable engine for it.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from pantherlake_ai_core import flux2_klein, npu
from pantherlake_ai_core.engine import ov_config_for

# FLUX.2 klein 4B: a picture in four steps instead of twenty-five or more,
# which is what makes six pictures for one page affordable, and Apache-2.0.
# Where the build comes from, and how it is made one the pipeline opens, is
# in pantherlake_ai_core.flux2_klein -- Portrait Studio draws with the same.
#
# Measured on the XPS 14 (2026-10-12), the eighteen pictures the planner asks
# for on three of the sample requests:
#   integrated GPU: loads in 7 s; 1024x576 in 4.1 s, 768x512 in 2.7 s,
#                   640x768 in 3.4 s; the process holds 7.7 GB
#   Arc Pro B60:    loads in 16 s; 1024x576 in 1.8 s, 768x512 in 1.3 s
#
# It took the place of FLUX.1-schnell that day. The same eighteen pictures by
# schnell: loads in 18 s and 35 s; 7.3, 5.0 and 6.2 s on the integrated GPU,
# 3.3 and 2.3 s on the B60; 13 GB held; 9.2 GB on disk against 4.6. Side by
# side the pictures are a match: klein keeps to a number said ("a single
# bean", "three people") where schnell drew a cupful and four, schnell's
# bicycles are a little finer; both letter a newspaper or a bag in gibberish.
#
# Before schnell, the lighter candidate was LCM Dreamshaper v7 (2.3 GB, 0.3
# to 0.8 s a picture): soft, painterly pictures that follow the description
# loosely -- asked for croissants on a counter it drew a room with bread-like
# shapes.
_STEPS = 4
_JPEG_QUALITY = 88


class ImageMaker:
    def __init__(
        self,
        device: str = "GPU",
        model_dir: str | None = None,
        on_downloading: Callable[[], None] | None = None,
    ):
        if npu.is_npu(device):
            raise ValueError("The image model does not run on the NPU: choose a GPU, or the CPU.")
        import openvino_genai as ov_genai

        self._model_dir = flux2_klein.resolve(model_dir, on_downloading=on_downloading)
        self.device = device
        self.pipeline = ov_genai.Text2ImagePipeline(self._model_dir, device, **ov_config_for(device))
        # About the last picture drawn: how many denoising steps, and what
        # they took by the runtime's own count.
        self.last_steps = 0
        self.last_denoise_seconds: float | None = None

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
        self.last_steps, self.last_denoise_seconds = _STEPS, self._denoise_seconds()
        image = Image.fromarray(np.array(tensor.data[0], dtype=np.uint8))
        if path.suffix.lower() in (".jpg", ".jpeg"):
            image.save(path, quality=_JPEG_QUALITY, optimize=True)
        else:
            image.save(path)
        return time.perf_counter() - started

    def _denoise_seconds(self) -> float | None:
        """What the last picture's denoising steps took, from the pipeline's
        own counters (one duration per step, in microseconds). None if they
        cannot be read: the figure is for show, and never costs a picture.

        Measured on the XPS 14's integrated GPU (2026-10-08, a machine
        running slow that evening): of 12.3 s for a 768x512 picture, 9.2 s
        were the four steps, 0.9 s reading the description and 2.1 s
        decoding the picture -- so "steps per second" and "pictures per
        minute" are two different figures, and both are reported."""
        try:
            durations = self.pipeline.get_performance_metrics().raw_metrics.iteration_durations
            return sum(durations) / 1e6 if durations else None
        except Exception:
            return None
