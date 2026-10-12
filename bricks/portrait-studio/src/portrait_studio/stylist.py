"""The image model at work: a photo and a style in, the same person in that
style out, through `openvino_genai.Image2ImagePipeline`.

For FLUX.2 klein "image to image" is not the old trick of adding noise to
the photo and cleaning it up again (which keeps the photo or loses the
person, with nothing in between): the photo is given to the model whole,
beside the picture it is drawing, as something to look at. That is why the
person stays the person while everything about how they are drawn changes.

Measured on the XPS 14's integrated GPU (Arc B390, OpenVINO 2026.3,
2026-10-12): the model loads in 7 to 9 s and a 768 x 768 picture takes 8 to
11 s -- four steps of about 2 s each, the rest reading the instruction and
decoding the picture. Half as many pixels a side (512) takes 3 to 4 s. The
NPU does not take this build: its compiler ends the process (OpenVINO
2026.4.1) or does not finish (2026.3).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from pantherlake_ai_core import flux2_klein, npu
from pantherlake_ai_core.engine import ov_config_for

from .photo import DEFAULT_SIZE, square
from .styles import Style

# The model is a distilled one: four steps is what it was made for, and more
# do not make a better picture.
STEPS = 4


@dataclass
class Restyled:
    style: str
    image: object  # a PIL image; None if it was stopped before the picture was whole
    size: int
    seed: int
    seconds: float
    steps: int = STEPS
    # What each step took by the runtime's own count, in seconds; empty if
    # it could not be read.
    step_seconds: list[float] = field(default_factory=list)
    cancelled: bool = False

    @property
    def steps_per_second(self) -> float | None:
        total = sum(self.step_seconds)
        return len(self.step_seconds) / total if total else None


class Stylist:
    def __init__(
        self,
        device: str = "GPU",
        model_dir: str | None = None,
        on_downloading: Callable[[], None] | None = None,
        on_preparing: Callable[[], None] | None = None,
    ):
        if npu.is_npu(device):
            raise ValueError("The image model does not run on the NPU: choose a GPU, or the CPU.")
        import openvino_genai as ov_genai

        self.device = device
        self.model_dir = flux2_klein.resolve(model_dir, on_downloading=on_downloading, on_preparing=on_preparing)
        self.pipeline = ov_genai.Image2ImagePipeline(self.model_dir, device, **ov_config_for(device))

    def restyle(
        self,
        photo,
        style: Style,
        *,
        size: int = DEFAULT_SIZE,
        seed: int = 0,
        on_step: Callable[[int, int], None] | None = None,
        stopped: Callable[[], bool] | None = None,
    ) -> Restyled:
        """Draw the person in `photo` (a PIL image, any shape) in `style`.
        `seed` picks the picture among the many the style allows: the same
        photo, style and seed give the same picture again. `on_step(done,
        of)` is called as each step ends; `stopped()` is asked then, and
        ends the drawing if it says so."""
        import numpy as np
        import openvino as ov
        from PIL import Image

        source = np.asarray(square(photo, size), dtype=np.uint8)
        cancelled = False

        def after_step(step: int, steps: int, _latent) -> bool:
            nonlocal cancelled
            if on_step is not None:
                on_step(step + 1, steps)
            if stopped is not None and stopped():
                cancelled = True
            return cancelled

        started = time.perf_counter()
        tensor = self.pipeline.generate(
            style.prompt, ov.Tensor(source[None]), width=size, height=size, num_inference_steps=STEPS, rng_seed=seed,
            callback=after_step,
        )
        seconds = time.perf_counter() - started
        image = None if cancelled else Image.fromarray(np.array(tensor.data[0], dtype=np.uint8))
        return Restyled(style=style.id, image=image, size=size, seed=seed, seconds=seconds,
                        step_seconds=self._step_seconds(), cancelled=cancelled)

    def _step_seconds(self) -> list[float]:
        """The last picture's steps, from the pipeline's own counters (one
        duration a step, in microseconds). Empty if they cannot be read: the
        figure is for show, and never costs a picture."""
        try:
            return [duration / 1e6 for duration in self.pipeline.get_performance_metrics().raw_metrics.iteration_durations]
        except Exception:
            return []
