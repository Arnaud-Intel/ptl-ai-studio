"""OpenVINO detection backend: YOLO11s via Intel's `openvino-model-api`,
targeting Intel CPU/iGPU/NPU.

Uses Intel's own `model_api` package (as recommended by the model card)
instead of hand-rolled YOLO anchor-decoding + NMS -- it's purpose-built for
exactly these OpenVINO Model Zoo detection models and already returns
plain pixel-space boxes with resolved label names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .types import Detection

# YOLO11**s**, not the nano it used to be. Measured on 30 identical frames
# of a rainy night street: nano found 7.8 relevant objects per frame at the
# 0.5 threshold, small found 13.6 -- and on this machine's iGPU it costs
# nothing for them (7.9ms vs 8.3ms), on the NPU about 4ms, and on the CPU
# 20.7ms against 13.6ms, which is still 48fps. Nano was quietly the reason
# cars in plain sight went uncounted.
_DEFAULT_REPO = "OpenVINO/YOLO11s-int8-ov"
# model_api fetches just the IR pair, never the repo's full snapshot -- so
# "is it cached" has to be asked about these two files, not the repo.
_DEFAULT_FILES = ("yolo11s.xml", "yolo11s.bin")


def _local_ir(model_dir: str) -> Path:
    """`--model-path` for this engine: an OpenVINO IR -- either the .xml
    itself or a folder holding exactly one."""
    path = Path(model_dir)
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.xml"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one OpenVINO .xml model under {model_dir}, found {len(candidates)}."
        )
    return candidates[0]


class OpenVINODetector:
    def __init__(
        self,
        device: str = "AUTO",
        model_dir: str | None = None,
        confidence_threshold: float = 0.5,
        on_downloading: Callable[[], None] | None = None,
    ):
        from model_api.adapters.openvino_adapter import OpenvinoAdapter, create_core, get_user_config
        from model_api.models import Model
        from pantherlake_ai_core.engine import ov_config_for
        from pantherlake_ai_core.model_cache import is_file_cached

        if model_dir:
            model_path = _local_ir(model_dir)
        else:
            from model_api.utils.hf_hub_helper import download_from_hf

            cached = all(is_file_cached(_DEFAULT_REPO, name) for name in _DEFAULT_FILES)
            if on_downloading is not None and not cached:
                on_downloading()
            model_path = download_from_hf(repo_id=_DEFAULT_REPO)

        # Assembled by hand rather than via Model.from_pretrained(): that
        # path has no way to pass OpenVINO's compile config (its `cache_dir`
        # is the Hub's), and handing it a Core trips an UnboundLocalError in
        # create_model. This is the same adapter it would build, plus
        # whatever `ov_config_for` decides this device should get -- for
        # this model that means the NPU's compiled-model cache and, very
        # deliberately, no GPU cache: a cached GPU YOLO11n returns one
        # garbage box instead of a scene full of them (see ov_config_for).
        adapter = OpenvinoAdapter(
            core=create_core(),
            model=str(model_path),
            device=device,
            plugin_config={**get_user_config(device, "1", None), **ov_config_for(device)},
        )
        self.model = Model.create_model(adapter)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model(frame)
        detections = []
        for box, score, name in zip(result.bboxes, result.scores, result.label_names):
            if score < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            detections.append(Detection(label=name, confidence=float(score), box=(x1, y1, x2, y2)))
        return detections
