"""OpenVINO detection backend: YOLO11n via Intel's `openvino-model-api`,
targeting Intel CPU/iGPU/NPU.

Uses Intel's own `model_api` package (as recommended by the model card)
instead of hand-rolled YOLO anchor-decoding + NMS -- it's purpose-built for
exactly these OpenVINO Model Zoo detection models and already returns
plain pixel-space boxes with resolved label names.
"""
from __future__ import annotations

import numpy as np

from .types import Detection

_DEFAULT_REPO = "OpenVINO/YOLO11n-int8-ov"


class OpenVINODetector:
    def __init__(self, device: str = "AUTO", model_dir: str | None = None, confidence_threshold: float = 0.5):
        from model_api.models import Model

        self.model = Model.from_pretrained(model_dir or _DEFAULT_REPO, device=device)
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
