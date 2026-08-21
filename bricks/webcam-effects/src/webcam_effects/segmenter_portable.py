"""Portable segmentation backend: the shared selfie-segmentation model via
ONNX Runtime, CPU."""
from __future__ import annotations

import numpy as np

from . import matte


class PortableSegmenter:
    def __init__(self, repo_id: str = matte.DEFAULT_REPO, filename: str = matte.DEFAULT_FILENAME, model_path: str | None = None):
        import onnxruntime as ort

        resolved_path = matte.resolve_model_path(repo_id, filename, model_path)
        self.session = ort.InferenceSession(resolved_path, providers=["CPUExecutionProvider"])

    def segment(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        tensor = matte.preprocess(frame)
        (alpha,) = self.session.run(["alphas"], {"pixel_values": tensor})
        return matte.postprocess(alpha, width, height)
