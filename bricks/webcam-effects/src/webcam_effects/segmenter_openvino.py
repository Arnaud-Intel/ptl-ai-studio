"""OpenVINO segmentation backend: the exact same selfie-segmentation ONNX
model as the portable engine, read directly by OpenVINO's ONNX frontend
(no separate IR conversion step) and compiled for Intel CPU/iGPU/NPU.

The model's ONNX export has a dynamic batch dimension, which OpenVINO's
shape inference (and the NPU compiler specifically) can't resolve on its
own -- reshaping to a static batch size of 1 before compiling is required,
not just a nice-to-have (verified: compiling without this raises an LLVM
shape-inference error rather than a clean Python exception).
"""
from __future__ import annotations

import numpy as np

from . import matte


class OpenVINOSegmenter:
    def __init__(
        self,
        device: str = "AUTO",
        repo_id: str = matte.DEFAULT_REPO,
        filename: str = matte.DEFAULT_FILENAME,
        model_path: str | None = None,
    ):
        from openvino import Core

        resolved_path = matte.resolve_model_path(repo_id, filename, model_path)
        core = Core()
        model = core.read_model(resolved_path)
        model.reshape({model.inputs[0].get_any_name(): [1, 3, matte.INPUT_SIZE, matte.INPUT_SIZE]})

        ov_config = {"CACHE_DIR": "ov_cache"} if device == "NPU" or "GPU" in device else {}
        compiled = core.compile_model(model, device_name=device, config=ov_config)
        self.infer_request = compiled.create_infer_request()

    def segment(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        tensor = matte.preprocess(frame)
        self.infer_request.infer(inputs=[tensor])
        alpha = self.infer_request.get_output_tensor().data
        return matte.postprocess(alpha, width, height)
