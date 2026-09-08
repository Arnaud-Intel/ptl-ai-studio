"""Portable detection backend: DETR (ResNet-50 backbone) via ONNX Runtime, CPU.

DETR is a set-prediction model -- unlike YOLO it needs no anchor decoding
and no non-max suppression, which keeps this backend's post-processing
small and easy to get right. The tradeoff is a heavier backbone than a
YOLO-nano model, so it's slower per frame on CPU; that's an acceptable
trade for "runs anywhere, simple to trust" in the portable engine.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from pantherlake_ai_core.model_cache import resolve_file

from .coco91_labels import ID2LABEL
from .types import Detection

_DEFAULT_REPO = "Xenova/detr-resnet-50"
_DEFAULT_FILENAME = "onnx/model_quantized.onnx"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_NO_OBJECT_INDEX = 91  # DETR appends one "no object" class after the 91 real labels


def _preprocess(frame_bgr: np.ndarray, short_side: int = 480, max_side: int = 800) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    scale = short_side / min(height, width)
    if max(height, width) * scale > max_side:
        scale = max_side / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))

    resized = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = normalized.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
    return np.ascontiguousarray(chw, dtype=np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _postprocess(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_width: int,
    orig_height: int,
    confidence_threshold: float,
) -> list[Detection]:
    probs = _softmax(logits, axis=-1)  # (num_queries, 92)
    class_ids = probs.argmax(axis=-1)
    scores = probs.max(axis=-1)

    detections = []
    for class_id, score, box in zip(class_ids, scores, pred_boxes):
        if class_id == _NO_OBJECT_INDEX or score < confidence_threshold:
            continue
        label = ID2LABEL.get(int(class_id))
        if not label or label == "N/A":
            continue
        cx, cy, w, h = box
        x1 = np.clip((cx - w / 2) * orig_width, 0, orig_width)
        y1 = np.clip((cy - h / 2) * orig_height, 0, orig_height)
        x2 = np.clip((cx + w / 2) * orig_width, 0, orig_width)
        y2 = np.clip((cy + h / 2) * orig_height, 0, orig_height)
        detections.append(
            Detection(label=label, confidence=float(score), box=(int(x1), int(y1), int(x2), int(y2)))
        )
    return detections


class PortableDetector:
    def __init__(
        self,
        repo_id: str = _DEFAULT_REPO,
        filename: str = _DEFAULT_FILENAME,
        model_path: str | None = None,
        confidence_threshold: float = 0.7,
        on_downloading: Callable[[], None] | None = None,
    ):
        import onnxruntime as ort

        resolved_path = resolve_file(repo_id, filename, local_path=model_path, on_downloading=on_downloading)
        self.session = ort.InferenceSession(resolved_path, providers=["CPUExecutionProvider"])
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        pixel_values = _preprocess(frame)
        # The traced ONNX graph fixes pixel_mask at (1, 64, 64) regardless of
        # the actual input size; DETR interpolates it internally to match the
        # real feature map, so an all-ones "nothing is padding" mask is correct.
        pixel_mask = np.ones((1, 64, 64), dtype=np.int64)

        logits, pred_boxes = self.session.run(
            ["logits", "pred_boxes"],
            {"pixel_values": pixel_values, "pixel_mask": pixel_mask},
        )
        return _postprocess(logits[0], pred_boxes[0], width, height, self.confidence_threshold)
