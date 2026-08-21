"""Shared pre/postprocessing and effect application for both segmentation
backends.

Both engines run the *exact same* tiny MediaPipe selfie-segmentation ONNX
model (~224KB quantized) -- OpenVINO can read an ONNX file directly via
`Core.read_model()`, no separate IR conversion step needed -- so there's
one preprocessing/postprocessing implementation shared by both, and only
the inference runtime (and therefore the device it can target) differs.
That means switching engines here isolates the hardware variable cleanly:
same model, same weights, different silicon.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULT_REPO = "onnx-community/mediapipe_selfie_segmentation"
DEFAULT_FILENAME = "onnx/model_quantized.onnx"

INPUT_SIZE = 256


def resolve_model_path(repo_id: str, filename: str, model_path: str | None) -> str:
    if model_path:
        return model_path
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = rgb.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(chw, dtype=np.float32)


def postprocess(alpha_output: np.ndarray, orig_width: int, orig_height: int) -> np.ndarray:
    """alpha_output: (1, 1, 256, 256) -> a smoothed (H, W) float32 mask in [0, 1]."""
    mask = alpha_output[0, 0]
    mask = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)  # soften the cutout edge
    return np.clip(mask, 0.0, 1.0)


def person_coverage(mask: np.ndarray) -> float:
    """Fraction of the frame the model attributes to a person -- a cheap,
    honest "is this doing anything" stat for the UI."""
    return float(mask.mean())


def apply_blur(frame: np.ndarray, mask: np.ndarray, strength: int = 41) -> np.ndarray:
    ksize = strength | 1  # cv2.GaussianBlur requires an odd kernel size
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    mask3 = mask[:, :, np.newaxis]
    blended = frame.astype(np.float32) * mask3 + blurred.astype(np.float32) * (1 - mask3)
    return blended.astype(np.uint8)


def apply_replace(frame: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    background = np.full_like(frame, color_bgr, dtype=np.uint8)
    mask3 = mask[:, :, np.newaxis]
    blended = frame.astype(np.float32) * mask3 + background.astype(np.float32) * (1 - mask3)
    return blended.astype(np.uint8)
