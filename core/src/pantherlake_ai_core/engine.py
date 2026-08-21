"""Inference-engine/device discovery shared by bricks that offer a
switchable local-AI backend: a portable one (e.g. CTranslate2, PyTorch —
CPU, or CUDA on an NVIDIA GPU) versus OpenVINO, which is what actually lets
a demo target Intel CPU/iGPU/NPU (e.g. Panther Lake) explicitly.
"""
from __future__ import annotations

from enum import Enum


class Engine(str, Enum):
    """Which runtime a brick uses to run its model."""

    PORTABLE = "portable"
    OPENVINO = "openvino"


def list_openvino_devices() -> list[str]:
    """Return OpenVINO's available compute devices (e.g. CPU, GPU, NPU).

    Returns an empty list if OpenVINO isn't installed, rather than raising,
    since it's an optional per-brick dependency (install via each brick's
    `openvino` extra).
    """
    try:
        from openvino import Core
    except ImportError:
        return []
    return list(Core().available_devices)


def describe_devices() -> str:
    """Human-readable summary of what each engine can currently target.

    Intended for a brick's --list-devices output.
    """
    lines = ["portable backend: cpu, cuda (if an NVIDIA GPU + CUDA are set up)"]
    ov_devices = list_openvino_devices()
    if ov_devices:
        lines.append(f"openvino backend: {', '.join(ov_devices)} (AUTO lets OpenVINO pick the best one)")
    else:
        lines.append(
            "openvino backend: not installed for this brick "
            "(install its `openvino` extra to enable Intel CPU/GPU/NPU acceleration)"
        )
    return "\n".join(lines)
