"""Inference-engine/device discovery shared by bricks that offer a
switchable local-AI backend: a portable one (e.g. CTranslate2, PyTorch —
CPU, or CUDA on an NVIDIA GPU) versus OpenVINO, which is what actually lets
a demo target Intel CPU/iGPU/NPU (e.g. Panther Lake) explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class GpuDevice:
    """One OpenVINO-visible GPU: its device id, human-readable name, and a
    key for correlating it with OS-level utilization telemetry."""

    id: str  # "GPU.0", "GPU.1", ... or bare "GPU" on a single-GPU machine
    full_name: str  # e.g. "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)"
    luid: str | None  # see _luid_to_perfcounter_key(); None if unavailable


def _luid_to_perfcounter_key(raw_luid: str) -> str | None:
    """Convert OpenVINO's DEVICE_LUID (a 16-hex-char string: 4 LowPart bytes
    then 4 HighPart bytes, both little-endian) into the
    "0x{HighPart:08x}_0x{LowPart:08x}" form Windows' `\\GPU Engine(*)`
    performance-counter instance names embed -- letting telemetry.py
    attribute a utilization reading to the exact physical GPU it came from,
    rather than guessing. Verified against real hardware: an Arc B390 iGPU's
    DEVICE_LUID '8df4090200000000' decodes to '0x00000000_0x0209f48d', an
    exact match to its live perf-counter LUID.
    """
    try:
        raw = bytes.fromhex(raw_luid)
    except ValueError:
        return None
    if len(raw) != 8:
        return None
    low = int.from_bytes(raw[0:4], "little")
    high = int.from_bytes(raw[4:8], "little")
    return f"0x{high:08x}_0x{low:08x}"


def list_gpu_devices() -> list[GpuDevice]:
    """Return every OpenVINO-visible GPU with its friendly name and (on
    Windows) a telemetry-correlation key -- empty if OpenVINO isn't
    installed. Each device's property lookups are independently
    best-effort: one GPU failing to report a name or LUID doesn't blank out
    the others.
    """
    try:
        from openvino import Core
    except ImportError:
        return []

    core = Core()
    devices = []
    for device_id in core.available_devices:
        if not device_id.upper().startswith("GPU"):
            continue
        try:
            full_name = core.get_property(device_id, "FULL_DEVICE_NAME")
        except Exception:
            full_name = device_id
        try:
            raw_luid = core.get_property(device_id, "DEVICE_LUID")
        except Exception:
            raw_luid = None
        luid = _luid_to_perfcounter_key(raw_luid) if raw_luid else None
        devices.append(GpuDevice(id=device_id, full_name=full_name, luid=luid))
    return devices


def describe_devices() -> str:
    """Human-readable summary of what each engine can currently target.

    Intended for a brick's --list-devices output.
    """
    lines = ["portable backend: cpu, cuda (if an NVIDIA GPU + CUDA are set up)"]
    ov_devices = list_openvino_devices()
    if ov_devices:
        gpu_names = {gd.id: gd.full_name for gd in list_gpu_devices()}
        labeled = [f"{d} ({gpu_names[d]})" if d in gpu_names else d for d in ov_devices]
        lines.append(f"openvino backend: {', '.join(labeled)} (AUTO lets OpenVINO pick the best one)")
    else:
        lines.append(
            "openvino backend: not installed for this brick "
            "(install its `openvino` extra to enable Intel CPU/GPU/NPU acceleration)"
        )
    return "\n".join(lines)
