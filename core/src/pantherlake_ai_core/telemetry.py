"""Best-effort local hardware-utilization telemetry (CPU/GPU/NPU), so the
launcher can show which silicon a running demo is actually using.

CPU is cheap and cross-platform (psutil). GPU/NPU utilization is
Windows-only, read from the same "GPU Engine" performance-counter category
Windows' own Task Manager uses for its GPU/NPU graphs. There is no
per-vendor API for this that works across Intel/AMD/NVIDIA GPUs and NPUs,
so instead of guessing which counter instance is which device, this
classifies them from their actual behavior:

- an adapter (LUID) whose engine instances are *only* ever "compute" type
  (no 3D/video/copy engines) is treated as the NPU -- NPUs don't do
  graphics, so this is a reliable signature, not a guess tied to one SKU.
- every other adapter (LUID) is a GPU candidate. Each one is correlated
  against `engine.list_gpu_devices()`'s DEVICE_LUID -- see
  `_luid_to_perfcounter_key` there -- so a multi-GPU machine (e.g. a
  Panther Lake iGPU plus a discrete Arc card) gets one reading *per
  physical GPU*, correctly named, instead of one merged number. A
  candidate that doesn't correlate to a known OpenVINO GPU (e.g. a virtual
  display adapter) is dropped rather than mislabeled. If OpenVINO isn't
  installed, or none of the candidates correlate, this falls back to the
  single "adapter with the most distinct engine types" heuristic this
  module used before multi-GPU support existed, so GPU telemetry stays
  available even without OpenVINO in the launcher's own environment.

On non-Windows platforms, or if the counter query fails for any reason,
gpu/npu readings come back as None/empty (`available=False`) rather than
a fabricated number -- the whole point of this feature is to be trustworthy
about what's actually running where.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field

import psutil

from . import engine as engine_mod

_IS_WINDOWS = platform.system() == "Windows"

# Single sample of every GPU-Engine instance's utilization, plus the
# human-readable NPU device name, as one compact JSON object -- one process
# spawn does the whole job instead of several. GPU names come from OpenVINO's
# FULL_DEVICE_NAME instead (see engine.list_gpu_devices), which is more
# precise -- it distinguishes an iGPU from a discrete GPU by name.
_POWERSHELL_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$samples = @{}
foreach ($s in (Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples) {
    $samples[$s.InstanceName] = [math]::Round($s.CookedValue, 1)
}
$npuName = Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'NPU|AI Boost' } | Select-Object -First 1 -ExpandProperty Name
@{ samples = $samples; npu_name = $npuName } | ConvertTo-Json -Compress -Depth 4
"""


@dataclass
class GpuReading:
    id: str  # e.g. "GPU.0" -- matches the id a brick's --compute-device takes
    name: str | None
    percent: float | None


@dataclass
class Utilization:
    available: bool = False
    cpu_percent: float | None = None
    gpus: list[GpuReading] = field(default_factory=list)
    npu_percent: float | None = None
    npu_name: str | None = None


def _engine_types_by_luid(samples: dict[str, float]) -> dict[str, set[str]]:
    engine_types: dict[str, set[str]] = {}
    for instance in samples:
        if "_luid_" not in instance or "_engtype_" not in instance:
            continue
        luid = instance.split("_luid_", 1)[1].split("_phys_", 1)[0]
        engine_type = instance.rsplit("_engtype_", 1)[1] or "unknown"
        engine_types.setdefault(luid, set()).add(engine_type)
    return engine_types


def _classify_luids(samples: dict[str, float]) -> tuple[list[GpuReading], str | None]:
    """Return (gpu readings, npu_luid), attributing each GPU-candidate LUID
    to a specific physical GPU where possible (see module docstring)."""
    engine_types = _engine_types_by_luid(samples)
    if not engine_types:
        return [], None

    npu_luid = next((luid for luid, types in engine_types.items() if types == {"compute"}), None)
    gpu_candidates = [luid for luid in engine_types if luid != npu_luid]

    luid_to_device = {gd.luid: gd for gd in engine_mod.list_gpu_devices() if gd.luid}
    matched = [
        GpuReading(id=luid_to_device[luid].id, name=luid_to_device[luid].full_name, percent=_sum_for_luid(samples, luid))
        for luid in gpu_candidates
        if luid in luid_to_device
    ]
    if matched:
        return matched, npu_luid

    # No OpenVINO GPU correlated (not installed, or none matched) -- fall
    # back to the pre-multi-GPU heuristic so a GPU reading is still shown.
    fallback_luid = max(gpu_candidates, key=lambda luid: len(engine_types[luid]), default=None)
    if fallback_luid is None:
        return [], npu_luid
    return [GpuReading(id="GPU", name=None, percent=_sum_for_luid(samples, fallback_luid))], npu_luid


def _sum_for_luid(samples: dict[str, float], luid: str | None) -> float | None:
    if luid is None:
        return None
    total = sum(value for instance, value in samples.items() if f"_luid_{luid}_" in instance)
    return round(min(total, 100.0), 1)


def read() -> Utilization:
    """Take one reading. The GPU/NPU query costs roughly 1-2 seconds on
    Windows (performance-counter wildcard expansion is inherently slow) --
    callers should poll this on a background thread, not per web request.
    """
    cpu_percent = psutil.cpu_percent(interval=0.1)

    if not _IS_WINDOWS:
        return Utilization(available=False, cpu_percent=cpu_percent)

    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _POWERSHELL_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout) if result.returncode == 0 and result.stdout.strip() else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        data = None

    if not data or "samples" not in data:
        return Utilization(available=False, cpu_percent=cpu_percent)

    samples = {k: float(v) for k, v in data["samples"].items()}
    gpus, npu_luid = _classify_luids(samples)

    return Utilization(
        available=True,
        cpu_percent=cpu_percent,
        gpus=gpus,
        npu_percent=_sum_for_luid(samples, npu_luid),
        npu_name=data.get("npu_name"),
    )
