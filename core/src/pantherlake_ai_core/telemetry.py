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
- the adapter (LUID) with the *most distinct engine types* (3D, video
  decode/process, copy, compute, ...) is treated as the primary GPU.

On non-Windows platforms, or if the counter query fails for any reason,
gpu_percent/npu_percent come back as None (`available=False`) rather than
a fabricated number -- the whole point of this feature is to be trustworthy
about what's actually running where.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass

import psutil

_IS_WINDOWS = platform.system() == "Windows"

# Single sample of every GPU-Engine instance's utilization, plus the
# human-readable GPU/NPU device names, as one compact JSON object -- one
# process spawn does the whole job instead of several.
_POWERSHELL_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$samples = @{}
foreach ($s in (Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples) {
    $samples[$s.InstanceName] = [math]::Round($s.CookedValue, 1)
}
$gpuName = Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name
$npuName = Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'NPU|AI Boost' } | Select-Object -First 1 -ExpandProperty Name
@{ samples = $samples; gpu_name = $gpuName; npu_name = $npuName } | ConvertTo-Json -Compress -Depth 4
"""


@dataclass
class Utilization:
    available: bool = False
    cpu_percent: float | None = None
    gpu_percent: float | None = None
    npu_percent: float | None = None
    gpu_name: str | None = None
    npu_name: str | None = None


def _classify_luids(samples: dict[str, float]) -> tuple[str | None, str | None]:
    """Return (gpu_luid, npu_luid) guessed from each LUID's engine-type mix."""
    engine_types: dict[str, set[str]] = {}
    for instance in samples:
        if "_luid_" not in instance or "_engtype_" not in instance:
            continue
        luid = instance.split("_luid_", 1)[1].split("_phys_", 1)[0]
        engine_type = instance.rsplit("_engtype_", 1)[1] or "unknown"
        engine_types.setdefault(luid, set()).add(engine_type)

    if not engine_types:
        return None, None

    npu_luid = next((luid for luid, types in engine_types.items() if types == {"compute"}), None)
    gpu_candidates = {luid: types for luid, types in engine_types.items() if luid != npu_luid}
    gpu_luid = max(gpu_candidates, key=lambda luid: len(gpu_candidates[luid]), default=None)
    return gpu_luid, npu_luid


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
    gpu_luid, npu_luid = _classify_luids(samples)

    return Utilization(
        available=True,
        cpu_percent=cpu_percent,
        gpu_percent=_sum_for_luid(samples, gpu_luid),
        npu_percent=_sum_for_luid(samples, npu_luid),
        gpu_name=data.get("gpu_name"),
        npu_name=data.get("npu_name"),
    )
