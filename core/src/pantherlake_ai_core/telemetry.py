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

import functools
import json
import platform
import subprocess
from dataclasses import dataclass, field

import psutil

from . import engine as engine_mod

_IS_WINDOWS = platform.system() == "Windows"

# Utilization of every GPU-Engine instance, as one compact JSON object.
# Deliberately *only* the counters: this is what runs on every poll, and
# the OS's own wildcard expansion over "GPU Engine(*)" already accounts for
# ~2s of it (measured -- it is the floor for this reading, not our overhead).
_COUNTERS_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$samples = @{}
foreach ($s in (Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples) {
    $samples[$s.InstanceName] = [math]::Round($s.CookedValue, 1)
}
@{ samples = $samples } | ConvertTo-Json -Compress -Depth 4
"""

# The NPU's human-readable name, split out of the poll and cached: walking
# Win32_PnPEntity measured ~0.57s -- a fifth of every reading -- to return a
# string that cannot change while the machine is running. GPU names come
# from OpenVINO's FULL_DEVICE_NAME instead (see engine.list_gpu_devices),
# which is more precise: it distinguishes an iGPU from a discrete GPU.
_NPU_NAME_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'NPU|AI Boost' } | Select-Object -First 1 -ExpandProperty Name
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


# Engine types a compute-only accelerator reports. Anything whose engines
# are a subset of these has no graphics pipeline, so it is the NPU.
_NPU_ENGINE_TYPES = {"compute", "neural"}


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

    # An adapter with no graphics engines at all is the NPU: NPUs don't do
    # 3D, video or copy work. Matching the *absence* of graphics rather than
    # one exact engine name matters -- this driver reports the NPU's engines
    # as "neural", where an older one said "compute", and pinning the rule to
    # {"compute"} silently lost the NPU gauge on the newer driver. The iGPU
    # also exposes a "neural" engine, so it is the full set that decides,
    # not the presence of any one type.
    npu_luid = next(
        (luid for luid, types in engine_types.items() if types <= _NPU_ENGINE_TYPES),
        None,
    )
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


def _powershell(script: str) -> str | None:
    """Run one PowerShell script, or None if it fails for any reason. The
    spawn itself measured ~0.12s; everything above that is the query."""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 and result.stdout.strip() else None


@functools.lru_cache(maxsize=1)
def npu_name() -> str | None:
    """The NPU's device name, looked up once per process -- it is fixed
    hardware, and asking costs ~0.57s."""
    if not _IS_WINDOWS:
        return None
    out = _powershell(_NPU_NAME_SCRIPT)
    return (out.strip() or None) if out else None


def read_cpu() -> float:
    """CPU utilization. Cheap and cross-platform: ~0.1s, which is the
    sampling window itself rather than overhead. Kept separate from
    read_devices() so a live CPU number never waits behind the slow
    GPU/NPU query."""
    return psutil.cpu_percent(interval=0.1)


def read_devices() -> Utilization:
    """GPU/NPU utilization -- the expensive half, ~2s on Windows, almost
    all of it the OS expanding the "GPU Engine(*)" wildcard. Poll it on a
    background thread, never per web request. `cpu_percent` is left None:
    see read_cpu()."""
    if not _IS_WINDOWS:
        return Utilization(available=False)

    raw = _powershell(_COUNTERS_SCRIPT)
    try:
        data = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        data = None

    if not data or "samples" not in data:
        return Utilization(available=False)

    samples = {k: float(v) for k, v in data["samples"].items()}
    gpus, npu_luid = _classify_luids(samples)

    return Utilization(
        available=True,
        gpus=gpus,
        npu_percent=_sum_for_luid(samples, npu_luid),
        npu_name=npu_name(),
    )


def read() -> Utilization:
    """One complete reading, both halves together. The launcher polls the
    two separately (see launcher/telemetry_poller.py) so the CPU gauge
    isn't stuck behind the device query; this is for callers that just
    want everything at once."""
    reading = read_devices()
    reading.cpu_percent = read_cpu()
    return reading
