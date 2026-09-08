"""Inference-engine/device discovery and defaults shared by every brick that
offers a switchable local-AI backend: a portable one (CTranslate2, ONNX
Runtime, PyTorch, llama.cpp -- CPU, or CUDA on an NVIDIA GPU) versus
OpenVINO, which is what actually lets a demo target Intel CPU/iGPU/NPU
(e.g. Panther Lake) explicitly.

Everything a CLI or the launcher needs to go from "what the user asked for"
to "which engine, on which device" lives here, so no brick carries its own
copy of the rules:

    engine = resolve_engine(args.engine)                  # explicit, else the best available
    device = args.compute_device or default_device(engine)
    ov_config_for(device)                                 # OpenVINO compile config (model cache)
    print_devices(mics=True, cameras=True)                # a brick's --list-devices output
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Engine(str, Enum):
    """Which runtime a brick uses to run its model."""

    PORTABLE = "portable"
    OPENVINO = "openvino"


# One compiled-model cache for every brick. OpenVINO keys entries by model
# and device, so a GPU/NPU compile happens once per machine instead of once
# per run. Absolute on purpose: the old per-brick `"ov_cache"` was relative
# to the current working directory, so the launcher, each CLI, and every
# directory a CLI was run from each recompiled into their own copy.
OV_CACHE_DIR = Path.home() / ".cache" / "pantherlake-ai-studio" / "ov_cache"


@functools.lru_cache(maxsize=1)
def _openvino_devices() -> tuple[str, ...]:
    try:
        from openvino import Core
    except ImportError:
        return ()
    return tuple(Core().available_devices)


def list_openvino_devices() -> list[str]:
    """OpenVINO's available compute devices (e.g. CPU, GPU.0, NPU) -- an
    empty list if OpenVINO isn't installed rather than an error, since it's
    an optional per-brick dependency (each brick's `openvino` extra).

    Cached after the first call: enumerating devices means instantiating
    an OpenVINO Core, which costs a noticeable fraction of a second, and
    the answer doesn't change while a process runs. See refresh_devices().
    """
    return list(_openvino_devices())


@dataclass(frozen=True)
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


@functools.lru_cache(maxsize=1)
def _gpu_devices() -> tuple[GpuDevice, ...]:
    try:
        from openvino import Core
    except ImportError:
        return ()

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
    return tuple(devices)


def list_gpu_devices() -> list[GpuDevice]:
    """Every OpenVINO-visible GPU with its friendly name and (on Windows) a
    telemetry-correlation key -- empty if OpenVINO isn't installed. Each
    device's property lookups are independently best-effort: one GPU
    failing to report a name or LUID doesn't blank out the others. Cached
    like list_openvino_devices().
    """
    return list(_gpu_devices())


def refresh_devices() -> None:
    """Forget the cached device lists so the next call re-enumerates (e.g.
    after a driver install, or in a test that fakes a different machine)."""
    _openvino_devices.cache_clear()
    _gpu_devices.cache_clear()


def resolve_engine(explicit: str | Engine | None = None) -> Engine:
    """The engine to use: `explicit` when given (a name or an Engine -- an
    unknown name raises ValueError, exactly as `Engine(name)` would), else
    openvino when it's installed and sees at least one device, else
    portable. The one rule every CLI's default and the launcher's share, so
    "run it with no flags" means the same thing everywhere."""
    if explicit:
        return Engine(explicit)
    return Engine.OPENVINO if list_openvino_devices() else Engine.PORTABLE


def default_device(engine: Engine | str) -> str:
    """Where a model goes when the user doesn't say: OpenVINO's "AUTO" (it
    picks the best device it has -- a GPU when present, else CPU) or the
    portable engine's "cpu". A brick whose openvino model needs a discrete
    GPU's VRAM should use preferred_large_model_device() for that case
    instead of this."""
    return "AUTO" if Engine(engine) == Engine.OPENVINO else "cpu"


def preferred_large_model_device() -> str:
    """The OpenVINO device to default a *large* model to (one that needs
    its own VRAM, e.g. a 30B coding LLM): a discrete GPU if the machine has
    one, otherwise "AUTO" -- so a brick never hardcodes one dev machine's
    card id (GPU.1) as everyone's default. Empty/unknown GPU list -> "AUTO".
    """
    discrete = [g for g in list_gpu_devices() if "dGPU" in g.full_name]
    if discrete:
        return discrete[-1].id
    return "AUTO"


def ov_config_for(device: str) -> dict[str, str]:
    """OpenVINO compile/pipeline config for `device`: the shared on-disk
    compiled-model cache (OV_CACHE_DIR) for GPU and NPU targets, where
    recompiling on every run costs seconds to minutes (the NPU especially),
    and nothing for CPU, which loads fast without it. "AUTO" deliberately
    gets nothing too: it may land on CPU, whose cache blobs for a large LLM
    run to gigabytes of disk for no load-time win.

    Use as `core.compile_model(model, device, ov_config_for(device))` or
    `ov_genai.SomePipeline(model_dir, device, **ov_config_for(device))`.
    """
    upper = device.upper()
    if upper == "NPU" or "GPU" in upper:
        OV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return {"CACHE_DIR": str(OV_CACHE_DIR)}
    return {}


def describe_devices() -> str:
    """Human-readable summary of what each engine can currently target --
    the inference-devices part of a brick's --list-devices output."""
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


def _section(title: str, items: list[str]) -> str:
    body = "\n".join(f"  - {item}" for item in items) if items else "  (none found)"
    return f"{title}\n{body}"


def print_devices(
    *,
    mics: bool = False,
    speakers: bool = False,
    cameras: bool = False,
    screens: bool = False,
    inference_flag: str = "--compute-device",
) -> None:
    """Print what a brick's --list-devices should show: the hardware inputs
    it can use (opt in to each kind) followed by the inference devices each
    engine can target -- one implementation instead of one per CLI.
    `inference_flag` names the option(s) the last section applies to."""
    sections: list[str] = []
    if mics or speakers:
        from . import audio

        if mics:
            sections.append(_section("Microphones:", audio.list_microphones()))
        if speakers:
            sections.append(_section("Output devices (system audio, captured via loopback):", audio.list_speakers()))
    if cameras or screens:
        from . import video

        if cameras:
            sections.append(_section("Cameras (--camera-index N):", [str(index) for index in video.list_cameras()]))
        if screens:
            found = [f"{s['index']}: {s['width']}x{s['height']}" for s in video.list_screens()]
            sections.append(_section("Screens (--screen-index N):", found))
    sections.append(f"Inference devices ({inference_flag}):\n{describe_devices()}")
    print("\n\n".join(sections))
