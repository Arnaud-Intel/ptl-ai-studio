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
    preferred_realtime_vision_device()                    # best device for a live video model
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
    portable engine's "cpu". A brick with a large openvino model (tens of
    GB) should use preferred_large_model_device() instead of this."""
    return "AUTO" if Engine(engine) == Engine.OPENVINO else "cpu"


def preferred_large_model_device() -> str:
    """The OpenVINO device to default a *large* model to (a 30B coding LLM,
    a 7B vision model): the discrete GPU if the machine has one, since it
    is faster, otherwise the integrated GPU -- named, not left to "AUTO".
    No GPU at all -> "AUTO". Never one dev machine's card id (GPU.1) baked
    in as everyone's default.

    The iGPU is not a consolation prize. Measured on the Dell XPS 14
    (2026-09-11): Qwen3-Coder-30B-A3B int4 takes 16.9 GB of the Arc B390's
    shared memory and streams ~38 tokens/s, first token in under 0.4 s; an
    Arc Pro B60 streams ~65 but starts no sooner. The Qwen2.5-VL-7B OCR
    model takes 5.9 GB and writes 20 tokens/s on the iGPU against 50 on the
    B60, while the iGPU reaches its first token sooner (2.0 s vs 2.5 s). A
    discrete GPU is faster; it is not required.
    """
    gpus = list_gpu_devices()
    discrete = [g for g in gpus if "dGPU" in g.full_name]
    if discrete:
        return discrete[-1].id
    if gpus:
        return gpus[0].id
    return "AUTO"


def preferred_realtime_vision_device() -> str:
    """The OpenVINO device to default a *live video* model to: the
    integrated GPU if there is one, otherwise "AUTO".

    Not AUTO, which is what every other brick defaults to, because for a
    small detection model AUTO is measurably the worst real choice. On 30
    identical frames of a street camera, YOLO11s: AUTO 33.6ms, iGPU 7.9ms,
    NPU 20.8ms, CPU 20.7ms -- identical detections, four times the latency.
    AUTO is picking for a different objective than "this frame, now", and a
    video brick wants the frame now.

    The *integrated* GPU specifically, i.e. the first one, even on a
    machine with a much larger discrete card. The discrete GPU carries a
    fixed per-frame penalty here -- measured at +19.6ms, +17.1ms and
    +18.2ms for YOLO11 n, s and m respectively. That it barely moves as the
    model triples in cost is the point: a card that were simply slower at
    compute would fall further behind on the bigger model. A constant
    penalty is the data path, not the silicon -- every frame is 2.6MB that
    has to cross to the card and come back, and this machine's dGPU sits on
    a narrow PCIe link without resizable BAR. Latency on GPU.1 is also far
    less predictable (p10 11ms, p90 32ms) where the iGPU holds 6.5-7.6ms,
    which is what transfer contention looks like.

    So the two preferences pull apart cleanly, and both are right: a live
    video model wants the iGPU, which shares system memory and never makes
    the trip; a large model wants `preferred_large_model_device()`, where
    weights are uploaded once and VRAM capacity is the thing that matters,
    and 18ms amortized over a multi-second generation is nothing.
    """
    gpus = list_gpu_devices()
    if gpus:
        integrated = [g for g in gpus if "dGPU" not in g.full_name]
        return (integrated or gpus)[0].id
    return "AUTO"


def ov_config_for(device: str) -> dict[str, str]:
    """OpenVINO compile/pipeline config for `device`: the shared on-disk
    compiled-model cache (OV_CACHE_DIR) for the NPU, and nothing for
    anything else.

    The NPU is where the cache earns its keep -- YOLO11n compiles in 2.4s
    cold and 0.04s from cache, and a large LLM takes minutes.

    **The GPU deliberately gets no cache**, even though it is slow to
    compile. On this machine (Arc B390 iGPU) a cached GPU model comes back
    *numerically wrong* rather than failing: YOLO11n-int8 finds 7-10
    objects in a street scene when it compiles, and 1 on every subsequent
    load from cache -- silently, with no error anywhere. It is not stale
    blobs: a brand-new cache directory does it on its very first hit, and
    no precision or execution-mode hint changes it. Measured, the cache
    was buying between nothing and 1.7s there anyway (a cache hit on
    YOLO11n costs the same 1.73s as compiling from scratch), so this trades
    an optimization worth ~0-1.7s for correctness.

    CPU loads fast without it, and "AUTO" is excluded too: it may land on
    CPU, whose cache blobs for a large LLM run to gigabytes of disk for no
    load-time win.

    Use as `core.compile_model(model, device, ov_config_for(device))` or
    `ov_genai.SomePipeline(model_dir, device, **ov_config_for(device))`.
    """
    if device.upper() == "NPU":
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
