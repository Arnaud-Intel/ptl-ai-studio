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
#
# Under it, one folder per OpenVINO version (`ov_cache_dir`): what one
# version compiled is of no use to the next, and until 2026-10-12 it all
# lay in this folder itself, where a year of versions and trials had left
# 26 GB on the demo laptop.
OV_CACHE_DIR = Path.home() / ".cache" / "pantherlake-ai-studio" / "ov_cache"
# The OpenVINO the app is tried on: what `uv.lock` installs. An older one
# still in an environment that was not synced is said at start (see
# `openvino_behind`), not refused: the bricks run on it, less well tried.
TESTED_OPENVINO = "2026.4.1"
# The first OpenVINO whose GPU gives the same answers from the cache as from
# compiling (see `ov_config_for`). A copy still on an older one -- updated by
# hand, its environment not synced -- keeps compiling its GPU models.
GPU_CACHE_SINCE = "2026.4.1"
# Set to 0 (or "off") to compile GPU models every time, as before 1.0.4:
# the way out if a model ever answers differently from the cache.
GPU_CACHE_SWITCH = "PTL_GPU_CACHE"
# What the GPU keeps is as large as the models: 15 GB for the 30B coder, 32 GB
# for everything the app loads on one GPU. With less than this free on the
# cache's drive the GPU compiles every time instead, as it did: a disk filled
# half-way through a 15 GB write is a demo that stops.
GPU_CACHE_FREE_GB = 20.0


def openvino_version() -> str | None:
    """The OpenVINO installed here ("2026.4.1"), or None if there is none.
    Read from the package's own record, so that asking does not load it."""
    from importlib import metadata

    try:
        return metadata.version("openvino")
    except metadata.PackageNotFoundError:
        return None


def _numbers(version: str) -> tuple[int, ...]:
    import re

    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def openvino_behind() -> str | None:
    """The installed OpenVINO, if it is older than the one the app is tried
    on; None when it is that one, a newer one, or not installed at all."""
    installed = openvino_version()
    return installed if installed and _numbers(installed) < _numbers(TESTED_OPENVINO) else None


def ov_cache_dir() -> Path:
    """Where this OpenVINO keeps what it compiles."""
    return OV_CACHE_DIR / (openvino_version() or "none")


def gpu_cache_allowed() -> bool:
    """Whether GPU models may be kept in the compiled-model cache and read
    back from it: on an OpenVINO that reads them back right, with room on
    the disk, unless it was switched off."""
    return gpu_cache_refusal() is None


def gpu_cache_refusal() -> str | None:
    """Why a GPU keeps nothing of what it compiles here, in a sentence for
    whoever is looking at the models; None when it does."""
    import os

    if os.environ.get(GPU_CACHE_SWITCH, "").strip().lower() in ("0", "off", "no", "false"):
        return f"switched off ({GPU_CACHE_SWITCH}=0)"
    installed = openvino_version()
    if not installed:
        return "OpenVINO is not installed"
    if _numbers(installed) < _numbers(GPU_CACHE_SINCE):
        return f"OpenVINO {installed} reads GPU models back wrong: it takes {GPU_CACHE_SINCE} or later"
    free = _free_gb(OV_CACHE_DIR)
    if free is not None and free < GPU_CACHE_FREE_GB:
        return f"{free:.0f} GB free on the disk, and it takes {GPU_CACHE_FREE_GB:.0f}"
    return None


def _free_gb(path: Path) -> float | None:
    """Free space on the drive `path` is (or will be) on; None if it can't be told."""
    import shutil

    for folder in (path, *path.parents):
        try:
            return shutil.disk_usage(folder).free / 1024**3
        except OSError:
            continue
    return None


def clear_stale_ov_cache() -> int:
    """Remove what older OpenVINO versions compiled: their folders, and
    everything from before the cache was kept per version. Returns the
    bytes freed. A newer version's folder is left alone -- a second copy of
    the app, further ahead, may be using it. Never raises: it is tidying."""
    import stat

    def remove(path: Path) -> int:
        """One file's size if it could be removed, 0 if it stays for next
        time (held open by another copy of the app)."""
        try:
            size = path.stat().st_size
            # The NPU's larger blobs are written read-only, and Windows
            # deletes no read-only file: 27 GB stayed behind for that.
            path.chmod(stat.S_IWRITE | stat.S_IREAD)
            path.unlink()
        except OSError:
            return 0
        return size

    current = openvino_version()
    if current is None or not OV_CACHE_DIR.is_dir():
        return 0
    freed = 0
    for entry in list(OV_CACHE_DIR.iterdir()):
        if not entry.is_dir():
            freed += remove(entry)
            continue
        versioned = _numbers(entry.name) if entry.name[:1].isdigit() else ()
        if versioned and versioned >= _numbers(current):
            continue
        try:
            inside = sorted(entry.rglob("*"), key=lambda path: len(path.parts), reverse=True)
        except OSError:
            continue
        for path in inside:
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass  # something in it stayed
            else:
                freed += remove(path)
        try:
            entry.rmdir()
        except OSError:
            pass
    return freed


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


def preferred_device() -> str:
    """The chip a brick gets when the choice is left to the app ("Auto"):
    the integrated GPU if there is one, otherwise the CPU. Always a real
    device, never OpenVINO's "AUTO".

    "AUTO" hides the one thing this showcase exists to show -- which chip is
    doing the work: a brick started on it reported "AUTO" as its device and
    so appeared under no chip at all. It was also measurably the worst
    choice for a small vision model (see preferred_realtime_vision_device)
    and fails outright for the vision-language model (screen_ocr).

    The integrated GPU rather than a discrete one: for the small models this
    covers, the trip to a discrete card costs more than its extra compute
    returns. A big model wants preferred_large_model_device() instead. The
    NPU is never picked automatically -- some models don't compile for it,
    so it stays an explicit choice.
    """
    gpus = list_gpu_devices()
    if gpus:
        integrated = [g for g in gpus if "dGPU" not in g.full_name]
        return (integrated or gpus)[0].id
    return "CPU"


def preferred_npu_device(devices: list[str] | None = None) -> str:
    """The chip for the work this app keeps on the NPU when nobody chose
    one -- speech (Whisper) and the meeting-notes model: the NPU if the
    machine has one in working order, otherwise what preferred_device()
    gives any brick. `devices` is the machine's OpenVINO devices, when the
    caller already has the list.

    The one exception to "the NPU is never picked automatically", for models
    measured there. On the XPS 14's NPU (2026-10-07) Whisper medium
    translates French 17 times faster than real time, against 22 on the
    integrated GPU, with the same words coming out; speech is a few tenths
    of a second of work per utterance, which is what the NPU is for, and it
    leaves the GPU to whatever else is on screen. An NPU that Windows reset
    earlier in the session is out of use until the app restarts (see the
    `npu` module) and is not picked.
    """
    from . import npu

    available = list_openvino_devices() if devices is None else devices
    if not npu.lost() and any(npu.is_npu(device) for device in available):
        return "NPU"
    return preferred_device()


def default_speech_device(engine: Engine | str) -> str:
    """Where a speech model goes when the user doesn't say: on OpenVINO the
    NPU if there is one (preferred_npu_device()), on the portable engine
    "cpu"."""
    return preferred_npu_device() if Engine(engine) == Engine.OPENVINO else "cpu"


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
    compiled-model cache (`ov_cache_dir()`, one folder per OpenVINO
    version) for the NPU and, where `gpu_cache_allowed()`, for a GPU; and
    nothing for anything else.

    The NPU is where the cache earns its keep -- YOLO11n compiles in 2.4s
    cold and 0.04s from cache, and a large LLM takes minutes.

    **The GPU has it from OpenVINO 2026.4.1, and never before.** Up to
    2026.3, on this machine (Arc B390 iGPU), a cached GPU model came back
    *numerically wrong* rather than failing: YOLO11n-int8 found 7-10
    objects in a street scene when it compiled, and 1 on every subsequent
    load from cache -- silently, with no error anywhere. It was not stale
    blobs: a brand-new cache directory did it on its very first hit, and
    no precision or execution-mode hint changed it. So a copy whose
    OpenVINO is older (updated by hand, not synced) still compiles its GPU
    models every time.

    On 2026.4.1, every model the app loads on a GPU was tried three ways,
    each in a process of its own -- compiled without the cache, compiled
    into it, read back from it -- on the Arc B390 and on the Arc Pro B60
    (2026-10-12): the detector's boxes, the embedder's vector, the answers
    of the 1.5B, 8B and 30B language models, Whisper's text, the vision
    model's reading of a receipt and the image model's pictures (drawn and
    restyled) are the same to the last digit and the last pixel, and the
    five HTML Creator pages of "Same page every time" are the same files.
    Loading, compiled then read back, in seconds on the B390 / the B60:
    30B coder 22.5 -> 5.3 / 38.7 -> 18.6, 8B 10.0 -> 2.9 / 17.9 -> 7.3,
    vision 7B 8.2 -> 4.5 / 17.0 -> 9.9, Whisper medium 7.7 -> 2.7 /
    11.8 -> 4.9, image model 7.8 -> 6.4 / 21.8 -> 11.8. The coder writes
    as fast either way (44.8 and 44.1 tokens/s over three alternating
    runs). What it costs is disk: 15 GB for the coder, 32 GB for
    everything on one GPU, as much again for a second GPU -- hence
    `GPU_CACHE_FREE_GB`, and `PTL_GPU_CACHE=0` to go without.

    CPU loads fast without it, and "AUTO" is excluded too: it may land on
    CPU, whose cache blobs for a large LLM run to gigabytes of disk for no
    load-time win.

    Use as `core.compile_model(model, device, ov_config_for(device))` or
    `ov_genai.SomePipeline(model_dir, device, **ov_config_for(device))`.
    """
    name = device.upper()
    if name == "NPU" or (name.split(".")[0] == "GPU" and gpu_cache_allowed()):
        folder = ov_cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        return {"CACHE_DIR": str(folder)}
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
