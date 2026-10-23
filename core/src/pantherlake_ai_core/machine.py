"""Whether this laptop can run a model that needs a lot of memory.

The 30B coding model alone takes about 17 GB. On the integrated GPU that is
taken from the laptop's own memory, which the GPU shares: under 32 GB it does
not work (the user's word, 2026-10-15; README "Which laptops"). A discrete GPU
with memory of its own -- the Arc Pro B60, 24 GB -- holds it instead, and then
the laptop's memory does not matter.

One rule, asked by everything that offers such a model: the demos' cards and
panels, the routes that load it, the models list, the Auto Demo and the setup
assistant (through the models list). `PTL_MEMORY_GB` stands in for the
laptop's memory, to rehearse a smaller machine on a larger one.

A second rule, for a laptop that has the memory: Windows lets the integrated
GPU use only part of it -- 57% unless Intel Graphics Software is told
otherwise ("Shared GPU/NPU Memory Override") -- and on a 32 GB laptop that is
18 GB, which the coder fills before it has written a line (the user's word,
2026-10-23: it ran once raised to 75%). That share is read where the slider
writes it and, until it is enough, the model is greyed out with what to change. The
studio cannot change it itself: Intel publishes no way but its own app's
slider, and it takes a restart. `PTL_GPU_SHARE_GB` stands in for it.
"""
from __future__ import annotations

import ctypes
import functools
import math
import os
import sys

MEMORY_SWITCH = "PTL_MEMORY_GB"
# What Windows reports is a little under the memory fitted (a 32 GB laptop
# shows 31.4 to 31.8 GB, less what the integrated GPU reserves): a laptop
# within this of what a model needs has it.
_SLACK_GB = 3.0
# A discrete GPU holds the model if it has this much of its own. Below it
# (an Arc A770 has 16 GB) the 17 GB model does not fit there either.
DISCRETE_GPU_MIN_GB = 20.0
GPU_SHARE_SWITCH = "PTL_GPU_SHARE_GB"
# Under this the laptop is short of memory for keeping models loaded between
# demos: a 32 GB one is (what its integrated GPU may use is about 24 GB at
# best, and the coder takes 15 of it), a 64 GB one is not.
TIGHT_BELOW_GB = 48.0


def total_memory_gb() -> float | None:
    """The laptop's memory, as Windows reports it; `PTL_MEMORY_GB` if set.
    None when it cannot be told."""
    switch = os.environ.get(MEMORY_SWITCH, "").strip()
    if switch:
        try:
            return float(switch)
        except ValueError:
            pass
    return _measured_memory_gb()


@functools.lru_cache(maxsize=1)
def _measured_memory_gb() -> float | None:
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                ("free", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong), ("page_free", ctypes.c_ulonglong),
                ("virtual_total", ctypes.c_ulonglong), ("virtual_free", ctypes.c_ulonglong),
                ("extended", ctypes.c_ulonglong),
            ]

        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total / 1024**3
        return None
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except Exception:
        return None


def gpu_share_gb() -> float | None:
    """How much of the laptop's memory Windows lets the integrated GPU use:
    the limit Intel's slider sets, of `total_memory_gb` (so `PTL_MEMORY_GB`
    alone plays a smaller laptop at the same setting); `PTL_GPU_SHARE_GB` if
    set. None when it cannot be told (not Windows)."""
    switch = os.environ.get(GPU_SHARE_SWITCH, "").strip()
    if switch:
        try:
            return float(switch)
        except ValueError:
            pass
    setting = gpu_share_setting()
    memory = total_memory_gb()
    if setting is None or memory is None:
        return None
    return memory * setting[0] / 100


# What Intel's slider writes (found on the XPS 14 on 2026-10-23, at 57%): a
# limit of Windows' video memory manager, in percent of the memory. The
# driver keeps a copy of its own (GpuSystemMemoryPinninglimit, under its
# class key), which it rewrote that day long after the restart: this one is
# written when the slider moves, so its time tells a change still waiting
# for a restart.
#
# Not DXGI: its SharedSystemMemory said 47.5 GB (75%) at that 57%, and an
# allocation test ran past 36 GB there before Windows paged memory out --
# that limit is about what may be resident, not what the slider sets.
_SHARE_KEY = r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers\MemoryManager"
_SHARE_VALUE = "SystemPartitionCommitLimitPercentage"
# With the override never turned on there is no value: Windows' own share,
# half the memory.
_WINDOWS_DEFAULT_PERCENT = 50


def gpu_share_setting() -> tuple[int, bool] | None:
    """(percent of the memory the integrated GPU may use, whether it was
    changed since the laptop last started and so is not in effect yet), or
    None off Windows."""
    if sys.platform != "win32":
        return None
    import time
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _SHARE_KEY) as key:
            percent, _ = winreg.QueryValueEx(key, _SHARE_VALUE)
            written = winreg.QueryInfoKey(key)[2] / 1e7 - 11644473600  # FILETIME -> Unix time
    except OSError:
        return _WINDOWS_DEFAULT_PERCENT, False
    kernel32 = ctypes.windll.kernel32
    kernel32.GetTickCount64.restype = ctypes.c_ulonglong
    started = time.time() - kernel32.GetTickCount64() / 1000
    return int(percent), written > started


def advised_share_percent(needs_gpu_gb: float, memory_gb: float) -> int:
    """The share to set in Intel's slider for `needs_gpu_gb`: the next 5%
    up, never under 75 (what the user found to work on 32 GB)."""
    return max(75, min(90, math.ceil(100 * needs_gpu_gb / memory_gb / 5) * 5))


def short_of(needs_gb: float, memory_gb: float | None) -> bool:
    """Whether a laptop with `memory_gb` has less than `needs_gb` (unknown
    memory is not short)."""
    return memory_gb is not None and memory_gb < needs_gb - _SLACK_GB


def big_discrete_gpu() -> str | None:
    """The id of a discrete GPU that can hold a large model by itself (its
    own memory unknown counts as enough: the Arc Pro B60 reports it), or
    None."""
    from .engine import list_gpu_devices

    for gpu in reversed(list_gpu_devices()):
        if "dGPU" not in gpu.full_name:
            continue
        memory = getattr(gpu, "memory_bytes", None)
        if memory is None or memory / 1024**3 >= DISCRETE_GPU_MIN_GB:
            return gpu.id
    return None


def unfit_reason(needs_gb: float | None, device: str | None = None) -> str | None:
    """Why a model needing `needs_gb` of the laptop's memory cannot run here
    -- on `device`, or on the chip the app would give it when none is said --
    in a sentence for the person looking; None when it can (or when nothing
    is needed, or the memory cannot be told)."""
    if not needs_gb:
        return None
    memory = total_memory_gb()
    if not short_of(needs_gb, memory):
        return None
    discrete = big_discrete_gpu()
    if discrete and (device is None or device.upper() in ("AUTO", discrete.upper())):
        return None
    return (
        f"Needs {needs_gb:g} GB of memory or a discrete GPU such as the Arc Pro B60: "
        f"this laptop has {memory:.0f} GB" + (" and no such GPU." if not discrete else f", and {device} is not that GPU.")
    )


# Where it is, as the user found it on 2026-10-23 (Intel Graphics Software
# for driver 32.0.101.8860): a switch, then a slider from 7% to 93%.
# The launcher's page that shows, with the user's screenshot, where that is.
SHARE_HELP = "/static/gpu-memory.html"
SHARE_SETTING = (
    "Intel Graphics Software: Graphics > General, turn on \"Shared GPU/NPU Memory Override\" and set its "
    "Memory Limit to"
)


def share_reason(needs_gpu_gb: float | None, device: str | None = None) -> str | None:
    """Why the integrated GPU is not allowed enough of the laptop's memory
    for a model needing `needs_gpu_gb` of it there, with what to change; None
    when it is (or nothing is needed, or the share cannot be told), and when
    the model goes elsewhere: a discrete GPU that holds it, or a chip that is
    not a GPU."""
    if not needs_gpu_gb:
        return None
    asked = (device or "AUTO").upper()
    if not (asked == "AUTO" or asked.startswith("GPU")):
        return None
    discrete = big_discrete_gpu()
    if discrete and asked in ("AUTO", discrete.upper()):
        return None
    share = gpu_share_gb()
    if share is None:
        return None
    memory = total_memory_gb()
    if share + 0.5 < needs_gpu_gb:
        percent = advised_share_percent(needs_gpu_gb, memory) if memory else 75
        of_memory = f" of its {memory:.0f} GB" if memory else ""
        return (
            f"Windows lets the integrated GPU use {share:.0f} GB{of_memory}, and the model needs about "
            f"{needs_gpu_gb:g} GB there while it writes. In {SHARE_SETTING} {percent}%, "
            "then restart the laptop."
        )
    # Enough, as set -- and moved since the laptop started, so not in effect
    # yet. On a laptop with memory to spare what it was before was enough too.
    setting = None if os.environ.get(GPU_SHARE_SWITCH, "").strip() else gpu_share_setting()
    if setting and setting[1] and memory_tight():
        return (
            f"The GPU memory limit was changed in Intel Graphics Software (now {setting[0]}%): "
            "it applies once the laptop restarts."
        )
    return None


def blocked_reason(needs_gb: float | None, needs_gpu_gb: float | None, device: str | None = None) -> str | None:
    """Both rules: too little memory (`unfit_reason`), or too little of it
    allowed to the integrated GPU (`share_reason`)."""
    return unfit_reason(needs_gb, device) or share_reason(needs_gpu_gb, device)


def memory_tight() -> bool:
    """Whether models left loaded by one demo should make room for the next
    (a 32 GB laptop: TIGHT_BELOW_GB)."""
    return short_of(TIGHT_BELOW_GB, total_memory_gb())


def describe() -> dict:
    """What the pages show about it: the memory, the integrated GPU's share
    of it, and the GPU that makes up for both."""
    memory = total_memory_gb()
    share = gpu_share_gb()
    setting = None if os.environ.get(GPU_SHARE_SWITCH, "").strip() else gpu_share_setting()
    return {
        "memory_gb": round(memory) if memory is not None else None,
        "gpu_share_gb": round(share, 1) if share is not None else None,
        "gpu_share_percent": round(100 * share / memory) if share is not None and memory else None,
        # Moved in Intel's app since the laptop started: not in effect yet.
        "gpu_share_pending": bool(setting and setting[1]),
        "big_discrete_gpu": big_discrete_gpu(),
        "simulated": bool(os.environ.get(MEMORY_SWITCH, "").strip() or os.environ.get(GPU_SHARE_SWITCH, "").strip()),
    }
