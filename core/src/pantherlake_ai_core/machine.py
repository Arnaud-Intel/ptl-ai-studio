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
"""
from __future__ import annotations

import ctypes
import functools
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


def describe() -> dict:
    """What the pages show about it: the memory, and the GPU that makes up
    for it."""
    memory = total_memory_gb()
    return {
        "memory_gb": round(memory) if memory is not None else None,
        "big_discrete_gpu": big_discrete_gpu(),
        "simulated": bool(os.environ.get(MEMORY_SWITCH, "").strip()),
    }
