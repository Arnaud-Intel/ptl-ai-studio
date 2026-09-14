"""Energy and battery readings, so the launcher can show what AI costs in
watts instead of asserting the NPU is efficient (BACKLOG R18).

Windows exposes the processor's RAPL energy counters as the "Energy Meter"
performance-counter set: one cumulative counter per rail, in picowatt-hours
-- `package` (the whole chip), `cores` (the CPU cores), `graphics` (the
iGPU) and `memory` (DRAM). Verified on the XPS 14: one second's package
delta of 5.19e9 pWh is 18.7 J, i.e. 18.4 W, which is exactly what Windows'
own derived Power counter reported for the same second. The NPU has no rail
of its own: its work shows up only in `package`, as power the core and
graphics rails don't account for -- say so, never invent a split.

Being cumulative, the counters make any window exact: read before, read
after, subtract. That is what lets a single request or a single utterance
carry its own energy, not an average smeared over a polling interval.

Read in-process through PDH (pdh.dll) with ctypes -- a persistent query
costs microseconds per read, where spawning PowerShell's Get-Counter costs
a second. No new dependency: pywin32 isn't installed and isn't needed.
Anywhere else (another OS, a machine without the counters) `available` is
False and reads return None: a gauge that says nothing beats one that says
zero.
"""
from __future__ import annotations

import ctypes
import platform
import threading
import time
from dataclasses import dataclass

import psutil

_IS_WINDOWS = platform.system() == "Windows"
_J_PER_PWH = 3.6e-9  # 1 pWh = 1e-12 Wh = 3.6e-9 J
_PDH_MORE_DATA = 0x800007D2
# Instance "RAPL_Package0_PKG" -> rail "pkg"; summed across packages if a
# machine has more than one.
_RAILS = {"pkg": "package", "pp0": "cores", "pp1": "graphics", "dram": "memory"}


@dataclass(frozen=True)
class EnergySample:
    at: float  # time.monotonic() when read
    joules: dict[str, float]  # cumulative, per rail: "package", "cores", "graphics", "memory"


def watts_between(a: EnergySample, b: EnergySample) -> dict[str, float]:
    """Average power per rail between two samples; empty if no time passed."""
    seconds = b.at - a.at
    if seconds <= 0:
        return {}
    return {rail: (b.joules[rail] - a.joules[rail]) / seconds for rail in b.joules if rail in a.joules}


if _IS_WINDOWS:
    from ctypes import wintypes

    class _FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    class _PDH_RAW_COUNTER(ctypes.Structure):
        _fields_ = [
            ("CStatus", wintypes.DWORD),
            ("TimeStamp", _FILETIME),
            ("FirstValue", ctypes.c_longlong),
            ("SecondValue", ctypes.c_longlong),
            ("MultiCount", wintypes.DWORD),
        ]

    class _PDH_RAW_COUNTER_ITEM_W(ctypes.Structure):
        _fields_ = [("szName", wintypes.LPWSTR), ("RawValue", _PDH_RAW_COUNTER)]


class EnergyMeter:
    """One persistent PDH query over `\\Energy Meter(*)\\Energy`. Thread-safe:
    the telemetry poller and the runners metering a request share one."""

    def __init__(self) -> None:
        self.available = False
        self._lock = threading.Lock()
        self._pdh = None
        if not _IS_WINDOWS:
            return
        try:
            pdh = ctypes.WinDLL("pdh")
            pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhAddEnglishCounterW.argtypes = [
                ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
            ]
            pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            pdh.PdhGetRawCounterArrayW.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
            ]
            for fn in (pdh.PdhOpenQueryW, pdh.PdhAddEnglishCounterW, pdh.PdhCollectQueryData, pdh.PdhGetRawCounterArrayW):
                fn.restype = wintypes.DWORD
            self._query = ctypes.c_void_p()
            self._counter = ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
                return
            if pdh.PdhAddEnglishCounterW(self._query, "\\Energy Meter(*)\\Energy", 0, ctypes.byref(self._counter)) != 0:
                return
            self._pdh = pdh
            self.available = self.read() is not None
        except (OSError, AttributeError):
            self.available = False

    def read(self) -> EnergySample | None:
        """Cumulative joules per rail, now; None if the counters can't be read."""
        if self._pdh is None:
            return None
        with self._lock:
            if self._pdh.PdhCollectQueryData(self._query) != 0:
                return None
            size, count = wintypes.DWORD(0), wintypes.DWORD(0)
            if self._pdh.PdhGetRawCounterArrayW(self._counter, ctypes.byref(size), ctypes.byref(count), None) != _PDH_MORE_DATA:
                return None
            buffer = (ctypes.c_byte * size.value)()
            if self._pdh.PdhGetRawCounterArrayW(self._counter, ctypes.byref(size), ctypes.byref(count), buffer) != 0:
                return None
            at = time.monotonic()
            items = ctypes.cast(buffer, ctypes.POINTER(_PDH_RAW_COUNTER_ITEM_W))
            joules: dict[str, float] = {}
            for i in range(count.value):
                item = items[i]
                rail = _RAILS.get((item.szName or "").lower().rsplit("_", 1)[-1])
                if rail and item.RawValue.CStatus == 0:
                    joules[rail] = joules.get(rail, 0.0) + item.RawValue.FirstValue * _J_PER_PWH
        return EnergySample(at=at, joules=joules) if "package" in joules else None


def battery() -> dict | None:
    """Charge, whether it's plugged in, and Windows' time-left estimate
    (None while charging or unknown). None on a machine with no battery."""
    try:
        reading = psutil.sensors_battery()
    except Exception:
        return None
    if reading is None:
        return None
    seconds_left = int(reading.secsleft)
    return {
        "percent": round(reading.percent),
        "plugged": bool(reading.power_plugged),
        "seconds_left": seconds_left if seconds_left >= 0 else None,
    }
