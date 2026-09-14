"""Background sampling of local hardware utilization (CPU/GPU/NPU), so
`/api/telemetry` can answer instantly from a cached reading instead of
paying the real query's cost on every request.

Two loops, not one, because the two halves of a reading cost wildly
different amounts (measured on this hardware):

- CPU, via psutil: ~0.1s, and that is the sampling window itself.
- GPU/NPU, via Windows' "GPU Engine" performance counters: ~2.3s, almost
  entirely the OS expanding the `GPU Engine(*)` wildcard. That is the
  floor for the reading, not overhead we added.

Polled together, the cheap number inherited the expensive one's latency
and the CPU gauge updated about as often as a lorry reverses. Split, the
CPU gauge is live and GPU/NPU refresh as fast as the OS will answer.
"""
from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Callable

from pantherlake_ai_core import power, telemetry

from . import energy


class TelemetryPoller:
    def __init__(
        self,
        cpu_interval: float = 0.3,
        device_interval: float = 1.0,
        power_interval: float = 1.0,
        is_idle: Callable[[], bool] = lambda: True,
    ):
        # Both are the *gap between* readings, not the period: add ~0.1s for
        # CPU and 3.5-4.5s for devices to get the real cadence.
        #
        # The device gap used to be 0.2s, on the reasoning that the query is
        # already the bottleneck so a longer pause only makes the gauges
        # staler. True as far as it goes, but it meant Get-Counter ran back
        # to back forever, expanding a 650-instance wildcard the whole time
        # a demo was running. Measured on the smart-city feed: 14.3 fps with
        # that, 20.7 fps with device polling off -- 45% of a video brick's
        # frame rate spent on the gauges watching it. Not the GIL (the
        # parse is 0.6ms); the OS-side work of the query itself.
        #
        # At 1.0s the gauges refresh every ~5s instead of ~4s and the frame
        # rate comes back. Cheap trade.
        self._cpu_interval = cpu_interval
        self._device_interval = device_interval
        # Power is a third loop, and a free one: the energy counters are read
        # in-process in well under a millisecond (pantherlake_ai_core.power).
        # `is_idle` says when no demo is running, so a sample can feed the
        # idle baseline energy-per-result is measured against.
        self._power_interval = power_interval
        self._is_idle = is_idle
        self._power: dict = {"available": energy.meter.available, "battery": power.battery()}
        self._snapshot = telemetry.Utilization(available=False)
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        for target in (self._poll_cpu, self._poll_devices, self._poll_power):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            # Each loop wakes from its wait() immediately on the event, but a
            # device read already in flight (~2.3s) has to finish first.
            thread.join(timeout=5.0)
        self._threads = []  # so a later start() actually starts again

    def snapshot(self) -> dict:
        with self._lock:
            return {**asdict(self._snapshot), "power": dict(self._power)}

    def _poll_cpu(self) -> None:
        while not self._stop_event.is_set():
            try:
                cpu = telemetry.read_cpu()
            except Exception:
                cpu = None
            with self._lock:
                self._snapshot.cpu_percent = cpu
            self._stop_event.wait(self._cpu_interval)

    def _poll_devices(self) -> None:
        while not self._stop_event.is_set():
            try:
                reading = telemetry.read_devices()
            except Exception:
                reading = telemetry.Utilization(available=False)
            # Only the device half -- cpu_percent belongs to the other loop.
            with self._lock:
                self._snapshot.available = reading.available
                self._snapshot.gpus = reading.gpus
                self._snapshot.npu_percent = reading.npu_percent
                self._snapshot.npu_name = reading.npu_name
            self._stop_event.wait(self._device_interval)

    def _poll_power(self) -> None:
        """Watts per rail from the energy counters' deltas, once a second."""
        previous = energy.meter.read()
        while not self._stop_event.wait(self._power_interval):
            current = energy.meter.read()
            reading: dict = {"available": False, "battery": power.battery()}
            watts = power.watts_between(previous, current) if previous and current else {}
            if "package" in watts:
                package = watts["package"]
                if self._is_idle():
                    energy.note_idle(package)
                accounted = watts.get("cores", 0.0) + watts.get("graphics", 0.0)
                reading.update(
                    available=True,
                    package_w=round(package, 2),
                    cores_w=_rounded(watts.get("cores")),
                    graphics_w=_rounded(watts.get("graphics")),
                    memory_w=_rounded(watts.get("memory")),
                    # The package minus the rails it has: the NPU is in here,
                    # with the memory controller and I/O, having no rail of its own.
                    rest_w=round(max(package - accounted, 0.0), 2),
                    idle_w=_rounded(energy.idle_watts()),
                )
            previous = current
            with self._lock:
                self._power = reading


def _rounded(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None
