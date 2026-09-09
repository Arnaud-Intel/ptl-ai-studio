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

from pantherlake_ai_core import telemetry


class TelemetryPoller:
    def __init__(self, cpu_interval: float = 0.3, device_interval: float = 0.2):
        # Both are the *gap between* readings, not the period: add ~0.1s for
        # CPU and ~2.3s for devices to get the real cadence. The device gap
        # is deliberately small -- the query is already the bottleneck, and
        # a longer pause only makes the gauges staler.
        self._cpu_interval = cpu_interval
        self._device_interval = device_interval
        self._snapshot = telemetry.Utilization(available=False)
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        for target in (self._poll_cpu, self._poll_devices):
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
            return asdict(self._snapshot)

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
