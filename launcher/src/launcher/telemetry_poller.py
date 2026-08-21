"""Background thread that periodically samples local hardware utilization
(CPU/GPU/NPU), so `/api/telemetry` can respond instantly from a cached
reading instead of paying the real query's cost (~1-3s on Windows) on
every request.
"""
from __future__ import annotations

import threading
from dataclasses import asdict

from pantherlake_ai_core import telemetry


class TelemetryPoller:
    def __init__(self, interval: float = 3.0):
        self._interval = interval
        self._snapshot = telemetry.Utilization(available=False)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def snapshot(self) -> dict:
        with self._lock:
            return asdict(self._snapshot)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                reading = telemetry.read()
            except Exception:
                reading = telemetry.Utilization(available=False)
            with self._lock:
                self._snapshot = reading
            self._stop_event.wait(self._interval)
