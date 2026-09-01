"""Holds the html-creator brick's session (a single composed LLM) and
exposes its blocking call for the web UI to run off the event loop (via
`fastapi.concurrency.run_in_threadpool`).
"""
from __future__ import annotations

import threading

from html_creator.session import HtmlCreatorSession
from html_creator.types import HtmlResult
from pantherlake_ai_core.engine import Engine

from . import activity, events

_DEMO_ID = "html-creator"


class HtmlCreatorRunner:
    def __init__(self) -> None:
        self._session: HtmlCreatorSession | None = None
        self._engine: str | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def generate(
        self, *, engine: str, device: str, mode: str, prompt: str | None, folder: str | None
    ) -> HtmlResult:
        """Blocking -- loads the LLM the first time or when the engine/device
        changes, then generates the HTML."""
        with self._lock:
            activity.set_active(_DEMO_ID, engine=engine, device=device)
            try:
                if self._session is None or self._engine != engine or self._device != device:
                    events.set_phase(_DEMO_ID, "loading", f"Loading model (engine={engine}, device={device})...")
                    self._session = HtmlCreatorSession(Engine(engine), compute_device=device)
                    self._engine = engine
                    self._device = device
                events.set_phase(_DEMO_ID, "running", "Generating HTML...")
                try:
                    result = self._session.generate(mode=mode, prompt=prompt, folder=folder)
                except Exception as exc:
                    events.set_phase(_DEMO_ID, "error", str(exc))
                    raise
                events.clear_phase(_DEMO_ID)
                return result
            finally:
                activity.clear_active(_DEMO_ID)
