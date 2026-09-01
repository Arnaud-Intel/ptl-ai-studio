"""Holds the screen-ocr brick's session (loaded extractor) and exposes its
blocking call for the web UI to run off the event loop (via
`fastapi.concurrency.run_in_threadpool`). Same shape as doc_qa_runner.py --
OCR is a one-call-in-one-result-out action, not a stream.
"""
from __future__ import annotations

import threading

from pantherlake_ai_core.engine import Engine
from screen_ocr.pipeline import OcrSession
from screen_ocr.types import ExtractionResult

from . import activity, events

_DEMO_ID = "screen-ocr"


class ScreenOcrRunner:
    def __init__(self) -> None:
        self._session: OcrSession | None = None
        self._engine: str | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def extract(self, *, image, engine: str, device: str, translate: bool) -> ExtractionResult:
        """Blocking -- loads the extractor the first time or when the
        engine/device changes, then runs one image through it."""
        with self._lock:
            activity.set_active(_DEMO_ID, engine=engine, device=device)
            try:
                if self._session is None or self._engine != engine or self._device != device:
                    events.set_phase(_DEMO_ID, "loading", f"Loading model (engine={engine}, device={device})...")

                    def on_downloading() -> None:
                        events.set_phase(_DEMO_ID, "loading", f"Downloading model (first run only, engine={engine})...")

                    self._session = OcrSession(Engine(engine), device=device, on_downloading=on_downloading)
                    self._engine = engine
                    self._device = device
                events.set_phase(_DEMO_ID, "running", "Extracting text...")
                try:
                    result = self._session.extract(image, translate=translate)
                except Exception as exc:
                    events.set_phase(_DEMO_ID, "error", str(exc))
                    raise
                events.clear_phase(_DEMO_ID)
                return result
            finally:
                activity.clear_active(_DEMO_ID)
