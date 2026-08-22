"""Runs the smart-recall brick's continuous capture loop on a background
thread (events streamed over a WebSocket, same shape as ExpenseExtractRunner
-- OCR and embedding are marked active as two simultaneous stages, not one
device around one blocking call) plus a request/response search action
(same shape as ScreenOcrRunner) for querying whatever's been indexed so far.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict

from pantherlake_ai_core.engine import Engine
from smart_recall import pipeline

from . import activity

_DEMO_ID = "smart-recall"


class SmartRecallRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self.error: str | None = None
        self._search_index: pipeline.RecallIndex | None = None
        self._search_device: str | None = None
        self._search_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        screen_index: int,
        interval_seconds: float,
        ocr_engine: Engine,
        ocr_device: str,
        embed_engine: Engine,
        embed_device: str,
    ) -> None:
        if self.running:
            raise RuntimeError("smart-recall is already running")

        self.error = None
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def emit(message: dict) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(message), loop)

        def on_event(event: pipeline.CaptureEvent) -> None:
            if event.kind == "indexed":
                emit({"type": "indexed", "chunk": asdict(event.chunk), "timestamp": event.timestamp})
            else:
                emit({"type": "skipped", "reason": event.reason})

        def target() -> None:
            activity.set_active(_DEMO_ID, engine=ocr_engine.value, device=ocr_device, stage="ocr", stage_label="OCR")
            activity.set_active(
                _DEMO_ID, engine=embed_engine.value, device=embed_device, stage="embed", stage_label="Indexing"
            )
            try:
                pipeline.run(
                    screen_index=screen_index,
                    interval_seconds=interval_seconds,
                    ocr_engine=ocr_engine,
                    ocr_device=ocr_device,
                    embed_engine=embed_engine,
                    embed_device=embed_device,
                    on_event=on_event,
                    stop_event=stop_event,
                )
            except Exception as exc:  # surfaced to the UI, not silently dropped
                self.error = str(exc)
                emit({"type": "error", "message": str(exc)})
            finally:
                activity.clear_active(_DEMO_ID, stage="ocr")
                activity.clear_active(_DEMO_ID, stage="embed")
                emit({"type": "stopped"})

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        self._thread = None

    def reset(self) -> None:
        if self.running:
            raise RuntimeError("Stop recording before resetting the index.")
        with self._search_lock:
            pipeline.reset_index()
            self._search_index = None
            self._search_device = None

    def search(self, *, question: str, top_k: int, device: str) -> list:
        """Blocking -- loads (or reuses) the search-side index, which only
        needs the embedder the index was actually built with, not OCR."""
        with self._search_lock:
            if self._search_index is None or self._search_device != device:
                self._search_index = pipeline.RecallIndex(device=device)
                self._search_device = device
            activity.set_active(
                _DEMO_ID, engine=self._search_index.embed_engine.value, device=device,
                stage="search", stage_label="Search",
            )
            try:
                return self._search_index.search(question, top_k=top_k)
            finally:
                activity.clear_active(_DEMO_ID, stage="search")
