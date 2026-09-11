"""Runs the expense-extract brick's two-stage batch pipeline on a
background thread and forwards progress events into an asyncio queue the
web UI drains over a WebSocket -- same background-thread/queue shape as
LiveTranslationRunner. A single demo instance runs at a time.

The one thing genuinely different from every other runner in this file:
`expense_extract.pipeline.run()` itself spans two concurrently-running
worker threads (OCR and LLM structuring), each pinned to its own device,
for the whole call -- so this marks *both* stages active up front and
clears both when the call returns, rather than one device around one
blocking call like every single-stage runner does.
"""
from __future__ import annotations

import asyncio
import threading

from expense_extract import pipeline
from expense_extract.types import totals_by_currency
from pantherlake_ai_core.engine import Engine

from . import activity, events, worker

_DEMO_ID = "expense-extract"
_STAGES = ("ocr", "llm")


class ExpenseExtractRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue,
        folder: str,
        ocr_engine: Engine,
        ocr_device: str,
        llm_engine: Engine,
        llm_device: str,
    ) -> None:
        worker.refuse_if_busy(_DEMO_ID, self._thread, self._stop_event)

        self.error = None
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def emit(message: dict) -> None:
            asyncio.run_coroutine_threadsafe(queue.put(message), loop)

        def on_ocr_start(path, index, total) -> None:
            emit({"type": "ocr_progress", "file": path.name, "index": index, "total": total})

        def on_structured(line) -> None:
            emit({"type": "structured", "line": line.to_dict()})

        def target() -> None:
            activity.set_active(_DEMO_ID, engine=ocr_engine.value, device=ocr_device, stage="ocr", stage_label="OCR")
            activity.set_active(_DEMO_ID, engine=llm_engine.value, device=llm_device, stage="llm", stage_label="Structuring")
            events.set_phase(_DEMO_ID, "running", "Extracting receipts...", stage="ocr")
            events.set_phase(_DEMO_ID, "running", "Structuring extracted text...", stage="llm")
            try:
                results = pipeline.run(
                    folder=folder,
                    ocr_engine=ocr_engine,
                    ocr_device=ocr_device,
                    llm_engine=llm_engine,
                    llm_device=llm_device,
                    on_ocr_start=on_ocr_start,
                    on_structured=on_structured,
                    stop_event=stop_event,
                )
                ok = [r for r in results if r.error is None]
                emit({"type": "done", "count": len(results), "structured": len(ok),
                      "needs_review": sum(r.needs_review for r in results), "totals": totals_by_currency(results)})
            except Exception as exc:  # surfaced to the UI, not silently dropped
                self.error = str(exc)
                for stage in _STAGES:
                    events.set_phase(_DEMO_ID, "error", str(exc), stage=stage)
                emit({"type": "error", "message": str(exc)})
            else:
                for stage in _STAGES:
                    events.clear_phase(_DEMO_ID, stage=stage)
            finally:
                for stage in _STAGES:
                    activity.clear_active(_DEMO_ID, stage=stage)

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not worker.request_stop(_DEMO_ID, self._thread, self._stop_event, stages=_STAGES):
            return
        self._thread = None
