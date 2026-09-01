"""Holds the doc-qa brick's session (embedder + LLM + index) and exposes
its blocking calls for the web UI to run off the event loop (via
`fastapi.concurrency.run_in_threadpool`).
"""
from __future__ import annotations

import threading

from doc_qa.pipeline import DocQASession
from doc_qa.types import Answer
from pantherlake_ai_core.engine import Engine

from . import activity, events

_DEMO_ID = "doc-qa"


class DocQARunner:
    def __init__(self) -> None:
        self._session: DocQASession | None = None
        self._engine: str | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def ingest(self, *, folder: str, engine: str, device: str, reindex: bool) -> tuple[int, str]:
        """Blocking -- loads the embedder/LLM the first time or when the
        engine/device changes, then (re)builds or loads the cached index."""
        with self._lock:
            activity.set_active(_DEMO_ID, engine=engine, device=device)
            try:
                if self._session is None or self._engine != engine or self._device != device:
                    events.set_phase(_DEMO_ID, "loading", f"Loading model (engine={engine}, device={device})...")
                    self._session = DocQASession(Engine(engine), device=device)
                    self._engine = engine
                    self._device = device
                events.set_phase(_DEMO_ID, "running", "Indexing documents...")
                try:
                    count = self._session.ingest(folder, force=reindex)
                except Exception as exc:
                    events.set_phase(_DEMO_ID, "error", str(exc))
                    raise
                events.clear_phase(_DEMO_ID)
                return count, str(self._session.folder)
            finally:
                activity.clear_active(_DEMO_ID)

    def ask(self, *, question: str, top_k: int) -> Answer:
        """Blocking."""
        with self._lock:
            if self._session is None:
                raise RuntimeError("Ingest a folder first.")
            activity.set_active(_DEMO_ID, engine=self._engine, device=self._device)
            events.set_phase(_DEMO_ID, "running", "Answering...")
            try:
                answer = self._session.ask(question, top_k=top_k)
            except Exception as exc:
                events.set_phase(_DEMO_ID, "error", str(exc))
                raise
            finally:
                activity.clear_active(_DEMO_ID)
            events.clear_phase(_DEMO_ID)
            return answer
