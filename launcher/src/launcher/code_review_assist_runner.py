"""Holds the code-review-assist brick's session (a single composed LLM) and
exposes its blocking call for the web UI to run off the event loop (via
`fastapi.concurrency.run_in_threadpool`).
"""
from __future__ import annotations

import threading

from code_review_assist.session import CodeReviewSession
from code_review_assist.types import ReviewResult
from pantherlake_ai_core.engine import Engine

from . import activity, events

_DEMO_ID = "code-review-assist"


class CodeReviewAssistRunner:
    def __init__(self) -> None:
        self._session: CodeReviewSession | None = None
        self._engine: str | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def review(
        self, *, engine: str, device: str, folder: str | None, against: str, diff_text: str | None
    ) -> ReviewResult:
        """Blocking -- loads the LLM the first time or when the engine/device
        changes, then makes the commit-message and review-notes calls."""
        with self._lock:
            activity.set_active(_DEMO_ID, engine=engine, device=device)
            try:
                if self._session is None or self._engine != engine or self._device != device:
                    self._session = CodeReviewSession(Engine(engine), compute_device=device)
                    self._engine = engine
                    self._device = device

                def on_ready() -> None:
                    events.set_phase(_DEMO_ID, "running", "Reviewing diff...")

                def on_downloading() -> None:
                    events.set_phase(_DEMO_ID, "loading", f"Downloading model (first run only, engine={engine})...")

                # The LLM itself is lazy (built on the session's first
                # review() call, reused after) -- this "loading" phase may
                # or may not turn into real work; on_ready flips it to
                # "running" only once the model is actually ready.
                events.set_phase(_DEMO_ID, "loading", f"Loading model (engine={engine}, device={device})...")
                try:
                    result = self._session.review(
                        folder=folder, against=against, diff_text=diff_text, on_ready=on_ready, on_downloading=on_downloading
                    )
                except Exception as exc:
                    events.set_phase(_DEMO_ID, "error", str(exc))
                    raise
                events.clear_phase(_DEMO_ID)
                return result
            finally:
                activity.clear_active(_DEMO_ID)
