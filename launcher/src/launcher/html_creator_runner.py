"""Holds the html-creator brick's session (a single composed LLM) and
exposes its blocking calls for the web UI to run off the event loop (via
`fastapi.concurrency.run_in_threadpool`).

It also keeps the page last made (`revisions`), so that one part of it can
be written again without the rest: see `change`.
"""
from __future__ import annotations

import threading

from html_creator import pictures as picture_kit
from html_creator.session import HtmlCreatorSession
from html_creator.types import HtmlResult
from pantherlake_ai_core.engine import Engine

from . import activity, energy, events, generation, metrics
from .errors import Conflict
from .page_revisions import PageRevisions

_DEMO_ID = "html-creator"


class HtmlCreatorRunner:
    def __init__(self) -> None:
        self._session: HtmlCreatorSession | None = None
        self._engine: str | None = None
        self._device: str | None = None
        self._lock = threading.Lock()
        self.revisions = PageRevisions()
        # The engine and the chip the page in `revisions` was written with: a
        # change to it is written with the same, even after the model was
        # unloaded from the hardware panel.
        self._made_with: tuple[str, str] | None = None

    def _use(self, engine: str, device: str) -> HtmlCreatorSession:
        if self._session is None or self._engine != engine or self._device != device:
            self._session = HtmlCreatorSession(Engine(engine), compute_device=device)
            self._engine = engine
            self._device = device
        return self._session

    def generate(
        self, *, engine: str, device: str, mode: str, prompt: str | None, folder: str | None,
        pictures: str | None = None, repeatable: bool = False,
    ) -> HtmlResult:
        """Blocking -- loads the LLM the first time or when the engine/device
        changes, then generates the HTML."""
        with self._lock:
            activity.set_active(_DEMO_ID, engine=engine, device=device)
            try:
                session = self._use(engine, device)
                started = []

                def on_ready() -> None:
                    events.set_phase(_DEMO_ID, "running", "Generating HTML...")
                    # The answer's energy window opens here, once the model is
                    # loaded: loading a 30B model isn't what an answer costs.
                    started.append(energy.mark())

                def on_downloading() -> None:
                    events.set_phase(_DEMO_ID, "loading", f"Downloading model (first run only, engine={engine})...")

                events.set_phase(_DEMO_ID, "loading", f"Loading model (engine={engine}, device={device})...")
                live = generation.get(_DEMO_ID)
                try:
                    result = session.generate(
                        mode=mode, prompt=prompt, folder=folder, pictures=pictures, repeatable=repeatable,
                        on_ready=on_ready,
                        on_downloading=on_downloading, control=live.begin(),
                    )
                except Exception as exc:
                    events.set_phase(_DEMO_ID, "error", str(exc))
                    raise
                finally:
                    live.end()
                if result.stats is not None and started:
                    result.stats.energy = energy.since(started[-1], _DEMO_ID)
                if result.stats is not None:
                    metrics.report(_DEMO_ID, result.stats.tokens_per_second, "tok/s", sticky=True)
                # The page as written, and the pictures it was offered: what a
                # change to one of its parts starts from.
                offered: list[picture_kit.Picture] = []
                if mode == "landing_page" and pictures and pictures.strip() and result.pictures_offered:
                    try:
                        offered = picture_kit.load(pictures.strip())[0]
                    except (OSError, ValueError):
                        offered = []
                self.revisions.start(result.html_source or result.html, offered)
                self._made_with = (engine, device)
                events.clear_phase(_DEMO_ID)
                return result
            finally:
                activity.clear_active(_DEMO_ID)

    def change(self, *, version: int, changes: list[dict]) -> list[dict]:
        """Blocking. Writes again the parts of the page that were pointed
        at, each as its comment asks, and keeps the page they leave as its
        next version. Returns what became of each change."""
        if not self.revisions.available or self._made_with is None:
            raise Conflict("There is no page to change yet: generate one first.")
        self.revisions.check(version, changes)  # refused before the model is waited for
        if not self._lock.acquire(blocking=False):
            raise Conflict("A page is being written -- wait for it, or stop it first.")
        engine, device = self._made_with
        try:
            activity.set_active(_DEMO_ID, engine=engine, device=device)
            session = self._use(engine, device)
            live = generation.get(_DEMO_ID)
            stop_asked = [False]

            def on_downloading() -> None:
                events.set_phase(_DEMO_ID, "loading", f"Downloading model (first run only, engine={engine})...")

            def on_start(index: int, total: int, change: dict) -> None:
                where[0] = f"Changing part {index} of {total}..." if total > 1 else "Changing the part..."
                events.set_phase(_DEMO_ID, "loading", f"Loading model (engine={engine}, device={device})...")

            def write(source, change, pictures):
                if change.get("picture"):
                    raise ValueError("This page's pictures are files from a folder: there is no image model here to draw one again.")
                control = live.begin()
                try:
                    return session.revise(
                        source, int(change["part"]), change["comment"], seen=change.get("seen"), pictures=pictures or None,
                        on_ready=lambda: events.set_phase(_DEMO_ID, "running", where[0]),
                        on_downloading=on_downloading, control=control,
                    )
                finally:
                    stop_asked[0] = stop_asked[0] or bool(control.should_stop and control.should_stop())
                    live.end()

            where = ["Changing the part..."]
            try:
                results = self.revisions.change(version, changes, write, on_start, stopped=lambda: stop_asked[0])
            except Exception as exc:
                events.set_phase(_DEMO_ID, "error", str(exc))
                raise
            rates = [result["tokens_per_second"] for result in results if result.get("tokens_per_second")]
            if rates:
                metrics.report(_DEMO_ID, rates[-1], "tok/s", sticky=True)
            events.clear_phase(_DEMO_ID)
            return results
        finally:
            activity.clear_active(_DEMO_ID)
            self._lock.release()
