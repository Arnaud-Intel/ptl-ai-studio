"""Holds the page-agent brick's three models and runs one build at a time,
off the event loop (via `fastapi.concurrency.run_in_threadpool`).

A build takes the better part of a minute and happens on three chips, two of
them at once, so beside the usual phase and activity reporting this runner
keeps a step-by-step account of the build in hand -- which step is on which
chip, the plan as soon as it is written, each picture as soon as it is
drawn -- for the page to ask for while it waits (`progress()`).
"""
from __future__ import annotations

import copy
import shutil
import tempfile
import threading
import time
from pathlib import Path

from page_agent.conductor import CHECK, IMAGES, PAGE, PLAN, STEPS, PageAgent
from page_agent.plan import PagePlan
from page_agent.types import Assignment, DrawnPicture, PageResult
from pantherlake_ai_core.engine import Engine

from . import activity, energy, events, generation, metrics
from .errors import Conflict

_DEMO_ID = "page-agent"
_ENGINE = Engine.OPENVINO.value
_CONDUCTOR = "conductor"

_LABELS = {PLAN: "Plan", IMAGES: "Pictures", PAGE: "Page", CHECK: "Check"}
_HELD_LABELS = {PLAN: "planner", IMAGES: "image model", PAGE: "coding model"}
# What the status line says while a step is at work.
_DOING = {
    (PLAN, "loading"): "loading the planner on {device}",
    (PLAN, "running"): "planning on {device}",
    (IMAGES, "loading"): "loading the image model on {device}",
    (IMAGES, "running"): "drawing {detail} on {device}",
    (PAGE, "loading"): "loading the coding model on {device}",
    (PAGE, "running"): "writing the page on {device}",
    (CHECK, "running"): "checking the page",
}


def _idle() -> dict:
    return {"running": False, "run": 0, "request": "", "assignment": None, "steps": [], "plan": None, "error": None}


class PageAgentRunner:
    def __init__(self) -> None:
        # The names `loaded.info` / `loaded.unload` look for: the session, its
        # engine, and the chip to list it under -- the coding model's, the
        # largest of the three.
        self._session: PageAgent | None = None
        self._engine: str | None = None
        self._device: str | None = None
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = _idle()
        self._work_dir: Path | None = None
        # True once a build has started in this process: how the launcher
        # knows to end without the runtimes' teardown (page_agent/leaving.py).
        self.has_built = False

    # ------------------------------------------------------------- for the page

    def held(self) -> list[dict]:
        """The models in memory between two builds, one entry per chip: the
        hardware panel lists each under its own chip. (Closing any of them
        unloads all three -- `loaded.unload` drops the session.)"""
        session = self._session
        if session is None:
            return []
        return [
            {"engine": _ENGINE, "device": device, "stage": step, "stage_label": _HELD_LABELS[step]}
            for step, device in session.loaded().items()
        ]

    def progress(self) -> dict:
        """The build in hand, or the last one: its steps, plan and pictures."""
        with self._state_lock:
            state = copy.deepcopy(self._state)
        now = time.time()
        for step in state["steps"]:
            since, spent = step.pop("since"), step.pop("spent")
            if since is not None:
                spent += now - since
            step["seconds"] = round(spent, 1) if (spent or step["state"] != "pending") else None
        return state

    def picture(self, name: str) -> Path | None:
        """The file of one of the current build's pictures, once drawn. Only
        names the plan gave: this is reached with a name from a URL."""
        with self._state_lock:
            plan, folder = self._state["plan"], self._work_dir
        if plan is None or folder is None:
            return None
        if not any(picture["name"] == name and picture["ready"] for picture in plan["pictures"]):
            return None
        path = folder / name
        return path if path.is_file() else None

    # ------------------------------------------------------------------ a build

    def _on_step(self, step: str, state: str, device: str, detail: str) -> None:
        with self._state_lock:
            entry = next(item for item in self._state["steps"] if item["id"] == step)
            # Time at work, added up: the pictures step works twice when the
            # page turns out to need more of them.
            now = time.time()
            if state in ("loading", "running") and entry["since"] is None:
                entry["since"] = now
            elif state in ("done", "failed") and entry["since"] is not None:
                entry["spent"] += now - entry["since"]
                entry["since"] = None
            entry.update(state=state, device=device, detail=detail)
            doing = [
                _DOING[(item["id"], item["state"])].format(device=item["device"], detail=item["detail"] or "the pictures")
                for item in self._state["steps"]
                if (item["id"], item["state"]) in _DOING
            ]
            loading_only = all(item["state"] != "running" for item in self._state["steps"])
        if state in ("loading", "running"):
            activity.set_active(_DEMO_ID, engine=_ENGINE, device=device, stage=step, stage_label=_LABELS[step].lower())
        else:
            activity.clear_active(_DEMO_ID, stage=step)
        if doing:
            message = " · ".join(doing)
            events.set_phase(_DEMO_ID, "loading" if loading_only else "running", message[0].upper() + message[1:] + "...")

    def _on_plan(self, plan: PagePlan) -> None:
        with self._state_lock:
            self._state["plan"] = {
                "title": plan.title,
                "style": plan.style,
                "sections": plan.sections,
                "notes": plan.notes,
                "pictures": [
                    {"name": p.name, "width": p.width, "height": p.height, "prompt": p.prompt, "ready": False, "seconds": None}
                    for p in plan.pictures
                ],
            }

    def _on_picture(self, drawn: DrawnPicture) -> None:
        with self._state_lock:
            pictures = self._state["plan"]["pictures"]
            for picture in pictures:
                if picture["name"] == drawn.name:
                    picture.update(ready=True, seconds=drawn.seconds)
                    return
            # Not in the plan: one the page turned out to need (page_agent/repeats.py).
            pictures.append(
                {"name": drawn.name, "width": drawn.width, "height": drawn.height, "prompt": drawn.prompt, "ready": True,
                 "seconds": drawn.seconds, "extra": True}
            )

    def build(self, *, request: str, assignment: Assignment) -> PageResult:
        """Blocking. Plans, draws, writes and checks one page."""
        if not self._lock.acquire(blocking=False):
            raise Conflict("A page is being built already -- wait for it, or stop it first.")
        try:
            self.has_built = True
            if self._session is None:
                self._session = PageAgent(Engine.OPENVINO)
                self._engine = _ENGINE
            self._device = assignment.page
            previous, self._work_dir = self._work_dir, Path(tempfile.mkdtemp(prefix="page-agent-"))
            if previous is not None:
                shutil.rmtree(previous, ignore_errors=True)  # the last build's pictures: they are in its page
            with self._state_lock:
                run = self._state["run"] + 1
                self._state = {
                    "running": True,
                    "run": run,
                    "request": request,
                    "assignment": {
                        "planner": assignment.planner, "images": assignment.images, "page": assignment.page,
                        "together": assignment.together,
                    },
                    "steps": [
                        {"id": step, "label": _LABELS[step], "device": device, "state": "pending", "detail": "",
                         "since": None, "spent": 0.0}
                        for step, device in zip(STEPS, (assignment.planner, assignment.images, assignment.page, "CPU"))
                    ],
                    "plan": None,
                    "error": None,
                }

            def on_downloading() -> None:
                events.set_phase(_DEMO_ID, "loading", "Downloading a model (first run only: the image model is 9 GB)...")

            # The conductor is code on the CPU, and says so in the hardware panel.
            activity.set_active(_DEMO_ID, engine=_ENGINE, device="CPU", stage=_CONDUCTOR, stage_label=_CONDUCTOR)
            events.set_phase(_DEMO_ID, "loading", "Starting...")
            live = generation.get(_DEMO_ID)
            started = energy.mark()
            try:
                result = self._session.build(
                    request,
                    assignment=assignment,
                    work_dir=self._work_dir,
                    on_step=self._on_step,
                    on_plan=self._on_plan,
                    on_picture=self._on_picture,
                    on_downloading=on_downloading,
                    control=live.begin(),
                )
            except Exception as exc:
                with self._state_lock:
                    self._state["error"] = str(exc)
                    for step in self._state["steps"]:
                        if step["state"] in ("loading", "running"):
                            step["state"] = "failed"
                            step["spent"] += time.time() - (step["since"] or time.time())
                            step["since"] = None
                events.set_phase(_DEMO_ID, "error", str(exc))
                raise
            finally:
                live.end()
                for stage in (*STEPS, _CONDUCTOR):
                    activity.clear_active(_DEMO_ID, stage=stage)
                with self._state_lock:
                    self._state["running"] = False
            if result.stats is not None:
                result.stats.energy = energy.since(started, _DEMO_ID)
                metrics.report(_DEMO_ID, result.stats.tokens_per_second, "tok/s", sticky=True)
            events.clear_phase(_DEMO_ID)
            return result
        finally:
            self._lock.release()
