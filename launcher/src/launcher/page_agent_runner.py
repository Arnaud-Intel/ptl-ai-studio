"""Holds the page-agent brick's three models and runs one build at a time,
off the event loop (via `fastapi.concurrency.run_in_threadpool`).

A build takes the better part of a minute and happens on three chips, two of
them at once, so beside the usual phase and activity reporting this runner
keeps a step-by-step account of the build in hand -- which step is on which
chip, the plan as soon as it is written, each picture as soon as it is
drawn -- for the page to ask for while it waits (`progress()`).

Each of the three models has its own figure in its own unit: tokens per
second for the planner and the coding model, images per minute for the
image model. They are reported under the step's own stage, which is what
puts each one under its own chip in the hardware panel -- while the step
works, and as "last" for as long as its model stays loaded.
"""
from __future__ import annotations

import copy
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

from html_creator import pictures as picture_kit
from html_creator import revise as revising
from html_creator.types import Revised
from page_agent import checks as checking
from page_agent.conductor import CHECK, IMAGES, PAGE, PLAN, STEPS, PageAgent
from page_agent.plan import PagePlan
from page_agent.types import Assignment, DrawnPicture, PageResult
from pantherlake_ai_core.engine import Engine

from . import activity, energy, events, generation, metrics
from .errors import Conflict
from .page_revisions import PageRevisions

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
        # The page last built, so that one part of it can be written again
        # (`change`), and what that takes from its build: the plan its
        # checks are made against, and the chip that writes pages.
        self.revisions = PageRevisions()
        self._built: PageResult | None = None

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
        """The build in hand, or the last one: its steps -- each with the
        time it has taken and how fast it works -- its plan and pictures."""
        with self._state_lock:
            state = copy.deepcopy(self._state)
        now = time.time()
        # The coding model's rate while it writes is counted from its tokens
        # as they come (generation.py): the conductor only knows it at the end.
        live = {m["stage"]: m for m in metrics.snapshot() if m["demo_id"] == _DEMO_ID and not m["sticky"]}
        for step in state["steps"]:
            since, spent = step.pop("since"), step.pop("spent")
            if since is not None:
                spent += now - since
            step["seconds"] = round(spent, 1) if (spent or step["state"] != "pending") else None
            if step["state"] == "running" and step["id"] in live:
                step["rate"] = {"value": live[step["id"]]["value"], "unit": live[step["id"]]["unit"]}
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

    def _on_metric(self, step: str, value: float, unit: str, final: bool) -> None:
        """How fast a step works, as the conductor tells it: kept with the
        step for the panel's own tiles, and reported under the step's stage
        for the hardware panel -- to stay ("last 19 tok/s") once it is the
        step's last word."""
        with self._state_lock:
            entry = next(item for item in self._state["steps"] if item["id"] == step)
            entry["rate"] = {"value": round(float(value), 2), "unit": unit}
        metrics.report(_DEMO_ID, value, unit, stage=step, sticky=final)

    def _on_plan(self, plan: PagePlan) -> None:
        with self._state_lock:
            self._state["plan"] = {
                "title": plan.title,
                "headline": plan.headline,
                "style": plan.style,
                "sections": plan.sections,
                "offers": plan.offers,
                "notes": plan.notes,
                "pictures": [
                    {"name": p.name, "width": p.width, "height": p.height, "prompt": p.prompt, "role": p.role,
                     "ready": False, "seconds": None}
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
                         "since": None, "spent": 0.0, "rate": None}
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
            # Under the page step's own stage, so that the rate counted from
            # its tokens shows under the chip that writes it.
            live = generation.get(_DEMO_ID, PAGE)
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
                    on_metric=self._on_metric,
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
            self._built = result
            self.revisions.start(result.html_source or "", [
                picture_kit.Picture(drawn.name, drawn.path, "image/jpeg", drawn.width, drawn.height, drawn.prompt)
                for drawn in result.pictures
            ])
            events.clear_phase(_DEMO_ID)
            return result
        finally:
            self._lock.release()

    # ----------------------------------------------------------- after a build

    def checks(self) -> list[dict]:
        """The build's own checks, made of the page as it is now: what they
        say of a page that has been changed in parts. They report; nothing
        is written again because of them."""
        built = self._built
        if built is None or not self.revisions.available:
            return []
        source = self.revisions.source
        used = picture_kit.referenced(source, [picture.name for picture in built.plan.pictures])
        return [
            {"name": check.name, "passed": check.passed, "detail": check.detail}
            for check in checking.review(source, built.plan, used, not source.rstrip().lower().endswith("</html>"))
        ]

    def pictures_in_page(self) -> dict[int, str]:
        """The parts of the page that show one of the build's pictures, by
        their number: an <img> of it, a part whose own style puts it behind
        its text, a part whose class has a rule that does. A comment on
        one of these can be about the picture, and be sent to the image
        model."""
        if not self.revisions.available:
            return {}
        source, names = self.revisions.source, [picture.name for picture in self.revisions.pictures]
        found: dict[int, str] = {}
        numbered = [part for part in revising.parts(source) if part.number is not None]
        for part in numbered:
            shown = picture_kit.referenced(source[part.start:part.opened], names)
            if len(shown) == 1:
                found[part.number] = shown[0]
        css = "\n".join(revising._STYLE_BLOCK.findall(source))
        for before, body in revising._rules(revising._COMMENT.sub("", css)):
            shown = picture_kit.referenced(body, names)
            if len(shown) != 1 or before.startswith("@"):
                continue
            for selector in before.split(","):
                classes = set(re.findall(r"\.(-?[_a-zA-Z][\w-]*)", selector.split()[-1] if selector.split() else ""))
                for part in numbered:
                    if classes and classes <= set(part.classes) and part.number not in found:
                        found[part.number] = shown[0]
        return found

    def _draw_again(self, session, built, source: str, change: dict, pictures: list) -> Revised:
        """The picture named in `change` drawn again as its comment asks,
        and the page showing the new one wherever it showed the old."""
        name, comment, part = str(change["picture"]), change["comment"], int(change["part"])
        picture = next((kept for kept in pictures if kept.name == name), None)
        if picture is None or not picture_kit.referenced(source, [name]):
            raise ValueError(f"The page shows no picture called {name}.")
        stem, dot, suffix = name.rpartition(".")
        stem = re.sub(r"-r\d+$", "", stem)
        taken, turn = {kept.name for kept in pictures}, 2
        while f"{stem}-r{turn}{dot}{suffix}" in taken:
            turn += 1
        new_name = f"{stem}-r{turn}{dot}{suffix}"
        started = time.perf_counter()
        drawn = session.redraw(
            picture, comment, assignment=built.assignment, work_dir=self._work_dir, name=new_name, on_step=self._on_step,
            on_downloading=lambda: events.set_phase(_DEMO_ID, "loading", "Downloading a model (first run only)..."),
        )
        self.revisions.add_picture(
            picture_kit.Picture(drawn.name, drawn.path, "image/jpeg", drawn.width, drawn.height, drawn.prompt)
        )
        # Beside the build's own pictures, for the page to show and to fetch.
        with self._state_lock:
            if self._state.get("plan"):
                self._state["plan"]["pictures"].append(
                    {"name": drawn.name, "width": drawn.width, "height": drawn.height, "prompt": drawn.prompt, "ready": True,
                     "seconds": drawn.seconds, "again": name}
                )
        return Revised(
            source=picture_kit._reference(name).sub(lambda _match: new_name, source), number=part, comment=comment.strip(),
            before=name, after=new_name, plan=f"The picture drawn again, from: {drawn.prompt}",
            seconds=round(time.perf_counter() - started, 2),
        )

    def change(self, *, version: int, changes: list[dict]) -> list[dict]:
        """Blocking. Writes again the parts of the built page that were
        pointed at, each as its comment asks, on the chip that wrote the
        page -- and draws again the pictures a comment is about, on the
        chip that drew them. Returns what became of each change."""
        built = self._built
        if built is None or not self.revisions.available:
            raise Conflict("There is no page to change yet: build one first.")
        self.revisions.check(version, changes)  # refused before the model is waited for
        if not self._lock.acquire(blocking=False):
            raise Conflict("A page is being built -- wait for it, or stop it first.")
        device = built.assignment.page
        try:
            if self._session is None:  # unloaded from the hardware panel since the build
                self._session = PageAgent(Engine.OPENVINO)
                self._engine = _ENGINE
            self._device = device
            session = self._session
            live = generation.get(_DEMO_ID, PAGE)
            activity.set_active(_DEMO_ID, engine=_ENGINE, device=device, stage=PAGE, stage_label=_LABELS[PAGE].lower())
            stop_asked = [False]
            where = ["Changing the part..."]

            def on_start(index: int, total: int, change: dict) -> None:
                where[0] = f"Changing part {index} of {total} on {device}..." if total > 1 else f"Changing the part on {device}..."
                events.set_phase(_DEMO_ID, "loading", f"Loading the coding model on {device}...")

            def write(source, change, pictures):
                if change.get("picture"):
                    return self._draw_again(session, built, source, change, pictures)
                control = live.begin()
                try:
                    return session.revise(
                        source, int(change["part"]), change["comment"], device=device, seen=change.get("seen"),
                        pictures=pictures or None,
                        on_ready=lambda: events.set_phase(_DEMO_ID, "running", where[0]),
                        on_downloading=lambda: events.set_phase(_DEMO_ID, "loading", "Downloading a model (first run only)..."),
                        control=control,
                    )
                finally:
                    stop_asked[0] = stop_asked[0] or bool(control.should_stop and control.should_stop())
                    live.end()

            try:
                results = self.revisions.change(version, changes, write, on_start, stopped=lambda: stop_asked[0])
            except Exception as exc:
                events.set_phase(_DEMO_ID, "error", str(exc))
                raise
            rates = [result["tokens_per_second"] for result in results if result.get("tokens_per_second")]
            if rates:
                metrics.report(_DEMO_ID, rates[-1], "tok/s", stage=PAGE, sticky=True)
            events.clear_phase(_DEMO_ID)
            return results
        finally:
            for stage in (PLAN, IMAGES, PAGE):
                activity.clear_active(_DEMO_ID, stage=stage)
            self._lock.release()
