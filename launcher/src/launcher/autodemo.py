"""The Auto Demo's director: the app running itself on a stand, scene after
scene, with nobody at the keyboard (docs/AUTO_DEMO.md has the design and
what was measured for it).

The launcher directs and the page follows. A thread here owns the playlist,
the clock and the failure rules, and says where it is at `GET
/api/autodemo`; the page has a stage of its own for it -- a caption, the
demo's outputs, the chips at work -- and draws what the director says. A
reload of the page picks up mid-scene, and a stalled browser does not stop
the show.

The director acts the way a person does: through the launcher's own routes,
on this machine. A scene is therefore data -- which route to call with what,
what to wait for, what to say -- and whatever a route checks for a person
(the device exists, the folder is there, a build is already running) it
checks for the director too.

What a scene says is the point of it, and it says it as a story: a sentence
or two at a time, each one shown when the thing it talks about is really
happening (the planner is at work, the pictures are being drawn, the result
is in) rather than at a time on a clock -- a model that takes forty seconds
to load one day takes fifteen the next.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from . import keep_awake as awake
from .errors import Conflict

# ------------------------------------------------------------------ what a scene is


@dataclass(frozen=True)
class Beat:
    """One moment of the story: a sentence or two, and when they are due.
    All that is set must hold: `after` seconds into the scene, `stage` -- a
    stage of the scene's demo, as the hardware panel names it -- seen at
    work, `result` once the scene's work is done. With `figure`, the stage
    must have a figure to show as well: a stage is "at work" from the moment
    its model starts loading, and a sentence about what it puts out is early
    until it puts something out."""

    text: str
    after: float = 0.0
    stage: str = ""
    result: bool = False
    figure: bool = False


@dataclass(frozen=True)
class Chip:
    """What a chip is doing in the scene, for the line under its gauge."""

    chip: str  # "NPU", "Integrated GPU", "Arc Pro B60", "CPU"
    label: str  # a few words: "Planner · Qwen3-8B"
    # Whose live figure belongs on that line, as the hardware panel knows it:
    # a demo and the stages of it this chip works on -- several when one
    # chip does two jobs in turn.
    demo: str = ""
    stages: tuple[str, ...] = ("default",)


@dataclass(frozen=True)
class Start:
    """Start a demo that runs until it is stopped, or ends by itself."""

    path: str
    body: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Ask:
    """Ask a demo that answers one request and wait for the answer, which is
    kept for the page to draw. `cancel` is the route that stops it part-way,
    for when the scene is interrupted."""

    path: str
    body: dict = field(default_factory=dict)
    cancel: str = ""
    timeout: float = 900.0


@dataclass(frozen=True)
class Wait:
    """Leave the scene running for a while: a stream to watch."""

    seconds: float


@dataclass(frozen=True)
class Until:
    """Wait until a route answers with `key` equal to `equals` -- a demo
    saying it has finished -- for `timeout` seconds at most."""

    path: str
    key: str
    equals: Any
    timeout: float = 240.0


Step = Start | Ask | Wait | Until


@dataclass(frozen=True)
class Scene:
    id: str
    title: str
    demo: str  # the brick it shows; its stages cue the story
    view: str  # which of the stage's output views draws it: "page", "receipts", "cameras"
    beats: tuple[Beat, ...]  # the story, in order
    chips: tuple[Chip, ...]
    steps: tuple[Step, ...]
    # What the output view needs that only the scene knows (the receipts to
    # show before they are read, say). Plain data: it goes to the page as is.
    props: dict = field(default_factory=dict)
    # Routes called (POST) when the scene ends, however it ends: nothing a
    # scene started is left running.
    stop: tuple[str, ...] = ()
    hold: float = 20.0  # seconds the result stays on screen once the work is done
    at_most: float = 420.0  # the scene is ended after this long, whatever it is doing


@dataclass(frozen=True)
class Skip:
    """A scene that cannot be played on this stand, and why."""

    title: str
    reason: str


LANGUAGES = ("en", "fr")


@dataclass
class Stand:
    """What the stand has, looked at once when the loop starts, and how the
    person starting it wants it played."""

    npu: bool = False
    igpu: str | None = None  # the integrated GPU's device name
    dgpu: str | None = None  # the discrete GPU's, if it is plugged in and wanted
    cameras: list = field(default_factory=list)
    internet: bool = False
    big_screen: bool = False
    lang: str = "en"  # the language the story is told in
    # A demo's bundled samples, by demo id, fetched when a scene asks.
    samples: Callable[[str], list[dict]] = lambda demo: []


# A scene is built for the stand it will play on -- its steps and what it
# says both depend on the chips there -- and for the loop it is in, so that a
# scene with several samples shows another one each time round.
Builder = Callable[[Stand, int], Scene | Skip]
# call(method, path, body, timeout) -> the route's JSON answer; raises on an error answer.
Call = Callable[[str, str, dict | None, float], dict]


class SceneFailed(RuntimeError):
    pass


class _Interrupted(Exception):
    """The scene was ended from outside: stopped or skipped."""


IDLE, CHECKING, PLAYING, PAUSED, STOPPED = "idle", "checking", "playing", "paused", "stopped"
# Told to stop, and still letting go of the demo it was in. A page being
# planned cannot be cut short: it ends when its model has answered, which
# can be half a minute after the stop was asked for.
STOPPING = "stopping"


def _never(host: str = "", port: int = 0) -> bool:
    return False


class Director:
    """Plays a playlist in a loop until told to stop.

    - A scene that fails is noted and skipped; `max_failures` in a row and
      the loop stops with a plain notice rather than go round on an error.
    - `pause()` holds the loop where it is: the scene in hand runs to its
      end and its result stays on screen, and the next one does not start
      until `resume()` -- or until nobody has asked for anything for
      `idle_resume` seconds, because a stand that waits for an answer from
      somebody who has walked away has stopped for the day.
    - Whatever a scene started is stopped when it ends, however it ends.
    """

    def __init__(
        self,
        call: Call,
        playlist: list[Builder],
        *,
        online: Callable[[], bool] = _never,
        idle_resume: float = 300.0,
        max_failures: int = 3,
        poll: float = 0.5,
        keep_awake: tuple[Callable[[], bool], Callable[[], None]] = (awake.hold, awake.release),
    ):
        self._call, self._playlist, self._online = call, list(playlist), online
        self._idle_resume, self._max_failures, self._poll = idle_resume, max_failures, poll
        self._hold_awake, self._release_awake = keep_awake
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()  # the loop is to end
        self._interrupt = threading.Event()  # the scene in hand is to end
        self._paused = False
        self._resumes_at = 0.0
        self._result: dict | None = None
        self._state: dict = self._fresh()

    # ------------------------------------------------------------------ from outside

    @staticmethod
    def _fresh() -> dict:
        return {
            "state": IDLE, "loop": 0, "playlist": [], "scene": None, "stand": None, "paused": None,
            "failures": [], "notice": "", "awake": False, "started_at": None,
        }

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            state = dict(self._state)
            state["scene"] = dict(state["scene"]) if state["scene"] else None
            state["now"] = time.time()
            return state

    def result(self) -> dict | None:
        """What the scene in hand was answered, for the page to draw:
        `{"scene": id, "data": the route's answer}`."""
        with self._lock:
            return self._result

    def check(self, *, dgpu: str = "auto", big_screen: bool = False, lang: str = "en") -> dict:
        """What the loop would do if it were started now: what the stand
        has, and each scene with whether it can play and, if not, why.
        Nothing is started."""
        stand = self._look_at_the_stand(dgpu, big_screen, lang)
        return {"stand": self._describe(stand), "playlist": self._listing(stand, 1)[1]}

    def start(self, *, dgpu: str = "auto", big_screen: bool = False, lang: str = "en") -> dict:
        """`dgpu`: "auto" uses the discrete GPU if it is plugged in, "off"
        plays as if it were not there. `lang`: the story's language."""
        if self.running:
            raise Conflict(
                "The Auto Demo is still stopping: the demo it was showing is finishing. Try again in a moment."
                if self._stop.is_set() else "The Auto Demo is already running."
            )
        if dgpu not in ("auto", "off"):
            raise ValueError("dgpu is 'auto' or 'off'.")
        if lang not in LANGUAGES:
            raise ValueError(f"lang is one of {', '.join(LANGUAGES)}.")
        self._stop.clear()
        self._interrupt.clear()
        self._paused = False
        with self._lock:
            self._state = {**self._fresh(), "state": CHECKING, "started_at": time.time()}
            self._result = None
        self._thread = threading.Thread(
            target=self._run, kwargs={"dgpu": dgpu, "big_screen": big_screen, "lang": lang}, daemon=True, name="autodemo"
        )
        self._thread.start()
        return self.snapshot()

    def stop(self, wait: float = 30.0) -> dict:
        """Ends the loop and waits up to `wait` seconds for it to have let
        go of everything. If it has not by then, the answer says "stopping"
        rather than pretend: the loop is idle when the demo it was in has
        finished or been cancelled."""
        self._stop.set()
        self._interrupt.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            self._set(state=STOPPING, paused=None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(wait)
        return self.snapshot()

    def pause(self) -> dict:
        """Hold the loop where it is. Nothing is interrupted: the scene in
        hand finishes and stays on screen. Asked again while paused, it
        puts the resumption off again."""
        if not self.running:
            return self.snapshot()
        with self._lock:
            since = (self._state["paused"] or {}).get("since") or time.time()
            self._paused = True
            self._resumes_at = time.time() + self._idle_resume
            self._state["paused"] = {"since": since, "resumes_at": self._resumes_at}
        return self.snapshot()

    def resume(self) -> dict:
        with self._lock:
            self._paused = False
            self._state["paused"] = None
        return self.snapshot()

    def skip(self) -> dict:
        """On to the next scene now."""
        if self.running:
            self._interrupt.set()
        return self.snapshot()

    # ---------------------------------------------------------------------- the loop

    def _set(self, **changes) -> None:
        with self._lock:
            self._state.update(changes)

    def _look_at_the_stand(self, dgpu: str, big_screen: bool, lang: str = "en") -> Stand:
        gpus = self._call("GET", "/api/system/gpu-devices", None, 15)
        cameras, devices = [], []
        try:
            seen = self._call("GET", "/api/object-detection/devices", None, 30)
            cameras, devices = seen.get("cameras") or [], seen.get("openvino_devices") or []
        except Exception:
            pass  # a stand without the vision demo still has the others
        discrete = [gpu["id"] for gpu in gpus if "dGPU" in gpu.get("full_name", "")]
        integrated = [gpu["id"] for gpu in gpus if "dGPU" not in gpu.get("full_name", "")]
        cache: dict[str, list[dict]] = {}

        def samples(demo: str) -> list[dict]:
            if demo not in cache:
                cache[demo] = self._call("GET", f"/api/{demo}/devices", None, 30).get("samples") or []
            return cache[demo]

        return Stand(
            npu=any(str(device).upper().startswith("NPU") for device in devices),
            igpu=integrated[0] if integrated else None,
            dgpu=discrete[-1] if discrete and dgpu == "auto" else None,
            cameras=list(cameras),
            internet=bool(self._online()),
            big_screen=big_screen,
            lang=lang if lang in LANGUAGES else "en",
            samples=samples,
        )

    @staticmethod
    def _describe(stand: Stand) -> dict:
        return {
            "npu": stand.npu, "igpu": stand.igpu, "dgpu": stand.dgpu, "cameras": len(stand.cameras),
            "internet": stand.internet, "big_screen": stand.big_screen, "lang": stand.lang,
        }

    def _listing(self, stand: Stand, loop: int) -> tuple[list[Scene | Skip], list[dict]]:
        """This turn of the loop: its scenes as built for the stand, and the
        same as a list to show -- what will play, what will not, and why."""
        built, listing = [], []
        for build in self._playlist:
            try:
                scene = build(stand, loop)
            except Exception as exc:  # a scene that cannot even be put together is one that cannot play
                scene = Skip(getattr(build, "__name__", "scene").replace("_", " "), f"could not be prepared: {exc}")
            playable = isinstance(scene, Scene)
            built.append(scene)
            listing.append({
                "id": scene.id if playable else "", "title": scene.title, "playable": playable,
                "demo": scene.demo if playable else "", "reason": "" if playable else scene.reason,
            })
        return built, listing

    def _run(self, dgpu: str, big_screen: bool, lang: str) -> None:
        self._set(awake=bool(self._hold_awake()))
        notice = ""
        try:
            stand = self._look_at_the_stand(dgpu, big_screen, lang)
            self._set(stand=self._describe(stand))
            failures_in_a_row, loop = 0, 0
            while not self._stop.is_set():
                loop += 1
                played = 0
                # What this turn of the loop will play, said before it starts.
                built, listing = self._listing(stand, loop)
                self._set(loop=loop, playlist=listing)
                for scene in built:
                    if self._stop.is_set():
                        break
                    if not isinstance(scene, Scene):
                        continue
                    self._wait_while_paused(between_scenes=True)
                    if self._stop.is_set():
                        break
                    played += 1
                    failures_in_a_row = 0 if self._play(scene) else failures_in_a_row + 1
                    if failures_in_a_row >= self._max_failures:
                        notice = (
                            f"The Auto Demo stopped itself: {failures_in_a_row} scenes failed one after the other. "
                            "The last reasons are listed below."
                        )
                        return
                if not played and not self._stop.is_set():
                    notice = "No scene of the playlist can be played on this stand: " + "; ".join(
                        f"{entry['title']} ({entry['reason']})" for entry in listing
                    )
                    return
        except Exception as exc:  # the stand could not even be looked at
            notice = f"The Auto Demo could not start: {exc}"
        finally:
            self._release_awake()
            self._set(state=STOPPED if notice else IDLE, scene=None, paused=None, notice=notice, awake=False)

    def _wait_while_paused(self, between_scenes: bool = False) -> None:
        """Hold here for as long as the loop is paused: until it is resumed,
        stopped, skipped, or left alone for `idle_resume` seconds."""
        if not self._paused:
            return
        if between_scenes:
            self._set(state=PAUSED, scene=None)
        while self._paused and not self._stop.is_set() and not self._interrupt.is_set():
            if time.time() >= self._resumes_at:
                break
            time.sleep(min(self._poll, 0.2))
        with self._lock:
            self._paused = False
            self._state["paused"] = None

    # --------------------------------------------------------------------- one scene

    def _play(self, scene: Scene) -> bool:
        """True unless the scene failed. Being interrupted is not failing."""
        if self._stop.is_set():  # stopped between the look at the playlist and here: nothing more is started
            return True
        self._interrupt.clear()
        now = time.time()
        with self._lock:
            self._result = None
            self._state.update(
                state=PLAYING,
                scene={
                    "id": scene.id, "title": scene.title, "demo": scene.demo, "view": scene.view,
                    "beats": [asdict(beat) for beat in scene.beats], "chips": [asdict(chip) for chip in scene.chips],
                    "props": scene.props, "phase": "working", "started_at": now, "ends_by": now + scene.at_most,
                    "result_ready": False,
                },
            )
        deadline = time.monotonic() + scene.at_most
        try:
            for step in scene.steps:
                self._do(step, scene, deadline)
            self._phase("showing", ends_by=time.time() + scene.hold)
            self._pause_for(scene.hold)
            self._wait_while_paused()  # held on its result for as long as the loop is
            return True
        except _Interrupted:
            return True
        except Exception as exc:
            with self._lock:
                self._state["failures"] = [
                    *self._state["failures"][-9:], {"scene": scene.id, "title": scene.title, "error": str(exc), "at": time.time()},
                ]
            return False
        finally:
            for path in scene.stop:
                try:
                    self._call("POST", path, {}, 60)
                except Exception:
                    pass  # stopping what is not running is not an error worth a notice

    def _phase(self, phase: str, **more) -> None:
        with self._lock:
            if self._state["scene"]:
                self._state["scene"] = {**self._state["scene"], "phase": phase, **more}

    def _pause_for(self, seconds: float) -> None:
        if self._interrupt.wait(seconds):
            raise _Interrupted()

    def _do(self, step: Step, scene: Scene, deadline: float) -> None:
        if self._interrupt.is_set():
            raise _Interrupted()
        if isinstance(step, Wait):
            self._pause_for(max(0.0, min(step.seconds, deadline - time.monotonic())))
        elif isinstance(step, Start):
            self._call("POST", step.path, step.body, 120)
        elif isinstance(step, Until):
            end = min(deadline, time.monotonic() + step.timeout)
            while self._call("GET", step.path, None, 30).get(step.key) != step.equals:
                if time.monotonic() >= end:
                    raise SceneFailed(f"{scene.title}: still not finished after {step.timeout:.0f} s")
                self._pause_for(self._poll)
        elif isinstance(step, Ask):
            self._ask(step, scene, deadline)

    def _ask(self, step: Ask, scene: Scene, deadline: float) -> None:
        """The request runs on a thread of its own, so that the director can
        still be interrupted while a page takes its minute and a half."""
        box: dict = {}

        def work() -> None:
            try:
                box["answer"] = self._call("POST", step.path, step.body, step.timeout)
            except Exception as exc:
                box["error"] = exc

        worker = threading.Thread(target=work, daemon=True, name="autodemo-ask")
        worker.start()
        while worker.is_alive():
            interrupted = self._interrupt.wait(0.1)
            late = time.monotonic() >= deadline
            if interrupted or late:
                if step.cancel:
                    try:
                        self._call("POST", step.cancel, {}, 30)
                    except Exception:
                        pass
                worker.join(60)
                if interrupted:
                    raise _Interrupted()
                raise SceneFailed(f"{scene.title}: no answer after {scene.at_most:.0f} s")
        if "error" in box:
            raise SceneFailed(f"{scene.title}: {box['error']}")
        with self._lock:
            self._result = {"scene": scene.id, "data": box.get("answer")}
        self._phase("answered", result_ready=True)
