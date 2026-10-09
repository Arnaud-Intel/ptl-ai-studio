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
    for when the scene is interrupted. `keep=False` for a request on the way
    to the one that matters -- a folder indexed before the question is asked
    -- whose answer is not the scene's result."""

    path: str
    body: dict = field(default_factory=dict)
    cancel: str = ""
    timeout: float = 900.0
    keep: bool = True


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
    cameras: list = field(default_factory=list)  # those the scenes may use
    cameras_found: int = 0  # those the machine has, used or not
    internet: bool = False
    big_screen: bool = False
    lang: str = "en"  # the language the story is told in
    # A demo's bundled samples, by demo id, fetched when a scene asks.
    samples: Callable[[str], list[dict]] = lambda demo: []


# A scene is built for the stand it will play on -- its steps and what it
# says both depend on the chips there -- and for the loop it is in, so that a
# scene with several samples shows another one each time round.
Builder = Callable[[Stand, int], Scene | Skip]


@dataclass(frozen=True)
class Entry:
    """A scene as the start screen lists it: the name it is ticked under,
    which stays the same whether it can play on this stand or not, and what
    builds it."""

    key: str
    build: Builder


# One place in the loop. Several entries in it take turns there: the first on
# the first turn of the loop, the second on the second, and round again. Two
# scenes of the same demo back to back read as one long one.
Slot = Builder | Entry | tuple[Entry, ...]
# call(method, path, body, timeout) -> the route's JSON answer; raises on an error answer.
Call = Callable[[str, str, dict | None, float], dict]


class SceneFailed(RuntimeError):
    pass


class Busy(RuntimeError):
    """A route answered that its demo is still at something (HTTP 409): a
    page still being planned after its scene was skipped, a batch of receipts
    still stopping. Not a failure yet: it is letting go, and is asked again."""


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
        playlist: list[Slot],
        *,
        online: Callable[[], bool] = _never,
        idle_resume: float = 300.0,
        max_failures: int = 3,
        poll: float = 0.5,
        busy_wait: float = 75.0,
        keep_awake: tuple[Callable[[], bool], Callable[[], None]] = (awake.hold, awake.release),
    ):
        self._call, self._online = call, online
        self._playlist: list[tuple[Entry, ...]] = [self._slot(slot, index) for index, slot in enumerate(playlist)]
        self._idle_resume, self._max_failures, self._poll = idle_resume, max_failures, poll
        # How long a demo that says it is busy is waited for. A page being
        # planned when its scene was skipped takes up to a minute to end, and
        # somebody skipping through the loop is back at it sooner than that:
        # three scenes refused for that would stop the loop on no fault at all.
        self._busy_wait = busy_wait
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
        # "Stopping" is only ever true of a loop that is still there. One
        # that ended in the instant it was told to stop wrote "idle" first
        # and had "stopping" written over it (seen once, on the test
        # machine): what is said is what is so.
        if state["state"] == STOPPING and not self.running:
            state["state"] = STOPPED if state["notice"] else IDLE
            state["scene"] = None
        return state

    def result(self) -> dict | None:
        """What the scene in hand was answered, for the page to draw:
        `{"scene": id, "data": the route's answer}`."""
        with self._lock:
            return self._result

    @staticmethod
    def _slot(slot: Slot, index: int) -> tuple[Entry, ...]:
        if isinstance(slot, tuple):
            return slot
        if isinstance(slot, Entry):
            return (slot,)
        name = getattr(slot, "__name__", "")
        return (Entry(name.replace("_", "-") if name.isidentifier() else f"scene-{index + 1}", slot),)

    def keys(self) -> list[str]:
        """Every scene of the playlist, by the name it is chosen under."""
        return [entry.key for slot in self._playlist for entry in slot]

    def _chosen(self, scenes: list[str] | None) -> set[str] | None:
        """The scenes asked for, checked. None is all of them."""
        if scenes is None:
            return None
        unknown = sorted(set(scenes) - set(self.keys()))
        if unknown:
            raise ValueError(f"No such scene: {', '.join(unknown)}. The playlist has: {', '.join(self.keys())}.")
        if not scenes:
            raise ValueError("Choose at least one demo to play.")
        return set(scenes)

    def check(
        self, *, dgpu: str = "auto", big_screen: bool = False, lang: str = "en", camera: str = "auto",
        scenes: list[str] | None = None,
    ) -> dict:
        """What the loop would do if it were started now: what the stand
        has, and each scene with whether it can play and, if not, why.
        Nothing is started."""
        chosen = self._chosen(scenes)
        stand = self._look_at_the_stand(dgpu, big_screen, lang, camera)
        return {"stand": self._describe(stand), "playlist": self._listing(stand, 1, chosen)[1]}

    def start(
        self, *, dgpu: str = "auto", big_screen: bool = False, lang: str = "en", camera: str = "auto",
        scenes: list[str] | None = None,
    ) -> dict:
        """`dgpu`: "auto" uses the discrete GPU if it is plugged in, "off"
        plays as if it were not there. `camera`: the same for the camera --
        "off" and no scene switches it on. `lang`: the story's language.
        `scenes`: the ones to play, by key; all of them if not said."""
        if self.running:
            raise Conflict(
                "The Auto Demo is still stopping: the demo it was showing is finishing. Try again in a moment."
                if self._stop.is_set() else "The Auto Demo is already running."
            )
        if dgpu not in ("auto", "off"):
            raise ValueError("dgpu is 'auto' or 'off'.")
        if camera not in ("auto", "off"):
            raise ValueError("camera is 'auto' or 'off'.")
        if lang not in LANGUAGES:
            raise ValueError(f"lang is one of {', '.join(LANGUAGES)}.")
        chosen = self._chosen(scenes)
        self._stop.clear()
        self._interrupt.clear()
        self._paused = False
        with self._lock:
            self._state = {**self._fresh(), "state": CHECKING, "started_at": time.time()}
            self._result = None
        self._thread = threading.Thread(
            target=self._run,
            kwargs={"dgpu": dgpu, "big_screen": big_screen, "lang": lang, "camera": camera, "chosen": chosen},
            daemon=True, name="autodemo",
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

    def _look_at_the_stand(self, dgpu: str, big_screen: bool, lang: str = "en", camera: str = "auto") -> Stand:
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
            # A camera shows whoever stands in front of it. One that was
            # left out is, for every scene, a camera that is not there.
            cameras=list(cameras) if camera == "auto" else [],
            cameras_found=len(cameras),
            internet=bool(self._online()),
            big_screen=big_screen,
            lang=lang if lang in LANGUAGES else "en",
            samples=samples,
        )

    @staticmethod
    def _describe(stand: Stand) -> dict:
        return {
            "npu": stand.npu, "igpu": stand.igpu, "dgpu": stand.dgpu, "cameras": len(stand.cameras),
            "cameras_found": max(stand.cameras_found, len(stand.cameras)),
            "internet": stand.internet, "big_screen": stand.big_screen, "lang": stand.lang,
        }

    @staticmethod
    def _build(entry: Entry, stand: Stand, turn: int) -> Scene | Skip:
        try:
            return entry.build(stand, turn)
        except Exception as exc:  # a scene that cannot even be put together is one that cannot play
            return Skip(entry.key.replace("-", " "), f"could not be prepared: {exc}")

    def _listing(self, stand: Stand, loop: int, chosen: set[str] | None = None) -> tuple[list[Scene], list[dict]]:
        """This turn of the loop: the scenes it plays, built for the stand,
        and every scene of the playlist as a list to show -- whether it was
        chosen, whether it can play here and if not why, and whether this
        is its turn.

        Where several scenes share a place in the loop, those that were
        chosen and can play take turns in it: if one of them cannot play
        here, the others have the place every turn rather than the loop
        going a scene short."""
        plays, listing = [], []
        for slot in self._playlist:
            looked = {entry.key: self._build(entry, stand, loop) for entry in slot}
            able = [entry for entry in slot
                    if (chosen is None or entry.key in chosen) and isinstance(looked[entry.key], Scene)]
            mine = able[(loop - 1) % len(able)] if able else None
            for entry in slot:
                scene = looked[entry.key]
                if entry is mine:
                    # Told how many times it has played itself, not which
                    # turn of the loop this is: a scene that shows another
                    # sample each time must not skip every other one.
                    again = self._build(entry, stand, (loop - 1) // len(able) + 1)
                    scene = again if isinstance(again, Scene) else scene
                    plays.append(scene)
                playable = isinstance(scene, Scene)
                listing.append({
                    "key": entry.key, "id": scene.id if playable else "", "title": scene.title, "playable": playable,
                    "demo": scene.demo if playable else "", "reason": "" if playable else scene.reason,
                    "chosen": chosen is None or entry.key in chosen, "plays": entry is mine,
                    "turns_with": [other.key for other in slot if other is not entry],
                })
        return plays, listing

    def _run(self, dgpu: str, big_screen: bool, lang: str, camera: str = "auto", chosen: set[str] | None = None) -> None:
        self._set(awake=bool(self._hold_awake()))
        notice = ""
        try:
            stand = self._look_at_the_stand(dgpu, big_screen, lang, camera)
            self._set(stand=self._describe(stand))
            failures_in_a_row, loop = 0, 0
            while not self._stop.is_set():
                loop += 1
                played = 0
                # What this turn of the loop will play, said before it starts.
                built, listing = self._listing(stand, loop, chosen)
                self._set(loop=loop, playlist=listing)
                for scene in built:
                    if self._stop.is_set():
                        break
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
                        f"{entry['title']} ({entry['reason']})" for entry in listing if entry["chosen"]
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
            self._post(step.path, step.body, 120, deadline)
        elif isinstance(step, Until):
            end = min(deadline, time.monotonic() + step.timeout)
            while self._call("GET", step.path, None, 30).get(step.key) != step.equals:
                if time.monotonic() >= end:
                    raise SceneFailed(f"{scene.title}: still not finished after {step.timeout:.0f} s")
                self._pause_for(self._poll)
        elif isinstance(step, Ask):
            self._ask(step, scene, deadline)

    def _post(self, path: str, body: dict, timeout: float, deadline: float) -> dict:
        """Call a route, asking again for a while if its demo says it is
        still busy with what it was doing before."""
        end = min(deadline, time.monotonic() + self._busy_wait)
        while True:
            try:
                return self._call("POST", path, body, timeout)
            except Busy:
                if time.monotonic() >= end:
                    raise
                if self._interrupt.wait(min(2.0, self._poll * 4)):
                    raise _Interrupted() from None

    def _ask(self, step: Ask, scene: Scene, deadline: float) -> None:
        """The request runs on a thread of its own, so that the director can
        still be interrupted while a page takes its minute and a half."""
        box: dict = {}

        def work() -> None:
            try:
                box["answer"] = self._post(step.path, step.body, step.timeout, deadline)
            except _Interrupted:
                pass  # the loop below has seen the same interruption
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
        if not step.keep:
            return
        with self._lock:
            self._result = {"scene": scene.id, "data": box.get("answer")}
        self._phase("answered", result_ready=True)
