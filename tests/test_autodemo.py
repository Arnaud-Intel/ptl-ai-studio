"""The Auto Demo: the app running itself on a stand (docs/AUTO_DEMO.md).

The director against a stand-in for the launcher's routes, so nothing here
loads a model or needs a browser: scenes play in order and go round, a scene
that fails is skipped and noted, three in a row stop the loop, a pause holds
the loop on what it is showing without interrupting it, and whatever a scene
started is stopped when it ends -- however it ends. Then the playlist: every
scene tells its story in both languages, for the stand it finds."""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from launcher import app as launcher_app
from launcher import autodemo, autodemo_scenes
from launcher.autodemo import Ask, Beat, Chip, Director, Entry, Scene, Skip, Stand, Start, Until, Wait
from launcher.errors import Conflict

GPUS = [
    {"id": "GPU.0", "full_name": "Intel(R) Arc(TM) B390 GPU (iGPU)"},
    {"id": "GPU.1", "full_name": "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)"},
]
SAMPLES = {
    "page-agent": [{"name": "Neighbourhood bakery", "prompt": "A page for a bakery."},
                   {"name": "Mountain bike rental", "prompt": "A page for a bike shop."}],
    "expense-extract": [
        {"name": "Start here · 3 everyday expenses", "folder": "C:/receipts/quick-start"},
        {"name": "Scanned & photographed · 5 worn receipts", "folder": "C:/receipts/scanned", "assets": [
            {"name": "01-cafe.png", "url": "/demo-assets/expenses/scanned/01-cafe.png", "image": True},
            {"name": "answers.json", "url": "/demo-assets/expenses/scanned/answers.json", "image": False},
            {"name": "02-taxi.png", "url": "/demo-assets/expenses/scanned/02-taxi.png", "image": True},
        ]},
    ],
    "smart-city-monitor": [
        {"name": "Shibuya", "group": "YouTube", "feeds": "https://video.example/live"},
        {"name": "Tower Bridge", "group": "Other", "feeds": "https://clips.example/1.mp4"},
        {"name": "Westminster Bridge", "group": "Other", "feeds": "https://clips.example/2.mp4"},
        {"name": "London, two chips", "group": "Other", "feeds": "https://clips.example/1.mp4|NPU\nhttps://clips.example/2.mp4"},
        {"name": "Cattle on the road", "group": "On this machine", "kind": "file", "ready": True, "counting": ["herd"], "feeds": "C:/v/cattle.webm"},
        {"name": "Bottle capping line", "group": "On this machine", "kind": "file", "ready": True, "counting": ["line"], "feeds": "C:/v/bottles.webm"},
    ],
    "doc-qa": [{"name": "Can the pilot launch?", "folder": "C:/samples/meridian-rollout-2026", "question": "Is it approved?", "assets": [
        {"name": "00-company-brief.md", "url": "/demo-assets/meridian-rollout-2026/00-company-brief.md", "image": False},
        {"name": "03-release-decision-log.md", "url": "/demo-assets/meridian-rollout-2026/03-release-decision-log.md", "image": False},
    ]}],
    "video-commentary": [
        {"name": "Cattle on the road", "path": "C:/v/cattle.webm", "ready": True},
        {"name": "Bottle capping line", "path": "C:/v/bottles.webm", "ready": True},
        {"name": "Tokyo", "path": "C:/v/tokyo.webm", "ready": False},
    ],
    "object-detection": [
        {"name": "Tyumen", "kind": "file", "path": "C:/v/tyumen.webm", "ready": False},
        {"name": "Toronto", "kind": "file", "path": "C:/v/toronto.webm", "ready": True},
    ],
}


class Stage:
    """Stands in for the launcher's routes: remembers every call, answers
    the ones a director makes when it looks at the stand, and can be told to
    fail or to take its time on a path."""

    def __init__(self, gpus=GPUS, cameras=(0,), devices=("CPU", "GPU.0", "GPU.1", "NPU")):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.gpus, self.cameras, self.devices = list(gpus), list(cameras), list(devices)
        self.fail: dict[str, str] = {}
        self.slow: dict[str, threading.Event] = {}
        self.cancels: dict[str, str] = {}  # a route that, called, lets a slow one go: what cancelling does
        self.busy: dict[str, int] = {}  # how many more times a route answers that its demo is still busy
        self.answers: dict[str, dict] = {}

    def __call__(self, method, path, body, timeout):
        self.calls.append((method, path, body))
        if self.busy.get(path, 0) > 0:
            self.busy[path] -= 1
            raise autodemo.Busy("A page is being built already -- wait for it, or stop it first.")
        if path in self.fail:
            raise RuntimeError(self.fail[path])
        if path in self.cancels:
            self.slow[self.cancels[path]].set()
        if path in self.slow:
            self.slow[path].wait(5)
        if path == "/api/system/gpu-devices":
            return self.gpus
        if path == "/api/object-detection/devices":
            return {"cameras": self.cameras, "openvino_devices": self.devices, "samples": SAMPLES["object-detection"]}
        if path.endswith("/devices"):
            return {"samples": SAMPLES.get(path.split("/")[2], [])}
        return self.answers.get(path, {"status": "ok"})

    def posted(self) -> list[str]:
        return [path for method, path, _body in self.calls if method == "POST"]


def _scene(name: str, *steps, stop=(), hold=0.0, at_most=5.0) -> Scene:
    return Scene(
        id=name, title=name.title(), demo=name, view="page",
        beats=(Beat(f"{name} begins."), Beat("The small model is at work.", stage="plan"), Beat("Done.", result=True)),
        chips=(Chip("NPU", "Plans · a small model", name, ("plan",)),), props={"request": name},
        steps=tuple(steps), stop=tuple(stop), hold=hold, at_most=at_most,
    )


def _director(stage: Stage, playlist, **options) -> Director:
    awake: list[str] = []
    options.setdefault("poll", 0.01)
    director = Director(stage, playlist, keep_awake=(lambda: awake.append("held") or True, lambda: awake.append("released")), **options)
    director.awake = awake
    return director


def _until(condition, seconds=3.0) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.005)
    return False


# --- the loop ---------------------------------------------------------------------------


def test_scenes_play_in_order_go_round_and_stop_what_they_started():
    stage = Stage()
    first = lambda stand, loop: _scene("first", Start("/a/start", {"n": loop}), Wait(0.01), stop=["/a/stop"])  # noqa: E731
    second = lambda stand, loop: _scene("second", Ask("/b/ask", {"q": "?"}))  # noqa: E731
    director = _director(stage, [first, second])
    director.start()
    assert _until(lambda: director.snapshot()["loop"] >= 3)
    director.stop()
    posted = stage.posted()
    assert posted[:6] == ["/a/start", "/a/stop", "/b/ask", "/a/start", "/a/stop", "/b/ask"]
    assert [body for _m, path, body in stage.calls if path == "/a/start"][:2] == [{"n": 1}, {"n": 2}]  # built for each turn
    state = director.snapshot()
    assert state["state"] == autodemo.IDLE and state["scene"] is None and state["notice"] == ""
    assert director.running is False and director.awake == ["held", "released"]  # awake for the loop, and no longer


def test_the_scene_in_hand_comes_with_its_story_and_what_its_view_needs():
    stage = Stage()
    stage.slow["/b/ask"] = threading.Event()
    stage.answers["/b/ask"] = {"html": "<html></html>"}
    director = _director(stage, [lambda stand, loop: _scene("page", Ask("/b/ask"), hold=0.3)])
    director.start(big_screen=True, lang="fr")
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("phase") == "working")
    state = director.snapshot()
    scene = state["scene"]
    assert (state["state"], scene["id"], scene["demo"], scene["view"]) == ("playing", "page", "page", "page")
    # The story, a moment at a time, each with when it is due: at once, when a stage is seen at work, when the work is done.
    assert scene["beats"] == [
        {"text": "page begins.", "after": 0.0, "stage": "", "result": False, "figure": False, "answer": ""},
        {"text": "The small model is at work.", "after": 0.0, "stage": "plan", "result": False, "figure": False, "answer": ""},
        {"text": "Done.", "after": 0.0, "stage": "", "result": True, "figure": False, "answer": ""},
    ]
    assert scene["chips"] == [{"chip": "NPU", "label": "Plans · a small model", "demo": "page", "stages": ("plan",)}]
    assert scene["props"] == {"request": "page"} and scene["result_ready"] is False and director.result() is None
    assert state["playlist"] == [{
        "key": "scene-1", "id": "page", "title": "Page", "playable": True, "demo": "page", "reason": "",
        "chosen": True, "plays": True, "turns_with": [],
    }]
    assert state["stand"] == {
        "npu": True, "igpu": "GPU.0", "dgpu": "GPU.1", "cameras": 1, "cameras_found": 1, "internet": False,
        "big_screen": True, "lang": "fr",
    }
    assert state["awake"] is True
    # The answer is kept for the page to draw, and stays on screen for the scene's hold.
    stage.slow["/b/ask"].set()
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("phase") == "showing")
    assert director.result() == {"scene": "page", "data": {"html": "<html></html>"}, "answers": {}}
    assert director.snapshot()["scene"]["result_ready"] is True
    director.stop()


def test_a_scene_that_fails_is_noted_and_skipped_and_three_in_a_row_stop_the_loop():
    stage = Stage()
    stage.fail["/bad/start"] = "the camera is not there"
    good = lambda stand, loop: _scene("good", Start("/good/start"), stop=["/good/stop"])  # noqa: E731
    bad = lambda stand, loop: _scene("bad", Start("/bad/start"), stop=["/bad/stop"])  # noqa: E731
    director = _director(stage, [good, bad])
    director.start()
    assert _until(lambda: len(director.snapshot()["failures"]) >= 2)
    director.stop()
    state = director.snapshot()
    assert state["notice"] == "" and state["failures"][0]["error"] == "the camera is not there"
    assert state["failures"][0]["scene"] == "bad" and "/bad/stop" in stage.posted()  # stopped all the same
    # With nothing but failures the loop gives up after three, and says so instead of going round on an error.
    alone = _director(stage, [bad])
    alone.start()
    assert _until(lambda: alone.snapshot()["state"] == autodemo.STOPPED)
    state = alone.snapshot()
    assert "3 scenes failed one after the other" in state["notice"] and len(state["failures"]) == 3
    assert alone.running is False and alone.awake[-1] == "released"


def test_a_demo_still_letting_go_of_its_last_run_is_waited_for_not_counted_as_a_failure():
    stage = Stage()
    stage.busy = {"/page/build": 3, "/batch/start": 2}  # skipped a moment ago, and not done ending
    page = lambda stand, loop: _scene("page", Ask("/page/build"))  # noqa: E731
    batch = lambda stand, loop: _scene("batch", Start("/batch/start"), stop=["/batch/stop"])  # noqa: E731
    director = _director(stage, [page, batch])
    director.start()
    assert _until(lambda: director.snapshot()["loop"] >= 2)
    director.stop()
    assert director.snapshot()["failures"] == [] and stage.posted().count("/page/build") >= 4  # asked again until it took
    # A demo that stays busy longer than it is waited for is a failure after all, in the route's own words.
    stage.busy = {"/page/build": 10_000}
    stuck = _director(stage, [page], busy_wait=0.1, max_failures=1)
    stuck.start()
    assert _until(lambda: stuck.snapshot()["state"] == autodemo.STOPPED)
    assert "being built already" in stuck.snapshot()["failures"][0]["error"]
    # And the wait itself can be skipped or stopped.
    patient = _director(stage, [page], busy_wait=30)
    patient.start()
    assert _until(lambda: stage.posted().count("/page/build") > 5)
    assert patient.stop(wait=3)["state"] == autodemo.IDLE and patient.snapshot()["failures"] == []


def test_a_scene_that_never_finishes_is_ended_and_a_request_that_takes_too_long_is_cancelled():
    stage = Stage()
    stage.answers["/job/status"] = {"running": True}
    stage.slow["/page/build"] = threading.Event()
    stage.cancels["/page/cancel"] = "/page/build"
    waiting = lambda stand, loop: _scene("waiting", Start("/job/start"), Until("/job/status", "running", False, timeout=0.05), stop=["/job/stop"])  # noqa: E731
    asking = lambda stand, loop: _scene("asking", Ask("/page/build", cancel="/page/cancel"), at_most=0.05)  # noqa: E731
    director = _director(stage, [waiting, asking], max_failures=2)
    director.start()
    assert _until(lambda: director.snapshot()["state"] == autodemo.STOPPED)
    errors = [failure["error"] for failure in director.snapshot()["failures"]]
    assert "still not finished" in errors[0] and "no answer after" in errors[1]
    assert "/job/stop" in stage.posted() and "/page/cancel" in stage.posted()


def test_a_pause_holds_the_loop_on_what_it_is_showing_and_interrupts_nothing():
    stage = Stage()
    stage.slow["/page/build"] = threading.Event()
    stage.cancels["/page/cancel"] = "/page/build"
    asking = lambda stand, loop: _scene("asking", Ask("/page/build", cancel="/page/cancel"), stop=["/page/stop"])  # noqa: E731
    stream = lambda stand, loop: _scene("stream", Start("/cam/start"), Wait(5), stop=["/cam/stop"])  # noqa: E731
    director = _director(stage, [asking, stream], idle_resume=30)
    director.start()
    assert _until(lambda: "/page/build" in stage.posted())
    paused = director.pause()
    # Somebody asked for a pause while a page was being built: the page goes on being built.
    assert paused["state"] == autodemo.PLAYING and paused["scene"]["id"] == "asking" and paused["paused"]["resumes_at"] > paused["now"]
    stage.slow["/page/build"].set()
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("phase") == "showing")
    time.sleep(0.15)
    state = director.snapshot()
    assert state["scene"]["id"] == "asking" and state["scene"]["result_ready"] is True  # held on its result
    assert "/page/cancel" not in stage.posted() and "/cam/start" not in stage.posted() and not state["failures"]
    # Asked again, the resumption is put off again; the time it has been paused since is kept.
    later = director.pause()["paused"]
    assert later["resumes_at"] > paused["paused"]["resumes_at"] and later["since"] == paused["paused"]["since"]
    # Resumed, the loop moves on to the next scene, and what the first one started is stopped.
    assert director.resume()["paused"] is None
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("id") == "stream")
    assert "/page/stop" in stage.posted() and "/cam/start" in stage.posted()
    # Skipping ends a scene at once, pause or no pause.
    director.pause()
    director.skip()
    assert _until(lambda: "/cam/stop" in stage.posted())
    director.stop()
    assert director.snapshot()["state"] == autodemo.IDLE


def test_a_pause_nobody_ends_ends_by_itself():
    stage = Stage()
    quick = lambda stand, loop: _scene("quick", Start("/q/start"))  # noqa: E731
    director = _director(stage, [quick], idle_resume=0.2)
    director.start()
    assert _until(lambda: director.snapshot()["loop"] >= 1)
    director.pause()
    held_at = director.snapshot()["loop"]
    assert _until(lambda: director.snapshot()["paused"] is None, seconds=2.0)  # nobody answered: the stand carries on
    assert _until(lambda: director.snapshot()["loop"] > held_at)
    director.stop()
    assert director.pause()["state"] == autodemo.IDLE  # nothing to pause when nothing runs


def test_a_stop_that_cannot_be_immediate_says_stopping_until_the_demo_in_hand_has_let_go():
    stage = Stage()
    stage.slow["/page/build"] = threading.Event()  # a page being planned: nothing cuts it short
    director = _director(stage, [lambda stand, loop: _scene("page", Ask("/page/build"), stop=["/page/stop"])])
    director.start()
    assert _until(lambda: "/page/build" in stage.posted())
    state = director.stop(wait=0.05)
    assert state["state"] == autodemo.STOPPING and director.running is True  # not "idle": it has not let go yet

    with pytest.raises(Conflict, match="still stopping"):
        director.start()
    stage.slow["/page/build"].set()  # the model has answered
    assert _until(lambda: director.snapshot()["state"] == autodemo.IDLE and not director.running)
    assert "/page/stop" in stage.posted() and director.awake[-1] == "released"
    director.start()  # and now it can be started again
    director.stop()
    # A loop that ended in the very instant it was told to stop: "stopping" written after its own "idle".
    with director._lock:
        director._state["state"] = autodemo.STOPPING
    assert director.snapshot()["state"] == autodemo.IDLE and director.pause()["state"] == autodemo.IDLE


def test_the_loop_cannot_be_started_twice_and_a_stand_with_nothing_to_play_says_so():
    stage = Stage()
    stage.slow["/x"] = threading.Event()
    director = _director(stage, [lambda stand, loop: _scene("x", Ask("/x"))])
    director.start()
    with pytest.raises(Conflict):
        director.start()
    with pytest.raises(ValueError):
        _director(stage, []).start(dgpu="maybe")
    with pytest.raises(ValueError):
        _director(stage, []).start(lang="de")
    director.stop()
    nothing = _director(stage, [lambda stand, loop: Skip("Cameras", "needs the internet")])
    nothing.start()
    assert _until(lambda: nothing.snapshot()["state"] == autodemo.STOPPED)
    assert "Cameras (needs the internet)" in nothing.snapshot()["notice"]


# --- which scenes play ------------------------------------------------------------------


def _named(key: str, *, skip: str = "") -> Entry:
    """A scene under a key, that plays a route of its own name -- or cannot play, and says why."""
    def build(stand, turn):
        return Skip(key.title(), skip) if skip else _scene(key, Start(f"/{key}/start", {"turn": turn}))
    return Entry(key, build)


def _started(stage: Stage, count: int) -> list[tuple[str, int]]:
    return [(path, body["turn"]) for _method, path, body in stage.calls if path.endswith("/start")][:count]


def test_only_the_scenes_chosen_at_the_start_are_played_and_a_wrong_choice_is_refused():
    stage = Stage()
    director = _director(stage, [_named("page"), _named("receipts"), _named("cameras")])
    assert director.keys() == ["page", "receipts", "cameras"]
    listed = director.check(scenes=["page", "cameras"])["playlist"]
    assert [(entry["key"], entry["chosen"], entry["plays"]) for entry in listed] == [
        ("page", True, True), ("receipts", False, False), ("cameras", True, True)]
    assert listed[1]["playable"] is True  # it could play: it was left out
    director.start(scenes=["cameras", "page"])
    assert _until(lambda: director.snapshot()["loop"] >= 3)
    director.stop()
    assert stage.posted()[:4] == ["/page/start", "/cameras/start", "/page/start", "/cameras/start"]  # in the playlist's order
    assert "/receipts/start" not in stage.posted()
    assert [entry["chosen"] for entry in director.snapshot()["playlist"]] == [True, False, True]
    # Every scene, when nothing is said; and a choice that cannot be honoured is refused before anything starts.
    assert all(entry["chosen"] for entry in director.check()["playlist"])
    with pytest.raises(ValueError, match="No such scene: weather"):
        director.start(scenes=["page", "weather"])
    with pytest.raises(ValueError, match="at least one"):
        director.start(scenes=[])
    assert director.running is False
    # Chosen, and it cannot play here: the loop says so rather than go round on nothing.
    alone = _director(stage, [_named("page"), _named("cameras", skip="needs the internet")])
    alone.start(scenes=["cameras"])
    assert _until(lambda: alone.snapshot()["state"] == autodemo.STOPPED)
    assert alone.snapshot()["notice"].endswith("stand: Cameras (needs the internet)")  # only what was asked for is explained


def test_scenes_that_share_a_place_take_turns_in_it_and_each_is_told_its_own_turn():
    stage = Stage()
    director = _director(stage, [_named("page"), (_named("streets"), _named("herd"))])
    assert director.keys() == ["page", "streets", "herd"]
    first = director.check()["playlist"]
    assert [(entry["key"], entry["plays"], entry["turns_with"]) for entry in first] == [
        ("page", True, []), ("streets", True, ["herd"]), ("herd", False, ["streets"])]
    director.start()
    assert _until(lambda: director.snapshot()["loop"] >= 5)
    director.stop()
    assert _started(stage, 8) == [
        ("/page/start", 1), ("/streets/start", 1), ("/page/start", 2), ("/herd/start", 1),
        ("/page/start", 3), ("/streets/start", 2), ("/page/start", 4), ("/herd/start", 2)]  # never both in one turn
    # One of the two left out, or unable to play here: the other has the place every turn.
    for playlist, scenes in (
        ([(_named("streets"), _named("herd"))], ["herd"]),
        ([(_named("streets", skip="needs the internet"), _named("herd"))], None),
    ):
        stage = Stage()
        only = _director(stage, playlist)
        only.start(scenes=scenes)
        assert _until(lambda: only.snapshot()["loop"] >= 4)
        only.stop()
        assert _started(stage, 3) == [("/herd/start", 1), ("/herd/start", 2), ("/herd/start", 3)]


def test_an_answer_on_the_way_to_the_one_that_matters_is_not_the_scenes_result():
    stage = Stage()
    stage.slow["/docs/ask"] = threading.Event()
    stage.answers["/docs/index"] = {"chunks": 12}
    stage.answers["/docs/ask"] = {"text": "Not yet approved."}
    scene = lambda stand, loop: _scene("docs", Ask("/docs/index", keep=False), Ask("/docs/ask"), hold=0.3)  # noqa: E731
    director = _director(stage, [scene])
    director.start()
    assert _until(lambda: "/docs/ask" in stage.posted())
    assert director.result() is None and director.snapshot()["scene"]["result_ready"] is False  # indexed, not answered
    stage.slow["/docs/ask"].set()
    assert _until(lambda: director.result() is not None)
    assert director.result() == {"scene": "docs", "data": {"text": "Not yet approved."}, "answers": {}}
    director.stop()


def test_a_scene_that_asks_twice_keeps_each_answer_by_name_and_says_when_each_is_in():
    stage = Stage()
    stage.slow["/docs/index"] = threading.Event()
    stage.slow["/docs/ask"] = threading.Event()
    stage.answers["/docs/alone"] = {"text": "The team's captain."}
    stage.answers["/docs/index"] = {"chunks": 12}
    stage.answers["/docs/ask"] = {"text": "Priya Desai, on September 17."}
    scene = lambda stand, loop: _scene(  # noqa: E731
        "docs", Ask("/docs/alone", keep=False, name="alone"), Ask("/docs/index", keep=False, name="read"),
        Ask("/docs/ask", name="with"), hold=0.3)
    director = _director(stage, [scene])
    director.start()
    # Asked alone: its answer is there for the page and for the story, and it is not the scene's result.
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("answered") == ["alone"])
    now = director.snapshot()["scene"]
    assert now["result_ready"] is False and now["phase"] == "working"
    assert director.result() == {"scene": "docs", "data": None, "answers": {"alone": {"text": "The team's captain."}}}
    stage.slow["/docs/index"].set()
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("answered") == ["alone", "read"])
    assert director.snapshot()["scene"]["result_ready"] is False
    stage.slow["/docs/ask"].set()
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("result_ready") is True)
    result = director.result()
    assert result["data"] == {"text": "Priya Desai, on September 17."} and list(result["answers"]) == ["alone", "read", "with"]
    assert director.snapshot()["scene"]["answered"] == ["alone", "read", "with"]
    director.stop()
    # The next scene starts with nothing answered: one scene's answers are not the next one's.
    stage.slow["/docs/index"].clear()
    again = _director(stage, [scene])
    again.start()
    assert _until(lambda: (again.snapshot()["scene"] or {}).get("answered") == ["alone"])
    assert list(again.result()["answers"]) == ["alone"]
    stage.slow["/docs/index"].set()
    again.stop()


# --- the playlist, on the stands it will meet -------------------------------------------


def _stand(stage: Stage, **options) -> Stand:
    return Director(stage, [], online=lambda: options.pop("internet", True))._look_at_the_stand(
        options.pop("dgpu", "auto"), options.pop("big_screen", False), options.pop("lang", "en"), options.pop("camera", "auto")
    )


def _builders() -> list:
    """Every scene of the playlist, those that share a place in it included."""
    return [entry.build for slot in autodemo_scenes.PLAYLIST for entry in (slot if isinstance(slot, tuple) else (slot,))]


def test_the_stand_is_looked_at_once_and_the_discrete_gpu_can_be_left_out():
    stage = Stage()
    with_it, without = _stand(stage), _stand(stage, dgpu="off", lang="fr")
    assert (with_it.igpu, with_it.dgpu, with_it.npu, with_it.cameras, with_it.internet, with_it.lang) == (
        "GPU.0", "GPU.1", True, [0], True, "en")
    assert without.dgpu is None and without.igpu == "GPU.0" and without.lang == "fr"
    laptop = _stand(Stage(gpus=GPUS[:1], cameras=[], devices=["CPU", "GPU.0"]), internet=False)
    assert (laptop.dgpu, laptop.npu, laptop.cameras, laptop.internet) == (None, False, [], False)
    # A camera left out is, for every scene, a camera that is not there -- and the stand still says it has one.
    covered = _stand(stage, camera="off")
    assert covered.cameras == [] and covered.cameras_found == 1
    assert Director._describe(covered)["cameras"] == 0 and Director._describe(covered)["cameras_found"] == 1
    with pytest.raises(ValueError, match="camera is"):
        _director(stage, []).start(camera="sometimes")


def test_the_page_agent_scene_tells_what_two_gpus_change_and_takes_another_brief_each_turn():
    stage = Stage()
    two = autodemo_scenes.page_agent(_stand(stage), 1)
    assert [chip.chip for chip in two.chips] == ["NPU", "Integrated GPU", "Arc Pro B60", "CPU"]
    assert [chip.stages for chip in two.chips[:3]] == [("plan",), ("images",), ("page",)]  # whose live figure goes on its line
    # The story follows the work: each step is spoken of when it is seen at work, and the result when it is in.
    assert [(beat.stage, beat.result) for beat in two.beats] == [
        ("", False), ("plan", False), ("images", False), ("page", False), ("page", False), ("", True)]
    # "That is the page's code appearing" waits for code: the step is at work a good while before, loading its model.
    assert [beat.figure for beat in two.beats] == [False, False, False, False, True, False] and "appearing" in two.beats[4].text
    assert "Neighbourhood bakery" in two.beats[0].text and "At the same time" in two.beats[3].text
    assert two.view == "page" and two.props == {"request": "Neighbourhood bakery"}
    build = two.steps[0]
    assert build.path == "/api/page-agent/build" and build.body == {"request": "A page for a bakery."}
    assert build.cancel == "/api/bricks/page-agent/stop?stage=page"
    one = autodemo_scenes.page_agent(_stand(stage, dgpu="off"), 2)
    assert [chip.chip for chip in one.chips] == ["NPU", "Integrated GPU", "CPU"] and "in turn" in one.beats[3].text
    assert one.chips[1].stages == ("images", "page")  # one GPU, two jobs: its line shows whichever is at work
    assert one.steps[0].body == {"request": "A page for a bike shop.", "image_device": "GPU.0", "page_device": "GPU.0"}
    assert one.at_most > two.at_most  # in turn takes longer, and is given longer
    assert isinstance(autodemo_scenes.page_agent(_stand(Stage(gpus=[])), 1), Skip)


def test_the_other_scenes_fit_the_stand_or_say_why_they_cannot_play():
    stage = Stage()
    stand = _stand(stage)
    receipts = autodemo_scenes.expense_extraction(stand, 1)
    start = next(step for step in receipts.steps if step.path == "/api/expense-extract/start")
    assert start.body["folder"] == "C:/receipts/scanned"  # the worn ones: receipts that look like receipts
    assert (start.body["ocr_compute_device"], start.body["llm_compute_device"]) == ("GPU.0", "NPU")
    assert receipts.stop == ("/api/expense-extract/stop", "/api/expense-extract/report/close")
    assert any(isinstance(step, Until) and step.key == "running" for step in receipts.steps)
    # The stage shows the receipts before they are read: their pictures come with the scene.
    assert receipts.view == "receipts" and receipts.props["receipts"] == [
        {"name": "01-cafe.png", "url": "/demo-assets/expenses/scanned/01-cafe.png"},
        {"name": "02-taxi.png", "url": "/demo-assets/expenses/scanned/02-taxi.png"},
    ]

    city = autodemo_scenes.smart_city(stand, 1)
    feeds = city.steps[0].body["feeds"]
    assert [feed["path"] for feed in feeds] == ["https://clips.example/1.mp4", "https://clips.example/2.mp4"]  # clips, not live video sites
    assert [feed["compute_device"] for feed in feeds] == ["GPU.0", "NPU"] and city.stop == ("/api/smart-city-monitor/stop",)
    assert city.view == "cameras" and city.props["feeds"] == [
        {"id": "feed-1", "name": "Tower Bridge", "chip": "Integrated GPU"}, {"id": "feed-2", "name": "Westminster Bridge", "chip": "NPU"}]
    # With neither its videos on disk nor the internet, it cannot play (tests/test_sample_videos.py has the rest).
    offline = autodemo_scenes.smart_city(_stand(stage, internet=False), 1)
    assert isinstance(offline, Skip) and "internet" in offline.reason



def test_a_scene_written_before_its_demo_is_ready_is_held_back_by_its_key(monkeypatch):
    monkeypatch.setattr(autodemo_scenes, "HELD_BACK", {"video-commentary": "held back until its demo has had a proofing pass"})
    held = autodemo_scenes.video_commentary(_stand(Stage()), 1)
    assert isinstance(held, Skip) and "proofing pass" in held.reason  # written, and not played yet


def test_the_playlist_names_its_scenes_for_the_start_screen_and_holds_none_back():
    director = Director(Stage(), autodemo_scenes.PLAYLIST, online=lambda: True)
    assert director.keys() == [
        "page-agent", "camera", "expense-extraction", "smart-city", "herd-and-line", "documents", "video-commentary"]
    assert autodemo_scenes.HELD_BACK == {}
    listed = director.check()["playlist"]
    # A scene's key is the id it plays under: what the start screen ticks is what the stage shows.
    assert [(entry["key"], entry["id"]) for entry in listed if entry["playable"]] == [(key, key) for key in director.keys()]
    assert [entry["turns_with"] for entry in listed[3:5]] == [["herd-and-line"], ["smart-city"]]
    assert [entry["plays"] for entry in listed] == [True, True, True, True, False, True, True]  # the streets on the first turn


def test_the_documents_scene_asks_the_same_question_without_the_files_and_then_with_them():
    stage = Stage()
    scene = autodemo_scenes.documents(_stand(stage), 1)
    question = "Who makes the final go/no-go decision on the Lyon pilot, and on which date?"
    asked = [step for step in scene.steps if isinstance(step, Ask)]
    assert [(step.path, step.name, step.keep) for step in asked] == [
        ("/api/doc-qa/ask", "alone", False), ("/api/doc-qa/ingest", "read", False), ("/api/doc-qa/ask", "with", True)]
    # First the model alone: nothing is indexed for it, the models are only loaded, on the NPU.
    assert asked[0].body == {"question": question, "alone": True, "engine": "openvino", "compute_device": "NPU"}
    # Then the folder, read again each time: "now the folder is read" is said of something happening.
    assert asked[1].body == {"folder": "C:/samples/meridian-rollout-2026", "engine": "openvino", "compute_device": "NPU", "reindex": True}
    assert asked[2].body == {"question": question}  # the same question, word for word
    assert [type(step) for step in scene.steps] == [Ask, Wait, Ask, Wait, Ask]  # time to read each before the next
    # The story waits for each of the three: alone, read, and the answer.
    assert [(beat.answer, beat.result) for beat in scene.beats] == [("", False), ("alone", False), ("read", False), ("", True)]
    assert "made up" in scene.beats[1].text and "Same question, same model" in scene.beats[3].text
    assert scene.view == "documents" and scene.props == {
        "question": question, "folder": "meridian-rollout-2026",
        "files": ["00-company-brief.md", "03-release-decision-log.md"]}  # the files, shown before they are read
    assert scene.stop == ("/api/bricks/doc-qa/stop",)  # nothing left loaded on the NPU
    # Another question the next turn; without an NPU, on the GPU; with neither chip or without its folder, not at all.
    assert "Dock C" in autodemo_scenes.documents(_stand(stage), 2).props["question"]
    shared = autodemo_scenes.documents(_stand(Stage(devices=["CPU", "GPU.0"])), 1)
    assert shared.steps[0].body["compute_device"] == "GPU.0" and shared.chips[0].chip == "Integrated GPU"
    assert isinstance(autodemo_scenes.documents(_stand(Stage(gpus=[], devices=["CPU"])), 1), Skip)
    elsewhere = _stand(stage)
    elsewhere.samples = lambda demo: [{"name": "Notes", "folder": "C:/samples/other-notes", "question": "?"}]
    assert isinstance(autodemo_scenes.documents(elsewhere, 1), Skip)  # its questions are about its own folder


def test_the_camera_scene_boxes_the_visitor_and_says_what_is_going_on_or_does_not_play():
    stage = Stage()
    scene = autodemo_scenes.camera(_stand(stage), 1)
    detect, wait, comment = scene.steps[:3]
    assert detect.path == "/api/object-detection/start" and detect.body == {
        "source": "webcam", "camera_index": 0, "engine": "openvino", "compute_device": "NPU"}
    # One of the two opens the camera; the other waits for its picture and watches what it watches.
    assert isinstance(wait, Until) and (wait.path, wait.key, wait.equals) == ("/api/object-detection/detections", "watching", True)
    assert comment.path == "/api/video-commentary/start" and comment.body == {
        "source": "detector", "people": True, "mood": "plain", "voice": "", "vision_device": "GPU.0", "mood_device": "NPU"}
    assert scene.stop == ("/api/video-commentary/stop", "/api/object-detection/stop")
    assert scene.view == "camera" and scene.demo == "object-detection" and scene.props == {"chip": "NPU"}
    assert [(chip.chip, chip.demo, chip.stages) for chip in scene.chips] == [
        ("NPU", "object-detection", ("default",)), ("Integrated GPU", "video-commentary", ("vision",))]
    assert "Nothing is recorded" in scene.beats[0].text and "never what they look like" in scene.beats[2].text
    # Without an NPU the detector takes the processor: the GPU is the vision model's.
    plain = autodemo_scenes.camera(_stand(Stage(devices=["CPU", "GPU.0"])), 1)
    assert plain.steps[0].body["compute_device"] == "CPU" and plain.chips[0].chip == "CPU" and "no NPU" in plain.beats[1].text
    # No camera, or one left out when the loop was started: the scene is about the camera, and does not play.
    left_out = autodemo_scenes.camera(_stand(stage, camera="off"), 1)
    assert isinstance(left_out, Skip) and "Use the camera" in left_out.reason
    none = autodemo_scenes.camera(_stand(Stage(cameras=[])), 1)
    assert isinstance(none, Skip) and none.reason == "needs a camera"


def test_the_commentator_scene_waits_for_its_first_line_then_changes_voice_and_stays_silent():
    stage = Stage()
    scene = autodemo_scenes.video_commentary(_stand(stage), 1)
    start = scene.steps[0]
    assert start.path == "/api/video-commentary/start" and start.body == {
        "source": "file", "path": "C:/v/cattle.webm", "loop": True, "mood": "plain", "voice": "",
        "vision_device": "GPU.0", "mood_device": "NPU"}  # no sound: a stand is too loud for it
    # The voices wait for the first plain sentence, not for a clock: two models load first.
    wait = scene.steps[1]
    assert isinstance(wait, Until) and (wait.path, wait.key, wait.equals) == ("/api/video-commentary/comments", "commenting", True)
    assert [step.body["mood"] for step in scene.steps if isinstance(step, Start) and step.path.endswith("/mood")] == [
        "sports", "documentary", "upbeat"]
    assert scene.stop == ("/api/video-commentary/stop",) and scene.view == "commentary"
    assert scene.props["video"] == "Cattle on the road" and set(scene.props["moods"]) == {"plain", "sports", "documentary", "upbeat"}
    assert [chip.stages for chip in scene.chips] == [("vision",), ("mood",)]
    assert "embroiders" in scene.beats[-1].text  # it says how far to trust the voice
    # Another video each turn, of those on the machine.
    assert autodemo_scenes.video_commentary(_stand(stage), 2).steps[0].body["path"] == "C:/v/bottles.webm"
    assert autodemo_scenes.video_commentary(_stand(stage), 3).steps[0].body["path"] == "C:/v/cattle.webm"
    # Without an NPU the small model shares the GPU, and the scene says so on one chip's line.
    shared = autodemo_scenes.video_commentary(_stand(Stage(devices=["CPU", "GPU.0"])), 1)
    assert shared.steps[0].body["mood_device"] == "GPU.0" and [chip.stages for chip in shared.chips] == [("vision", "mood")]
    assert "shares the graphics chip" in shared.beats[2].text
    assert isinstance(autodemo_scenes.video_commentary(_stand(Stage(gpus=[])), 1), Skip)


def test_every_scene_tells_its_story_a_sentence_or_two_at_a_time_in_both_languages():
    stage = Stage()
    for options in ({}, {"dgpu": "off"}, {"camera": "off"}):
        english, french = _stand(stage, **options), _stand(stage, lang="fr", **options)
        for build in _builders():
            scene, scène = build(english, 1), build(french, 1)
            if build is autodemo_scenes.camera and options.get("camera") == "off":
                assert isinstance(scene, Skip) and scène.title != scene.title  # it is about the camera: it does not play
                continue
            assert isinstance(scene, Scene) and isinstance(scène, Scene)
            assert len(scene.beats) >= 3 and scene.beats[0].stage == "" and scene.beats[0].after == 0  # it opens at once
            assert len(scène.beats) == len(scene.beats) and scène.title != scene.title
            stages = {stage for chip in scene.chips for stage in chip.stages}
            for beat, temps in zip(scene.beats, scène.beats):
                assert 40 < len(beat.text) <= 320 and beat.text.count(". ") <= 2  # short enough to read standing up
                assert temps.text != beat.text
                assert (temps.stage, temps.after, temps.result, temps.figure) == (beat.stage, beat.after, beat.result, beat.figure)
                assert not beat.stage or beat.stage in stages  # a cue is a stage one of its chips really works on
                assert not beat.figure or beat.stage  # a figure is some stage's
            for chip, puce in zip(scene.chips, scène.chips):
                assert len(chip.label) <= 60 and puce.label != chip.label and puce.chip == chip.chip


# --- the routes -------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    stage = Stage()
    stage.slow["/page/build"] = threading.Event()
    stage.cancels["/page/cancel"] = "/page/build"
    director = _director(stage, [lambda stand, loop: _scene("page", Ask("/page/build", cancel="/page/cancel"))])
    monkeypatch.setattr(launcher_app, "autodemo_director", director)
    yield TestClient(launcher_app.app), stage, director
    director.stop()


def test_the_loop_is_checked_started_followed_and_stopped_through_the_api(client):
    web, stage, director = client
    assert web.get("/api/autodemo").json()["state"] == "idle"
    check = web.get("/api/autodemo/check?dgpu=off&lang=fr").json()
    assert check["stand"]["dgpu"] is None and check["stand"]["lang"] == "fr"
    assert check["playlist"][0]["playable"] is True and director.running is False
    assert check["playlist"][0]["key"] == "scene-1"
    assert web.get("/api/autodemo/check?camera=off").json()["stand"]["cameras"] == 0
    assert web.get("/api/autodemo/result").status_code == 404

    assert web.post("/api/autodemo/start", json={"lang": "de"}).status_code == 400
    refused = web.post("/api/autodemo/start", json={"scenes": ["nothing-of-the-kind"]})
    assert refused.status_code == 400 and "No such scene" in refused.json()["error"] and director.running is False
    started = web.post("/api/autodemo/start", json={
        "dgpu": "auto", "big_screen": True, "lang": "fr", "camera": "off", "scenes": ["scene-1"]})
    assert started.status_code == 200 and web.post("/api/autodemo/start", json={}).status_code == 409
    assert _until(lambda: (web.get("/api/autodemo").json()["scene"] or {}).get("id") == "page")
    assert web.get("/api/autodemo").json()["stand"]["lang"] == "fr"
    assert web.get("/api/autodemo").json()["stand"]["cameras"] == 0  # left out when it was started
    assert "Auto Demo" in launcher_app._busy_demos()  # no upgrade while the loop is on

    held = web.post("/api/autodemo/pause").json()
    assert held["paused"] is not None and held["state"] == "playing" and "/page/cancel" not in stage.posted()
    assert web.post("/api/autodemo/resume").json()["paused"] is None
    assert web.post("/api/autodemo/skip").status_code == 200
    assert web.post("/api/autodemo/stop").json()["state"] == "idle" and director.running is False
    assert "Auto Demo" not in launcher_app._busy_demos()
