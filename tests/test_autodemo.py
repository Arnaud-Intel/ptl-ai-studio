"""The Auto Demo: the app running itself on a stand (docs/AUTO_DEMO.md).

The director against a stand-in for the launcher's routes, so nothing here
loads a model or needs a browser: scenes play in order and go round, a scene
that fails is skipped and noted, three in a row stop the loop, a visitor's
touch steps the loop aside and it comes back, and whatever a scene started
is stopped when it ends -- however it ends."""
from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from launcher import app as launcher_app
from launcher import autodemo, autodemo_scenes
from launcher.autodemo import Ask, Chip, Director, Scene, Skip, Stand, Start, Until, Wait
from launcher.errors import Conflict

GPUS = [
    {"id": "GPU.0", "full_name": "Intel(R) Arc(TM) B390 GPU (iGPU)"},
    {"id": "GPU.1", "full_name": "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)"},
]
SAMPLES = {
    "page-agent": [{"name": "Neighbourhood bakery", "prompt": "A page for a bakery."},
                   {"name": "Mountain bike rental", "prompt": "A page for a bike shop."}],
    "expense-extract": [{"name": "Scanned & photographed", "folder": "C:/receipts/scanned"},
                        {"name": "Start here · 3 everyday expenses", "folder": "C:/receipts/quick-start"}],
    "smart-city-monitor": [
        {"name": "Shibuya", "group": "YouTube", "feeds": "https://video.example/live"},
        {"name": "Tower Bridge", "group": "Other", "feeds": "https://clips.example/1.mp4"},
        {"name": "Westminster Bridge", "group": "Other", "feeds": "https://clips.example/2.mp4"},
        {"name": "London, two chips", "group": "Other", "feeds": "https://clips.example/1.mp4|NPU\nhttps://clips.example/2.mp4"},
    ],
    "doc-qa": [{"name": "Can the pilot launch?", "folder": "C:/docs", "question": "Is it approved?"}],
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
        self.answers: dict[str, dict] = {}

    def __call__(self, method, path, body, timeout):
        self.calls.append((method, path, body))
        if path in self.fail:
            raise RuntimeError(self.fail[path])
        if path in self.cancels:
            self.slow[self.cancels[path]].set()
        if path in self.slow:
            self.slow[path].wait(5)
        if path == "/api/system/gpu-devices":
            return self.gpus
        if path == "/api/object-detection/devices":
            return {"cameras": self.cameras, "openvino_devices": self.devices}
        if path.endswith("/devices"):
            return {"samples": SAMPLES.get(path.split("/")[2], [])}
        return self.answers.get(path, {"status": "ok"})

    def posted(self) -> list[str]:
        return [path for method, path, _body in self.calls if method == "POST"]


def _scene(name: str, *steps, stop=(), hold=0.0, at_most=5.0) -> Scene:
    return Scene(
        id=name, title=name.title(), demo=name, happening=f"{name} is happening.",
        chips=(Chip("NPU", "A small model.", "It is small.", name),), look_at=("The result.",),
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


def test_the_scene_in_hand_says_what_is_happening_on_which_chip_and_why():
    stage = Stage()
    stage.slow["/b/ask"] = threading.Event()
    stage.answers["/b/ask"] = {"html": "<html></html>"}
    director = _director(stage, [lambda stand, loop: _scene("page", Ask("/b/ask"), hold=0.3)])
    director.start(big_screen=True)
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("phase") == "working")
    state = director.snapshot()
    scene = state["scene"]
    assert (state["state"], scene["id"], scene["demo"], scene["happening"]) == ("playing", "page", "page", "page is happening.")
    assert scene["chips"] == [{"chip": "NPU", "runs": "A small model.", "why": "It is small.", "demo": "page", "stage": "default"}]
    assert scene["look_at"] == ["The result."] and scene["result_ready"] is False and director.result() is None
    assert state["playlist"] == [{"id": "page", "title": "Page", "playable": True, "demo": "page", "reason": ""}]
    assert state["stand"] == {"npu": True, "igpu": "GPU.0", "dgpu": "GPU.1", "cameras": 1, "internet": False, "big_screen": True}
    assert state["awake"] is True
    # The answer is kept for the page to draw, and stays on screen for the scene's hold.
    stage.slow["/b/ask"].set()
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("phase") == "showing")
    assert director.result() == {"scene": "page", "data": {"html": "<html></html>"}}
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


def test_a_visitors_touch_steps_the_loop_aside_and_it_comes_back_at_the_next_scene():
    stage = Stage()
    stage.slow["/page/build"] = threading.Event()
    stage.cancels["/page/cancel"] = "/page/build"
    asking = lambda stand, loop: _scene("asking", Ask("/page/build", cancel="/page/cancel"), stop=["/page/stop"])  # noqa: E731
    stream = lambda stand, loop: _scene("stream", Start("/cam/start"), Wait(5), stop=["/cam/stop"])  # noqa: E731
    director = _director(stage, [asking, stream], idle_resume=0.25)
    director.start()
    assert _until(lambda: "/page/build" in stage.posted())
    director.touch()
    assert _until(lambda: director.snapshot()["state"] == autodemo.PAUSED)
    state = director.snapshot()
    assert state["scene"] is None and state["paused"]["resumes_at"] > state["now"]
    assert "/page/cancel" in stage.posted() and "/page/stop" in stage.posted() and not state["failures"]  # interrupted is not failed
    # Still at it: the resumption is put off again.
    time.sleep(0.15)
    later = director.touch()["paused"]["resumes_at"]
    assert later > state["paused"]["resumes_at"]
    # Left alone, the loop takes up again with the scene after the one that was interrupted.
    assert _until(lambda: (director.snapshot()["scene"] or {}).get("id") == "stream")
    assert "/cam/start" in stage.posted()
    # Skipping ends a scene at once, and resuming by hand does not wait for the idle time.
    director.skip()
    assert _until(lambda: "/cam/stop" in stage.posted())
    director.touch()
    assert _until(lambda: director.snapshot()["state"] == autodemo.PAUSED)
    director.resume()
    assert _until(lambda: director.snapshot()["state"] == autodemo.PLAYING)
    director.stop()
    assert director.snapshot()["state"] == autodemo.IDLE


def test_the_loop_cannot_be_started_twice_and_a_stand_with_nothing_to_play_says_so():
    stage = Stage()
    stage.slow["/x"] = threading.Event()
    director = _director(stage, [lambda stand, loop: _scene("x", Ask("/x"))])
    director.start()
    with pytest.raises(Conflict):
        director.start()
    with pytest.raises(ValueError):
        _director(stage, []).start(dgpu="maybe")
    director.stop()
    nothing = _director(stage, [lambda stand, loop: Skip("Cameras", "needs the internet")])
    nothing.start()
    assert _until(lambda: nothing.snapshot()["state"] == autodemo.STOPPED)
    assert "Cameras (needs the internet)" in nothing.snapshot()["notice"]


# --- the playlist, on the stands it will meet -------------------------------------------


def _stand(stage: Stage, **options) -> Stand:
    return Director(stage, [], online=lambda: options.pop("internet", True))._look_at_the_stand(
        options.pop("dgpu", "auto"), options.pop("big_screen", False)
    )


def test_the_stand_is_looked_at_once_and_the_discrete_gpu_can_be_left_out():
    stage = Stage()
    with_it, without = _stand(stage), _stand(stage, dgpu="off")
    assert (with_it.igpu, with_it.dgpu, with_it.npu, with_it.cameras, with_it.internet) == ("GPU.0", "GPU.1", True, [0], True)
    assert without.dgpu is None and without.igpu == "GPU.0"
    laptop = _stand(Stage(gpus=GPUS[:1], cameras=[], devices=["CPU", "GPU.0"]), internet=False)
    assert (laptop.dgpu, laptop.npu, laptop.cameras, laptop.internet) == (None, False, [], False)


def test_the_page_agent_scene_says_what_two_gpus_change_and_takes_another_brief_each_turn():
    stage = Stage()
    two = autodemo_scenes.page_agent(_stand(stage), 1)
    assert [chip.chip for chip in two.chips] == ["NPU", "Integrated GPU", "Arc Pro B60", "CPU"]
    assert [chip.stage for chip in two.chips[:3]] == ["plan", "images", "page"]  # where each chip's live figure comes from
    assert "at the same time" in two.happening and "Neighbourhood bakery" in two.happening
    build = two.steps[0]
    assert build.path == "/api/page-agent/build" and build.body == {"request": "A page for a bakery."}
    assert build.cancel == "/api/bricks/page-agent/stop?stage=page"
    one = autodemo_scenes.page_agent(_stand(stage, dgpu="off"), 2)
    assert [chip.chip for chip in one.chips] == ["NPU", "Integrated GPU", "CPU"] and "one after the other" in one.happening
    assert one.steps[0].body == {"request": "A page for a bike shop.", "image_device": "GPU.0", "page_device": "GPU.0"}
    assert one.at_most > two.at_most  # in turn takes longer, and is given longer
    assert isinstance(autodemo_scenes.page_agent(_stand(Stage(gpus=[])), 1), Skip)


def test_the_other_scenes_fit_the_stand_or_say_why_they_cannot_play():
    stage = Stage()
    stand = _stand(stage)
    receipts = autodemo_scenes.expense_extraction(stand, 1)
    start = next(step for step in receipts.steps if step.path == "/api/expense-extract/start")
    assert start.body["folder"] == "C:/receipts/quick-start"  # the short one: three receipts, about a minute
    assert (start.body["ocr_compute_device"], start.body["llm_compute_device"]) == ("GPU.0", "NPU")
    assert receipts.stop == ("/api/expense-extract/stop", "/api/expense-extract/report/close")
    assert any(isinstance(step, Until) and step.key == "running" for step in receipts.steps)

    city = autodemo_scenes.smart_city(stand, 1)
    feeds = city.steps[0].body["feeds"]
    assert [feed["path"] for feed in feeds] == ["https://clips.example/1.mp4", "https://clips.example/2.mp4"]  # clips, not live video sites
    assert [feed["compute_device"] for feed in feeds] == ["GPU.0", "NPU"] and city.stop == ("/api/smart-city-monitor/stop",)
    offline = autodemo_scenes.smart_city(_stand(stage, internet=False), 1)
    assert isinstance(offline, Skip) and "internet" in offline.reason

    held = autodemo_scenes.seeing_and_answering(stand, 1)
    assert isinstance(held, Skip) and "proofing pass" in held.reason  # written, and not played yet


def test_seeing_and_answering_is_ready_for_the_day_it_is_let_through(monkeypatch):
    monkeypatch.setattr(autodemo_scenes, "HELD_BACK", {})
    stage = Stage()
    scene = autodemo_scenes.seeing_and_answering(_stand(stage), 1)
    assert scene.steps[0].body == {"source": "camera", "camera_index": 0, "engine": "openvino", "compute_device": "GPU.0"}
    assert "Nothing the camera sees is recorded." in scene.chips[0].why
    assert [step.path for step in scene.steps] == ["/api/object-detection/start", "/api/doc-qa/ingest", "/api/doc-qa/ask"]
    no_camera = autodemo_scenes.seeing_and_answering(_stand(Stage(cameras=[])), 1)
    assert no_camera.steps[0].body["source"] == "screen" and "recorded" not in no_camera.chips[0].why


def test_every_scene_of_the_playlist_explains_itself():
    stage = Stage()
    stand = _stand(stage)
    for build in autodemo_scenes.PLAYLIST:
        scene = build(stand, 1)
        if isinstance(scene, Skip):
            assert scene.title and scene.reason
            continue
        assert len(scene.happening) > 120 and len(scene.look_at) >= 2 and len(scene.chips) >= 2
        for chip in scene.chips:
            assert chip.chip and len(chip.runs) > 20 and len(chip.why) > 40  # which engine, and why this chip


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
    check = web.get("/api/autodemo/check?dgpu=off").json()
    assert check["stand"]["dgpu"] is None and check["playlist"][0]["playable"] is True and director.running is False
    assert web.get("/api/autodemo/result").status_code == 404

    started = web.post("/api/autodemo/start", json={"dgpu": "auto", "big_screen": True})
    assert started.status_code == 200 and web.post("/api/autodemo/start", json={}).status_code == 409
    assert web.post("/api/autodemo/start", json={"dgpu": "maybe"}).status_code in (400, 409)
    assert _until(lambda: (web.get("/api/autodemo").json()["scene"] or {}).get("id") == "page")
    assert "Auto Demo" in launcher_app._busy_demos()  # no upgrade while the loop is on

    assert web.post("/api/autodemo/touch").json()["state"] in ("playing", "paused")
    assert _until(lambda: web.get("/api/autodemo").json()["state"] == "paused")
    assert web.post("/api/autodemo/resume").status_code == 200
    assert _until(lambda: web.get("/api/autodemo").json()["state"] == "playing")
    assert web.post("/api/autodemo/skip").status_code == 200
    assert web.post("/api/autodemo/stop").json()["state"] == "idle" and director.running is False
    assert "Auto Demo" not in launcher_app._busy_demos()
