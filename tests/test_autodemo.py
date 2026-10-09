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
from launcher.autodemo import Ask, Beat, Chip, Director, Scene, Skip, Stand, Start, Until, Wait
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
        {"text": "page begins.", "after": 0.0, "stage": "", "result": False, "figure": False},
        {"text": "The small model is at work.", "after": 0.0, "stage": "plan", "result": False, "figure": False},
        {"text": "Done.", "after": 0.0, "stage": "", "result": True, "figure": False},
    ]
    assert scene["chips"] == [{"chip": "NPU", "label": "Plans · a small model", "demo": "page", "stages": ("plan",)}]
    assert scene["props"] == {"request": "page"} and scene["result_ready"] is False and director.result() is None
    assert state["playlist"] == [{"id": "page", "title": "Page", "playable": True, "demo": "page", "reason": ""}]
    assert state["stand"] == {
        "npu": True, "igpu": "GPU.0", "dgpu": "GPU.1", "cameras": 1, "internet": False, "big_screen": True, "lang": "fr",
    }
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


# --- the playlist, on the stands it will meet -------------------------------------------


def _stand(stage: Stage, **options) -> Stand:
    return Director(stage, [], online=lambda: options.pop("internet", True))._look_at_the_stand(
        options.pop("dgpu", "auto"), options.pop("big_screen", False), options.pop("lang", "en")
    )


def test_the_stand_is_looked_at_once_and_the_discrete_gpu_can_be_left_out():
    stage = Stage()
    with_it, without = _stand(stage), _stand(stage, dgpu="off", lang="fr")
    assert (with_it.igpu, with_it.dgpu, with_it.npu, with_it.cameras, with_it.internet, with_it.lang) == (
        "GPU.0", "GPU.1", True, [0], True, "en")
    assert without.dgpu is None and without.igpu == "GPU.0" and without.lang == "fr"
    laptop = _stand(Stage(gpus=GPUS[:1], cameras=[], devices=["CPU", "GPU.0"]), internet=False)
    assert (laptop.dgpu, laptop.npu, laptop.cameras, laptop.internet) == (None, False, [], False)


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

    held = autodemo_scenes.seeing_and_answering(stand, 1)
    assert isinstance(held, Skip) and "proofing pass" in held.reason  # written, and not played yet


def test_seeing_and_answering_is_ready_for_the_day_it_is_let_through(monkeypatch):
    monkeypatch.setattr(autodemo_scenes, "HELD_BACK", {})
    stage = Stage()
    scene = autodemo_scenes.seeing_and_answering(_stand(stage), 1)
    assert scene.steps[0].body == {"source": "camera", "camera_index": 0, "engine": "openvino", "compute_device": "GPU.0"}
    assert "Nothing it sees is recorded." in scene.beats[0].text
    assert [step.path for step in scene.steps] == ["/api/object-detection/start", "/api/doc-qa/ingest", "/api/doc-qa/ask"]
    no_camera = autodemo_scenes.seeing_and_answering(_stand(Stage(cameras=[])), 1)
    assert no_camera.steps[0].body["source"] == "screen" and "recorded" not in no_camera.beats[0].text


def test_every_scene_tells_its_story_a_sentence_or_two_at_a_time_in_both_languages(monkeypatch):
    monkeypatch.setattr(autodemo_scenes, "HELD_BACK", {})
    stage = Stage()
    for options in ({}, {"dgpu": "off"}):
        english, french = _stand(stage, **options), _stand(stage, lang="fr", **options)
        for build in autodemo_scenes.PLAYLIST:
            scene, scène = build(english, 1), build(french, 1)
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
    assert web.get("/api/autodemo/result").status_code == 404

    assert web.post("/api/autodemo/start", json={"lang": "de"}).status_code == 400
    started = web.post("/api/autodemo/start", json={"dgpu": "auto", "big_screen": True, "lang": "fr"})
    assert started.status_code == 200 and web.post("/api/autodemo/start", json={}).status_code == 409
    assert _until(lambda: (web.get("/api/autodemo").json()["scene"] or {}).get("id") == "page")
    assert web.get("/api/autodemo").json()["stand"]["lang"] == "fr"
    assert "Auto Demo" in launcher_app._busy_demos()  # no upgrade while the loop is on

    held = web.post("/api/autodemo/pause").json()
    assert held["paused"] is not None and held["state"] == "playing" and "/page/cancel" not in stage.posted()
    assert web.post("/api/autodemo/resume").json()["paused"] is None
    assert web.post("/api/autodemo/skip").status_code == 200
    assert web.post("/api/autodemo/stop").json()["state"] == "idle" and director.running is False
    assert "Auto Demo" not in launcher_app._busy_demos()
