"""The video commentator: when it speaks, when it keeps quiet, and in which voice.

Two models do the work (a vision model that says what it sees, a language
model that gives the line a mood); neither is loaded here. The rules that
decide whether there is something new to say are tested with stand-ins for
both, and the launcher's side through its routes with the pipeline replaced."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from launcher import app as launcher_app
from pantherlake_ai_core import sample_videos
from video_commentary import moods, pipeline, samples
from video_commentary.pipeline import Comment, Commentator


def _picture(shade: int) -> np.ndarray:
    return np.full((90, 160, 3), shade, np.uint8)


class Models:
    """Stands in for the two models: what the vision model would say of the
    next picture, and a language model that marks what it rewrote."""

    def __init__(self, *sees: str):
        self.to_see = list(sees)
        self.looked = 0
        self.said: list[tuple[str, str]] = []

    def see(self, picture):
        self.looked += 1
        return self.to_see.pop(0) if len(self.to_see) > 1 else self.to_see[0]

    def say(self, instruction, line):
        self.said.append((instruction, line))
        return f'"{line.rstrip(".")}, and what a sight!"'


def _commentator(models: Models, **options):
    clock = [0.0]
    commentator = Commentator(models.see, models.say, clock=lambda: clock[0], **options)
    return commentator, clock


def test_it_says_what_it_sees_in_the_mood_asked_and_keeps_what_was_seen_beside_it():
    models = Models("Cattle walk along a road.")
    commentator, _clock = _commentator(models)
    comment = commentator.consider(_picture(40), "upbeat")
    assert comment.seen == "Cattle walk along a road."
    assert comment.said == "Cattle walk along a road, and what a sight!"  # the quotation marks a model adds are gone
    assert comment.mood == "upbeat" and "upbeat commentator" in models.said[0][0]
    # The plain mood is the vision model's own line: the language model is not asked at all.
    plain_models = Models("Bottles move along a line.")
    plain = _commentator(plain_models)[0].consider(_picture(40), "plain")
    assert plain.said == plain.seen == "Bottles move along a line." and plain_models.said == []
    with pytest.raises(ValueError, match="Unknown mood"):
        commentator.consider(_picture(200), "fashion-critic")


def test_it_keeps_quiet_while_nothing_changes_and_looks_again_after_a_while():
    models = Models("A quiet street.", "A quiet street, still.")
    commentator, clock = _commentator(models, still_seconds=20.0)
    assert commentator.consider(_picture(40), "upbeat") is not None
    for seconds in (4, 8, 12):
        clock[0] = seconds
        assert commentator.consider(_picture(40), "upbeat") is None  # the same picture: not even looked at
    assert models.looked == 1
    clock[0] = 25  # a still scene gets a fresh look in the end
    again = commentator.consider(_picture(40), "upbeat")
    assert models.looked == 2 and again.seen == "A quiet street, still."


def test_a_picture_that_moved_but_shows_the_same_thing_is_not_said_twice():
    models = Models("People cross the street.", "People cross the street.", "A bus arrives.")
    commentator, clock = _commentator(models)
    assert commentator.consider(_picture(40), "sports") is not None
    clock[0] = 4
    assert commentator.consider(_picture(120), "sports") is None  # looked, and it is the same thing happening
    clock[0] = 8
    assert commentator.consider(_picture(200), "sports").seen == "A bus arrives."
    assert models.looked == 3 and len(models.said) == 2


def test_a_change_of_mood_is_said_at_once_without_looking_at_the_picture_again():
    models = Models("A rider follows the herd.")
    commentator, clock = _commentator(models)
    commentator.consider(_picture(40), "upbeat")
    clock[0] = 4
    comment = commentator.consider(_picture(40), "deadpan")  # nothing moved, but the voice did
    assert comment is not None and comment.mood == "deadpan" and comment.seen == "A rider follows the herd."
    assert models.looked == 1 and "deadpan" in models.said[-1][0]
    assert comment.seeing_seconds < 0.05  # no picture was read for it


def test_between_two_looks_a_new_mood_says_the_last_line_again_in_its_voice():
    models = Models("A rider follows the herd.")
    commentator, _clock = _commentator(models)
    assert commentator.revoice("sports") is None and commentator.mood == ""  # nothing seen yet: nothing to say again
    commentator.consider(_picture(40), "upbeat")
    assert commentator.revoice("upbeat") is None  # that is the voice it was said in
    again = commentator.revoice("sports")
    assert again.mood == "sports" and again.seen == "A rider follows the herd." and again.seeing_seconds == 0.0
    assert models.looked == 1 and "sports commentator" in models.said[-1][0] and commentator.mood == "sports"
    assert commentator.revoice("plain").said == "A rider follows the herd."


def test_the_picture_is_shown_to_the_model_small_and_a_small_one_is_left_alone():
    assert pipeline.shown(np.zeros((1080, 1920, 3), np.uint8)).shape == (378, 672, 3)
    small = np.zeros((240, 320, 3), np.uint8)
    assert pipeline.shown(small) is small


def test_the_moods_change_a_voice_and_none_of_them_judges_anybody():
    assert [mood.key for mood in moods.MOODS] == ["plain", "upbeat", "sports", "documentary", "deadpan"]
    assert moods.get(moods.DEFAULT).instruction and moods.get("plain").instruction is None
    for mood in moods.MOODS[1:]:
        assert "Keep every fact; add none." in mood.instruction
        # A commentator on a street is one thing; one that rates the passers-by is another brick.
        assert not any(word in mood.instruction.lower() for word in ("wear", "outfit", "fashion", "looks", "appearance"))


def test_its_samples_are_the_studios_own_videos_and_it_opens_on_one():
    assert samples.SAMPLES[0].default and [sample.default for sample in samples.SAMPLES].count(True) == 1
    for sample in samples.SAMPLES:
        video = sample_videos.for_path(sample.path)
        assert video is not None and sample.videos == (video.key,) and video.credit.split(",")[0] in sample.description


# --- in the launcher --------------------------------------------------------------------


@pytest.fixture
def running(monkeypatch):
    """The launcher with the two models replaced: a pipeline that shows a
    frame, says one thing in the mood of the moment, then waits to be stopped."""
    said = threading.Event()

    def run(*, mood, on_frame, on_comment, on_ready, stop_event, **settings):
        run.settings = settings
        on_ready()
        on_frame(np.zeros((1080, 1920, 3), np.uint8))
        on_comment(Comment(seen="Cattle walk along a road.", said="What a herd!", mood=mood(), seeing_seconds=0.9, saying_seconds=0.6, at=time.time()))
        said.set()
        stop_event.wait(5)

    monkeypatch.setattr(pipeline, "run", run)
    monkeypatch.setattr(launcher_app, "_video_commentary_devices", lambda vision, mood: ("GPU.0", "NPU"))
    monkeypatch.setattr(sample_videos, "present", lambda video: True)
    web = TestClient(launcher_app.app)
    yield web, run, said
    web.post("/api/video-commentary/stop")


def test_the_panel_is_told_the_moods_the_samples_and_which_chips_auto_means(monkeypatch):
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    monkeypatch.setattr(launcher_app, "_video_commentary_devices", lambda vision, mood: ("GPU.0", "NPU"))
    monkeypatch.setattr(launcher_app, "_DEVICE_SOURCES", {**launcher_app._DEVICE_SOURCES, "cameras": lambda: [0], "screens": lambda: []})
    told = TestClient(launcher_app.app).get("/api/video-commentary/devices").json()
    assert [mood["key"] for mood in told["moods"]] == [mood.key for mood in moods.MOODS] and told["default_mood"] == "upbeat"
    assert told["auto_devices"] == {"vision": "GPU.0", "mood": "NPU"}
    assert told["samples"][0]["name"] == "Cattle on the road" and "ready" in told["samples"][0]
    demo = next(d for d in TestClient(launcher_app.app).get("/api/demos").json() if d["id"] == "video-commentary")
    assert demo["experimental"] is True and demo["engines"] == ["openvino"]


def test_it_starts_shows_the_video_lists_what_was_said_and_stops(running):
    web, run, said = running
    started = web.post("/api/video-commentary/start", json={"source": "file", "path": str(sample_videos.CATTLE_DRIVE.path), "mood": "sports"})
    assert started.status_code == 200 and started.json()["devices"] == {"vision": "GPU.0", "mood": "NPU"}
    assert said.wait(3)
    assert run.settings["vision_device"] == "GPU.0" and run.settings["mood_device"] == "NPU" and run.settings["every"] == 4.0
    listed = web.get("/api/video-commentary/comments").json()
    assert listed["running"] is True and listed["mood"] == "sports"
    assert [(c["number"], c["said"], c["seen"], c["mood"]) for c in listed["comments"]] == [
        (1, "What a herd!", "Cattle walk along a road.", "sports")]
    assert web.get("/api/video-commentary/comments?after=1").json()["comments"] == []  # only what is new is sent again
    # The picture goes to the page smaller than the film: only the newest frame matters, as for any live feed.
    jpeg = launcher_app.video_commentary_runner.latest_jpeg()
    import cv2

    assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (540, 960)
    assert web.post("/api/video-commentary/start", json={"source": "file", "path": "x.mp4"}).status_code == 409
    assert web.post("/api/video-commentary/stop").json() == {"status": "stopped"}
    assert web.get("/api/video-commentary/comments").json()["running"] is False


def test_the_mood_can_change_while_it_runs_and_a_wrong_request_is_refused_by_name(running):
    web, _run, said = running
    assert web.post("/api/video-commentary/start", json={"source": "webcam", "camera_index": 0}).status_code == 200
    assert said.wait(3)
    assert web.post("/api/video-commentary/mood", json={"mood": "documentary"}).json() == {"mood": "documentary"}
    assert web.get("/api/video-commentary/comments").json()["mood"] == "documentary"
    refused = web.post("/api/video-commentary/mood", json={"mood": "fashion-critic"})
    assert refused.status_code == 400 and "Unknown mood" in refused.text
    web.post("/api/video-commentary/stop")
    assert web.post("/api/video-commentary/start", json={"source": "file", "path": "  "}).status_code == 400  # no file named
    assert web.post("/api/video-commentary/start", json={"source": "drone"}).status_code == 400


def test_a_sample_video_not_fetched_yet_is_fetched_before_it_is_watched(running, monkeypatch):
    web, _run, said = running
    fetched = []
    monkeypatch.setattr(sample_videos, "present", lambda video: False)
    monkeypatch.setattr(sample_videos, "download", lambda video, *args, **kwargs: fetched.append(video.key))
    assert web.post("/api/video-commentary/start", json={"source": "file", "path": str(sample_videos.CAPPING_LINE.path)}).status_code == 200
    assert said.wait(3) and fetched == ["bottle-capping-line"]


def test_the_vision_model_is_not_sent_to_the_npu(monkeypatch):
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    refused = TestClient(launcher_app.app).post("/api/video-commentary/start", json={"source": "webcam", "vision_device": "NPU"})
    assert refused.status_code == 400 and "does not run on the NPU" in refused.text
