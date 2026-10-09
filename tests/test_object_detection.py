"""Object detection: what its proofing pass of 2026-10-10 settled.

A video file is a source like the camera and the screen; a source that is
wrong is refused before a model is loaded for it; a big picture is looked at
and drawn no wider than it needs to be, with boxes sized to it; and the
launcher says how many of each thing are in the picture. No model is loaded
and no camera is opened here."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from launcher import app as launcher_app
from object_detection import pipeline, samples
from object_detection.draw import draw_detections
from object_detection.types import Detection
from pantherlake_ai_core import sample_videos
from pantherlake_ai_core.engine import Engine


def test_a_wrong_source_is_refused_before_a_model_is_loaded_for_it(monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(pipeline, "create_detector", lambda *args, **kwargs: loaded.append(args))
    with pytest.raises(ValueError, match="Unknown source 'camera': one of webcam, screen, file"):
        pipeline.run(source="camera", engine=Engine.PORTABLE, compute_device="cpu", on_frame=lambda frame, found: None)
    with pytest.raises(ValueError, match="A video file is needed"):
        pipeline.run(source="file", engine=Engine.PORTABLE, compute_device="cpu", on_frame=lambda frame, found: None)
    with pytest.raises(FileNotFoundError, match="Not a file"):
        pipeline.run(source="file", path=str(tmp_path / "gone.mp4"), engine=Engine.PORTABLE, compute_device="cpu",
                     on_frame=lambda frame, found: None)
    assert loaded == []


def test_a_video_file_is_watched_like_a_camera_and_a_big_picture_is_looked_at_smaller(monkeypatch, tmp_path):
    clip = tmp_path / "street.mp4"
    clip.write_bytes(b"not really a video")
    asked = {}

    def frames(path, *, loop, stop_event):
        asked.update(path=path, loop=loop)
        yield np.zeros((1800, 2880, 3), np.uint8)  # a laptop's whole screen
        yield np.zeros((432, 768, 3), np.uint8)  # a small clip

    class Detector:
        seen: list[tuple] = []

        def detect(self, frame):
            self.seen.append(frame.shape)
            return [Detection("car", 0.9, (10, 10, 60, 40))]

    monkeypatch.setattr(pipeline.video, "stream_video_file_frames", frames)
    monkeypatch.setattr(pipeline, "create_detector", lambda *args, **kwargs: Detector())
    shown, ready = [], []
    pipeline.run(source="file", path=str(clip), loop=False, engine=Engine.PORTABLE, compute_device="cpu",
                 on_frame=lambda frame, found: shown.append((frame.shape, found[0].label)), on_ready=lambda: ready.append(True))
    assert asked == {"path": str(clip), "loop": False} and ready == [True]
    # The model looks at 640 pixels whatever it is given: nothing is lost at 1280, and a frame's time is halved.
    assert Detector.seen == [(800, 1280, 3), (432, 768, 3)]
    assert shown == [((800, 1280, 3), "car"), ((432, 768, 3), "car")]  # the boxes belong to the picture handed over
    small = np.zeros((432, 768, 3), np.uint8)
    assert pipeline.shown(small) is small


def test_boxes_are_drawn_to_the_size_of_the_picture():
    def edge(height: int) -> int:
        """How thick the box's left edge is, in pixels, half-way down it."""
        frame = np.zeros((height, height * 16 // 9, 3), np.uint8)
        box = (height // 2, height // 4, height, height * 3 // 4)
        drawn = draw_detections(frame, [Detection("person", 0.8, box)])
        row = drawn[height // 2, : height * 3 // 4]
        return int(np.count_nonzero(row.any(axis=1)))

    assert edge(1080) >= 2 * edge(540) - 1 > edge(540)  # a hairline on a tall frame is no box once the page shows it small
    untouched = np.zeros((90, 160, 3), np.uint8)
    draw_detections(untouched, [Detection("person", 0.8, (10, 10, 50, 50))])
    assert not untouched.any()  # the frame handed in is left as it was


def test_its_samples_are_the_studios_own_videos_and_it_opens_on_the_busiest():
    assert samples.SAMPLES[0].default and samples.SAMPLES[0].name == sample_videos.TORONTO.name
    assert [sample.default for sample in samples.SAMPLES].count(True) == 1
    for sample in samples.SAMPLES:
        video = sample_videos.for_path(sample.path)
        assert sample.kind == "file" and video is not None and sample.videos == (video.key,)


# --- in the launcher --------------------------------------------------------------------


@pytest.fixture
def web(monkeypatch):
    """The launcher with the detector replaced: a pipeline that shows one
    frame with a few things in it, then waits to be stopped."""
    shown = threading.Event()

    def run(*, on_frame, on_ready, stop_event, **settings):
        run.settings = settings
        on_ready()
        on_frame(np.zeros((720, 1280, 3), np.uint8), [
            Detection("person", 0.91, (10, 10, 80, 200)), Detection("car", 0.8, (300, 300, 500, 400)),
            Detection("person", 0.6, (100, 10, 180, 200)),
        ])
        shown.set()
        stop_event.wait(5)

    monkeypatch.setattr(pipeline, "run", run)
    monkeypatch.setattr(sample_videos, "present", lambda video: True)
    client = TestClient(launcher_app.app)
    yield client, run, shown
    client.post("/api/object-detection/stop")
    deadline = time.time() + 3
    while launcher_app.object_detection_runner.running and time.time() < deadline:
        time.sleep(0.02)


def test_the_launcher_refuses_a_wrong_source_itself_instead_of_answering_started(web, tmp_path):
    client, _run, _shown = web
    body = {"engine": "portable", "compute_device": "cpu"}
    # "camera" is what an Auto Demo scene once asked for: answered "started", and failed in a thread nobody watched.
    refused = client.post("/api/object-detection/start", json={**body, "source": "camera"})
    assert refused.status_code == 400 and "one of webcam, screen, file" in refused.json()["error"]
    assert client.post("/api/object-detection/start", json={**body, "source": "file", "path": " "}).status_code == 400
    gone = client.post("/api/object-detection/start", json={**body, "source": "file", "path": str(tmp_path / "gone.mp4")})
    assert gone.status_code == 400 and "Not a file" in gone.json()["error"]
    assert launcher_app.object_detection_runner.running is False


def test_a_video_file_is_played_and_the_page_is_told_what_is_in_the_picture(web, tmp_path):
    client, run, shown = web
    clip = tmp_path / "street.mp4"
    clip.write_bytes(b"not really a video")
    started = client.post("/api/object-detection/start", json={
        "source": "file", "path": f"  {clip} ", "engine": "portable", "compute_device": "cpu"})
    assert started.json() == {"status": "started"} and shown.wait(3)
    assert (run.settings["source"], run.settings["path"], run.settings["loop"]) == ("file", str(clip), True)
    told = client.get("/api/object-detection/detections").json()
    assert told["running"] is True and told["error"] is None and len(told["detections"]) == 3
    assert list(told["counts"].items()) == [("person", 2), ("car", 1)]  # most first: what the Auto Demo's stage shows
    assert client.post("/api/object-detection/stop").json() == {"status": "stopped"}
    assert client.get("/api/object-detection/detections").json()["counts"] == {}


def test_a_sample_video_not_fetched_yet_is_fetched_before_it_is_watched(web, monkeypatch):
    client, _run, shown = web
    fetched = []
    monkeypatch.setattr(sample_videos, "present", lambda video: False)
    monkeypatch.setattr(sample_videos, "download", lambda video, *args, **kwargs: fetched.append(video.key))
    monkeypatch.setattr(pipeline, "frames_from", lambda *args, **kwargs: (frame for frame in ()))  # the file is not there in this test
    started = client.post("/api/object-detection/start", json={
        "source": "file", "path": str(sample_videos.TORONTO.path), "engine": "portable", "compute_device": "cpu"})
    assert started.status_code == 200 and shown.wait(3) and fetched == [sample_videos.TORONTO.key]


def test_the_panel_is_offered_the_sample_videos(monkeypatch):
    monkeypatch.setattr(launcher_app, "_DEVICE_SOURCES", {**launcher_app._DEVICE_SOURCES, "cameras": lambda: [], "screens": lambda: []})
    told = TestClient(launcher_app.app).get("/api/object-detection/devices").json()
    assert told["samples"][0]["name"] == sample_videos.TORONTO.name and told["samples"][0]["default"] is True
    assert all(sample["kind"] == "file" and "ready" in sample for sample in told["samples"])


def test_another_demo_can_watch_what_the_detector_watches_without_the_boxes(web, tmp_path):
    """A camera is opened by one demo only: the Video Commentator beside the
    detector is handed the detector's frames, as captured."""
    client, _run, shown = web
    runner = launcher_app.object_detection_runner
    with pytest.raises(Exception, match="not running"):
        next(runner.frames())
    assert client.get("/api/object-detection/detections").json()["watching"] is False
    clip = tmp_path / "street.mp4"
    clip.write_bytes(b"not really a video")
    client.post("/api/object-detection/start", json={"source": "file", "path": str(clip), "engine": "portable", "compute_device": "cpu"})
    assert shown.wait(3) and runner.source == "file"
    assert client.get("/api/object-detection/detections").json()["watching"] is True
    stop = threading.Event()
    frames = runner.frames(stop, every=0.01)
    first = next(frames)
    assert first.shape == (720, 1280, 3) and not first.any()  # the picture as captured: no box was drawn on it
    stop.set()
    assert list(frames) == []  # and it ends when it is told to, or when the detector stops
