"""The video commentator: when it speaks, when it keeps quiet, and in which voice.

Two models do the work (a vision model that says what it sees, a language
model that gives the line a mood); neither is loaded here. The rules that
decide whether there is something new to say are tested with stand-ins for
both, and the launcher's side through its routes with the pipeline replaced.

Then the line said aloud: which voice says it and how, what a voice that
fails costs (the sound, not the commentary), and the page being handed the
sound of each spoken line. No voice is loaded and nothing is played."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from launcher import app as launcher_app
from pantherlake_ai_core import sample_videos
from video_commentary import moods, pipeline, samples, voices
from video_commentary.pipeline import Comment, Commentator, Voicing


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


# --- said aloud -------------------------------------------------------------------------


class Voices:
    """Stands in for the voices: a second of sound per line, and a record of who was asked to say what."""

    def __init__(self, fail: str = ""):
        self.fail = fail
        self.asked: list[tuple[str, str, str]] = []

    def speak(self, voice_key, text, mood_key):
        self.asked.append((voice_key, text, mood_key))
        if self.fail:
            raise RuntimeError(self.fail)
        return np.zeros(22050, np.float32), 22050


def _comment(said="What a herd!", mood="sports") -> Comment:
    return Comment(seen="Cattle walk along a road.", said=said, mood=mood, seeing_seconds=0.9, saying_seconds=0.6, at=0.0)


def test_a_line_is_spoken_only_when_a_voice_is_on_and_the_next_look_waits_for_it():
    speaker, voice, work = Voices(), [""], []
    voiced = Voicing(speaker, lambda: voice[0], on_work=lambda stage, working, stats: work.append((stage, working)), clock=lambda: 100.0)
    comment, speech = voiced(_comment())
    assert speech is None and comment.voice == "" and speaker.asked == [] and voiced.quiet_after == 0.0  # written only
    voice[0] = "studio"  # switched on while the video plays
    comment, (audio, rate) = voiced(_comment())
    assert speaker.asked == [("studio", "What a herd!", "sports")]  # the line as said, and the mood it is said in
    assert (comment.voice, comment.speech_seconds, len(audio), rate) == ("studio", 1.0, 22050, 22050)
    assert comment.said == "What a herd!" and comment.voicing_seconds >= 0
    assert voiced.quiet_after == 101.0  # the picture is not looked at again until the line has been said
    assert work == [("voice", True), ("voice", False)]
    voice[0] = "off"
    assert voiced(_comment())[1] is None


def test_a_voice_that_fails_costs_the_sound_not_the_commentary_and_is_not_tried_again():
    speaker, voice, told = Voices(fail="the model would not load"), ["cloned"], []
    voiced = Voicing(speaker, lambda: voice[0], on_failed=told.append)
    comment, speech = voiced(_comment())
    assert speech is None and comment.said == "What a herd!" and told == ["the model would not load"]
    voiced(_comment())
    voiced(_comment())
    assert len(speaker.asked) == 1 and len(told) == 1  # said once, and left alone
    voice[0] = "studio"  # another voice is another try
    voiced(_comment())
    assert len(speaker.asked) == 2 and speaker.asked[-1][0] == "studio"
    with pytest.raises(ValueError, match="Unknown voice"):
        Voicing(speaker, lambda: "robot")(_comment())


def test_the_studio_voice_takes_its_delivery_from_the_mood_and_the_cloned_one_is_lent():
    assert [voice.key for voice in voices.VOICES] == ["studio", "cloned"]
    assert (voices.get(None), voices.get(""), voices.get("Off"), voices.get(" Studio ")) == ("", "", "", "studio")
    assert set(voices.DELIVERY) == {mood.key for mood in moods.MOODS}  # every mood is read some way
    assert voices.DELIVERY["sports"] == "excited" and voices.DELIVERY["upbeat"] == "cheerful"
    assert voices.spoken('"What a *herd*,"  she said.') == "What a herd , she said."  # what a caption carries and a voice cannot

    made, said = [], []

    class Studio:
        def __init__(self):
            made.append(self)

        def speak(self, text, mood_key):
            said.append((text, mood_key))
            return np.zeros(10, np.float32), 22050

    cloned = []
    speaker = voices.Speaker(clone=lambda text: (cloned.append(text) or [0.0] * 5, 24000), studio=Studio)
    assert made == []  # a commentary nobody asked to hear loads no voice
    speaker.speak("studio", "Go on!", "sports")
    speaker.speak("studio", "And again.", "deadpan")
    assert len(made) == 1 and said == [("Go on!", "sports"), ("And again.", "deadpan")]
    audio, rate = speaker.speak("cloned", '"In my own voice."', "sports")
    assert cloned == ["In my own voice."] and rate == 24000 and audio.dtype == np.float32 and len(made) == 1
    with pytest.raises(RuntimeError, match="No cloned voice"):
        voices.Speaker(studio=Studio).speak("cloned", "Hello.", "plain")
    with pytest.raises(ValueError, match="Nothing to say"):
        speaker.speak("studio", ' "" ', "plain")


def test_the_studio_voice_is_only_asked_for_deliveries_it_has():
    voice_model = pytest.importorskip("voice_clone_studio.voice_model")
    assert set(voices.DELIVERY.values()) <= set(voice_model.STYLES)


# --- in the launcher --------------------------------------------------------------------


class Enrolled:
    """Stands in for the Voice Clone Studio's runner: a voice enrolled, or not."""

    def __init__(self, enrolled=True, device="CPU"):
        self.enrolled, self.device, self.said = enrolled, device, []

    def speak(self, text):
        self.said.append(text)
        return np.zeros(2400, np.float32), 24000


@pytest.fixture
def running(monkeypatch):
    """The launcher with the two models replaced: a pipeline that shows a
    frame, says one thing in the mood of the moment -- aloud, if a voice is
    on -- then waits to be stopped."""
    said = threading.Event()

    def run(*, mood, on_frame, on_comment, on_ready, stop_event, voice=lambda: "", clone=None, **settings):
        run.settings = settings
        on_ready()
        on_frame(np.zeros((1080, 1920, 3), np.uint8))
        comment = Comment(seen="Cattle walk along a road.", said="What a herd!", mood=mood(), seeing_seconds=0.9, saying_seconds=0.6, at=time.time())
        if voice():
            speech = clone(comment.said) if voice() == "cloned" else (np.zeros(22050, np.float32), 22050)
            comment = Comment(**{**comment.__dict__, "voice": voice(), "voicing_seconds": 0.5, "speech_seconds": len(speech[0]) / speech[1]})
            on_comment(comment, speech)
        else:
            on_comment(comment, None)
        said.set()
        stop_event.wait(5)

    monkeypatch.setattr(pipeline, "run", run)
    monkeypatch.setattr(launcher_app, "_video_commentary_devices", lambda vision, mood: ("GPU.0", "NPU"))
    monkeypatch.setattr(sample_videos, "present", lambda video: True)
    monkeypatch.setattr(launcher_app.video_commentary_runner, "_cloned", Enrolled(enrolled=False))
    monkeypatch.setattr(launcher_app.video_commentary_runner, "_voice", "")
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
    assert listed["commenting"] is True and listed["voice"] == "" and listed["comments"][0]["speech"] is False
    assert web.get("/api/video-commentary/speech/1").status_code == 404  # written only: there is no sound to ask for
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


def test_a_line_said_aloud_is_handed_to_the_page_with_its_sound(running):
    web, _run, said = running
    started = web.post("/api/video-commentary/start", json={"source": "webcam", "voice": "studio"})
    assert started.status_code == 200 and started.json()["voice"] == "studio"
    assert said.wait(3)
    listed = web.get("/api/video-commentary/comments").json()
    comment = listed["comments"][0]
    assert listed["voice"] == "studio" and (comment["speech"], comment["voice"], comment["speech_seconds"]) == (True, "studio", 1.0)
    sound = web.get("/api/video-commentary/speech/1")
    assert sound.status_code == 200 and sound.headers["content-type"] == "audio/wav" and sound.content[:4] == b"RIFF"
    # The voice has a row of its own in the hardware panel, on the chip it is made on, with how fast it went.
    rows = {entry["stage"]: entry for entry in web.get("/api/telemetry").json()["active"] if entry["demo_id"] == "video-commentary"}
    assert rows["voice"]["device"] == "CPU" and rows["voice"]["stage_label"] == "Speaks"
    # Switched off while it plays: the row goes, and the choice is kept for the next Start.
    assert web.post("/api/video-commentary/voice", json={"voice": ""}).json() == {"voice": ""}
    assert "voice" not in {entry["stage"] for entry in web.get("/api/telemetry").json()["active"] if entry["demo_id"] == "video-commentary"}
    refused = web.post("/api/video-commentary/voice", json={"voice": "robot"})
    assert refused.status_code == 400 and "Unknown voice" in refused.text
    web.post("/api/video-commentary/stop")
    assert web.get("/api/video-commentary/speech/1").status_code == 404  # stopped: nothing is kept


def test_the_cloned_voice_is_the_one_enrolled_in_the_voice_clone_studio_or_it_is_refused(running, monkeypatch):
    web, _run, said = running
    told = web.get("/api/video-commentary/devices").json()["voices"]
    assert [(voice["key"], voice["ready"]) for voice in told] == [("studio", True), ("cloned", False)]
    refused = web.post("/api/video-commentary/start", json={"source": "webcam", "voice": "cloned"})
    assert refused.status_code == 400 and "Voice Clone Studio" in refused.json()["error"]
    assert launcher_app.video_commentary_runner.running is False  # refused before anything was started
    lent = Enrolled(device="NPU")
    monkeypatch.setattr(launcher_app.video_commentary_runner, "_cloned", lent)
    told = web.get("/api/video-commentary/devices").json()["voices"]
    assert [(voice["key"], voice["ready"], voice["device"]) for voice in told] == [("studio", True, "CPU"), ("cloned", True, "NPU")]
    assert web.post("/api/video-commentary/start", json={"source": "webcam", "voice": "cloned"}).status_code == 200
    assert said.wait(3) and lent.said == ["What a herd!"]
    assert web.get("/api/video-commentary/speech/1").content[:4] == b"RIFF"
    rows = {entry["stage"]: entry for entry in web.get("/api/telemetry").json()["active"] if entry["demo_id"] == "video-commentary"}
    assert rows["voice"]["device"] == "NPU"  # the chip the enrolled voice is made on
    web.post("/api/video-commentary/stop")
    # Started again without a word about the voice, after the enrolment has gone: silent, not refused.
    lent.enrolled = False
    said.clear()
    assert web.post("/api/video-commentary/start", json={"source": "webcam"}).status_code == 200
    assert said.wait(3) and web.get("/api/video-commentary/comments").json()["voice"] == ""


def test_the_voice_clone_studio_lends_its_voice_without_claiming_the_work(monkeypatch):
    from launcher import activity
    from launcher.errors import Conflict
    from launcher.voice_clone_studio_runner import VoiceCloneStudioRunner

    runner = VoiceCloneStudioRunner()
    with pytest.raises(Conflict, match="Enroll a voice first"):
        runner.speak("Hello.")
    assert runner.device is None

    class Session:
        def synthesize(self, text):
            return [0.0, 0.0], 24000

    runner._session, runner._enrolled, runner._engine, runner._device = Session(), True, "portable", "NPU"
    assert runner.speak("Hello.") == ([0.0, 0.0], 24000)
    assert runner.device == "CPU"  # Chatterbox is made on the CPU, whatever was asked
    runner._engine = "openvino"
    assert runner.device == "NPU"
    assert not [entry for entry in activity.snapshot() if entry["demo_id"] == "voice-clone-studio"]  # the borrower shows it
