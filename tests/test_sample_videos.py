"""The street videos the city monitor plays from disk.

They are not in the repository: each is fetched from where its author
published it, checked byte for byte, and then offered first in the demo --
which opens on them, and which the Auto Demo plays without a network. No
video is fetched here: the source is a stand-in."""
from __future__ import annotations

import hashlib
import re
import urllib.error
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from launcher import app as launcher_app
from launcher import autodemo_scenes
from launcher.autodemo import Director, Scene, Skip, Stand
from pantherlake_ai_core import models, sample_videos
from pantherlake_ai_core.sample_videos import SampleVideo, VideoUnavailable
from smart_city_monitor import samples, sources

FOOTAGE = bytes(range(256)) * 40  # 10,240 bytes standing in for a video


class Source:
    """Stands in for the sites the videos come from: what it serves at each
    address, what was asked of it, and whether it resumes."""

    def __init__(self, resumes=True):
        self.files: dict[str, bytes] = {}
        self.asked: list[tuple[str, int]] = []
        self.resumes = resumes
        self.down: set[str] = set()

    def __call__(self, url, start):
        self.asked.append((url, start))
        if url in self.down or url not in self.files:
            raise urllib.error.URLError("no route to host")
        body = self.files[url]
        source = self

        class Response:
            status = 206 if (start and source.resumes) else 200

            def __init__(self):
                self.left = body[start:] if (start and source.resumes) else body

            def read(self, size):
                piece, self.left = self.left[:size], self.left[size:]
                return piece

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Response()


@pytest.fixture
def there(tmp_path, monkeypatch):
    """A machine with an empty video folder and a source that has the clip."""
    monkeypatch.setattr(sample_videos, "VIDEO_DIR", tmp_path / "videos")
    source = Source()
    monkeypatch.setattr(sample_videos, "_open", source)
    video = SampleVideo(
        key="clip", name="A street", description="", filename="street.webm",
        urls=("https://first.example/street.webm", "https://second.example/street.webm"),
        size_bytes=len(FOOTAGE), sha256=hashlib.sha256(FOOTAGE).hexdigest(),
        licence="CC0 1.0", credit="Somebody", source_page="https://first.example/page",
    )
    monkeypatch.setattr(SampleVideo, "path", property(lambda self: sample_videos.VIDEO_DIR / self.filename))
    source.files[video.urls[0]] = FOOTAGE
    return video, source


def test_a_video_is_fetched_once_checked_and_put_in_place(there):
    video, source = there
    assert not sample_videos.present(video)
    seen = []
    assert sample_videos.download(video, seen.append) == video.path
    assert video.path.read_bytes() == FOOTAGE and sample_videos.present(video)
    assert seen[-1] == len(FOOTAGE) and not video.path.with_name("street.webm.part").exists()
    sample_videos.download(video)
    assert len(source.asked) == 1  # already there: nothing is asked of anybody


def test_what_is_not_the_chosen_file_is_refused_and_nothing_of_it_is_kept(there):
    video, source = there
    source.files[video.urls[0]] = FOOTAGE[:-1] + b"!"  # the right size, another video
    with pytest.raises(VideoUnavailable, match="changed at the source"):
        sample_videos.download(video)
    assert not video.path.exists() and not list(sample_videos.VIDEO_DIR.iterdir())


def test_a_download_cut_short_resumes_and_a_source_that_cannot_resume_starts_again(there):
    video, source = there
    part = video.path.with_name("street.webm.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(FOOTAGE[:4000])
    sample_videos.download(video)
    assert source.asked == [(video.urls[0], 4000)] and video.path.read_bytes() == FOOTAGE
    # The same, from a server that answers with the whole file whatever is asked.
    video.path.unlink()
    part.write_bytes(FOOTAGE[:4000])
    source.resumes = False
    sample_videos.download(video)
    assert video.path.read_bytes() == FOOTAGE


def test_the_next_address_is_tried_and_all_failing_says_why(there):
    video, source = there
    source.down.add(video.urls[0])
    source.files[video.urls[1]] = FOOTAGE
    sample_videos.download(video)
    assert [url for url, _start in source.asked] == list(video.urls) and sample_videos.present(video)
    video.path.unlink()
    source.down.add(video.urls[1])
    with pytest.raises(VideoUnavailable, match="first.example could not be reached.*second.example could not be reached"):
        sample_videos.download(video)


def test_a_stop_keeps_what_was_fetched_for_next_time(there):
    video, _source = there
    with pytest.raises(VideoUnavailable, match="stopped"):
        sample_videos.download(video, stopped=lambda: True)
    assert not video.path.exists()


# --- the three that were chosen ---------------------------------------------------------


def test_every_video_says_where_it_comes_from_and_under_which_licence():
    readme = (Path(sample_videos.SAMPLE_ROOT) / "videos" / "README.md").read_text(encoding="utf-8")
    assert len({video.key for video in sample_videos.VIDEOS}) == len(sample_videos.VIDEOS) == 6
    assert [video.counts for video in sample_videos.VIDEOS] == ["street"] * 4 + ["line", "herd"]
    for video in sample_videos.VIDEOS:
        assert video.urls and all(url.startswith("https://") for url in video.urls)
        assert re.fullmatch(r"[0-9a-f]{64}", video.sha256) and video.size_bytes > 1_000_000
        assert video.licence.startswith(("CC", "Public domain")) and video.credit and video.source_page.startswith("https://")
        # Two of the licences ask for the credit: it travels with the files.
        assert video.filename in readme and video.credit.split(",")[0] in readme
    assert sample_videos.for_path(sample_videos.TORONTO.path) is sample_videos.TORONTO
    assert sample_videos.for_path("C:/somewhere/else.mp4") is None and sample_videos.for_path("https://x.example/a.mp4") is None


def test_a_street_video_is_named_by_its_city_and_nothing_more():
    """On screen and on disk: a street, a square or a shop named on a stand
    is somebody's interest, and a city is not."""
    streets = [video for video in sample_videos.VIDEOS if video.counts == "street"][:3]
    assert [(video.name, video.filename, video.key) for video in streets] == [
        ("Toronto", "toronto.webm", "toronto"), ("Tyumen", "tyumen.webm", "tyumen"), ("Tokyo", "tokyo.webm", "tokyo")]
    readme = (Path(sample_videos.SAMPLE_ROOT) / "videos" / "README.md").read_text(encoding="utf-8")
    shown = " ".join([readme.split("## Licences and credits")[0]]
                     + [f"{video.name} {video.description}" for video in sample_videos.VIDEOS]
                     + [f"{sample.name} {sample.description}" for sample in samples.SAMPLES if sample.kind == "file"])
    for place in ("Yonge", "Dundas", "Shibuya", "Respubliki", "Ordzhonikidze"):
        assert place not in shown, f"{place} is named where a visitor reads it"


def test_the_videos_are_fetched_with_the_models_and_weigh_what_they_weigh(monkeypatch):
    spec = models.BY_KEY["street-videos"]
    assert spec.demos == ("smart-city-monitor",) and models.remote_size(spec) == sample_videos.total_bytes()
    state = {"present": False, "bytes": 1234, "fetched": 0}
    monkeypatch.setattr(sample_videos, "download_all", lambda on_progress=None: state.__setitem__("fetched", state["fetched"] + 1))
    monkeypatch.setattr(sample_videos, "all_present", lambda: state["present"])
    monkeypatch.setattr(sample_videos, "bytes_on_disk", lambda: state["bytes"])
    spec = models._street_videos_spec()  # as built, with the stand-ins above
    assert models.is_cached(spec) is False and models.cached_bytes(spec) == 1234
    models.download(spec)
    assert state["fetched"] == 1
    state["present"] = True
    assert models.is_cached(spec) is True


# --- in the demo ------------------------------------------------------------------------


def test_the_city_monitor_offers_its_own_videos_first_and_opens_on_two_of_them():
    usual = samples.SAMPLES[0]
    assert usual.default and usual.kind == "file" and usual.group == samples.LOCAL
    # The two the detector does best on, by measurement (the table in sample_videos): and they come first.
    assert usual.feeds == f"{sample_videos.TORONTO.path}|NPU\n{sample_videos.TYUMEN.path}|GPU"
    assert sample_videos.VIDEOS[:2] == (sample_videos.TORONTO, sample_videos.TYUMEN)
    assert [sample.default for sample in samples.SAMPLES].count(True) == 1
    alone = [sample for sample in samples.SAMPLES if sample.kind == "file" and len(sample.videos) == 1]
    assert [sample.videos[0] for sample in alone] == [video.key for video in sample_videos.VIDEOS]
    # Whoever is credited is credited where the sample is chosen.
    assert "RG72" in alone[1].description and "CC BY-SA 4.0" in alone[1].description
    assert "Basile Morin" in alone[2].description
    # A feed playing one of them is called by its name, not by a file name.
    assert sources.display_name(str(sample_videos.TOKYO.path)) == "Tokyo"
    assert sources.display_name(r"C:\videos\crossing.mp4") == "crossing.mp4"


def test_the_page_is_told_which_videos_are_still_to_be_fetched(monkeypatch):
    monkeypatch.setattr(sample_videos, "present", lambda video: video.key == "toronto")
    listed = TestClient(launcher_app.app).get("/api/smart-city-monitor/devices").json()["samples"]
    by_name = {sample["name"]: sample for sample in listed}
    assert by_name["Toronto"]["ready"] is True
    assert by_name["Tokyo"]["ready"] is False and by_name["Two streets, two chips"]["ready"] is False
    assert by_name["Tyumen"]["ready"] is False
    assert "ready" not in by_name["Westminster Bridge, London"]  # a camera on the network has nothing to fetch


def test_starting_a_feed_on_a_video_not_fetched_yet_fetches_it_first(monkeypatch):
    order = []
    monkeypatch.setattr(sample_videos, "present", lambda video: False)
    monkeypatch.setattr(sample_videos, "download", lambda video, *args, **kwargs: order.append(f"fetch {video.key}"))
    started = {}
    monkeypatch.setattr(launcher_app.smart_city_monitor_runner, "start",
                        lambda **kwargs: (order.append("start"), started.update(kwargs)))
    web = TestClient(launcher_app.app)
    # The portable engine: the one a machine without OpenVINO has, as where these tests run.
    body = {"feeds": [{"path": str(sample_videos.TORONTO.path), "engine": "portable"},
                      {"path": r"C:\videos\mine.mp4", "engine": "portable"}]}
    assert web.post("/api/smart-city-monitor/start", json=body).status_code == 200
    assert order == ["fetch toronto", "start"]  # the visitor's own file is none of its business
    assert [feed.counting for feed in started["feeds"]] == ["street", "street"]
    # A video that cannot be fetched is said, and nothing is started on it.
    order.clear()

    def unreachable(video, *args, **kwargs):
        raise VideoUnavailable(f"{video.name}: upload.wikimedia.org could not be reached.")

    monkeypatch.setattr(sample_videos, "download", unreachable)
    refused = web.post("/api/smart-city-monitor/start", json=body)
    assert refused.status_code >= 400 and "could not be reached" in refused.text and order == []


def test_a_feed_counts_what_it_is_watched_for_and_the_same_person_under_another_name():
    from smart_city_monitor.pipeline import FeedCounters
    from smart_city_monitor.types import COUNTING, FeedSpec, TrackedDetection, labels_counted

    def new(label, number):
        return TrackedDetection(track_id=number, label=label, confidence=0.9, box=(0, 0, 10, 10), is_new=True)

    assert FeedSpec(feed_id="feed-1", path="x.mp4", compute_device="CPU").counting == "street"  # unless told otherwise
    line, herd = FeedCounters(labels_counted("line")), FeedCounters(labels_counted("herd"))
    line.record([new("bottle", 1), new("bottle", 2), new("person", 3)], now=1.0)
    herd.record([new("cow", 1), new("dog", 2), new("person", 3)], now=1.0)
    assert line.snapshot(2.0)[1] == {"Bottles": 2, "Workers": 1}
    assert herd.snapshot(2.0)[1] == {"Cattle": 1, "Dogs": 1, "People": 1}
    assert COUNTING["street"]["person"] == "Pedestrians" and "bottle" not in COUNTING["street"] and "car" not in COUNTING["herd"]
    with pytest.raises(ValueError, match="one of street, line, herd"):
        labels_counted("defects")  # the detector names things; it does not judge them


def test_the_second_pair_is_a_herd_and_a_line_each_counted_for_what_it_shows(monkeypatch):
    pair = next(sample for sample in samples.SAMPLES if sample.name == "A herd and a line, two chips")
    assert pair.feeds == f"{sample_videos.CATTLE_DRIVE.path}|GPU\n{sample_videos.CAPPING_LINE.path}|NPU"
    assert pair.counting == ("herd", "line") and not pair.default and pair.group == samples.LOCAL
    alone = {sample.videos[0]: sample for sample in samples.SAMPLES if sample.kind == "file" and len(sample.videos) == 1}
    assert alone["cattle-drive"].counting == ("herd",) and alone["bottle-capping-line"].counting == ("line",)
    assert alone["toronto"].counting == ("street",)
    # Started with nothing said, a sample video is counted for what it shows; told, a feed counts what it is told to.
    started = {}
    monkeypatch.setattr(sample_videos, "present", lambda video: True)
    monkeypatch.setattr(launcher_app.smart_city_monitor_runner, "start", lambda **kwargs: started.update(kwargs))
    web = TestClient(launcher_app.app)
    feeds = [{"path": str(sample_videos.CATTLE_DRIVE.path), "engine": "portable"},
             {"path": str(sample_videos.CAPPING_LINE.path), "engine": "portable"},
             {"path": "C:/videos/my-line.mp4", "engine": "portable", "counting": "line"}]
    answer = web.post("/api/smart-city-monitor/start", json={"feeds": feeds})
    assert answer.status_code == 200 and [feed.counting for feed in started["feeds"]] == ["herd", "line", "line"]
    assert [feed["counting"] for feed in answer.json()["feeds"]] == ["herd", "line", "line"]  # for the page to rebuild its cards
    refused = web.post("/api/smart-city-monitor/start", json={"feeds": [{"path": "C:/v/x.mp4", "engine": "portable", "counting": "defects"}]})
    assert refused.status_code == 400 and "feed 1" in refused.text and "defects" in refused.text


# --- in the Auto Demo -------------------------------------------------------------------


def _stand(samples_listed, **has) -> Stand:
    return Stand(npu=has.get("npu", True), igpu="GPU.0", internet=has.get("internet", False), lang=has.get("lang", "en"),
                 samples=lambda demo: samples_listed)


def _listed(ready=True):
    return [
        {"name": "Two streets, two chips", "kind": "file", "ready": ready, "group": "On this machine", "feeds": "C:/v/a.webm|GPU\nC:/v/b.webm|NPU"},
        {"name": "Toronto", "kind": "file", "ready": ready, "group": "On this machine", "feeds": "C:/v/a.webm"},
        {"name": "Tokyo", "kind": "file", "ready": ready, "group": "On this machine", "feeds": "C:/v/b.webm"},
        {"name": "Small clip", "kind": "file", "ready": ready, "group": "On this machine", "feeds": "C:/v/c.mp4"},
        {"name": "Shibuya live", "kind": "url", "group": "YouTube", "feeds": "https://video.example/live"},
        {"name": "Tower Bridge", "kind": "url", "group": "Other", "feeds": "https://clips.example/1.mp4"},
        {"name": "Westminster Bridge", "kind": "url", "group": "Other", "feeds": "https://clips.example/2.mp4"},
    ]


def test_the_auto_demo_plays_the_videos_on_disk_with_no_network_and_swaps_their_chips_each_turn():
    scene = autodemo_scenes.smart_city(_stand(_listed()), 1)
    assert isinstance(scene, Scene)  # no internet, and it plays
    feeds = scene.steps[0].body["feeds"]
    assert feeds == [{"path": "C:/v/a.webm", "compute_device": "GPU.0"}, {"path": "C:/v/b.webm", "compute_device": "NPU"}]
    assert [feed["name"] for feed in scene.props["feeds"]] == ["Toronto", "Tokyo"]
    assert "from this laptop's disk" in scene.beats[0].text and "London" not in scene.beats[0].text
    assert "disque" in autodemo_scenes.smart_city(_stand(_listed(), lang="fr"), 1).beats[0].text
    again = autodemo_scenes.smart_city(_stand(_listed()), 2)
    assert [feed["path"] for feed in again.steps[0].body["feeds"]] == ["C:/v/b.webm", "C:/v/a.webm"]  # the other chip each


def test_the_auto_demo_counts_a_herd_and_a_line_with_the_same_detector_when_their_videos_are_there():
    listed = _listed() + [
        {"name": "Bottle capping line", "kind": "file", "ready": True, "counting": ["line"], "feeds": "C:/v/bottles.webm"},
        {"name": "Cattle on the road", "kind": "file", "ready": True, "counting": ["herd"], "feeds": "C:/v/cattle.webm"},
    ]
    scene = autodemo_scenes.herd_and_line(_stand(listed), 1)
    assert isinstance(scene, Scene) and scene.demo == "smart-city-monitor" and scene.view == "cameras"
    assert scene.steps[0].body["feeds"] == [
        {"path": "C:/v/cattle.webm", "compute_device": "GPU.0", "counting": "herd"},  # the 30-frame clip on the GPU
        {"path": "C:/v/bottles.webm", "compute_device": "NPU", "counting": "line"}]
    assert [feed["name"] for feed in scene.props["feeds"]] == ["Cattle on the road", "Bottle capping line"]
    assert "never trained" in scene.beats[1].text and "counted twice" in scene.beats[3].text  # what it is, and how far to trust it
    assert scene.stop == ("/api/smart-city-monitor/stop",)
    # The street scene goes on playing streets: a herd is not a second street.
    street = autodemo_scenes.smart_city(_stand(listed[4:] + listed[:4]), 1)
    assert [feed["path"] for feed in street.steps[0].body["feeds"]] == ["C:/v/a.webm", "C:/v/b.webm"]
    missing = autodemo_scenes.herd_and_line(_stand(_listed()), 1)
    assert isinstance(missing, Skip) and "Prepare models" in missing.reason


def test_the_loop_plays_one_counting_scene_a_turn_the_streets_then_the_herd_and_the_line():
    listed = _listed() + [
        {"name": "Bottle capping line", "kind": "file", "ready": True, "counting": ["line"], "feeds": "C:/v/bottles.webm"},
        {"name": "Cattle on the road", "kind": "file", "ready": True, "counting": ["herd"], "feeds": "C:/v/cattle.webm"},
    ]
    # One place in the playlist for the two, not two scenes of the same brick back to back.
    place = next(slot for slot in autodemo_scenes.PLAYLIST if isinstance(slot, tuple))
    assert [entry.key for entry in place] == ["smart-city", "herd-and-line"]
    director = Director(lambda *call: {}, [place])

    def turns(stand, loops=(1, 2, 3, 4, 5), chosen=None):
        return [director._listing(stand, loop, chosen)[0] for loop in loops]

    stand = _stand(listed)
    played = [scenes[0] for scenes in turns(stand)]
    assert [scene.id for scene in played] == ["smart-city", "herd-and-line", "smart-city", "herd-and-line", "smart-city"]
    assert all(len(scenes) == 1 for scenes in turns(stand))  # never both in one turn
    # The streets still change chips each time they play, which is now every other turn.
    streets = [[feed["path"] for feed in scene.steps[0].body["feeds"]] for scene in played[::2]]
    assert streets == [["C:/v/a.webm", "C:/v/b.webm"], ["C:/v/b.webm", "C:/v/a.webm"], ["C:/v/a.webm", "C:/v/b.webm"]]
    # The one whose turn it is cannot play: the other does, rather than the loop going a scene short.
    only_streets = _stand(_listed())
    assert [scenes[0].id for scenes in turns(only_streets, (1, 2))] == ["smart-city", "smart-city"]
    only_herd = _stand([entry for entry in listed if entry.get("counting")])
    assert [scenes[0].id for scenes in turns(only_herd, (1, 2))] == ["herd-and-line", "herd-and-line"]
    # One of the two left out when the loop was started: the other every turn, changing chips every turn.
    chosen = turns(stand, (1, 2), {"smart-city"})
    assert [scenes[0].id for scenes in chosen] == ["smart-city", "smart-city"]
    assert [feed["path"] for feed in chosen[1][0].steps[0].body["feeds"]] == ["C:/v/b.webm", "C:/v/a.webm"]
    # Neither can: nothing plays there, and each says why.
    scenes, listing = director._listing(_stand(_listed(ready=False)), 2)
    assert scenes == [] and all("Prepare models" in entry["reason"] for entry in listing)


def test_without_its_videos_the_auto_demo_falls_back_on_the_live_cameras_or_says_what_it_needs():
    online = autodemo_scenes.smart_city(_stand(_listed(ready=False), internet=True), 1)
    assert [feed["path"] for feed in online.steps[0].body["feeds"]] == ["https://clips.example/1.mp4", "https://clips.example/2.mp4"]
    assert "London" in online.beats[0].text
    offline = autodemo_scenes.smart_city(_stand(_listed(ready=False)), 1)
    assert isinstance(offline, Skip) and "Prepare models" in offline.reason and "internet" in offline.reason
