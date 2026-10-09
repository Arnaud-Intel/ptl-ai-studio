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
from launcher.autodemo import Scene, Skip, Stand
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
    assert len({video.key for video in sample_videos.VIDEOS}) == len(sample_videos.VIDEOS) == 4
    for video in sample_videos.VIDEOS:
        assert video.urls and all(url.startswith("https://") for url in video.urls)
        assert re.fullmatch(r"[0-9a-f]{64}", video.sha256) and video.size_bytes > 1_000_000
        assert video.licence.startswith("CC") and video.credit and video.source_page.startswith("https://")
        # Two of the licences ask for the credit: it travels with the files.
        assert video.filename in readme and video.credit.split(",")[0] in readme
    assert sample_videos.for_path(sample_videos.TORONTO.path) is sample_videos.TORONTO
    assert sample_videos.for_path("C:/somewhere/else.mp4") is None and sample_videos.for_path("https://x.example/a.mp4") is None


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
    assert sources.display_name(str(sample_videos.SHIBUYA.path)) == "Shibuya Crossing, Tokyo"
    assert sources.display_name(r"C:\videos\crossing.mp4") == "crossing.mp4"


def test_the_page_is_told_which_videos_are_still_to_be_fetched(monkeypatch):
    monkeypatch.setattr(sample_videos, "present", lambda video: video.key == "toronto-crossing")
    listed = TestClient(launcher_app.app).get("/api/smart-city-monitor/devices").json()["samples"]
    by_name = {sample["name"]: sample for sample in listed}
    assert by_name["Yonge-Dundas crossing, Toronto"]["ready"] is True
    assert by_name["Shibuya Crossing, Tokyo"]["ready"] is False and by_name["Two streets, two chips"]["ready"] is False
    assert by_name["Respubliki-Ordzhonikidze crossing, Tyumen"]["ready"] is False
    assert "ready" not in by_name["Westminster Bridge, London"]  # a camera on the network has nothing to fetch


def test_starting_a_feed_on_a_video_not_fetched_yet_fetches_it_first(monkeypatch):
    order = []
    monkeypatch.setattr(sample_videos, "present", lambda video: False)
    monkeypatch.setattr(sample_videos, "download", lambda video, *args, **kwargs: order.append(f"fetch {video.key}"))
    monkeypatch.setattr(launcher_app.smart_city_monitor_runner, "start", lambda **kwargs: order.append("start"))
    web = TestClient(launcher_app.app)
    # The portable engine: the one a machine without OpenVINO has, as where these tests run.
    body = {"feeds": [{"path": str(sample_videos.TORONTO.path), "engine": "portable"},
                      {"path": r"C:\videos\mine.mp4", "engine": "portable"}]}
    assert web.post("/api/smart-city-monitor/start", json=body).status_code == 200
    assert order == ["fetch toronto-crossing", "start"]  # the visitor's own file is none of its business
    # A video that cannot be fetched is said, and nothing is started on it.
    order.clear()

    def unreachable(video, *args, **kwargs):
        raise VideoUnavailable(f"{video.name}: upload.wikimedia.org could not be reached.")

    monkeypatch.setattr(sample_videos, "download", unreachable)
    refused = web.post("/api/smart-city-monitor/start", json=body)
    assert refused.status_code >= 400 and "could not be reached" in refused.text and order == []


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


def test_without_its_videos_the_auto_demo_falls_back_on_the_live_cameras_or_says_what_it_needs():
    online = autodemo_scenes.smart_city(_stand(_listed(ready=False), internet=True), 1)
    assert [feed["path"] for feed in online.steps[0].body["feeds"]] == ["https://clips.example/1.mp4", "https://clips.example/2.mp4"]
    assert "London" in online.beats[0].text
    offline = autodemo_scenes.smart_city(_stand(_listed(ready=False)), 1)
    assert isinstance(offline, Skip) and "Prepare models" in offline.reason and "internet" in offline.reason
