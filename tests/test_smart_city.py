"""smart-city-monitor's counting logic: the IoU tracker, the trailing-60s
counters, and the CLI's `path|DEVICE` source syntax. No video, no model."""
from __future__ import annotations

import pytest
from object_detection.types import Detection

from pantherlake_ai_core.engine import Engine

from smart_city_monitor.cli import _parse_source
from smart_city_monitor.pipeline import FeedCounters, _group_by_detector
from smart_city_monitor.tracker import Tracker
from smart_city_monitor.types import FeedSpec, TrackedDetection


def det(label: str, box: tuple[int, int, int, int], confidence: float = 0.9) -> Detection:
    return Detection(label=label, confidence=confidence, box=box)


def test_a_stationary_object_keeps_one_id_and_is_new_only_once():
    tracker = Tracker()
    (first,) = tracker.update([det("car", (10, 10, 110, 110))], now=0.0)
    (second,) = tracker.update([det("car", (12, 11, 112, 111))], now=0.1)
    assert first.is_new and not second.is_new
    assert first.track_id == second.track_id


def test_tracks_never_match_across_labels():
    tracker = Tracker()
    tracker.update([det("car", (0, 0, 100, 100))], now=0.0)
    (person,) = tracker.update([det("person", (0, 0, 100, 100))], now=0.1)
    assert person.is_new


def test_two_objects_get_two_ids():
    tracker = Tracker()
    tracks = tracker.update([det("car", (0, 0, 100, 100)), det("car", (500, 500, 600, 600))], now=0.0)
    assert len({t.track_id for t in tracks}) == 2


def test_a_track_unseen_for_over_a_second_expires_instead_of_being_reused():
    tracker = Tracker()
    (before,) = tracker.update([det("car", (0, 0, 100, 100))], now=0.0)
    (after,) = tracker.update([det("car", (0, 0, 100, 100))], now=1.5)
    assert after.is_new and after.track_id != before.track_id


def test_counters_count_a_track_once_and_age_out_of_the_trailing_minute():
    counters = FeedCounters()
    car = TrackedDetection(1, "car", 0.9, (0, 0, 1, 1), is_new=True)
    counters.record([car], now=0.0)
    counters.record([TrackedDetection(1, "car", 0.9, (0, 0, 1, 1), is_new=False)], now=1.0)
    counters.record([TrackedDetection(2, "person", 0.9, (0, 0, 1, 1), is_new=True)], now=30.0)
    assert counters.snapshot(now=31.0) == ({"Cars": 1, "Pedestrians": 1}, {"Cars": 1, "Pedestrians": 1})
    last_60s, total = counters.snapshot(now=70.0)
    assert last_60s == {"Cars": 0, "Pedestrians": 1}
    assert total == {"Cars": 1, "Pedestrians": 1}


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("clip.mp4", ("clip.mp4", "AUTO")),
        ("clip.mp4|NPU", ("clip.mp4", "NPU")),
        (r"C:\videos\a.mp4|GPU.0", (r"C:\videos\a.mp4", "GPU.0")),  # a drive-letter colon is not a separator
        ("clip.mp4|", ("clip.mp4", "AUTO")),
    ],
)
def test_parse_source(raw, expected):
    assert _parse_source(raw, "AUTO") == expected


# --- one detector per (engine, device, model) --------------------------------


def feed(feed_id: str, device: str, **kwargs) -> FeedSpec:
    return FeedSpec(feed_id=feed_id, path=f"{feed_id}.mp4", compute_device=device, **kwargs)


def test_feeds_wanting_the_same_thing_share_one_detector():
    """Loading a model is the expensive part, so feeds that ask for the
    same engine, chip and model must not each load their own."""
    feeds = [feed("f1", "NPU"), feed("f2", "NPU"), feed("f3", "GPU")]
    groups = _group_by_detector(feeds, Engine.OPENVINO)
    assert len(groups) == 2
    assert [f.feed_id for f in groups[(Engine.OPENVINO, "NPU", None)]] == ["f1", "f2"]
    assert [f.feed_id for f in groups[(Engine.OPENVINO, "GPU", None)]] == ["f3"]


def test_a_feed_with_its_own_engine_gets_its_own_detector():
    """Two feeds on one chip running different backends is the whole point
    of per-feed engines -- they must not be collapsed together."""
    feeds = [feed("f1", "CPU"), feed("f2", "CPU", engine=Engine.PORTABLE)]
    groups = _group_by_detector(feeds, Engine.OPENVINO)
    assert set(groups) == {(Engine.OPENVINO, "CPU", None), (Engine.PORTABLE, "CPU", None)}


def test_the_model_is_part_of_what_makes_a_detector_distinct():
    feeds = [feed("f1", "NPU"), feed("f2", "NPU", model_path="/models/other.xml")]
    groups = _group_by_detector(feeds, Engine.OPENVINO)
    assert set(groups) == {(Engine.OPENVINO, "NPU", None), (Engine.OPENVINO, "NPU", "/models/other.xml")}


def test_a_feed_without_an_engine_falls_back_to_the_run():
    (key,) = _group_by_detector([feed("f1", "CPU")], Engine.PORTABLE)
    assert key == (Engine.PORTABLE, "CPU", None)
