"""Telling a feed's three kinds of source apart, and naming them.

No network and no yt-dlp: what's worth testing here is the dispatch and
the labelling, not YouTube's page format.
"""
from __future__ import annotations

import pytest
from smart_city_monitor import samples, sources


@pytest.mark.parametrize(
    "source",
    [
        "https://www.youtube.com/watch?v=DjdUEyjx8GM",
        "http://cam.example/mjpeg",
        "rtsp://192.168.1.50:554/stream1",
        "rtmp://example.test/live",
    ],
)
def test_a_stream_url_is_recognised(source):
    assert sources.is_url(source)


@pytest.mark.parametrize(
    "source",
    [
        r"C:\videos\crossing.mp4",  # a drive letter is not a URL scheme
        "/home/me/clip.mp4",
        "clip.mp4",
        "",
    ],
)
def test_a_file_path_is_not_a_url(source):
    assert not sources.is_url(source)


def test_only_youtube_hosts_need_resolving():
    assert sources.is_youtube("https://youtu.be/DjdUEyjx8GM")
    assert sources.is_youtube("https://m.youtube.com/watch?v=DjdUEyjx8GM")
    assert not sources.is_youtube("https://notyoutube.com/watch?v=x")
    assert not sources.is_youtube("rtsp://cam.local/1")
    assert not sources.is_youtube("clip.mp4")


def test_a_curated_camera_keeps_its_real_name():
    """Every YouTube feed reduces to "www.youtube.com" -- and running two
    at once, each on its own chip, is the whole point of this brick, so
    the per-feed counts have to be told apart."""
    names = [sources.display_name(f.url) for f in samples.LIVE_FEEDS]
    assert names == [f.name for f in samples.LIVE_FEEDS]
    assert len(set(names)) == len(names), "two feeds would be indistinguishable"


def test_anything_else_is_named_by_host_or_filename():
    assert sources.display_name("rtsp://cam.local:554/stream1") == "cam.local"
    assert sources.display_name("https://other.example/watch?v=abc") == "other.example"
    assert sources.display_name(r"C:\videos\crossing.mp4") == "crossing.mp4"


def test_every_sample_lists_sources_the_pipeline_can_open():
    assert samples.SAMPLES
    for sample in samples.SAMPLES:
        lines = [line for line in sample.feeds.splitlines() if line.strip()]
        assert lines, f"{sample.name} has no feed"
        assert all(sources.is_url(line) for line in lines)


def test_the_biggest_stream_within_the_cap_wins(monkeypatch):
    """Not the first one under it: the format list is not sorted, and a
    360p stream when 720p is available is a worse demo for free."""
    formats = [
        {"protocol": "m3u8_native", "vcodec": "avc1", "url": "u360", "height": 360},
        {"protocol": "m3u8_native", "vcodec": "avc1", "url": "u1080", "height": 1080},
        {"protocol": "m3u8_native", "vcodec": "avc1", "url": "u720", "height": 720},
        {"protocol": "https", "vcodec": "avc1", "url": "not-hls", "height": 720},
        {"protocol": "m3u8_native", "vcodec": "none", "url": "audio-only", "height": None},
    ]
    monkeypatch.setattr(sources, "_extract_formats", lambda url: formats)
    assert sources.resolve_youtube_stream("https://youtu.be/x") == "u720"


def test_a_stream_above_the_cap_is_better_than_none(monkeypatch):
    formats = [{"protocol": "m3u8_native", "vcodec": "avc1", "url": "u1080", "height": 1080}]
    monkeypatch.setattr(sources, "_extract_formats", lambda url: formats)
    assert sources.resolve_youtube_stream("https://youtu.be/x") == "u1080"


def test_a_page_with_no_video_stream_says_so(monkeypatch):
    monkeypatch.setattr(sources, "_extract_formats", lambda url: [])
    with pytest.raises(RuntimeError, match="actually live"):
        sources.resolve_youtube_stream("https://youtu.be/x")


def test_a_reconnect_waits_but_a_stop_returns_at_once():
    """The backoff must not depend on there being a stop_event: without
    one, a dead URL would otherwise be retried in a tight loop."""
    import threading
    import time

    from pantherlake_ai_core import video

    start = time.monotonic()
    assert video.sleep_unless_stopped(None, 0.05) is False
    # a tolerance, not sloppiness: Windows' sleep can return a hair early
    assert time.monotonic() - start >= 0.04

    stopped = threading.Event()
    stopped.set()
    start = time.monotonic()
    assert video.sleep_unless_stopped(stopped, 5.0) is True
    assert time.monotonic() - start < 1.0  # returned immediately, didn't wait 5s
