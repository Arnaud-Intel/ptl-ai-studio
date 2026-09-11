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


def test_a_bot_check_is_explained_as_a_block_not_a_stale_parser():
    """Seen on 2026-09-11 on every camera at once. Telling the operator to
    upgrade yt-dlp -- already the newest -- sends them the wrong way."""
    msg = sources._explain_youtube_failure(
        "https://youtu.be/x", "ERROR: [youtube] x: Sign in to confirm you’re not a bot. Use --cookies"
    )
    assert "upgrade-package" not in msg
    assert "local video" in msg and "RTSP" in msg


def test_any_other_failure_still_suggests_upgrading():
    msg = sources._explain_youtube_failure("https://youtu.be/x", "Unable to extract initial player response")
    assert "upgrade-package yt-dlp" in msg


def test_terminal_colour_codes_never_reach_the_ui():
    msg = sources._explain_youtube_failure("https://youtu.be/x", "\x1b[0;31mERROR:\x1b[0m something broke")
    assert "\x1b" not in msg and "something broke" in msg


def test_every_curated_camera_sits_in_the_right_section():
    """The picker splits YouTube from everything else so that, when YouTube
    blocks the network, it is obvious which cameras still work. A camera
    filed in the wrong section would lie about exactly that."""
    assert {f.group for f in samples.LIVE_FEEDS} <= {"YouTube", "Other"}
    for feed in samples.LIVE_FEEDS:
        assert sources.is_youtube(feed.url) == (feed.group == "YouTube"), feed.name



def test_a_clip_url_is_told_apart_from_a_stream():
    assert sources.is_clip_url("https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.06592.mp4")
    assert not sources.is_clip_url("https://wzmedia.dot.ca.gov/D7/CCTV-196.stream/playlist.m3u8")
    assert not sources.is_clip_url("https://www.youtube.com/watch?v=x")
    assert not sources.is_clip_url("rtsp://cam.local/stream.mp4")
    assert not sources.is_clip_url(r"C:\videos\clip.mp4")


def test_a_clip_url_goes_to_the_refreshing_reader(monkeypatch):
    seen = {}

    def fake(url, stop_event=None):
        seen["url"] = url
        yield "frame"

    monkeypatch.setattr(sources.video, "stream_refreshing_clip", fake)
    assert list(sources.open_frames("https://cams.example/one.mp4")) == ["frame"]
    assert seen["url"] == "https://cams.example/one.mp4"


def test_a_refreshing_clip_plays_each_revision_exactly_once():
    """Measured on TfL JamCams: one clip stays up ~5-7 minutes. Replaying it
    until the next one lands would count the same cars ~30 times."""
    import threading

    from pantherlake_ai_core import video

    stop = threading.Event()
    versions = iter(["v1", "v1", "v1", "v2", "v2"])

    def version(url):
        try:
            return next(versions)
        except StopIteration:
            stop.set()
            return "v2"

    plays = []

    def play(url, stop_event):
        plays.append(url)
        yield "frame"

    list(video.stream_refreshing_clip("u", stop_event=stop, poll_seconds=0, _version=version, _play=play))
    assert len(plays) == 2  # v1 once, v2 once -- v1 never again


def test_a_server_that_wont_say_is_not_replayed_on_a_hunch(monkeypatch):
    from pantherlake_ai_core import video

    waits = []

    def fake_sleep(stop_event, seconds):
        waits.append(seconds)
        return True  # asked to stop during the wait

    monkeypatch.setattr(video, "sleep_unless_stopped", fake_sleep)
    plays = []

    def play(url, stop_event):
        plays.append(url)
        yield "frame"

    list(video.stream_refreshing_clip("u", _version=lambda url: None, _play=play))
    assert plays == ["u"]
    assert waits == [video._CLIP_FALLBACK_WAIT]


def test_a_later_failed_play_is_a_blip_not_a_dead_feed():
    import threading

    from pantherlake_ai_core import video

    stop = threading.Event()
    versions = iter(["v1", "v2", "v3"])
    calls = []

    def version(url):
        try:
            return next(versions)
        except StopIteration:
            stop.set()
            return "v3"

    def play(url, stop_event):
        calls.append(len(calls))
        if len(calls) == 2:
            raise RuntimeError("Could not open clip: u")
        yield "frame"

    list(video.stream_refreshing_clip("u", stop_event=stop, poll_seconds=0, _version=version, _play=play))
    assert len(calls) == 3  # played, blipped, played the next revision
