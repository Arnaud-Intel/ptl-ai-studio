"""Network failures, shared cooldowns and cancellation without contacting YouTube."""
import subprocess
import threading
from types import SimpleNamespace

import pytest

from smart_city_monitor import sources, youtube, pipeline
from smart_city_monitor.types import FeedSpec, TrackedDetection
from pantherlake_ai_core.engine import Engine


def test_blocked_network_stops_other_feeds_and_survives_new_runs(monkeypatch):
    resolver = youtube.Resolver()
    calls = []
    now = [100.0]
    monkeypatch.setattr(youtube.time, 'monotonic', lambda: now[0])
    def extract(url, stop):
        calls.append(url)
        raise youtube.classify("Sign in to confirm you're not a bot")
    monkeypatch.setattr(youtube, '_extract', extract)
    for url in ('https://youtu.be/a', 'https://youtu.be/b', 'https://youtu.be/a'):
        with pytest.raises(youtube.SourceError) as exc:
            resolver.resolve(url)
        assert exc.value.blocked
    assert len(calls) == 1
    now[0] += youtube.BLOCK_COOLDOWN + 1
    with pytest.raises(youtube.SourceError):
        resolver.resolve('https://youtu.be/a')
    assert len(calls) == 2


def test_same_video_uses_one_resolution_and_invalidates_on_drop(monkeypatch):
    resolver = youtube.Resolver()
    calls = []
    monkeypatch.setattr(youtube, 'MIN_INTERVAL', 0)
    monkeypatch.setattr(youtube, '_extract', lambda url, stop: calls.append(url) or [{'url': 'stream'}])
    resolver.resolve('https://youtu.be/abc')
    resolver.resolve('https://www.youtube.com/watch?v=abc')
    assert len(calls) == 1
    resolver.invalidate('https://youtu.be/abc')
    resolver.resolve('https://youtu.be/abc')
    assert len(calls) == 2


def test_extractions_are_paced_across_different_feeds(monkeypatch):
    resolver = youtube.Resolver()
    now, starts = [0.0], []
    monkeypatch.setattr(youtube.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(youtube.time, 'sleep', lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(youtube, '_extract', lambda url, stop: starts.append(now[0]) or [{'url': 'stream'}])
    resolver.resolve('https://youtu.be/a')
    resolver.resolve('https://youtu.be/b')
    assert starts[1] - starts[0] >= youtube.MIN_INTERVAL


@pytest.mark.parametrize('message,blocked,retryable', [
    ('HTTP Error 429: Too Many Requests', True, False),
    ('Video unavailable', False, False),
    ('HTTP Error 403: Forbidden', False, False),
    ('connection timed out', False, True),
    ('HTTP Error 503', False, True),
])
def test_error_classification(message, blocked, retryable):
    error = youtube.classify(message)
    assert (error.blocked, error.retryable) == (blocked, retryable)


def test_private_stream_urls_are_redacted():
    assert 'secret' not in youtube.safe_message('Failed https://user:secret@host/live?token=secret')


def test_stop_while_waiting_for_another_extraction(monkeypatch):
    resolver = youtube.Resolver()
    stop = threading.Event()
    stop.set()
    resolver.lock.acquire()
    try:
        with pytest.raises(youtube.Cancelled):
            resolver.resolve('https://youtu.be/a', stop)
    finally:
        resolver.lock.release()


def test_extractor_child_is_killed_on_cancel(monkeypatch):
    stop = threading.Event()
    class Child:
        killed = False
        returncode = None
        def communicate(self, timeout=None):
            if self.killed:
                return '', ''
            stop.set()
            raise subprocess.TimeoutExpired('extract', timeout)
        def poll(self): return self.returncode
        def kill(self): self.killed = True; self.returncode = -9
    child = Child()
    monkeypatch.setattr(youtube.subprocess, 'Popen', lambda *a, **kw: child)
    with pytest.raises(youtube.Cancelled):
        youtube._extract('https://youtu.be/a', stop)
    assert child.killed


def test_unstable_stream_has_finite_backoff_even_if_it_returns_a_frame(monkeypatch):
    waits, statuses = [], []
    monkeypatch.setattr(sources.video, 'stream_live_frames', lambda *a, **kw: iter(['one frame']))
    monkeypatch.setattr(sources.video, 'sleep_unless_stopped', lambda stop, delay: waits.append(delay) or False)
    frames = sources.open_frames('rtsp://camera/live', on_status=lambda *s: statuses.append(s))
    with pytest.raises(RuntimeError, match='5 unstable'):
        list(frames)
    assert waits == [5, 10, 20, 40]
    assert any('Reconnecting' in message for _, message in statuses)


def test_bot_failure_is_not_retried(monkeypatch):
    def resolve(*a, **kw): raise youtube.classify('not a bot')
    monkeypatch.setattr(sources, 'resolve_youtube_stream', resolve)
    monkeypatch.setattr(sources.video, 'sleep_unless_stopped', lambda *a: pytest.fail('must not retry'))
    with pytest.raises(youtube.SourceError):
        list(sources.open_frames('https://youtu.be/a'))


def test_counts_expire_without_new_frames_and_keep_totals():
    feed = FeedSpec('f', 'file.mp4', 'CPU')
    state = pipeline._SharedState([feed])
    state.record('f', [TrackedDetection(1, 'car', .9, (0, 0, 1, 1), True)], 0)
    state.finish('f')
    snap = state.snapshot(61)
    assert snap.per_feed_last_60s['f']['Cars'] == 0
    assert snap.per_feed_total['f']['Cars'] == 1
    assert snap.active_feeds == []


def test_one_failed_feed_does_not_stop_a_healthy_feed(monkeypatch):
    monkeypatch.setattr(pipeline, 'create_detector', lambda *a, **kw: SimpleNamespace(detect=lambda f: []))
    def frames(path, **kwargs):
        if path == 'bad': raise RuntimeError('unavailable')
        yield 'frame'
    monkeypatch.setattr(sources, 'open_frames', frames)
    errors, received, ready, statuses = [], [], [], []
    pipeline.run(feeds=[FeedSpec('bad', 'bad', 'CPU'), FeedSpec('good', 'good', 'CPU')],
                 engine=Engine.PORTABLE, loop=False, on_frame=lambda fid, *a: received.append(fid),
                 on_counts=lambda s: None, on_ready=ready.append,
                 on_feed_status=lambda fid, phase, msg: statuses.append((fid, phase)),
                 on_feed_error=lambda fid, msg: errors.append(fid))
    assert errors == ['bad'] and received == ['good'] and ready == ['good']
    assert statuses == [('good', 'idle')]


def test_failed_inference_does_not_report_ready(monkeypatch):
    def detect(frame): raise RuntimeError('device failed')
    monkeypatch.setattr(pipeline, 'create_detector', lambda *a, **kw: SimpleNamespace(detect=detect))
    monkeypatch.setattr(sources, 'open_frames', lambda *a, **kw: iter(['frame']))
    ready = []
    with pytest.raises(RuntimeError, match='device failed'):
        pipeline.run(feeds=[FeedSpec('f', 'file', 'CPU')], engine=Engine.PORTABLE,
                     on_ready=ready.append, on_frame=lambda *a: None, on_counts=lambda *a: None)
    assert not ready


def test_above_cap_chooses_smallest_not_4k(monkeypatch):
    monkeypatch.setattr(sources, '_extract_formats', lambda url: [
        {'protocol': 'm3u8', 'vcodec': 'h264', 'url': str(h), 'height': h} for h in (2160, 1080)
    ])
    assert sources.resolve_youtube_stream('https://youtu.be/a') == '1080'


def test_unavailable_video_is_not_requested_on_every_start(monkeypatch):
    resolver = youtube.Resolver()
    calls = []
    def extract(url, stop):
        calls.append(url)
        raise youtube.classify('Video unavailable')
    monkeypatch.setattr(youtube, '_extract', extract)
    for _ in range(3):
        with pytest.raises(youtube.SourceError):
            resolver.resolve('https://youtu.be/a')
    assert len(calls) == 1


def test_empty_camera_clips_have_a_retry_limit(monkeypatch):
    from pantherlake_ai_core import video
    waits = []
    monkeypatch.setattr(video, 'sleep_unless_stopped', lambda stop, delay: waits.append(delay) or False)
    with pytest.raises(RuntimeError, match='five times'):
        list(video.stream_refreshing_clip('clip', _version=lambda url: 'v1', _play=lambda *a: iter([])))
    assert waits == [15, 30, 60, 60]


def test_network_capture_has_open_and_read_timeouts_and_releases(monkeypatch):
    import cv2
    from pantherlake_ai_core import video
    opened = []
    class Capture:
        released = False
        def isOpened(self): return False
        def release(self): self.released = True
    capture = Capture()
    def open_capture(*args):
        opened.append(args)
        return capture
    monkeypatch.setattr(cv2, 'VideoCapture', open_capture)
    with pytest.raises(RuntimeError):
        list(video.stream_live_frames('rtsp://camera', reconnect=False))
    assert opened[0][2] == [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000]
    assert capture.released


def test_runner_retains_feed_errors_when_other_feed_finishes(monkeypatch, tmp_path):
    from launcher import events
    from launcher.smart_city_monitor_runner import SmartCityMonitorRunner
    monkeypatch.setattr(events, 'LOG_FILE', tmp_path / 'events.log')
    def run(**kw):
        kw['on_feed_error']('bad', 'Unavailable')
        kw['on_ready']('good')
    monkeypatch.setattr(pipeline, 'run', run)
    runner = SmartCityMonitorRunner()
    runner.start(feeds=[FeedSpec('bad', 'bad', 'CPU'), FeedSpec('good', 'good', 'CPU')], engine=Engine.PORTABLE, loop=False)
    runner._thread.join(timeout=3)
    assert not runner.running
    assert runner.health()['bad']['phase'] == 'error'
    assert runner.health()['good']['phase'] == 'idle'
    assert events.status_snapshot()['smart-city-monitor:bad']['message'] == 'Unavailable'


def test_runner_reports_all_feeds_failed_after_initial_success(monkeypatch, tmp_path):
    from launcher import events
    from launcher.smart_city_monitor_runner import SmartCityMonitorRunner
    monkeypatch.setattr(events, 'LOG_FILE', tmp_path / 'events.log')
    def run(**kw):
        kw['on_ready']('f')
        kw['on_feed_error']('f', 'Stream dropped')
    monkeypatch.setattr(pipeline, 'run', run)
    runner = SmartCityMonitorRunner()
    runner.start(feeds=[FeedSpec('f', 'file', 'CPU')], engine=Engine.PORTABLE, loop=False)
    runner._thread.join(timeout=3)
    assert runner.error.startswith('All feeds stopped')


def test_runner_labels_stale_image(monkeypatch):
    from launcher.smart_city_monitor_runner import SmartCityMonitorRunner
    import time
    runner = SmartCityMonitorRunner()
    runner._health = {'f': {'phase': 'running', 'message': 'Monitoring', 'last_frame': time.monotonic() - 20}}
    assert runner.health()['f']['phase'] == 'waiting'
    assert runner.health()['f']['frame_age_seconds'] >= 20


def test_extractor_deadline_kills_a_stuck_child(monkeypatch):
    moments = iter([0, 0, 100])
    monkeypatch.setattr(youtube.time, 'monotonic', lambda: next(moments))
    class Child:
        killed = False
        def communicate(self, timeout=None):
            if self.killed: return '', ''
            raise subprocess.TimeoutExpired('extract', timeout)
        def poll(self): return -9 if self.killed else None
        def kill(self): self.killed = True
    child = Child()
    monkeypatch.setattr(youtube.subprocess, 'Popen', lambda *a, **kw: child)
    with pytest.raises(youtube.SourceError, match='timed out'):
        youtube._extract('https://youtu.be/a')
    assert child.killed


def test_transient_stream_failure_can_recover(monkeypatch):
    stop = threading.Event()
    calls = []
    def stream(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('temporary network failure')
        yield 'recovered'
        stop.set()
    monkeypatch.setattr(sources.video, 'stream_live_frames', stream)
    monkeypatch.setattr(sources.video, 'sleep_unless_stopped', lambda *a: False)
    assert list(sources.open_frames('rtsp://camera', stop_event=stop)) == ['recovered']
    assert len(calls) == 2


def test_doctor_fails_without_supported_runtime(monkeypatch):
    from smart_city_monitor import doctor
    monkeypatch.setattr(doctor.importlib.metadata, 'version', lambda package: 'test-version')
    monkeypatch.setattr(doctor.shutil, 'which', lambda name: None)
    monkeypatch.delenv('PTL_YOUTUBE_COOKIES', raising=False)
    assert doctor.main([]) == 1


def test_doctor_accepts_supported_node_without_contacting_youtube(monkeypatch):
    from smart_city_monitor import doctor
    monkeypatch.setattr(doctor.importlib.metadata, 'version', lambda package: 'test-version')
    monkeypatch.setattr(doctor.shutil, 'which', lambda name: 'node' if name == 'node' else None)
    monkeypatch.setattr(doctor.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='v22.1.0', returncode=0))
    monkeypatch.delenv('PTL_YOUTUBE_COOKIES', raising=False)
    monkeypatch.setattr(sources, 'resolve_youtube_stream', lambda *a, **kw: pytest.fail('offline check must not contact YouTube'))
    assert doctor.main([]) == 0
