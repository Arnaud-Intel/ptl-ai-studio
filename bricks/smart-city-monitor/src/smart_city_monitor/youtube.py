"""Bounded YouTube extraction, shared pacing/cache and a process-wide circuit breaker.

The extractor runs in a child process so Stop can interrupt a stuck network or
JavaScript call. No account, proxy rotation or background feed probing is used.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

MIN_INTERVAL = 8.0
BLOCK_COOLDOWN = 900.0
EXTRACT_TIMEOUT = 60.0
CACHE_SECONDS = 90.0


class SourceError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, blocked: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.blocked = blocked


class Cancelled(Exception):
    pass


def safe_message(message: str) -> str:
    """Extractor/CDN errors can contain signed URLs. Never send these to logs/UI."""
    message = re.sub(r"\x1b\[[0-9;]*m", "", str(message))
    message = re.sub(r"(?:https?|rtsp|rtmp)://\S+", "[URL omitted]", message)
    return " ".join(message.split())[:600]


def classify(message: str) -> SourceError:
    clean = safe_message(message)
    lower = clean.lower()
    fallback = ' Use Other > London, two chips, a direct camera stream (RTSP/HLS), or a local video.'
    if any(s in lower for s in ("not a bot", "sign in to confirm", "429", "too many requests", "rate limit", "this content isn't available, try again later")):
        return SourceError('YouTube blocked or rate-limited this session. New YouTube connections are paused for 15 minutes.' + fallback,
                           blocked=True)
    if any(s in lower for s in ("video unavailable", "private video", "removed", "not available in your country", "copyright", "members-only", "age-restricted")):
        return SourceError('This YouTube video is unavailable or restricted; this alone does not prove an account ban.' + fallback)
    if any(s in lower for s in ("403", "forbidden", "po token")):
        return SourceError('YouTube denied playback (403/token/session restriction). Repeated retries may worsen blocking.' + fallback)
    if any(s in lower for s in ("timed out", "timeout", "connection", "temporary failure", "502", "503", "504", "network is unreachable")):
        return SourceError('Temporary YouTube connection failure: ' + clean, retryable=True)
    return SourceError('YouTube extraction failed: ' + clean +
                       '. Check Deno and yt-dlp[default]; update with uv sync --extra openvino --upgrade-package yt-dlp.' + fallback)


def _check_stop(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise Cancelled()


def _extract(url: str, stop_event=None) -> list[dict]:
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    child = subprocess.Popen(
        [sys.executable, '-m', 'smart_city_monitor.youtube', url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
        creationflags=flags,
    )
    deadline = time.monotonic() + EXTRACT_TIMEOUT
    try:
        while True:
            _check_stop(stop_event)
            if time.monotonic() >= deadline:
                raise SourceError('YouTube extraction timed out after 60 seconds.', retryable=True)
            try:
                output, stderr = child.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        try:
            result = json.loads(output)
        except (ValueError, TypeError):
            raise SourceError('YouTube helper failed: ' + safe_message(stderr or 'invalid response')) from None
        for warning in result.get('warnings', []):
            logging.getLogger(__name__).warning('YouTube: %s', safe_message(warning))
        if 'error' in result:
            raise classify(result['error'])
        if child.returncode:
            raise SourceError('YouTube helper exited unexpectedly.')
        return result['formats']
    finally:
        if child.poll() is None:
            child.kill()
        try:
            child.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            # A descendant may still hold a pipe handle. Never let cleanup
            # turn a bounded extraction into an unbounded Stop.
            child.stdout.close()
            child.stderr.close()


class Resolver:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.blocked_until = 0.0
        self.cache: dict[str, tuple[float, list[dict]]] = {}
        self.failures: dict[str, tuple[float, str]] = {}

    @staticmethod
    def key(url):
        parsed = urlparse(url)
        video_id = parse_qs(parsed.query).get('v', [''])[0]
        if parsed.hostname in ('youtu.be', 'www.youtu.be'):
            video_id = parsed.path.strip('/')
        return video_id or url

    def invalidate(self, url, stop_event=None):
        while not self.lock.acquire(timeout=0.1):
            if stop_event is not None and stop_event.is_set():
                return
        try:
            self.cache.pop(self.key(url), None)
        finally:
            self.lock.release()

    def resolve(self, url, stop_event=None):
        # Serializes both extraction and writes to an opted-in cookie jar.
        while not self.lock.acquire(timeout=0.1):
            _check_stop(stop_event)
        try:
            _check_stop(stop_event)
            now = time.monotonic()
            if now < self.blocked_until:
                remaining = int(self.blocked_until - now) + 1
                raise SourceError(f'YouTube cooldown: {remaining} seconds remaining. Use Other > London, two chips or a local video.', blocked=True)
            key = self.key(url)
            self.failures = {k: v for k, v in self.failures.items() if v[0] > now}
            if key in self.failures:
                raise SourceError(self.failures[key][1] + ' This video will be checked again after a one-minute cooldown.')
            self.cache = {k: v for k, v in self.cache.items() if v[0] > now}
            if key in self.cache:
                return self.cache[key][1]
            while time.monotonic() < self.next_request:
                _check_stop(stop_event)
                time.sleep(max(0, min(0.1, self.next_request - time.monotonic())))
            self.next_request = time.monotonic() + MIN_INTERVAL
            try:
                formats = _extract(url, stop_event)
            except SourceError as exc:
                if exc.blocked:
                    self.blocked_until = time.monotonic() + BLOCK_COOLDOWN
                    self.cache.clear()
                elif not exc.retryable:
                    if len(self.failures) >= 128:
                        self.failures.pop(next(iter(self.failures)))
                    self.failures[key] = (time.monotonic() + 60, str(exc))
                raise
            if not any(f.get('url') and f.get('vcodec') != 'none' for f in formats):
                message = 'No playable HLS video stream -- is it actually live? Check Deno and yt-dlp[default], or choose Other > London, two chips.'
                self.failures[key] = (time.monotonic() + 60, message)
                raise SourceError(message)
            if len(self.cache) >= 128:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = (time.monotonic() + CACHE_SECONDS, formats)
            return formats
        finally:
            self.lock.release()


resolver = Resolver()


def main():
    # Worker protocol stays off the UI; only redacted diagnostics leave the parent.
    from .sources import _ytdlp_options
    warnings = []

    class Logger:
        def debug(self, message): pass
        def warning(self, message): warnings.append(safe_message(message))
        def error(self, message): pass

    try:
        import yt_dlp
        options = _ytdlp_options()
        options['logger'] = Logger()
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(sys.argv[1], download=False) or {}
        formats = [{k: f.get(k) for k in ('protocol', 'vcodec', 'url', 'height')}
                   for f in info.get('formats', []) if f.get('protocol') in ('m3u8', 'm3u8_native')]
        print(json.dumps({'formats': formats, 'warnings': warnings}))
    except Exception as exc:
        print(json.dumps({'error': str(exc), 'warnings': warnings}))


if __name__ == '__main__':
    main()
