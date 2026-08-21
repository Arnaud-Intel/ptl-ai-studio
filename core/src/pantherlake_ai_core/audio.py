"""Audio capture from a microphone or from system audio (loopback).

Uses `soundcard`, which supports WASAPI loopback recording on Windows so
whatever is currently playing on the device (e.g. a video) can be captured
directly, with no "Stereo Mix" device required.
"""
from __future__ import annotations

import threading

import numpy as np
import soundcard as sc

SAMPLE_RATE = 16000


def list_microphones() -> list[str]:
    return [m.name for m in sc.all_microphones(include_loopback=False)]


def list_speakers() -> list[str]:
    return [s.name for s in sc.all_speakers()]


def _find_microphone(name_filter: str | None):
    mics = sc.all_microphones(include_loopback=False)
    if not mics:
        raise RuntimeError("No microphone found.")
    if name_filter:
        for m in mics:
            if name_filter.lower() in m.name.lower():
                return m
        raise RuntimeError(
            f"No microphone matching '{name_filter}'. Available: {[m.name for m in mics]}"
        )
    return sc.default_microphone()


def _find_loopback(name_filter: str | None):
    speakers = sc.all_speakers()
    if not speakers:
        raise RuntimeError("No output/speaker device found.")
    if name_filter:
        for s in speakers:
            if name_filter.lower() in s.name.lower():
                return sc.get_microphone(id=s.name, include_loopback=True)
        raise RuntimeError(
            f"No output device matching '{name_filter}'. Available: {[s.name for s in speakers]}"
        )
    default_speaker = sc.default_speaker()
    return sc.get_microphone(id=default_speaker.name, include_loopback=True)


def _find_speaker(name_filter: str | None):
    speakers = sc.all_speakers()
    if not speakers:
        raise RuntimeError("No output/speaker device found.")
    if name_filter:
        for s in speakers:
            if name_filter.lower() in s.name.lower():
                return s
        raise RuntimeError(
            f"No output device matching '{name_filter}'. Available: {[s.name for s in speakers]}"
        )
    return sc.default_speaker()


def play(data: np.ndarray, sample_rate: int, device: str | None = None) -> None:
    """Blocking playback of a mono/stereo float32 array through a speaker
    -- for bricks that talk back (e.g. voice-assistant)."""
    speaker = _find_speaker(device)
    speaker.play(data, samplerate=sample_rate)


def get_input(source: str, device: str | None):
    """Return a soundcard microphone/loopback handle for 'mic' or 'system'."""
    if source == "mic":
        return _find_microphone(device)
    if source == "system":
        return _find_loopback(device)
    raise ValueError(f"Unknown source '{source}', expected 'mic' or 'system'.")


def stream_blocks(
    source: str,
    device: str | None,
    block_duration: float = 0.03,
    stop_event: threading.Event | None = None,
):
    """Yield mono float32 audio blocks at SAMPLE_RATE from the chosen source.

    Runs until `stop_event` is set, or forever if none is given. Passing a
    stop_event lets a long-running consumer (e.g. a web UI's background
    thread) break the loop promptly instead of blocking on Ctrl+C.
    """
    mic = get_input(source, device)
    block_size = max(1, int(SAMPLE_RATE * block_duration))
    with mic.recorder(samplerate=SAMPLE_RATE, blocksize=block_size) as rec:
        while stop_event is None or not stop_event.is_set():
            data = rec.record(numframes=block_size)
            if data.ndim == 2 and data.shape[1] > 1:
                mono = data.mean(axis=1)
            else:
                mono = data.reshape(-1)
            yield mono.astype(np.float32)
