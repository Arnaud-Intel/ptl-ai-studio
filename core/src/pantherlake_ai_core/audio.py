"""Audio capture from a microphone or from system audio (loopback).

Uses `soundcard`, which supports WASAPI loopback recording on Windows so
whatever is currently playing on the device (e.g. a video) can be captured
directly, with no "Stereo Mix" device required.
"""
from __future__ import annotations

import threading

import numpy as np

# soundcard binds to the platform's audio service when imported (WASAPI,
# CoreAudio, PulseAudio) and fails outright where there is none -- a
# headless CI runner, a container. Importing *this* module must still work
# there (the segmenter, the launcher's registry and routes, and the tests
# all import through it); only actually capturing or playing audio needs
# the backend, so that is where the failure is raised, with the reason.
#
# Deliberately `Exception`, not a specific type: soundcard signals this in
# whatever way its per-platform backend happens to -- on Linux with no
# PulseAudio daemon it's a bare `assert` on the connection state, elsewhere
# an OSError or a cffi error. There is no useful way to enumerate them, and
# every one of them means the same thing here: no audio on this machine.
try:
    import soundcard as _soundcard
except Exception as exc:  # pragma: no cover - depends on the host
    _soundcard = None
    _SOUNDCARD_ERROR: Exception | None = exc
else:
    _SOUNDCARD_ERROR = None

SAMPLE_RATE = 16000


def _backend():
    if _soundcard is None:
        reason = str(_SOUNDCARD_ERROR) or type(_SOUNDCARD_ERROR).__name__
        raise RuntimeError(f"Audio isn't available on this machine: {reason}")
    return _soundcard


def list_microphones() -> list[str]:
    if _soundcard is None:
        return []
    return [m.name for m in _soundcard.all_microphones(include_loopback=False)]


def list_speakers() -> list[str]:
    if _soundcard is None:
        return []
    return [s.name for s in _soundcard.all_speakers()]


def _find_microphone(name_filter: str | None):
    sc = _backend()
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
    sc = _backend()
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
    sc = _backend()
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
