"""Energy-based voice-activity segmentation.

Turns a continuous stream of small audio blocks into discrete speech
utterances. This avoids depending on a natively-compiled VAD library
(e.g. webrtcvad), which can be awkward to install on Windows.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class VADConfig:
    block_duration: float = 0.03
    # How far back the noise floor looks, and which of those blocks it takes
    # for "the room with nobody talking" (see segment_stream).
    floor_window: float = 10.0
    floor_percentile: float = 5.0
    threshold_multiplier: float = 3.5
    min_threshold: float = 0.004
    min_speech_duration: float = 0.35
    max_speech_duration: float = 14.0
    silence_hangover: float = 0.6
    pre_roll: float = 0.25


def _rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block))))


def segment_stream(blocks, config: VADConfig | None = None):
    """Consume an iterator of mono float32 blocks; yield complete speech
    segments as concatenated float32 numpy arrays.

    Speech is whatever is `threshold_multiplier` times louder than the
    room, and the room is measured all the time: the noise floor is a low
    percentile of the last `floor_window` seconds of block levels, so it
    sits in the gaps between words whoever is talking.

    It used to be measured once, over the first 0.6 s after Start, and kept
    for the session. That works for someone who presses Start and then
    speaks. Someone presenting is already speaking: the floor was taken
    from their voice, the threshold landed above it, and the session heard
    little or nothing -- on a simulated microphone, not one segment out of
    39 s of speech (2026-10-06, live translation restarted four times in
    ninety seconds during a demo). A floor that follows the room also
    survives what a call does to a microphone: a gain that moves, a fan
    that starts.
    """
    config = config or VADConfig()  # a fresh default per call, not one shared mutable instance
    pre_roll_blocks = max(1, int(config.pre_roll / config.block_duration))
    hangover_blocks = max(1, int(config.silence_hangover / config.block_duration))
    min_speech_blocks = max(1, int(config.min_speech_duration / config.block_duration))
    max_speech_blocks = max(1, int(config.max_speech_duration / config.block_duration))
    floor_blocks = max(1, int(config.floor_window / config.block_duration))

    pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_blocks)
    recent_levels: deque[float] = deque(maxlen=floor_blocks)

    in_speech = False
    speech_blocks: list[np.ndarray] = []
    silence_run = 0

    for block in blocks:
        level = _rms(block)
        recent_levels.append(level)
        noise_floor = float(np.percentile(recent_levels, config.floor_percentile))
        threshold = max(config.min_threshold, noise_floor * config.threshold_multiplier)

        is_loud = level > threshold

        if not in_speech:
            pre_roll.append(block)
            if is_loud:
                in_speech = True
                speech_blocks = list(pre_roll)
                silence_run = 0
            continue

        speech_blocks.append(block)
        silence_run = 0 if is_loud else silence_run + 1

        hit_max = len(speech_blocks) >= max_speech_blocks
        hit_silence = silence_run >= hangover_blocks

        if hit_max or hit_silence:
            if hit_silence and silence_run > 0:
                speech_blocks = speech_blocks[: len(speech_blocks) - silence_run]
            if len(speech_blocks) >= min_speech_blocks:
                yield np.concatenate(speech_blocks).astype(np.float32)
            in_speech = False
            speech_blocks = []
            silence_run = 0
            pre_roll.clear()
            pre_roll.append(block)

    # The stream ended (stop requested, or the source ran out) mid-utterance:
    # flush what was captured rather than silently dropping the last thing
    # said -- trimming any trailing silence the same way a normal cut does.
    if in_speech and silence_run > 0:
        speech_blocks = speech_blocks[: len(speech_blocks) - silence_run]
    if in_speech and len(speech_blocks) >= min_speech_blocks:
        yield np.concatenate(speech_blocks).astype(np.float32)
