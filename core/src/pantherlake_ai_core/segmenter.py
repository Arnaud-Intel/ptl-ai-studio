"""Energy-based voice-activity segmentation.

Turns a continuous stream of small audio blocks into discrete speech
utterances. This avoids depending on a natively-compiled VAD library
(e.g. webrtcvad), which can be awkward to install on Windows.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE


@dataclass
class VADConfig:
    block_duration: float = 0.03
    calibration_seconds: float = 0.6
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


def segment_stream(blocks, config: VADConfig = VADConfig()):
    """Consume an iterator of mono float32 blocks; yield complete speech
    segments as concatenated float32 numpy arrays.

    The first `calibration_seconds` of audio are used to estimate the
    ambient noise floor and set a speech-detection threshold from it.
    """
    pre_roll_blocks = max(1, int(config.pre_roll / config.block_duration))
    hangover_blocks = max(1, int(config.silence_hangover / config.block_duration))
    min_speech_blocks = max(1, int(config.min_speech_duration / config.block_duration))
    max_speech_blocks = max(1, int(config.max_speech_duration / config.block_duration))
    calib_blocks = max(1, int(config.calibration_seconds / config.block_duration))

    pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_blocks)
    calibration: list[float] = []
    threshold = config.min_threshold

    in_speech = False
    speech_blocks: list[np.ndarray] = []
    silence_run = 0

    for block in blocks:
        level = _rms(block)

        if len(calibration) < calib_blocks:
            calibration.append(level)
            pre_roll.append(block)
            if len(calibration) == calib_blocks:
                noise_floor = float(np.median(calibration))
                threshold = max(config.min_threshold, noise_floor * config.threshold_multiplier)
            continue

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
