"""The energy-based voice-activity segmenter, on synthetic audio: quiet
gaussian noise for silence, louder noise for speech. No microphone, no
model -- just the block bookkeeping the streaming bricks depend on."""
from __future__ import annotations

import numpy as np

from pantherlake_ai_core.segmenter import VADConfig, segment_stream

RATE = 16000
CONFIG = VADConfig()  # 30 ms blocks; 0.6 s calibration; 0.6 s hangover; 0.35 s minimum; 0.25 s pre-roll
BLOCK = int(RATE * CONFIG.block_duration)


def _blocks(seconds: float, amplitude: float, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    count = int(round(seconds / CONFIG.block_duration))
    return [(rng.standard_normal(BLOCK) * amplitude).astype(np.float32) for _ in range(count)]


def quiet(seconds: float) -> list[np.ndarray]:
    return _blocks(seconds, 0.001)


def loud(seconds: float) -> list[np.ndarray]:
    return _blocks(seconds, 0.3)


def segments(stream: list[np.ndarray]) -> list[np.ndarray]:
    return list(segment_stream(iter(stream), VADConfig()))


def test_an_utterance_followed_by_silence_is_one_segment():
    found = segments(quiet(0.6) + quiet(0.3) + loud(1.0) + quiet(1.0))
    assert len(found) == 1
    assert found[0].dtype == np.float32
    # ~1.0 s of speech plus up to 0.25 s of pre-roll, trailing silence trimmed
    assert RATE * 0.9 <= found[0].size <= RATE * 1.4


def test_a_stream_that_ends_mid_utterance_still_yields_it():
    # No trailing silence at all: the source stopped (or Stop was pressed)
    # while someone was talking. That last sentence must not be dropped.
    found = segments(quiet(0.6) + quiet(0.3) + loud(1.0))
    assert len(found) == 1
    assert found[0].size >= RATE * 0.9


def test_the_final_flush_trims_trailing_silence_like_a_normal_cut():
    # Silence shorter than the hangover, then end-of-stream.
    (segment,) = segments(quiet(0.6) + quiet(0.3) + loud(1.0) + quiet(0.3))
    assert segment.size <= RATE * 1.3


def test_a_burst_shorter_than_the_minimum_is_dropped():
    assert segments(quiet(0.6) + quiet(0.3) + loud(0.05) + quiet(1.0)) == []


def test_two_utterances_are_two_segments():
    found = segments(quiet(0.6) + quiet(0.3) + loud(0.8) + quiet(1.0) + loud(0.8) + quiet(1.0))
    assert len(found) == 2


def test_default_config_is_not_shared_between_calls():
    # segment_stream(blocks) must build its own VADConfig each time -- a
    # shared mutable default would let one caller's tweak leak into another.
    stream = quiet(0.6) + quiet(0.3) + loud(1.0) + quiet(1.0)
    assert len(list(segment_stream(iter(stream)))) == 1
    assert len(list(segment_stream(iter(stream)))) == 1
