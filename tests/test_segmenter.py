"""The energy-based voice-activity segmenter, on synthetic audio: quiet
gaussian noise for silence, louder noise for speech. No microphone, no
model -- just the block bookkeeping the streaming bricks depend on."""
from __future__ import annotations

import numpy as np

from pantherlake_ai_core.segmenter import VADConfig, segment_stream

RATE = 16000
CONFIG = VADConfig()  # 30 ms blocks; 0.6 s hangover; 0.35 s minimum; 0.25 s pre-roll; noise floor from the last 10 s
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


# --- the noise floor is measured all the time, not once at the start ----------------------


def talking(seconds: float, seed: int = 1) -> list[np.ndarray]:
    """Speech-like: bursts of sound with the short gaps there are between
    words, where steady noise has none."""
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    while len(blocks) * CONFIG.block_duration < seconds:
        blocks += [(rng.standard_normal(BLOCK) * 0.3).astype(np.float32) for _ in range(6)]  # 0.18 s of voice
        blocks += [(rng.standard_normal(BLOCK) * 0.001).astype(np.float32) for _ in range(2)]  # 0.06 s between words
    return blocks[: int(round(seconds / CONFIG.block_duration))]


def test_someone_already_talking_when_the_stream_opens_is_heard():
    # Presenting while pressing Start. The floor used to be taken from the
    # first 0.6 s -- from the voice -- and the session then heard nothing:
    # not this sentence, not the next one, until it was restarted in silence.
    found = segments(talking(2.0) + quiet(1.0) + talking(2.0, seed=2) + quiet(1.0))
    assert len(found) == 2
    assert found[0].size >= RATE * 1.6  # all but the first word or so of the sentence in progress
    assert found[1].size >= RATE * 1.8


def test_a_room_that_gets_louder_stops_being_taken_for_speech():
    # A fan starts, a call's noise suppression lets go: steady sound ten
    # times the old floor. Against a floor fixed at the start it was speech
    # for the rest of the session, 14 s of it at a time.
    fan = _blocks(30.0, 0.01, seed=3)
    voice = [block + extra for block, extra in zip(talking(1.5, seed=4), _blocks(1.5, 0.01, seed=5))]
    found = segments(quiet(1.0) + fan + voice + _blocks(2.0, 0.01, seed=6))
    assert RATE * 1.2 <= found[-1].size <= RATE * 2.2  # the sentence, found over the fan
    heard_after_settling = [s for s in found[1:-1]]
    assert len(found) <= 2 and not heard_after_settling  # at most the fan's first seconds, then quiet


def test_a_microphone_turned_down_mid_session_is_still_heard():
    # What a call's automatic gain control does: everything a third as loud.
    later = [block * 0.3 for block in quiet(1.0) + talking(1.5, seed=7) + quiet(1.0)]
    found = segments(quiet(1.0) + talking(1.5) + quiet(1.0) + later)
    assert len(found) == 2
