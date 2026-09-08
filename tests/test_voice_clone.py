"""voice-clone-studio's wrapper around the vendored OpenVoice extractor.

The vendored code states its input preconditions as bare `assert`s. Those
are bad-input conditions, so `enroll` re-raises them as ValueError -- which
the launcher answers with a 400 rather than a 500, and which survives
`python -O` (asserts don't).
"""
from __future__ import annotations

import pytest
from voice_clone_studio import voice_model


def test_a_clip_with_no_speech_is_a_value_error(monkeypatch):
    def refuse(path, converter, **kwargs):
        raise AssertionError("no speech detected in the reference clip -- try a clearer recording")

    monkeypatch.setattr(voice_model.se_extractor, "get_se", refuse)
    with pytest.raises(ValueError, match="no speech detected"):
        voice_model.enroll(converter=object(), reference_audio_path="quiet.wav")


def test_a_bare_assert_still_gets_a_usable_message(monkeypatch):
    def refuse(path, converter, **kwargs):
        raise AssertionError()

    monkeypatch.setattr(voice_model.se_extractor, "get_se", refuse)
    with pytest.raises(ValueError, match="can't be used"):
        voice_model.enroll(converter=object(), reference_audio_path="quiet.wav")


def test_a_good_clip_passes_the_embedding_through(monkeypatch):
    monkeypatch.setattr(voice_model.se_extractor, "get_se", lambda path, converter, **kw: ("embedding", None))
    assert voice_model.enroll(converter=object(), reference_audio_path="speech.wav") == "embedding"
