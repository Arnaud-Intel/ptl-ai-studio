"""core.audio has to import on a machine with no audio backend.

`soundcard` binds to the platform's audio service at import time, and on a
headless runner (no PulseAudio daemon) it fails with a bare AssertionError
from deep inside its connection setup. Everything imports through this
module -- the segmenter's callers, the launcher's routes, this test suite
-- so an unguarded import there takes all of it down. Only actual capture
or playback should need the backend.
"""
from __future__ import annotations

import pytest

from pantherlake_ai_core import audio


@pytest.fixture
def no_backend(monkeypatch):
    """Pretend the import failed the way a headless Linux runner does."""
    monkeypatch.setattr(audio, "_soundcard", None)
    monkeypatch.setattr(audio, "_SOUNDCARD_ERROR", AssertionError())


def test_device_lists_are_empty_rather_than_raising(no_backend):
    assert audio.list_microphones() == []
    assert audio.list_speakers() == []


@pytest.mark.parametrize("source", ["mic", "system"])
def test_capture_says_why_it_cannot_run(no_backend, source):
    with pytest.raises(RuntimeError, match="Audio isn't available on this machine"):
        audio.get_input(source, None)


def test_the_reason_is_named_even_when_the_error_has_no_message(no_backend):
    # soundcard's headless failure is a bare `assert`, i.e. str(exc) == "".
    with pytest.raises(RuntimeError, match="AssertionError"):
        audio.get_input("mic", None)


def test_an_unknown_source_is_still_a_value_error(no_backend):
    with pytest.raises(ValueError, match="Unknown source"):
        audio.get_input("telepathy", None)
