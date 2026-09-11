"""live-translation outlives one failed utterance: it reloads the model and
retries, rather than ending the session over a single dropped request."""
from __future__ import annotations

import pytest
from live_translation import pipeline
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import TranslationResult


class _FlakyTranslator:
    """Raises once per entry in `failures` (shared across reloads), then
    translates normally."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = failures

    def translate(self, segment: str) -> TranslationResult:
        if self.failures:
            raise RuntimeError(self.failures.pop(0))
        return TranslationResult(text=f"heard {segment}", detected_language="fr", language_probability=1.0)


def _run(monkeypatch, failures: list[str]):
    loads: list[str] = []

    def fake_create_translator(**kwargs):
        loads.append(kwargs["device"])
        return _FlakyTranslator(failures)

    monkeypatch.setattr(pipeline, "create_translator", fake_create_translator)
    monkeypatch.setattr(pipeline.audio, "stream_blocks", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(pipeline, "segment_stream", lambda blocks, config: iter(["one", "two", "three"]))
    results: list[TranslationResult] = []
    recovering: list[Exception] = []
    ready: list[bool] = []
    pipeline.run(
        source="mic",
        audio_device=None,
        engine=Engine.OPENVINO,
        model_size="base",
        compute_device="NPU",
        on_result=results.append,
        on_ready=lambda: ready.append(True),
        on_recovering=recovering.append,
    )
    return loads, results, recovering, ready


def test_one_failed_utterance_reloads_the_model_and_carries_on(monkeypatch):
    loads, results, recovering, ready = _run(monkeypatch, ["ZE_RESULT_ERROR_INVALID_ARGUMENT"])
    assert [r.text for r in results] == ["heard one", "heard two", "heard three"]  # the failed one retried
    assert loads == ["NPU", "NPU"]
    assert len(recovering) == 1 and "ZE_RESULT" in str(recovering[0])
    assert len(ready) == 2  # back to "running" after the reload


def test_a_failure_the_reload_does_not_fix_still_surfaces(monkeypatch):
    with pytest.raises(RuntimeError, match="still broken"):
        _run(monkeypatch, ["dropped request", "still broken"])


def test_a_healthy_session_never_reloads(monkeypatch):
    loads, results, recovering, _ = _run(monkeypatch, [])
    assert loads == ["NPU"] and not recovering and len(results) == 3
