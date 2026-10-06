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


# --- the NPU reset under a running session (pantherlake_ai_core.npu) ---------------------


class _MovingTranslator:
    """Leaves the NPU on its second utterance, the way OpenVINOTranslator
    does when Windows resets the chip under it."""

    def __init__(self) -> None:
        self.device = "NPU"
        self.calls = 0

    def translate(self, segment: str) -> TranslationResult:
        self.calls += 1
        if self.calls == 2:
            self.device = "GPU.0"
        return TranslationResult(text=f"heard {segment}", detected_language="fr", language_probability=1.0)


def _run_with(monkeypatch, translator, **callbacks):
    loads: list[str] = []

    def fake_create_translator(**kwargs):
        loads.append(kwargs["device"])
        return translator

    monkeypatch.setattr(pipeline, "create_translator", fake_create_translator)
    monkeypatch.setattr(pipeline.audio, "stream_blocks", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(pipeline, "segment_stream", lambda blocks, config: iter(["one", "two", "three"]))
    results: list[TranslationResult] = []
    pipeline.run(
        source="mic", audio_device=None, engine=Engine.OPENVINO, model_size="base", compute_device="NPU",
        on_result=results.append, **callbacks,
    )
    return loads, results


def test_a_model_that_had_to_leave_the_npu_is_reported_once_and_the_session_carries_on(monkeypatch):
    moved: list[str] = []
    recovering: list[Exception] = []
    loads, results = _run_with(monkeypatch, _MovingTranslator(), on_device=moved.append, on_recovering=recovering.append)
    assert [r.text for r in results] == ["heard one", "heard two", "heard three"]
    assert moved == ["GPU.0"]  # the hardware panel follows the model to its new chip
    assert loads == ["NPU"] and not recovering  # no reload: the model moved itself


def test_a_lost_npu_is_never_answered_with_a_reload(monkeypatch):
    from pantherlake_ai_core import npu

    class _Lost:
        device = "NPU"

        def translate(self, segment: str):
            raise npu.NpuLost(npu.LOST_MESSAGE)

    loads: list[str] = []
    monkeypatch.setattr(pipeline, "create_translator", lambda **kwargs: loads.append(kwargs["device"]) or _Lost())
    monkeypatch.setattr(pipeline.audio, "stream_blocks", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(pipeline, "segment_stream", lambda blocks, config: iter(["one", "two"]))
    recovering: list[Exception] = []
    with pytest.raises(npu.NpuLost):
        pipeline.run(
            source="mic", audio_device=None, engine=Engine.OPENVINO, model_size="base", compute_device="NPU",
            on_result=lambda result: None, on_recovering=recovering.append,
        )
    # Reloading onto a chip Windows has reset is what took the launcher down
    # on 2026-10-05 and 2026-10-06: the driver ended the process 20-30 s later.
    assert loads == ["NPU"] and not recovering
