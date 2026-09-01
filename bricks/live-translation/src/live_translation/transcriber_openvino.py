"""OpenVINO translation backend: runs Whisper on Intel CPU, integrated GPU,
or NPU (e.g. Panther Lake) via `openvino_genai`.

This is what actually showcases the target hardware, as opposed to
transcriber_portable.py which is CPU/CUDA only. Requires this brick's
`openvino` extra: `uv sync --extra openvino` (see the workspace root).

By default this downloads a pre-converted multilingual model from Intel's
`OpenVINO` org on Hugging Face (e.g. OpenVINO/whisper-base-fp16-ov) the
first time it's used, then runs fully offline. Pass --ov-model-dir to
point at a model you converted yourself with `optimum-cli export openvino`
(see this brick's README for that command) -- useful for sizes or
quantizations Intel hasn't pre-published.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from pantherlake_ai_core.types import TranslationResult

# Multilingual (not "*.en") variants only -- translation needs to recognize
# non-English source speech, which the English-only models can't do.
_AVAILABLE_SIZES = ("tiny", "base", "medium", "large-v3")
_DEFAULT_REPO_TEMPLATE = "OpenVINO/whisper-{size}-fp16-ov"


def _resolve_model_dir(
    model_size: str, model_dir: str | None, on_downloading: Callable[[], None] | None = None
) -> str:
    if model_dir:
        return model_dir
    if model_size not in _AVAILABLE_SIZES:
        raise ValueError(
            f"No pre-converted multilingual OpenVINO model for size '{model_size}'. "
            f"Available sizes: {', '.join(_AVAILABLE_SIZES)}. "
            "For another size/model, convert it yourself with `optimum-cli export "
            "openvino` and pass --ov-model-dir."
        )
    from huggingface_hub import snapshot_download
    from pantherlake_ai_core.model_cache import is_repo_cached

    repo_id = _DEFAULT_REPO_TEMPLATE.format(size=model_size)
    if on_downloading is not None and not is_repo_cached(repo_id):
        on_downloading()
    return snapshot_download(repo_id)


def _load_pipeline_class():
    import openvino_genai as ov_genai

    # openvino_genai renamed WhisperPipeline -> ASRPipeline in newer
    # releases (it now also covers non-Whisper ASR models). Support both so
    # this brick works across the versions users may have installed.
    pipeline_cls = getattr(ov_genai, "ASRPipeline", None) or getattr(ov_genai, "WhisperPipeline", None)
    if pipeline_cls is None:
        raise RuntimeError(
            "Installed openvino_genai has neither ASRPipeline nor WhisperPipeline. "
            "Please upgrade the `openvino-genai` package."
        )
    return pipeline_cls


class OpenVINOTranslator:
    """Loads a local OpenVINO Whisper model once and translates audio chunks
    to English text, targeting whichever OpenVINO device is requested."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "AUTO",
        model_dir: str | None = None,
        task: str = "translate",
        on_downloading: Callable[[], None] | None = None,
    ):
        pipeline_cls = _load_pipeline_class()
        resolved_dir = _resolve_model_dir(model_size, model_dir, on_downloading)

        ov_config = {}
        if device == "NPU" or "GPU" in device:
            # Cache compiled models on disk for GPU/NPU: recompiling on every
            # run is slow, and isn't needed for CPU.
            ov_config["CACHE_DIR"] = "ov_cache"

        self.pipeline = pipeline_cls(resolved_dir, device, **ov_config)
        self.task = task

    def translate(self, audio: np.ndarray) -> TranslationResult | None:
        result = self.pipeline.generate(audio.tolist(), task=self.task)
        text = str(result).strip()
        if not text:
            return None
        # result.languages is a list of per-chunk detected codes (e.g.
        # ["fr"], same 2-letter format as faster-whisper's `info.language`);
        # a short single-utterance chunk normally yields exactly one.
        detected_language = result.languages[0] if result.languages else "auto"
        return TranslationResult(
            text=text,
            detected_language=detected_language,
            language_probability=1.0,
        )
