"""Capture -> segment -> translate loop, shared by the CLI and any UI
front-end (e.g. the launcher) so the streaming logic lives in one place.
"""
from __future__ import annotations

import threading
from typing import Callable

from pantherlake_ai_core import audio
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.segmenter import VADConfig, segment_stream
from pantherlake_ai_core.types import TranslationResult

from .transcriber import create_translator


def run(
    *,
    source: str,
    audio_device: str | None,
    engine: Engine,
    model_size: str,
    compute_device: str,
    compute_type: str = "auto",
    ov_model_dir: str | None = None,
    on_result: Callable[[TranslationResult], None],
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread, calling `on_result` for each translated
    utterance, until `stop_event` is set (or forever if none is given)."""
    translator = create_translator(
        engine=engine,
        model_size=model_size,
        device=compute_device,
        compute_type=compute_type,
        ov_model_dir=ov_model_dir,
    )
    blocks = audio.stream_blocks(source, audio_device, stop_event=stop_event)
    for segment in segment_stream(blocks, VADConfig()):
        result = translator.translate(segment)
        if result is not None:
            on_result(result)
