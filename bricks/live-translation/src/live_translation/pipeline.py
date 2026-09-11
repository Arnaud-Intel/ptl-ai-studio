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
    on_ready: Callable[[], None] | None = None,
    on_downloading: Callable[[], None] | None = None,
    on_recovering: Callable[[Exception], None] | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread, calling `on_result` for each translated
    utterance, until `stop_event` is set (or forever if none is given).
    `on_ready`, if given, fires once the model is loaded and capture is
    about to start -- the model load is the only slow, blocking step here,
    so this is the real "loading -> actually running" boundary, not
    something a caller can infer from timing or from the first result
    (which may be seconds or minutes away depending on when someone speaks).
    `on_downloading`, if given, fires before that load has to fetch the
    model from the network rather than just reading it off local disk.
    `on_recovering(exc)`, if given, fires when an utterance fails to
    translate and the model is being reloaded to retry it; `on_ready`
    fires again once it's back.
    """

    def load():
        return create_translator(
            engine=engine,
            model_size=model_size,
            device=compute_device,
            compute_type=compute_type,
            ov_model_dir=ov_model_dir,
            on_downloading=on_downloading,
        )

    translator = load()
    if on_ready is not None:
        on_ready()
    blocks = audio.stream_blocks(source, audio_device, stop_event=stop_event)
    for segment in segment_stream(blocks, VADConfig()):
        try:
            result = translator.translate(segment)
        except RuntimeError as exc:
            # Seen once on the NPU (2026-09-11, 42s into a session): the
            # driver rejected a single request -- Level Zero
            # ZE_RESULT_ERROR_INVALID_ARGUMENT -- and over a hundred attempts
            # to provoke it again all went through. One dropped request
            # shouldn't end the session, so reload the model and retry that
            # utterance. If the fresh model fails too it isn't a hiccup, and
            # that second error surfaces as before.
            if on_recovering is not None:
                on_recovering(exc)
            del translator
            translator = load()
            result = translator.translate(segment)
            if on_ready is not None:
                on_ready()
        if result is not None:
            on_result(result)
