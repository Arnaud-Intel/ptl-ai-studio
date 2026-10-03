"""Shared result types used across demo bricks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class TranslationResult:
    text: str
    detected_language: str
    language_probability: float
    # How long the utterance was and how long it took to process, filled in
    # by the capture loop: speech is measured in times real time, not tokens.
    audio_seconds: float | None = None
    processing_seconds: float | None = None


@dataclass
class GenerationControl:
    """A way into a model call while it runs: `on_text` receives each piece
    of the answer as it is written, and `should_stop` is asked after every
    piece -- answering True ends the call with what has been written so far.
    `on_tokens` is told how many tokens were just produced -- what a
    tokens-per-second figure counts, since a piece of text is not a token
    (in HTML or code, three pieces are about four tokens). `on_heading`
    receives the lines a brick adds itself between two answers; left unset,
    they go to `on_text` with the rest.

    One object rather than two parameters, so the bricks between a runner
    and the model pass a single thing down. With no control, a call behaves
    exactly as it did before streaming existed and pays nothing for it."""

    on_text: Callable[[str], None] | None = None
    should_stop: Callable[[], bool] | None = None
    on_heading: Callable[[str], None] | None = None
    on_tokens: Callable[[int], None] | None = None


def say(control: GenerationControl | None, text: str) -> None:
    """Put a line of the brick's own (a heading between two answers) into
    the running text. Kept apart from the model's pieces so whoever counts
    tokens per second doesn't count these."""
    if control is None:
        return
    receive = control.on_heading or control.on_text
    if receive is not None:
        receive(text)


def stopped(control: GenerationControl | None) -> bool:
    return control is not None and control.should_stop is not None and control.should_stop()


def openvino_streamer(control: GenerationControl, ov_genai, tokenizer) -> tuple[object, Callable[[], bool]]:
    """An openvino_genai streamer for `control`, and a way to ask afterwards
    whether it stopped the answer. `ov_genai` is the `openvino_genai` module
    and `tokenizer` the pipeline's, passed in so this module needs no
    OpenVINO.

    It sees every token, and leaves turning them into text to the runtime's
    own `TextStreamer`. A plain callable would be simpler, but the runtime
    only calls one when a token completes some text: 154 calls for 220
    tokens of HTML, so a rate counted from it read 30% low.

    Measured on the XPS 14 (Qwen2.5-1.5B, 2026-10-03): every token is seen,
    at a rate that matches the runtime's own to the decimal, on the iGPU and
    the NPU; throughput is the same as without it, within run-to-run
    variation (docs/STREAMING.md has the figures); CANCEL returns within a
    second with the text so far, and the pipeline answers its next request
    normally."""
    status = ov_genai.StreamingStatus
    state = {"cancelled": False}

    def on_piece(piece: str):
        if control.on_text is not None:
            control.on_text(piece)
        if control.should_stop is not None and control.should_stop():
            state["cancelled"] = True
            return status.CANCEL
        return status.RUNNING

    class Streamer(ov_genai.StreamerBase):
        def __init__(self) -> None:
            super().__init__()
            self._text = ov_genai.TextStreamer(tokenizer, on_piece)

        def write(self, token):  # one token id, or a list of them
            if control.on_tokens is not None:
                control.on_tokens(len(token) if isinstance(token, (list, tuple)) else 1)
            return self._text.write(token)

        def end(self) -> None:
            self._text.end()

    return Streamer(), lambda: state["cancelled"]


@dataclass
class GenerationStats:
    """How fast a model produced an answer, so a panel can show the audience
    the number rather than assert it. `tokens_per_second` is the rate once
    text is flowing; `first_token_seconds` the wait before it starts (None
    where the backend can't tell); `device` is where it actually ran."""

    device: str
    tokens: int
    seconds: float
    tokens_per_second: float
    first_token_seconds: float | None = None
    # Filled in by the launcher, which owns the energy meter (BACKLOG R18):
    # package joules over the answer and the part above the idle baseline.
    energy: dict | None = None
    # The answer was stopped part-way: these numbers describe an incomplete one.
    cancelled: bool = False

    @classmethod
    def from_openvino(cls, result, device: str, cancelled: bool = False) -> GenerationStats | None:
        """From an openvino_genai LLM or VLM result, whose `perf_metrics`
        time the generation inside the runtime (durations in ms). None if
        they can't be read: the numbers are for show, and must never cost
        the caller the answer itself."""
        try:
            metrics = result.perf_metrics
            return cls(
                device=device,
                tokens=int(metrics.get_num_generated_tokens()),
                seconds=round(metrics.get_generate_duration().mean / 1000, 3),
                tokens_per_second=round(metrics.get_throughput().mean, 1),
                first_token_seconds=round(metrics.get_ttft().mean / 1000, 3),
                cancelled=cancelled,
            )
        except Exception:
            return None


def combine_stats(parts: list[GenerationStats]) -> GenerationStats | None:
    """Several generations reported as one (a commit message, then review
    notes): tokens and time add up, the rate is weighted by tokens, and the
    first token is the first call's."""
    if not parts:
        return None
    tokens = sum(p.tokens for p in parts)
    rate = sum(p.tokens_per_second * p.tokens for p in parts) / tokens if tokens else 0.0
    return GenerationStats(
        device=parts[0].device,
        tokens=tokens,
        seconds=round(sum(p.seconds for p in parts), 3),
        tokens_per_second=round(rate, 1),
        first_token_seconds=parts[0].first_token_seconds,
        cancelled=any(p.cancelled for p in parts),
    )
