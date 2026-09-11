"""Shared result types used across demo bricks."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranslationResult:
    text: str
    detected_language: str
    language_probability: float


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

    @classmethod
    def from_openvino(cls, result, device: str) -> GenerationStats | None:
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
    )
