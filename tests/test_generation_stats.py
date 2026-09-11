"""How fast an answer came, carried from the runtime to the panel (BACKLOG
R20): the stats type, and the two bricks that report it for their 30B
model."""
from __future__ import annotations

from types import SimpleNamespace

from code_review_assist import session as code_review_session
from code_review_assist.session import CodeReviewSession
from html_creator import session as html_session
from html_creator.session import HtmlCreatorSession
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import GenerationStats, combine_stats


def _mean(value):
    return SimpleNamespace(mean=value)


class _PerfMetrics:
    """openvino_genai's PerfMetrics shape: durations in ms, rates per second."""

    def get_num_generated_tokens(self):
        return 256

    def get_generate_duration(self):
        return _mean(7100.0)

    def get_throughput(self):
        return _mean(38.46)

    def get_ttft(self):
        return _mean(390.0)


def test_openvino_metrics_become_seconds_and_tokens_per_second():
    stats = GenerationStats.from_openvino(SimpleNamespace(perf_metrics=_PerfMetrics()), "GPU.0")
    assert stats == GenerationStats(
        device="GPU.0", tokens=256, seconds=7.1, tokens_per_second=38.5, first_token_seconds=0.39
    )


def test_unreadable_metrics_never_cost_the_answer():
    assert GenerationStats.from_openvino(SimpleNamespace(), "GPU.0") is None


def test_combined_stats_weight_the_rate_by_tokens():
    a = GenerationStats("GPU.0", tokens=100, seconds=3.0, tokens_per_second=40.0, first_token_seconds=0.3)
    b = GenerationStats("GPU.0", tokens=300, seconds=9.0, tokens_per_second=36.0, first_token_seconds=0.5)
    both = combine_stats([a, b])
    assert (both.tokens, both.seconds, both.tokens_per_second, both.first_token_seconds) == (400, 12.0, 37.0, 0.3)
    assert combine_stats([]) is None


class _FakeLLM:
    def __init__(self, rates):
        self.rates = list(rates)
        self.last_stats = None

    def answer(self, system_prompt, user_prompt, max_tokens=512):
        rate = self.rates.pop(0)
        self.last_stats = GenerationStats(
            "GPU.0", tokens=100, seconds=100 / rate, tokens_per_second=rate, first_token_seconds=0.4
        )
        return "<!DOCTYPE html><html></html>" if "HTML" in system_prompt else "text"


def test_code_review_reports_both_answers_as_one(monkeypatch):
    monkeypatch.setattr(code_review_session, "create_llm", lambda *a, **k: _FakeLLM([40.0, 30.0]))
    result = CodeReviewSession(Engine.OPENVINO, compute_device="GPU.0").review(diff_text="diff --git a/x b/x\n+1\n")
    assert (result.stats.tokens, result.stats.tokens_per_second, result.stats.device) == (200, 35.0, "GPU.0")


def test_html_creator_reports_its_answer(monkeypatch):
    monkeypatch.setattr(html_session, "create_llm", lambda *a, **k: _FakeLLM([38.0]))
    result = HtmlCreatorSession(Engine.OPENVINO, compute_device="GPU.0").generate(prompt="a bakery")
    assert result.stats.tokens_per_second == 38.0
