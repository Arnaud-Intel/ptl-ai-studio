"""Small deterministic helpers that guard model output and CLI input."""
from __future__ import annotations

import argparse

import pytest
from code_review_assist.diff import MAX_DIFF_CHARS, truncate_diff
from expense_extract.parsing import coerce_amount, parse_expense_json
from html_creator.html_cleanup import strip_code_fence
from webcam_effects.cli import _parse_color


def test_expense_json_plain():
    assert parse_expense_json('{"vendor": "Cafe", "amount": 4.5}') == {"vendor": "Cafe", "amount": 4.5}


def test_expense_json_survives_a_markdown_fence():
    assert parse_expense_json('```json\n{"vendor": "Cafe"}\n```') == {"vendor": "Cafe"}


def test_expense_json_survives_surrounding_prose():
    assert parse_expense_json('Sure! Here it is: {"vendor": "Cafe"} Let me know.') == {"vendor": "Cafe"}


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", "{not json}"])
def test_expense_json_rejects_what_it_cannot_parse(text):
    assert parse_expense_json(text) is None


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), (12, 12.0), (4.5, 4.5), ("$12.50", 12.5), ("12.50 USD", 12.5), ("abc", None), ("", None), ([1], None)],
)
def test_coerce_amount(value, expected):
    assert coerce_amount(value) == expected


def test_truncate_diff():
    assert truncate_diff("short") == ("short", False)
    truncated, was_truncated = truncate_diff("x" * (MAX_DIFF_CHARS + 5))
    assert was_truncated and len(truncated) == MAX_DIFF_CHARS


def test_strip_code_fence():
    assert strip_code_fence("```html\n<p>hi</p>\n```") == ("<p>hi</p>", True)
    assert strip_code_fence("```html\n<p>cut off by max_tokens") == ("<p>cut off by max_tokens", True)
    assert strip_code_fence("  <p>plain</p>\n") == ("<p>plain</p>", False)


def test_parse_color_is_rgb_in_and_bgr_out():
    assert _parse_color("0,104,181") == (181, 104, 0)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_color("blue")
