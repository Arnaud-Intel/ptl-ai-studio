"""Deterministic cleanup of an LLM's raw HTML output.

Small local models sometimes wrap "output raw code" instructions in a
markdown code fence anyway (```html ... ```). Rather than hoping a more
insistent prompt fixes it, strip a leading/trailing fence if present --
same "guard, don't hope" philosophy as code-review-assist's two-separate-
calls decision.
"""
from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n(.*)\n```$", re.DOTALL)
# A generation that got cut off by max_tokens before reaching its closing
# fence still starts with one -- strip just the opening marker in that case
# (confirmed to happen in practice: a truncated response left a literal
# "```html" line in front of the DOCTYPE, which broke the preview).
_LEADING_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n")


def strip_code_fence(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip(), True
    if _LEADING_FENCE_RE.match(stripped):
        return _LEADING_FENCE_RE.sub("", stripped, count=1).strip(), True
    return stripped, False
