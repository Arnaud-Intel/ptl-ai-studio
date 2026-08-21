"""Turns the LLM's reply into a plain dict, tolerating the ways a small
local model deviates from "respond with only JSON" -- wrapping it in a
sentence, fencing it in ```json, trailing commentary after the closing
brace. Small models do all three, not hypothetically -- this is what
survived testing against this brick's own default model.
"""
from __future__ import annotations

import json
import re

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_expense_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    return None


def coerce_amount(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9.\-]", "", value)
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None
    return None
