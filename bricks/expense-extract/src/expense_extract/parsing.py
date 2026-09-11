"""Turns the LLM's reply into a plain dict, tolerating the ways a small
local model deviates from "respond with only JSON" -- wrapping it in a
sentence, fencing it in ```json, trailing commentary after the closing
brace. Small models do all three, not hypothetically -- this is what
survived testing against this brick's own default model.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation

# Explicitly supported currencies. Unknown currencies remain reviewable, never USD by default.
CURRENCIES = frozenset("EUR USD GBP CAD AUD CHF CNY SEK NOK DKK PLN CZK INR BRL NZD SGD HKD MXN ZAR ILS TRY JPY KRW".split())
_CODES = re.compile(r"\b(" + "|".join(sorted(CURRENCIES)) + r")\b", re.I)


def currency_from_text(text: str) -> str | None:
    found = {match.upper() for match in _CODES.findall(text)}
    if "€" in text:
        found.add("EUR")
    if "£" in text:
        found.add("GBP")
    # A bare dollar or yen symbol is ambiguous across countries.
    return next(iter(found)) if len(found) == 1 else None

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_expense_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)

    try:
        parsed = json.loads(text, parse_float=Decimal)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0), parse_float=Decimal)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    return None


def coerce_amount(value) -> Decimal | None:
    """Parse unambiguous money; never remove punctuation and change its magnitude."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    text = str(value).strip()
    if isinstance(value, str):
        text = _CODES.sub("", text).strip()
        text = re.sub(r"^[€£$¥]\s*|\s*[€£$¥]$", "", text).strip()
        negative = text.startswith("(") and text.endswith(")")
        if negative:
            text = text[1:-1].strip()
        text = text.replace("\u00a0", " ").replace("\u202f", " ")
        if re.fullmatch(r"-?\d{1,3}(?: \d{3})+(?:[.,]\d{1,2})?", text):
            text = text.replace(" ", "").replace(",", ".")
        elif re.fullmatch(r"-?\d{1,3}(?:,\d{3})+\.\d{1,2}", text):
            text = text.replace(",", "")
        elif re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+,\d{1,2}", text):
            text = text.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"-?\d+(?:[.,]\d{1,2})?", text):
            text = text.replace(",", ".")
        else:
            return None  # e.g. 1,234 could mean either 1234 or 1.234
        if negative:
            if text.startswith("-"):
                return None
            text = "-" + text
    try:
        amount = Decimal(text)
        if not amount.is_finite() or abs(amount) > Decimal("999999999999.99"):
            return None
        return amount if amount == amount.quantize(Decimal("0.01")) else None
    except InvalidOperation:
        return None
