from decimal import Decimal
import json

import pytest

from expense_extract.parsing import coerce_amount, currency_from_text
from expense_extract.pipeline import _structure
from expense_extract.types import totals_by_currency


@pytest.mark.parametrize("text,expected", [
    ("12,50 EUR", "12.50"), ("1 234,56 EUR", "1234.56"),
    ("1\u202f234,56 EUR", "1234.56"), ("1,234.56 USD", "1234.56"),
    ("1.234,56 EUR", "1234.56"), ("-12,50 EUR", "-12.50"),
    ("(12.50)", "-12.50"), (0.1, "0.1"),
])
def test_money_preserves_magnitude(text, expected):
    assert coerce_amount(text) == Decimal(expected)


@pytest.mark.parametrize("text", ["1,234", "1.234", "12,3,4", "12 34", "--12", "12 USD garbage",
                                      "NaN", "Infinity", float("nan"), float("inf"), True, "1e5", "(-12.50)"])
def test_ambiguous_and_invalid_money_requires_review(text):
    assert coerce_amount(text) is None


def structure(amount="12,50", currency="EUR", date="2026-09-10", raw=None):
    class FakeLLM:
        def answer(self, *args, **kwargs):
            return json.dumps(dict(vendor="Cafe", date=date, amount=amount, currency=currency, category="Meals"))
    return _structure(FakeLLM(), raw or f"Cafe\n2026-09-10\nTotal {amount} {currency}", "receipt.png")


def test_totals_use_decimal_and_never_mix_currencies_or_review_items():
    lines = [structure("0.10"), structure("0.20"), structure("2.00", "USD"),
             structure("99.00", "USD", raw="Cafe total $99.00"), structure("1,234")]
    assert totals_by_currency(lines) == {"EUR": "0.30", "USD": "2.00"}
    assert lines[-2].needs_review and lines[-1].needs_review
    assert json.loads(json.dumps(lines[0].to_dict()))["amount"] == "0.10"


def test_currency_is_not_invented_from_dollar_sign():
    assert currency_from_text("total $12.50") is None
    assert currency_from_text("EUR 12 / USD 13") is None
    assert currency_from_text("total €12,50") == "EUR"


def test_bad_date_and_ungrounded_amount_are_flagged():
    assert structure(date="2026-02-30").needs_review
    assert structure(amount="999.00", raw="Cafe\nTotal 12.50 EUR").needs_review
    assert structure(amount="12.50", currency="JPY").needs_review
    assert not structure(amount="-12.50").needs_review
