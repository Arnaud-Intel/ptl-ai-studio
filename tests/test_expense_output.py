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
    assert structure(date="2026-02-30", raw="Cafe\n2026-02-30\nTotal 12,50 EUR").needs_review
    assert structure(amount="999.00", raw="Cafe\nTotal 12.50 EUR").needs_review
    assert structure(amount="12.50", currency="JPY").needs_review
    assert not structure(amount="-12.50").needs_review


def test_the_model_is_not_shown_the_buyer_and_so_cannot_take_it_for_the_vendor():
    seen = []

    class Reads:
        def answer(self, system, text, **kwargs):
            seen.append(text)
            return json.dumps(dict(vendor="Rive Transfer", date="2026-09-03", amount="46,80", currency="EUR", category="Travel"))

    receipt = ("Rive Transfer\nTRANSFER RECEIPT\nDate: 2026-09-03\nBilled to: Meridian Robotics / Alex Morgan\n"
               "Client : Alex Demo\nFacturé à: Meridian\nCustomer service: 0800 000\nTOTAL PAID\nEUR 46,80")
    line = _structure(Reads(), receipt, "05.png")
    assert "Meridian" not in seen[0] and "Alex Demo" not in seen[0]
    assert "Rive Transfer" in seen[0] and "Customer service: 0800 000" in seen[0]  # a label, not any line that says customer
    assert line.raw_text == receipt and not line.needs_review  # what is kept for the person who checks is the whole receipt


@pytest.mark.parametrize("said,printed,currency,expected", [
    (None, "Date: 2026-09-03", "EUR", "2026-09-03"),  # the model gave none: the one printed date is read by rule
    ("2026-12-09", "12/09/2026 13:24", "EUR", "2026-09-12"),  # day first, where the model read a US date
    ("2026-09-12", "12/09/2026 13:24", "EUR", "2026-09-12"),
    (None, "09/12/2026", "USD", "2026-09-12"),  # a US receipt
    (None, "25.12.2026", "USD", "2026-12-25"),  # only one way round is a date
    ("2026-09-13", "Sejour: 11/09/2026 - 13/09/2026", "EUR", "2026-09-13"),  # two dates: the model's choice stands
    ("2026-09-04", "Valid until 2027-01-01", "EUR", "2026-09-04"),  # a date that is not a swap of the printed one stands
])
def test_a_date_printed_once_is_read_by_rule_when_the_model_gave_none_or_swapped_it(said, printed, currency, expected):
    line = structure(date=said, currency=currency, raw=f"Cafe\n{printed}\nTotal 12,50 {currency}")
    assert line.date == expected and not line.needs_review


def test_no_date_at_all_or_several_and_none_from_the_model_is_for_a_person():
    assert structure(date=None, raw="Cafe\nTotal 12,50 EUR").needs_review
    assert structure(date=None, raw="Cafe\n11/09/2026 - 13/09/2026\nTotal 12,50 EUR").needs_review
