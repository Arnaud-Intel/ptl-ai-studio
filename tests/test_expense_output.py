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


def test_an_amount_that_is_nowhere_on_the_receipt_is_not_shown():
    """Asked for a total too faded to have been read, the small model wrote
    one (324.00 for a hotel folio whose amounts are blank, 2026-10-10). The
    line used to keep it, flagged: an invented number in front of whoever
    reads the line. It keeps none now, and says why."""
    folio = "ATELIER QUAY HOTEL\nDate: 13/09/2026\n2 nuits x 180,00 EUR\nTOTAL TTC EUR\nMontant regle EUR\nSolde EUR"
    line = structure(amount="324.00", date="2026-09-13", raw=folio)
    assert line.amount is None and line.needs_review
    assert line.review_reasons == ["Amount could not be matched to the receipt text"]
    assert totals_by_currency([line]) == {}
    assert line.to_dict()["amount"] is None


@pytest.mark.parametrize("receipt,said,flagged", [
    # The fare where the total was asked for: on the receipt, and not what was paid.
    ("Rive Transfer\nDate: 2026-09-03\nDemo station\nEUR 38,00\nEvening supplement\nEUR 8,80\nTOTAL PAID\nEUR 46,80", "38,00", True),
    ("Rive Transfer\nDate: 2026-09-03\nDemo station\nEUR 38,00\nEvening supplement\nEUR 8,80\nTOTAL PAID\nEUR 46,80", "46,80", False),
    ("Cafe\n2026-09-10\nSOUS-TOTAL 35,30\nRemise -3,00\nTOTAL EUR 32,30\nCB 32,30", "35,30", True),  # the sub-total is not the total
    ("Cafe\n2026-09-10\nSOUS-TOTAL 35,30\nRemise -3,00\nTOTAL EUR 32,30\nCB 32,30", "32,30", False),
    ("Workbench\n2026-09-05\nReturned cable\nEUR -18,00\nTOTAL REFUND\nEUR -18,00", "-18,00", False),  # a refund's total is negative
    # Nothing to go by: two totals that differ, a total of another kind, a total that could not be read.
    ("Cafe\n2026-09-10\nTotal HT 10,00 EUR\nTotal TTC 12,00 EUR", "10,00", False),
    ("Cafe\n2026-09-10\nTotal items: 3\nA payer 12,00 EUR", "12,00", False),
    ("Hotel\n2026-09-10\nChambre 12,00 EUR\nTOTAL TTC EUR\nMontant regle EUR", "12,00", False),
])
def test_an_amount_that_is_not_the_printed_total_is_for_a_person(receipt, said, flagged):
    """Qwen3-8B on the NPU gave 38.00 for a transfer whose total line says
    46,80 -- a figure that is on the receipt, so nothing flagged it. The
    line a receipt opens with "TOTAL" settles it when it carries one amount
    of money. The model's figure is kept beside the flag, to be compared."""
    line = structure(amount=said, raw=receipt)
    assert ("Amount is not the printed total" in line.review_reasons) == flagged
    assert line.amount is not None and line.needs_review == flagged


def test_a_date_that_reads_two_ways_is_for_a_person_when_no_currency_says_which():
    """Day first everywhere but on a US receipt, and the currency is what
    tells them apart. "$24.00" is not a currency: 04/09/2026 is then the 4th
    of September or the 9th of April, and nobody at this desk knows which."""
    kiosk = structure(amount="24.00", currency=None, date="2026-09-04", raw="Harbor Kiosk\n04/09/2026\nTOTAL PAID\n$24.00")
    assert "Date can be read two ways" in kiosk.review_reasons and kiosk.date == "2026-09-04"
    assert "Date can be read two ways" not in structure(amount="24,00", raw="Kiosk\n04/09/2026\nTOTAL 24,00 EUR").review_reasons
    assert "Date can be read two ways" not in structure(
        amount="24.00", currency=None, date="2026-09-25", raw="Kiosk\n25/09/2026\nTOTAL PAID\n$24.00").review_reasons
