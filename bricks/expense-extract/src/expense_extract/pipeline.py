"""Batch receipt -> structured expense pipeline, genuinely two-stage and
concurrent: OCR (the screen-ocr brick) and LLM structuring (the doc-qa
brick's LLM) run on separate threads connected by a small bounded queue,
each pinned to its own device -- so while the LLM is structuring receipt
N, OCR is already reading receipt N+1. Both devices are visibly busy at
the same time, not one after the other; that overlap is the entire point
of this brick, not an implementation detail. Shared by the CLI and the
launcher.
"""
from __future__ import annotations

import queue
import threading
from datetime import date
import re
from pathlib import Path
from typing import Callable

import cv2
from doc_qa import language_models
from doc_qa.engine_factory import create_llm
from pantherlake_ai_core.engine import Engine
from screen_ocr.pipeline import OcrSession

from .parsing import CURRENCIES, coerce_amount, currency_from_text, parse_expense_json
from .types import ExpenseLine

# The language model that makes a line of what was read (doc-qa's language_models).
DEFAULT_MODEL = language_models.DEFAULT

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_SYSTEM_PROMPT = (
    "You extract structured expense data from OCR'd receipt text, which "
    "may include OCR errors or misread characters. Respond with ONLY a "
    "single JSON object -- no other words, no markdown fencing -- with "
    'exactly these keys: "vendor" (string), "date" (string, "YYYY-MM-DD" '
    'if explicit, otherwise null), "amount" (string preserving the printed separators, the final '
    'total paid, or null), "currency" (explicit ISO currency code, or null if ambiguous), '
    '"category" (one of: "Meals", "Travel", "Lodging", '
    '"Office Supplies", "Software", "Other"). Never guess missing fields. '
    "A bare $ does not establish USD. If the text is not a receipt, return null fields. "
    # What follows was measured on the fourteen sample receipts that have a
    # checked answer, with Qwen2.5-1.5B on the NPU (2026-10-09). Without it:
    # vendor 5 of 14, category 7 of 14. With it, and with the two rules
    # below: vendor 13, date 14, category 12. Said last and in plain
    # sentences: the same hints inside the list of keys cost three categories.
    "The vendor is the business that issued the receipt, named at its top. "
    "Categories: a cafe, a restaurant, food or drink is Meals; a flight, a train, a taxi, a transfer or "
    "parking is Travel; a hotel stay is Lodging; physical goods such as equipment, hardware, cables, "
    "adapters or stationery are Office Supplies, and so is a refund of them; "
    "Software is only for a subscription, a licence or an online service."
)

# A receipt names two parties, and the small model took the one after a label
# for the vendor on every sample receipt that has one ("Billed to: Meridian
# Robotics": nine of nine, measured 2026-10-09). An expense line has no use
# for the buyer, so the model is not shown that line.
_BUYER_LINE = re.compile(
    r"^\s*(billed to|bill to|invoiced? to|sold to|ship(ped)? to|deliver(ed)? to|customer|client|guest|attn|attention"
    r"|factur[ée]e? [àa]|adress[ée]e? [àa]|livr[ée]e? [àa])\b\s*[:\-]",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_NUMERIC_DATE = re.compile(r"(?<![\d/.\-])(\d{1,2})([/.\-])(\d{1,2})\2(\d{4})(?!\d)")


def _for_the_model(raw_text: str) -> str:
    return "\n".join(line for line in raw_text.splitlines() if not _BUYER_LINE.match(line))


def _the_one_printed_date(raw_text: str, currency: str | None = None) -> str:
    """The receipt's date when it prints exactly one, in figures
    ("2026-09-03", "12/09/2026"): read by rule, since the model returned
    none for four of the nine sample receipts that say "Date: 2026-09-0x",
    and read "12/09/2026" on a French till receipt as the 9th of December.
    Day first, as everywhere but on a US receipt, unless only the other way
    round is a date. Two different dates (a hotel stay) are a question for
    the model, or for a person."""
    found = set()
    for match in _ISO_DATE.findall(raw_text):
        try:
            date.fromisoformat(match)
        except ValueError:
            continue
        found.add(match)
    for first, _separator, second, year in _NUMERIC_DATE.findall(raw_text):
        first, second = int(first), int(second)
        month_first = second > 12 >= first or (currency == "USD" and first <= 12 and second <= 12)
        day, month = (second, first) if month_first else (first, second)
        try:
            found.add(date(int(year), month, day).isoformat())
        except ValueError:
            continue
    return found.pop() if len(found) == 1 else ""


def _day_and_month_swapped(one: str, other: str) -> bool:
    return one[:4] == other[:4] and one[5:7] == other[8:10] and one[8:10] == other[5:7] and one != other


def _a_date_reads_two_ways(raw_text: str) -> bool:
    """Whether the receipt prints a date in figures that is one day read
    day first and another read month first ("04/09/2026")."""
    return any(
        int(first) <= 12 and int(second) <= 12 and first != second
        for first, _separator, second, _year in _NUMERIC_DATE.findall(raw_text)
    )


_AMOUNT_TOKEN = r"-?\d[\d.,   ]*\d|-?\d"
# A line that opens on the word: "TOTAL PAID", "Total EUR 32,30", "TOTAL
# REFUND". A sub-total opens on something else ("SOUS-TOTAL", "Subtotal").
_TOTAL_LINE = re.compile(r"^\s*total\b(?P<rest>.*)$", re.IGNORECASE)
_MONEY_MARK = re.compile(r"[€£$¥]|\b(" + "|".join(sorted(CURRENCIES)) + r")\b", re.IGNORECASE)


def _the_printed_total(raw_text: str):
    """The amount printed on the receipt's total line -- beside the word, or
    on the line under it -- when the receipt has one such amount. None when
    it has none that can be read (a faded total) or several that differ."""
    lines = raw_text.splitlines()
    found = set()
    for index, line in enumerate(lines):
        match = _TOTAL_LINE.match(line)
        if not match:
            continue
        below = next((later for later in lines[index + 1:] if later.strip()), "")
        for place in (match.group("rest"), below):
            # Money, not any figure: "Total items: 3" is a total of another kind.
            priced = bool(_MONEY_MARK.search(place))
            amounts = [
                coerce_amount(token) for token in re.findall(_AMOUNT_TOKEN, place)
                if priced or re.search(r"[.,]\d{2}$", token)
            ]
            amounts = [amount for amount in amounts if amount is not None]
            if amounts:
                found.add(amounts[-1])
                break
    return found.pop() if len(found) == 1 else None


# Small enough to bound memory, large enough that a faster OCR stage can
# get ahead of a slower LLM stage instead of stalling on every item.
_QUEUE_SIZE = 3


def list_receipt_images(folder: str | Path) -> list[Path]:
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)


def _structure(llm, raw_text: str, source_name: str) -> ExpenseLine:
    if not raw_text or not raw_text.strip():
        return ExpenseLine(
            source_file=source_name, vendor="", date="", amount=None, category="Other",
            raw_text=raw_text or "", error="No text detected by OCR",
        )

    reply = llm.answer(_SYSTEM_PROMPT, _for_the_model(raw_text), max_tokens=200)
    parsed = parse_expense_json(reply)
    if parsed is None:
        return ExpenseLine(
            source_file=source_name, vendor="", date="", amount=None, category="Other",
            raw_text=raw_text, error="Could not parse a structured response from the LLM",
        )

    amount = coerce_amount(parsed.get("amount"))
    currency = str(parsed.get("currency") or "").upper().strip() or None
    evidence_currency = currency_from_text(raw_text)
    reasons = []
    if currency not in CURRENCIES or currency != evidence_currency:
        currency = evidence_currency
    if currency is None:
        reasons.append("Currency is missing, unsupported or ambiguous")
    if amount is None:
        reasons.append("Amount is missing or ambiguous")
    elif currency in {"JPY", "KRW"} and amount != amount.to_integral_value():
        reasons.append("Fractional amount for a zero-decimal currency")
    elif not any(coerce_amount(token) == amount for token in re.findall(_AMOUNT_TOKEN, raw_text)):
        # A figure that is nowhere on the receipt is the model's own: asked
        # for a total too faded to have been read, it wrote one (324.00 for a
        # hotel folio whose three amounts are blank). It used to be kept with
        # this flag beside it, which still put an invented number in front of
        # whoever reads the line. None is shown; the reason says why.
        reasons.append("Amount could not be matched to the receipt text")
        amount = None
    else:
        # On the receipt, and still not what was paid: the fare instead of the
        # fare with its supplement (Qwen3-8B on the NPU, 2026-10-10: 38.00 for
        # a transfer whose total line says 46,80, with nothing to say so).
        # The line a receipt opens with "TOTAL" settles it when it carries
        # one amount; the model's figure stays, for a person to compare.
        printed_total = _the_printed_total(raw_text)
        if printed_total is not None and printed_total != amount:
            reasons.append("Amount is not the printed total")
    date_text = str(parsed.get("date") or "")
    printed = _the_one_printed_date(raw_text, currency)
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
            raise ValueError
        date.fromisoformat(date_text)
        if printed and _day_and_month_swapped(date_text, printed):
            date_text = printed
    except ValueError:
        date_text = printed
        if not date_text:
            reasons.append("Date is missing or invalid")
    # Day first is the rule everywhere but on a US receipt, and the currency
    # is what tells one from the other. With no currency established there is
    # nothing to tell them apart by: the date stands, and is for a person.
    if date_text and currency is None and _a_date_reads_two_ways(raw_text):
        reasons.append("Date can be read two ways")
    if not parsed.get("vendor"):
        reasons.append("Vendor is missing")
    category = str(parsed.get("category") or "Other")
    if category not in {"Meals", "Travel", "Lodging", "Office Supplies", "Software", "Other"}:
        reasons.append("Category is invalid")
        category = "Other"
    return ExpenseLine(
        source_file=source_name,
        vendor=str(parsed.get("vendor") or ""),
        date=date_text,
        amount=amount,
        currency=currency,
        review_reasons=reasons,
        category=category,
        raw_text=raw_text,
    )


def run(
    *,
    folder: str,
    ocr_engine: Engine,
    ocr_device: str,
    llm_engine: Engine,
    llm_device: str,
    llm_model: str | None = None,
    on_ocr_start: Callable[[Path, int, int], None] = lambda path, index, total: None,
    on_structured: Callable[[ExpenseLine], None] = lambda line: None,
    on_llm_device: Callable[[str], None] = lambda device: None,
    stop_event: threading.Event | None = None,
) -> list[ExpenseLine]:
    """Blocks the calling thread until every image is processed (or
    `stop_event` is set). Returns the collected results in completion
    order (which, because the two stages run concurrently, is not
    necessarily the same order the images were listed in).

    `on_llm_device(device)` fires if the structuring model ends up on a
    chip other than `llm_device`: the NPU can be reset under a running
    model, and the model then carries on elsewhere (see core's `npu`
    module) rather than fail every receipt that is left.

    `llm_model`: which language model makes the lines, by its key in
    doc-qa's `language_models` (None for this brick's default)."""
    llm_repo = language_models.repo_for(llm_model or DEFAULT_MODEL, llm_engine, llm_device)
    images = list_receipt_images(folder)
    if not images:
        raise ValueError(f"No receipt images found under {folder} ({', '.join(sorted(SUPPORTED_SUFFIXES))})")

    ocr_session = OcrSession(ocr_engine, device=ocr_device)
    llm = create_llm(llm_engine, device=llm_device, model_repo=llm_repo)

    llm_on = [llm_device]

    def report_llm_device() -> None:
        now_on = getattr(llm, "device", llm_on[0])
        if now_on != llm_on[0]:
            llm_on[0] = now_on
            on_llm_device(now_on)

    report_llm_device()  # already elsewhere if the NPU was lost before this batch
    handoff: queue.Queue = queue.Queue(maxsize=_QUEUE_SIZE)
    results: list[ExpenseLine] = []
    results_lock = threading.Lock()
    _DONE = object()

    def stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    def put_unless_stopped(item) -> bool:
        # Bounded queue: never block forever on put(), or a stop requested
        # while the LLM stage is busy could never be honored.
        while not stopped():
            try:
                handoff.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def ocr_worker() -> None:
        try:
            for index, path in enumerate(images, start=1):
                if stopped():
                    break
                on_ocr_start(path, index, len(images))
                image = cv2.imread(str(path))
                text = ocr_session.extract(image).text if image is not None else ""
                if not put_unless_stopped((path.name, text)):
                    break
        finally:
            handoff.put(_DONE)

    def llm_worker() -> None:
        while True:
            item = handoff.get()
            if item is _DONE:
                return
            if stopped():
                continue  # drain without structuring so the OCR side can finish and stop promptly
            source_name, text = item
            # One bad receipt must not take the structuring thread down: if
            # it died, the OCR thread would block forever on the bounded
            # queue with nothing left to drain it (the same deadlock
            # smart-recall's pipeline guards against).
            try:
                line = _structure(llm, text, source_name)
            except Exception as exc:
                line = ExpenseLine(
                    source_file=source_name, vendor="", date="", amount=None, category="Other",
                    raw_text=text, error=f"Structuring failed: {exc}",
                )
            report_llm_device()
            with results_lock:
                results.append(line)
            on_structured(line)

    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    llm_thread = threading.Thread(target=llm_worker, daemon=True)
    ocr_thread.start()
    llm_thread.start()
    ocr_thread.join()
    llm_thread.join()

    return results
