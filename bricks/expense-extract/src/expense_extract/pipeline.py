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
from doc_qa.engine_factory import create_llm
from pantherlake_ai_core.engine import Engine
from screen_ocr.pipeline import OcrSession

from .parsing import CURRENCIES, coerce_amount, currency_from_text, parse_expense_json
from .types import ExpenseLine

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
    "A bare $ does not establish USD. If the text is not a receipt, return null fields."
)

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

    reply = llm.answer(_SYSTEM_PROMPT, raw_text, max_tokens=200)
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
    elif not any(coerce_amount(token) == amount for token in re.findall(r"-?\d[\d., \u00a0\u202f]*\d|-?\d", raw_text)):
        reasons.append("Amount could not be matched to the receipt text")
    date_text = str(parsed.get("date") or "")
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_text):
            raise ValueError
        date.fromisoformat(date_text)
    except ValueError:
        reasons.append("Date is missing or invalid")
        date_text = ""
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
    on_ocr_start: Callable[[Path, int, int], None] = lambda path, index, total: None,
    on_structured: Callable[[ExpenseLine], None] = lambda line: None,
    stop_event: threading.Event | None = None,
) -> list[ExpenseLine]:
    """Blocks the calling thread until every image is processed (or
    `stop_event` is set). Returns the collected results in completion
    order (which, because the two stages run concurrently, is not
    necessarily the same order the images were listed in)."""
    images = list_receipt_images(folder)
    if not images:
        raise ValueError(f"No receipt images found under {folder} ({', '.join(sorted(SUPPORTED_SUFFIXES))})")

    ocr_session = OcrSession(ocr_engine, device=ocr_device)
    llm = create_llm(llm_engine, device=llm_device)

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
