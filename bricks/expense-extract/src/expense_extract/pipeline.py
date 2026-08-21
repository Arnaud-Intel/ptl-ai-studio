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
from pathlib import Path
from typing import Callable

import cv2
from doc_qa.engine_factory import create_llm
from pantherlake_ai_core.engine import Engine
from screen_ocr.pipeline import OcrSession

from .parsing import coerce_amount, parse_expense_json
from .types import ExpenseLine

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_SYSTEM_PROMPT = (
    "You extract structured expense data from OCR'd receipt text, which "
    "may include OCR errors or misread characters. Respond with ONLY a "
    "single JSON object -- no other words, no markdown fencing -- with "
    'exactly these keys: "vendor" (string), "date" (string, "YYYY-MM-DD" '
    'if you can tell, else your best guess), "amount" (number, the final '
    'total paid), "category" (one of: "Meals", "Travel", "Lodging", '
    '"Office Supplies", "Software", "Other"). If the text barely looks '
    "like a receipt, still return your best-guess JSON."
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

    return ExpenseLine(
        source_file=source_name,
        vendor=str(parsed.get("vendor") or ""),
        date=str(parsed.get("date") or ""),
        amount=coerce_amount(parsed.get("amount")),
        category=str(parsed.get("category") or "Other"),
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

    def ocr_worker() -> None:
        try:
            for index, path in enumerate(images, start=1):
                if stop_event is not None and stop_event.is_set():
                    break
                on_ocr_start(path, index, len(images))
                image = cv2.imread(str(path))
                text = ocr_session.extract(image).text if image is not None else ""
                handoff.put((path.name, text))
        finally:
            handoff.put(_DONE)

    def llm_worker() -> None:
        while True:
            item = handoff.get()
            if item is _DONE:
                return
            source_name, text = item
            line = _structure(llm, text, source_name)
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
