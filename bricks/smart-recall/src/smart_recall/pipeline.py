"""Continuous screen capture -> OCR -> embed -> index loop, plus the
search side that reads it back. Shared by the CLI and the launcher.

Two threads, one queue -- the same concurrent-pipeline shape
`expense-extract` uses, just driven by a timer instead of a finite file
list: OCR (pinned to one device) and embedding (pinned to another) run
at the same time, so an embedding call in progress never stalls the next
capture's OCR.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
from doc_qa.engine_factory import create_embedder
from doc_qa.store import VectorStore
from doc_qa.types import Chunk, RetrievedChunk
from pantherlake_ai_core import video
from pantherlake_ai_core.engine import Engine
from screen_ocr.pipeline import OcrSession

from .change_detection import frame_changed

_CACHE_ROOT = Path.home() / ".cache" / "pantherlake-ai-studio" / "smart-recall"
SCREENSHOTS_DIR = _CACHE_ROOT / "screenshots"
INDEX_DIR = _CACHE_ROOT / "index"
_INDEX_META_PATH = INDEX_DIR / "meta.json"

# Skip indexing near-empty OCR results (e.g. a blank desktop, a loading
# screen) -- there's nothing meaningful to search for there.
MIN_TEXT_LENGTH = 20

# Small enough to bound memory, large enough that a faster OCR stage can
# get ahead of a slower embedding stage instead of stalling on every capture.
_QUEUE_SIZE = 3


def _load_store() -> VectorStore:
    return VectorStore.load(INDEX_DIR) or VectorStore()


def _load_index_meta() -> dict | None:
    if _INDEX_META_PATH.exists():
        return json.loads(_INDEX_META_PATH.read_text(encoding="utf-8"))
    return None


def _save_index_meta(embed_engine: str) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _INDEX_META_PATH.write_text(json.dumps({"embed_engine": embed_engine}), encoding="utf-8")


def index_status() -> dict:
    """How many chunks are indexed and which embedding engine they're
    locked to (`None` if nothing's been recorded yet) -- cheap, no model
    load, for a status display that doesn't need to actually search."""
    meta = _load_index_meta()
    return {"indexed_count": _load_store().size, "embed_engine": meta["embed_engine"] if meta else None}


def reset_index() -> None:
    """Wipes the index, its metadata, and every saved screenshot -- for
    starting over (e.g. switching embedding engines; see the module
    docstring's note on why that can't just be mixed in)."""
    import shutil

    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    shutil.rmtree(SCREENSHOTS_DIR, ignore_errors=True)


class RecallIndex:
    """The search side: loads the persisted index and *only* the embedder
    needed to embed a query -- no OCR, no screen capture. The index's own
    embedding engine (recorded in meta.json at first capture) is used
    regardless of what's passed in, since a query embedded by a different
    model than the one that indexed the chunks would produce meaningless
    similarity scores -- vectors from two different embedding spaces
    aren't comparable, mixing them isn't a quality tradeoff, it's just
    wrong. Only the compute *device* is actually a free choice here.
    """

    def __init__(self, *, device: str = "AUTO"):
        meta = _load_index_meta()
        if meta is None:
            raise RuntimeError("Nothing has been recorded yet -- run `smart-recall record` first.")
        self.embed_engine = Engine(meta["embed_engine"])
        self.embedder = create_embedder(self.embed_engine, device=device)
        self.store = _load_store()

    def search(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.store.size == 0:
            return []
        query_vector = self.embedder.embed_query(question)
        return self.store.search(query_vector, top_k=top_k)

    @staticmethod
    def screenshot_path(chunk: Chunk) -> Path:
        return SCREENSHOTS_DIR / chunk.source


@dataclass
class CaptureEvent:
    kind: str  # "indexed" | "skipped"
    reason: str | None = None
    chunk: Chunk | None = None
    timestamp: str = ""


def run(
    *,
    screen_index: int,
    interval_seconds: float,
    ocr_engine: Engine,
    ocr_device: str,
    embed_engine: Engine,
    embed_device: str,
    on_event: Callable[[CaptureEvent], None] = lambda event: None,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread until `stop_event` is set. Captures the
    screen every `interval_seconds`, skips frames that haven't visibly
    changed, OCRs and embeds the rest, and appends them to the persisted
    index -- picking up where a previous recording session left off."""
    existing_meta = _load_index_meta()
    if existing_meta is not None and existing_meta["embed_engine"] != embed_engine.value:
        raise ValueError(
            f"The existing index was built with the '{existing_meta['embed_engine']}' embedding engine. "
            f"Mixing in '{embed_engine.value}' vectors would make search results meaningless (different "
            "embedding spaces). Pass the matching engine, or call reset_index() / `smart-recall record --reset` "
            "to start over."
        )
    if existing_meta is None:
        _save_index_meta(embed_engine.value)

    ocr_session = OcrSession(ocr_engine, device=ocr_device)
    embedder = create_embedder(embed_engine, device=embed_device)
    store = _load_store()
    next_chunk_index = store.size

    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    handoff: queue.Queue = queue.Queue(maxsize=_QUEUE_SIZE)
    _DONE = object()

    def capture_and_ocr_worker() -> None:
        previous_frame = None
        try:
            while stop_event is None or not stop_event.is_set():
                frame = video.capture_screen_frame(screen_index)
                if not frame_changed(previous_frame, frame):
                    on_event(CaptureEvent(kind="skipped", reason="no change since last capture"))
                else:
                    previous_frame = frame
                    text = ocr_session.extract(frame).text
                    if text and len(text.strip()) >= MIN_TEXT_LENGTH:
                        handoff.put((frame, text))
                    else:
                        on_event(CaptureEvent(kind="skipped", reason="no readable text"))

                waited = 0.0
                while waited < interval_seconds:
                    if stop_event is not None and stop_event.is_set():
                        break
                    time.sleep(0.1)
                    waited += 0.1
        finally:
            handoff.put(_DONE)

    def embed_and_index_worker() -> None:
        nonlocal next_chunk_index
        while True:
            item = handoff.get()
            if item is _DONE:
                return
            frame, text = item

            # Per-item, not per-thread: a bad frame (or a callback that
            # chokes on OCR'd text -- e.g. a Windows console crashing on
            # non-ASCII characters, found via testing, not hypothetical)
            # must not take the whole indexing thread down. If it did, the
            # OCR thread would keep enqueueing behind a now-unconsumed
            # bounded queue and deadlock the whole pipeline once it fills,
            # since nothing else drains it.
            try:
                timestamp = datetime.now()
                filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{next_chunk_index}.jpg"
                cv2.imwrite(str(SCREENSHOTS_DIR / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

                vector = embedder.embed_documents([text])[0]
                chunk = Chunk(text=text, source=filename, chunk_index=next_chunk_index)
                next_chunk_index += 1
                store.add([chunk], [vector])
                store.save(INDEX_DIR)

                on_event(CaptureEvent(kind="indexed", chunk=chunk, timestamp=timestamp.isoformat(timespec="seconds")))
            except Exception as exc:
                on_event(CaptureEvent(kind="skipped", reason=f"indexing failed: {exc}"))

    ocr_thread = threading.Thread(target=capture_and_ocr_worker, daemon=True)
    embed_thread = threading.Thread(target=embed_and_index_worker, daemon=True)
    ocr_thread.start()
    embed_thread.start()
    ocr_thread.join()
    embed_thread.join()
