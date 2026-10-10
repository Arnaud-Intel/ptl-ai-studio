"""Ingest a folder of documents and answer questions about them -- shared
by the CLI and any UI front-end (e.g. the launcher) so this logic lives in
one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from pantherlake_ai_core.engine import Engine

from . import documents
from .engine_factory import TEMPLATE_TOKENS, create_embedder, create_llm
from .store import VectorStore, cache_dir_for
from .types import Answer

# Measured on the sample folder with six questions, two of which the
# documents cannot answer, Qwen2.5-1.5B (2026-10-09). The prompt before this
# one ended with "Refer to which excerpt(s) you used by their [number]", and
# the whole answer to four of the six was "[1]". This one got a sentence
# every time, the right facts for the four that have an answer, and "not
# mentioned" for the two that have none -- where a version without the last
# clause answered "the chief executive's home address" with his name.
_SYSTEM_PROMPT = (
    "You answer a question about the user's own documents, using ONLY the excerpts you are given. "
    "Answer in full sentences and say which file each fact comes from. "
    "If the excerpts do not contain what is asked, say that the documents do not say, and stop: "
    "a name, a figure or a date that is not written in the excerpts must not appear in your answer."
)


# What is answered, without asking the model, when no passage of the folder
# is close to the question. A small model handed four passages beside the
# point does not say "I don't know": asked for a pilot's budget with only a
# travel policy to read, it answered with the hotel rate.
NOTHING_CLOSE = (
    "The documents do not say. No passage in them is close to this question, "
    "so no answer was written from them."
)


# What the model is told when it is asked without the documents: the same
# model, the same question, and nothing to read. It is there to show what
# the documents change. Measured on the sample folder (2026-10-09, the 1.5B
# model on the NPU): of three questions about the fictional company it said
# of two that it did not have the information, and of the third -- who
# decides on the Lyon pilot, and when -- that it was "the team's captain",
# "on September 16, 2023". Nothing here asks it to invent, or not to.
ALONE_PROMPT = "Answer the question in two or three sentences. If you do not have the information, say so."


class DocQASession:
    """Holds one loaded embedder + LLM + index. Ingest once, ask many times."""

    def __init__(
        self,
        engine: Engine,
        *,
        device: str = "AUTO",
        model_dir: str | None = None,
        on_downloading: Callable[[], None] | None = None,
    ):
        self.engine = engine
        self.embedder = create_embedder(engine, device=device, model_dir=model_dir, on_downloading=on_downloading)
        self.llm = create_llm(engine, device=device, model_dir=model_dir, on_downloading=on_downloading)
        self.store = VectorStore()
        self.folder: Path | None = None

    def ingest(
        self,
        folder: str | Path,
        *,
        chunk_size: int = 900,
        overlap: int = 150,
        force: bool = False,
    ) -> int:
        """Build (or load a cached) index for `folder`. Returns the chunk count.

        The cached index is used only if the folder still holds exactly
        what it was built from. It used to be kept by the folder's path
        alone: take a file out, index again, and the answers went on
        quoting it -- "it says it read three files and I gave it two"
        (reported 2026-10-09)."""
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Not a folder: {folder}")

        # Vectors made another way are vectors of another index: how the
        # embedder reads a text is part of what an index is kept under.
        model_key = f"{self.engine.value}:{getattr(self.embedder, 'identity', '')}:{chunk_size}:{overlap}"
        cache_dir = cache_dir_for(folder, model_key)
        holds = documents.fingerprint(folder)

        if not force:
            cached = VectorStore.load(cache_dir, fingerprint=holds)
            if cached is not None:
                self.store = cached
                self.folder = folder
                return self.store.size

        # From here on the session answers about this folder or about
        # nothing: an indexing that fails must not leave the last folder's
        # documents answering under this one's name.
        self.store = VectorStore()
        self.folder = None
        chunks = documents.build_chunks(folder, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            suffixes = ", ".join(sorted(documents.SUPPORTED_SUFFIXES))
            raise ValueError(f"No supported documents ({suffixes}) found under {folder}")

        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.build(chunks, vectors)
        self.store.save(cache_dir, fingerprint=holds)
        self.folder = folder
        return self.store.size

    def ask_alone(self, question: str, *, max_tokens: int = 160, control=None) -> Answer:
        """The question put to the language model with no document in the
        conversation: what it answers from what it learned, which is not
        the user's files. No index is needed, and none is touched."""
        text = self.llm.answer(ALONE_PROMPT, question, max_tokens=max_tokens, control=control)
        return Answer(text=text, stats=getattr(self.llm, "last_stats", None))

    def ask(self, question: str, *, top_k: int = 4, max_tokens: int = 512, control=None) -> Answer:
        if self.store.size == 0:
            raise RuntimeError("No documents ingested yet -- call ingest() first.")

        query_vector = self.embedder.embed_query(question)
        retrieved = self.store.search(query_vector, top_k=top_k)
        # An embedder that knows how its scores read says how alike a passage
        # must be to be worth reading (see embedder_openvino).
        floor = getattr(self.embedder, "relevance_floor", None)
        if floor is not None:
            retrieved = [hit for hit in retrieved if hit.score >= floor]
            if not retrieved:
                return Answer(text=NOTHING_CLOSE)
        used = fit_excerpts(self.llm, retrieved, question, max_tokens)
        text = self.llm.answer(_SYSTEM_PROMPT, _user_prompt(used, question), max_tokens=max_tokens, control=control)
        # Only what the model actually saw: the files listed under an answer
        # are the ones it was given, not the ones the search first returned.
        return Answer(text=text, sources=used, stats=getattr(self.llm, "last_stats", None))


def _user_prompt(retrieved: list, question: str) -> str:
    # Named by their file, not numbered: "[3]" in an answer read as "a third
    # file" to somebody who had given two.
    excerpts = "\n\n".join(f"Excerpt from the file {r.chunk.source}:\n{r.chunk.text}" for r in retrieved)
    return f"{excerpts}\n\nQuestion: {question}"


def fit_excerpts(llm, retrieved: list, question: str, max_tokens: int) -> list:
    """The best-ranked excerpts that fit the model's window on this device.

    Dropping the weakest match beats refusing the question: the NPU runs an
    LLM compiled for a fixed prompt length, and a few long passages can pass
    it. The best one is always kept -- if even that doesn't fit, the model
    says so plainly (PromptTooLong)."""
    budget = llm.prompt_budget(max_tokens) - llm.count_tokens(_SYSTEM_PROMPT) - TEMPLATE_TOKENS
    used = list(retrieved)
    while len(used) > 1 and llm.count_tokens(_user_prompt(used, question)) > budget:
        used.pop()
    return used
