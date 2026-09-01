"""Turns a git diff into a commit message and review notes using doc-qa's
local LLM.

This brick deliberately has no LLM code of its own: it imports
`doc_qa.engine_factory` for the model, the same way meeting-notes composes
live-translation and doc-qa rather than re-implementing a third Whisper or
llama.cpp wrapper. Unlike doc-qa's own default (a small general-purpose
chat model), this brick asks for a coding-specialized model via
`create_llm`'s `model_repo` override -- code review needs different
strengths than document Q&A.
"""
from __future__ import annotations

from typing import Callable

from doc_qa.engine_factory import create_llm
from pantherlake_ai_core.engine import Engine

from . import diff
from .types import ReviewResult

# Mixture-of-experts coding model (30B total params, ~3B active per token):
# a general-purpose small model is noticeably weaker at code review than at
# document Q&A, but a large dense coding model would be slow token-by-token.
# MoE gives large-model code knowledge at small-model decode cost. Verified
# loadable via openvino_genai.LLMPipeline on this machine's Arc B60 -- see
# README.md for the verification run and the dense fallback if a future
# openvino_genai version regresses qwen3_moe support.
_DEFAULT_OPENVINO_REPO = "OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov"
_DEFAULT_PORTABLE_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
_PORTABLE_N_CTX = 8192  # a real diff + prompt routinely exceeds doc-qa's 4096-token default

_COMMIT_MESSAGE_SYSTEM_PROMPT = (
    "You write git commit messages. Given a diff, write ONE conventional-commit-style "
    "message: a short imperative summary line (max ~72 chars), optionally followed by a "
    "blank line and a few bullet points of detail if the diff is non-trivial. Only "
    "describe what the diff actually changes -- don't invent context, files, or intent "
    "that isn't visible in it. Output only the commit message, nothing else."
)

_REVIEW_NOTES_SYSTEM_PROMPT = (
    "You are a careful code reviewer. Given a diff, list concrete issues: bugs, edge "
    "cases, security problems, missing tests, or style inconsistencies with the "
    "surrounding code. One bullet per issue, referencing the specific line or change. "
    "Only flag what the diff actually shows -- don't guess at code you can't see. If you "
    "find nothing worth flagging, say so plainly in one line instead of inventing filler."
)


class CodeReviewSession:
    def __init__(self, engine: Engine, *, compute_device: str):
        self.engine = engine
        self.compute_device = compute_device
        self._llm = None  # built lazily -- no reason to load it before a diff is ready

    def review(
        self,
        *,
        folder: str | None = None,
        against: str = "HEAD",
        diff_text: str | None = None,
        max_tokens: int = 400,
        on_ready: Callable[[], None] | None = None,
        on_downloading: Callable[[], None] | None = None,
    ) -> ReviewResult:
        """`on_ready`/`on_downloading`, if given: the LLM is lazy (built on
        the first `review()` call, reused after) so a caller wanting to
        distinguish "building the model" from "actually reviewing" needs a
        seam here rather than around `__init__`, which does no loading."""
        raw = diff.read_diff(folder=folder, against=against, diff_text=diff_text)
        if not raw.strip():
            raise RuntimeError(f"No changes to review against '{against}'.")
        text, truncated = diff.truncate_diff(raw)

        if self._llm is None:
            if self.engine == Engine.OPENVINO:
                self._llm = create_llm(
                    self.engine,
                    device=self.compute_device,
                    model_repo=_DEFAULT_OPENVINO_REPO,
                    on_downloading=on_downloading,
                )
            else:
                self._llm = create_llm(
                    self.engine, model_repo=_DEFAULT_PORTABLE_REPO, n_ctx=_PORTABLE_N_CTX
                )
        if on_ready is not None:
            on_ready()

        commit_message = self._llm.answer(_COMMIT_MESSAGE_SYSTEM_PROMPT, text, max_tokens=120).strip()
        review_notes = self._llm.answer(_REVIEW_NOTES_SYSTEM_PROMPT, text, max_tokens=max_tokens).strip()
        return ReviewResult(
            commit_message=commit_message,
            review_notes=review_notes,
            diff_char_count=len(raw),
            diff_truncated=truncated,
        )
