"""Turns a text prompt or a folder of documents into a self-contained HTML
page using doc-qa's local LLM.

This brick deliberately has no LLM code of its own -- see
code-review-assist/session.py for the pattern this mirrors. It asks for
the same coding-specialized model code-review-assist does: generating
clean HTML/CSS is a code-generation task, not a document-Q&A task, so
doc-qa's own small general-purpose default is the wrong tool here too.
"""
from __future__ import annotations

from doc_qa.engine_factory import create_llm
from pantherlake_ai_core.engine import Engine

from . import folder_input
from .html_cleanup import strip_code_fence
from .types import HtmlResult

_DEFAULT_OPENVINO_REPO = "OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov"
_DEFAULT_PORTABLE_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
# Higher than code-review-assist's 8192: a full HTML page's output alone can
# run 2000-5000 tokens, on top of a capped document input plus the prompt.
_PORTABLE_N_CTX = 16384

_MAX_TOKENS_BY_MODE = {
    # 4096 was tried first and empirically wasn't always enough -- a richly
    # detailed prompt produced a page that got cut off mid-section.
    "landing_page": 6144,
    "document": 3072,
}

_HTML_RULES = (
    "Output ONE complete, self-contained HTML document: start with "
    "<!DOCTYPE html>, include <meta charset=\"utf-8\"> and a responsive "
    "viewport meta tag, and put all styling in a single inline <style> "
    "block in the <head>. Do not link to any external stylesheet, font, "
    "CDN script, or image -- everything must work from this one file with "
    "no network access. If you need any JavaScript, put it inline in a "
    "<script> tag at the end of <body>. Output raw HTML only: no markdown "
    "code fences, no commentary before or after the document."
)

_LANDING_PAGE_SYSTEM_PROMPT = (
    "You are a web designer who writes clean, modern, self-contained HTML "
    "landing pages. Given a short description of a page, design and write "
    "one. Invent plausible, on-topic copy, section headings, and layout for "
    "whatever the description asks for -- a real landing page needs real-"
    "looking content, not placeholder text. " + _HTML_RULES
)

_DOCUMENT_SUMMARY_SYSTEM_PROMPT = (
    "You turn a set of documents into a well-organized HTML summary report "
    "-- a clean, readable replacement for a PDF handout. Given the "
    "documents' text below (each preceded by its filename), write a "
    "summary with clear headings and, where useful, lists or a table. "
    "Only use what's actually in the documents -- don't invent facts, "
    "figures, or sources that aren't there. If a source's content is worth "
    "attributing, name the file it came from. " + _HTML_RULES
)


class HtmlCreatorSession:
    def __init__(self, engine: Engine, *, compute_device: str):
        self.engine = engine
        self.compute_device = compute_device
        self._llm = None  # built lazily -- no reason to load it before there's a prompt/folder

    def generate(
        self,
        *,
        mode: str = "landing_page",
        prompt: str | None = None,
        folder: str | None = None,
        max_tokens: int | None = None,
    ) -> HtmlResult:
        if mode == "landing_page":
            if not prompt or not prompt.strip():
                raise ValueError("Provide a prompt describing the page to generate.")
            system_prompt = _LANDING_PAGE_SYSTEM_PROMPT
            source_text = prompt.strip()
            truncated = False
        elif mode == "document":
            if not folder:
                raise ValueError("Provide a folder of documents to summarize.")
            raw = folder_input.read_documents(folder)
            if not raw.strip():
                raise RuntimeError(f"No supported documents found in '{folder}'.")
            system_prompt = _DOCUMENT_SUMMARY_SYSTEM_PROMPT
            source_text, truncated = folder_input.truncate_documents(raw)
        else:
            raise ValueError(f"Unknown mode '{mode}' (expected 'landing_page' or 'document').")

        if self._llm is None:
            if self.engine == Engine.OPENVINO:
                self._llm = create_llm(self.engine, device=self.compute_device, model_repo=_DEFAULT_OPENVINO_REPO)
            else:
                self._llm = create_llm(
                    self.engine, model_repo=_DEFAULT_PORTABLE_REPO, n_ctx=_PORTABLE_N_CTX
                )

        tokens = max_tokens or _MAX_TOKENS_BY_MODE[mode]
        raw_output = self._llm.answer(system_prompt, source_text, max_tokens=tokens)
        html, fence_stripped = strip_code_fence(raw_output)

        return HtmlResult(
            html=html,
            mode=mode,
            source_char_count=len(source_text) if mode == "landing_page" else len(raw),
            source_truncated=truncated,
            fence_stripped=fence_stripped,
            html_truncated=not html.rstrip().lower().endswith("</html>"),
        )
