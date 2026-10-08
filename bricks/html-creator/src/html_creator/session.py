"""Turns a text prompt or a folder of documents into a self-contained HTML
page using doc-qa's local LLM.

This brick deliberately has no LLM code of its own -- see
code-review-assist/session.py for the pattern this mirrors. It asks for
the same coding-specialized model code-review-assist does: generating
clean HTML/CSS is a code-generation task, not a document-Q&A task, so
doc-qa's own small general-purpose default is the wrong tool here too.
"""
from __future__ import annotations

import gc
from typing import Callable

from doc_qa.engine_factory import create_llm
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import GenerationControl

from . import folder_input
from . import pictures as picture_kit
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
    "<script> tag at the end of <body>. The page is shown in a sandboxed "
    "frame: keep state in JavaScript variables, and do not use "
    "localStorage, sessionStorage, cookies, alert(), confirm() or prompt(). "
    "Output raw HTML only: no markdown "
    "code fences, no commentary before or after the document."
)

# Added when the request comes with pictures: the one exception to "no
# image files", and the reason it is safe -- they are embedded afterwards.
_PICTURE_RULES = (
    " The request ends with a PICTURES list. Those files are the only "
    "images that exist: use them wherever the page calls for a picture, "
    "each by its exact file name with nothing in front of it (they are "
    "embedded into the page afterwards). Give every <img> a meaningful alt "
    "text, and never reference any other image file or URL."
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
        pictures: str | None = None,
        picture_list: list[picture_kit.Picture] | None = None,
        before_embed: Callable[[], None] | None = None,
        closing: str | None = None,
        repeatable: bool = False,
        max_tokens: int | None = None,
        on_ready: Callable[[], None] | None = None,
        on_downloading: Callable[[], None] | None = None,
        control: GenerationControl | None = None,
    ) -> HtmlResult:
        """`on_ready`/`on_downloading`, if given: the LLM is lazy (built on
        the first `generate()` call, reused after) so a caller wanting to
        distinguish "building the model" from "actually generating" needs a
        seam here rather than around `__init__`, which does no loading.

        `pictures`, for a landing page, is a folder of images the page may
        place: the model is told their names and captions, and the ones it
        references are embedded into the page.

        `picture_list` is the same thing for pictures already described --
        name, size, caption, and the path each will be at. The page is
        written from the descriptions alone, so the files need not exist
        until it is; `before_embed`, if given, is called once the page is
        written and returns when they do. That is what lets another model
        draw the pictures on another chip while this one writes the page
        (see the page-agent brick). `closing`, if given, is a last line of
        the request, placed after the list of pictures: the last thing the
        model reads before it starts to write.

        `repeatable` makes the page a function of its prompt alone, by
        loading the model afresh and taking its most likely next token each
        time. Both are needed, one per engine. On the GPU the runtime
        specialises itself to the requests it has served and its arithmetic
        shifts with it: measured on the XPS 14, the same prompt gave one
        page on a fresh model, another after a different prompt and a third
        straight afterwards -- with or without sampling, whose draw is
        seeded -- while four fresh models gave the same page four times.
        llama.cpp seeds its draw at random, so there it is the sampling
        that has to go. The price is the model's load time on every page.
        Off by default."""
        offered: list[picture_kit.Picture] = []
        picture_notes: list[str] = []
        if mode == "landing_page":
            if not prompt or not prompt.strip():
                raise ValueError("Provide a prompt describing the page to generate.")
            system_prompt = _LANDING_PAGE_SYSTEM_PROMPT
            source_text = prompt.strip()
            truncated = False
            if picture_list:
                offered = list(picture_list)
            elif pictures and pictures.strip():
                offered, picture_notes = picture_kit.load(pictures.strip())
            if offered:
                system_prompt += _PICTURE_RULES
                source_text += "\n\n" + picture_kit.manifest(offered)
            if closing and closing.strip():
                source_text += "\n\n" + closing.strip()
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

        if repeatable and self._llm is not None:
            # Released before the next one loads: two 30B models do not fit
            # side by side.
            self._llm = None
            gc.collect()

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

        tokens = max_tokens or _MAX_TOKENS_BY_MODE[mode]
        raw_output = self._llm.answer(
            system_prompt, source_text, max_tokens=tokens, control=control, sample=not repeatable
        )
        html, fence_stripped = strip_code_fence(raw_output)
        truncated_output = not html.rstrip().lower().endswith("</html>")
        written = html
        if before_embed is not None:
            before_embed()
        html, used = picture_kit.embed(html, offered)

        return HtmlResult(
            html=html,
            mode=mode,
            source_char_count=len(source_text) if mode == "landing_page" else len(raw),
            source_truncated=truncated,
            fence_stripped=fence_stripped,
            html_truncated=truncated_output,
            stats=self._llm.last_stats,
            pictures_offered=len(offered),
            pictures_used=used,
            picture_notes=picture_notes,
            html_source=written if used else None,
            repeatable=repeatable,
        )
