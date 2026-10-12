"""One model, loaded once on one chip the way its brick loads it, so that
what the chip compiles lands in the compiled-model cache -- then gone.

    python -m launcher.compile_worker MODEL_KEY DEVICE

A process of its own, started by `compile_ahead`, and for three reasons. A
compile that goes wrong can end its process without a word (the NPU's
compiler did, on a model it does not take), and that must not be the
Studio. What a model held is only surely given back when its process ends.
And a process that has loaded several models in a row has been seen to spin
in the runtime's own teardown instead of exiting (page-agent's README), so
this one says what it did and leaves outright.

The last line it prints is its report: `{"ok": true, "seconds": 12.3}` or
`{"ok": false, "error": "..."}`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable

from pantherlake_ai_core import models


def _whisper(size: str) -> Callable[[str], object]:
    def load(device: str):
        from live_translation.transcriber import create_translator
        from pantherlake_ai_core.engine import Engine

        return create_translator(Engine.OPENVINO, size, device)

    return load


def _language_model(key: str) -> Callable[[str], object]:
    def load(device: str):
        from doc_qa.llm_openvino import OpenVINOLLM

        return OpenVINOLLM(device=device, model_repo=models.BY_KEY[key].repo_id)

    return load


def _embedder(device: str):
    from doc_qa.embedder_openvino import OpenVINOEmbedder

    return OpenVINOEmbedder(device=device)


def _vision(device: str):
    from screen_ocr.extractor_openvino import OpenVINOExtractor

    return OpenVINOExtractor(device=device)


def _detector(device: str):
    from object_detection.detector_openvino import OpenVINODetector

    return OpenVINODetector(device=device)


def _image_model(device: str):
    # Two pipelines over the same files: the one that draws (the Page Agent)
    # and the one that draws a photo again (Portrait Studio). The second
    # adds little to what the first compiled.
    from page_agent.images import ImageMaker
    from portrait_studio.stylist import Stylist

    return ImageMaker(device=device), Stylist(device=device)


# How each OpenVINO model of the inventory is loaded: through its brick's
# own class, so that what is compiled here is what the brick asks for later
# (the NPU's window for a language model is part of what is compiled).
# tests/test_compile_ahead.py holds this against the inventory.
LOADERS: dict[str, Callable[[str], object]] = {
    "whisper-medium-ov": _whisper("medium"),
    "whisper-base-ov": _whisper("base"),
    "llm-1.5b-ov": _language_model("llm-1.5b-ov"),
    "notes-8b-ov": _language_model("notes-8b-ov"),
    "notes-8b-npu-ov": _language_model("notes-8b-npu-ov"),
    "embed-ov": _embedder,
    "vlm-7b-ov": _vision,
    "yolo11s-ov": _detector,
    "coder-30b-ov": _language_model("coder-30b-ov"),
    "flux2-klein-ov": _image_model,
}


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2 or arguments[0] not in LOADERS:
        print(json.dumps({"ok": False, "error": f"usage: compile_worker MODEL_KEY DEVICE; models: {', '.join(LOADERS)}"}))
        return 2
    key, device = arguments
    began = time.perf_counter()
    try:
        LOADERS[key](device)
    except Exception as exc:  # the report is the point: what went wrong, in a line
        report = {"ok": False, "error": " ".join(f"{type(exc).__name__}: {exc}".split())[:400]}
    else:
        report = {"ok": True, "seconds": round(time.perf_counter() - began, 1)}
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # not sys.exit: see the top of this file
