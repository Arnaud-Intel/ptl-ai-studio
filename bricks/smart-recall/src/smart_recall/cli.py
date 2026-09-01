"""Command-line entry point for local screen memory."""
from __future__ import annotations

import argparse
import sys

from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core import video

from . import pipeline
from .samples import SAMPLES

_OCR_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}
_EMBED_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "AUTO"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smart-recall",
        description="Semantic search over your own screen history -- fully local.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Continuously capture, OCR, and index the screen until Ctrl+C.")
    record.add_argument("--screen-index", type=int, default=1, help="Screen/monitor index. Default: 1")
    record.add_argument(
        "--interval", type=float, default=5.0,
        help="Seconds between capture attempts (a capture is skipped entirely if the screen "
             "hasn't visibly changed). Default: 5.0",
    )
    record.add_argument(
        "--ocr-engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Backend for the OCR stage. Default: openvino if installed and a device is available, "
             "otherwise portable.",
    )
    record.add_argument("--ocr-device", default=None, help="openvino OCR engine only: AUTO, CPU, GPU, or NPU.")
    record.add_argument(
        "--embed-engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Backend for the embedding stage. Fixed for the life of the index -- see --reset. "
             "Default: openvino if installed and a device is available, otherwise portable.",
    )
    record.add_argument("--embed-device", default=None, help="openvino embed engine only: AUTO, CPU, GPU, or NPU.")
    record.add_argument(
        "--reset", action="store_true",
        help="Wipe any existing index and saved screenshots before starting (e.g. to switch --embed-engine).",
    )

    search = sub.add_parser("search", help="Search everything recorded so far.")
    search.add_argument(
        "question", nargs="?", default=None,
        help="What you're looking for, in plain language. Or use --sample.",
    )
    search.add_argument("--sample", default=None, help="Use a named example question instead (see list-samples).")
    search.add_argument("--top-k", type=int, default=5, help="Number of matches to show. Default: 5")
    search.add_argument("--embed-device", default="AUTO", help="AUTO, CPU, GPU, or NPU (openvino index only).")

    sub.add_parser("list-devices", help="List available screens and inference devices, then exit.")
    sub.add_parser("list-samples", help="List available example search questions, then exit.")

    return p


def _list_devices() -> None:
    print("Screens (--screen-index N):")
    for screen in video.list_screens():
        print(f"  - {screen['index']}: {screen['width']}x{screen['height']}")
    print("\nInference devices (--ocr-device / --embed-device):")
    print(engine_mod.describe_devices())


def _run_record(args: argparse.Namespace) -> int:
    if args.reset:
        pipeline.reset_index()
        print("Existing index and screenshots cleared.")

    from pantherlake_ai_core.engine import list_openvino_devices

    default_engine = engine_mod.Engine.OPENVINO if list_openvino_devices() else engine_mod.Engine.PORTABLE
    ocr_engine = engine_mod.Engine(args.ocr_engine) if args.ocr_engine else default_engine
    embed_engine = engine_mod.Engine(args.embed_engine) if args.embed_engine else default_engine
    ocr_device = args.ocr_device or _OCR_ENGINE_DEFAULTS[ocr_engine]["device"]
    embed_device = args.embed_device or _EMBED_ENGINE_DEFAULTS[embed_engine]["device"]

    print(
        f"OCR stage: engine={ocr_engine.value}, device={ocr_device}\n"
        f"Embed stage: engine={embed_engine.value}, device={embed_device}\n"
        f"Capturing screen {args.screen_index} every {args.interval:.0f}s (only when it changes). "
        "Press Ctrl+C to stop.\n"
    )

    def handle_event(event: pipeline.CaptureEvent) -> None:
        if event.kind == "indexed":
            snippet = event.chunk.text.strip().replace("\n", " ")[:80]
            print(f"[{event.timestamp}] indexed: {snippet}...")
        else:
            print(f"  (skipped -- {event.reason})")

    try:
        pipeline.run(
            screen_index=args.screen_index,
            interval_seconds=args.interval,
            ocr_engine=ocr_engine,
            ocr_device=ocr_device,
            embed_engine=embed_engine,
            embed_device=embed_device,
            on_event=handle_event,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


def _run_search(args: argparse.Namespace) -> int:
    if bool(args.question) == bool(args.sample):
        print("Error: provide a question or --sample (not both, not neither).", file=sys.stderr)
        return 1
    if args.sample:
        matches = [s for s in SAMPLES if s.name == args.sample]
        if not matches:
            print(f"Error: no sample named '{args.sample}' (see: smart-recall list-samples)", file=sys.stderr)
            return 1
        args.question = matches[0].question

    try:
        index = pipeline.RecallIndex(device=args.embed_device)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    results = index.search(args.question, top_k=args.top_k)
    if not results:
        print("No matches yet -- has `smart-recall record` indexed anything?")
        return 0

    for r in results:
        snippet = r.chunk.text.strip().replace("\n", " ")[:160]
        screenshot = pipeline.RecallIndex.screenshot_path(r.chunk)
        print(f"[{r.score:.2f}] {r.chunk.source}")
        print(f"  {snippet}")
        print(f"  screenshot: {screenshot}\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    # OCR'd screen text can contain anything -- any script, any symbol --
    # and Windows consoles default to a codepage (e.g. cp1252) that can't
    # encode most of it. Reconfiguring here instead of leaving Python's
    # default means "found this while recording" prints as best-effort
    # text instead of crashing the CLI outright, which is what happened
    # in testing the first time OCR picked up a non-Latin character.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)

    if args.command == "list-devices":
        _list_devices()
        return 0
    if args.command == "list-samples":
        for s in SAMPLES:
            print(f"{s.name}: {s.description}")
        return 0
    if args.command == "record":
        return _run_record(args)
    if args.command == "search":
        return _run_search(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
