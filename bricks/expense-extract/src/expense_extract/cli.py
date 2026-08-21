"""Command-line entry point for batch receipt-to-CSV extraction."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys

from pantherlake_ai_core import engine as engine_mod

from . import pipeline

_OCR_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}
_LLM_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "AUTO"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="expense-extract",
        description=(
            "Batch-convert a folder of receipt photos into a CSV of structured expense lines. "
            "OCR and LLM structuring run concurrently on two independently chosen devices -- "
            "e.g. NPU for OCR while the GPU structures the previous receipt -- not one after "
            "the other."
        ),
    )
    p.add_argument("folder", help="Folder of receipt image files (.png/.jpg/.jpeg/.bmp/.tif/.webp).")
    p.add_argument("--output", default="expenses.csv", help="Output CSV path. Default: expenses.csv")
    p.add_argument(
        "--ocr-engine", choices=[e.value for e in engine_mod.Engine], default=engine_mod.Engine.PORTABLE.value,
        help="Backend for the OCR stage. Default: portable",
    )
    p.add_argument("--ocr-device", default=None, help="openvino OCR engine only: AUTO, CPU, GPU, or NPU.")
    p.add_argument(
        "--llm-engine", choices=[e.value for e in engine_mod.Engine], default=engine_mod.Engine.PORTABLE.value,
        help="Backend for the LLM structuring stage. Default: portable",
    )
    p.add_argument("--llm-device", default=None, help="openvino LLM engine only: AUTO, CPU, GPU, or NPU.")
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available inference devices, then exit.",
    )
    return p


def list_devices() -> None:
    from pantherlake_ai_core.engine import describe_devices

    print("Inference devices (--ocr-device / --llm-device):")
    print(describe_devices())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    ocr_engine = engine_mod.Engine(args.ocr_engine)
    llm_engine = engine_mod.Engine(args.llm_engine)
    ocr_device = args.ocr_device or _OCR_ENGINE_DEFAULTS[ocr_engine]["device"]
    llm_device = args.llm_device or _LLM_ENGINE_DEFAULTS[llm_engine]["device"]

    print(
        f"OCR stage: engine={ocr_engine.value}, device={ocr_device}\n"
        f"LLM stage: engine={llm_engine.value}, device={llm_device}\n"
        "Both stages run concurrently on separate threads once processing starts -- watch the "
        "timestamps below overlap, not queue up.\n"
    )

    def handle_ocr_start(path, index, total):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] OCR   {index}/{total}: {path.name}")

    def handle_structured(line):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        if line.error:
            print(f"[{ts}] LLM   {line.source_file}: SKIPPED ({line.error})")
        else:
            print(f"[{ts}] LLM   {line.source_file}: {line.vendor or '?'} -- {line.date or '?'} -- "
                  f"${line.amount if line.amount is not None else '?'} -- {line.category}")

    try:
        results = pipeline.run(
            folder=args.folder,
            ocr_engine=ocr_engine,
            ocr_device=ocr_device,
            llm_engine=llm_engine,
            llm_device=llm_device,
            on_ocr_start=handle_ocr_start,
            on_structured=handle_structured,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_file", "vendor", "date", "amount", "category", "error"])
        for line in results:
            writer.writerow([line.source_file, line.vendor, line.date, line.amount, line.category, line.error or ""])

    ok = [r for r in results if r.error is None]
    total_amount = sum(r.amount for r in ok if r.amount is not None)
    print(f"\nWrote {args.output}: {len(results)} receipt(s), {len(ok)} structured cleanly, total ${total_amount:.2f}.")

    by_category: dict[str, float] = {}
    for r in ok:
        if r.amount is not None:
            by_category[r.category] = by_category.get(r.category, 0.0) + r.amount
    if by_category:
        print("By category:")
        for category, amount in sorted(by_category.items(), key=lambda kv: -kv[1]):
            print(f"  {category}: ${amount:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
