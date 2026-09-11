"""Command-line entry point for batch receipt-to-CSV extraction."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys

from pantherlake_ai_core import engine as engine_mod

from . import pipeline
from .types import totals_by_currency


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
    # Optional at parse time only so --list-devices works on its own; main()
    # enforces it for an actual run.
    p.add_argument("folder", nargs="?", default=None, help="Folder of receipt image files (.png/.jpg/.jpeg/.bmp/.tif/.webp).")
    p.add_argument("--output", default="expenses.csv", help="Output CSV path. Default: expenses.csv")
    p.add_argument(
        "--ocr-engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Backend for the OCR stage. Default: openvino if installed and a device is available, "
             "otherwise portable.",
    )
    p.add_argument(
        "--ocr-device", default=None,
        help="openvino OCR engine only: AUTO, CPU, GPU, or NPU. Default: AUTO.",
    )
    p.add_argument(
        "--llm-engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Backend for the LLM structuring stage. Default: openvino if installed and a device is "
             "available, otherwise portable.",
    )
    p.add_argument(
        "--llm-device", default=None,
        help="openvino LLM engine only: AUTO, CPU, GPU, or NPU. Default: AUTO.",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available inference devices, then exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        engine_mod.print_devices(inference_flag="--ocr-device / --llm-device")
        return 0

    if not args.folder:
        parser.error("the following arguments are required: folder")

    ocr_engine = engine_mod.resolve_engine(args.ocr_engine)
    llm_engine = engine_mod.resolve_engine(args.llm_engine)
    ocr_device = args.ocr_device or engine_mod.default_device(ocr_engine)
    llm_device = args.llm_device or engine_mod.default_device(llm_engine)

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
                  f"{line.currency or 'Unknown currency'} {line.amount if line.amount is not None else '?'} -- {line.category}"
                  + (f" -- NEEDS REVIEW: {'; '.join(line.review_reasons)}" if line.needs_review else ""))

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
        writer.writerow(["source_file", "vendor", "date", "amount", "currency", "category", "needs_review", "review_reasons", "error"])
        for line in results:
            writer.writerow([line.source_file, line.vendor, line.date, line.amount, line.currency or "", line.category,
                             line.needs_review, "; ".join(line.review_reasons), line.error or ""])

    ok = [r for r in results if r.error is None]
    print(f"\nWrote {args.output}: {len(results)} receipt(s), {len(ok)} structured, "
          f"{sum(r.needs_review for r in results)} need review.")
    print("Validated-field totals (verify against originals):")
    for currency, amount in totals_by_currency(results).items():
        print(f"  {currency}: {amount}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
