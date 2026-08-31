"""Command-line entry point for the commit & code review assistant."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pantherlake_ai_core import engine as engine_mod

from .session import CodeReviewSession

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    # GPU.1 is this dev machine's Arc B60 card id, not a portable default the
    # way "AUTO" is for every other brick -- override with --compute-device
    # on a machine without that exact device.
    engine_mod.Engine.OPENVINO: {"device": "GPU.1"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="code-review-assist",
        description="Locally turn a git diff into a commit message and review notes with a local LLM.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", default=None, help="Git working tree to run `git diff` in.")
    source.add_argument("--diff-file", default=None, help="Read a diff from this file instead of running git.")
    p.add_argument(
        "--against", default="HEAD",
        help="Git ref to diff against (only used with --folder). Default: HEAD, i.e. all uncommitted changes.",
    )
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=engine_mod.Engine.PORTABLE.value,
        help="Inference backend: 'portable' (llama.cpp, CPU) or 'openvino' (Intel CPU/iGPU/NPU -- "
             "requires this brick's `openvino` extra). Default: portable",
    )
    p.add_argument("--compute-device", default=None, help="Device for the openvino engine. Ignored for portable.")
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available inference devices, then exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        from pantherlake_ai_core.engine import describe_devices

        print(describe_devices())
        return 0

    engine = engine_mod.Engine(args.engine)
    compute_device = args.compute_device or _ENGINE_DEFAULTS[engine]["device"]

    diff_text = Path(args.diff_file).read_text() if args.diff_file else None

    print(f"Loading model (engine={engine.value}, device={compute_device})... this may download a model on first use.")
    session = CodeReviewSession(engine, compute_device=compute_device)

    try:
        result = session.review(folder=args.folder, against=args.against, diff_text=diff_text)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result.diff_truncated:
        from .diff import MAX_DIFF_CHARS

        print(
            f"(diff was {result.diff_char_count} characters -- truncated to the first "
            f"{MAX_DIFF_CHARS} before review; some changes may not be reflected)\n",
            file=sys.stderr,
        )

    print("Commit message:\n")
    print(result.commit_message)
    print("\nReview notes:\n")
    print(result.review_notes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
