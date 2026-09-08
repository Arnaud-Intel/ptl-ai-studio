"""Command-line entry point for the commit & code review assistant."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pantherlake_ai_core import engine as engine_mod

from .samples import SAMPLES
from .session import CodeReviewSession

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    # None: resolved at run time by pantherlake_ai_core.engine's
    # preferred_large_model_device() -- the machine's discrete GPU if it has
    # one (this brick's 30B coding model needs real VRAM), else AUTO. Never a
    # hardcoded card id from one dev machine.
    engine_mod.Engine.OPENVINO: {"device": None},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="code-review-assist",
        description="Locally turn a git diff into a commit message and review notes with a local LLM.",
    )
    # Not required at parse time -- --list-devices/--list-samples need to work
    # standalone too; "exactly one of these unless listing" is checked in main().
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--folder", default=None, help="Git working tree to run `git diff` in.")
    source.add_argument("--diff-file", default=None, help="Read a diff from this file instead of running git.")
    source.add_argument("--sample", default=None, help="Use a named example diff instead (see --list-samples).")
    p.add_argument(
        "--against", default="HEAD",
        help="Git ref to diff against (only used with --folder). Default: HEAD, i.e. all uncommitted changes.",
    )
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Inference backend: 'portable' (llama.cpp, CPU) or 'openvino' (Intel CPU/iGPU/NPU -- "
             "requires this brick's `openvino` extra). Default: openvino if installed and a device "
             "is available, otherwise portable.",
    )
    p.add_argument("--compute-device", default=None, help="Device for the openvino engine. Ignored for portable.")
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available inference devices, then exit.",
    )
    p.add_argument(
        "--list-samples", action="store_true",
        help="List available example diffs, then exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        from pantherlake_ai_core.engine import describe_devices

        print(describe_devices())
        return 0

    if args.list_samples:
        for s in SAMPLES:
            print(f"{s.name}: {s.description}")
        return 0

    if sum(bool(v) for v in (args.folder, args.diff_file, args.sample)) != 1:
        parser.error("exactly one of --folder, --diff-file, --sample is required")

    if args.engine:
        engine = engine_mod.Engine(args.engine)
    else:
        from pantherlake_ai_core.engine import list_openvino_devices

        engine = engine_mod.Engine.OPENVINO if list_openvino_devices() else engine_mod.Engine.PORTABLE
    compute_device = args.compute_device or _ENGINE_DEFAULTS[engine]["device"]
    if compute_device is None:
        from pantherlake_ai_core.engine import preferred_large_model_device

        compute_device = preferred_large_model_device()

    if args.diff_file:
        diff_text = Path(args.diff_file).read_text()
    elif args.sample:
        matches = [s for s in SAMPLES if s.name == args.sample]
        if not matches:
            parser.error(f"no sample named '{args.sample}' (see --list-samples)")
        diff_text = matches[0].diff_text
    else:
        diff_text = None

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
