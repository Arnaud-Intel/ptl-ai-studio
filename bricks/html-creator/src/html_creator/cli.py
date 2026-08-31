"""Command-line entry point for the HTML creator."""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

from pantherlake_ai_core import engine as engine_mod

from .session import HtmlCreatorSession

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    # GPU.1 is this dev machine's Arc B60 card id, not a portable default the
    # way "AUTO" is for every other brick -- override with --compute-device
    # on a machine without that exact device.
    engine_mod.Engine.OPENVINO: {"device": "GPU.1"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="html-creator",
        description="Locally generate a self-contained HTML page from a prompt or a folder of documents.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", default=None, help="Describe a landing page to generate.")
    source.add_argument("--folder", default=None, help="Summarize every document in this folder instead.")
    p.add_argument("--out", default=None, help="Write the generated HTML to this file instead of stdout.")
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
    mode = "landing_page" if args.prompt else "document"

    print(f"Loading model (engine={engine.value}, device={compute_device})... this may download a model on first use.")
    session = HtmlCreatorSession(engine, compute_device=compute_device)

    try:
        result = session.generate(mode=mode, prompt=args.prompt, folder=args.folder)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result.source_truncated:
        print(f"(source was {result.source_char_count} characters -- truncated before generation)", file=sys.stderr)
    if result.html_truncated:
        print("(warning: output doesn't end with </html> -- it may have been cut off)", file=sys.stderr)
    if result.fence_stripped:
        print("(note: stripped a markdown code fence the model wrapped its output in)", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(result.html, encoding="utf-8")
        print(f"Wrote {len(result.html)} characters to {args.out}")
    else:
        print(result.html)

    return 0


if __name__ == "__main__":
    sys.exit(main())
