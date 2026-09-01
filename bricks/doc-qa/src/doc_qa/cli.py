"""Command-line entry point for local document Q&A."""
from __future__ import annotations

import argparse
import sys

from pantherlake_ai_core import engine as engine_mod

from .pipeline import DocQASession
from .samples import SAMPLES

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="doc-qa",
        description="Locally ask questions about your own documents (retrieval-augmented, fully on-device).",
    )
    p.add_argument(
        "folder", nargs="?", default=None,
        help="Folder of .txt/.md/.pdf files to answer questions about. Not needed with --list-samples.",
    )
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Inference backend: 'portable' (llama.cpp, CPU) or 'openvino' (Intel CPU/iGPU/NPU -- "
             "requires this brick's `openvino` extra). Default: openvino if installed and a device "
             "is available, otherwise portable.",
    )
    p.add_argument(
        "--compute-device", default=None,
        help="Device for the openvino engine (AUTO, CPU, GPU, NPU). Ignored for the portable engine.",
    )
    p.add_argument("--reindex", action="store_true", help="Rebuild the index even if a cached one exists.")
    p.add_argument("--top-k", type=int, default=4, help="Number of source chunks to retrieve per question. Default: 4")
    p.add_argument("--question", default=None, help="Ask a single question and exit, instead of an interactive loop.")
    p.add_argument("--sample", default=None, help="Use a named example question instead (see --list-samples).")
    p.add_argument(
        "--list-samples", action="store_true",
        help="List available example questions, then exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_samples:
        for s in SAMPLES:
            print(f"{s.name}: {s.description}")
        return 0

    if args.question and args.sample:
        parser.error("--question and --sample are mutually exclusive")
    if args.sample:
        matches = [s for s in SAMPLES if s.name == args.sample]
        if not matches:
            parser.error(f"no sample named '{args.sample}' (see --list-samples)")
        args.question = matches[0].question
        # A sample carries its own bundled demo folder -- use it unless the
        # caller already gave an explicit folder to ask the same kind of
        # question about their own documents instead.
        args.folder = args.folder or matches[0].folder

    if not args.folder:
        parser.error("folder is required (unless using --list-samples, or --sample with its bundled folder)")

    if args.engine:
        engine = engine_mod.Engine(args.engine)
    else:
        from pantherlake_ai_core.engine import list_openvino_devices

        engine = engine_mod.Engine.OPENVINO if list_openvino_devices() else engine_mod.Engine.PORTABLE
    defaults = _ENGINE_DEFAULTS[engine]
    device = args.compute_device or defaults["device"]

    session = DocQASession(engine, device=device)

    print(f"Indexing '{args.folder}' (engine={engine.value})... this downloads models on first use.")
    count = session.ingest(args.folder, force=args.reindex)
    print(f"Indexed {count} chunk(s) from {session.folder}.\n")

    def ask_and_print(question: str) -> None:
        answer = session.ask(question, top_k=args.top_k)
        print(f"\n{answer.text}\n")
        print("Sources:")
        for r in answer.sources:
            print(f"  [{r.score:.2f}] {r.chunk.source} (chunk {r.chunk.chunk_index})")
        print()

    if args.question:
        ask_and_print(args.question)
        return 0

    print("Ask a question (Ctrl+C or empty line to quit).\n")
    try:
        while True:
            question = input("> ").strip()
            if not question:
                break
            ask_and_print(question)
    except KeyboardInterrupt:
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
