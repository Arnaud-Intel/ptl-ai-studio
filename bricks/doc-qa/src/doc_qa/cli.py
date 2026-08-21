"""Command-line entry point for local document Q&A."""
from __future__ import annotations

import argparse
import sys

from pantherlake_ai_core import engine as engine_mod

from .pipeline import DocQASession

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="doc-qa",
        description="Locally ask questions about your own documents (retrieval-augmented, fully on-device).",
    )
    p.add_argument("folder", help="Folder of .txt/.md/.pdf files to answer questions about.")
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=engine_mod.Engine.PORTABLE.value,
        help="Inference backend: 'portable' (llama.cpp, CPU) or 'openvino' (Intel CPU/iGPU/NPU -- "
             "requires this brick's `openvino` extra). Default: portable",
    )
    p.add_argument(
        "--compute-device", default=None,
        help="Device for the openvino engine (AUTO, CPU, GPU, NPU). Ignored for the portable engine.",
    )
    p.add_argument("--reindex", action="store_true", help="Rebuild the index even if a cached one exists.")
    p.add_argument("--top-k", type=int, default=4, help="Number of source chunks to retrieve per question. Default: 4")
    p.add_argument("--question", default=None, help="Ask a single question and exit, instead of an interactive loop.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = engine_mod.Engine(args.engine)
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
