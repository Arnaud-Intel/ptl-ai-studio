"""Command-line entry point for live meeting notes."""
from __future__ import annotations

import argparse
import sys

from pantherlake_ai_core import engine as engine_mod

from .session import MeetingSession

# Whisper size per engine -- see live-translation's CLI for why.
_WHISPER_SIZE_DEFAULTS = {engine_mod.Engine.PORTABLE: "small", engine_mod.Engine.OPENVINO: "base"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="meeting-notes",
        description="Locally transcribe a meeting/call and generate running notes + action items with a local LLM.",
    )
    p.add_argument(
        "--source", choices=["mic", "system"], default="system",
        help="Audio source. Default: system (the call/video itself, not just your mic).",
    )
    p.add_argument(
        "--audio-device", default=None,
        help="Substring to match a specific microphone/output device name.",
    )
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Inference backend for BOTH transcription (faster-whisper/OpenVINO Whisper, via the "
             "live-translation brick) and notes generation (llama.cpp/OpenVINO LLM, via the doc-qa "
             "brick): 'portable' or 'openvino' (requires this brick's `openvino` extra). Default: "
             "openvino if installed and a device is available, otherwise portable.",
    )
    p.add_argument(
        "--compute-device", default=None,
        help="openvino engine only: AUTO, CPU, GPU, or NPU. Default: AUTO (cpu for portable).",
    )
    p.add_argument("--whisper-model", default=None, help="Whisper model size override (tiny/base/small/medium/large-v3).")
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available microphones, output devices, and inference devices, then exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        engine_mod.print_devices(mics=True, speakers=True)
        return 0

    engine = engine_mod.resolve_engine(args.engine)
    compute_device = args.compute_device or engine_mod.default_device(engine)
    whisper_model = args.whisper_model or _WHISPER_SIZE_DEFAULTS[engine]

    print(
        f"Loading transcription engine (engine={engine.value}, device={compute_device})... "
        "this may download models on first use."
    )
    session = MeetingSession(engine, compute_device=compute_device, whisper_model_size=whisper_model)

    label = "microphone" if args.source == "mic" else "system audio (loopback)"
    print(f"Listening on {label}. Press Ctrl+C to stop and generate notes.\n")

    def handle_line(line) -> None:
        print(f"[{line.timestamp}] ({line.detected_language}) {line.text}")

    try:
        session.transcribe(source=args.source, audio_device=args.audio_device, on_line=handle_line)
    except KeyboardInterrupt:
        print("\nStopped. Generating notes...\n")

    try:
        notes = session.generate_notes()
    except RuntimeError as exc:
        print(f"Could not generate notes: {exc}", file=sys.stderr)
        return 1

    print(notes.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
