"""Command-line entry point for the local voice assistant."""
from __future__ import annotations

import argparse
import sys

from pantherlake_ai_core import audio, engine as engine_mod

from . import session
from .wake_word import AVAILABLE_WAKE_WORDS, DEFAULT_WAKE_WORD

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"whisper_model": "small", "device": "auto"},
    engine_mod.Engine.OPENVINO: {"whisper_model": "base", "device": "AUTO"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice-assistant",
        description="Say the wake word, ask a question, hear a spoken answer -- fully on-device.",
    )
    p.add_argument(
        "--wake-word", choices=AVAILABLE_WAKE_WORDS, default=DEFAULT_WAKE_WORD,
        help=f"Wake word to listen for. Default: {DEFAULT_WAKE_WORD}",
    )
    p.add_argument(
        "--wake-threshold", type=float, default=0.5,
        help="Wake-word detection score threshold (0-1). Lower triggers more easily "
             "(and more falsely). Default: 0.5",
    )
    p.add_argument("--audio-device", default=None, help="Substring to match a specific microphone name.")
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=engine_mod.Engine.PORTABLE.value,
        help="Inference backend for speech-to-text, the LLM, and text-to-speech: 'portable' "
             "(CPU) or 'openvino' (Intel CPU/iGPU/NPU -- requires this brick's `openvino` "
             "extra). Wake-word detection always runs the same way regardless -- see this "
             "brick's README for why. Default: portable",
    )
    p.add_argument(
        "--whisper-model", default=None,
        help="Whisper model size override (tiny, base, small, medium, large-v3). "
             "Default depends on --engine.",
    )
    p.add_argument("--compute-device", default=None, help="openvino engine only: AUTO, CPU, GPU, or NPU.")
    p.add_argument(
        "--no-speak", action="store_true",
        help="Print replies instead of speaking them out loud (e.g. for a machine with no speakers).",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available microphones and inference devices, then exit.",
    )
    return p


def list_devices() -> None:
    print("Microphones:")
    for name in audio.list_microphones():
        print(f"  - {name}")
    print("\nInference devices (--compute-device):")
    print(engine_mod.describe_devices())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    engine = engine_mod.Engine(args.engine)
    defaults = _ENGINE_DEFAULTS[engine]
    whisper_model = args.whisper_model or defaults["whisper_model"]
    compute_device = args.compute_device or defaults["device"]

    print(
        f"Loading wake word '{args.wake_word}', Whisper '{whisper_model}', LLM, and TTS "
        f"(engine={engine.value}, device={compute_device})... this may download models on first use."
    )
    print(f'Say "{args.wake_word.replace("_", " ")}" to start. Press Ctrl+C to stop.\n')

    def handle_wake() -> None:
        print("(wake word heard -- listening for your question...)")

    def handle_heard(text: str) -> None:
        print(f"You: {text}")

    def handle_reply(text: str) -> None:
        print(f"Assistant: {text}")

    try:
        session.run(
            audio_device=args.audio_device,
            engine=engine,
            whisper_model_size=whisper_model,
            compute_device=compute_device,
            wake_word=args.wake_word,
            wake_threshold=args.wake_threshold,
            on_wake=handle_wake,
            on_heard=handle_heard,
            on_reply=handle_reply,
            speak_replies=not args.no_speak,
        )
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
