"""Command-line entry point for the local voice assistant."""
from __future__ import annotations

import argparse
import sys
import threading

from pantherlake_ai_core import engine as engine_mod

from . import session
from .wake_word import AVAILABLE_WAKE_WORDS

# Whisper size per engine -- see live-translation's CLI for why.
_WHISPER_SIZE_DEFAULTS = {engine_mod.Engine.PORTABLE: "small", engine_mod.Engine.OPENVINO: "base"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice-assistant",
        description="Press Enter, ask a question, press Enter again, hear a spoken answer -- fully on-device.",
    )
    p.add_argument(
        "--wake-word", choices=AVAILABLE_WAKE_WORDS, default=None,
        help="Listen for this wake word instead of waiting for Enter. Its model is openWakeWord's, "
             "for non-commercial use (CC BY-NC-SA 4.0). Default: none.",
    )
    p.add_argument(
        "--wake-threshold", type=float, default=0.5,
        help="Wake-word detection score threshold (0-1). Lower triggers more easily "
             "(and more falsely). Default: 0.5",
    )
    p.add_argument("--audio-device", default=None, help="Substring to match a specific microphone name.")
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Inference backend for speech-to-text, the LLM, and text-to-speech: 'portable' "
             "(CPU) or 'openvino' (Intel CPU/iGPU/NPU -- requires this brick's `openvino` "
             "extra). Wake-word detection always runs the same way regardless -- see this "
             "brick's README for why. Default: openvino if installed and a device is "
             "available, otherwise portable.",
    )
    p.add_argument(
        "--whisper-model", default=None,
        help="Whisper model size override (tiny, base, small, medium, large-v3). "
             "Default depends on --engine.",
    )
    p.add_argument(
        "--compute-device", default=None,
        help="openvino engine only: AUTO, CPU, GPU, or NPU, for listening and answering. The voice itself is "
             "always made on the CPU. Default: AUTO.",
    )
    p.add_argument(
        "--no-speak", action="store_true",
        help="Print replies instead of speaking them out loud (e.g. for a machine with no speakers).",
    )
    p.add_argument(
        "--llm-model", choices=("1.5b", "8b"), default=None,
        help="openvino engine: which language model answers. Default: 1.5b -- the voice waits for the whole answer, "
             "and the 8b takes three times as long to write one.",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available microphones and inference devices, then exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        engine_mod.print_devices(mics=True)
        return 0

    engine = engine_mod.resolve_engine(args.engine)
    whisper_model = args.whisper_model or _WHISPER_SIZE_DEFAULTS[engine]
    compute_device = args.compute_device or engine_mod.default_device(engine)

    print(
        (f"Loading wake word '{args.wake_word}', " if args.wake_word else "Loading ")
        + f"Whisper '{whisper_model}', LLM, and TTS "
        f"(engine={engine.value}, device={compute_device})... this may download models on first use."
    )
    # With no wake word, Enter is the button: once to start a question, once to end it.
    talk = threading.Event()
    if args.wake_word:
        print(f'Say "{args.wake_word.replace("_", " ")}" to start. Press Ctrl+C to stop.\n')
    else:
        print("Press Enter, ask, and press Enter again. Press Ctrl+C to stop.\n")

        def enter_is_the_button() -> None:
            for _ in sys.stdin:
                if talk.is_set():
                    talk.clear()
                else:
                    talk.set()

        threading.Thread(target=enter_is_the_button, daemon=True).start()

    def handle_wake() -> None:
        print("(wake word heard -- listening for your question...)" if args.wake_word else "(listening -- Enter when you are done)")

    def handle_missed(reason: str) -> None:
        print(f"({reason})")

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
            talk=talk,
            llm_model=args.llm_model,
            on_wake=handle_wake,
            on_missed=handle_missed,
            on_heard=handle_heard,
            on_reply=handle_reply,
            speak_replies=not args.no_speak,
        )
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
