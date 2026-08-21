"""Command-line entry point for live local speech-to-English translation."""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from pantherlake_ai_core import audio, engine as engine_mod

from . import pipeline

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"model": "small", "device": "auto"},
    engine_mod.Engine.OPENVINO: {"model": "base", "device": "AUTO"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live-translate",
        description=(
            "Locally translate speech from a microphone, or from whatever is "
            "playing on this device (system audio loopback), into English text."
        ),
    )
    p.add_argument(
        "--source", choices=["mic", "system"], default="mic",
        help="Audio source: 'mic' for a microphone, 'system' to capture device "
             "audio output (e.g. a playing video) via loopback. Default: mic",
    )
    p.add_argument(
        "--audio-device", default=None,
        help="Substring to match a specific microphone/output device name. "
             "Default: the system default device for the chosen source.",
    )
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=engine_mod.Engine.PORTABLE.value,
        help="Inference backend: 'portable' (faster-whisper, CPU/CUDA) or "
             "'openvino' (Intel CPU/iGPU/NPU, e.g. Panther Lake -- requires "
             "this brick's `openvino` extra). Default: portable",
    )
    p.add_argument(
        "--model", default=None,
        help="Model size (tiny, base, small, medium, large-v3). Default depends on "
             "--engine: 'small' for portable, 'base' for openvino (the largest size "
             "Intel doesn't pre-convert as multilingual is large-v3).",
    )
    p.add_argument(
        "--compute-device", default=None,
        help="Which device the engine should run on. For --engine portable: "
             "auto, cpu, or cuda. For --engine openvino: AUTO, CPU, GPU, or NPU. "
             "Default depends on --engine.",
    )
    p.add_argument(
        "--compute-type", default="auto",
        help="faster-whisper compute type, portable engine only (auto, int8, float16, ...). Default: auto",
    )
    p.add_argument(
        "--ov-model-dir", default=None,
        help="openvino engine only: path to a model you converted yourself with "
             "`optimum-cli export openvino`, instead of downloading Intel's default.",
    )
    p.add_argument("--output", default=None, help="Also append translated lines to this text file.")
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available microphones, output devices, and inference devices, then exit.",
    )
    return p


def list_devices() -> None:
    print("Microphones (--source mic):")
    for name in audio.list_microphones():
        print(f"  - {name}")
    print("\nOutput devices (--source system, captured via loopback):")
    for name in audio.list_speakers():
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
    model_size = args.model or defaults["model"]
    compute_device = args.compute_device or defaults["device"]

    print(
        f"Loading '{model_size}' Whisper model on engine={engine.value}, device={compute_device}... "
        "(first run may download the model; every run after that is fully offline)"
    )

    label = "microphone" if args.source == "mic" else "system audio (loopback)"
    device_note = f" matching '{args.audio_device}'" if args.audio_device else " (default device)"
    print(f"Listening on {label}{device_note}. Press Ctrl+C to stop.\n")

    out_file = open(args.output, "a", encoding="utf-8") if args.output else None

    def handle_result(result) -> None:
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] ({result.detected_language}) {result.text}"
        print(line)
        if out_file:
            out_file.write(line + "\n")
            out_file.flush()

    try:
        pipeline.run(
            source=args.source,
            audio_device=args.audio_device,
            engine=engine,
            model_size=model_size,
            compute_device=compute_device,
            compute_type=args.compute_type,
            ov_model_dir=args.ov_model_dir,
            on_result=handle_result,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if out_file:
            out_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
