"""Command-line entry point for local voice cloning."""
from __future__ import annotations

import argparse
import sys

from pantherlake_ai_core import audio, engine as engine_mod

from .pipeline import VoiceCloneSession
from .voice_model import STYLES

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "CPU"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice-clone-studio",
        description="Enroll a short voice sample, then synthesize any text in that voice -- fully on-device.",
    )
    source = p.add_mutually_exclusive_group()
    source.add_argument("--reference", metavar="PATH", help="Audio file of the voice to clone (5-30s of clear speech).")
    source.add_argument(
        "--record", type=float, metavar="SECONDS",
        help="Record that many seconds from the default microphone instead of using a file.",
    )
    p.add_argument("--text", required=False, help="Text to speak in the cloned voice. If omitted, only enrolls and exits.")
    p.add_argument("--style", choices=STYLES, default="default", help="Base delivery style before tone cloning. Default: default")
    p.add_argument("--tau", type=float, default=0.3, help="Tone-conversion strength (higher = closer to the reference tone). Default: 0.3")
    p.add_argument("--output", default="cloned.wav", help="Output WAV path. Default: cloned.wav")
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=engine_mod.Engine.PORTABLE.value,
        help="Inference backend: 'portable' (PyTorch, CPU) or 'openvino' (Intel CPU/iGPU/NPU -- "
             "requires this brick's `openvino` extra). Both run the identical checkpoints, so "
             "this only changes which silicon runs them. Default: portable",
    )
    p.add_argument("--compute-device", default=None, help="openvino engine only: AUTO, CPU, GPU, or NPU.")
    p.add_argument("--model-path", default=None, help="Use a local model instead of downloading the default.")
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available microphones and inference devices, then exit.",
    )
    return p


def list_devices() -> None:
    from pantherlake_ai_core.engine import describe_devices

    print("Microphones (--record captures from the default one):")
    mics = audio.list_microphones()
    if mics:
        for name in mics:
            print(f"  - {name}")
    else:
        print("  (none found)")

    print("\nInference devices (--compute-device):")
    print(describe_devices())


def _record_reference(seconds: float) -> str:
    import tempfile

    import numpy as np
    import soundfile as sf

    print(f"Recording {seconds:.0f}s from the default microphone...")
    blocks = []
    captured = 0.0
    for block in audio.stream_blocks("mic", None):
        blocks.append(block)
        captured += len(block) / audio.SAMPLE_RATE
        if captured >= seconds:
            break
    clip = np.concatenate(blocks)

    fd, path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    sf.write(path, clip, audio.SAMPLE_RATE)
    print("Recording complete.")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    if not args.reference and not args.record:
        parser.error("one of the arguments --reference --record is required")

    engine = engine_mod.Engine(args.engine)
    compute_device = args.compute_device or _ENGINE_DEFAULTS[engine]["device"]

    reference_path = args.reference or _record_reference(args.record)

    print(f"Loading voice cloner (engine={engine.value}, device={compute_device})... this may download a model on first use.")
    session = VoiceCloneSession(engine, device=compute_device, model_path=args.model_path)

    print(f"Enrolling voice from '{reference_path}'...")
    session.enroll(reference_path)
    print("Voice enrolled.")

    if not args.text:
        return 0

    print(f"Synthesizing (style={args.style}, tau={args.tau})...")
    audio_out, sample_rate = session.synthesize(args.text, style=args.style, tau=args.tau)

    import soundfile as sf

    sf.write(args.output, audio_out, sample_rate)
    print(f"Wrote {args.output} ({len(audio_out) / sample_rate:.1f}s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
