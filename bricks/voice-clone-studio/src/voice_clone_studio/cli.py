"""Command-line entry point for local voice cloning."""
from __future__ import annotations

import argparse
import os
import sys

from pantherlake_ai_core import audio, engine as engine_mod

from .chatterbox_model import PARALINGUISTIC_TAGS
from .engine_factory import DEFAULT_MODEL, MODEL_ENGINES, MODELS
from .pipeline import VoiceCloneSession
from .samples import SAMPLES
from .voice_model import STYLES


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
    p.add_argument("--sample", default=None, help="Use a named example text instead (see --list-samples).")
    p.add_argument(
        "--model", choices=MODELS, default=DEFAULT_MODEL,
        help="Which voice model. 'chatterbox' (default) is Chatterbox-Turbo: it clones a real "
             "speaker far more closely (measured speaker similarity 0.65 vs 0.29 against a human "
             "reference) and takes paralinguistic tags such as "
             + " ".join(PARALINGUISTIC_TAGS[:3]) + " written into the text -- but it is CPU-only. "
             "'openvoice' is the older tone-color model, lower fidelity but the one that runs on "
             "the NPU and iGPU, with nine delivery styles.",
    )
    p.add_argument("--style", choices=STYLES, default="default", help="openvoice model only: base delivery style before tone cloning. Default: default")
    p.add_argument("--tau", type=float, default=0.3, help="openvoice model only: tone-conversion strength (higher = closer to the reference tone). Default: 0.3")
    p.add_argument("--output", default="cloned.wav", help="Output WAV path. Default: cloned.wav")
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Inference backend: 'portable' (PyTorch, CPU) or 'openvino' (Intel CPU/iGPU/NPU -- "
             "requires this brick's `openvino` extra). Both run the identical checkpoints, so "
             "this only changes which silicon runs them. Default: openvino if installed and a "
             "device is available, otherwise portable.",
    )
    p.add_argument(
        "--compute-device", default=None,
        help="openvino engine only: AUTO, CPU, GPU, or NPU. Default: AUTO.",
    )
    p.add_argument("--model-path", default=None, help="Use a local model instead of downloading the default.")
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available microphones and inference devices, then exit.",
    )
    p.add_argument(
        "--list-samples", action="store_true",
        help="List available example texts, then exit.",
    )
    return p


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
    os.close(fd)
    sf.write(path, clip, audio.SAMPLE_RATE)
    print("Recording complete.")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        engine_mod.print_devices(mics=True)
        return 0

    if args.list_samples:
        for s in SAMPLES:
            print(f"{s.name}: {s.description}")
        return 0

    if args.text and args.sample:
        parser.error("--text and --sample are mutually exclusive")
    if args.sample:
        matches = [s for s in SAMPLES if s.name == args.sample]
        if not matches:
            parser.error(f"no sample named '{args.sample}' (see --list-samples)")
        args.text = matches[0].text

    if not args.reference and not args.record:
        parser.error("one of the arguments --reference --record is required")

    # Chatterbox runs on one engine only, so resolving "best available"
    # would hand it openvino and fail. An engine asked for explicitly is
    # still passed through, to get the factory's reason rather than a
    # silent substitution.
    allowed = MODEL_ENGINES[args.model]
    engine = engine_mod.resolve_engine(args.engine) if args.engine else (
        allowed[0] if len(allowed) == 1 else engine_mod.resolve_engine(None)
    )
    compute_device = args.compute_device or engine_mod.default_device(engine)

    reference_path = args.reference or _record_reference(args.record)

    print(f"Loading {args.model} (engine={engine.value}, device={compute_device})... this may download a model on first use.")
    try:
        session = VoiceCloneSession(engine, model=args.model, device=compute_device, model_path=args.model_path)
    except ValueError as exc:
        # A model/engine pairing that can't work is the user's choice to
        # correct, so say so and stop -- not a traceback.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Enrolling voice from '{reference_path}'...")
    session.enroll(reference_path)
    print("Voice enrolled.")

    if not args.text:
        return 0

    if session.supports_styles:
        print(f"Synthesizing (style={args.style}, tau={args.tau})...")
        audio_out, sample_rate = session.synthesize(args.text, style=args.style, tau=args.tau)
    else:
        print("Synthesizing...")
        audio_out, sample_rate = session.synthesize(args.text)

    import soundfile as sf

    sf.write(args.output, audio_out, sample_rate)
    print(f"Wrote {args.output} ({len(audio_out) / sample_rate:.1f}s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
