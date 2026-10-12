"""Command-line entry point for Portrait Studio (experimental)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core import npu
from pantherlake_ai_core.demo_samples import SAMPLE_ROOT

from .photo import DEFAULT_SIZE, SIZES
from .samples import SAMPLES
from .styles import STYLES, get


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="portrait-studio",
        description="EXPERIMENTAL. Draw the person in a photo again in a chosen style -- a cartoon, a corporate "
                    "headshot, a manga panel -- with an image-editing model running on this machine.",
    )
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--image", default=None, help="The photo: a JPEG, PNG or WebP file.")
    source.add_argument("--sample", default=None, help="Use a bundled portrait instead (see --list-samples).")
    p.add_argument(
        "--style", action="append", default=None,
        help="A style (see --list-styles); repeat it for several, or 'all' for every one. Default: cartoon.",
    )
    p.add_argument("--out", default=".", help="A folder for the pictures, or a .png/.jpg file for a single one. Default: here.")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, choices=SIZES, help=f"Pixels a side. Default: {DEFAULT_SIZE}")
    p.add_argument("--seed", type=int, default=0, help="Another seed, another picture of the same style. Default: 0")
    p.add_argument("--compute-device", default=None, help="Chip for the image model. Default: the integrated GPU.")
    p.add_argument("--model-path", default=None, help="A FLUX.2 klein OpenVINO build already on disk, instead of the Hub's.")
    p.add_argument("--list-devices", action="store_true", help="List available inference devices, then exit.")
    p.add_argument("--list-samples", action="store_true", help="List the bundled portraits, then exit.")
    p.add_argument("--list-styles", action="store_true", help="List the styles, then exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        engine_mod.print_devices()
        return 0
    if args.list_samples:
        for sample in SAMPLES:
            print(f"{sample.name}\n    {sample.description}")
        return 0
    if args.list_styles:
        for style in STYLES:
            print(f"{style.id:<12} {style.name}: {style.hint}")
        return 0

    path = Path(args.image) if args.image else None
    if args.sample:
        chosen = next((s for s in SAMPLES if s.name.lower().startswith(args.sample.lower())), None)
        if chosen is None:
            parser.error(f"No sample called '{args.sample}' (see --list-samples).")
        path = SAMPLE_ROOT / chosen.image
    if path is None:
        parser.error("Give a photo with --image, or pick one with --sample.")
    if not path.is_file():
        parser.error(f"There is no file at {path}.")

    asked = args.style or ["cartoon"]
    try:
        styles = list(STYLES) if "all" in asked else [get(style_id) for style_id in asked]
    except ValueError as exc:
        parser.error(str(exc))
    out = Path(args.out)
    one_file = out.suffix.lower() in (".png", ".jpg", ".jpeg")
    if one_file and len(styles) > 1:
        parser.error("--out names one file, and several styles were asked for: give a folder.")

    device = args.compute_device or engine_mod.preferred_device()
    if npu.is_npu(device):
        parser.error("The image model does not run on the NPU: choose a GPU, or the CPU.")

    from .photo import read
    from .stylist import Stylist

    try:
        photo = read(path.read_bytes())
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Loading the image model on {device}...", file=sys.stderr)
    stylist = Stylist(
        device, model_dir=args.model_path,
        on_downloading=lambda: print("Downloading the model (first run only, about 4.6 GB)...", file=sys.stderr),
        on_preparing=lambda: print("Preparing the model (once)...", file=sys.stderr),
    )
    (out.parent if one_file else out).mkdir(parents=True, exist_ok=True)
    for style in styles:
        drawn = stylist.restyle(photo, style, size=args.size, seed=args.seed)
        target = out if one_file else out / f"{path.stem}-{style.id}.png"
        drawn.image.save(target)
        rate = f", {drawn.steps_per_second:.2f} steps/s" if drawn.steps_per_second else ""
        print(f"{style.name}: {target}  ({drawn.seconds:.1f} s{rate})")
    return 0


def run() -> None:
    sys.exit(main())


if __name__ == "__main__":
    run()
