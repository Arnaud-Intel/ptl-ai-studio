"""Command-line entry point for local text extraction."""
from __future__ import annotations

import argparse
import sys

from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core import video

from .pipeline import OcrSession

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="screen-ocr",
        description="Locally extract (and optionally translate) text from a screenshot, webcam frame, or image file.",
    )
    p.add_argument(
        "--source", choices=["screen", "webcam", "image"], default="screen",
        help="Where to read the image from. Default: screen",
    )
    p.add_argument("--image", default=None, help="Path to an image file (required if --source image).")
    p.add_argument("--screen-index", type=int, default=1, help="Screen/monitor index. Default: 1")
    p.add_argument("--camera-index", type=int, default=0, help="Webcam index. Default: 0")
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Inference backend: 'portable' (RapidOCR, CPU) or 'openvino' (vision-language model, "
             "Intel CPU/iGPU/NPU -- requires this brick's `openvino` extra, and is required for "
             "--translate). Default: openvino if installed and a device is available, otherwise "
             "portable.",
    )
    p.add_argument("--compute-device", default=None, help="openvino engine only: AUTO, CPU, GPU, or NPU.")
    p.add_argument("--model-path", default=None, help="Use a local model instead of downloading the default.")
    p.add_argument(
        "--translate", action="store_true",
        help="Translate the extracted text to English (requires --engine openvino).",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available cameras, screens, and inference devices, then exit.",
    )
    return p


def list_devices() -> None:
    from pantherlake_ai_core.engine import describe_devices

    print("Cameras (--source webcam --camera-index N):")
    cameras = video.list_cameras()
    if cameras:
        for index in cameras:
            print(f"  - {index}")
    else:
        print("  (none found)")

    print("\nScreens (--source screen --screen-index N):")
    for screen in video.list_screens():
        print(f"  - {screen['index']}: {screen['width']}x{screen['height']}")

    print("\nInference devices (--compute-device):")
    print(describe_devices())


def _load_image(args):
    import cv2

    if args.source == "image":
        if not args.image:
            print("--image PATH is required when --source image", file=sys.stderr)
            return None
        image = cv2.imread(args.image)
        if image is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
        return image

    if args.source == "webcam":
        return video.capture_camera_frame(args.camera_index)

    return video.capture_screen_frame(args.screen_index)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    if args.engine:
        engine = engine_mod.Engine(args.engine)
    else:
        from pantherlake_ai_core.engine import list_openvino_devices

        engine = engine_mod.Engine.OPENVINO if list_openvino_devices() else engine_mod.Engine.PORTABLE
    compute_device = args.compute_device or _ENGINE_DEFAULTS[engine]["device"]

    image = _load_image(args)
    if image is None:
        return 1

    print(f"Loading extractor (engine={engine.value}, device={compute_device})... this may download a model on first use.")
    session = OcrSession(engine, device=compute_device, model_path=args.model_path)

    try:
        result = session.extract(image, translate=args.translate)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result.translated_text is not None:
        print("Translation (English):\n")
        print(result.translated_text)
    else:
        print("Extracted text:\n")
        print(result.text or "(no text detected)")
        if result.regions:
            print("\nRegions:")
            for region in result.regions:
                print(f"  [{region.confidence:.2f}] {region.text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
