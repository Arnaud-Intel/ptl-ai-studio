"""Command-line entry point for local webcam background effects."""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core import video

from . import matte, pipeline

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}


def _parse_color(value: str) -> tuple[int, int, int]:
    """Accepts 'R,G,B'; returns a BGR tuple (OpenCV's native order)."""
    try:
        r, g, b = (int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--color must be 'R,G,B', e.g. '0,104,181'") from exc
    return (b, g, r)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webcam-effects",
        description="Locally blur or replace your webcam background using on-device person segmentation.",
    )
    p.add_argument("--camera-index", type=int, default=0, help="Webcam index (see --list-devices). Default: 0")
    p.add_argument("--effect", choices=["blur", "replace"], default="blur", help="Background effect. Default: blur")
    p.add_argument(
        "--color", type=_parse_color, default=(181, 104, 0),
        help="Replace-mode background color as 'R,G,B'. Default: Intel blue (0,104,181).",
    )
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Inference backend: 'portable' (ONNX Runtime, CPU) or 'openvino' (Intel CPU/iGPU/NPU -- "
             "requires this brick's `openvino` extra). Both run the identical segmentation model, so "
             "this only changes which silicon runs it. Default: openvino if installed and a device "
             "is available, otherwise portable.",
    )
    p.add_argument("--compute-device", default=None, help="openvino engine only: AUTO, CPU, GPU, or NPU.")
    p.add_argument("--model-path", default=None, help="Use a local model instead of downloading the default.")
    p.add_argument(
        "--show", action="store_true",
        help="Also open a live annotated window (needs a display; off by default so this works headlessly).",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available cameras and inference devices, then exit.",
    )
    return p


def list_devices() -> None:
    from pantherlake_ai_core.engine import describe_devices

    print("Cameras (--camera-index N):")
    cameras = video.list_cameras()
    if cameras:
        for index in cameras:
            print(f"  - {index}")
    else:
        print("  (none found)")

    print("\nInference devices (--compute-device):")
    print(describe_devices())


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

    print(f"Loading segmenter (engine={engine.value}, device={compute_device})... this may download a model on first use.")
    print(f"Watching camera {args.camera_index}, effect={args.effect}. Press Ctrl+C to stop.\n")

    show_window = args.show
    cv2 = None
    if show_window:
        import cv2 as _cv2

        cv2 = _cv2

    def handle_frame(frame, mask) -> None:
        coverage = matte.person_coverage(mask)
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] person coverage: {coverage:.0%}")

        if show_window:
            if args.effect == "replace":
                annotated = matte.apply_replace(frame, mask, args.color)
            else:
                annotated = matte.apply_blur(frame, mask)
            cv2.imshow("Panther Lake AI Studio - Webcam Background Effects", annotated)
            cv2.waitKey(1)

    try:
        pipeline.run(
            camera_index=args.camera_index,
            engine=engine,
            compute_device=compute_device,
            model_path=args.model_path,
            on_frame=handle_frame,
        )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if show_window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
