"""Command-line entry point for local object detection."""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core import video

from . import pipeline

_ENGINE_DEFAULTS = {
    engine_mod.Engine.PORTABLE: {"device": "cpu"},
    engine_mod.Engine.OPENVINO: {"device": "AUTO"},
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="object-detect",
        description="Locally detect objects in a webcam or screen feed and print/overlay bounding boxes.",
    )
    p.add_argument(
        "--source", choices=["webcam", "screen"], default="screen",
        help="Video source. 'screen' works on any machine; 'webcam' needs a camera. Default: screen",
    )
    p.add_argument("--camera-index", type=int, default=0, help="Webcam index (see --list-devices). Default: 0")
    p.add_argument("--screen-index", type=int, default=1, help="Screen/monitor index (see --list-devices). Default: 1")
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=engine_mod.Engine.PORTABLE.value,
        help="Inference backend: 'portable' (DETR via ONNX Runtime, CPU) or 'openvino' "
             "(YOLO11n via OpenVINO, Intel CPU/iGPU/NPU -- requires this brick's `openvino` extra). "
             "Default: portable",
    )
    p.add_argument(
        "--compute-device", default=None,
        help="openvino engine only: AUTO, CPU, GPU, or NPU. Ignored for the portable engine.",
    )
    p.add_argument("--model-path", default=None, help="Use a local model instead of downloading the default.")
    p.add_argument(
        "--show", action="store_true",
        help="Also open a live annotated window (needs a display; off by default so this works headlessly).",
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    engine = engine_mod.Engine(args.engine)
    compute_device = args.compute_device or _ENGINE_DEFAULTS[engine]["device"]

    print(f"Loading detector (engine={engine.value}, device={compute_device})... this may download a model on first use.")

    source_label = "webcam" if args.source == "webcam" else f"screen {args.screen_index}"
    print(f"Watching {source_label}. Press Ctrl+C to stop.\n")

    show_window = args.show
    cv2 = None
    if show_window:
        import cv2 as _cv2

        cv2 = _cv2

    def handle_frame(frame, detections) -> None:
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        if detections:
            summary = ", ".join(f"{d.label} ({d.confidence:.0%})" for d in detections)
            print(f"[{timestamp}] {len(detections)} object(s): {summary}")
        else:
            print(f"[{timestamp}] (nothing detected)")

        if show_window:
            from .draw import draw_detections

            annotated = draw_detections(frame, detections)
            cv2.imshow("Panther Lake AI Studio - Object Detection", annotated)
            cv2.waitKey(1)

    try:
        pipeline.run(
            source=args.source,
            camera_index=args.camera_index,
            screen_index=args.screen_index,
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
