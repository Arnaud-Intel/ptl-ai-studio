"""Command-line entry point for multi-feed pedestrian/vehicle counting."""
from __future__ import annotations

import argparse
import sys
import threading
import time

from pantherlake_ai_core import engine as engine_mod

from . import pipeline, sources
from .draw import draw_tracks
from .types import FeedSpec

_SUMMARY_INTERVAL_SECONDS = 10.0


def _parse_source(raw: str, default_device: str) -> tuple[str, str]:
    """'path' or 'path|device' -> (path, device). Pipe, not colon, since a
    Windows path already contains a drive-letter colon."""
    if "|" in raw:
        path, device = raw.rsplit("|", 1)
        return path, device or default_device
    return raw, default_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="smart-city-monitor",
        description=(
            "Locally count pedestrians/cars/bikes across one or more video feeds -- each feed "
            "trackable so objects are counted once, and independently pinnable to its own compute "
            "device (e.g. one feed on the NPU, another on a GPU, running genuinely at once)."
        ),
    )
    p.add_argument(
        "--source", dest="sources", action="append", default=[], metavar="PATH[|DEVICE]",
        help="A feed to monitor, repeatable: a video file, or a live stream URL (RTSP/HTTP/HLS, "
             "or a YouTube live page). Optionally suffixed '|DEVICE' (e.g. 'clip.mp4|GPU.0') to pin "
             "that feed to a specific openvino device; omitted, it uses --compute-device. "
             "Required (at least one).",
    )
    p.add_argument(
        "--engine", choices=[e.value for e in engine_mod.Engine], default=None,
        help="Inference backend for every feed: 'portable' (DETR via ONNX Runtime, CPU) or "
             "'openvino' (YOLO11s via OpenVINO, Intel CPU/iGPU/NPU -- requires this brick's "
             "`openvino` extra). Default: openvino if installed and a device is available, "
             "otherwise portable.",
    )
    p.add_argument(
        "--compute-device", default=None,
        help="openvino engine only: default device for any --source without its own '|DEVICE'. "
             "Default: the integrated GPU, which for live video is about four times faster than "
             "AUTO for identical detections.",
    )
    p.add_argument("--model-path", default=None, help="Use a local model instead of downloading the default.")
    p.add_argument(
        "--no-loop", action="store_true",
        help="Stop each feed at end-of-file instead of restarting it from the beginning.",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Also open one live annotated window per feed (needs a display; off by default).",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="List available inference devices, then exit.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_devices:
        engine_mod.print_devices(inference_flag="--compute-device or a --source's |DEVICE suffix")
        return 0

    if not args.sources:
        print("Error: at least one --source is required.", file=sys.stderr)
        return 1

    engine = engine_mod.resolve_engine(args.engine)
    # The iGPU, not AUTO: for a small detection model on live video AUTO
    # measures four times slower for identical results (see
    # preferred_realtime_vision_device). Matches what the launcher does.
    default_device = args.compute_device or (
        engine_mod.preferred_realtime_vision_device()
        if engine == engine_mod.Engine.OPENVINO
        else engine_mod.default_device(engine)
    )

    feeds = []
    for i, raw in enumerate(args.sources, start=1):
        path, device = _parse_source(raw, default_device)
        feeds.append(
            FeedSpec(feed_id=f"feed-{i}", path=path, compute_device=device, name=sources.display_name(path))
        )

    print(f"Monitoring {len(feeds)} feed(s) (engine={engine.value}):")
    print("Experimental track counts, not unique objects: lost tracks may be counted again.")
    for feed in feeds:
        print(f"  - {feed.feed_id}: {feed.path} -> {feed.compute_device}")
    print("Press Ctrl+C to stop.\n")

    show_windows = args.show
    cv2 = None
    if show_windows:
        import cv2 as _cv2

        cv2 = _cv2

    def handle_ready(feed_id: str) -> None:
        print(f"[ready] {feed_id}")

    def handle_frame(feed_id, frame, tracks) -> None:
        if show_windows:
            annotated = draw_tracks(frame, tracks)
            cv2.imshow(feed_id, annotated)
            cv2.waitKey(1)

    # time.monotonic() is an arbitrary epoch, not 0-based -- start the
    # throttle from "now" so the first summary genuinely waits a full
    # interval instead of firing on the very first frame.
    last_summary = [time.monotonic()]

    def handle_counts(snapshot) -> None:
        now = time.monotonic()
        if now - last_summary[0] < _SUMMARY_INTERVAL_SECONDS:
            return
        last_summary[0] = now
        # Per feed, not summed. Two cameras added together give a number
        # about nowhere -- and which feed is busy is the thing worth
        # reading when each one is pinned to its own chip.
        names = {feed.feed_id: feed.name for feed in feeds}
        printed = False
        for feed_id in snapshot.active_feeds:
            counts = {k: v for k, v in sorted(snapshot.per_feed_last_60s.get(feed_id, {}).items()) if v}
            if not counts:
                continue
            parts = ", ".join(f"{label}: {count}/min" for label, count in counts.items())
            print(f"  {names.get(feed_id, feed_id)}: {parts}")
            printed = True
        if not printed:
            print("(no relevant objects counted yet)")

    stop_event = threading.Event()
    try:
        pipeline.run(
            feeds=feeds,
            engine=engine,
            model_path=args.model_path,
            loop=not args.no_loop,
            on_ready=handle_ready,
            on_frame=handle_frame,
            on_counts=handle_counts,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if show_windows:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
