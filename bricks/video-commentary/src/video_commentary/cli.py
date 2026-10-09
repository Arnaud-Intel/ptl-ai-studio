"""video-commentary on the command line: a video watched, and a line about it every few seconds."""
from __future__ import annotations

import argparse
import sys
import threading

from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core import sample_videos

from . import moods, pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-commentary",
        description="EXPERIMENTAL. A vision-language model watches a video and says what is happening; "
                    "a small language model gives each line a mood. OpenVINO engine only.",
    )
    p.add_argument("--source", choices=("file", "webcam", "screen"), default="file", help="What to watch. Default: file.")
    p.add_argument("--path", help="The video file, for --source file. Default: the first sample video (fetched if need be).")
    p.add_argument("--camera", type=int, default=0, help="Camera index, for --source webcam.")
    p.add_argument("--screen", type=int, default=1, help="Screen index, for --source screen.")
    p.add_argument("--mood", choices=[mood.key for mood in moods.MOODS], default=moods.DEFAULT, help=f"The voice. Default: {moods.DEFAULT}.")
    p.add_argument("--every", type=float, default=pipeline.EVERY_SECONDS, help="Seconds between two looks. Default: %(default)s.")
    p.add_argument("--vision-device", help="Where the vision model runs. Default: a GPU.")
    p.add_argument("--mood-device", help="Where the language model runs. Default: the NPU if there is one.")
    p.add_argument("--once", action="store_true", help="Play the file once instead of in a loop.")
    p.add_argument("--list-devices", action="store_true", help="List available inference devices, then exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        engine_mod.print_devices(cameras=True, inference_flag="--vision-device / --mood-device")
        return 0

    path = args.path
    if args.source == "file" and not path:
        video = sample_videos.CATTLE_DRIVE
        if not sample_videos.present(video):
            print(f"Fetching {video.name} ({video.size_bytes / 1e6:.0f} MB, first use only)...", file=sys.stderr)
        path = str(sample_videos.download(video))

    devices = engine_mod.list_openvino_devices()
    vision = args.vision_device or engine_mod.preferred_large_model_device()
    voice = args.mood_device or ("NPU" if any(d.upper().startswith("NPU") for d in devices) else vision)
    print(f"Watching {path or args.source}: seen on {vision}, said on {voice}, mood {args.mood}. Ctrl+C stops.", file=sys.stderr)

    stop = threading.Event()

    def on_comment(comment: pipeline.Comment) -> None:
        print(comment.said, flush=True)
        if comment.said != comment.seen:
            print(f"    (seen: {comment.seen} | {comment.seeing_seconds:.1f} s + {comment.saying_seconds:.1f} s)", flush=True)

    try:
        pipeline.run(
            source=args.source, path=path or "", camera_index=args.camera, screen_index=args.screen, loop=not args.once,
            vision_device=vision, mood_device=voice, mood=lambda: args.mood, every=args.every,
            on_comment=on_comment, stop_event=stop,
        )
    except KeyboardInterrupt:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
