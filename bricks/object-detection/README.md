# object-detection

Detects objects in a video file, a live webcam or a screen-capture feed and
overlays labeled, confidence-scored bounding boxes — fully on-device.

It supports two interchangeable inference engines, each using a
genuinely different model family (not just a different runtime for the
same model), because that's what a well-supported model actually looks
like for each:

- **`portable`** (default) — [DETR](https://huggingface.co/facebook/detr-resnet-50)
  (ResNet-50 backbone) via ONNX Runtime, CPU only. DETR is a set-prediction
  model, so there's no anchor decoding or non-max suppression to get
  wrong — simpler, easier to trust, at the cost of a heavier backbone.
- **`openvino`** — [YOLO11s](https://huggingface.co/OpenVINO/YOLO11s-int8-ov)
  via [`openvino-model-api`](https://github.com/open-edge-platform/model_api)
  (Intel's own inference-wrapper package for OpenVINO Model Zoo detection
  models). Targets Intel hardware explicitly: `CPU`, `GPU` (iGPU), or `NPU`.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engine only
uv sync --extra openvino   # also installs the OpenVINO engine
```

This brick requires **Python >= 3.11** (`openvino-model-api` doesn't
support 3.10), unlike the other bricks in this workspace.

> **First run note:** the first time you run with a given `--engine`, its
> detection model is downloaded from Hugging Face and cached
> (`~/.cache/huggingface`). Every run after that is fully offline.

## Usage

List available cameras, screens, and inference devices:

```bash
uv run object-detect --list-devices
```

Watch the screen (works on any machine, no camera needed) and print
detections to the console:

```bash
uv run object-detect --source screen
```

Watch a webcam, with a live annotated window (needs a display):

```bash
uv run object-detect --source webcam --show
```

Play a video file — with no `--path`, the studio's Toronto sample video,
fetched the first time (see [`sample-data/videos`](../../sample-data/videos/README.md)):

```bash
uv run object-detect --source file --show
uv run object-detect --source file --path street.mp4 --once
```

Run on Intel NPU via OpenVINO:

```bash
uv run object-detect --source screen --engine openvino --compute-device NPU
```

Press `Ctrl+C` to stop.

## Options

| Flag | Description |
| --- | --- |
| `--source {webcam,screen,file}` | Video source. `screen` works everywhere; `webcam` needs a camera; `file` plays `--path`. Default: `screen`. |
| `--path FILE` | The video, for `--source file`. Default: the Toronto sample video, fetched if need be. |
| `--once` | Play the file once instead of in a loop. |
| `--camera-index N` | Which webcam (see `--list-devices`). Default: `0`. |
| `--screen-index N` | Which screen/monitor (see `--list-devices`). Default: `1`. |
| `--engine {portable,openvino}` | Inference backend. Default: `portable`. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--model-path PATH` | Use a local model file/dir instead of downloading the default. |
| `--show` | Also open a live annotated window (`cv2.imshow`) -- off by default so this works headlessly (e.g. over SSH, in the launcher's background thread). |
| `--list-devices` | List cameras, screens, and inference devices, then exit. |

## How it works

1. **Capture** ([`pantherlake_ai_core.video`](../../core/src/pantherlake_ai_core/video.py),
   shared with other bricks) — OpenCV for webcam frames and video files (a
   file is played at its own frame rate, in a loop), [`mss`](https://github.com/BoboTiG/python-mss)
   for screen frames. All yield plain BGR `numpy` arrays, so the rest of
   the pipeline doesn't care which source is in use. A frame wider than
   1280 pixels is shrunk to that before anything else is done with it.
2. **Detect** ([`engine_factory.py`](src/object_detection/engine_factory.py)) —
   picks [`detector_portable.py`](src/object_detection/detector_portable.py)
   or [`detector_openvino.py`](src/object_detection/detector_openvino.py);
   both expose one `.detect(frame) -> list[Detection]` call, hiding very
   different pre/post-processing (DETR's softmax-over-queries vs. YOLO's
   anchor decode + NMS, handled for us by `model_api`).
3. **Emit** ([`pipeline.py`](src/object_detection/pipeline.py)) — the
   shared capture-then-detect loop, taking an `on_frame(frame, detections)`
   callback. Deliberately doesn't draw anything: the CLI's `--show` window
   and the launcher's video stream both call
   [`draw.py`](src/object_detection/draw.py) themselves, since "how to
   present a frame" differs per consumer while "how to detect objects in
   it" doesn't.

## What its proofing pass found (2026-10-09)

The Auto Demo's "Seeing and answering" scene waited for this brick to be
gone over (the scene has since become "The camera sees you": the detector
on the camera, with the Video Commentator watching the same frames, which
the launcher's runner hands over as captured). On the XPS 14, YOLO11s, through the launcher:

| Source | Integrated GPU | NPU | CPU |
| --- | --- | --- | --- |
| A sample video (Toronto, 1080p, 24 frames a second) | 24.0 | 23.8 | 24.0 |
| The webcam | 30.4 | | |
| The screen (2880x1800) | 18.8 (12.0 before) | 12.6 before | |

DETR on the portable engine: 4.3 frames a second on the same video.

- **There was nothing to point it at.** A covered camera or a screen full
  of windows gives a detector nothing to find (no box at all on this
  desktop). A video file is now a source, the studio's sample videos are
  offered, and the panel opens on the busiest of them.
- **A wrong source was answered "started".** The source was checked after
  the model had loaded, on a thread nobody waited for; the Auto Demo's own
  scene asked for `camera`, which does not exist (`webcam` does), and would
  have shown an empty picture. It is refused at once now, by name, before
  any model is loaded.
- **A big screen cost three frames in four.** Of a frame's 83 ms, 20 went
  to resizing 2880x1800 pixels for a model that looks at 640, and 21 to
  drawing and encoding all of them for a page that shows them 700 wide.
  Frames are shrunk to 1280 first, and the screen capture hands over
  packed pixels instead of a view with gaps (15 ms a resize). What is left
  is the capture itself, 41 ms a grab.
- **Boxes were hairlines on a tall picture**: two pixels and a 13-pixel
  label whatever the frame. They are sized to it, as the city monitor's are.
- **What it still misses**: small people. On the Toronto crossing YOLO11s
  boxes 5 people a frame where DETR boxes 21, and in a crowd seen from far
  above it finds a handful of several hundred. Its 640-pixel look is the
  reason; a larger input or tiles is the remedy, and it is not done.
- **Not looked at**: the camera's picture itself. The webcam was run once
  for its frame rate and its labels, without its picture being fetched.

## Notes / current limitations

- The two engines use different label vocabularies: DETR here uses
  COCO-91 (some category-file gaps, filtered out), YOLO11s uses COCO-80.
  Same idea (common everyday objects), not byte-identical class lists.
- DETR (portable) is noticeably slower per frame on CPU than YOLO11s
  (openvino) is on CPU/GPU/NPU -- that gap is itself a fair demonstration
  of what hardware acceleration buys you, not a bug to fix.
- No frame-skipping/throttling: every captured frame is run through the
  detector. On a slow path (e.g. DETR on a large screen capture) this
  means a lower effective frame rate rather than dropped detections --
  reasonable for a demo, worth revisiting if this needs to hit a target FPS.
- A video file is played at its own pace and every frame is looked at: a
  detector slower than the file (DETR, at 4 frames a second) plays it in
  slow motion rather than skip.
