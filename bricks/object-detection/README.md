# object-detection

Detects objects in a live webcam or screen-capture feed and overlays
labeled, confidence-scored bounding boxes — fully on-device.

It supports two interchangeable inference engines, each using a
genuinely different model family (not just a different runtime for the
same model), because that's what a well-supported model actually looks
like for each:

- **`portable`** (default) — [DETR](https://huggingface.co/facebook/detr-resnet-50)
  (ResNet-50 backbone) via ONNX Runtime, CPU only. DETR is a set-prediction
  model, so there's no anchor decoding or non-max suppression to get
  wrong — simpler, easier to trust, at the cost of a heavier backbone.
- **`openvino`** — [YOLO11n](https://huggingface.co/OpenVINO/YOLO11n-int8-ov)
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

Run on Intel NPU via OpenVINO:

```bash
uv run object-detect --source screen --engine openvino --compute-device NPU
```

Press `Ctrl+C` to stop.

## Options

| Flag | Description |
| --- | --- |
| `--source {webcam,screen}` | Video source. `screen` works everywhere; `webcam` needs a camera. Default: `screen`. |
| `--camera-index N` | Which webcam (see `--list-devices`). Default: `0`. |
| `--screen-index N` | Which screen/monitor (see `--list-devices`). Default: `1`. |
| `--engine {portable,openvino}` | Inference backend. Default: `portable`. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--model-path PATH` | Use a local model file/dir instead of downloading the default. |
| `--show` | Also open a live annotated window (`cv2.imshow`) -- off by default so this works headlessly (e.g. over SSH, in the launcher's background thread). |
| `--list-devices` | List cameras, screens, and inference devices, then exit. |

## How it works

1. **Capture** ([`pantherlake_ai_core.video`](../../core/src/pantherlake_ai_core/video.py),
   shared with other bricks) — OpenCV for webcam frames, [`mss`](https://github.com/BoboTiG/python-mss)
   for screen frames. Both yield plain BGR `numpy` arrays, so the rest of
   the pipeline doesn't care which source is in use.
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

## Notes / current limitations

- The two engines use different label vocabularies: DETR here uses
  COCO-91 (some category-file gaps, filtered out), YOLO11n uses COCO-80.
  Same idea (common everyday objects), not byte-identical class lists.
- DETR (portable) is noticeably slower per frame on CPU than YOLO11n
  (openvino) is on CPU/GPU/NPU -- that gap is itself a fair demonstration
  of what hardware acceleration buys you, not a bug to fix.
- No frame-skipping/throttling: every captured frame is run through the
  detector. On a slow path (e.g. DETR on a large screen capture) this
  means a lower effective frame rate rather than dropped detections --
  reasonable for a demo, worth revisiting if this needs to hit a target FPS.
