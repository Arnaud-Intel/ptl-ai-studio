# screen-ocr

Extracts text from a screenshot, a webcam frame, or an uploaded photo —
fully on-device, with optional translation to English.

Unlike `live-translation`'s continuous loop or `object-detection`'s video
feed, this is a one-shot action: grab or load one image, get its text back.
There's no capture loop here (see [`pipeline.py`](src/screen_ocr/pipeline.py)) --
just a session you can call `extract()` on as many times as you like.

It supports two interchangeable engines, using genuinely different
approaches (not just a different runtime for the same model):

- **`portable`** (default) — [RapidOCR](https://github.com/RapidAI/RapidOCR)
  (PaddleOCR detection + recognition models) via ONNX Runtime, CPU only.
  Fast, dedicated OCR models that return real per-region boxes and
  confidence scores. Can't translate -- there's no language model attached.
- **`openvino`** — a vision-language model
  ([Qwen2.5-VL-7B-Instruct](https://huggingface.co/OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov))
  via `openvino_genai.VLMPipeline`, targeting Intel `CPU`/`GPU`/`NPU` in
  principle. Slower and much heavier to download, but it's a full language
  model looking at the image, so it can translate what it reads in the
  same pass. **In practice, `NPU` currently fails to compile this model on
  this hardware** -- see the note right below.

> **Why not RapidOCR's own OpenVINO backend?** RapidOCR does have an
> `engine_type: "openvino"` option, but as of this writing its OpenVINO
> backend hardcodes `device_name="CPU"` when compiling the model (see
> `rapidocr/inference_engine/openvino/main.py` in their source) -- so it
> can't actually target the NPU/GPU. Using it would have made "openvino
> engine" a lie for this brick. The VLM path is what genuinely exposes
> device choice, at the cost of being a much bigger download.

> **A real technical finding: this VLM doesn't currently compile for
> NPU.** `--compute-device NPU` was tested (via `expense-extract`, which
> uses this engine unmodified) and reliably fails with a compiler error
> from OpenVINO's NPU backend: `[vpux-compiler] UnrollDistributedOps Pass
> failed: Can't convert 76 Bit to Byte`, surfacing as `RuntimeError:
> Compilation failed`. `CPU` and `GPU` were both verified working --
> `GPU` is the one that's actually fast. At 7B parameters (even int4),
> this VLM is a lot larger than the other OpenVINO models this workspace
> runs on NPU (Whisper-base, small TTS/embedding models), and this looks
> like it's hitting a real limit in the current NPU compiler for a model
> this size, not a configuration mistake on our end. Worth re-testing
> against a newer OpenVINO/NPU driver release; not something fixable from
> this brick's code.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engine only
uv sync --extra openvino   # also installs the OpenVINO engine
```

> **First run note:** RapidOCR's models ship inside the `rapidocr` package
> itself (no download needed). The OpenVINO engine's VLM (~4-5GB, int4) is
> downloaded from Hugging Face on first use and cached
> (`~/.cache/huggingface`); every run after that is fully offline.

## Usage

```bash
uv run screen-ocr --list-devices
```

Extract text from the screen (portable engine, no translation):

```bash
uv run screen-ocr --source screen
```

Extract text from a photo file:

```bash
uv run screen-ocr --source image --image ./receipt.jpg
```

Read and translate foreign-language text on screen, via the OpenVINO
engine on GPU (see the note above for why GPU, not NPU, here):

```bash
uv run screen-ocr --source screen --engine openvino --compute-device GPU --translate
```

## Options

| Flag | Description |
| --- | --- |
| `--source {screen,webcam,image}` | Where to read the image from. Default: `screen`. |
| `--image PATH` | Path to an image file (required if `--source image`). |
| `--screen-index N` | Which screen/monitor. Default: `1`. |
| `--camera-index N` | Which webcam. Default: `0`. |
| `--engine {portable,openvino}` | Inference backend. Default: `portable`. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--model-path PATH` | Use a local model instead of downloading the default. |
| `--translate` | Translate the extracted text to English. Requires `--engine openvino`. |
| `--list-devices` | List cameras, screens, and inference devices, then exit. |

## How it works

1. **Capture/load** — a single BGR frame from
   [`pantherlake_ai_core.video`](../../core/src/pantherlake_ai_core/video.py)'s
   `capture_screen_frame`/`capture_camera_frame` (added alongside this brick;
   `object-detection` has the equivalent continuous-stream versions), or a
   file read via OpenCV.
2. **Extract** ([`engine_factory.py`](src/screen_ocr/engine_factory.py)) —
   picks [`extractor_portable.py`](src/screen_ocr/extractor_portable.py) or
   [`extractor_openvino.py`](src/screen_ocr/extractor_openvino.py); both
   expose one `.extract(image, translate=False) -> ExtractionResult` call.
3. **Result** ([`types.py`](src/screen_ocr/types.py)) — `text` (or
   `translated_text` when translation was requested) plus `regions` (only
   populated by the portable engine, since the VLM path gives one holistic
   read of the image rather than per-region boxes).

## Notes / current limitations

- Translation is only available on the `openvino` engine; the portable
  engine raises a clear error if asked to translate rather than silently
  ignoring the flag or degrading.
- The VLM path answers as text generation, not as a bounding-box detector
  -- honest text, but no `regions` to overlay on the image, unlike the
  portable engine.
- No frame-averaging/retry: one capture, one pass. Good enough for a
  screenshot or photo; if it misses text on a blurry webcam frame, capture
  again.
