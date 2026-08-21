# webcam-effects

Blurs or replaces the background behind you in a live webcam feed via
on-device person segmentation -- fully on-device.

Unlike every other brick in this workspace, both engines here run the
**exact same model** -- there was no need for two different model
families, because the model file itself (a small ONNX graph) loads
directly into either runtime:

- **`portable`** (default) -- ONNX Runtime, CPU only.
- **`openvino`** -- raw OpenVINO `Core`, targeting `CPU`, `GPU` (iGPU),
  or `NPU` explicitly.

Both load
[`onnx-community/mediapipe_selfie_segmentation`](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation)
(Google's official MediaPipe selfie-segmentation model, ONNX-converted,
~224KB quantized) -- see [`matte.py`](src/webcam_effects/matte.py) for the
shared pre/postprocessing both backends call. That makes this demo a
clean "same model, different silicon" comparison: switching engines is
purely a CPU-vs-NPU/iGPU question, not a model-quality one. OpenVINO's
`Core().read_model()` reads `.onnx` files directly -- no separate IR
conversion step needed.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engine only
uv sync --extra openvino   # also installs the OpenVINO engine
```

> **First run note:** the first time you run either engine, the model is
> downloaded from Hugging Face and cached (`~/.cache/huggingface`). Every
> run after that is fully offline.

## Usage

```bash
uv run webcam-effects --list-devices
```

Blur your background, with a live preview window (needs a display):

```bash
uv run webcam-effects --show
```

Replace your background with solid Intel blue instead:

```bash
uv run webcam-effects --show --effect replace --color "0,104,181"
```

Run segmentation on the Intel NPU via OpenVINO:

```bash
uv run webcam-effects --show --engine openvino --compute-device NPU
```

Press `Ctrl+C` to stop.

## Options

| Flag | Description |
| --- | --- |
| `--camera-index N` | Which webcam (see `--list-devices`). Default: `0`. |
| `--effect {blur,replace}` | Background treatment. Default: `blur`. |
| `--color "R,G,B"` | `replace` effect only: solid background color. Default: `0,104,181` (Intel blue). |
| `--engine {portable,openvino}` | Inference backend. Default: `portable`. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--model-path PATH` | Use a local model file instead of downloading the default. |
| `--show` | Also open a live preview window (`cv2.imshow`) -- off by default so this works headlessly (e.g. over SSH, in the launcher's background thread). |
| `--list-devices` | List cameras and inference devices, then exit. |

## How it works

1. **Capture** ([`pantherlake_ai_core.video`](../../core/src/pantherlake_ai_core/video.py),
   shared with other bricks) -- OpenCV webcam frames.
2. **Segment** ([`engine_factory.py`](src/webcam_effects/engine_factory.py)) --
   picks [`segmenter_portable.py`](src/webcam_effects/segmenter_portable.py)
   or [`segmenter_openvino.py`](src/webcam_effects/segmenter_openvino.py);
   both expose one `.segment(frame) -> mask` call and share every bit of
   pre/postprocessing via [`matte.py`](src/webcam_effects/matte.py) --
   there's no per-engine numerical difference to account for, only
   which silicon ran the same math.
3. **Emit** ([`pipeline.py`](src/webcam_effects/pipeline.py)) -- the
   shared capture-then-segment loop, taking an `on_frame(frame, mask)`
   callback. Deliberately doesn't apply an effect: `matte.apply_blur()` /
   `matte.apply_replace()` are called by the consumer instead, which is
   what lets the launcher change `blur` <-> `replace` (and the replace
   color) **live**, without restarting the camera or re-running
   segmentation setup.

## A real technical finding: OpenVINO needs a static input shape

The downloaded ONNX file declares a symbolic `batch_size` dimension
(`['batch_size', 3, 256, 256]`). Compiling it as-is with
`core.compile_model(model, device_name='NPU')` -- and, in practice, even
`'CPU'` in the same process -- doesn't raise a catchable Python
exception. It crashes the process with:

```
[ERROR] ... [IE::FrontEnd::importNetwork]   Upper bounds are not specified for node ...
LLVM ERROR: Failed to infer result type(s):
"IE.Interpolate"(...) : (...) -> ( ??? )
```

The fix is to reshape the model to a concrete static shape immediately
after loading it, before compiling:

```python
model = core.read_model(resolved_path)
model.reshape({model.inputs[0].get_any_name(): [1, 3, 256, 256]})
compiled = core.compile_model(model, device_name=device)
```

See [`segmenter_openvino.py`](src/webcam_effects/segmenter_openvino.py).
Verified against a real photo that all three paths (portable/CPU,
openvino/CPU, openvino/NPU) produce near-identical person-coverage
(0.4697 / 0.4697 / 0.4686) -- the reshape doesn't change what the model
computes, it's purely a compilation requirement.

## Notes / current limitations

- The segmentation mask is resized and Gaussian-blurred at output
  (`matte.postprocess`) to soften the cutout edge -- there's no
  temporal smoothing across frames, so a fast-moving edge can flicker
  slightly frame to frame.
- `--color` takes RGB on the CLI for readability but is stored/applied
  as BGR internally (matching OpenCV's frame layout) -- see
  `cli.py`'s `_parse_color`.
- No virtual-camera/OS-level webcam output -- this brick shows the
  effect in its own preview window or the launcher's video panel, it
  doesn't publish a fake webcam device for other apps (e.g. a video
  call) to consume.
