<p align="center">
  <img src="docs/panther-lake-ai-studio-banner.png" alt="Panther Lake AI Studio" width="720" />
</p>

<p align="center">
  <a href="https://github.com/Arnaud-Intel/ptl-ai-studio/tags"><img src="https://img.shields.io/github/v/tag/Arnaud-Intel/ptl-ai-studio?label=version&color=0068B5" alt="Version" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-0068B5" alt="Python" /></a>
  <a href="https://docs.openvino.ai/"><img src="https://img.shields.io/badge/runtime-OpenVINO-8A2BE2" alt="OpenVINO" /></a>
  <img src="https://img.shields.io/badge/cloud%20calls-zero-4ade80" alt="Cloud calls: zero" />
  <img src="https://img.shields.io/badge/platform-Windows-0078D6" alt="Platform: Windows" />
</p>

**A local AI Studio for Intel Panther Lake -- nine on-device AI demos,
one launcher, zero cloud calls.**

Speech translation. A voice assistant that talks back in your own voice.
Live meeting notes. Object detection. Receipt-to-spreadsheet automation
that runs your NPU and GPU *at the same time*. Every demo here runs
entirely on the machine in front of you -- no API key, no network call,
no data that leaves the device -- and every one of them can point
directly at your Intel CPU, integrated GPU, or NPU and show you, on a
live gauge, exactly which chip is doing the work.

This isn't a slide deck about on-device AI. It's nine working
applications that prove it.

## See it in 60 seconds

```bash
git clone https://github.com/Arnaud-Intel/ptl-ai-studio.git
cd ptl-ai-studio
uv sync --extra openvino
uv run panther-lake-launcher
```

A local web UI opens at `http://127.0.0.1:8765`: pick a demo, pick an
engine -- portable CPU or OpenVINO on your NPU/iGPU -- hit Launch, and
watch the CPU/GPU/NPU gauges in the header light up with *which demo is
using which chip, right now*. That live attribution is real, not
decorative: it comes from the exact device string each demo handed the
inference runtime, not a guess.

## The demo suite

| Demo | What it does | Runs on |
| --- | --- | --- |
| **Live Speech Translation** | Any spoken language, live, straight to English text | CPU / NPU / GPU |
| **Local Voice Assistant** | Say a wake word, ask a question, hear a spoken answer | CPU / NPU / GPU |
| **Live Meeting Notes** | Transcribes a call and generates a running summary + action items on demand | CPU / NPU / GPU |
| **Voice Clone Studio** | Enroll a 10-second voice sample, then speak any text back in that voice | CPU / NPU / GPU |
| **Webcam Background Effects** | Real-time background blur or replacement, no video ever leaves the machine | CPU / NPU / GPU |
| **Object Detection Overlay** | Live labeled bounding boxes over a webcam or screen feed | CPU / NPU / GPU |
| **Screen / Image Text Extraction** | Pull text out of a screenshot or photo, with optional on-device translation | CPU / GPU |
| **Local Document Q&A** | Chat with your own files -- retrieval-augmented, nothing indexed in the cloud | CPU / NPU / GPU |
| **Expense Report Extractor** | Photograph a folder of receipts, get a structured CSV -- OCR and the LLM run *concurrently* on two different chips | GPU **+** NPU, at once |

*Every "Runs on" cell is real, tested hardware routing -- not a spec
sheet claim. `expense-extract` in particular is the showcase: OCR reads
receipt N+1 on the GPU while the LLM is still structuring receipt N on
the NPU, both gauges lit at the same time, proof captured live from the
telemetry API during testing.*

Also on the roadmap and already visible as "Coming soon" cards in the
launcher: a local voice assistant for inbox triage, a commit/code-review
assistant, semantic recall over your own screen history, and live noise
suppression -- the suite is built to keep growing without touching what
already ships.

## Why this is worth a look

- **Genuine hardware routing, not a toggle that does nothing.** Every
  switchable-backend demo runs [OpenVINO](https://docs.openvino.ai/) for
  the Intel path, because `faster-whisper`, PyTorch, and ONNX Runtime's
  default provider are CPU/CUDA-only -- they physically cannot target an
  NPU or iGPU. OpenVINO is what actually exposes `CPU` / `GPU` / `NPU` as
  selectable devices on a chip like Panther Lake, which is the entire
  point of demonstrating *local* AI *on this hardware*, not just on a
  laptop.
- **Composable, not copy-pasted.** Nine demos, and the newest ones barely
  add code: `meeting-notes` has no transcriber or LLM of its own -- it
  composes `live-translation` and `doc-qa` directly. `voice-assistant`
  composes three bricks and adds exactly one new model (wake-word
  detection). Shared capture, VAD, and device-discovery code lives in one
  `core` package every brick depends on.
- **Verified against real hardware, not assumed.** This suite was built
  and tested against an actual Intel NPU and Arc GPU, end to end, down to
  finding (and routing around) a real OpenVINO NPU compiler limitation on
  a 7B vision-language model -- documented, not hidden, in
  [`screen-ocr`'s README](bricks/screen-ocr/README.md).
- **One launcher, no build step.** The front end is vanilla HTML/CSS/JS
  served straight from FastAPI -- no npm install, no bundler, just
  `uv run panther-lake-launcher`.

## Command line, if you'd rather skip the UI

Every demo also installs its own console script:

```bash
uv run live-translate --source system --engine openvino --compute-device NPU
uv run voice-assistant --engine openvino --compute-device NPU
uv run voice-clone-studio --record 15 --text "Hello from my own cloned voice."
uv run expense-extract ./receipts --ocr-engine openvino --ocr-device GPU --llm-engine openvino --llm-device NPU
```

See each brick's own README for its full set of options.

## Under the hood

One [`uv` workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/),
one shared `.venv`, every brick and the launcher installed together so
they can depend on each other freely. Full layout, the pattern for adding
a new brick, and the auto-versioning mechanism (the badge at the top of
this page updates itself on every merge to `main`) are in
[CONTRIBUTING.md](CONTRIBUTING.md).

---

Built for [Dell](https://www.dell.com/) hardware powered by
[Intel(R) Core(TM) Ultra](https://www.intel.com/) and
[Intel(R) Arc(TM) Graphics](https://www.intel.com/) -- see it running live
in the launcher's own footer.
