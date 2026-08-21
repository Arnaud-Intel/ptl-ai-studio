# Panther Lake AI Studio

A growing collection of small, local-only AI demos ("bricks"), each showing
one on-device AI capability running without any cloud dependency, brought
together under one front end: **Panther Lake AI Studio**
(`uv run panther-lake-launcher`). Several bricks support a **switchable
backend**: a portable one that runs anywhere (CPU, or CUDA on an NVIDIA
GPU), and an [OpenVINO](https://docs.openvino.ai/) one that can target Intel
hardware explicitly -- CPU, integrated GPU, or NPU (e.g. Panther Lake) --
which is the actual point of this showcase.

## Layout

```
local_demo/
  pyproject.toml          <- uv workspace root (no code of its own)
  core/                    <- pantherlake-ai-core: shared code every brick can use
    src/pantherlake_ai_core/
      audio.py             <- mic / system-audio (loopback) capture, speaker playback
      video.py               <- webcam / screen capture
      segmenter.py             <- lightweight energy-based voice-activity segmentation
      engine.py                 <- Engine enum + OpenVINO device discovery
      telemetry.py               <- CPU/GPU/NPU utilization reading
      types.py                    <- small shared result types (e.g. TranslationResult)
  bricks/
    live-translation/       <- speech -> English translation (streaming-demo template)
      pyproject.toml
      src/live_translation/
        pipeline.py          <- capture->segment->translate loop, shared by the CLI and the launcher
        cli.py
    doc-qa/                  <- retrieval-augmented Q&A over local files (request/response-demo template)
      pyproject.toml
      src/doc_qa/
        pipeline.py           <- ingest()/ask(), shared by the CLI and the launcher
        cli.py
    object-detection/        <- live bounding-box overlay (latest-frame-stream-demo template)
      pyproject.toml
      src/object_detection/
        pipeline.py            <- capture->detect loop, shared by the CLI and the launcher
        cli.py
    screen-ocr/               <- text extraction from a screenshot/webcam/photo (one-shot-demo template)
      pyproject.toml
      src/screen_ocr/
        pipeline.py             <- OcrSession.extract(), shared by the CLI and the launcher
        cli.py
    meeting-notes/             <- live transcript + LLM notes (brick-composing-bricks template)
      pyproject.toml
      src/meeting_notes/
        session.py               <- composes live_translation.pipeline + doc_qa.engine_factory;
                                     no transcriber or LLM code of its own
        cli.py
    webcam-effects/            <- live background blur/replace (same-model-both-engines template)
      pyproject.toml
      src/webcam_effects/
        matte.py                 <- pre/postprocessing + effect application, shared by both engines
        pipeline.py               <- capture->segment loop, shared by the CLI and the launcher
        cli.py
    voice-clone-studio/        <- zero-shot voice cloning (vendored-third-party-model template)
      pyproject.toml
      src/voice_clone_studio/
        _openvoice/               <- trimmed vendor copy of myshell-ai/OpenVoice (MIT), not on PyPI as one package
        voice_model.py            <- checkpoint loading + OpenVINO wrapper classes, shared by both engines
        pipeline.py               <- enroll-once/synthesize-many session, shared by the CLI and the launcher
        cli.py
    voice-assistant/           <- wake word + LLM + TTS (three-bricks-composed-plus-one-new-piece template)
      pyproject.toml
      src/voice_assistant/
        wake_word.py              <- openWakeWord (ONNX Runtime), the one genuinely new model in this brick
        session.py                 <- wake->listen->think->speak loop, composing live-translation + doc-qa + voice-clone-studio
        cli.py
    expense-extract/           <- receipts -> CSV (concurrent-two-device-pipeline template)
      pyproject.toml
      src/expense_extract/
        pipeline.py               <- two threads, one queue: OCR and LLM structuring run on two devices at once, not in turn
        parsing.py                 <- tolerant JSON extraction from the LLM's reply
        cli.py
    <next-brick>/
      pyproject.toml
      src/<next_brick>/
  launcher/                <- panther-lake-launcher: the web UI that runs the bricks
    src/launcher/
      registry.py           <- every demo card shown, including not-yet-built ones
      app.py                <- FastAPI app (REST + WebSocket)
      static/                <- vanilla HTML/CSS/JS front end, no build step
```

This is a single [`uv` workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
one `uv.lock` and one shared `.venv` at the repo root cover every brick and
`core` together, so bricks can freely depend on `core` and on each other
without separate installs -- `meeting-notes` is the proof: it has no
transcriber or LLM of its own, it composes `live-translation` and `doc-qa`
directly (see its README for the shape of that, and a real hallucination
bug that composition surfaced and how it got fixed).

## Setup

```bash
uv sync
```

This installs `core` plus every brick's default (portable) dependencies.
Heavier, hardware-specific dependencies -- like OpenVINO -- are kept as
per-brick *extras* so they're opt-in:

```bash
uv sync --extra openvino
```

(`uv` applies extras across all workspace members that declare them, so
this pulls in OpenVINO support for every brick that has it, not just one.)

## Running the launcher (the front UI)

```bash
uv run panther-lake-launcher
```

Opens a local web UI at `http://127.0.0.1:8765` listing every demo as a
card, grouped by category (Speech, Vision, Text, Audio). Available demos
have a working **Launch** button that opens a control panel (pick source /
engine / device, Start/Stop, live output); demos not built yet show as
**Coming soon** with a description, so the showcase reads as a full suite
from day one. See [launcher/README.md](launcher/README.md).

The launcher doesn't reimplement anything -- it's a thin FastAPI front end
that imports each available brick's package directly (e.g.
`live_translation.pipeline`) and drives it in a background thread.

## Running a brick from the command line

Each brick also installs its own standalone console script, if you'd rather
skip the UI:

```bash
uv run live-translate --list-devices
uv run live-translate --source system --engine openvino --compute-device NPU

uv run doc-qa ./my-notes --question "What did we decide about the launch date?"
uv run doc-qa ./my-notes --engine openvino --compute-device NPU

uv run object-detect --source screen
uv run object-detect --source screen --engine openvino --compute-device NPU

uv run screen-ocr --source screen
uv run screen-ocr --source screen --engine openvino --compute-device NPU --translate

uv run meeting-notes --source system
uv run meeting-notes --source system --engine openvino --compute-device NPU

uv run webcam-effects --show
uv run webcam-effects --show --effect replace --engine openvino --compute-device NPU

uv run voice-clone-studio --reference my_voice.wav --text "Hello from my own cloned voice."
uv run voice-clone-studio --record 15 --text "Hello from the NPU." --engine openvino --compute-device NPU

uv run voice-assistant
uv run voice-assistant --engine openvino --compute-device NPU

uv run expense-extract ./receipts
uv run expense-extract ./receipts --ocr-engine openvino --ocr-device NPU --llm-engine openvino --llm-device GPU
```

See each brick's own README for its specific options.

## Adding a new brick

1. `bricks/<name>/` with its own `pyproject.toml` (`name`, `[project.scripts]`
   entry point) and `src/<package>/`.
2. Depend on shared code instead of copying it -- and that's not limited to
   `core`. If another brick already wraps the model/runtime you need
   (Whisper, a local LLM, ...), depend on that brick directly rather than
   wrapping it a second time; `meeting-notes` depends on both
   `live-translation` and `doc-qa` this way, with zero transcriber/LLM code
   of its own:
   ```toml
   dependencies = ["pantherlake-ai-core", "live-translation", "doc-qa", ...]

   [tool.uv.sources]
   pantherlake-ai-core = { workspace = true }
   live-translation = { workspace = true }
   doc-qa = { workspace = true }
   ```
3. If the demo has a real, hardware-relevant local/cloud or CPU/NPU choice,
   follow the `live-translation` pattern: one module per backend
   (`transcriber_portable.py`, `transcriber_openvino.py`, ...) behind a tiny
   factory function, selected by an `--engine` CLI flag. Put anything
   reusable beyond this one demo (a capture/IO helper, a shared result type,
   a device-discovery helper) in `core/` instead of the brick, so the next
   brick doesn't reimplement it. Put the actual run loop (capture -> process
   -> emit) in its own `pipeline.py` function that takes an `on_result`
   callback, the way `live-translation` does -- that's what let the CLI and
   the launcher share one implementation instead of forking it.
4. Flip its entry in [`launcher/src/launcher/registry.py`](launcher/src/launcher/registry.py)
   from `status="planned"` to `status="available"`, and add routes +
   control panel for it in the launcher. Three templates to follow,
   depending on the demo's shape (see [launcher/README.md](launcher/README.md)
   for the detail): `live-translation`'s WebSocket/background-thread routes
   for a stream where every result matters, `doc-qa`'s (and `screen-ocr`'s)
   plain `run_in_threadpool` routes for a one-call-in-one-result-out demo,
   or `object-detection`'s single-overwritten-buffer routes for a
   continuous feed where only the newest result matters (e.g. more video).
   Prefer request/response unless the demo is genuinely a live feed --
   OCR, for instance, could have been built as a continuous per-frame
   stream like object detection, but a discrete "capture, get text back"
   action matches how OCR is actually used, and is simpler to boot.
5. `uv sync` from the root to pick up the new member. If the brick needs a
   dependency that requires a newer Python than the workspace's baseline
   (e.g. `object-detection` needs `>=3.11` for `openvino-model-api`, while
   the others run on `>=3.10`), bump just that brick's (and the launcher's,
   since it depends on every available brick) `requires-python` -- no need
   to raise it workspace-wide.

## Why OpenVINO for the Intel-hardware path

`faster-whisper` (CTranslate2), PyTorch, ONNX Runtime's default execution
provider, etc. are portable but CPU/CUDA-only -- they cannot target an Intel
NPU or iGPU. OpenVINO is Intel's inference runtime and is what actually
exposes `CPU` / `GPU` / `NPU` as selectable devices on a chip like Panther
Lake, which is the whole point of demonstrating *local* AI *on this
hardware* rather than just *on a laptop*.

## Versioning

The whole workspace shares one version number, in the [`VERSION`](VERSION)
file at the repo root (plain `MAJOR.MINOR.PATCH`, no `v` prefix in the
file itself). The launcher reads it at `GET /api/version` and shows it in
the page footer.

It bumps itself: [`.github/workflows/version-bump.yml`](.github/workflows/version-bump.yml)
runs on every push to `main` (in practice, every merged PR), bumps
`PATCH` by one, commits `VERSION` back with `[skip ci]`, and tags the
commit `vX.Y.Z`. Nothing to run by hand -- don't hand-edit `VERSION` in a
PR, since the bot commit after merge would just bump past whatever you set.
Bump `MAJOR`/`MINOR` yourself (edit `VERSION` directly, on `main`, outside
the normal PR flow) for an intentional jump; the bot only ever increments
`PATCH`.
