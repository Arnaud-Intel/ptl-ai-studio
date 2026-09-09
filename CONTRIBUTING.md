# Contributing to Panther Lake AI Studio

This is where the developer-facing detail lives -- workspace layout, how
to add a new brick, and how versioning is automated. If you're looking
for what this project *is*, see the [root README](README.md) instead.

## Layout

```
local_demo/
  pyproject.toml          <- uv workspace root (no code of its own)
  core/                    <- pantherlake-ai-core: shared code every brick can use
    src/pantherlake_ai_core/
      audio.py             <- mic / system-audio (loopback) capture, speaker playback
      video.py               <- webcam / screen capture
      segmenter.py             <- lightweight energy-based voice-activity segmentation
      engine.py                 <- Engine enum, device discovery, and the shared engine/device
                                   defaults every CLI and the launcher resolve through
      model_cache.py             <- Hub model resolution (repo id or --model-path -> local path)
                                    with an "about to download" callback
      telemetry.py                <- CPU/GPU/NPU utilization reading
      types.py                     <- small shared result types (e.g. TranslationResult)
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
    smart-recall/              <- screen history search (continuous-concurrent-pipeline template)
      pyproject.toml
      src/smart_recall/
        pipeline.py               <- same two-device-at-once shape as expense-extract, driven by a timer instead of a file list
        change_detection.py        <- skip indexing a screen that hasn't visibly changed
        cli.py
    <next-brick>/
      pyproject.toml
      src/<next_brick>/
  launcher/                <- panther-lake-launcher: the web UI that runs the bricks
    src/launcher/
      registry.py           <- every demo card shown, including not-yet-built ones, plus what
                               each one's /devices route should enumerate
      app.py                <- FastAPI app (REST + WebSocket): one route set per brick on top of
                               shared helpers (resolve(), error_response(), mjpeg_stream(), ws_drain())
      *_runner.py            <- one per brick: owns its thread/session, reports phases + devices
      worker.py               <- the start/stop bookkeeping every threaded runner shares,
                                 including reporting "stopping" when a worker is mid-call
      errors.py                <- Conflict: the "not in a state to do that" error (-> HTTP 409)
      events.py / activity.py <- per-brick lifecycle phase (for /api/status) / device in use (for gauges)
      static/                  <- vanilla HTML/CSS/JS front end, no build step
```

This is a single [`uv` workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
one `uv.lock` and one shared `.venv` at the repo root cover every brick and
`core` together, so bricks can freely depend on `core` and on each other
without separate installs -- `meeting-notes` is the proof: it has no
transcriber or LLM of its own, it composes `live-translation` and `doc-qa`
directly (see its README for the shape of that, and a real hallucination
bug that composition surfaced and how it got fixed).

## Development setup

```bash
uv sync                    # portable engines only
uv sync --extra openvino   # also installs every brick's OpenVINO engine
```

`uv` applies extras across every workspace member that declares them, so
the second command pulls in OpenVINO support for every brick that has it,
not just one.

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

   The `core` helpers every brick is expected to use rather than re-implement:
   - `engine.resolve_engine(args.engine)` and `engine.default_device(engine)`
     for the `--engine` / `--compute-device` defaults. The launcher applies
     the same two, so "no choice" means the same thing in the UI and on the
     command line. `preferred_large_model_device()` is only for a model that
     genuinely needs a discrete GPU's VRAM (see code-review-assist).
   - `engine.print_devices(mics=..., cameras=..., ...)` for `--list-devices`.
   - `model_cache.resolve_snapshot(repo_id, local_dir=args.model_path,
     on_downloading=...)` / `resolve_file(...)` to turn a Hub repo id (or the
     user's `--model-path`) into a local path. Thread `on_downloading` (and
     `on_ready`, fired once the model is actually up) from your pipeline /
     session through to the factory: the launcher wires them to its
     "Downloading (first run only)" / "loading -> running" status.
   - `engine.ov_config_for(device)` as the config for every OpenVINO
     `compile_model` / `openvino_genai` pipeline, so an NPU compile lands in
     the one shared cache under `~/.cache/pantherlake-ai-studio/ov_cache`
     instead of a per-CWD `ov_cache/`. Don't add `CACHE_DIR` yourself: that
     helper deliberately gives the GPU no cache, because a cached GPU model
     comes back numerically wrong on this hardware (see its docstring) --
     silently, which is far worse than the second it saves.
4. Flip its entry in [`launcher/src/launcher/registry.py`](launcher/src/launcher/registry.py)
   from `status="planned"` to `status="available"`, declare which hardware
   lists its controls need (`devices=("microphones", "cameras", ...)`) and
   its samples module (`samples="my_brick.samples"`) -- that is all
   `GET /api/<id>/devices` needs -- and add routes + a control panel for it
   in the launcher. Routes go through `resolve()` for engine/device and
   `error_response()` for errors; a runner raises `launcher.errors.Conflict`
   for "already running" / "do X first" (a 409), `ValueError` /
   `FileNotFoundError` for bad input (a 400), and lets a real failure
   propagate (a 500 carrying the message). The panel itself is one
   `<section>` in `static/index.html` plus one entry in `static/app.js`'s
   `PANELS` table (a `StreamPanel` or a `Panel` config -- opening,
   rehydrating, the status pill, Start/Stop and the "Now running" strip are
   all inherited). Three server-side shapes to follow, depending on the
   demo's (see [launcher/README.md](launcher/README.md) for the detail):
   `live-translation`'s WebSocket/background-thread routes for a stream
   where every result matters, `doc-qa`'s (and `screen-ocr`'s) plain
   `run_in_threadpool` routes for a one-call-in-one-result-out demo, or
   `object-detection`'s single-overwritten-buffer routes for a continuous
   feed where only the newest result matters (e.g. more video).
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

## Tests

```bash
uv sync --extra openvino   # the dev group (pytest, httpx) comes with any sync
uv run pytest
```

(`uv sync` makes the environment match exactly what you ask for, so a
plain `uv sync` on a machine that had the `openvino` extra removes it
again -- keep passing `--extra openvino` when you have Intel hardware.)

`tests/` at the repo root covers what can be checked without hardware or a
model: the pure logic in `core` (the voice-activity segmenter, the
engine/device rules, telemetry's LUID attribution), each brick's small
deterministic helpers (expense JSON parsing, diff truncation, code-fence
stripping, the smart-city tracker and counters), and the launcher's API
contract through FastAPI's `TestClient` with every hardware probe
monkeypatched (`/api/demos`, every `/devices` route, the 400/409/500 error
policy, the 409-on-double-start contract). The same suite runs on every
push and pull request via
[`.github/workflows/test.yml`](.github/workflows/test.yml), on a plain
Ubuntu runner with no Intel hardware.

**CI installs with a plain `uv sync`, without the `openvino` extra**, so a
test module has to be *importable* without it. Keep `import openvino` (and
`openvino_genai`, `model_api`) inside the function or method that uses it,
the way every `*_openvino.py` backend already does -- a module-level one
turns into a collection error that takes the whole suite down, not a
single skipped test.

A new brick should bring a test for its pure parts -- a parser, a
post-processing step, a CLI argument helper -- and, if it adds routes, an
entry in `test_launcher_api.py`'s expectations. Anything that needs a
device or a model stays a manual check (see each brick's README for what
was verified and on what).

One platform quirk worth knowing before you bump a Python version: on
**Linux**, `voice-assistant` -> `openwakeword` -> `tflite-runtime`, and
that package publishes no wheels past **cp311**, so the workspace can't
install on Linux with 3.12+ (which is why CI pins 3.11). Windows never
installs `tflite-runtime` at all -- it's a `sys_platform == 'linux'`
dependency -- so a Windows dev machine is free to run 3.12 or later.

## Versioning

The whole workspace shares one version number, in the [`VERSION`](VERSION)
file at the repo root (plain `MAJOR.MINOR.PATCH`, no `v` prefix in the
file itself). The launcher reads it at `GET /api/version` and shows it in
the page footer; the badge at the top of the README pulls it live from
the repo's latest git tag.

It bumps itself: [`.github/workflows/version-bump.yml`](.github/workflows/version-bump.yml)
runs on every push to `main` (in practice, every merged PR), bumps
`PATCH` by one, commits `VERSION` back with `[skip ci]`, and tags the
commit `vX.Y.Z`. Nothing to run by hand -- don't hand-edit `VERSION` in a
PR, since the bot commit after merge would just bump past whatever you set.
Bump `MAJOR`/`MINOR` yourself (edit `VERSION` directly, on `main`, outside
the normal PR flow) for an intentional jump; the bot only ever increments
`PATCH`.
