# Panther Lake AI Studio (`panther-lake-launcher`)

The local web UI for discovering and running the demo bricks in this
workspace. No cloud dependency, no build step for the front end (plain
HTML/CSS/JS served as static files).

Visual identity: dominant black surfaces, Intel blue (`#0068b5`) for
primary actions and accents, a lighter blue (`#3fa9f5`) reserved for hover
states. See `static/style.css`'s `:root` block for the token list.

The header's "Running on" strip is currently plain text badges (Dell,
Intel Core Ultra 3, Panther Lake NPU) rather than the actual company
logos/badge artwork -- see `static/index.html`'s `.hardware-strip`. Swap in
real logo files there if/when you have licensed access to them (Intel's
"Core Ultra" badge in particular is normally gated behind their partner
co-marketing program, so don't just pull one off the web).

## Run

From the workspace root:

```bash
uv run panther-lake-launcher
```

Opens `http://127.0.0.1:8765` in your default browser. Leave it running in
a terminal; `Ctrl+C` stops it.

## What it does

- `GET /api/demos` — returns the demo registry (`src/launcher/registry.py`):
  every demo card the UI shows, `status: "available" | "planned"`.
- `GET /api/version` — reads the workspace-root `VERSION` file and returns
  `{"version": "X.Y.Z"}`; the page footer fetches this once at load and
  shows it as `vX.Y.Z`. See [CONTRIBUTING.md](../CONTRIBUTING.md#versioning)
  for how that file gets bumped.
- For the **available** `live-translation` demo:
  - `GET /api/live-translation/devices` — real microphones, output devices,
    and OpenVINO devices (`CPU`/`GPU`/`NPU`) detected on this machine, used
    to populate the control panel's dropdowns.
  - `POST /api/live-translation/start` / `/stop` — starts/stops
    `live_translation.pipeline.run(...)` on a background thread.
  - `WS /ws/live-translation` — streams each translated utterance to the
    page as it happens.
- For the **available** `voice-assistant` demo:
  - `GET /api/voice-assistant/devices` — microphones, OpenVINO devices, and
    the available wake words, for the control panel's dropdowns.
  - `POST /api/voice-assistant/start` / `/stop` — starts/stops
    `voice_assistant.session.run(...)` on a background thread, same
    background-thread/queue shape as `live_translation_runner.py`.
  - `WS /ws/voice-assistant` — streams three event types as they happen:
    `wake` (the wake word was heard), `heard` (the transcribed question),
    and `reply` (the spoken answer's text) -- the page renders each
    differently rather than treating them as one generic "result" like
    `live-translation`'s single-shape stream.
  - Speech synthesis and playback both happen **inside** the background
    thread, through this machine's own speakers
    (`pantherlake_ai_core.audio.play`) -- there's no audio route to poll or
    stream, unlike `voice-clone-studio`'s synthesize endpoint, because the
    server here *is* the device doing the talking, not a remote client
    that needs the clip delivered to it.
- For the **available** `doc-qa` demo:
  - `GET /api/doc-qa/devices` — OpenVINO devices detected, for the control
    panel's compute-device dropdown.
  - `POST /api/doc-qa/ingest` — (re)builds/loads the index for a folder path
    typed into the panel, via `doc_qa.pipeline.DocQASession`. Request/response,
    not streamed (indexing a folder of notes is seconds, not a live feed).
  - `POST /api/doc-qa/ask` — answers one question against the current index.
  - Both routes run the brick's blocking calls via
    `fastapi.concurrency.run_in_threadpool` (see `doc_qa_runner.py`) so a
    slow embed/generate call doesn't block the event loop -- there's no
    background thread/queue to manage here, unlike `live-translation`,
    because this isn't a continuous stream.
- For the **available** `object-detection` demo:
  - `GET /api/object-detection/devices` — cameras, screens, and OpenVINO
    devices detected, for the control panel's dropdowns.
  - `POST /api/object-detection/start` / `/stop` — starts/stops
    `object_detection.pipeline.run(...)` on a background thread.
  - `GET /api/object-detection/stream` — an MJPEG stream (`multipart/x-mixed-replace`)
    of the latest annotated frame; the modal points an `<img>` tag straight
    at it. `GET /api/object-detection/detections` is polled separately for
    the text list beside the video.
  - Neither a queue nor `run_in_threadpool`: video only needs the *newest*
    frame (an old one is worthless the instant a new one exists), so
    `object_detection_runner.py` just keeps one JPEG buffer the background
    thread overwrites in place, and the MJPEG generator polls that buffer
    at its own pace. A third distinct shape from the other two demos —
    see the decision list under "Adding UI for a brick" below.
- For the **available** `screen-ocr` demo:
  - `GET /api/screen-ocr/devices` — cameras, screens, and OpenVINO devices
    detected, for the control panel's dropdowns.
  - `POST /api/screen-ocr/extract` — captures one screen or webcam frame
    server-side and runs it through `screen_ocr.pipeline.OcrSession`.
  - `POST /api/screen-ocr/extract-upload` — same, but for a photo the
    browser uploads (`multipart/form-data`, via FastAPI's `UploadFile`/
    `Form`; needs the `python-multipart` dependency). Both routes run
    through `run_in_threadpool` like `doc-qa` -- one call in, one result
    out, no persistent stream.
- For the **available** `meeting-notes` demo:
  - `GET /api/meeting-notes/devices` — microphones, output devices, and
    OpenVINO devices detected, for the control panel's dropdowns.
  - `POST /api/meeting-notes/start` / `/stop` — starts/stops
    `meeting_notes.session.MeetingSession.transcribe(...)` on a background
    thread (same shape as `live_translation_runner.py`).
  - `WS /ws/meeting-notes` — streams each transcribed line to the page,
    same shape as `/ws/live-translation`.
  - `POST /api/meeting-notes/generate` — runs `MeetingSession.generate_notes()`
    via `run_in_threadpool`, same shape as `/api/doc-qa/ask`. Works whether
    or not transcription is still running, since the session (and its
    transcript) outlives the capture thread.
  - This demo genuinely needs **both** shapes at once -- see
    `meeting_notes_runner.py`, which is the launcher-side proof that the
    "pick one template" framing below is a starting point, not a rule: it
    runs a background-thread/queue stream for the live transcript and a
    `run_in_threadpool` request/response for on-demand notes, on the same
    underlying session.
- For the **available** `webcam-effects` demo:
  - `GET /api/webcam-effects/devices` — cameras and OpenVINO devices
    detected, for the control panel's dropdowns.
  - `POST /api/webcam-effects/start` / `/stop` — starts/stops
    `webcam_effects.pipeline.run(...)` on a background thread, same
    latest-frame-buffer shape as `object_detection_runner.py`.
  - `GET /api/webcam-effects/stream` — an MJPEG stream of the latest
    frame with the current effect already applied.
  - `POST /api/webcam-effects/effect` — changes the effect (`blur` vs
    `replace`) and replace-color **while capture keeps running**. This
    is a genuinely new shape on top of latest-frame-stream: the
    background thread doesn't just overwrite a frame buffer, it reads a
    small piece of *mutable state* (`self._effect` / `self._color` in
    `webcam_effects_runner.py`) on every frame, so a control the operator
    changes mid-run takes effect on the very next frame with no
    restart. `object-detection` has no equivalent -- its overlay isn't
    a runtime choice.
  - `GET /api/webcam-effects/stats` — polled separately for the current
    person-coverage percentage shown beside the video.
- For the **available** `voice-clone-studio` demo:
  - `GET /api/voice-clone-studio/devices` — microphones and OpenVINO
    devices detected, for the control panel's dropdowns.
  - `POST /api/voice-clone-studio/enroll-record` — records N seconds from
    this machine's own default microphone (via `pantherlake_ai_core.audio`,
    the same server-side capture every other demo here uses -- not the
    browser's own mic APIs) and enrolls it as the target voice.
  - `POST /api/voice-clone-studio/enroll-upload` — same, but for an audio
    file the browser uploads (`multipart/form-data`), same shape as
    `screen-ocr`'s `/extract-upload`.
  - `GET /api/voice-clone-studio/status` — whether a voice is currently
    enrolled, so reopening the modal can re-enable the synthesize controls
    without re-enrolling.
  - `POST /api/voice-clone-studio/synthesize` — synthesizes text in the
    enrolled voice and returns the WAV bytes directly as the response body
    (`media_type="audio/wav"`), rather than a JSON payload with a URL to
    fetch separately. The page does `fetch(...).then(r => r.blob())` and
    points an `<audio>` element's `src` at `URL.createObjectURL(blob)` --
    no server-side temp file to track or clean up.
  - All four routes are request/response through `run_in_threadpool`
    (`voice_clone_studio_runner.py`, same shape as `doc_qa_runner.py`) --
    enrollment and synthesis are each one call in, one result out, not a
    continuous stream, so there's no background thread here at all.
- For the **available** `expense-extract` demo:
  - `GET /api/expense-extract/devices` — OpenVINO devices detected, for
    the control panel's **two** compute-device dropdowns (OCR and LLM
    are picked independently, unlike every other demo here).
  - `POST /api/expense-extract/start` / `/stop` — starts/stops
    `expense_extract.pipeline.run(...)` on a background thread, same
    shape as `live_translation_runner.py`. The pipeline call itself spans
    two more threads internally (OCR and LLM structuring, genuinely
    concurrent) -- see `expense_extract_runner.py` for why it marks
    *both* stages active in `activity.py` for the whole call rather than
    one device around one blocking call like every single-stage runner.
  - `WS /ws/expense-extract` — streams `ocr_progress` (which file is
    being read), `structured` (one receipt's finished expense line), and
    `done` (final counts and total) as they happen.
- **Planned** demos render as disabled "Coming soon" cards with their
  description, so clicking one doesn't error — it's just not wired up yet.
- `GET /api/telemetry` — CPU/GPU/NPU utilization plus which demo (if any) is
  currently driving each device. See **Hardware telemetry** below.

## Hardware telemetry

The header shows a live CPU/GPU/NPU gauge strip, and highlights whichever
one a running demo is actually using (e.g. starting Live Speech Translation
with `--engine openvino --compute-device NPU` lights up the NPU gauge with
"Live Speech Translation" underneath it). This is the answer to "what
silicon is this actually using" -- the whole point of the showcase.

The same three gauges also appear as a **sticky footer inside every demo
modal** (`.modal-footer`, pinned to the bottom of the panel via
`position: sticky` so it stays visible even if the modal's content
scrolls) -- that's the one you actually watch while a demo is running,
without needing to see past the modal to the dimmed header behind it.
It's not a second telemetry system: `static/index.html` defines the three
gauges once in `#telemetry-footer-template`, and `initTelemetryFooters()`
(called from `initTelemetry()` in `app.js`) clones it into every element
with the `.demo-modal` class at page load. `renderTelemetry()` then
updates *every* `.telemetry-gauge[data-device="..."]` on the page via
`querySelectorAll` on each poll -- header and every modal footer move in
lockstep from one `/api/telemetry` response, there's nothing per-modal to
keep in sync by hand. Adding a fifth demo modal only requires giving its
`.modal` the `demo-modal` class; the footer and its live updates come for
free.

How it's real, not decorative:

- **CPU** comes from `psutil`, cross-platform, cheap.
- **GPU/NPU** come from Windows' own "GPU Engine" performance-counter
  category (the same one Task Manager reads for its GPU/NPU graphs) --
  see [`pantherlake_ai_core/telemetry.py`](../../core/src/pantherlake_ai_core/telemetry.py).
  There's no per-vendor API that reports "NPU %" directly, so it
  *classifies* the counter instances by behavior instead of guessing a
  fixed ID: an adapter (LUID) whose engine instances are *only* ever
  "compute" type is the NPU (NPUs don't do graphics); the adapter with the
  *most distinct engine types* (3D, video, copy, compute, ...) is the
  primary GPU. Verified against this machine's real Intel NPU + Arc GPU
  before shipping -- it isn't a guess.
- **Which demo is active** comes from [`activity.py`](src/launcher/activity.py):
  each runner records `{engine, device}` for the exact device string it
  passed to the brick, for the duration of the call -- ground truth, not
  inferred. A device string like `AUTO` or `cuda` isn't pinned to one
  gauge (OpenVINO's `AUTO` can pick any device internally, and there's no
  CUDA gauge here), so it just won't highlight anything -- still honest,
  no fabricated attribution.
  - Keyed by `(demo_id, stage)`, not just `demo_id` -- every demo before
    `expense-extract` only ever has one thing running at a time, so
    `stage` defaults to a fixed name and those runners never pass it.
    `expense-extract` genuinely runs OCR and LLM structuring
    concurrently on two different devices for the whole call, and needs
    *both* showing as active at once -- `set_active(..., stage="ocr")`
    and `set_active(..., stage="llm")` side by side, not one overwriting
    the other. `snapshot()` returns a list for this reason (one entry
    per active stage) rather than a dict keyed by demo id; `app.js`'s
    `renderTelemetry()` labels each gauge from whichever entries target
    it, appending the stage label in parentheses when there is one --
    e.g. "Expense Report Extractor (OCR)" on the GPU gauge and "Expense
    Report Extractor (Structuring)" on the NPU gauge, at the same time.
- **Non-Windows / query failure**: `available: false`, GPU/NPU render as
  "N/A" rather than a fake 0%.

The GPU/NPU query itself is genuinely slow (the OS's wildcard expansion
over "GPU Engine" instances takes ~1-3s, not something to pay per web
request), so it isn't queried per `/api/telemetry` call: a background
thread ([`telemetry_poller.py`](src/launcher/telemetry_poller.py)) samples
it every 3s and the route just returns the cached snapshot. The panel
polls that route every 2s, so it's a bit laggier than a "live" gauge, but
never blocks anything else in the app.

## Adding UI for a brick once it's built

1. Flip its `registry.py` entry to `status="available"`.
2. Pick the shape that matches the demo, and add routes in `app.py` that
   call into the brick's package -- keep the actual demo logic in the brick
   itself (e.g. a `pipeline.py`), the launcher should only own wiring:
   - **Continuous stream** (a live feed of results, like `live-translation`):
     a background thread pushes into an `asyncio.Queue`, drained by a
     `WebSocket` route. See `live_translation_runner.py` + `/ws/live-translation`.
   - **Request/response** (one call in, one result out, like `doc-qa`): a
     plain route that runs the brick's blocking call via
     `fastapi.concurrency.run_in_threadpool` and returns JSON. See
     `doc_qa_runner.py` + `/api/doc-qa/ask`. Simpler -- prefer this unless
     the demo is genuinely a live feed.
   - **Latest-frame stream** (a continuous feed where only the newest
     result matters, like `object-detection`'s video): a background thread
     overwrites one buffer in place; an HTTP route (MJPEG for video, or
     just a GET returning the latest value for anything else) reads it at
     its own pace. See `object_detection_runner.py` + `/api/object-detection/stream`.
     Use this instead of the WebSocket/queue shape when older results are
     simply stale, not something a client needs delivered.
3. In `static/index.html`, add a modal (or extend the shared one) with that
   demo's controls, and in `static/app.js` an `openDemo()` branch for its
   id that opens it and wires it up the way `openLiveTranslation()` (stream),
   `openDocQA()` (request/response), or `openObjectDetection()` (latest-frame
   stream) does. Give the modal's `.modal` element the `demo-modal` class --
   that's the only thing needed to get the live telemetry footer (see
   **Hardware telemetry** above); don't hand-write the gauge markup.
4. Call `activity.set_active("<demo-id>", engine=..., device=...)` around
   the brick's actual inference call, and `activity.clear_active(...)` when
   it's done (`finally`) -- that's what lets the telemetry gauges say which
   demo is using a device, instead of just a bare percentage. Add the demo's
   display name to `DEMO_NAMES_BY_ID` in `static/app.js` too.

## Notes / current limitations

- One `live-translation` run at a time; starting a second while one is
  running returns `409`.
- The result WebSocket is a single shared queue, not a broadcast -- fine
  for one operator with one browser tab, but a second concurrently open
  tab would only get every other message rather than a full duplicate
  stream. Worth revisiting (e.g. an `asyncio.Queue` per connection) if this
  becomes a multi-viewer kiosk. (`object-detection`'s MJPEG stream doesn't
  have this problem -- each `GET /api/object-detection/stream` connection
  reads the shared frame buffer independently, so multiple tabs each get
  every frame, not a split.)
