# Panther Lake AI Studio (`panther-lake-launcher`)

The local web UI for discovering and running the demo bricks in this
workspace. No cloud dependency, no build step for the front end (plain
HTML/CSS/JS served as static files).

Visual identity: dominant black surfaces, Intel blue (`#0068b5`) for
primary actions and accents, a lighter blue (`#3fa9f5`) reserved for text
and border highlights. See `static/style.css`'s `:root` block for the token
list -- every muted text color there sits at 4.5:1 or better on every
surface, and every keyboard-reachable control has a visible focus ring.

The header's "Running on" strip is plain text badges (Dell, Intel Core
Ultra 3, Panther Lake NPU) rather than the actual company logos/badge
artwork -- swap in real logo files in `static/index.html` if/when you have
licensed access to them (Intel's "Core Ultra" badge in particular is
normally gated behind their partner co-marketing program).

## Run

From the workspace root:

```bash
uv run panther-lake-launcher
```

Opens `http://127.0.0.1:8765` in your default browser. `--host`, `--port`,
and `--no-browser` do what they say. Leave it running in a terminal;
`Ctrl+C` stops it.

## How the UI is put together

- **Home** is the card grid: one card per brick (name, tagline, and only a
  badge that actually distinguishes it -- "Discrete GPU" or "Coming soon").
  A card whose brick is running or loading says so on the card.
- **Opening a brick** (`#/brick/<id>`) swaps the grid for that brick's
  panel: its description, its controls, its output. Browser Back, Escape, or
  "All demos" returns to the grid with focus back on the card you came from.
- **Leaving a panel never stops the brick.** A run is a server-side thread;
  the panel is just a window onto it. The **"Now running" strip** under the
  header (polled from `/api/status` every 1.5s, devices from
  `/api/telemetry`) shows one chip per brick that is loading or running,
  wherever you are; click a chip to jump to it. Reopening a panel
  rehydrates from that same status -- Start/Stop reflect the real state, a
  video stream reattaches, the WebSocket reconnects -- and the bricks with
  their own state route (`doc-qa`, `voice-clone-studio`, `smart-recall`)
  use it to pick up an index, an enrolled voice, or a recording that
  predates the visit (or a page reload).
- **The status pill** next to each Start/Run button mirrors the brick's
  real backend phase while something is in flight: "Downloading model
  (first run only)", "Loading model", the running message, or the error --
  never an optimistic "Running..." before the model is actually up.
- **The header gauges** (CPU, one per GPU, NPU) are the only gauges; each
  is labeled with whichever brick -- and stage -- is driving that device.

### `static/app.js` in one paragraph

`Panel` is one brick's controller: it fetches `/api/<id>/devices` once,
wires the engine/device selects (`wireEngineAndDevice` -- the OpenVINO
option is enabled only when the brick has a device, the device list follows
the engine, GPU ids get friendly names), rehydrates, and runs
request/response actions through `run()` (disable the button, mirror the
backend phase into the pill, show the result or the error). `StreamPanel`
adds Start/Stop, the WebSocket (with reconnect while the panel is open) or
the MJPEG `<img>` plus a polled side channel, and the running-state
bookkeeping. The `PANELS` table at the bottom holds the thirteen configs;
each carries only what is genuinely unique to that brick -- how to fill its
extra selects, the request body, how to render its messages -- typically
30-80 lines.

## What the server does

- `GET /api/demos` -- the demo registry (`src/launcher/registry.py`): every
  card, `status: "available" | "planned"`, plus which device lists and
  samples module each brick's controls need.
- `GET /api/{demo_id}/devices` -- one registry-driven route for every
  brick: `openvino_devices` always, plus whichever of `microphones`,
  `speakers`, `cameras`, `screens`, `wake_words` its registry entry asks for,
  and `samples` when it has a samples module.
- `GET /api/version`, `GET /api/status`, `GET /api/logs`,
  `GET /api/telemetry`, `GET /api/system/gpu-devices` -- version, per-brick
  lifecycle phase, the persisted event log, utilization + which brick is on
  which device, and the machine's GPUs (for gauges and dropdown labels).
- Per brick, one of three shapes -- the launcher only owns wiring, the demo
  logic stays in the brick's own `pipeline.py`/`session.py`:
  - **Stream** (`live-translation`, `meeting-notes`, `voice-assistant`,
    `expense-extract`, `smart-recall`): `POST .../start` / `.../stop` run the
    brick on a background thread (`<brick>_runner.py`); `WS /ws/<id>` drains
    its queue to the page (`ws_drain` in `app.py` -- one shared queue per
    brick, so a second tab on the same brick would only get every other
    message; the disconnect is watched concurrently so a closed tab never
    swallows the next message).
  - **Latest-frame stream** (`object-detection`, `webcam-effects`,
    `smart-city-monitor`): the same start/stop, but the thread overwrites
    one JPEG buffer in place and `GET .../stream` serves it as MJPEG
    (`mjpeg_stream` -- a 404 when the brick isn't running), with a polled
    side route (`/detections`, `/stats`, `/counts`) for the text beside the
    video. `webcam-effects` additionally takes `POST .../effect` while
    running, changing the look on the next frame.
  - **Request/response** (`doc-qa`, `screen-ocr`, `voice-clone-studio`,
    `code-review-assist`, `html-creator`): plain routes that run the brick's
    blocking call via `run_in_threadpool`. `meeting-notes` and
    `smart-recall` are both shapes at once (a stream plus notes generation /
    search on the same session).
- Every route resolves engine/device through `resolve()` -- the same rule
  the CLIs use, so the UI and the command line agree on what "no choice"
  means -- and reports errors as `{"error": message}` under one policy
  (`error_response`): 400 for a bad input, 409 when the brick isn't in a
  state to do that (`launcher.errors.Conflict`), 500 with the message for a
  failure inside a model. A malformed body gets the same shape (400), not
  FastAPI's 422 detail list.

## Hardware telemetry

The header shows a live CPU / GPU / NPU gauge strip and highlights whichever
gauge a running brick is actually using (e.g. starting Live Speech
Translation on the NPU lights up the NPU gauge with "Live Speech
Translation" underneath it). This is the answer to "what silicon is this
actually using" -- the whole point of the showcase.

How it's real, not decorative:

- **CPU** comes from `psutil`, cross-platform, cheap.
- **GPU/NPU** come from Windows' own "GPU Engine" performance-counter
  category (the same one Task Manager reads) -- see
  [`pantherlake_ai_core/telemetry.py`](../core/src/pantherlake_ai_core/telemetry.py).
  There's no per-vendor API that reports "NPU %" directly, so it
  *classifies* the counter instances by behavior: an adapter whose engine
  instances are only ever "compute" type is the NPU (NPUs don't do
  graphics); each GPU is matched to its OpenVINO device id through the
  adapter LUID, so a machine with an iGPU and a discrete card gets one
  gauge each, tracked independently. Verified against real hardware.
- **Which brick is active** comes from [`activity.py`](src/launcher/activity.py):
  each runner records `{engine, device}` for the exact device string it
  passed to the brick, for the duration of the call -- ground truth, not
  inferred. A device string like `AUTO` or `cuda` isn't pinned to one gauge
  (OpenVINO's `AUTO` can pick any device internally), so it just won't
  highlight anything -- still honest, no fabricated attribution. Keyed by
  `(demo_id, stage)`: `expense-extract` and `smart-recall` run two stages
  on two devices at once, `smart-city-monitor` one stage per feed, and each
  needs its own gauge label ("Expense Report Extractor (OCR)" on one,
  "(Structuring)" on another, simultaneously).
- **Non-Windows / query failure**: `available: false`, GPU/NPU render as
  "N/A" rather than a fake 0%.

The GPU/NPU query itself is slow (the OS's wildcard expansion over "GPU
Engine" instances takes ~1-3s), so it isn't queried per `/api/telemetry`
call: a background thread ([`telemetry_poller.py`](src/launcher/telemetry_poller.py))
samples it every 3s and the route returns the cached snapshot. The page
polls that route every 2s.

## Adding UI for a brick once it's built

1. Flip its `registry.py` entry to `status="available"` and declare what its
   controls need: `devices=("microphones", "cameras", ...)` and
   `samples="my_brick.samples"`. That is all `GET /api/<id>/devices` needs.
2. Add its routes in `app.py`, following the shape that matches the demo
   (see **What the server does**): a `<brick>_runner.py` on a background
   thread for a stream or a latest-frame feed, plain `run_in_threadpool`
   routes for request/response -- prefer request/response unless the demo is
   genuinely a live feed. Go through `resolve()` and `error_response()`;
   raise `launcher.errors.Conflict` for "already running" / "do X first".
3. In `static/index.html`, add a `<section id="<prefix>-panel"
   class="brick-panel" hidden>` inside `#view-brick` with the controls
   (`.controls` > `.field`s), a `.run-row` holding the buttons and a
   `<span id="<prefix>-status" class="run-status">`, and the output boxes.
4. In `static/app.js`, add an entry to `PANELS`: `new StreamPanel({...})`
   with `transport: "ws"` or `"mjpeg"` (+ `video: "<img id>"`), or
   `new Panel({...})` for request/response. Fill in `populate(data)` (call
   `wireEngineAndDevice` and whatever else the controls need), `body()` for
   a stream's start request, `onMessage` / `onRunning` for its output, or a
   `wire()` that binds the action button to `this.run({...})`. Everything
   else -- opening, rehydrating, the status pill, Start/Stop, reconnecting,
   the running strip -- is inherited.
5. In the runner, call `activity.set_active(...)` / `clear_active(...)`
   around the inference and `events.set_phase(...)` at the loading ->
   running -> done boundaries (with `stage=` if the brick runs several
   things at once) -- that is what the gauges, the status pill, and the
   running strip are all reading.
