# smart-city-monitor

**Accuracy status:** experimental new-track counts, not unique people/vehicles.
Track loss can overcount. Frame-wide strongest-overlap matching now prevents a
weaker early detection from stealing a stronger match, with a synthetic regression
fixture. The historical ~2.3× live-feed overcount has not been reproduced or
resolved on an annotated real clip yet; do not treat these counts as validated
traffic statistics. The API snapshot includes `count_basis` and `accuracy_warning`.

Counts pedestrians, bicycles, cars, motorcycles, buses, and trucks across
one or more video feeds at once — a local file, a camera on your network,
or a public live city camera — fully on-device. Each feed can be
pinned to its own compute device (e.g. one feed on the NPU, another on a
GPU), so N feeds genuinely run in parallel, correctly attributed on the
telemetry gauges.

This brick has **no detection model of its own**: it composes
[`object-detection`](../object-detection/README.md)'s
`engine_factory.create_detector` directly, the same way `meeting-notes`
composes `live-translation` and `doc-qa` rather than wrapping Whisper or
an LLM a third time. What it adds on top:

- **Tracking** ([`tracker.py`](src/smart_city_monitor/tracker.py)) — a
  small IoU-based greedy tracker, one instance per feed, so an object is
  counted once as it moves through frame instead of once per detection.
- **Counting** ([`pipeline.py`](src/smart_city_monitor/pipeline.py)) — a
  trailing-60-second count per class, continuously refreshed, plus a
  running total, **reported per feed**. The launcher shows one block per
  feed and the CLI prints one line each; the snapshot still carries a
  combined total for anything that wants it, but nothing displays it.
  Summing two cameras gives a number about nowhere, and *which* feed is
  busy is the thing worth reading when each is pinned to its own chip.
- **Real-time-paced file playback**
  ([`pantherlake_ai_core.video.stream_video_file_frames`](../../core/src/pantherlake_ai_core/video.py)) —
  a video file is played back at its own frame rate rather than as fast
  as the CPU/GPU can chew through it, so "N per minute" reflects one
  real minute of footage, not processing speed.
- **Live stream input** ([`sources.py`](src/smart_city_monitor/sources.py)) —
  a feed can equally be an RTSP/HTTP/HLS URL, read as it arrives and
  reopened if it drops; a video-clip URL its publisher replaces in place,
  played once per revision; or a YouTube live page, resolved to its
  underlying stream first. [`samples.py`](src/smart_city_monitor/samples.py)
  ships public city cameras in two sections — YouTube, and Other (London's
  TfL JamCams) — so the demo has something real to count without you
  sourcing footage. See **Where the live cameras come from** below.
- **N feeds, each on its own engine, model and device**
  ([`pipeline.py`](src/smart_city_monitor/pipeline.py)) — one detector per
  distinct `(engine, device, model)` among the feeds, shared by every feed
  asking for the same three so it loads exactly once, with each group's
  feeds running on their own threads. Feeds on different devices run, and
  load, genuinely in parallel — and because the engine is part of that key,
  one run can have YOLO11s on the NPU and DETR on the CPU at the same time.

## Setup

From the workspace root:

```bash
uv sync                    # portable engine only
uv sync --extra openvino   # also installs the OpenVINO engine
```

This brick requires **Python >= 3.11** (inherited from `object-detection`,
which needs it for `openvino-model-api`), same as `object-detection`
itself.

## Usage

List available inference devices:

```bash
uv run smart-city-monitor --list-devices
```

Monitor one video file:

```bash
uv run smart-city-monitor --source intersection.mp4 --engine openvino
```

Monitor a live camera — anything FFmpeg can open, including a YouTube
live page:

```bash
uv run smart-city-monitor --source "rtsp://camera.local/stream1" --engine openvino
```

Monitor a London traffic camera (a TfL JamCam clip, played once per revision):

```bash
uv run smart-city-monitor --source "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.03500.mp4" --engine openvino
```

Monitor two feeds at once, each pinned to a different chip, with a live
annotated window per feed:

```bash
uv run smart-city-monitor \
    --source "intersection.mp4|GPU.0" --source "crosswalk.mp4|NPU" \
    --engine openvino --show
```

Press `Ctrl+C` to stop.

## Options

| Flag | Description |
| --- | --- |
| `--source SOURCE[\|DEVICE]` | A feed to monitor, repeatable for multiple feeds: a video file, a stream URL (RTSP/HTTP/HLS), a video-clip URL, or a YouTube live page. Optional `\|DEVICE` suffix pins that one feed to a specific device (e.g. `clip.mp4\|GPU.0`); omitted, it uses `--compute-device`. Required (at least one). |
| `--engine {portable,openvino}` | Default inference backend for the run. Default: `openvino` if installed and a device is available, otherwise `portable`. (The CLI applies it to every feed; the launcher UI sets one per feed.) |
| `--compute-device NAME` | `openvino` engine only: default device for any `--source` without its own `\|DEVICE`. |
| `--model-path PATH` | Use a local model file/dir instead of downloading the default. |
| `--no-loop` | Stop each feed at end-of-file instead of restarting it from the beginning. |
| `--show` | Also open one live annotated window per feed (`cv2.imshow`) — off by default so this works headlessly. |
| `--list-devices` | List inference devices, then exit. |

## How it works

1. **Capture** — `sources.open_frames` decides what a feed actually is.
   A file goes through `pantherlake_ai_core.video.stream_video_file_frames`,
   paced to its own `CAP_PROP_FPS` and looping from frame 0 at EOF by
   default (so a short clip can stand in for a continuous camera). A URL
   to a whole video file served over HTTP — an `.mp4` a publisher
   overwrites in place, like a TfL JamCam — goes through
   `stream_refreshing_clip`: each revision plays once at its own frame
   rate, then the reader waits for the file's ETag to change. A clip ends,
   and replaying it would count the same objects again. Any other URL goes
   through `stream_live_frames`, which reads frames as they arrive and
   reopens the stream if it drops rather than seeking backwards. A YouTube
   URL is resolved to its underlying HLS stream by `yt-dlp` first, and
   re-resolved on a reconnect, since those links expire.
2. **Detect** — one `object_detection.engine_factory.create_detector`
   instance per distinct device among the running feeds, guarded by one
   lock per device so feeds sharing a device serialize their inference
   calls safely while feeds on different devices run truly concurrently.
3. **Track** — each feed's own `Tracker` matches this frame's detections
   (filtered to the classes below) against its live tracks by IoU,
   assigning a persistent id; an unmatched detection becomes a new track
   — the actual "count" event.
4. **Count** — each new track increments that feed's per-class trailing-
   60-second window and running total; the launcher and CLI report each
   feed's own numbers, and the snapshot still carries a combined total.

Relevant classes (present in both engines' vocabularies with matching
label strings — see `object-detection`'s own README for the COCO-91 vs
COCO-80 caveat): `person` → Pedestrians, `bicycle` → Bicycles, `car` →
Cars, `motorcycle` → Motorcycles, `bus` → Buses, `truck` → Trucks.
Anything else the detector reports is dropped before tracking, so neither
the drawn boxes nor the counts are cluttered with irrelevant classes.

## Choosing a device

Leave a feed on **AUTO and it will run about four times slower than it
needs to.** Measured on 30 identical frames of a night street camera, the
same YOLO11s detection:

| Device | Per frame | Objects found |
| --- | --- | --- |
| AUTO | 33.6 ms | 13.6 |
| **iGPU (GPU.0)** | **7.9 ms** | 13.0 |
| NPU | 20.8 ms | 13.6 |
| CPU | 20.7 ms | 13.6 |
| dGPU (GPU.1) | 48.4 ms | 13.8 |

Identical detections, four times the latency: AUTO is optimizing for
something other than "this frame, now". The launcher and the CLI therefore
default a feed to the integrated GPU rather than AUTO.

The discrete card is skipped on purpose, which surprises people who bought
one. It carries a **fixed** per-frame penalty -- +19.6 ms, +17.1 ms and
+18.2 ms for YOLO11 n, s and m. That it barely moves while the model
triples in cost is the tell: a card that were simply slower at compute
would fall further behind on the bigger model. A constant penalty is the
*data path* -- every frame is 2.6 MB that has to cross to the card and come
back, over a link that on this machine is narrow and lacks resizable BAR.
Its latency is also far less predictable (p10 11 ms, p90 32 ms) where the
iGPU holds 6.5-7.6 ms. Pin a feed to it if you want to watch the dGPU
gauge move; leave it on the iGPU if you want frames.

## Where the live cameras come from

The launcher's camera picker has two sections.

**YouTube** — public 24/7 city cameras, resolved to their HLS stream with
`yt-dlp`. YouTube can refuse a whole network at once with *"Sign in to
confirm you're not a bot"*. On 2026-09-11 it did exactly that for every
curated camera from this machine; neither the newest `yt-dlp` nor any of
its alternative player clients got past it without a signed-in session.
A feed that hits the wall now fails with a message that says so, instead
of suggesting a `yt-dlp` upgrade that wouldn't help.

**Other** — cameras that don't go through YouTube. Today these are four
Transport for London JamCams, plus a "London, two chips" pair that keeps
the dual-chip showcase working when YouTube is blocked. A JamCam is a
~10-second clip TfL replaces every few minutes, so each revision is played
once: one clip stays up long enough for ~30 replays, and replaying would
count the same vehicles on every pass. Measured against the Tower Bridge
camera: one 232-frame burst at 25.1 fps, then nothing until the clip
changed. Between clips the picture holds its last frame and the
per-minute counts decay. TfL deliberately downsamples the footage for GDPR,
so these cameras count vehicles well and people poorly.

JamCam footage is TfL open data — **Powered by TfL Open Data**. TfL staff
have confirmed on TfL's developer forum that running vehicle detection and
similar ML over the feeds is permitted, provided TfL is credited.

Any other RTSP, HLS (`.m3u8`) or HTTP video-clip URL works in a feed's URL
box too. California's Caltrans publishes around 1,300 public HLS freeway
cameras, but in testing only about one camera in ten opened reliably, so
none are curated.

## Notes / current limitations

- **The tracker is a pragmatic heuristic, not real multi-object-tracking
  or re-identification** — no appearance embedding, no motion model, just
  IoU-overlap matching between consecutive frames (same spirit as
  `smart-recall`'s `change_detection.py`). An object that leaves the
  frame and re-enters, or is fully occluded for more than ~1 second, gets
  a new id and is counted again. Good enough to stop massively
  over-counting a slow/stationary object; not a claim to solve MOT.
- **"Per minute" is a trailing 60-second count**, continuously refreshed
  — not an extrapolated instantaneous rate. It's only meaningful because
  playback is paced to the source video's own frame rate (see above); a
  video file played back faster than real time would inflate it.
- **The counts currently run high**, by roughly 2.3x on every feed
  measured — the tracker drops and re-acquires objects that are still on
  screen, and each re-acquisition counts again. Treat the numbers as
  relative (this feed is busier than that one) rather than absolute for
  now; see `BACKLOG.md`.
- **A live feed is somebody else's camera.** The *video* arrives over the
  network — detection still runs entirely on local silicon and no frame is
  sent anywhere — but a public stream can be renamed, rate-limited or taken
  down without notice, so a local file stays the option that always works.
  Reading a YouTube page needs `yt-dlp`, which is a declared dependency of
  this brick and occasionally needs upgrading when YouTube changes
  (`uv sync --upgrade-package yt-dlp`) — though not when YouTube is
  blocking the network outright; see **Where the live cameras come from**.
- **A feed dropped onto the launcher UI is copied, not referenced.** A
  browser never tells a page where a dropped file actually lives, so the
  only way drag-and-drop can work at all is to send the bytes to the
  launcher, which stages them under
  `~/.cache/pantherlake-ai-studio/uploads/` and uses that path. Typing a
  path into the feed's own box instead reads the file where it already is,
  with nothing duplicated — which is what you want for anything large.
  Nothing prunes the staging folder; delete it when it gets big.
- Inherits `object-detection`'s own label-vocabulary and performance
  caveats (see its README) for whichever engine is selected.
