# Backlog

Fixes and ideas found via manual review of `logs/events.log` (see
`CLAUDE.md`), or filed directly by the user. Not a roadmap -- a holding
pen for things worth doing that aren't being done right now.

<!-- - [ ] **Title.** Short description. (filed YYYY-MM-DD, source) -->

- [ ] **The Activity Log forgets everything the launcher didn't see itself.**
  `events.py` keeps two records: a persisted append-only `logs/events.log`,
  and an in-memory `deque(maxlen=200)`. `GET /api/logs` -- the only thing
  the Activity Log viewer reads -- returns the deque, so restarting the
  launcher empties the panel while the file keeps going. Measured today:
  169 events on disk back to 2026-09-01, 13 available to the UI, oldest
  16:41 the same day. That makes the log much less useful for the thing it
  is for, since a first-run failure is often exactly what you restart
  after. Have `recent_events()` fall back to (or seed the deque from) the
  tail of `logs/events.log`; it is one JSON object per line and already
  sorted. Watch the file size while doing it -- nothing rotates it today
  (21KB after a week of demos, so this is not urgent, just unbounded).
  (filed 2026-09-08, from the activity-log review)

- [ ] **Smart-city counts run roughly 2.3x high, because tracks are lost and
  re-created.** The tracker is supposed to make an object count once, but it
  drops and re-acquires objects that are plainly still on screen, and each
  re-acquisition counts again. Measured over 60 consecutive frames per feed
  on CPU (so no device is involved): Shinjuku 4.7 objects on screen but 11
  new tracks, Shibuya 6.6/13, Abbey Road 4.3/11, Melbourne 10.4/24 -- a
  consistent ~2.0-2.5x, i.e. about one spurious re-count per object every
  couple of seconds. Live feeds make it obvious in a way local clips didn't:
  a 50-second run of the Melbourne camera reported "352 Cars/min" with about
  five cars visible, most of them stopped at a light. Worth checking whether
  the IoU gate is too tight for detections that flicker below the confidence
  threshold for a frame, and whether the time-based expiry (`time.monotonic`)
  suits a live stream, which is not paced to any frame rate the way a file is.
  (filed 2026-09-09, found while adding live feeds)

- [ ] **webcam-effects' segmentation model returns NaNs on the GPU.** Same
  frame, same model: CPU produces a clean matte (output sum 257.9, no NaNs),
  the GPU produces 473 NaNs and a sum of 40905. It is not the compiled-model
  cache -- it reproduces with no cache, a cold cache and a warm one alike, so
  it is unrelated to the GPU caching bug fixed in `ov_config_for` on
  2026-09-09. Suspect fp16 overflow in the selfie-segmentation ONNX graph on
  this driver; try `INFERENCE_PRECISION_HINT: f32` and see whether the matte
  matches CPU. Until then, webcam-effects on an explicit GPU device is
  quietly wrong rather than broken-looking. (filed 2026-09-09, found while
  investigating the smart-city GPU bug)
