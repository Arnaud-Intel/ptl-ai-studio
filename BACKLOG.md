# Backlog

Fixes and ideas found via manual review of `logs/events.log` (see
`CLAUDE.md`), or filed directly by the user. Original reports are preserved
below. The [implementation roadmap](#implementation-roadmap--2026-09-10)
turns them and the project review findings into sequenced, reviewable work.

<!-- - [ ] **Title.** Short description. (filed YYYY-MM-DD, source) -->

## Original reports

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

- [ ] **Smart-city tops out near 25 fps when the pipeline can do ~70.** With
  detection on the iGPU the per-frame work measured standalone is 9.3 ms
  detect, 3.9 ms capture, 3.1 ms JPEG encode and 1.4 ms draw -- about 18 ms,
  so ~55 fps against a source that offers ~30. Through the launcher the
  producer reached 14.3 fps, and 24.5 fps once the telemetry device poll
  stopped running back to back (fixed 2026-09-09). The remaining gap to the
  source rate is unexplained: it is not the GIL from telemetry parsing
  (0.6 ms per cycle), not the MJPEG poll (two concurrent clients each still
  got the full rate, so delivery is not the ceiling), and not the source
  (raw capture sustains 30+ fps). Worth profiling the feed thread inside the
  running launcher rather than standalone. Encoding a 194 KB JPEG per frame
  at quality 80 is the next-largest cost after detection and is pure display
  overhead -- a smaller streamed frame would buy some of it back.
  (filed 2026-09-09, from the smart-city frame-rate investigation)

## Implementation roadmap — 2026-09-10

### Trustworthy-output implementation update

Implemented in the working tree on 2026-09-10 (see
[validation notes](docs/TRUSTWORTHY_OUTPUT.md)). Checkboxes below retain their
full acceptance meaning, including hardware gates.

- **R01:** device identity and selection checks implemented; the false USB/NPU
  label is covered by regression tests. Real Intel NPU name validation is pending.
- **R02:** decimal/currency parsing, field-review status, grouped totals, CSV and
  WebSocket changes implemented with automated fixtures. Manual correction remains
  through the CSV; no in-app approval flow is claimed.
- **R03:** GPU/AUTO webcam gate, finite/range mask checks, frame-wide tracker matching
  and experimental-count disclosure implemented. Annotated live-clip accuracy and
  CPU/GPU model qualification remain open; the reported overcount is not marked fixed.
- **R04:** bounded restart recovery, malformed-record tolerance, rotation and
  concurrent-write tests implemented. History is restored without resurrecting
  previously active workers.

Based on [the project review](docs/PROJECT_REVIEW.md), version 0.2.36,
commit `db0a7f7`. Completed implementation items are checked; remaining gates
are described in the update above. This is a relative
timeline from implementation kickoff, not a commitment to calendar dates.

**Capacity assumption:** one full-time engineer, with part-time product/QA
support and access to the target Intel hardware. Estimates include coding
and focused verification; they total **40–57 engineer-days**. Twelve weeks
provide about 60 working days, leaving 3–20 days for integration and unknowns.
The upper estimate leaves little contingency: replan after week 2 if hardware
investigations grow. Waiting for devices/drivers is not included.

**Priority:** P1 = correctness, data integrity, or reliability gate;
P2 = usability, performance, or maintainability after those gates.
There is no demonstrated emergency requiring a P0 in this review.
The engineer owns delivery; the product owner approves behavior and targets;
QA supplies fixture review and target-machine verification.

| Window | Work, in suggested order | Effort | Exit gate |
| --- | --- | --- | --- |
| Weeks 1–2: trustworthy output | R01 hardware identity → R02 receipt correctness → R03 vision reproduction/fallback → R04 durable logs | 8–10 days | No false NPU identity; locale/currency fixtures pass; known broken vision configurations fixed or explicitly unavailable; logs survive restart |
| Weeks 3–4: reliable sessions and indexes | R05 event delivery/shutdown → R06 index freshness → R07 atomic persistence | 8–11 days | Disconnect/reconnect and shutdown checks pass; edited/deleted documents are reflected; interrupted writes recover safely |
| Weeks 5–6: controlled first use | R08 recording retention → R09 request boundaries → R10 setup/model readiness | 9–13 days | Recording policy and quotas work; invalid requests are bounded; fresh-install and prepared-offline paths are verified |
| Weeks 7–8: sustained workloads | R11 resource lifecycle → R12 measured performance | 6–9 days | Repeated starts do not grow retained resources without bound; representative concurrent session passes; performance report includes accuracy |
| Weeks 9–10: usability and maintainability | R13 guided controls → R14 module extraction | 4–6 days | New user completes the sample flow; keyboard navigation works; extracted modules preserve behavior |
| Weeks 11–12: release confidence | R15 platform/quality checks → R16 release gate and pilot | 5–8 days | Target matrix and four pilot flows pass; version/tag corresponds to tested commit; limitations are published |

Tests for each fix belong in that fix. R15 expands the platform and quality
matrix; it is not a reason to defer testing until week 11. Leave unused time
inside each phase available for integration and unexpected hardware work.

### Weeks 1–2 — trustworthy output

- [ ] **R01 · P1 · Correct hardware identity and capability labels.**
  Replace substring-based NPU discovery with validated device identity;
  distinguish hardware branding, telemetry visibility, and engine support.
  Current live API labels `Logitech USB Input Device` as the NPU because
  `NPU` matches inside `Input`; the header also hardcodes Panther Lake badges.
  **Done:** negative fixtures for USB input devices pass; a real NPU is named
  correctly on target hardware; unavailable devices say unavailable; device
  selections are checked against the chosen engine.
  **Estimate:** 1–2 days. **Depends on:** none.
  **Files:** `core/src/pantherlake_ai_core/telemetry.py`, `engine.py`, launcher
  device APIs and static header. (filed 2026-09-10, live UI/API and code review)

- [x] **R02 · P1 · Make receipt amounts and currencies trustworthy.**
  Preserve currency, handle unambiguous decimal/group separators, reject
  ambiguous or non-finite amounts, and represent money with decimal precision.
  Remove dollar-only rendering and cross-currency totals. Replace best-guess
  acceptance with a visible review state for missing/uncertain fields.
  **Done:** fixtures cover `12,50 EUR`, `1 234,56 EUR`, `1,234.56 USD`, refunds,
  ambiguous separators, invalid dates and unknown currency; totals stay grouped
  by currency and only reviewed/valid amounts contribute.
  **Estimate:** 3 days. **Depends on:** none.
  **Files:** `bricks/expense-extract/src/expense_extract/{parsing,types,pipeline}.py`,
  `launcher/src/launcher/expense_extract_runner.py`, frontend expense panel.
  (filed 2026-09-10, code review)

- [ ] **R03 · P1 · Reproduce vision defects and fix or gate affected paths.**
  Covers original smart-city overcount and webcam GPU NaN reports above.
  Preserve a small annotated clip/frame corpus, record model/runtime/driver,
  add finite-mask validation, investigate precision and tracker behavior.
  Do not treat an IoU tweak or FP32 as a proven solution before measurement.
  **Done:** supported configurations produce finite valid masks; annotated
  counting fixtures meet an agreed threshold (initial target: ≤10% aggregate
  count error per clip with at least 20 reference objects); unsupported paths
  are disabled with a tested fallback. Short clips use absolute error too.
  **Estimate:** 3 days, timeboxed; deeper fixes are re-estimated after reproduction.
  **Depends on:** R01 for capability gating and access to affected Intel hardware.
  **Files:** smart-city tracker/pipeline, webcam segmenter/matte, device controls.
  (filed 2026-09-10, existing hardware reports and code review)

- [x] **R04 · P1 · Restore activity history and bound log storage.**
  Implements the original Activity Log report. Load a bounded tail after restart,
  rotate files, tolerate malformed trailing records, and retain useful error
  context without copying captured content into logs.
  **Done:** a failure remains visible after restart; corrupt last line is skipped;
  configured retention/size limits hold; concurrent entries remain parseable.
  **Estimate:** 1–2 days. **Depends on:** none.
  **Files:** `launcher/src/launcher/events.py`, log API and viewer.
  (filed 2026-09-10, original report and code review)

### Weeks 3–4 — reliable sessions and indexes

- [ ] **R05 · P1 · Bound event delivery and coordinate shutdown.**
  Introduce per-run IDs and bounded event retention, with an explicit overflow
  policy and replay/fan-out or enforced single-viewer ownership. Never silently
  split results between tabs. Shut down runners before closing their event loop;
  propagate pipeline-thread failures and report genuinely stuck native calls.
  **Done:** two tabs receive a consistent result or a clear ownership message;
  30 minutes without a consumer cannot grow the event buffer beyond its limit;
  reconnect cannot mix old/new runs; shutdown releases mock capture handles;
  producer failure and slow inference preserve truthful stopping/error states.
  **Estimate:** 4–5 days. **Depends on:** R04.
  **Files:** launcher lifespan, `ws_drain`, worker and streaming runners.
  (filed 2026-09-10, code review)

- [ ] **R06 · P1 · Invalidate stale document and embedding caches.**
  Add a source manifest and embedding identity including model/revision and
  chunking configuration; detect changed/deleted/added files on ingest. Record
  full embedding identity for screen memory too. Fail clearly or rebuild when
  old metadata cannot establish compatibility.
  **Done:** edit/delete/add fixtures update search results without a manual force
  option; unchanged inputs reuse cache; model changes cannot reuse incompatible
  vectors. Wire pinned revisions from R10 when available.
  **Estimate:** 2–3 days. **Depends on:** none; coordinates with R10.
  **Files:** document pipeline/store, smart-recall index metadata.
  (filed 2026-09-10, code review)

- [ ] **R07 · P1 · Make index persistence atomic and recoverable.**
  Write a complete generation before atomically switching its manifest; validate
  vector/chunk counts, schema and dimensions on load. Keep the previous valid
  generation and coordinate concurrent readers/writers.
  **Done:** injected failure between vector and metadata writes never exposes a
  mixed index; simultaneous reads see a complete generation; corrupt cache gives
  an actionable recovery state rather than taking down the panel.
  **Estimate:** 2–3 days. **Depends on:** R06 metadata design.
  **Files:** `bricks/doc-qa/src/doc_qa/store.py`, screen-memory persistence.
  (filed 2026-09-10, code review)

### Weeks 5–6 — controlled first use

- [ ] **R08 · P1 · Give screen memory retention and deletion controls.**
  Show what is recorded, storage location and usage before recording; provide
  pause, configurable age/size limits, and selective deletion in addition to
  reset. Keep screenshots and searchable text consistent during cleanup.
  **Done:** deterministic clock/quota checks enforce configured limits; deleted
  captures disappear from disk and search; pause stops new captures; failures
  during indexing do not leave screenshots indefinitely orphaned.
  **Estimate:** 3–4 days. **Depends on:** R05, R07.
  **Files:** smart-recall pipeline, runner, API and panel.
  (filed 2026-09-10, code review)

- [ ] **R09 · P1 · Bound local API inputs and access.**
  Keep local-only binding as the supported default; validate Host/Origin for
  privileged HTTP/WebSocket operations with a local session mechanism as needed.
  Add server-side numeric bounds, image/audio/video upload limits and cleanup;
  define a restrictive network policy for generated HTML previews. Network
  sharing requires a separate authenticated design, not just `--host 0.0.0.0`.
  **Done:** unexpected origins/hosts cannot trigger capture or file operations;
  valid UI/CLI flows still work; oversized uploads reject and clean up; invalid
  top-k/interval values fail before model work; preview external-resource tests
  match the documented offline policy. No exploit was demonstrated in this review.
  **Estimate:** 3–4 days. **Depends on:** R05 transport decisions.
  **Files:** launcher app/request schemas, upload routes, generated-HTML preview.
  (filed 2026-09-10, code review)

- [ ] **R10 · P1 · Make installation and model readiness predictable.**
  Add a preflight for Python/runtime, FFmpeg where needed, devices, disk space
  and selected-model availability. Create a model inventory with pinned revisions,
  approximate download sizes, prepare/download status, retry and explicit offline
  loading. Clarify local inference versus optional downloads/live video sources.
  Verify documented install/start commands preserve the intended dependency set.
  **Done:** clean Windows setup reaches one sample; missing FFmpeg has a scoped
  remedy; prepared sample starts with network disabled; absent offline model gives
  a clear error; partial download and insufficient disk have recoverable states.
  **Estimate:** 3–5 days. **Depends on:** R01; integrate identity with R06.
  **Files:** model cache, factories, setup scripts, README and readiness UI.
  (filed 2026-09-10, installation warning and code review)

### Weeks 7–8 — sustained workloads

- [ ] **R11 · P2 · Manage retained models and concurrent workload capacity.**
  Measure retained memory first; add explicit unload/release for idle sessions
  and explain capacity conflicts. Keep simultaneous workloads supported. Use
  process isolation only where measured native-call behavior justifies it.
  **Done:** 20 start/stop/unload cycles show no monotonic retained-memory growth
  after warmup; capture devices reopen; a 30-minute agreed concurrent workload
  passes without out-of-memory errors; capacity rejection gives recovery steps.
  **Estimate:** 3–4 days. **Depends on:** R05, R10.
  **Files:** model-owning runners, shared lifecycle/capacity helpers.
  (filed 2026-09-10, architecture review; memory behavior requires measurement)

- [ ] **R12 · P2 · Profile before optimizing video and screen history.**
  Covers the original smart-city throughput report. Instrument capture, infer,
  tracking, encode, delivery and telemetry overhead; benchmark cold/warm and
  concurrent cases. Measure screen-memory insertion/save cost as history grows.
  Choose smaller previews, batching or incremental storage only from results.
  **Done:** reproducible report for target hardware; initial target of ≥25 delivered
  fps on an agreed 30-fps single-feed source while meeting R03 accuracy; no stale
  frame buildup; 10,000-entry history remains usable against a recorded latency
  budget. Revise targets from baseline before implementation if infeasible.
  **Estimate:** 3–5 days. **Depends on:** R03, R07, R11.
  **Files:** smart-city pipeline/runner, MJPEG, telemetry, vector store.
  (filed 2026-09-10, original report and code review)

### Weeks 9–10 — usability and maintainability

- [ ] **R13 · P2 · Guide users to a successful first demo.**
  Offer a recommended ready configuration, place engine/model settings under
  advanced controls, make missing-device/download/error recovery actionable,
  and improve modal focus handling. Keep samples and visible active capture.
  **Done:** a new operator completes a prepared sample in ≤3 minutes without
  instructions; keyboard-only home → demo → log → return works; status/error
  announcements and a narrow-window layout pass manual checks.
  **Estimate:** 2–3 days. **Depends on:** R01, R08, R10.
  **Files:** frontend panels, header, log dialog and CSS.
  (filed 2026-09-10, UI and code review)

- [ ] **R14 · P2 · Extract per-demo routes and panel modules incrementally.**
  Keep FastAPI, vanilla JS, the registry and shared panel abstractions; migrate
  a representative demo first, then remaining groups in reviewable changes.
  Establish light formatting/linting for touched first-party code.
  **Done:** contracts and browser smoke checks stay green; adding a demo does not
  require editing a monolithic route/panel implementation; no bundled-framework
  migration or broad behavior changes are mixed into the extraction.
  **Estimate:** 2–3 days. **Depends on:** R05 and stability of earlier API changes.
  **Files:** launcher app/static JS and contributor guidance.
  (filed 2026-09-10, architecture review)

### Weeks 11–12 — release confidence

- [ ] **R15 · P1 · Expand platform and output-quality regression checks.**
  Retain Linux portable CI; add Windows portable/OpenVINO import and API checks,
  mocked browser flows, and opt-in target-hardware fixtures. Evaluate citation
  grounding, receipt fields, tracking counts and speech accuracy on small
  versioned datasets. Separate deterministic CI from expensive hardware runs.
  **Done:** install/import/test matrix passes; browser start/error/reconnect flows
  pass; target-machine report records all runtime/model/driver versions and
  accuracy metrics, including unsupported configurations and offline startup.
  **Estimate:** 3–5 days. **Depends on:** earlier fixes and R03/R10 fixtures.
  **Files:** tests, workflows and benchmark/quality fixture documentation.
  (filed 2026-09-10, existing CI and test review)

- [ ] **R16 · P1 · Publish only verified releases and run a small pilot.**
  Gate version publication on the tested commit, serialize bumps, verify
  installation instructions, and publish compatibility/known-limitations notes.
  Review third-party/model attribution and license inventory before wider
  distribution. Run object detection, document Q&A, translation and receipt
  extraction through setup → sample → error/recovery → stop/restart with an
  operator unfamiliar with the code.
  **Done:** failed checks cannot publish a release tag; a rehearsed release installs
  cleanly; all four pilot journeys pass; no open P1 affecting the supported pilot
  configurations; deferred defects have explicit limitations and follow-up owners.
  **Estimate:** 2–3 days. **Depends on:** R15 and applicable P1 gates.
  **Files:** version workflow, release documentation and pilot report.
  (filed 2026-09-10, workflow and product review)

### Deferred beyond the first cycle

- [ ] **New demos:** inbox triage and noise suppression after the four pilot
  journeys meet their gates and user demand is confirmed.
- [ ] **Remote multi-user access:** separate design for authentication, sessions,
  permissions and data boundaries before expanding local-only deployment.
- [ ] **Large architectural changes:** database migration, automatic model sharing,
  routing optimization or frontend-framework migration only with measured need.

Review priorities at the end of each phase. Carry forward unfinished gate items
before starting optional P2 work; do not mark a ticket complete merely because
its timebox expired.
