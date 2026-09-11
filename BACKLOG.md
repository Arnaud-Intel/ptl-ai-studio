# Backlog

This app exists to prove, live on one laptop, that Panther Lake runs useful
AI on the device itself. This file is the plan: what to show next, and what
has to be fixed before it can be shown. Findings from `logs/events.log`
reviews (see `CLAUDE.md`) and requests from the user land in the
[Inbox](#inbox) first.

Rebuilt on 2026-09-11 in a full pass: the roadmap is organized around the
hardware claims below and ordered Now / Next / Later instead of by calendar.
Every earlier ticket keeps its number and wording; finished ones are under
[Done](#done), and the original reports are preserved at the end.

## Inbox

Unsorted findings, newest last. Planning sorts each into an existing ticket,
a new one or the deferred list, and moves its original wording to
[Original reports](#original-reports).

<!-- - [ ] **Title.** Short description. (filed YYYY-MM-DD, source) -->

*Empty -- last sorted 2026-09-11.*

## What we are showing

The demo unit is a Dell XPS 14 (DA14260). These are the figures OpenVINO and
Windows report on it (2026-09-11), not datasheet numbers.

| Part | On this machine |
| --- | --- |
| CPU | Intel Core Ultra X7 358H: 16 cores (4 performance, 8 efficient, 4 low-power efficient) |
| iGPU | Arc B390, 12 Xe3 cores (96 EUs): 122.9 INT8 TOPS, 61.4 FP16 TFLOPS, about 36 GB addressable in shared memory |
| NPU | NPU 5 (architecture 5010, driver 32.0.100.5540): 50.4 INT8 TOPS, 25.2 FP16 TFLOPS |
| Memory | 64 GB LPDDR5X-9600, shared by all three |
| Power | readable per rail from Windows' RAPL counters: package, CPU cores, graphics, DRAM; the NPU has no rail of its own |
| Not on stage | the Arc Pro B60 on the desk is an external card, so every demo is planned without it (R20) |

Intel's launch message for the platform is up to 180 TOPS -- 50 from the NPU
and about 120 from the GPU -- running LLMs and agents locally
([Intel, CES 2026](https://newsroom.intel.com/client-computing/ces-2026-intel-core-ultra-series-3-debut-first-built-on-intel-18a)).
The demos should make five claims visible:

- **C1 · Every chip at once.** CPU, iGPU and NPU run different AI at the same
  time, and the gauges show it.
- **C2 · Efficient.** Sustained AI for a few watts on the NPU -- measured and
  shown, not asserted.
- **C3 · Big models, no discrete GPU.** The iGPU's share of system memory
  holds models that used to need a graphics card.
- **C4 · Real work, offline and private.** Tasks people do every day, with
  the network off and nothing leaving the machine.
- **C5 · Real time.** Speech and vision fast enough to feel live; first
  tokens fast enough to feel conversational.

### Where the demos stand

| Demo | Proves today | What holds it back |
| --- | --- | --- |
| Live speech translation | C4, C5 | Its efficiency on the NPU is invisible (R18) |
| Voice assistant | C4, C5 | -- |
| Meeting notes | C1 (two bricks at once), C4 | -- |
| Voice clone studio | C5 | The more faithful model, Chatterbox, is CPU-only |
| Webcam effects | C5 | iGPU gated off after NaN masks (R03) |
| Object detection | C5 | Detector licence (R17) |
| Screen OCR | C3 (7B vision-language model, 6 GB on the iGPU) | -- |
| Smart city | C1 (a chip per feed), C5 | Counts not yet trustworthy (R03); live feeds can vanish (R27) |
| Document Q&A | C4 | Stale answers after documents change (R06) |
| Expense extraction | C1 (two stages, two chips), C4 | -- |
| Code review, HTML creator | C3 (30B model on the iGPU, ~40 tokens/s), C4 | Each loads its own copy of the model (R11) |
| Screen memory | C1, C4 | Out of the pilot until retention lands (R08) |

Missing altogether: nothing measures power, so **C2 has no proof** (R18);
there is no generative visual demo, the thing most audiences read as "AI"
(R22); showing C1 takes a dozen settings rather than one click (R21); and
nothing shows the local agents Intel's own launch pitches (R26).

## How this backlog works

- **P1** -- a demo can show wrong output or fail on stage, a headline claim
  has no proof at all, or a release could ship something untested or
  unlicensable. **P2** -- makes the show stronger or the app sturdier once
  P1 holds. **P3** -- worth doing when there is time or a show asks for it.
- **Order, not dates.** Now / Next / Later replaces the twelve-week calendar,
  whose estimates were far off (R01-R04 were estimated at 8-10 days; their
  code landed in one). Per-ticket estimates stay, as a sense of size.
- **Numbers are stable.** R01-R16 come from [the project review](docs/PROJECT_REVIEW.md)
  (v0.2.36, `db0a7f7`); R17 onward from backlog reviews. A finished ticket
  moves to Done with its evidence.
- **Showcase tickets need a number,** not just a feature: each names the
  claim it proves, and its Done line says what gets measured.
- Tests for each fix belong in that fix. Don't mark a ticket complete because
  its timebox expired, and carry unfinished P1 gates forward before starting
  P2 work.

## Now -- P1, in this order

R17 comes first because later measurements depend on which models survive
it; R18 next because it is small and gives the efficiency claim its first
proof (R20, the other quick proof, is done); then the fixes that stop a demo showing something wrong, and
the release gate.

- [ ] **R17 · P1 · Settle model licences before measuring on the models.**
  Every YOLO11 size is AGPL-3.0 (model cards of `OpenVINO/YOLO11s-int8-ov`,
  the default detector since 2026-09-09, and `OpenVINO/YOLO11n-int8-ov`,
  checked 2026-09-11). It drives object detection, a pilot journey, and
  smart-city. If AGPL is unacceptable for partner demos the detector changes,
  and R03's count fixtures and R12's fps targets would have to be re-measured,
  so this runs first rather than inside R16. Inventory every model the app
  downloads: repo, revision, licence, gated or not.
  **Done:** the inventory is committed; each non-permissive or gated model has
  a recorded decision (keep with its conditions, replace, or drop) approved by
  the product owner; R03 and R12 measure on the chosen detector.
  **Estimate:** 1 day for the inventory; a replacement detector is estimated
  separately if one is needed. **Depends on:** none.
  **Files:** a model inventory (shared with R10's), detector factories and
  README attribution. (filed 2026-09-11, backlog review)
  Include the candidates the showcase tickets name, so choosing one is not a
  second review: FLUX.1-schnell, gpt-oss-20b, Qwen3-30B-A3B and Qwen3-VL
  (Apache-2.0 on Intel's pre-converted cards) and LCM Dreamshaper v7 (MIT);
  SD 1.5 and SDXL are OpenRAIL, with use restrictions.

- [ ] **R18 · P1 · Measure power and show energy per result.** *Proves C2.*
  Nothing in the app measures power, so the efficiency story -- the reason an
  NPU exists -- is an assertion. Windows exposes this machine's RAPL counters
  (`\Energy Meter(*)\Power`: package, CPU cores, graphics, DRAM). Sample them
  in the telemetry poller; show package watts in the header beside the
  utilisation gauges, and energy per result in each streaming brick (J per
  translated utterance, mJ per detected frame, J per answer), measured over
  the brick's active window against an idle baseline. The NPU has no rail of
  its own: its work shows only as package power that the core and graphics
  rails don't explain -- say that on screen rather than inventing a split.
  Show battery state and, on battery, the drain rate.
  **Evidence (2026-09-11):** YOLO11s on one 640×640 frame, run flat out,
  package power over an 11.7 W idle: CPU +35.1 W, iGPU +17.4 W, NPU +6.3 W.
  The counters see all three chips: the graphics rail moves only for the iGPU
  (0.03 to 8.6 W), and NPU work appears only in the package total. The core
  rail rises with every chip (+4.3 W even while the NPU runs the model)
  because the CPU prepares frames and decodes boxes, so energy per result is
  a whole-package figure, not a per-chip one.
  **Done:** watts update at least once a second with no measurable inference
  slowdown; every streaming brick shows energy per result; the method and its
  limits are written down; on a machine without the counters the gauge is
  hidden, never zero.
  **Estimate:** 2–3 days. **Depends on:** none.
  **Files:** `core/src/pantherlake_ai_core/telemetry.py`, telemetry poller,
  header, streaming panels. (filed 2026-09-11, backlog review)

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
  **Show prep (added 2026-09-11):** one command downloads every model a show's
  scenarios use, compiles the NPU cache, runs each sample once, then confirms
  each starts with the network off -- the difference between a warm demo and
  a first-run download on stage.

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
  **Depends on:** R01 for capability gating (done), R17 for which detector to
  measure on, and access to affected Intel hardware.
  **Files:** smart-city tracker/pipeline, webcam segmenter/matte, device controls.
  (filed 2026-09-10, existing hardware reports and code review)
  **Status 2026-09-10:** the webcam GPU/AUTO gate, finite/range mask checks,
  frame-wide tracker matching and the experimental-count disclosure are in;
  annotated count accuracy and CPU/GPU qualification remain open, and the
  overcount is not marked fixed. **Quick first step:** run the webcam model on
  the iGPU with `INFERENCE_PRECISION_HINT: f32` and compare its mask with the
  CPU's -- minutes, not the timebox.

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

- [ ] **R16 · P1 · Publish only verified releases and run a small pilot.**
  Gate version publication on the tested commit, serialize bumps, verify
  installation instructions, and publish compatibility/known-limitations notes.
  Review third-party attribution before wider distribution (model licences
  are settled up front, in R17). Run object detection, document Q&A, translation and receipt
  extraction through setup → sample → error/recovery → stop/restart with an
  operator unfamiliar with the code.
  **Done:** failed checks cannot publish a release tag; a rehearsed release installs
  cleanly; all four pilot journeys pass; no open P1 affecting the supported pilot
  configurations; deferred defects have explicit limitations and follow-up owners.
  **Estimate:** 2–3 days. **Depends on:** applicable P1 gates. R15 widens what a
  release is tested on but does not block tagging one.
  **Files:** version workflow, release documentation and pilot report.
  (filed 2026-09-10, workflow and product review)
  If R18 and R21 are done by then, the pilot adds a fifth, showcase journey:
  a stage scenario with the power readout.

## Next -- P2

### Showcase

- [ ] **R19 · P2 · The same task on every chip, side by side.** *Proves C1, C2, C5.*
  Every brick lets you pick CPU, iGPU or NPU, but comparing them takes a
  presenter a dozen dropdowns and a good memory. One panel runs a fixed,
  versioned workload on each chip -- YOLO11s on reference frames, Whisper on a
  reference clip, a batch of embeddings, one LLM prompt -- and shows latency,
  throughput, watts and energy per item in one table, recording runtime,
  driver and model revision. The table doubles as R12's baseline and R15's
  hardware report.
  **Evidence (2026-09-11, first data point):** YOLO11s on one frame, flat out: CPU 49.8 fps
  at 704 mJ/frame, iGPU 168 fps at 103 mJ/frame, NPU 48.6 fps at 130 mJ/frame
  (package energy above idle). The iGPU is fastest and, flat out, slightly
  cheaper per frame; the NPU draws by far the least power. Which chip is
  "most efficient" depends on whether work is batch or paced to a live
  source, so the table has to show both. Moving frame preparation and box
  decoding into the model graph (OpenVINO's PrePostProcessor) is the obvious
  lever for the NPU case.
  **Done:** one click produces the table on this machine in under two minutes,
  each workload run both flat out and paced to a live rate (30 fps video,
  real-time audio); the same run works from the CLI and exports CSV/JSON; a
  chip that is absent or unqualified (R03) says so instead of showing zero.
  **Estimate:** 3–4 days. **Depends on:** R18. (filed 2026-09-11, backlog review)

- [ ] **R21 · P2 · One-click stage scenarios.** *Proves C1, and C2 with R18.*
  The app's strongest moment -- every chip busy at once -- takes several panels
  and a dozen settings today. Add a few named scenarios that start a
  known-good combination on known-good chips and land on a view of all gauges
  and watts: *Hybrid meeting* (meeting notes and webcam effects on the NPU,
  document Q&A on the iGPU), *City operations* (smart city, one feed per
  chip), *Back office* (expense extraction, OCR and structuring on separate
  chips). Each says what the audience should notice and stops cleanly.
  **Done:** each scenario starts from the home page in one click and reaches
  steady state within a minute, warm and offline (R10's show prep); stopping
  leaves nothing running; a missing device or model is caught before starting.
  **Estimate:** 3–4 days. **Depends on:** R10, R20; R18 for watts.
  (filed 2026-09-11, backlog review)

- [ ] **R22 · P2 · Image generation on the integrated GPU.** *Proves C3, C5.*
  The app has no generative visuals -- what most audiences recognise as "AI",
  and the best showcase for the iGPU's 120 TOPS. OpenVINO GenAI's
  `Text2ImagePipeline` and `InpaintingPipeline` are already installed. Intel
  pre-converts FLUX.1-schnell (int4/int8/fp16, Apache-2.0), which needs only a
  few steps; LCM Dreamshaper v7 (MIT) is the fast fallback. Inpainting on a
  photo the audience provides shows the result isn't canned.
  **Done:** an image from a prompt at an agreed resolution and time on the
  iGPU, warm, with seconds and joules per image shown (R18); inpainting works
  on an uploaded image; generation can be cancelled; the model is in R17's
  inventory. **Estimate:** 3–4 days. **Depends on:** R17.
  (filed 2026-09-11, backlog review)

- [ ] **R23 · P2 · Live video narration with a vision-language model.** *Proves C3, C5; C1 with speech.*
  The colleague's commentary app (from the Inbox, original wording under
  Original reports), rebuilt on ungated models if its source doesn't arrive:
  a VLM on the iGPU describes a webcam or city-camera feed every few seconds
  and a synthetic voice reads it out -- two chips, one experience.
  `OpenVINO/Qwen3-VL-4B-Instruct-int4-ov` (Apache-2.0, cached here) measured
  1.4 s per caption on the iGPU and 17.3 s on CPU; its NPU compile hung. The
  8B variant is the quality step up if latency allows.
  **Done:** a caption at least every ~3 s on the iGPU, warm; a person reviewing
  the reference clips finds no invented objects; speech optional; works on the
  smart-city sources, TfL included. **Estimate:** 3–5 days. **Depends on:** the
  colleague's source or a decision to rebuild; R17. (filed 2026-09-11, user request)

- [ ] **R24 · P2 · Two chips, one answer: speculative decoding.** *Proves C1, C5.* Spike first.
  OpenVINO GenAI can pair a large model with a small draft model on another
  device (`openvino_genai.draft_model(path, device)` and `num_assistant_tokens`
  are in the installed 2026.3). A small draft on the NPU proposing tokens for
  a larger model on the iGPU would make heterogeneous compute visible in the
  most common workload, chat -- or show it doesn't pay here, also worth knowing.
  **Done (spike):** first-token latency and tokens/s with and without the NPU
  draft on a fixed prompt set, identical outputs; go or no-go recorded here.
  A doc Q&A or agent toggle is a follow-up ticket only on "go".
  **Estimate:** 2 days. **Depends on:** R20. (filed 2026-09-11, backlog review)

- [ ] **R25 · P2 · Listening all day for a few watts.** *Proves C2.*
  The NPU's purpose is sustained low-power inference; show it in watts and
  battery hours. The voice assistant's wake word and Whisper, plus live noise
  suppression (the planned brick; candidates include Intel's Open Model Zoo
  noise-suppression models), run on the NPU with package power compared
  against idle and against the same pipeline on CPU and iGPU, on battery.
  **Done:** a 10-minute run on battery reports the added package power and the
  projected battery impact for NPU against CPU; noise suppression keeps up
  with the microphone in real time on the NPU; the planned noise-suppression
  demo becomes available or is dropped with a reason.
  **Estimate:** 4–6 days. **Depends on:** R18, R17. (filed 2026-09-11, backlog review)

- [ ] **R27 · P2 · Health-check the curated live feeds.** From the Inbox
  (original wording under Original reports). Check each curated source in the
  background at start-up and every few minutes -- a metadata request for TfL
  clips, a yt-dlp resolve for YouTube -- and mark dead or bot-blocked ones
  unavailable in the picker, with the reason, before anyone picks them.
  **Done:** a feed that fails its check is marked in the picker; the check
  never blocks the UI and is rate-limited; samples prefer feeds that pass.
  **Estimate:** 1 day. **Depends on:** none. (filed 2026-09-11, live smart-city use)

### Sturdiness

- [ ] **R05 · P2 · Bound event delivery and coordinate shutdown.**
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

- [ ] **R07 · P2 · Make index persistence atomic and recoverable.**
  Write a complete generation before atomically switching its manifest; validate
  vector/chunk counts, schema and dimensions on load. Keep the previous valid
  generation and coordinate concurrent readers/writers.
  **Done:** injected failure between vector and metadata writes never exposes a
  mixed index; simultaneous reads see a complete generation; corrupt cache gives
  an actionable recovery state rather than taking down the panel.
  **Estimate:** 2–3 days. **Depends on:** R06 metadata design.
  **Files:** `bricks/doc-qa/src/doc_qa/store.py`, screen-memory persistence.
  (filed 2026-09-10, code review)

- [ ] **R09 · P2 · Bound local API inputs and access.**
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
  **Re-ranked P2 on 2026-09-11** (local-only and supervised today). Do the
  cheap half first: the Host/Origin check and upload size limits --
  `POST /api/smart-city-monitor/upload` copies whatever it is sent, uncapped.

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
  **Seen 2026-09-11:** code review and HTML creator each keep their own copy
  of the same 30B model loaded, 17 GB apiece, so using both and then screen
  OCR asks for about 40 GB against the ~36 GB the iGPU can address. It
  worked in the R20 run, but one loaded model shared between bricks is the
  first thing to fix here.

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
  Shares its workloads and method with R19's side-by-side table.

- [ ] **R13 · P2 · Guide users to a successful first demo.**
  Offer a recommended ready configuration, place engine/model settings under
  advanced controls, make missing-device/download/error recovery actionable,
  and improve modal focus handling. Keep samples and visible active capture.
  **Done:** a new operator completes a prepared sample in ≤3 minutes without
  instructions; keyboard-only home → demo → log → return works; status/error
  announcements and a narrow-window layout pass manual checks.
  **Estimate:** 2–3 days. **Depends on:** R01, R10 (and R08 before screen
  memory is included).
  **Files:** frontend panels, header, log dialog and CSS.
  (filed 2026-09-10, UI and code review)
  Shares its UI work with R21's stage scenarios.

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

- [ ] **R15 · P2 · Expand platform and output-quality regression checks.**
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
  Add voice-cloning speaker similarity (the ECAPA score that justified
  Chatterbox) to the quality set.

## Later -- P3

- [ ] **R26 · P3 · A local agent that chains the demos.** *Proves C3, C4.*
  Intel pitches the platform as running AI agents locally; today each brick is
  a separate demo. An agent on a 20-30B model on the iGPU (gpt-oss-20b or
  Qwen3-30B-A3B-Instruct-2507, both pre-converted by Intel under Apache-2.0)
  with tool calls into existing bricks -- search documents, extract receipts,
  summarise a meeting, draft an email -- turns them into workflows such as
  "collect last week's receipts into an expense report and draft the note to
  finance". The planned inbox triage becomes one of its tasks.
  **Done:** three scripted tasks complete end to end offline; every tool call is
  shown, nothing is written or sent without confirmation, and failures are
  reported rather than improvised. Promote to P2 if productivity is a show's theme.
  **Estimate:** 1–2 weeks. **Depends on:** R20, R06, R17. (filed 2026-09-11, backlog review)

- [ ] **R28 · P3 · Extend dropped-request recovery to the other Whisper
  demos, if the fault recurs.** From the Inbox (original wording under
  Original reports). Meeting notes and the voice assistant get the reload and
  retry live translation has had since 0.2.40 -- only once the activity log
  shows the fault again ("reloading the model and retrying" entries).
  **Estimate:** half a day. **Depends on:** evidence. (filed 2026-09-11, activity log)

- [ ] **R29 · P3 · Search photos by description, indexed on the NPU.** *Proves C2, C4.*
  Image-text embeddings for a local photo folder, built on the NPU in the
  background and searched in plain language ("the whiteboard from Tuesday's
  workshop"), reusing document Q&A's index. Needs a permissively licensed
  CLIP/SigLIP-class model (R17). **Done:** a 1,000-photo folder indexes in the
  background with its energy shown (R18); relevant photos rank first on an
  agreed query set. **Estimate:** 3–4 days. **Depends on:** R17, R18.
  (filed 2026-09-11, backlog review)

- [ ] **R30 · P3 · An assistant in every app.** *Proves C4, C5.*
  A small tray helper with a global hotkey sends the selected text or the
  clipboard to the local LLM -- rewrite, summarise, translate -- and pastes the
  result back: the pattern people know from cloud assistants, offline. It
  lives outside the browser, so it is a new native component; opt-in only.
  **Done:** works in any text field on Windows; first words appear within a
  second on the iGPU; nothing is sent anywhere. **Estimate:** 3–5 days.
  **Depends on:** R20. (filed 2026-09-11, backlog review)

- [ ] **R31 · P3 · Spike: text-to-video, and one speech-to-speech model.**
  `Text2VideoPipeline` and `OmniPipeline` are in the installed OpenVINO GenAI.
  Measure whether a few-second clip renders in demo time on the iGPU, and
  whether a single omni model could replace the voice assistant's
  speech-LLM-speech chain with lower latency. Go or no-go only.
  **Estimate:** 2 days. **Depends on:** R17. (filed 2026-09-11, backlog review)

## Deferred

- [ ] **R08 · Deferred · Give screen memory retention and deletion controls.**
  Show what is recorded, storage location and usage before recording; provide
  pause, configurable age/size limits, and selective deletion in addition to
  reset. Keep screenshots and searchable text consistent during cleanup.
  **Done:** deterministic clock/quota checks enforce configured limits; deleted
  captures disappear from disk and search; pause stops new captures; failures
  during indexing do not leave screenshots indefinitely orphaned.
  **Estimate:** 3–4 days. **Depends on:** R05, R07.
  **Files:** smart-recall pipeline, runner, API and panel.
  (filed 2026-09-10, code review)
  **Deferred 2026-09-11:** the project review keeps screen memory out of the
  pilot until this lands, so it gates screen memory joining the pilot, not
  the pilot itself. Re-rank it when screen memory is wanted on stage.

- [ ] **New demos beyond the tickets above:** after the pilot journeys meet
  their gates. Inbox triage now folds into R26, noise suppression into R25.
- [ ] **Remote multi-user access:** separate design for authentication, sessions,
  permissions and data boundaries before expanding local-only deployment.
- [ ] **Large architectural changes:** database migration, automatic model sharing,
  routing optimization or frontend-framework migration only with measured need.

## Done

R20 landed on 2026-09-11. R01, R02 and R04 were implemented on 2026-09-10 (see the
[validation notes](docs/TRUSTWORTHY_OUTPUT.md)); R01's hardware gate was
verified on 2026-09-11. For R02, manual correction remains through the CSV;
no in-app approval flow is claimed. R04 restores history without resurrecting
previously active workers.

- [x] **R01 · P1 · Correct hardware identity and capability labels.**
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
  **Verified 2026-09-11** on the target machine (Core Ultra X7 358H, NPU driver
  32.0.100.5540, OpenVINO 2026.3): OpenVINO reports the NPU as
  `Intel(R) AI Boost`, and `/api/telemetry` returns the same `npu_name` with a
  numeric `npu_percent`; no USB device is named anywhere. The `npu_name: null`
  in the validation notes came from a machine where OpenVINO sees no NPU,
  which is the correct "unavailable" answer. The USB-input negative fixtures
  (`test_telemetry.py`) and the unavailable-device and unknown-engine
  rejections (`test_launcher_api.py`) pass, and the header's platform badges
  now sit under "Designed for".

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

- [x] **R04 · P1 · Restore activity history and bound log storage.**
  Implements the original Activity Log report. Load a bounded tail after restart,
  rotate files, tolerate malformed trailing records, and retain useful error
  context without copying captured content into logs.
  **Done:** a failure remains visible after restart; corrupt last line is skipped;
  configured retention/size limits hold; concurrent entries remain parseable.
  **Estimate:** 1–2 days. **Depends on:** none.
  **Files:** `launcher/src/launcher/events.py`, log API and viewer.
  (filed 2026-09-10, original report and code review)

- [x] **R20 · P1 · Show the big models on the laptop's own GPU.** *Proves C3.*
  Code review and HTML creator carry a "Discrete GPU" badge (`requires_dgpu`),
  and they and screen OCR's 7B vision model default to the discrete GPU when
  one is plugged in, else to `AUTO`. The B60 on this desk is external: on
  stage the XPS 14 is alone, and the badge tells the audience the opposite of
  the story. It is also not true -- the iGPU addresses about 36 GB.
  **Evidence (2026-09-11):** Qwen3-Coder-30B-A3B int4, one 256-token coding
  prompt: the iGPU loads it in 21.6 s, gives the first token in 0.39 s and
  streams about 38 tokens/s; the B60 takes 31.7 s, 0.40 s, about 65 tokens/s.
  The discrete card is faster, not required. Only ~3B of the 30B parameters
  are active per token, which is why it streams this fast from shared memory
  -- worth saying on stage.
  **Done:** with no discrete GPU the large-model default is the iGPU, chosen
  explicitly rather than left to `AUTO`; the badge becomes a measured memory
  requirement, with "faster on a discrete GPU" only where measured; code
  review, HTML creator and screen OCR each complete their sample on the XPS 14
  alone; the panel shows tokens/s so the audience sees the number.
  **Estimate:** 1 day. **Depends on:** none; re-measure if R17 swaps a model.
  **Files:** `core/src/pantherlake_ai_core/engine.py`
  (`preferred_large_model_device`), registry and badge, code-review-assist,
  html-creator, screen-ocr. (filed 2026-09-11, backlog review)
  **Done 2026-09-11:** large models default to the discrete GPU when there is
  one, else to the integrated GPU by name, in core and in the UI, with tests;
  the badge states the model and its measured memory ("30B model · 17 GB",
  "7B vision model · 6 GB"), with the discrete-GPU comparison in its tooltip;
  code review, HTML creator and screen OCR show tokens/s, token count and
  first-token time under each answer, from OpenVINO's own metrics. Verified
  through the launcher pinned to the iGPU (the B60 attached but unused):
  code review 445 tokens at 44.4 tokens/s, first token 0.77 s; HTML creator
  a complete 8.8 KB page, 2,175 tokens at 40.5 tokens/s; screen OCR read a
  receipt correctly at 23.2 tokens/s. Memory once loaded: 16.9 GB (30B) and
  5.9 GB (7B) of shared memory.

## Original reports

Preserved as filed. A ticked box means fixed; a "covered by" or "sorted into"
line names the ticket the work now lives in.

- [x] **The Activity Log forgets everything the launcher didn't see itself.**
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
  (filed 2026-09-08, from the activity-log review; fixed by R04, 2026-09-10)

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
  Covered by **R03**.

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
  Covered by **R03**.

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
  Covered by **R12**.

- [ ] **Curated live feeds can all go dark at once, and nothing says so until
  someone picks one.** On 2026-09-11 YouTube answered every curated camera
  with "Sign in to confirm you're not a bot", even with the newest yt-dlp; the
  first Amsterdam stream had already died within days of being added.
  Mitigated in 0.2.38 and 0.2.39, not fixed: the picker separates YouTube from
  other sources (TfL JamCams, which were unaffected), the error says what
  happened, and `PTL_YOUTUBE_COOKIES` opts in to an exported signed-in
  session. Still missing: a health check that marks a dead feed unavailable
  before it is picked on stage. Probably belongs with R10 or R13.
  (filed 2026-09-11, live smart-city use)
  Sorted into **R27** on 2026-09-11.

- [ ] **Whisper on the NPU can drop a request, and only live translation
  recovers.** 2026-09-11 10:34: live translation on the NPU failed with Level
  Zero `ZE_RESULT_ERROR_INVALID_ARGUMENT` 42 s into a session, 13 s after
  smart-city released its NPU detector. Over a hundred attempts to reproduce
  it passed (Whisper alone, next to YOLO on the NPU, after YOLO's release,
  0.4-30 s segments, noise and silence), so it reads as a rare driver fault
  (NPU driver 32.0.100.5540). Live translation now reloads the model and
  retries the utterance (0.2.40); meeting notes and the voice assistant can
  run the same Whisper on the NPU without that recovery. A recurrence shows in
  the activity log as "reloading the model and retrying" -- count those before
  extending it. (filed 2026-09-11, activity log)
  Sorted into **R28** on 2026-09-11.

- [ ] **Video commentary demo, from a colleague's app.** Blocked on its source
  (`app4.py`, `index4.html`); only the install guide is here. The original
  runs gated Gemma 3 4B; the ungated `OpenVINO/Qwen3-VL-4B-Instruct-int4-ov`
  (Apache-2.0) measured 1.4 s per caption on the iGPU and 17.3 s on CPU, and
  its NPU compile hung. If it is wanted for a show it needs a ticket, not the
  deferred-demos list. (filed 2026-09-11, user request)
  Sorted into **R23** on 2026-09-11.
