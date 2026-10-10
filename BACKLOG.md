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

Refreshed on 2026-10-10, at v0.2.93: a month and some fifty versions later,
with the Auto Demo, the Page Agent, the Video Commentator and a setup
assistant built in between. The Inbox was sorted (its entries are under
[Original reports](#original-reports), each with the ticket it went into), R21
was closed, tickets the work had touched got a dated note, and R33 to R51 were
added: what the app needs next, six new bricks among them. Tickets keep their
numbers and their wording.

## Inbox

Unsorted findings, newest last. Planning sorts each into an existing ticket,
a new one or the deferred list, and moves its original wording to
[Original reports](#original-reports).

<!-- - [ ] **Title.** Short description. (filed YYYY-MM-DD, source) -->

Sorted on 2026-10-10: nothing is waiting. The twenty-six entries that were
here are under Original reports, each with the ticket it went into.

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

Fifteen demos and the Auto Demo, on 2026-10-10.

| Demo | Proves today | What holds it back |
| --- | --- | --- |
| Auto Demo | C1 in one click, C3 | Never left alone for hours, never a whole turn of six scenes (R34) |
| Live speech translation | C2 (joules per line), C4, C5 | In a call the other participants are transcribed as the presenter (R40); the chips are not side by side (R19) |
| Voice assistant | C4, C5 | Not run with a microphone since its voice moved to the CPU (R35); speaks only once the whole answer is written (R38); its wake word is a non-commercial model (R17) |
| Meeting notes | C1 (two bricks at once), C4 | On the NPU it finds 11 stated tasks of 20; 17-18 on the integrated GPU (R36) |
| Voice clone studio | C5 | The voice models do not compile for the NPU; the more faithful voice takes 8 to 11 s a line (R38) |
| Webcam effects | C5 | Integrated GPU gated off after NaN masks (R03) |
| Object detection | C5 | Detector licence (R17) |
| Screen OCR | C2 (joules per read), C3 (7B vision-language model, 6 GB on the iGPU) | 7 to 9 s an image; a 4B model read as much in 5 s, on the next OpenVINO (R37) |
| Smart city | C1 (a chip per feed), C5 | Counts run high, and small people in a crowd are mostly missed (R03) |
| Video commentator (experimental) | C1, C3, C5 | The mood model embroiders on what was seen; its lines are too long to be said in time (R23) |
| Document Q&A | C4 | The 1.5B model falls for a trick question and garbles a long answer (R36) |
| Expense extraction | C1 (two stages, two chips), C4 | Refunds misfiled, day-first dates read month-first, an invented total still shown as a number (R36) |
| Code review, HTML creator | C2 (joules per answer), C3 (30B model on the iGPU, ~40 tokens/s), C4 | Each loads its own copy of the model (R11); HTML Creator can run a page into a loop (R47) |
| Page agent (experimental) | C1 (three models, two or three chips), C3 | Page length unbounded, one skeleton, pictures never looked at (R39) |
| Screen memory | C1, C4 | Out of the pilot until retention lands (R08); its search reads the embedding model the old way (R06) |

What the September list called missing is there now: the Auto Demo shows C1 in
one click (R21, done), the Page Agent draws its pictures on the integrated GPU
(half of R22), and it is three models working on one job, though plain code
conducts them, not a model (R26).

What is thin on 2026-10-10 is of another kind. Nothing has been installed or
run on a laptop other than this one, and the README says Lunar Lake and
Panther Lake (R33). Changes to the speech demos are checked without a
microphone or a loudspeaker, so the latest have not been heard (R35). The Auto
Demo has not been left alone for more than a few turns (R34). The small model on the NPU is what the
quality of most demos rests on, and it is their weakest part (R36). Energy per
result is on four demos and no view puts the three chips side by side (R18,
R19). And licences are not settled, with one model in use already found to be
non-commercial (R17).

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
- **New bricks are tickets too,** marked *New brick*: each names the claim it
  proves, what it reuses from the bricks that exist, and what to measure
  before anything is built. An idea that is not there yet is a line under
  Deferred.

## Now -- P1, in this order

R17 still comes first: it decides which models everything else is measured on,
and one model in use has turned out to be non-commercial. Then three things
that would embarrass a show or a new user, and that no amount of code written
at this desk can settle: an install on another laptop (R33), the Auto Demo
left alone for a day (R34), the speech demos heard (R35). R18 and R10 are
mostly done and keep what is left of them; then the fixes that stop a demo
showing something wrong, and the release gate.

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
  **Status 2026-10-10:** not started, and the list has grown. In use today,
  and to go in the inventory: Qwen2.5-1.5B, Qwen3-8B (two builds),
  Qwen3-Coder-30B, Qwen2.5-VL-7B, Qwen3-Embedding, Whisper (base and medium),
  FLUX.1-schnell, YOLO11s, the selfie-segmentation model, OpenVoice and
  Chatterbox, Silero's voice detection, the wake word, the portable engines'
  models, and the six sample videos (already credited in
  `sample-data/videos/README.md`). **One finding already:** the wake word
  ("hey jarvis") is one of openWakeWord's own models, and that package's
  README puts all of them under CC BY-NC-SA 4.0 -- non-commercial -- while its
  code is Apache-2.0 (read in the installed package, 2026-10-10). The Voice
  Assistant at a company's stand needs a decision on it: a wake word trained
  for the project, a button instead of a wake word, or the licence cleared.
  The new bricks name their candidates' licences as questions for this ticket
  (R42, R44, R49).

- [ ] **R33 · P1 · Prove the install on laptops that are not this one.**
  *Every claim rests on it.*
  The README says Lunar Lake and Panther Lake, and that a laptop with nothing
  installed needs only `first_launch.bat`. Everything measured so far -- every
  figure, every screenshot, the setup assistant's walk -- comes from one Dell
  XPS 14 with 64 GB, and the bare laptop was an imitation (an empty profile,
  Windows alone on the PATH). Not seen anywhere: Windows' warning on a
  downloaded `.bat`, a laptop without the Visual C++ runtime, a company
  laptop's rules, and a Lunar Lake machine at all -- 16 or 32 GB where this
  one has 64, another integrated GPU, the NPU of the generation before. Memory
  is the part most likely to bite: the 30B coder takes 17 GB of what the
  integrated GPU can address (about 36 GB here), and the image model 13 GB
  beside it.
  Two installs from the ZIP, by somebody who did not write the assistant: a
  new Panther Lake laptop and a Lunar Lake one. Write down what the six steps
  did, which demos start, and one figure per demo.
  **Done:** both installs reach a working sample with no help beyond the
  README, or each place they stopped is fixed and tried again; the README's
  "Which laptops" has a table of what runs at 16, 32 and 64 GB with measured
  speeds, in place of "at its own speeds"; a demo whose model cannot fit says
  so before loading, on that machine, and offers the one that does; the items
  the assistant's report lists as not checked are each seen or struck.
  **Estimate:** 1 day per laptop with the machine in hand, then 1-2 days for
  defaults that follow the memory if the smaller machine needs them. **Depends
  on:** the laptops. **Files:** `setup/`, README, `core`'s engine defaults,
  the registry's memory badges. (filed 2026-10-10, backlog refresh)

- [ ] **R34 · P1 · The Auto Demo, left alone for a day.** *Proves C1 where it
  will be shown.*
  The loop is what a stand runs with nobody at the keyboard, and it has been
  round a few times: never for hours, never with all six scenes in one turn.
  What is known to be waiting for it: an expense report is added to the
  database at every turn and none is removed; the large models are not
  unloaded between scenes on a machine short of memory; after two hours of
  builds this laptop held itself to 15 W and everything took half as long
  again (R45); the camera scene has met one person sitting at the laptop, not
  a visitor holding things up, nor a crowd; the stage for a stand without the
  discrete GPU is built and tested, not watched; the commentator's lines are
  in English whatever language was chosen.
  **Done:** eight hours on mains with every scene ticked: no failure the loop
  does not recover from by itself, nothing left loaded at the end, memory and
  disk flat from the first hour to the last, and each scene's figures at hour
  one and hour eight side by side; when it stops the loop writes a short
  report (turns, scenes played, failures, the fastest and slowest of each
  scene); it removes its own expense reports; half an hour of the camera scene
  in front of several people, with somebody watching; `docs/AUTO_DEMO.md` ends
  on a checklist for a stand, made from what the run showed.
  **Estimate:** 2-3 days, most of it the machine running. **Depends on:**
  none; R45 explains what it finds. (filed 2026-10-10, backlog refresh;
  carries what is left of R21)

- [ ] **R35 · P1 · A rehearsal with real sound.** *A release should not ship
  unheard.*
  When a speech demo is changed it is checked with synthesised or recorded
  speech fed in place of the microphone, and what it says is read back by
  Whisper instead of listened to. That is a choice: nothing is recorded or
  played on this laptop without its owner's word. Live Translation has been
  shown in real calls by its owner, which is how R40's faults were found; but
  several things have not been spoken to or heard since they were built or
  last changed: the Voice Assistant, wake word included, since its voice moved
  to the CPU (v0.2.91); the speech speed in the hardware panel (2026-10-03);
  Meeting Notes written from the page during a real meeting; the commentator's
  line said aloud, in the studio's voice and in a cloned one; a voice enrolled
  and used in the Voice Clone Studio since the same change.
  A checklist of some ten steps for the owner of the laptop, half an hour,
  each with what should be seen and heard and a place to write what was.
  **Done:** every step done once on the XPS 14 with its own microphone and
  speakers, and the speech ones once more in a Teams call, which is how the
  app is shown (R40); each result written down; whatever was wrong is in the
  Inbox.
  **Estimate:** half a day to write the list and what it needs, half an hour
  to run it. **Depends on:** the user. (filed 2026-10-10, backlog refresh)

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
  **Status 2026-09-14, in progress:** in -- a Power gauge in the dock (package
  watts once a second, rails and battery in its tooltip, hidden without
  counters), and energy per result above an idle baseline learned while no
  demo runs, on live translation's lines and on code review, HTML creator and
  screen OCR answers. A counter read costs 0.06 ms in-process. Through the
  launcher, screen OCR on the iGPU spent 386 J over 9 s, 223 J above idle.
  The method and its limits are in the README. Still open: energy per result
  for meeting notes, the voice assistant, document Q&A, expense extraction,
  screen memory, voice cloning and the video demos (which first need a frame
  count), and a drain rate on battery. Seen: a freshly started launcher learns
  its baseline while the machine is still settling (18.6 W, against ~12 W
  quiet), so its first results understate their above-idle cost.
  **Status 2026-10-10:** since 0.2.53 the hardware panel shows every running
  demo under the chip it is on with its own figure (frames or tokens per
  second), live while a language model writes (R32). Energy per result is
  still on the same four demos. Added from the Inbox: say on screen whether
  the figures are on mains or on battery -- on 2026-10-03 the integrated GPU's
  tokens per second fell by about a third on battery (86-88 to 50-58 on the
  1.5B model) and nothing said so. R45 has the rest of that. The language
  calls inside the loop bricks (expense lines, screen memory, the voice
  assistant's reply) still have no live figure.

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
  **Measured 2026-09-21:** with the environment installed and the models
  cached, the suite does run with no network -- screen OCR read an image on
  the iGPU offline, and the launcher starts in 4 s. Two breaks were found and
  fixed: `uv run` re-syncs on every start and fails offline (start_launcher.bat
  now tries `uv run --offline` first), and llama.cpp resolved its GGUF filename
  pattern over the network even for a cached model (core.model_cache.resolve_gguf).
  Still needs a connection: the live cameras, the update check (which says so
  and carries on), and the first use of any model that isn't cached yet -- the
  prefetch above is what makes that last one a before-the-show step.
  **Part done 2026-09-21:** the prefetch exists -- `panther-lake-prefetch` and
  the footer's "Prepare models" button list every model with its size, what
  uses it and whether it is cached, and download the missing ones with live
  speed and time left (core.models, core.prefetch). Still open here: the rest
  of the preflight (Python/FFmpeg/devices/disk space), pinned revisions, and
  the "starts with the network off" rehearsal as one command.
  **Status 2026-10-10:** the preflight is built, as the setup assistant
  (v0.2.92): processor, memory, disk, drivers, the Visual C++ runtime, the
  folder and the hosts are checked before anything is downloaded; the new
  environment is asked which packages load and which chips it lists; models
  are offered in sets with their sizes; a step that stops or fails can be run
  again and keeps what it had. Too little disk, a host that does not answer
  and a refused certificate were shown with scripted failures. FFmpeg turns
  out not to be needed: on a copy with none, the Studio starts and warns once
  (the voice brick reads WAV without it). Left: pinned model revisions (with
  R06 and R17); the rehearsal with the network off as one command (now part of
  R45); a mirror for the two sample videos that are Wikimedia's own encodes,
  which a re-encode there would make fail their checksum; a model download cut
  half-way, which was not tried. The clean Windows setup this ticket's Done
  line asks for was imitated, not done: that is R33.

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
  **Status 2026-10-10:** the counts still run high, and the app says so on
  screen. On the sample clips now fetched with the app: the cattle clip is a
  few dozen animals and was counted as 90, a track lost and found being
  counted twice; the Shibuya clip gives a median of 3 boxes among several
  hundred people. Six openly licensed clips are on disk (streets, a bottling
  line, a herd): they can be the annotated corpus this ticket asks for, and
  nobody has counted them by hand yet. Two things to try that came out of
  them: a larger input, or tiles, for small people; and counting at a line
  that is crossed rather than at a track that is created. The webcam's GPU
  path is still gated.

- [ ] **R06 · P1 · Invalidate stale document and embedding caches.**
  *2026-10-09:* reported from the field -- "it said it had looked at three
  files when only two were given, and made things up". Replayed with the
  real models: a folder indexed with three files, one taken out, indexed
  again without "Rebuild the index", went on answering from the file that
  was gone. Done for Document Q&A: an index is kept with a fingerprint of
  what it was built from (every supported file, its size and when it was
  written) and with how the embedder reads a text, and is built again when
  either differs; an indexing that fails no longer leaves the last
  folder's documents answering. The panel names the files indexed and the
  files each answer was written from. Left of this ticket: screen memory's
  index, and pinned model revisions (R10).
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
  *2026-10-10:* screen memory's side is the larger half now. Its index still
  reads the embedding model at its first token -- the fault that, in Document
  Q&A, put the right file first for 2 questions of 10 instead of 8 -- and
  moving it means embedding again everything it has kept.

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
  **Status 2026-10-10:** the gate is not there. The version is bumped by a
  workflow of its own on every push, whatever the tests say: on 2026-10-09 a
  push whose tests failed on CI was tagged v0.2.87, and mended in v0.2.88.
  Making the bump wait for the test job is the first half-day of this ticket,
  and R46 needs it (a tag that means "tested"). The install instructions were
  rewritten and walked on an imitated bare laptop (R10, R33). The four pilot
  journeys named above still fit; the Auto Demo is the fifth.

## Next -- P2

### Showcase

In a suggested order, least work for most effect first: the answers people
read (R36) and the voices (R38), which make what is already shown better; then
the two bricks that need no new model (R22, an Image Studio, and R41, Look and
Ask); the side-by-side table (R19); the two experimental bricks out of their
label (R23, R39); the call (R40); the OpenVINO upgrade (R37); then the bricks
that start with a question to settle (R42, R43, R44), and the tickets that
were here before.

- [ ] **R36 · P2 · A better small model on the NPU.** *Proves C2, C4 -- with
  answers worth showing.*
  Most of what the NPU says in this app comes from Qwen2.5-1.5B, and it is the
  weakest part of the demos it serves. Measured in October: of 7 action items
  in a meeting it found none; it fills in 47 expense fields of 70; it files a
  refund under "Other" and a credit note under "Software", takes a banner line
  for the vendor, and when a total is too faded to read it invents one
  (flagged, kept out of the sums, and still shown as a number); asked for "the
  chief executive's home address" it gives his name; in the commentator it
  turns a rider into "brave cowboys". Qwen3-8B, in its channel-wise build for
  the NPU, is already on disk and already writes Meeting Notes: 62 expense
  fields of 70 and 4 action items of 7, at 19 tokens/s against 58, 1.2 s to
  its first token, 95 s to compile the first time.
  In order. (1) Let Document Q&A, the Expense Extractor, the Voice Assistant
  and the commentator's voice choose the 8B on the NPU, and measure each on
  its own samples, quality and seconds both. (2) The two receipt faults no
  model fixes: a day-first date read month-first (`12/09/2026`, by every model
  tried) and an amount shown although it could not be matched to the receipt.
  Say the date order in the prompt and flag a date that reads both ways; show
  no amount when none can be matched. (3) The one thing that would make the
  NPU's notes good: export Qwen3-8B channel-wise with calibration data and
  measure it on the four test meetings (11 stated tasks of 20 today; 17-18 on
  the integrated GPU). It needs the 16 GB original weights, an hour or more of
  CPU, and an account to host the result.
  **Done:** a table per brick, 1.5B against 8B on the NPU, on the brick's own
  samples; each brick's default chosen from it and said in its README; the two
  French receipts dated 12/09/2026 come back as 12 September or flagged; an
  unreadable total shows no number; the calibrated export is measured and
  adopted, or dropped with its figures.
  **Estimate:** 3-4 days, plus 1-2 for the export. **Depends on:** R17 for the
  model's place in the inventory. (filed 2026-10-10, from six Inbox entries)

- [ ] **R38 · P2 · Voices that keep up with the talk.** *Proves C5.*
  The Voice Assistant writes its whole answer, then speaks it. The commentator
  waits for a line to have been said before it looks again. A cloned voice
  takes 8 to 11 s a line by itself and 18 to 30 s beside a video, and its
  first line pays twenty seconds of warming up. The studio's own voice is not
  the limit: on the CPU it makes a sentence in 0.2 to 0.5 s.
  (1) Speak sentence by sentence while the answer is being written -- step C
  of the streaming work (R32), whose control object is there. (2) Warm a
  cloned voice when it is chosen, not at its first line, and look at how many
  threads it takes while a video is decoded on the same cores. (3) Keep the
  commentator's lines to what can be said in five or six seconds (one ran to
  28 words and 11 s).
  **Done:** the time from the end of a question to the first sound of the
  answer, measured before and after on ten questions; the same for a cloned
  voice's first line; with the studio's voice the commentator says a line
  every six seconds or better; none of it moves a voice model to the NPU,
  where it does not compile (v0.2.91).
  **Estimate:** 2-3 days. **Depends on:** R35, to hear it. (filed 2026-10-10,
  from three Inbox entries)

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
  **Status 2026-10-10, half proven:** FLUX.1-schnell has been drawing on the
  integrated GPU since 2026-10-08, inside the Page Agent: 4.5 to 8.5 s a
  picture, about 13 GB loaded, six pictures a page. What this ticket still
  asks is a brick of its own -- *New brick:* an **Image Studio**: a prompt and
  a picture with its seconds and joules, a picture changed by a sentence, a
  region painted again on a photo the audience brings, a stop button. The
  model is on the disk of anyone who fetched the Auto Demo's set, which makes
  this the cheapest new brick on the list. To measure first: painting a region
  again and changing a picture with this model in the installed OpenVINO
  GenAI, neither of which the Page Agent uses.

- [ ] **R41 · P2 · New brick: Look and Ask.** *Proves C1, C5.*
  Hold something up to the camera and ask about it aloud; the laptop answers
  aloud. Three chips in one exchange, and the kind of demo people walk up to
  -- the user's remark on the camera scene was that it is "potentially
  engaging for people to be interacting with". Nothing new to download:
  Whisper hears the question on the NPU, the 7B vision model answers from the
  frame on the integrated GPU (today it describes a frame in 0.8 to 1.0 s,
  first word after 0.2 s), the studio's voice says it from the CPU (0.2 to 0.5
  s a sentence). It composes the Voice Assistant and the commentator's eye, as
  Meeting Notes composes two bricks. What it must not do is already written
  down: what is said of people is kept to what they do, by a rule in code
  (`about_what_they_do`), and nothing is kept.
  **Done:** the answer starts to be spoken within three seconds of the end of
  the question on the XPS 14, measured on ten questions about ten objects;
  each of the three chips has its row in the hardware panel; a typed question
  works where a stand is too loud to be heard; a question about a person's
  looks, age or identity is declined in one sentence; the frame that was
  looked at is shown beside the answer and then dropped.
  **Estimate:** 3-4 days. **Depends on:** R35 and R38, for the voice. (filed
  2026-10-10, backlog refresh)

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
  *2026-10-10:* two things to fold in. A scene for the Auto Demo comes nearly
  free once the table exists ("one job, three chips"). And the chips are less
  independent than "a chip each" says: on 2026-10-08, with pictures drawn on
  one GPU and a page written on the other, the coder wrote at 45 to 64
  tokens/s instead of 65 to 68, and the planner on the NPU took 23 to 31 s
  instead of 14 -- they share the CPU that feeds them, and the memory. The
  table wants a column for "while the other two work".

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
  **Status 2026-10-10, built and not yet Done:** this is the Video Commentator
  (experimental, 2026-10-09), made without the colleague's source. Qwen2.5-VL
  7B on the integrated GPU says what is in a frame in 0.8 to 1.0 s,
  Qwen2.5-1.5B on the NPU says it again in a mood in 0.5 to 0.8 s: a line
  every four seconds, said aloud if asked (every eight to ten seconds then),
  on the sample videos, a webcam or the screen, and as a scene of the Auto
  Demo. Against the Done line above: the caption rate is met; "no invented
  objects" is not -- the vision model's plain sentence holds, the mood model
  embroiders on it (a rider becomes "brave cowboys"), which is why the plain
  line is shown under each; the smart-city sources, TfL included, were not
  tried. Left before it loses its label: the 8B on the NPU as the voice that
  keeps to the facts (R36), lines short enough to be said (R38), French, a
  city camera as a source, and a count of invented details on the six sample
  clips, before and after. **Not to be built as a mood:** a judge of how
  people look (see Deferred).

- [ ] **R39 · P2 · The Page Agent out of "experimental".** *Proves C1, C3.*
  It opens the Auto Demo and it is the picture at the top of the README. What
  keeps the label on, from its own list: a page's length is unbounded (a build
  is 82 to 92 s once the models are loaded, 145 to 162 s the first time) and a
  presenter cannot ask for a short one; every page has the same skeleton; a
  picture is never looked at before it is used, and lettering in one is
  gibberish; a long brief is not always carried whole, and the check that says
  which figures are missing puts none back; three models stay loaded after a
  build, 13 GB for the image model alone, until the brick is closed; a French
  request was tried once.
  **Done:** a "short page" choice that builds in under a minute once loaded;
  two or three skeletons, picked by the planner; each picture shown to the
  vision model the studio already has, and drawn again once if it is not what
  was asked for; a figure the brief gave and the page lost is put back, or
  named on screen; the image model is let go when the page is done on a
  machine where the three do not fit beside another large model; ten briefs,
  three of them in French, built and looked at, with the count of pages a
  presenter would show.
  **Estimate:** 4-6 days. **Depends on:** R33 for the memory rule. (filed
  2026-10-10, from the brick's Inbox entry)

- [ ] **R40 · P2 · Live translation in a real call, with captions that stay on
  top.** *Proves C4, C5.*
  The app is shown in Teams meetings, from the laptop's own microphone and
  speakers, by a presenter who talks while clicking. Fixed for that on
  2026-10-06: the noise floor, and choosing the spoken language. Not handled:
  the other participants' voices come out of the speakers, reach the
  microphone and are transcribed as the presenter's; what a call does to this
  microphone (gain, noise suppression in communications mode) has not been
  measured; short noises come back as stock phrases ("Thank you.", "you"), and
  a clip cut at its 14 s limit leaves a tail that becomes a phrase of its own.
  And the result is in a browser tab, which the audience sees only if the
  presenter shares that tab.
  (1) Two captures the app already has, used together: what the speakers play
  is a reference for what to ignore on the microphone, and a second line of
  captions in its own right ("them", beside "me"). (2) Drop a phrase Whisper
  gives for a sound too short or too quiet to be one. (3) A caption bar that
  floats over the meeting: a small window that stays on top, opened from the
  page (the browser's document picture-in-picture, to be tried in Edge), two
  lines, the language beside them.
  **Done:** in a two-person Teams test call on the laptop's own speakers, no
  line of the other person's is given to the presenter (ten minutes, counted);
  stock phrases counted before and after on a conversation recorded for the
  purpose; the caption bar stays over a shared presentation and follows within
  the delay the panel shows today; what the call does to the microphone is
  written down.
  **Estimate:** 4-5 days. **Depends on:** R35; a second person for the call.
  (filed 2026-10-10, from the Inbox and from how the app is shown)

- [ ] **R37 · P2 · OpenVINO 2026.4, and a choice of model where one has been
  earned.**
  Nine candidate models were tried on 2026-10-05 and all but one need OpenVINO
  2026.4; the project is on 2026.3. What the upgrade would buy, from that
  trial: Qwen3.5-4B reads the receipts as well as today's 7B vision model (68
  values of 68) in 5 s an image against 7 to 9, and as a small language model
  on the integrated GPU it scored 7, 17 and 63 (action items of 7, document
  facts of 17, expense fields of 70) where the 1.5B on the NPU scores 0, 15
  and 47; Qwen3.6-35B-A3B and Gemma 4 26B named all three planted faults in
  the review sample, where today's coder names two among false alarms. What it
  costs: the newer models load through the vision pipeline, think aloud unless
  told not to, and Gemma's tokenizer mangles ten HTML tags; and the upgrade by
  itself changes what today's coder writes -- with "Same page every time",
  Neon Breakout drew its game and never moved the ball on 2026.4.
  **Done:** the project runs on 2026.4 with the full test suite green, and the
  five HTML Creator scenarios and the Auto Demo's page looked at again; Screen
  OCR and the Expense Extractor read with the 4B model if it holds its 68 of
  68 inside the app; Code Review offers a second model if one still finds the
  three faults inside the app; a brick with two qualified models shows a Model
  menu, the others none; the hundred-odd gigabytes of trial models in the
  Hugging Face cache are in use or, on the owner's word, deleted.
  **Estimate:** 2 days for the upgrade and its checks, 2-3 per model adopted.
  **Depends on:** R17; after R36, which needs no upgrade. (filed 2026-10-10,
  from the model trial)

- [ ] **R42 · P2 · New brick: Visual Inspection.** *Proves C1, C2, C5.* Spike
  first.
  The user asked the counting demo for defects ("recognise items, count them,
  defects") and the detector cannot give them: it names everyday objects, it
  does not judge them. The tool for that is anomaly detection, and Intel
  publishes a library for it, Anomalib, which exports to OpenVINO: shown a few
  dozen pictures of good parts, a model learns what good looks like and marks
  what departs from it with a heat map. The demo that follows: teach it on the
  spot -- twenty good bottle caps, biscuits or printed cards in front of the
  camera -- then pass it a bad one. Taught on the integrated GPU, watching on
  the NPU.
  The spike: which of the library's models can be taught on this laptop in a
  minute or two and then compiles for the NPU; how many good pictures a cap or
  a card needs under a stand's light; frames per second. Pictures to practise
  on have to be found: MVTec, the usual set, is non-commercial; another openly
  licensed set, or pictures taken for the project.
  **Done (spike):** time to teach, pictures needed, frames per second on each
  chip and defects found of those shown, on two kinds of object; go or no-go
  recorded here. **Done (brick, on "go"):** teach, watch, a tally of good and
  bad, the heat map over the picture, energy per part inspected (R18); a scene
  for the Auto Demo on a recorded line.
  **Estimate:** 2 days for the spike, 4-5 for the brick. **Depends on:** R17,
  for the library's models and for the pictures. (filed 2026-10-10, from the
  counting videos' Inbox entry)

- [ ] **R43 · P2 · New brick: Ask Your Spreadsheet.** *Proves C3, C4.*
  A workbook nobody would upload -- sales, salaries, a customer list -- and
  questions in plain words: "which region lost most since March?". The model
  does not answer from memory: it writes one query, the query runs on the
  laptop against the file opened read-only, and the page shows the table, a
  chart and the query itself, so that a wrong answer can be seen to be wrong.
  That is the receipts demo's habit of showing its work, and it uses the model
  the studio already has for writing code. Python's own SQLite is enough to
  try it, with no new dependency; what the model writes is never run as a
  program, and a query that would change anything is refused by the database,
  not by a prompt.
  To measure first: the 30B coder and the 8B on twenty questions over a
  workbook of the sample documents' fictional company, right answers and
  seconds each.
  **Done:** sixteen of twenty right on the sample workbook with the model
  chosen, and a wrong one recognisable as wrong from what is shown; every
  answer carries its query and the rows it used; a file of 100,000 rows
  answers in the time a small one does, the model seeing only the column names
  and a few rows; a question the data cannot answer is declined.
  **Estimate:** 4-5 days. **Depends on:** none; R36 if the 8B on the NPU
  writes the query. (filed 2026-10-10, backlog refresh)

- [ ] **R44 · P2 · New brick: Redact Before Sharing.** *Proves C4.*
  The one job that cannot be handed to a cloud service: taking the names,
  addresses, account numbers and faces out of a document before it is sent.
  Drop a PDF, a screenshot or a photo; the page lists what it found, by kind,
  each one ticked; untick what should stay; save a copy with the rest blacked
  out. The parts exist: the vision model reads a page, the language model on
  the NPU can tell a name from a word, patterns catch what has a shape (an
  IBAN, an email address, a phone number), and the sample company's documents
  are fictional. Two things to find out before promising anything: where on
  the page a word is -- the vision model gives text, a black box needs
  coordinates -- and a face detector with a licence that suits (R17). Like the
  receipts demo it has to be honest about what it misses: it shows what it is
  unsure of, and never says a document is clean.
  **Done:** on ten sample documents with sixty planted items, the count found,
  missed and wrongly marked, by kind; a saved copy in which a blacked-out word
  cannot be selected or searched for (pages saved as pictures if that is what
  it takes, and said so); faces in a photo blurred; nothing but the saved copy
  written to disk.
  **Estimate:** 1-2 days to settle the two questions, 5-6 for the brick.
  **Depends on:** R17, R36. (filed 2026-10-10, backlog refresh)

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
  *2026-10-10:* a first number, from the model trial: Qwen3.8-27B with its own
  draft head went from 7.6 to 14 tokens/s on the integrated GPU and from 22 to
  35 on the B60. That is a draft on the same chip, not on the NPU, and the
  model needs OpenVINO 2026.4 (R37). The spike as written is still to do.

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
  *2026-10-10:* the voice models do not compile for the NPU (v0.2.91), so
  "listening" here is the wake word, Whisper and noise suppression, not the
  answer spoken back; and the wake word's licence is an open question (R17).
  Noise suppression is still a planned card in the app: it is the *new brick*
  this ticket delivers.

- [ ] **R27 · P2 · Health-check the curated live feeds.** From the Inbox
  (original wording under Original reports). Check each curated source in the
  background at start-up and every few minutes -- a metadata request for TfL
  clips, a yt-dlp resolve for YouTube -- and mark dead or bot-blocked ones
  unavailable in the picker, with the reason, before anyone picks them.
  **Done:** a feed that fails its check is marked in the picker; the check
  never blocks the UI and is rate-limited; samples prefer feeds that pass.
  **Estimate:** 1 day. **Depends on:** none. (filed 2026-09-11, live smart-city use)
  **Log review 2026-09-22:** three YouTube sources (`6dp-bvQ7RWo`,
  `zMCea32gpmg`, `dfVK7ld38Ys`) reported Video unavailable on 2026-09-22;
  subsequent starts ran successfully. Source: `logs/events.log`.
  **Investigation 2026-09-22 (recurring YouTube blocking):** earlier logs
  contain explicit sign-in/bot challenges; the latest generic Video unavailable
  failures do not establish an account ban. `sources.open_frames` re-extracts
  after stream drops with a fixed two-second delay and no shared pacing or
  reconnect budget. Consider shared extraction pacing, bounded exponential
  backoff, and a cooldown after explicit rate-limit/bot responses; avoid adding
  periodic YouTube health probes that increase requests. The lockfile has
  yt-dlp 2026.8.19 but no yt-dlp-ejs, and setup does not provision/check a JS
  runtime. Audit against [upstream EJS setup](https://github.com/yt-dlp/yt-dlp/wiki/EJS)
  and expose useful extractor warnings currently hidden by `no_warnings`.
  Cookie failure should not be attributed to expiry without evidence.
  [Upstream PO-token guidance](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide)
  currently exempts HLS live streams except the iOS client, so tokens are not
  the first proposed fix for this HLS-only reader. Retain direct-camera/local
  clip alternatives and clearly label recordings. Investigation only; no live
  extraction or runtime changes made.
  **Hardening implemented 2026-09-22:** shared eight-second YouTube pacing,
  short successful/failed-resolution caches, a 15-minute bot/rate-limit
  cooldown across feed restarts, interruptible extraction with a 60-second
  deadline, and bounded live/clip reconnects. Added network read/open timeouts,
  per-feed recovery/freshness reporting, retained individual errors, and a
  heartbeat that ages counts while cameras are silent. The matching yt-dlp
  JavaScript solver is now locked; `smart-city-doctor` checks setup offline
  and can probe one explicitly requested URL. No periodic YouTube scanning
  or automatic source substitution. Live verification: an anonymous YouTube
  probe returned Video unavailable and was classified correctly; the TfL
  Piccadilly/St James's camera decoded successfully with the new timeouts.
  The curated-source health picker part of R27 remains open.
  *2026-10-10:* less urgent than when filed. Since 2026-10-09 the panel and
  the Auto Demo open on videos kept on the machine, so a dead camera no longer
  breaks a show. The mark in the picker is still worth its day for manual use.

### Sturdiness

The earlier tickets in their order, then three that came out of October: what
state the machine is in (R45), upgrades for the copies the setup assistant
installs (R46), and the runtime's faults (R47).

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
  **Part done 2026-10-03:** an idle brick's model can now be unloaded from
  the hardware panel (or `POST /api/bricks/<id>/stop`); unloading the 7B OCR
  model took GPU memory from 6.8 GB to 2.2 GB. Still open: one model shared
  between code review and HTML creator, a memory figure per brick in the
  panel, and the 20-cycle growth test.
  *2026-10-10:* two more for this ticket. The Page Agent keeps three models
  loaded after a build, about 13 GB for the image model alone. And on
  2026-10-09 its models failed to load twice (`[GPU] ProgramBuilder build
  failed`, `CL_INVALID_EVENT`) while the Expense Extractor was reading
  receipts -- not reproduced, possibly memory. A memory figure per brick in
  the hardware panel would have said.

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
  *2026-10-10:* much of this arrived by other doors: the app opens on a choice
  of two (Auto Demo, Manual demo), the Auto Demo needs no settings, the setup
  assistant ends on a working Studio, and demos come with samples. Left as
  written: engine and model settings under an "advanced" fold, the
  keyboard-only path, the narrow window.

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
  *2026-10-10:* the case is stronger. Since the review `app.py` has gone from
  1,224 lines to 2,111 and `app.js` from 2,282 to 5,580, with five bricks and
  a director added; nothing has been extracted.

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
  *2026-10-10:* the setup assistant's own tests are PowerShell and CI is
  Linux, so nothing on CI runs them (on Windows,
  `tests/test_setup_assistant.py` does). A Windows job that runs the two
  `.Tests.ps1` files and walks the assistant's page once with a plan is the
  first step of the Windows half of this ticket.

- [ ] **R45 · P2 · Know the machine's state before a show.**
  The same demo on the same laptop is not always the same speed, and nothing
  on screen says why. On battery the integrated GPU's tokens per second fell
  by about a third (2026-10-03). After two hours of builds back to back the
  laptop held itself to 15 W and every demo took half as long again
  (2026-10-08; cause unknown: heat, a Dell profile, the charger). The external
  card changes the GPUs' names and which chip the large models take. And a
  model that is not on the disk yet is a download in front of an audience.
  A "Before the show" page, and the same as one command: mains or battery; the
  package power under a short fixed load against what this machine gave when
  fresh, which is how the 15 W state shows; which GPUs are there; the drivers;
  for the scenes or demos ticked, whether every model is on disk and compiled,
  with a button that runs each sample once; then the same with the network
  off, which is what R10 called show prep.
  **Done:** the page says "plugged in, not held back, models ready" or names
  what is not, in under two minutes; a laptop in the 15 W state is told apart
  from a fresh one (provoke it once to prove it, and time how long it lasts);
  the hardware panel shows mains or battery beside the power figure; the
  presenter's notes say what to do about each.
  **Estimate:** 3 days. **Depends on:** R18. (filed 2026-10-10, from three
  Inbox entries and R10's show prep)

- [ ] **R46 · P2 · Upgrades for a copy that came from a ZIP.**
  The setup assistant made "Download ZIP" the first way to get the app -- it
  is the one that needs nothing installed -- and a copy from a ZIP cannot
  upgrade itself: the footer's Upgrade works through git, and the README says
  to download the ZIP again. With several versions a week, laptops installed
  that way will stay on the version they were given.
  Give those copies the same button: fetch the tagged version's archive from
  GitHub, check it, put it in place of the code while keeping what belongs to
  the laptop (the environment, the installer kept in the project, the logs,
  the sample videos), bring the environment up to date, start again. The
  assistant's page is the natural place to show it happening.
  **Done:** a ZIP copy at one version upgrades to the next from the footer, on
  the imitated bare laptop and on a real one (R33); a download that fails or
  is cut leaves the old version working; a file the user changed is named, not
  silently overwritten; a copy made with git keeps upgrading as it does today.
  **Estimate:** 3 days. **Depends on:** R16, for a tag that means "tested".
  (filed 2026-10-10, backlog refresh)

- [ ] **R47 · P2 · Runtime faults to pin down and report upstream.**
  Six things seen in the OpenVINO runtime or the drivers, worked around in the
  app and understood in none. (1) The NPU's compiler ends the whole process,
  with no Python error, when it is handed the voice model ("LLVM ERROR: Failed
  to infer result type(s)", exit code 127); worked around by never handing it
  over (v0.2.91). (2) After image and language models have shared the
  integrated GPU a process can spin forever at exit, five times of five on one
  sequence; worked around with `TerminateProcess`; it matters to the in-app
  upgrade, which waits for the old process to end. (3) On the GPU an answer
  depends on what the model generated before it, with sampling or without;
  HTML Creator reloads the model when asked for the same page every time,
  nothing else does. (4) On the external B60 the 30B model's second and later
  answers wait about 20 s for their first token. (5) `CL_INVALID_EVENT`,
  twice, while two large models loaded beside a third at work (R11). (6) At
  temperature 0.2 the 30B coder ran three long pages of fifteen into the same
  CSS rules until the tokens ran out; the Page Agent now writes at 0.7 with a
  watch that stops a loop, HTML Creator still writes at 0.2 with none.
  **Done:** each of (1) to (5) has a script of its own under `scripts/` that
  shows it on this laptop, or a line saying it could not be made to happen
  again; those that reproduce are written up for the OpenVINO project, for
  whoever holds the account to send, and the issue is linked here; for (6) the
  watch lives where both bricks use it, and HTML Creator's samples have been
  counted for loops at its temperature.
  **Estimate:** 3-4 days, a fault at a time. **Depends on:** none. (filed
  2026-10-10, from six Inbox entries)

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
  *2026-10-10:* the Page Agent (2026-10-08) is a first step this ticket did
  not foresee: three models on one job, conducted by plain code. What is still
  missing is what Intel's pitch means by an agent -- a model that decides
  which tool to call. Qwen3-8B on the NPU as the one that decides and the 30B
  on the GPU as the one that writes is the pairing to try first; R43 and R44
  would be two more tools for it.

- [ ] **R28 · P3 · Extend dropped-request recovery to the other Whisper
  demos, if the fault recurs.** From the Inbox (original wording under
  Original reports). Meeting notes and the voice assistant get the reload and
  retry live translation has had since 0.2.40 -- only once the activity log
  shows the fault again ("reloading the model and retrying" entries).
  **Estimate:** half a day. **Depends on:** evidence. (filed 2026-09-11, activity log)
  *2026-10-10:* wider than Whisper now. A lost NPU took the launcher down
  twice, on 2026-10-05 and -06 (upstream openvinotoolkit/openvino#38403,
  driver 32.0.100.5540); since v0.2.63 bricks take turns on the NPU, and after
  a loss the speech and language models carry on on the GPU. Since 2026-10-07
  two models on the NPU is the ordinary case (speech and the notes) and it has
  held. Not tried: that pair while the integrated GPU loads a large model,
  which is what was going on both times. To do when the next NPU driver
  arrives: the scenario that lost it, ten times with turn-taking and ten
  without.

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

- [ ] **R48 · P3 · New brick: Interpreter.** *Proves C1, C5.*
  Two people, two languages, one laptop between them: each speaks in turn and
  hears the other in their own language. Live Translation already hears any
  language and writes English; what is missing is the way back (English into
  French, say) and the voice. The 8B on the NPU can translate a sentence.
  Whether the studio's voice can speak French is the first thing to find out
  -- the commentator's French is waiting on the same answer.
  **Done:** a sentence comes back spoken in the other language within four
  seconds of its end, both ways, on the fourteen French sentences already used
  for Live Translation and on their English; the two sides are told apart on
  screen; nothing is kept.
  **Estimate:** 1 day to find out about the voice, 4-5 for the brick.
  **Depends on:** R36, R38, R40. (filed 2026-10-10, backlog refresh)

- [ ] **R49 · P3 · New brick: Move Counter.** *Proves C2, C5.*
  A second thing for a visitor to do in front of the camera: a pose model on
  the NPU draws the skeleton and counts squats or arm raises for thirty
  seconds, with the watts beside the count. It is a game, and it is the NPU
  doing something continuous for very little power. It counts and measures
  angles; it says nothing about a body. It needs a pose model with a licence
  that suits: the detector's family has one, under the same AGPL that R17 has
  to settle, so another is to be found.
  **Done:** 25 frames a second on the NPU with the package power shown; ten
  repetitions counted as ten, give or take one, on five recorded people; the
  camera's picture is not kept; the model is in R17's inventory.
  **Estimate:** 3-4 days. **Depends on:** R17. (filed 2026-10-10, backlog
  refresh)

- [ ] **R50 · P3 · Meeting notes that know who spoke.** *Proves C4.*
  On the scripted meeting of 2026-10-07 the notes gave one speaker's task to
  another: the model has no way of knowing who said what. Half of the answer
  is free. The app captures the microphone and what the speakers play
  separately, so "me" and "them" can be told apart with no model at all (the
  same two captures as R40). The other half -- telling "them" from one another
  -- needs a voice print for each speaker, which the Voice Clone Studio
  already computes for the voices it enrols.
  **Done:** lines labelled "me" and "them" in a Teams call; on a scripted
  meeting of three synthetic voices, nine lines of ten given to the right
  voice; on the four test meetings, the tasks given to the right owner,
  counted before and after.
  **Estimate:** 1 day for "me" and "them", 3-4 for the rest. **Depends on:**
  R40. (filed 2026-10-10, backlog refresh)

- [ ] **R51 · P3 · The app in French.**
  The Auto Demo tells its story in English or in French. Everything else is
  English: the panels, the setup assistant, the commentator's lines, the
  README. The commentator's French is an open item of its own (R23); the rest
  has not been asked for, and is written down here so that it is a decision
  rather than an oversight. If it is wanted: the panels' wording moved out of
  the code into one file per language, the commentator and the assistant's
  voice answering in the language chosen, the setup assistant's page in both.
  **Done:** a French speaker installs the app and runs three demos without
  meeting an English sentence that is the app's own.
  **Estimate:** 4-6 days. **Depends on:** a decision. (filed 2026-10-10,
  backlog refresh)

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
  *2026-10-10:* one more reason it is not on stage: its search reads the
  embedding model the old way (R06).

- [ ] **New demos beyond the tickets above:** after the pilot journeys meet
  their gates. Inbox triage now folds into R26, noise suppression into R25.
  *2026-10-10:* overtaken. Five bricks and the Auto Demo were built on request
  since this was written. New bricks are tickets now (R22's Image Studio,
  R25's noise suppression, R41 to R44, R48, R49), and the ideas that are not
  there yet are listed below.

- [ ] **Remote multi-user access:** separate design for authentication, sessions,
  permissions and data boundaries before expanding local-only deployment.

- [ ] **Large architectural changes:** database migration, automatic model sharing,
  routing optimization or frontend-framework migration only with measured need.

- [ ] **Email drafted in the writer's own voice** (the app's planned "Inbox
  Triage & Draft Assistant" card): set aside by the user on 2026-10-04 -- "not
  convinced by this demo". The plan and its feasibility test stay in
  [docs/EMAIL_VOICE.md](docs/EMAIL_VOICE.md); nothing is to be built from them
  for now. The card still shows in the app as planned.

- [ ] **A judge of how people look** (the "fashion" mood asked for with the
  commentator): not to be built as one more mood. A machine passing judgement
  on real people at a company's stand goes wrong in ways a bottle counter does
  not. A version that could be shown -- asked for by the person, one still,
  about the clothes and colours only, kind only, nothing kept -- waits for a
  word with legal.

- [ ] **Brick ideas that are not tickets yet,** a line each, to be given a
  number when a show asks for one:
  - *Stand host* -- the Voice Assistant answering from a folder (the laptop's
    own fact sheet) instead of from memory: Document Q&A behind a wake word.
  - *Whiteboard to notes* -- a photo of a whiteboard becomes tidy notes and a
    diagram, with the vision model; a mode of Screen OCR more than a brick.
  - *Depth for the webcam* -- a depth model for a blur that follows distance,
    in Webcam Effects.
  - *Sharper pictures* -- an old photo or a small video frame enlarged on the
    integrated GPU, before and after.
  - *Ask the code* -- Document Q&A over a repository, answered by the 30B
    coder with the files named.
  - Already tickets: photo search (R29), an assistant in every app (R30), a
    local agent (R26), text-to-video and speech-to-speech (R31).

## Done

R20 landed on 2026-09-11. R01, R02 and R04 were implemented on 2026-09-10 (see the
[validation notes](docs/TRUSTWORTHY_OUTPUT.md)); R01's hardware gate was
verified on 2026-09-11. For R02, manual correction remains through the CSV;
no in-app approval flow is claimed. R04 restores history without resurrecting
previously active workers.

Since the rebuild: R32 on 2026-10-03, and R21 on 2026-10-09 as the Auto Demo.
Closed straight from the Inbox, with their reports under Original reports: the
same page every time (2026-10-04), a page answered with a code fence
(2026-10-08), the voice that took the launcher down on the NPU (v0.2.91), the
setup assistant (v0.2.92) and a Studio that stays stopped (v0.2.93).

- [x] **R21 · P2 · One-click stage scenarios.** *Proves C1, and C2 with R18.*
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
  **Done 2026-10-09, in another shape: the Auto Demo.** Not three named
  scenarios but a loop of six scenes, ticked when it is started from the app's
  first screen, each told in a caption that says what the audience should
  notice, in English or French, with the chips at work down the right. Against
  the Done line: one click from the first screen; a scene whose model or
  device is missing is shown as unable to play, with the reason, before the
  loop starts; a three-scene playlist ran eight times with no failure and
  nothing left loaded. Not as written: "steady state within a minute" does not
  fit a scene that builds a web page (2 min 23 s); *Hybrid meeting* was not
  built, the loop making no sound; energy per result is not on the stage
  (R18). What is left is R34.

- [x] **R32 · P2 · Stream the language models: live speed, answers as they
  are written, and cancel.** *Proves C5.*
  Every LLM call blocks until the whole answer exists: the panel's tokens/s
  appears afterwards, the audience watches a spinner (54 s for an HTML page),
  and a brick that answers one request at a time can't be stopped mid-way.
  One control object through both LLM backends and the vision-language
  extractor gives all three. Design, alternatives and the open decisions are
  in [docs/STREAMING.md](docs/STREAMING.md).
  **Evidence (2026-10-03):** Qwen2.5-1.5B, 200 tokens: streaming costs 8% on
  the iGPU (83.9 against 91.0 tok/s) and nothing on the NPU (55.8 against
  53.9); first text after 0.09 s and 0.39 s; a cancel returns in 0.30 s and
  0.79 s with the partial text, and the model answers the next request
  normally. llama.cpp streams and stops the same way.
  **Done:** step A -- every brick's ✕ works mid-answer and the panel shows a
  live tokens/s while it generates; step B -- document Q&A, code review, HTML
  creator, screen OCR and meeting notes show their answer as it is written,
  and a stopped answer is labelled incomplete. With no control passed, a
  call behaves exactly as today.
  **Estimate:** A about 1 day, B about 2 days; C (the voice assistant speaking
  sentence by sentence) optional, 1-2 days. **Depends on:** none; partial text
  is polled so it adds no delivery mechanism while R05 is open.
  (filed 2026-10-03, user request)
  **Done 2026-10-03:** steps A and B, for the five bricks that answer a
  request with a language model. Verified through the launcher on the XPS 14
  (on battery): document Q&A on the NPU 51.5 tok/s with a live reading of
  34-55 while writing; code review 33.0 and HTML creator 32.2 on the iGPU's
  30B model; screen OCR 20.1 on the 7B vision model; a long meeting on the
  NPU summarised part by part at 46.8. Each stop returned in 0.30-0.36 s with
  the text so far and the model still loaded; from the page, the text grows
  every 300 ms and a stopped answer stays under "Stopped -- this answer is
  incomplete". Two things the evidence above got wrong, corrected in the
  build: the 8% cost was run-to-run variation (interleaved medians show
  none, on the 1.5B and the 30B); and a callback is not a token (154
  callbacks for 220 tokens of HTML), so the live rate counts tokens through
  a streamer that sees each one. "Every brick's ✕" holds for those five; the
  loop bricks stop as before, between items. Step C and those loops are in
  the Inbox.

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

- [ ] **Stream the language models: live tokens/s, answers word by word,
  and cancelling one mid-way.** The hardware panel (0.2.53) shows tokens/s
  only once an answer is finished, because no LLM call streams; a brick that
  answers one request at a time can't be stopped mid-answer, so its ✕ is
  disabled until it finishes. One change covers all three: a streamer in the
  two LLM backends that counts tokens as they arrive, passes text on, and can
  return "stop". Agreed with the user as the step after the panel; likely
  its own ticket next to R19/R21. (filed 2026-10-03, user request)
  Sorted into **R32** on 2026-10-03.

- [x] **The studio's voice takes the launcher down when it is put on the
  NPU, and is unusable on a GPU.** Seen while measuring voices for the
  Video Commentator, in a script of its own: `voice_model.
  accelerate_tts_with_openvino(tts, device="NPU")` -- what the Voice
  Assistant does when its chip is the NPU, and what the Voice Clone Studio's
  OpenVoice model does there -- ends in the NPU compiler's "LLVM ERROR:
  Failed to infer result type(s)" (the model's shapes have no upper bound)
  and the process exits with it: no Python error to catch. On the
  integrated GPU the same voice compiled again for every sentence of a new
  length, 16 to 22 s each (0.05 s for one it had seen); on the CPU through
  OpenVINO, 0.5 s. One observation, on today's driver, with the B60
  unplugged; the Voice Assistant's README offers `--compute-device NPU`.
  To check in the app itself, then either give the voice bounded shapes or
  keep it on the CPU whatever chip the rest of the brick uses, as the
  commentator does. (filed 2026-10-09, voice timings)
  *2026-10-10:* seen again through the two bricks' own code, with the B60
  plugged in, no microphone opened and nothing played:
  `VoiceAssistantSession(Engine.OPENVINO, device="NPU")` and
  `VoiceCloneSession(Engine.OPENVINO, model="openvoice", device="NPU")`
  both end the Python process with the same LLVM error (exit code 127).
  So choosing the NPU for either demo in the launcher takes the launcher
  down. The README no longer lists the NPU for them and says so under
  "Good to know"; the menus still offer it.
  *Fixed the same day, at the user's request.* Tried part by part, one
  process each: the speaker ends the process on the NPU with or without
  bounded shapes; the tone converter is refused there with an ordinary
  error; on a GPU both work and are compiled again for every new length
  (16 to 22 s and 5 s); on the CPU they take 0.2 and 0.5 s. So the voice
  models are never handed to the NPU: `voice_model.compile_device` refuses
  it by name before anything is imported or loaded, and treats "AUTO" as
  the CPU. The Voice Assistant listens and answers on the chip chosen and
  speaks from the CPU, with a row of its own in the hardware panel; the
  Voice Clone Studio shows the NPU greyed out with the reason, and its
  routes answer 400. Checked with the two scripts that used to exit with
  127: the assistant now loads on the NPU and speaks, the clone is refused
  with the message. Not run in the launcher with a microphone.
  Fixed in v0.2.91. Sorted on 2026-10-10: what was not run, the Voice
  Assistant with a microphone, into **R35**; the report upstream into **R47**.

- [ ] **A cloned voice is slowed threefold by a video playing beside it.**
  Chatterbox made a line in 8 to 11 s alone and in 18 to 30 s inside the
  Video Commentator, where a 1080p video is decoded and re-encoded for the
  page at 30 frames a second on the same CPU. Its first line also pays a
  warm-up of about twenty seconds. Worth a look at how many threads it
  takes, and at warming it when the voice is chosen rather than at the
  first line. (filed 2026-10-09, voice timings)
  Sorted into **R38** on 2026-10-10.

- [ ] **Speech speed in the panel is untested with real audio.** Times real
  time is computed per utterance (unit-tested) for live translation and
  meeting notes, but was not watched live -- nothing was played through the
  speakers while building it. Check it on the next run with audio.
  (filed 2026-10-03, hardware panel)
  Sorted into **R35** on 2026-10-10.

- [ ] **The iGPU's tokens per second fall by about a third on battery, and
  nothing on screen says so.** Measured while testing streaming, on battery
  at 28% with the Balanced plan: Qwen2.5-1.5B ran at 86-88 tok/s for its
  first two answers, then 50-58 for the next twenty; the 30B coder held
  about 29-33 tok/s. Earlier the same day the same models gave 87 and
  44-45. The NPU did not move (51-55 throughout). Not yet separated: the
  battery's power limit against heat after repeated runs. Worth knowing
  before a show (plug in, or say the number is the on-battery one), and the
  hardware panel could show mains or battery beside the power figure.
  Related: R18's battery drain rate. (filed 2026-10-03, streaming tests)
  Sorted into **R45** on 2026-10-10, with a line in **R18** for the panel.

- [ ] **What streaming left out.** R32's optional step C -- the voice
  assistant speaking its reply sentence by sentence as it is written -- is
  not built. The language-model calls inside the loop bricks (expense
  structuring, screen-memory OCR, the voice assistant's reply) have no live
  tokens per second and stop between items, not mid-answer. Meeting notes'
  streaming was verified through its runner with a typed transcript, not
  from the page with real audio. (filed 2026-10-03, R32)
  Sorted on 2026-10-10: the voice speaking as the answer is written into
  **R38**; live figures inside the loop bricks into **R18**; Meeting Notes
  from the page with real audio into **R35**.

- [ ] **On the external B60, the 30B model's second and later answers wait
  about 20 s for their first token.** Seen while tuning the HTML Creator
  scenarios with the card plugged in: first answer after loading, first token
  in 0.7 s; the next two, 20.2 s and 18.5 s, then 60-63 tok/s as usual. The
  integrated GPU does not do this (0.3-0.6 s every time, same prompts, same
  session code). Not on the stage path, where the card is absent; worth a
  look before any demo that shows the B60. (filed 2026-10-03, HTML Creator
  scenarios)
  Sorted into **R47** on 2026-10-10.

- [x] **A generated page is not the same twice, so a rehearsed page is not
  the page the audience gets.** The language models sample (temperature 0.2
  over the model's own settings): three runs of one HTML Creator prompt gave
  three pages, two of them checked and correct, and while tuning the new
  scenarios every looser wording produced at least one page with a visible
  fault (text zooming with the hero picture, a game that restarted its score
  on every lost life, a counter ending on 123.0 instead of 122.9). Decoding
  without sampling for the HTML Creator would make a sample prompt give one
  known page; it costs "Generate again gives a new design", so it may want
  to be a switch. To decide, then re-verify the five scenarios under it.
  (filed 2026-10-03, HTML Creator scenarios)
  **Done 2026-10-04, and the cause above was wrong.** Sampling is seeded: on
  the CPU and NPU the same prompt gives the same text every time. On the GPU
  a page depends on what the model generated before it, with or without
  sampling; a freshly loaded model gives the same page every time (30B: the
  board three times, the game and the travel site twice each, other pages in
  between). The HTML Creator has a "Same page every time" switch, off by
  default, that reloads the model before each page (about 20 s) and takes
  the most likely token (which is what makes llama.cpp repeat). The five
  scenarios' pages under the switch were checked. Still open: the same
  history-dependence applies to every GPU answer in the app (code review,
  document Q&A on the GPU), where nothing reloads the model.
  Done 2026-10-04. Sorted on 2026-10-10: what is left, an answer on the GPU
  depending on what came before it, into **R47**.

- [ ] **The drafting demo should learn how we write, then write email our
  way.** The user's aim for the planned "Inbox Triage & Draft Assistant"
  card, which the roadmap had folded into the agent ticket (R26). Planned in
  [docs/EMAIL_VOICE.md](docs/EMAIL_VOICE.md): a style card plus the writer's
  own most similar emails as examples, a check that marks facts the draft
  made up, a fictional mailbox for the stage. A feasibility test on the XPS
  14: the 30B model on the integrated GPU went from 3 to 6 of 7 of a
  writer's habits when guided, in 3.3 s a draft; the 1.5B model on the NPU
  cannot write in a voice; guided drafts invented a deadline. Five decisions
  wait in the note, and it may deserve its own ticket rather than a place
  inside R26. (filed 2026-10-04, user request)
  **Set aside by the user on 2026-10-04:** not convinced by this demo. The
  note stays as it is; nothing is to be built from it for now.
  Sorted into the deferred list on 2026-10-10: set aside by the user on
  2026-10-04.

- [ ] **An Auto Demo for a stand: the app runs itself in a loop, several
  demos at once, and says what is happening.** Groundwork in
  [docs/AUTO_DEMO.md](docs/AUTO_DEMO.md): scenes as data (which is R21), a
  director in the launcher that the page follows, captions with live
  figures, and what it takes to run for hours with nobody there. Measured on
  the XPS 14: the iGPU carries object detection and the 30B model together
  (43.5 tok/s against 45.7, 10.3 fps against 12.0), the NPU is unaffected by
  GPU work, a model that takes every CPU core costs the other two chips a
  third; a three-scene playlist ran eight times with no failure and nothing
  left loaded. Needs first: R21, a keep-awake request (the app has none), a
  way to delete one expense report. Five decisions wait in the note.
  (filed 2026-10-04, user request)
  *2026-10-08:* the five decisions are taken and the foundation is built --
  scenes as data, the director, its routes, keep-awake, tests against
  stand-ins. The note's "Where it stands" lists what is there and what is
  not: the page's side, a long run on the real machine, the proofing pass
  on Object Detection and Document Q&A that the fourth scene waits for,
  and deleting the loop's own expense reports.
  *2026-10-09:* the page's side is built and the loop has been round once
  on the real machine with no failure (three scenes, a pause, a stop). Next,
  in the note's order: a run of several hours, a caption that stays in view
  beside the result, and the proofing pass for the fourth scene.
  *2026-10-09, later:* rebuilt on the user's feedback after watching it
  ("a wall of text on top, the result is below the fold, scrolling pauses
  it"). The app now opens on a start screen (Auto Demo / Manual demo), and
  the Auto Demo has a stage of its own: a caption always on top that tells
  the demo as a story, a sentence or two at a time, in English or French;
  the demo's outputs only, made to fit without scrolling; the chips down the
  right. A touch brings up a popup (keep playing, pause, stop) that answers
  itself after fifteen seconds; a pause interrupts nothing. Watched for a
  whole turn in English and most of one in French. The receipts scene plays
  the five worn receipts, as asked. Next: a run of several hours, and the
  proofing pass on Object Detection for the fourth scene.
  *2026-10-09:* asked for by the user: a scene for the Video Commentator,
  the proofing pass on Object Detection, and a choice of scenes when the
  loop is started. All three are built. The start dialog ticks each scene
  (and whether the camera may be used); the commentator has its scene and
  its view; Object Detection had its pass (a video file as a source, a
  wrong source refused before a model loads, boxes sized to the picture,
  the screen from 12 to 19 frames a second: its README has the rest) and
  "Seeing and answering" plays, on a view of its own, with nothing held
  back. Watched in both languages on their own; not yet a whole turn of
  six scenes, nor a run of several hours.
  *2026-10-09:* the user, having watched "Seeing and answering": "not very
  convincing -- we can't really understand the principle behind the Q&A
  part", and of the camera, "potentially engaging for people to be
  interacting with". It is two scenes now. **Documents**: one question
  asked of the model alone, then the folder read, then the same question
  again, with the files marked as they are read and used (alone, the model
  answers "the team's captain, on September 16, 2023"; with the files,
  Priya Desai on September 17). **The camera**: the detector on the NPU
  boxes the visitor at 31 frames a second while the vision model says in a
  plain sentence what is going on -- and what it says of people is kept to
  what they do, by a rule in code, the model having said "wearing glasses"
  in every line whatever it was asked. Document Q&A's panel gained a
  "Without the documents" tick for the same contrast by hand. Left: the
  camera scene in front of visitors (it was tried on one person sitting at
  the laptop), a whole turn of six, and the run of several hours.
  Built, 2026-10-08 and -09: it closes **R21**. Sorted on 2026-10-10: what is
  left into **R34**.

- [ ] **Street videos on disk for the city monitor, and what the detector makes of them.**
  Asked for by the user on 2026-10-09: a local video as the demo's default,
  for a stand and for manual use, retrieved by the installer from where it
  was published. Three were chosen from openly licensed footage and are
  fetched, checksummed, into `sample-data/videos/` (not in git) by the
  first-launch helper, by Prepare models, or at first use; the panel opens
  on two of them, one per chip, and the Auto Demo plays them with no
  network. Both play at their full 24 and 25 frames a second on the
  integrated GPU and the NPU together.
  Measured with the brick's own detector (YOLO11s, 0.5): the Toronto
  crossing gives 9 to 23 things a frame and never none; **the Shibuya clip
  gives a median of 4 among several hundred people, and nothing at all in
  one frame of six** (its middle is a tight shot of a packed crossing; at
  a threshold of 0.25 the median person count rises from 1 to 5); Intel's
  own sample clip is an empty street most of the time (nothing in 63 frames
  of 108) and plays at 12 frames a second. So one of the three is a strong
  demo, and the default pair showed it beside a crowd the detector mostly
  cannot box.
  *Same day, settled:* two more clips from Wikimedia Commons were fetched
  and measured with the brick's detector and tracker over one pass each. A
  crossing in Tyumen (CC BY-SA 4.0, 31 s): a median of 9 boxes on screen,
  never none, 80 things crossing the picture a minute. Slovenska street in
  Ljubljana (CC BY 3.0, 54 s): 7, none in 1% of frames, 32 a minute, with
  more kinds of thing (buses, cyclists). Toronto: 11, never none, 122.
  Shibuya: 3, none in 28%, 27. Tyumen took Shibuya's place in the default
  pair and in the Auto Demo; Shibuya stays in the list as an extra;
  Slovenska was not kept. Tyumen is short: it plays twice and a bit in the
  Auto Demo's seventy seconds, so its counts are of the same cars again.
  Left: a detector that sees small people (a larger input, or tiles).
  Also: the two Commons files are Commons' own 1080p encodes, which it may
  redo one day -- the checksum would then refuse them; a mirror the user
  hosts, added to `urls`, is the remedy. (filed 2026-10-09, user request)
  Built 2026-10-09. Sorted on 2026-10-10: the counts and the small people into
  **R03**, the mirror for the two Commons files into **R10**.

- [ ] **Video Commentator, experimental: a video watched and commented on in a mood.**
  Asked for by the user on 2026-10-09 ("video commentary with a gentle
  twist of mood... maybe a fashion judgment one too"). Measured first, on
  the XPS 14: the vision model the studio already has (Qwen2.5-VL 7B) says
  what is happening in a frame in 0.8 to 1.0 s on the integrated GPU (25
  tok/s, first word after 0.2 s, a frame 672 wide; no better read at 1280,
  and 1.3 s), and Qwen2.5-1.5B gives the line a mood in 0.5 to 0.8 s on the
  NPU. Asked for the mood directly, the vision model writes the same plain
  sentence with "bustling" in it, which is why there are two models.
  Built: the brick (`bricks/video-commentary`, composing screen-ocr and
  doc-qa, no model of its own), five moods, the studio's sample videos, a
  webcam or the screen, the mood changed while it plays, a line every four
  seconds, quiet while nothing changes. In the app: first line 14 s after
  Start, then 0.9 s to see and 0.7 s to say, both chips on their own rows.
  What it does not do well yet: the small model embroiders in a mood (a
  rider becomes "brave cowboys", a herd goes "to market") -- the plain line
  is shown under each for that reason; "deadpan" is barely a voice; a line
  is about a second and a half behind the picture.
  **Not built, and not to be built as one more mood: the fashion judge.**
  A machine passing judgement on how real people look, at a company's
  stand, goes wrong in ways a bottle counter does not (it drifts from
  clothes to bodies, age, dress worn for faith; and a camera that judges
  visitors is not one that counts them -- worth a word with legal first).
  A version that could be shown: asked for by the person, one still, about
  the clothes and colours only, kind only, nothing kept; or on outfits and
  not on people. Also open: speaking the line (the voice brick exists), a
  scene on the Auto Demo's stage, and Qwen3-8B on the NPU for a voice that
  keeps to the facts. (filed 2026-10-09, user request)
  *2026-10-09:* the scene is on the Auto Demo's stage, and the line can be
  said aloud (asked for by the user: "an option for speaking the line
  aloud (with our cloned voice?)"). Off by default. The studio's own voice,
  read in a delivery that follows the mood, is made on the CPU in 0.6 to
  1.0 s for a line of 5 to 8 s, and the next look waits for it to have been
  said: a line every eight to ten seconds. The cloned voice is the one
  enrolled in the Voice Clone Studio, lent; it works and it cannot keep up
  -- 18 to 30 s a line with Chatterbox while the video plays, 22 s with
  OpenVoice on the GPU -- so it is two lines a minute, each twenty seconds
  behind its picture. Checked without playing anything: each line's sound,
  read back by Whisper, gave the line. Left: lines short enough to be
  said (an upbeat one ran to 28 words and 11 s), a faster cloned voice,
  French, and a sound option for the Auto Demo's scene if a stand ever
  wants one.
  Built 2026-10-09. Sorted on 2026-10-10: it is **R23**, which says what its
  Done line still asks; the judge of how people look into the deferred list.

- [ ] **A second pair of videos for counting: a factory line, a herd.**
  Asked for by the user on 2026-10-09, after the street pair: "a
  manufacturing duo -- recognise items, count them, defects -- or sheep
  counting or alike". Four openly licensed clips were fetched to a scratch
  folder (not into the project) and put through the city monitor's detector
  and tracker, one pass each:

  | Clip | About | Boxes on screen | Frames with none | Counted | Mistaken for |
  | --- | --- | --- | --- | --- | --- |
  | Bottle capping machine (Commons, CC BY 3.0, 21 s, 1080p, 7.1 MB) | bottles | median 3 | 14% | 15 bottles | nothing |
  | Cattle on a road (Commons, public domain, 28 s, 720p, 20.7 MB) | cows | median 8 | 9% | 90 cows, and the rider, horses, a dog | 1 elephant |
  | Fruit on a conveyor (Intel sample-videos, CC BY 4.0, 61 s, 17.8 MB) | fruit | median 0 | 60% | 35 apples, 10 oranges, 9 bananas, 7 broccoli | vase, ball, bird, teddy bear |
  | Sheep filing past (Commons, CC BY 3.0, 47 s, 640x480, 13.2 MB) | sheep | median 0 | 77% | 113 sheep | 122 "cows" |

  The bottles and the cattle are good demos; the fruit comes one at a time
  with an empty belt between, is counted several times over and plays at 60
  frames a second, more than the NPU detects (49); the sheep clip is
  hand-held and blurred, and half its sheep are called cows. Counts run
  high everywhere (a track lost and found is counted twice): the herd is a
  few dozen animals, not 90.
  *Same day, built on the user's word:* the bottles and the cattle are in
  the fetched set (six clips, about 120 MB), a feed says what it counts
  (`COUNTING` in the brick's `types.py`: street, line, herd; the **Counts**
  menu on a feed's card), "A herd and a line, two chips" is offered beside
  the street pair, and the Auto Demo plays it in turn with the streets --
  the streets on the first turn of the loop, this on the second, as the
  user asked after seeing them back to back. Left: a
  second factory clip as good as the bottles, and counts that do not run
  high. **Defects are out of this detector's
  reach**: it names everyday objects, it does not judge them; Intel's bolt
  clip is for a model trained for it (and MVTec, the usual defect set, is
  non-commercial). (filed 2026-10-09, user request)
  Built 2026-10-09. Sorted on 2026-10-10: the counts into **R03**, the defects
  into **R42**.

- [ ] **Document Q&A was answering from passages drawn nearly by lot.**
  Found while replaying the stale-index report (R06). The embedding model,
  Qwen3-Embedding, is trained to be read at its last token; the pipeline
  read it at its first. On the sample folder, with ten questions whose
  answer is in one known file, the right file came first for 2 of 10, and
  for 8 of 10 once read at the last token (9 of 10 among the four passages
  read), on the integrated GPU and the NPU alike. On top of that the prompt
  ended on "refer to the excerpts by their [number]", and the whole answer
  to four of six questions was "[1]". Both fixed on 2026-10-09; with the
  new prompt, five questions of five that have an answer got the right
  facts and three of four that have none were declined. Two things on the
  NPU as well: indexing any folder of more than one passage failed inside
  the plugin (a call with two texts), and one text took 1.6 s; now one text
  a call in a fixed shape of 512 tokens, 0.07 s each.
  What is left: (1) **screen memory still reads the model at its first
  token**, because its index on disk was built that way and the two kinds
  of vector cannot be mixed -- its search is as weak as Document Q&A's
  was, and moving it means embedding again every text it has kept;
  (2) Qwen2.5-1.5B still falls for a trick ("the chief executive's home
  address" gets his name) and garbles a long four-part answer on the NPU;
  Qwen3-8B on the NPU is the thing to measure there; (3) a passage longer
  than 512 tokens (dense text, or a language that takes more tokens a
  character) is cut short on the NPU. (filed 2026-10-09, proofing pass
  asked for by the user)
  Fixed 2026-10-09. Sorted on 2026-10-10: screen memory's index into **R06**,
  the small model's slips into **R36**.

- [ ] **Expense lines: what the small model still gets wrong.** The Auto
  Demo's stage puts each receipt beside its line, and the first watched run
  showed the customer as the vendor of every receipt. Fixed the same day
  (see the note in [docs/AUTO_DEMO.md](docs/AUTO_DEMO.md): vendor 5 to 13
  right of 14, date 8 to 14, category 7 to 12, Qwen2.5-1.5B on the NPU).
  What is left: a refund is filed "Other" and a credit note "Software"
  (2 of 14 categories); one sample's banner line ("NOT FOR PAYMENT") is
  taken for its vendor; and when a total is too faded for the vision model
  to read, the language model invents one -- flagged ("Amount could not be
  matched to the receipt text") and left out of the totals, but still shown
  as a number. Worth trying: showing no amount at all when it cannot be
  matched, and Qwen3-8B on the NPU for this step (it is already on disk for
  the Page Agent; about 19 tok/s there against 50, for some sixty tokens a
  receipt). (filed 2026-10-09, Auto Demo watched runs)
  Sorted into **R36** on 2026-10-10.

- [ ] **A model picker for each brick: nine candidate models tried, and what
  each would take.** Asked for with Qwen3.8 27B for coding, an NPU
  alternative for summarising and Gemma 4 as examples. Tried on the XPS 14,
  plugged in, B60 attached: every model was given the bricks' own prompts
  and bundled samples, speeds were taken one job at a time, one run per
  cell.
  *Large models*, review notes in tok/s, integrated GPU / B60:
  Qwen3-Coder-30B-A3B (today's) 49 / 77; Qwen3.6-35B-A3B 44 / 60; Gemma 4
  26B-A4B 37 / 68; Qwen3.8-27B 7.6 / 22 (14 / 35 with its draft head);
  Gemma 4 31B 6.4 / 19. All four candidates named the three planted faults
  in the tenant-export diff; today's coder names two, among false alarms.
  *On the NPU* only today's Qwen2.5-1.5B (58 tok/s) and Qwen3-8B int4-cw
  (18.8 tok/s, 1.2 s to the first token, 95 s first compile) run. Qwen3-4B
  int4 loads and answers garbage; Gemma 4 E2B and Qwen3.5-4B had not
  finished compiling after 40 minutes; Gemma 4 E4B fails to compile.
  *Small-model scores* (meeting action items of 7 / document facts of 17 /
  expense fields of 70): 1.5B on the NPU 0 / 15 / 47; 8B on the NPU
  4 / 15 / 62; on the integrated GPU Qwen3.5-4B 7 / 17 / 63 at 38 tok/s,
  Gemma 4 E4B 6 / 16 / 64 at 32, Gemma 4 E2B 3 / 12 / 63 at 59.
  *Reading receipts* on the integrated GPU, expected values read of 68 and
  time per image: Qwen2.5-VL-7B (today's) 68, 7-9 s; Qwen3.5-4B 68, 5 s;
  Gemma 4 E4B 67, 5 s; Gemma 4 E2B 61, 2.6 s; the 35B and Gemma 26B read
  all 68 too.
  *What adopting them takes.* Everything but Qwen3-8B needs OpenVINO 2026.4
  (the project is on 2026.3) and loads through the vision pipeline, not the
  text one `doc_qa.llm_openvino` uses. Qwen3 and later think aloud unless
  the chat history carries `enable_thinking: false`. The tokenizer in the
  Gemma 4 builds turns ten tags (`<html>`, `<body>`, `<span>`, `<code>`,
  `</div>` among them) into U+FFFD, so its HTML has to be rebuilt from
  token ids; and without sampling Gemma did not stop cleanly (the 26B
  started its page again, the 31B repeated `</html>` to the token limit).
  The 35B loads in 70-84 s against 19 s for today's coder and holds 21 GB
  against 17.
  *The HTML Creator scenarios do not transfer.* With "Same page every
  time", Neon Breakout plays only on today's model with today's OpenVINO:
  on 2026.4 today's coder draws the game and never moves the ball, the 35B
  misspells a variable, the 27B never launches the ball. An OpenVINO
  upgrade alone needs the five scenarios checked again.
  *Worth a slot, least work first:* Qwen3-8B int4-cw as a second, better
  NPU model (runs on 2026.3; needs only the thinking switch); after an
  OpenVINO upgrade, Qwen3.5-4B for screen OCR and as the small model on the
  integrated GPU, and Qwen3.6-35B-A3B, or Qwen3.8-27B on a B60, as a
  code-review choice. All nine are Apache-2.0 on their cards (R17). The
  models (91 GB) are in the Hugging Face cache; nothing in the app uses
  them. (filed 2026-10-05, user request)
  *2026-10-07:* live translation can now hand its transcript to Meeting
  Notes, which puts today's 1.5B notes model in front of more people. On a
  scripted 12-line meeting with three stated tasks (one run, AUTO device)
  its summary was sound but it wrote "Action items: None identified" and
  gave one speaker's task to someone else -- the 0 of 7 above, seen on the
  page.
  *2026-10-07, later:* Meeting Notes now writes with Qwen3-8B on the
  OpenVINO engine (standard int4 build on a GPU or the CPU, channel-wise on
  the NPU) -- it needs no OpenVINO upgrade. Four test meetings, 20 stated
  tasks: 17-18 found on the integrated GPU at 23 tok/s against 6 for the
  1.5B; 11 on the NPU at 19 tok/s. The table and what the instructions had
  to learn are in `bricks/meeting-notes/README.md`. Still open from this
  item: Qwen3.5-4B found 7 of 7 on the first of those meetings at 38 tok/s
  but needs OpenVINO 2026.4; nothing better than the channel-wise 8B runs
  on the NPU; the other bricks on the 1.5B (documents, voice assistant,
  expenses) are unchanged. `Qwen3-4B-int4-ov` in the cache is no longer a
  candidate for anything.
  *2026-10-07, evening:* asked to keep the notes on the NPU as much as
  possible. They now go there by default, on a chip of their own
  ("Notes written on"), with the transcription left on the GPU. Every
  published NPU candidate was tried and none beats the channel-wise
  Qwen3-8B (11 of 20): the int8 builds of Qwen3-4B and -8B compile and
  then never answer, Phi-3.5-mini invents owners and dates, Mistral-7B
  v0.3 finds 13 but sets deadlines nobody gave. The channel-wise 8B was
  quantised without calibration data. **Open, and the one thing that would
  make the NPU's notes good:** export Qwen3-8B channel-wise with
  calibration (AWQ, scale estimation, wikitext2), measure it on the same
  four meetings, and host it where the app can download it -- needs the
  16 GB original weights, an hour or more of CPU, and someone's Hub
  account. Five models from this search (about 20 GB) are in the Hugging
  Face cache and used by nothing: `Qwen3-4B-int8-ov`, `Qwen3-8B-int8-ov`,
  `Phi-3.5-mini-instruct-int4-cw-ov`, `Mistral-7B-Instruct-v0.3-int4-cw-ov`
  and `Qwen3-4B-int4-ov`.
  Sorted into **R37** on 2026-10-10; the calibrated export for the NPU into
  **R36**.

- [ ] **Live translation lost the NPU when the expense extractor started
  beside it.** `logs/events.log`, 2026-10-05 at 10:41-10:44: live
  translation was running on the NPU; the expense extractor started
  about two minutes later; half a minute on, an utterance failed with
  `ZE_RESULT_ERROR_DEVICE_LOST` ("device hung, reset, was removed, or
  driver update occurred"), and the reload-and-retry failed with
  `zeMemAllocHost ... ZE_RESULT_ERROR_UNKNOWN`, leaving the brick in error.
  The log does not record which chip the expense run's language step used;
  if it was the NPU, this is two models on the NPU at once. Worth
  reproducing before a scenario or an Auto Demo scene puts two NPU
  workloads side by side (R21, C1). (filed 2026-10-05, events.log)
  **2026-10-06:** it took the launcher down a second time (11:14), and both
  times the process died inside the NPU driver 20-30 s after the loss, on
  the retry. Reproduced outside the launcher and handled in v0.2.63
  (`pantherlake_ai_core.npu`): bricks take turns on the NPU, and after a
  loss nothing touches it again -- speech and language models carry on on
  the GPU. The cause is upstream (openvinotoolkit/openvino#38403, NPU
  driver 32.0.100.5540). Left open: turn-taking kept the NPU in four runs
  of the scenario that lost it in two of three without, which is an
  indication, not proof; the detector, segmenter, embedder and voice models
  stop with a clear message on a loss rather than move; retest on the next
  NPU driver.
  **2026-10-07:** by request, speech (Whisper medium) and the meeting-notes
  model (Qwen3-8B, 4.5 GB) now both default to the NPU, so two models there
  is the ordinary case. It held in every run made that evening: four sets
  of notes with an utterance every few seconds, three more with 850
  utterances back to back, and the launcher's own flow twice. None of those
  had the integrated GPU busy loading a large model, which is what was going
  on both times the NPU was lost -- that combination is still untested with
  the new defaults.
  Handled in v0.2.63. Sorted on 2026-10-10: the retest on the next driver and
  the combination not yet tried into **R28**.

- [ ] **Day-first dates on receipts are read month-first, whichever model
  structures them.** In the model trial, the two French scanned receipts
  dated `12/09/2026` came back as 2026-12-09 from every model that
  answered, small or large (expected 2026-09-12 in
  `scanned-receipts-expected.json`), and none returned the scanned hotel's
  balance of 406.00 (504.00, 374.00, 180.00 or nothing). The structuring
  prompt says nothing about date order, and a date that is valid both ways
  is not flagged for review the way an ambiguous amount is. (filed
  2026-10-05, model trial)
  Sorted into **R36** on 2026-10-10.

- [ ] **Live translation during a call: what is still not handled.** Fixed
  on 2026-10-06 after a demo given in a Teams meeting: the segmenter took
  its noise floor from the first 0.6 s, so a presenter talking while
  pressing Start was heard badly or not at all (now measured continuously);
  and a spoken language can be chosen instead of detected. Not done:
  (1) with the laptop's own speakers and microphone, the other
  participants' voices reach the microphone and are transcribed as if the
  presenter said them, in whatever language they speak -- needs the
  speaker output as a reference to gate or cancel, or a headset;
  (2) what a call does to this microphone (Cirrus Logic array: gain, noise
  suppression in communications mode) has not been measured -- record what
  the segmenter receives during a Teams test call and compare;
  (3) checked only on synthesised English speech: no recorded speech in
  another language is on this machine.
  (filed 2026-10-06, user report)
  **2026-10-07:** (3) is done with 14 French sentences from FLEURS, which is
  what showed "base" giving the gist and "medium" a translation (table in
  `bricks/live-translation/README.md`); "medium" on the NPU is now the
  default. Seen in a real French conversation transcribed with "base" that
  evening and not addressed: short noises come back as stock phrases
  ("Thank you.", "you", "Mm-hmm.") and short utterances get a wrong
  language label; a clip cut at the 14 s limit leaves a tail that becomes a
  phrase of its own ("Bye!"). Whether "medium" reduces these on real
  speech has not been measured. Also fixed that evening: after "Summarise
  in Meeting Notes" the Meeting Notes panel showed a frozen copy of the
  transcript beside its own Start, which started a second, separate
  transcription of system audio -- it now follows the live transcript and
  its own Start steps aside.
  Sorted into **R40** on 2026-10-10.

- [ ] **Page Agent (experimental brick): what is not done.** Built on
  request as a first multi-model brick: a planner on the NPU, FLUX.1-schnell
  for the pictures, the 30B coder for the page, code on the CPU conducting
  (`bricks/page-agent/README.md` has the design, the timings and what was
  learned). Open: (1) the page's length is unbounded -- 30 s to over a
  minute of writing -- and nothing lets a presenter ask for a short one;
  (2) the planner describes pictures in the request's language, where the
  image model wants English; (3) pictures are never checked against what
  was asked for, and lettering in them is gibberish; (4) on one GPU the
  extra pictures cost an unload and reload of the coder; (5) the planner
  shares the NPU with speech and meeting notes when those run -- not tried
  together; (6) three models stay loaded after a build (about 13 GB on the
  integrated GPU for FLUX alone) until the brick is closed in the hardware
  panel, which on a 32 GB machine leaves little for the other large-model
  bricks; (7) it could draw from a brand's own pictures or a logo as well
  as generate; (8) a licence check of FLUX.1-schnell's OpenVINO build
  (Apache-2.0 on the card) belongs with R17. (filed 2026-10-08, user
  request)
  *2026-10-08, evening, after "the pages are a bit bland":* the pages are
  now art-directed (the brick's README says how: a fuller plan with six
  pictures each given a place, a specification of the page in every
  request, some twenty layout rules put into every stylesheet), and the
  samples are briefs. That moves the list. (1) stands and is longer: a
  build is 82-92 s once the models are loaded, 145-162 s the first time.
  (2) is better, not closed: the one French request tried since got English
  photographs. (3) stands; what asks for lettering is now taken out of the
  plan. New: (9) every page has the same skeleton -- a choice of two or
  three would be the next step, picked by the planner; (10) a long brief is
  not always carried whole (week prices, a year per project): a check says
  which figures are missing and nothing puts them back; (11) the planner's
  photographs are plain, and a picture is still never looked at before it
  is used; (12) the planner reads a brief's wording closely: "a launch page
  for Kivu Nightfall, the new coffee of..." got "Roasted after dark | How to
  Brew | Subscription" as its three things, and "a one-page site for a
  coffee roaster" with the same three coffees listed got the coffees. The
  sample was reworded; a presenter's own brief will meet the same thing.
  Sorted into **R39** on 2026-10-10.

- [x] **The 30B coder can answer a page request with an opening code fence
  and nothing else.** Seen while building the Page Agent: for a request
  made of a description, a title, a list of sections and a PICTURES list,
  `html-creator`'s landing-page instructions got "```" and the end -- on a
  freshly loaded model, with sampling or without, twice out of two; the
  same request ending in "Start your reply with <!DOCTYPE html>." got the
  page four times out of four. HTML Creator's own requests have not been
  seen to do it, but nothing in it would notice: it would show an empty
  page. Worth a guard there (an answer with no `<html` is not a page) or
  the same closing line. (filed 2026-10-08, page-agent build)
  *Done 2026-10-08:* the closing line stopped being enough the day the
  requests grew (two in four came back as "```" at 2,300 tokens), so the
  answer is now begun for the model -- `begin` in doc-qa's `answer`, which
  applies the chat template itself and lets the runtime continue from
  `<!DOCTYPE html>`. HTML Creator passes it for every page, document
  summaries included. The portable engine takes the argument and cannot
  honour it.
  Done 2026-10-08; nothing left to sort.

- [ ] **A process can spin forever at exit after image and language models
  have shared the integrated GPU.** Found with the Page Agent's
  command-line tool on the XPS 14: after one build (image model, 30B coder,
  image model again, all on GPU.0, the bike sample) the process never
  exited, five times out of five -- one thread at 100% after Python had
  finished tearing down (`PYTHONVERBOSE` ends on "clear sys.audit hooks"),
  so inside OpenVINO's or a driver's own teardown. Releasing every model
  first changes nothing; `os._exit` hangs too; eight shorter sequences with
  the same models exit normally. Worked around, not understood: the tool,
  and the launcher once a page has been built, end with `TerminateProcess`
  (`page_agent/leaving.py`). Open: which sequence sets it off, whether
  other bricks can reach it (HTML Creator then an image model, say), and a
  report upstream once it is small enough to hand over. It matters beyond
  this brick because an in-app upgrade waits for the old process to end.
  (filed 2026-10-08, page-agent build)
  Sorted into **R47** on 2026-10-10.

- [ ] **HTML Creator writes its pages at the temperature that made the Page
  Agent's loop.** At 0.2, three of fifteen long pages from the 30B coder
  ran away into the same CSS rules repeated until the tokens ran out; at the
  model's own 0.7, none of thirteen (and one page stopped by the watch in
  the twenty-two built after that). The Page Agent now writes at 0.7 and
  has a watch that stops a looping page (`page_agent/runaway.py`); HTML
  Creator still writes at 0.2 and would sit through the whole 6,144 tokens.
  Its samples have not been seen to loop -- they were not looked at for it
  either. Worth a count over its samples, and the watch moved somewhere
  both can use it. Note that `repeatable` (most likely token every time)
  can only make a loop more certain. (filed 2026-10-08, page-agent pages)
  Sorted into **R47** on 2026-10-10.

- [ ] **A model on one GPU slows down while another works on the other.**
  Seen while the Page Agent drew its pictures on the integrated GPU and
  wrote its page on the B60: the coder wrote at 45 to 64 tokens/s instead
  of 65 to 68, FLUX took 5 to 10 s a picture instead of 4.5 to 8.5, and the
  planner on the NPU 23 to 31 s a plan instead of 14 while pictures were
  drawn. Two chips at once is the brick's selling point, and it is true --
  the build is shorter than in turn -- but "each model has its own chip"
  overstates it: they share the CPU that feeds them, and the memory.
  Not measured: where the time goes. (filed 2026-10-08, page-agent pages)
  Sorted into **R19** on 2026-10-10, where it gets measured.

- [ ] **After two hours of builds back to back the laptop holds itself to
  15 W, and every demo takes half as long again.** Seen on the XPS 14 on
  the evening of 2026-10-08, mains power, Windows on "Best performance":
  from about 21:05 the processor package drew 15 W with eight cores busy --
  what it draws at idle -- and stayed there through a few idle minutes. The
  30B coder on the B60 fell from 52-64 tokens/s to 32-38, the integrated
  GPU from 40 to 25-31, a FLUX picture from 5-10 s to 11-20 s; a warm
  two-GPU Page Agent build went from 82-92 s to 130-140 s. Nothing else was
  using the machine (sampled). Not known: what triggers it (heat, a Dell
  thermal profile, the charger), how long it lasts, whether a rehearsal
  before a show would put the machine in that state for the show. Worth an
  hour with the hardware panel's power line on screen before the next long
  demo, and a line in the presenter's notes either way. (filed 2026-10-08,
  page-agent pages)
  Sorted into **R45** on 2026-10-10.

- [x] **A setup assistant for a laptop with nothing installed.** Asked for
  on 2026-10-10: the first launch as a guided, visual page, sure to work
  from a fresh laptop. `first_launch.bat` now opens a page served by
  Windows PowerShell (`setup/`): six steps -- this laptop, the installer,
  the Studio, the chips, the models, ready -- each saying what it found,
  asking before a download and, when it fails, what to do. The console
  helper stays behind it (`first_launch.bat console`, and by itself when
  the page cannot be served). `start_launcher.bat` and the in-app upgrade
  find an installer kept in the project (`.tools\uv`), and the command
  files are committed with Windows line endings so that a ZIP from GitHub
  runs.
  *How it was checked.* No fresh laptop and no Windows Sandbox were at
  hand, so one was imitated on the XPS 14: a clean copy of the project in a
  folder 87 characters deep, an empty user profile, and a PATH with Windows
  alone on it (no uv, no Python, no git, no winget). From there
  `first_launch.bat` was walked to the end in a headless browser with real
  downloads: uv 0.13.0 fetched and its SHA-256 checked in 8 s, the
  environment installed in 80 s, the chips found (CPU, two GPUs, NPU) after
  a first check of 76 s, the Studio answering 104 s after its first start,
  the four sets sized at 2.2, 6.9, 37 and 48 GB with none of 21 models
  present, one model fetched, the Studio opened. `start_launcher.bat` and
  `stop_launcher.bat` were then run from the same bare PATH, and
  `first_launch.bat` alone in an empty folder, as Windows runs it from
  inside a ZIP. The failures were scripted and read on the page: too
  little disk, a host that does not answer, winget and the download both
  failing, a certificate refused (and the retry with Windows' own), a
  missing NPU, a package that does not load.
  *Not checked, and worth a real new laptop:* Windows' own warning on a
  downloaded `.bat`; a company laptop's rules (scripts forbidden, a proxy);
  a laptop without the Visual C++ runtime -- the assistant looks for it in
  step 1 and links to Microsoft's installer, which could only be shown with
  a scripted report here, since this laptop has it; a Lunar Lake laptop.
  Models read from a folder with accents in its name were tried and load.
  *Found on the way, and fixed:* a Studio stopped with `stop_launcher.bat`
  came back by itself when the window that had started it was still open.
  `start_launcher.bat` ran the Studio a second time "with the network"
  whenever the first run ended with an error, and a Studio that is stopped
  ends with one. Whether the network is needed is now asked before the
  Studio starts, with a command that does nothing (0.15 s); the Studio is
  then started once. Checked on the bare copy both ways: stopped, it stays
  stopped; with one package taken out of the environment and an empty
  cache, it says something is missing, fetches it and starts.
  (filed 2026-10-10, user request)
  Done 2026-10-10 (v0.2.92, v0.2.93): most of **R10**. Sorted the same day:
  what was not checked, on a laptop that is not this one, into **R33**.

- [ ] **The Page Agent's models failed to load on the GPU twice, while the
  receipts demo was at work.** From `logs/events.log`, 2026-10-09 at 13:24
  and 13:26: `[GPU] ProgramBuilder build failed!` with `clWaitForEvents,
  error code: -58 CL_INVALID_EVENT`, both times as the Page Agent loaded
  its image model on GPU.0 and its coding model on GPU.1 and the Expense
  Report Extractor was reading receipts. Two events in one afternoon of
  tests, not reproduced since and not looked into: it may be memory (a
  17 GB model arriving beside a vision model), or the B60's cable. If it
  comes back, note what else was running and how much memory was free.
  (filed 2026-10-10, events.log review)
  Sorted into **R11** on 2026-10-10.
