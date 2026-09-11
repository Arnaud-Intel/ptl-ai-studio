# Project review — Panther Lake AI Studio

Reviewed 2026-09-10 · version 0.2.36 · commit `db0a7f7`

## Assessment

This is a compelling local AI demonstration suite with a sound reusable foundation. Its distinguishing feature is showing several real workloads on different devices through one interface. The next investment should make those demonstrations dependable and their outputs trustworthy before expanding the catalog.

The project is suitable for supervised demonstrations today. This review does **not** establish readiness for unattended capture, business-critical extraction, or network deployment. Those uses expose gaps in persistence, validation, resource limits, and access boundaries that matter less during a short single-operator demonstration.

The [implementation backlog](../BACKLOG.md#implementation-roadmap--2026-09-10) sequences a proposed 12-week improvement cycle. No application behavior was changed during this review.

## Evidence and limits

- Inspected shared engines, telemetry, model resolution, the launcher API and workers, frontend controls, document indexing, receipt extraction, screen memory, vision tracking, existing backlog, and CI workflows. This is a targeted architecture and reliability review, not an exhaustive audit of every model backend or vendored file.
- The installation earlier in this task passed **142 tests in 13.96 seconds**, on this same application commit, with Windows, Python 3.12, and OpenVINO installed. No application code changed afterward; that suite was not rerun for this documentation change.
- The live homepage and read-only APIs responded. The browser and `/api/telemetry` both showed `Logitech USB Input Device` as the NPU name. The application's regex also directly matched that string.
- `/api/logs` returned an empty list in this new installation. There is no new history of model failures here. Prior hardware measurements in the original backlog are historical reports, not measurements reproduced in this review.
- No large models were downloaded or inferred for this analysis. Accuracy, accelerator support, long-session memory growth, and performance targets still need dedicated validation. Estimates below are planning judgments.

## What is good

| Strength | Evidence | Why it matters |
| --- | --- | --- |
| Reuse follows real responsibilities | `core/` owns capture, device policy, telemetry, and model resolution; meeting notes and the voice assistant compose existing bricks | Fixes can improve several demos together, and new features need less model-specific code |
| CLI and UI share pipelines | Brick `pipeline.py` / `session.py` modules are called by both entry points | Prevents separate implementations drifting apart |
| Practical transport choices | `launcher/app.py` uses request/response for discrete work, queues for events, and a latest-frame buffer for video | Keeps the basic design understandable and avoids accumulating obsolete video frames |
| Useful lifecycle semantics | `worker.py` refuses duplicate starts and distinguishes cooperative stopping; `events.py` records failures | Operators get more information than a spinner or a silent failure |
| Intentional concurrency | Expense extraction and screen memory use bounded inter-stage queues | There is real overlap between workloads, with some protection against producer/consumer imbalance |
| Low frontend setup cost | Plain HTML/CSS/JS, reusable `Panel` / `StreamPanel`, sample prompts and inputs | Easy to start, inspect, and demonstrate; a framework migration is not necessary |
| Good baseline regression coverage | Deterministic helpers, API contracts, worker behavior, and telemetry tests; automated Linux CI | Provides a useful safety net for incremental improvements |
| Several thoughtful UI details | Categories, running-workload strip, status live regions, skip link, sandboxed generated-HTML frame, restart/version indication | These are valuable foundations worth preserving |
| Honest engineering history | Existing backlog records numerical GPU problems and tracker limitations with measurements | Known failures can become reproducible regression cases rather than repeatedly rediscovered problems |

## What needs improvement

### 1. Correctness and hardware credibility come first

**Observed: the NPU label is wrong on this machine.** In `core/src/pantherlake_ai_core/telemetry.py`, `_NPU_NAME_SCRIPT` uses the unanchored, case-insensitive pattern `NPU|AI Boost`. The letters “npu” occur inside “Input”, so a Logitech input device qualifies. Separately, `static/index.html` hardcodes a “Running on” banner with Intel Core Ultra 3 / Panther Lake NPU badges. Branding and measured machine capabilities should be distinct. A displayed GPU also needs a clear distinction between a device visible to telemetry and one supported by the selected inference engine. See **R01**.

**Confirmed by code: receipt amounts can be silently wrong.** `coerce_amount()` removes commas, so string `12,50` becomes `1250`. `ExpenseLine` has no currency field, the frontend prefixes amounts with `$`, and the runner sums all successful amounts without currency grouping. The prompt explicitly encourages a best guess even for poor receipt text. Add explicit currency, unambiguous parsing, validation, and review status; unknown values should remain unknown. See **R02**.

**Reported, not reproduced here: tracking overcounts and GPU segmentation produces NaNs.** The original backlog documents both. The current tracker still uses greedy IoU matching with a one-second expiry, and the segmentation path has no explicit finite-output check. Preserve the original reports, first create fixtures and a supported-device matrix, then choose a fix or disable the affected configuration with an explanation. Do not assume forcing FP32 or changing an IoU threshold is sufficient. See **R03**.

### 2. Persistence is too optimistic

**Confirmed by code: document cache reuse does not notice changed files.** `doc_qa/pipeline.py` returns a cached index using a key made from folder, engine, and chunk settings. It does not fingerprint document contents, and the key does not contain the actual embedding model revision. A changed or deleted document can therefore remain in answers until a forced reindex. See **R06**.

**Confirmed by code: index saves are not atomic.** `doc_qa/store.py` writes vectors and chunk metadata separately; a crash or concurrent read can encounter mismatched generations. Screen memory rewrites the entire index after each capture and appends vectors by copying the array. That design is simple for small collections but gets more expensive as history grows. Fix consistency before considering a database replacement. See **R07**, then **R12**.

**Confirmed by code: activity history is lost from the UI on restart.** The file is appended, but `recent_events()` only returns a process-local deque. This matches the original backlog. Add bounded recovery from disk and rotation. See **R04**.

### 3. Long-running work needs an explicit ownership and delivery model

**Confirmed by code: frontend event queues are unbounded.** `lifespan()` creates plain `asyncio.Queue()` instances, unlike the bounded queues inside the two-stage pipelines. Workers continue producing while no tab consumes messages. The shared queue is also consumed competitively by multiple WebSocket clients, so tabs can split events rather than both receive them. The code documents a one-tab assumption, but the interface does not enforce it. See **R05**.

**Confirmed by code: application shutdown only stops the telemetry poller.** There is no central runner teardown in `lifespan()`. Cooperative stop helpers are a good start, but shutdown, producer failures, replay, and held model resources need explicit policies and tests. Process isolation should be considered only where native calls cannot be cancelled reliably; it is not a prerequisite for every demo. See **R05**, **R11**.

### 4. Local operation still needs data controls and bounded inputs

Screen memory persists screenshots and extracted text in the user cache. There is an explicit recording button and reset action, which is good, but no automatic retention or quota in the reviewed pipeline. Add a visible storage policy, pause, selective deletion, and deletion consistency between screenshots and the index. See **R08**.

The API defaults to loopback, which is appropriate. However, it exposes local-file and capture operations without application authentication or explicit Host/Origin checks, and `--host` can change the bind address. Treat this as a boundary to harden, **not a demonstrated remote exploit**. Keep local-only operation the supported default; a shared network product would need separate design. See **R09**.

Request models leave values such as `top_k` and capture interval unconstrained. Some image/audio uploads read their entire bodies; video uploads are copied in chunks but lack an application quota and cleanup lifecycle. The generated HTML iframe is sandboxed, but its preview policy does not explicitly restrict external resources. Validate ranges and sizes on the server and define the preview's network policy. See **R09**.

### 5. First-run confidence is weaker than the feature catalog

The current install emitted a missing-FFmpeg warning. Model resolution announces downloads but does not expose a model inventory, pinned revisions, disk requirements, or an explicit offline mode; download helpers call Hub resolution without a revision or `local_files_only` option. This does not prove that user content is uploaded, but “local inference” and “no network activity after setup” are different promises. See **R10**.

The homepage is easy to browse, but engine/device choices require background knowledge. Add one recommended ready-to-run path per demo, hide advanced controls initially, show why a configuration is unavailable, and offer actionable recovery. Preserve sample inputs and accessibility foundations. See **R13**.

### 6. Maintenance and release checks should match the target platform

`app.py` is 1,224 lines and `app.js` is 2,282 lines. File length is not itself a defect, but adding a demo currently touches large shared route and panel files. Extract per-demo modules incrementally behind the existing abstractions. Avoid mixing a broad rewrite into correctness fixes. See **R14**.

The test workflow runs Linux/Python 3.11 without OpenVINO, while the product targets Windows and accelerators. Browser interactions, offline operation, restart recovery, model quality, and real hardware behavior require additional layers of testing. The version-bump workflow also tags on pushes independently of test success; branch protection settings were not inspected. Couple releases to a verified commit and serialize version publication. See **R15–R16**.

## Product direction

Keep the project focused on a dependable local AI studio. Prioritize four representative end-to-end demonstrations for a pilot: object detection, document Q&A, speech translation, and expense extraction. Together they exercise video, retrieval, audio, and two-device processing. Screen memory should join the pilot after retention and recovery are implemented.

Defer the two “Coming soon” demos, remote multi-user access, automatic routing optimization, and a frontend-framework rewrite until the reliability gates pass. More feature cards will not resolve incorrect output or a failed first run.

## How to measure improvement

The backlog's acceptance checks are proposed release targets, not current capabilities. Record OS, driver, runtime, model revision, device, input fixture, and cold/warm state for every hardware result. Evaluate output quality alongside speed. A faster pipeline that overcounts people or corrupts a segmentation mask is a regression.

Use the end of each two-week phase to review evidence and adjust scope. If an accelerator issue remains unresolved, ship a tested fallback or mark that exact configuration unsupported rather than stretching the schedule around an unverified fix.
