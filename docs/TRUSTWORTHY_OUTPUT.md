# Trustworthy-output implementation

Implemented 2026-09-10 against version 0.2.36. No version bump is made manually.

## Changes

- NPU naming now uses an OpenVINO-enumerated NPU and its runtime property. USB
  device substring matches are removed. Known GPU identities take precedence
  over compute-engine heuristics, and unverified NPUs get no utilization claim.
- The header says “Designed for” for sponsor/target-platform badges and “Monitored
  hardware” for telemetry. Explicit OpenVINO selections are checked against the
  runtime's device list before launching work.
- Receipt parsing preserves decimal magnitude, rejects ambiguous/non-finite values,
  and uses exact decimal totals per currency. Currency evidence, amount evidence,
  date validity and category checks produce a visible review status. Flagged lines
  are excluded from totals. A validated-field total is not a claim of accounting
  accuracy; OCR/LLM mistakes still require comparison with the receipt.
- WebSocket amount values are now decimal strings; `done.totals` replaces
  `done.total`. CSV adds currency and review columns. See the expense README.
- Webcam OpenVINO GPU/AUTO requests fail before model loading. CPU is the default;
  explicit NPU remains selectable. All segmentation backends reject malformed,
  non-finite or materially out-of-range masks before rendering.
- Tracker assignment now considers strongest IoU pairs across the frame first,
  preventing one demonstrated order-dependent extra track. The UI, CLI and API
  describe new-track counts as experimental, since track loss still causes recounts.
- Activity history recovers up to 200 recent valid events from bounded file tails.
  Logs rotate at 1 MiB with three backups (about 4 MiB total for newly written logs).
  Messages are truncated to 4,000 characters to bound single records. A legacy
  oversized log is bounded during reading and rotates on the next write. History
  loading does not mark old workers as currently running. The viewer includes dates.

## Verification and remaining gates

- Final Python regression run: **189 passed in 7.05 seconds**, Windows/Python 3.12
  with OpenVINO installed (47 more cases than the original 142-test baseline).
- JavaScript syntax check passed.
- Restarted the idle launcher and verified the live API returns `npu_name: null`
  and `npu_percent: null` on this machine, with no Logitech NPU label.
- Browser check: webcam selects OpenVINO CPU, GPU/AUTO options are disabled with
  a reason, and activity entries from before restart appear with dates.
- Hardware-dependent model inference was not run. No CPU/GPU accuracy parity or
  real-camera counting improvement is claimed by the automated checks.

Automated fixtures cover locale separators, refunds, unknown currencies, exact
mixed-currency totals, invalid dates, model-output serialization, log recovery,
malformed tails, rotation, concurrent writes, device validation, invalid masks and
the order-dependent tracker case. Existing API and worker tests remain included.

No camera/microphone capture or large-model download is needed for these checks.
Real Intel NPU naming, CPU/GPU numerical mask comparison and annotated live-camera
count accuracy still need the affected hardware and agreed fixtures. GPU inference
is gated rather than claimed repaired, and the historical ~2.3× tracking defect is
not claimed resolved. Those gates remain open in R01/R03.

**2026-09-11 follow-up:** real NPU naming is verified on the target machine
(OpenVINO and `/api/telemetry` both report `Intel(R) AI Boost`), so R01 is
closed. The R03 gates above are unchanged.

For hardware qualification, record OS, GPU/NPU and driver, runtime version, model
revision, source frames/annotations and cold/warm state. Compare mask finiteness,
range and CPU numerical similarity, and count error per clip (initial target ≤10%
for clips with at least 20 reference objects). Re-enable a device only after this
evidence passes; do not remove the gate based solely on a successful compilation.
