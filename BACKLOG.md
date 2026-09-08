# Backlog

Fixes and ideas found via manual review of `logs/events.log` (see
`CLAUDE.md`), or filed directly by the user. Not a roadmap -- a holding
pen for things worth doing that aren't being done right now.

<!-- - [ ] **Title.** Short description. (filed YYYY-MM-DD, source) -->

- [ ] **Say "Stopping…" while a slow brick finishes its in-flight call.**
  Every threaded runner's `stop()` sets the stop event and joins with a 3s
  timeout, then returns. If the worker is inside a long call when Stop is
  pressed -- e.g. smart-recall's OCR stage mid-inference on the 7B
  vision-language model, which can take a minute -- the join times out and
  the brick keeps reporting `running` with its normal message ("Recording
  and indexing...") until the call returns. It does stop, and nothing
  leaks (verified: the stop event is set, the thread exits on its own, and
  `running` follows `is_alive()`), but for that stretch the UI says the
  opposite of what the user just asked for. Either set a `"stopping"`
  phase when the join times out, or move the join into `start()` (which is
  the only thing the 3s wait actually protects -- a Start racing a
  still-winding-down thread) and let `stop()` return immediately.
  (filed 2026-09-08, found during the refactor round's regression pass)
