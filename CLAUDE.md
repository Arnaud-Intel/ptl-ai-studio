# Agent instructions

See `CONTRIBUTING.md` for architecture and conventions.

## Check the activity log periodically

The launcher records a lifecycle event (loading a model, running, or
failing) for every brick to `logs/events.log` (one JSON object per line:
`demo_id`, `phase`, `message`, `at`) and via `GET /api/logs`. When working
in this repo, skim recent entries for recurring or notable errors and, if
you find something actionable, add it to `BACKLOG.md`'s checklist (same
format as the entries already there) rather than fixing it unprompted.
