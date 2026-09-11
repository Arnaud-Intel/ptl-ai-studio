# Agent instructions

See `CONTRIBUTING.md` for architecture and conventions.

## Check the activity log periodically

The launcher records a lifecycle event (loading a model, running, or
failing) for every brick to `logs/events.log` (one JSON object per line:
`demo_id`, `phase`, `message`, `at`) and via `GET /api/logs`. When working
in this repo, skim recent entries for recurring or notable errors and, if
you find something actionable, add it to the **Inbox** at the top of
`BACKLOG.md` (the commented template there shows the format) rather than
fixing it unprompted. If a roadmap ticket already covers it, add a dated
note to that ticket instead of a duplicate. Don't file straight into the
R-tickets or change their priorities: sorting the Inbox into them happens
when the roadmap is reviewed.
