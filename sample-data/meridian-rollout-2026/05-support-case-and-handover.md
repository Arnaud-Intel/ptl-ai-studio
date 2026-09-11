# Support case DEMO-SUP-184 | Dock C controlled stop

FICTIONAL DEMO DATA. Opened September 4; updated September 8, 2026.
Owner: Sofia Bianchi. QA: Marcus Okafor. Customer contact: Elise Laurent.

Status: fix in review; not yet accepted for release. A stale pallet-buffer polygon
allowed the planner to propose a route too close to Dock C. The independent
collision-avoidance layer stopped motion. No injuries, collision or damage occurred.
Do not describe this as a disabled safety system or a resolved production incident.

Fix: reject inconsistent map revisions at import, preserve the last validated
map, and surface a specific operator message. Due September 12. Verification:
Marcus replays 200 scenarios including stale polygons, partial imports and
restart recovery by September 16. A unit-test pass alone does not close the case.

Operator handover: confirm the map revision, keep Dock C clear, watch for the
controlled-stop alert, call the shift lead, and log the mission ID. Do not
restart motion until the operator checklist permits it. Alex publishes the
training checklist September 15. Escalation email: support@meridian.example.
