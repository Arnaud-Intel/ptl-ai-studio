# Engineering Planning Sync — July 22, 2026

Attendees: Priya Desai (CTO), Tomas Whitfield (CEO), Jae-won Lim
(Hardware Lead), Sofia Bianchi (Software Lead), Marcus Okafor (QA Lead)

## Summary

The team reviewed progress on the Wayfinder cold storage variant and the
Fleet OS 3.0 pilot program. Cold storage testing at the Salem test
facility has been going well, but battery performance at -20°C is still
below target. The Fleet OS 3.0 pilot at two customer sites has had one
minor incident (a mobile robot stopped short of a collision, no injuries
or damage) that needs to be root-caused before GA.

## Discussion points

- Battery vendor Ionix Cells proposed a new cell chemistry that could
  close the cold-weather performance gap, but it would need to be
  requalified, adding roughly six weeks to the schedule.
- The Fleet OS near-miss incident was traced to a mapping edge case, not
  a sensor failure -- the collision-avoidance system itself performed
  correctly.
- Marcus raised that regression test coverage for Fleet OS's pathing
  logic is thin, and asked for headcount to build it out before GA.

## Action items

- **Jae-won**: Get a sample of Ionix Cells' new cell chemistry and run a
  48-hour cold test by August 8, 2026.
- **Sofia**: Fix the mapping edge case that caused the Fleet OS near-miss
  and ship a patch to both pilot sites by July 31, 2026.
- **Marcus**: Draft a test plan for expanded Fleet OS pathing regression
  coverage and bring a headcount estimate to the August 5 planning
  meeting.
- **Tomas**: Follow up with both pilot customers about the near-miss
  incident directly, before end of week (July 26, 2026).
- **Priya**: Decide by August 1 whether the cold storage variant launch
  slips past June 30 based on Jae-won's battery test results.
