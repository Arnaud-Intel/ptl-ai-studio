"""What the conductor verifies about a finished page -- plain code, no model.

A model that wrote the page is a poor judge of whether it finished it. These
are the faults that are both common and mechanical to detect; two of them
(a page cut off mid-way, a planned picture left out) are worth one more try
with the fault named, a third (one picture shown several times) is mended
by drawing more pictures (see repeats.py), and the others are reported as
they are.
"""
from __future__ import annotations

import re

from .plan import PagePlan
from .types import Check

_REMOTE_RESOURCE = re.compile(
    r"""(?:\bsrc\s*=\s*["']?|url\(\s*["']?|<link\b[^>]*\bhref\s*=\s*["']?|@import\s+["']?)(?:https?:)?//""", re.IGNORECASE
)
_VIEWPORT = re.compile(r"""<meta\b[^>]*name\s*=\s*["']?viewport""", re.IGNORECASE)
_HEADING = re.compile(r"<h1\b", re.IGNORECASE)

COMPLETE = "complete"
PICTURES = "pictures placed"
ONCE = "each picture once"
SELF_CONTAINED = "self-contained"
RESPONSIVE = "fits a phone"
HEADING = "has a main heading"


def review(written: str, plan: PagePlan, used: list[str], truncated: bool) -> list[Check]:
    """`written` is the page as the model wrote it (file names, not embedded
    pictures); `used` the planned pictures it placed."""
    missing = [picture.name for picture in plan.pictures if picture.name not in used]
    # Three cards wearing the same photograph is the first thing a reader
    # sees: a page with more items than pictures tends to fill them that way.
    repeated = [f"{picture.name} x{written.count(picture.name)}" for picture in plan.pictures if written.count(picture.name) > 1]
    remote = len(_REMOTE_RESOURCE.findall(written))
    return [
        Check(COMPLETE, not truncated, "" if not truncated else "the page stops before </html>"),
        Check(
            PICTURES,
            not missing,
            f"all {len(plan.pictures)}" if not missing else "not placed: " + ", ".join(missing),
        ),
        Check(ONCE, not repeated, "" if not repeated else "used more than once: " + ", ".join(repeated)),
        Check(
            SELF_CONTAINED,
            remote == 0,
            "" if remote == 0 else f"{remote} resource(s) would be fetched from the network (a font, a script, an image)",
        ),
        Check(RESPONSIVE, bool(_VIEWPORT.search(written)), "" if _VIEWPORT.search(written) else "no viewport meta tag"),
        Check(HEADING, bool(_HEADING.search(written)), "" if _HEADING.search(written) else "no <h1>"),
    ]


def repair_note(checks: list[Check]) -> str | None:
    """What to add to the request for a second try, or None when nothing a
    second try could fix is wrong."""
    failed = {check.name: check for check in checks if not check.passed}
    notes = []
    if COMPLETE in failed:
        notes.append(
            "A first version of this page was cut off before the end. Write a more compact page: fewer sections, "
            "shorter copy, less CSS -- and finish it with </html>."
        )
    if PICTURES in failed:
        notes.append(
            "A first version of this page left pictures out. Every picture of the PICTURES list must appear in the "
            "page, each exactly once, by its exact file name."
        )
    return " ".join(notes) or None
