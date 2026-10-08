"""What the conductor verifies about a finished page -- plain code, no model.

A model that wrote the page is a poor judge of whether it finished it. These
are the faults that are both common and mechanical to detect; two of them
(a page that stops mid-way, a page that left most of its pictures out) are
worth one more try with the fault named, a third (one picture shown several
times) is mended by drawing more pictures (see repeats.py), and the others
are reported as they are.

A second try is a minute and a half of writing, so it is kept for a page
that cannot be shown. A page with five of its six pictures is a good page
with a picture to spare, and is shown.
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
# A figure of two characters or more: 39, 1,400, 6.80, 21:00. A lone digit is found in any page.
_FIGURE = re.compile(r"\d[\d,.:]*\d")
_TAG = re.compile(r"<[^>]+>")

COMPLETE = "complete"
PICTURES = "pictures placed"
ONCE = "each picture once"
SELF_CONTAINED = "self-contained"
RESPONSIVE = "fits a phone"
HEADING = "has a main heading"
FIGURES = "keeps the request's figures"


def _uses(written: str, name: str) -> int:
    """How many places show the file: each <img>, and one for all the CSS
    that puts it behind something (a rule written twice is one place)."""
    file = re.escape(name)
    images = len(re.findall(r"<img\b[^>]*" + file, written, re.IGNORECASE))
    return images + (1 if re.search(r"url\([^)]*" + file, written, re.IGNORECASE) else 0)


def _figures_kept(request: str, written: str) -> tuple[list[str], list[str]]:
    """The figures the request gives, and those of them the page does not
    carry. A page may write 1,400 as 1400 or 1 400; it may also turn 19:00
    into 7 PM, which this counts as missing -- hence a check that reports
    more than it judges."""
    asked = list(dict.fromkeys(_FIGURE.findall(request)))
    text = _TAG.sub(" ", written)
    plain = text.replace(",", "").replace("\u202f", "").replace("\u00a0", "")
    missing = [figure for figure in asked if figure not in text and figure.replace(",", "") not in plain]
    return asked, missing


def review(written: str, plan: PagePlan, used: list[str], truncated: bool) -> list[Check]:
    """`written` is the page as the model wrote it (file names, not embedded
    pictures); `used` the planned pictures it placed."""
    missing = [picture.name for picture in plan.pictures if picture.name not in used]
    placed = len(plan.pictures) - len(missing)
    # The first picture is the one the page opens on. Of six, one may be left
    # over; of three, none.
    spare = 1 if len(plan.pictures) >= 4 else 0
    opens_on_it = not plan.pictures or plan.pictures[0].name not in missing
    # Three cards wearing the same photograph is the first thing a reader
    # sees: a page with more items than pictures tends to fill them that way.
    repeated = [f"{picture.name} x{_uses(written, picture.name)}" for picture in plan.pictures if _uses(written, picture.name) > 1]
    asked, lost = _figures_kept(plan.request, written)
    remote = len(_REMOTE_RESOURCE.findall(written))
    return [
        Check(COMPLETE, not truncated, "" if not truncated else "the page stops before </html>", retry=truncated),
        Check(
            PICTURES,
            opens_on_it and len(missing) <= spare,
            f"all {len(plan.pictures)}" if not missing else f"{placed} of {len(plan.pictures)}, not placed: " + ", ".join(missing),
            retry=not opens_on_it or placed * 2 < len(plan.pictures),
        ),
        Check(ONCE, not repeated, "" if not repeated else "used more than once: " + ", ".join(repeated)),
        Check(
            SELF_CONTAINED,
            remote == 0,
            "" if remote == 0 else f"{remote} resource(s) would be fetched from the network (a font, a script, an image)",
        ),
        Check(RESPONSIVE, bool(_VIEWPORT.search(written)), "" if _VIEWPORT.search(written) else "no viewport meta tag"),
        Check(HEADING, bool(_HEADING.search(written)), "" if _HEADING.search(written) else "no <h1>"),
        Check(
            FIGURES,
            len(lost) * 3 <= len(asked),  # two thirds of them at least
            "the request gave none" if not asked
            else f"all {len(asked)}" if not lost
            else f"{len(asked) - len(lost)} of {len(asked)}, not found: " + ", ".join(lost),
        ),
    ]


def repair_note(checks: list[Check], ran_away: bool = False) -> str | None:
    """What to add to the request for a second try, or None when nothing is
    wrong that a second try could fix and is worth one. `ran_away`: the
    first page was stopped because it had started repeating itself (see
    runaway.py), which is another fault than running out of room."""
    again = {check.name for check in checks if check.retry}
    notes = []
    if COMPLETE in again and ran_away:
        notes.append(
            "A first version of this page went round in circles: the same CSS rules written again and again with "
            "slightly different selectors. Write every rule once, style the parts of the page with a few shared "
            "classes rather than a rule per place, and go on to the <body>."
        )
    elif COMPLETE in again:
        notes.append(
            "A first version of this page was cut off before the end. Write a more compact page: fewer sections, "
            "shorter copy, less CSS -- and finish it with </html>."
        )
    if PICTURES in again:
        notes.append(
            "A first version of this page left pictures out. Every picture of the PICTURES list must appear in the "
            "page, each exactly once, by its exact file name."
        )
    return " ".join(notes) or None
