"""Named example search phrasings -- suggestions for how to ask, not
guaranteed matches, since results depend on whatever's actually been
recorded."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    name: str
    description: str
    question: str


SAMPLES: list[Sample] = [
    Sample(
        name="Recall a webpage",
        description="Search for a page you remember browsing.",
        question="that pricing page I had open earlier",
    ),
    Sample(
        name="Recall an error",
        description="Search for a terminal or console error you saw.",
        question="the error message I saw in the terminal",
    ),
    Sample(
        name="Recall a chart",
        description="Search for a chart or spreadsheet you viewed.",
        question="a chart or spreadsheet I was looking at",
    ),
    Sample(
        name="Recall an email",
        description="Search for an email you had open.",
        question="the email with the meeting link",
    ),
]
