"""Named example questions -- phrasing suggestions, not guaranteed answers,
since the actual answer depends on whatever the user has ingested."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    name: str
    description: str
    question: str


SAMPLES: list[Sample] = [
    Sample(
        name="Document overview",
        description="Ask what the ingested documents are broadly about.",
        question="What are the main topics covered in these documents?",
    ),
    Sample(
        name="Summary",
        description="Ask for a condensed bullet summary.",
        question="Summarize the key points in a few bullet points.",
    ),
    Sample(
        name="Dates and deadlines",
        description="Ask for any time-sensitive information.",
        question="Are there any dates, deadlines, or timelines mentioned?",
    ),
    Sample(
        name="Action items",
        description="Ask for follow-up tasks and owners.",
        question="What action items or next steps are mentioned, and who owns them?",
    ),
]
