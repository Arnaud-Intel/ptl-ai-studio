"""Named example questions. Each points at the bundled fictional-company
sample-data folder (see /sample-data/README.md at the repo root) so a
sample is a real, self-contained demo -- not just phrasing with nothing to
answer against -- while still letting a caller override the folder to ask
the same kind of question about their own documents instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# bricks/doc-qa/src/doc_qa/samples.py -> repo root is 4 levels up.
_SAMPLE_DATA_DIR = Path(__file__).resolve().parents[4] / "sample-data" / "meridian-robotics"


@dataclass
class Sample:
    name: str
    description: str
    question: str
    folder: str | None = None


_MERIDIAN = str(_SAMPLE_DATA_DIR) if _SAMPLE_DATA_DIR.is_dir() else None

SAMPLES: list[Sample] = [
    Sample(
        name="Document overview",
        description="Ask what the ingested documents are broadly about.",
        question="What are the main topics covered in these documents?",
        folder=_MERIDIAN,
    ),
    Sample(
        name="Summary",
        description="Ask for a condensed bullet summary.",
        question="Summarize the key points in a few bullet points.",
        folder=_MERIDIAN,
    ),
    Sample(
        name="Dates and deadlines",
        description="Ask for any time-sensitive information.",
        question="Are there any dates, deadlines, or timelines mentioned?",
        folder=_MERIDIAN,
    ),
    Sample(
        name="Action items",
        description="Ask for follow-up tasks and owners.",
        question="What action items or next steps are mentioned, and who owns them?",
        folder=_MERIDIAN,
    ),
]
