"""Named example inputs. The one sample points at the bundled folder of
synthetic receipts (see /sample-data/README.md at the repo root) -- three
rendered, fictional receipts, so the demo has something real to structure
without anyone hunting for photos of their own first.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# bricks/expense-extract/src/expense_extract/samples.py -> repo root is 4 levels up.
_SAMPLE_DATA_DIR = Path(__file__).resolve().parents[4] / "sample-data" / "receipts"


@dataclass
class Sample:
    name: str
    description: str
    folder: str


SAMPLES: list[Sample] = (
    [
        Sample(
            name="Sample receipts",
            description="Three synthetic receipts -- a cafe, a taxi ride, an office-supply store. Fictional vendors, nothing real.",
            folder=str(_SAMPLE_DATA_DIR),
        )
    ]
    if _SAMPLE_DATA_DIR.is_dir()
    else []
)
