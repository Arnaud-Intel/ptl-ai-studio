"""One checked-in fictional sample catalog shared by CLI and launcher."""
from dataclasses import dataclass, field
from pathlib import Path
import json

SAMPLE_ROOT = Path(__file__).resolve().parents[3] / "sample-data"


@dataclass
class DemoSample:
    name: str
    description: str
    folder: str | None = None
    question: str | None = None
    prompt: str | None = None
    text: str | None = None
    mode: str | None = None
    source: str | None = None
    diff_text: str | None = None
    image: str | None = None
    assets: list[str] = field(default_factory=list)
    next_step: str = ""


def load_samples(demo_id: str) -> list[DemoSample]:
    catalog = json.loads((SAMPLE_ROOT / "catalog.json").read_text(encoding="utf-8"))
    samples = []
    for item in catalog.get(demo_id, []):
        values = dict(item)
        if values.get("diff_file"):
            path = values.pop("diff_file")
            values["diff_text"] = (SAMPLE_ROOT / path).read_text(encoding="utf-8")
            values["assets"] = [*values.get("assets", []), path]
        if values.get("folder"):
            values["folder"] = str(SAMPLE_ROOT / values["folder"])
        samples.append(DemoSample(**values))
    return samples
