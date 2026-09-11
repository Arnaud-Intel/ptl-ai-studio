"""Read-only links to checked-in demo content; never links to arbitrary user paths."""
from pathlib import Path
from urllib.parse import quote
import json

from pantherlake_ai_core.demo_samples import SAMPLE_ROOT


def asset(relative: str) -> dict:
    path = (SAMPLE_ROOT / relative).resolve()
    safe_relative = path.relative_to(SAMPLE_ROOT.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"Missing bundled demo asset: {relative}")
    return {"name": path.name, "url": "/demo-assets/" + quote(safe_relative.as_posix()),
            "image": path.suffix.lower() in {".png", ".jpg", ".jpeg"}}


def enrich_sample(sample: dict) -> dict:
    result = dict(sample)
    paths = list(result.get("assets") or [])
    if result.get("folder"):
        folder = Path(result["folder"]).resolve()
        folder.relative_to(SAMPLE_ROOT.resolve())
        paths = [p.relative_to(SAMPLE_ROOT).as_posix() for p in sorted(folder.iterdir())
                 if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".md", ".txt"}] + paths
    if result.get("image"):
        result["image_url"] = asset(result["image"])["url"]
        paths.insert(0, result["image"])
    result["assets"] = [asset(path) for path in dict.fromkeys(paths)]
    return result


def guide(demo_id: str) -> dict | None:
    guides = json.loads((SAMPLE_ROOT / "guides.json").read_text(encoding="utf-8"))
    item = guides.get(demo_id)
    return {**item, "assets": [asset(path) for path in item.get("assets", [])]} if item else None
