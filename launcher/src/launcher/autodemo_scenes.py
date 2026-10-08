"""The Auto Demo's playlist: its scenes, each built for the stand it plays on.

A scene is what the director does (which routes, with what, waiting for
what) and what the screen says meanwhile: what is happening, which chip
does which part on which engine and why that one, and what to look at in
the result. The second half is the reason the mode exists -- a stand is
watched by people nobody is there to talk to.

What goes in was decided with the user on 2026-10-08:

- The discrete GPU is not always there. Every scene says what it does with
  one and without, and the person starting the loop can leave it out.
- No sound: a stand at a large event is too loud for it.
- Smart City plays only when the internet is reachable.
- "Seeing and answering" (object detection with document Q&A) is written
  and held back until those two bricks have had a proofing pass: it does
  not play until its entry is taken out of `HELD_BACK`.
"""
from __future__ import annotations

from .autodemo import Ask, Builder, Chip, Scene, Skip, Stand, Start, Until, Wait

# Scenes that are written but not to be played yet, and why.
HELD_BACK: dict[str, str] = {
    "seeing-and-answering": "held back until Object Detection and Document Q&A have had their proofing pass",
}

_IGPU, _DGPU, _NPU, _CPU = "Integrated GPU", "Arc Pro B60", "NPU", "CPU"


def _sample(stand: Stand, demo: str, turn: int = 0, where=lambda sample: True) -> dict | None:
    """One of a demo's bundled samples: the `turn`-th of those `where` keeps,
    going round."""
    kept = [sample for sample in stand.samples(demo) if where(sample)]
    return kept[turn % len(kept)] if kept else None


def page_agent(stand: Stand, loop: int) -> Scene | Skip:
    title = "Three models, three chips, one web page"
    if not stand.igpu:
        return Skip(title, "needs a GPU for the image model and the coding model")
    sample = _sample(stand, "page-agent", loop - 1)
    if sample is None:
        return Skip(title, "its sample requests could not be read")
    two = stand.dgpu is not None
    planner = Chip(
        _NPU if stand.npu else _IGPU,
        "The planner: Qwen3-8B on OpenVINO. It names the page, says what it presents and briefs six photographs.",
        "A short answer from a small model is exactly the NPU's job: it costs a few watts, and both GPUs stay free "
        "for the heavy work." if stand.npu else "This machine has no NPU, so the small model shares the GPU.",
        "page-agent", "plan",
    )
    conductor = Chip(
        _CPU, "The conductor: ordinary code, no model.",
        "It decides who works where, gives the page its layout rules, checks it and mends what it can. "
        "Deciding is cheap; it does not need a chip of its own.",
    )
    if two:
        chips = (
            planner,
            Chip(_IGPU, "The image model: FLUX.1-schnell on OpenVINO, six photographs in four steps each.",
                 "Image generation needs a GPU and about 13 GB of memory, which the integrated GPU takes from the "
                 "laptop's own -- no graphics card required.", "page-agent", "images"),
            Chip(_DGPU, "The coding model: Qwen3-Coder 30B on OpenVINO, writing the whole page.",
                 "The largest model goes to the largest GPU, so the pictures are drawn while the page is written "
                 "instead of before it.", "page-agent", "page"),
            conductor,
        )
        how = "at the same time: the page is written from the photographs' descriptions while they are being taken"
    else:
        chips = (
            planner,
            Chip(_IGPU, "The image model, FLUX.1-schnell, then the coding model, Qwen3-Coder 30B -- both on OpenVINO, in turn.",
                 "One GPU does both, with no graphics card: each model is unloaded before the other is loaded, "
                 "because a 30B model and an image model do not fit in memory together.", "page-agent", "images"),
            conductor,
        )
        how = "one after the other on the integrated GPU: the photographs first, then the page"
    devices = {} if two else {"image_device": stand.igpu, "page_device": stand.igpu}
    return Scene(
        id="page-agent",
        title=title,
        demo="page-agent",
        happening=(
            f"A request for a web page goes in -- this time: {sample['name']}. A small model plans the page, an image "
            f"model takes its photographs and a large coding model writes the HTML, {how}. Everything runs on this "
            "laptop; nothing is sent anywhere."
        ),
        chips=chips,
        look_at=(
            "The plan comes first: a name, a headline, three things on offer, and six photographs with the place "
            "each one is meant for.",
            "Each step shows its own speed: tokens per second for the two language models, images per minute for "
            "the image model.",
            "The finished page is one self-contained file. Every photograph in it was generated here, a moment ago.",
        ),
        steps=(Ask("/api/page-agent/build", {"request": sample["prompt"], **devices},
                   cancel="/api/bricks/page-agent/stop?stage=page"),),
        hold=35.0,
        at_most=480.0 if two else 780.0,
    )


def expense_extraction(stand: Stand, loop: int) -> Scene | Skip:
    title = "Two chips share one job"
    if not stand.igpu:
        return Skip(title, "needs a GPU to read the receipts")
    sample = _sample(stand, "expense-extract", 0, lambda s: s.get("name", "").startswith("Start here"))
    if sample is None or not sample.get("folder"):
        return Skip(title, "its sample receipts could not be found")
    structuring = "NPU" if stand.npu else stand.igpu
    return Scene(
        id="expense-extraction",
        title=title,
        demo="expense-extract",
        happening=(
            "A folder of receipts -- photographs and scans -- becomes an expense report. One model reads each "
            "receipt, a second turns what was read into a dated, categorised expense line. They work at the same "
            "time, each on its own chip, passing receipts from one to the other."
        ),
        chips=(
            Chip(_IGPU, "Reading: Qwen2.5-VL 7B, a vision-language model, on OpenVINO.",
                 "Reading a photograph is the heavy half of the job. A vision model of this size needs a GPU, and "
                 "the integrated one carries it.", "expense-extract", "ocr"),
            Chip(_NPU if stand.npu else _IGPU, "Structuring: Qwen2.5 1.5B, a small language model, on OpenVINO.",
                 "Turning text into fields is light, steady work. On the NPU it takes a few watts and nothing from "
                 "the GPU that is busy reading." if stand.npu else "This machine has no NPU, so both models share the GPU.",
                 "expense-extract", "llm"),
        ),
        look_at=(
            "Each receipt shows up as it is read, then its line fills in: vendor, date, amount, currency, category.",
            "Two figures, one per chip, in receipts per minute: the reading sets the pace, the structuring keeps up.",
            "A line the models are not sure of is flagged for review instead of going into the total.",
        ),
        steps=(
            Start("/api/expense-extract/report/close"),
            Start("/api/expense-extract/start", {
                "folder": sample["folder"], "ocr_engine": "openvino", "ocr_compute_device": stand.igpu,
                "llm_engine": "openvino", "llm_compute_device": structuring,
            }),
            Wait(3.0),
            Until("/api/expense-extract/report", "running", False, timeout=300.0),
        ),
        stop=("/api/expense-extract/stop", "/api/expense-extract/report/close"),
        hold=25.0,
        at_most=360.0,
    )


def smart_city(stand: Stand, loop: int) -> Scene | Skip:
    title = "Street cameras, one per chip"
    if not stand.internet:
        return Skip(title, "needs the internet for its live street cameras")
    if not stand.igpu:
        return Skip(title, "needs a GPU")
    # The traffic-camera clips: plain video files, where the live streams
    # need a video site to let the laptop in.
    clips = [s for s in stand.samples("smart-city-monitor")
             if s.get("group") != "YouTube" and "|" not in str(s.get("feeds", "")) and "\n" not in str(s.get("feeds", ""))]
    if len(clips) < 2:
        return Skip(title, "fewer than two street-camera sources to choose from")
    first, second = clips[(loop - 1) % len(clips)], clips[loop % len(clips)]
    other = "NPU" if stand.npu else "CPU"
    return Scene(
        id="smart-city",
        title=title,
        demo="smart-city-monitor",
        happening=(
            f"Two traffic cameras -- {first['name']} and {second['name']} -- are watched at once. A detector finds "
            "every person and vehicle in every frame and counts them as they pass, one camera on each chip."
        ),
        chips=(
            Chip(_IGPU, "Camera 1: YOLO11s, an object detector, on OpenVINO.",
                 "Video is many small jobs a second. The integrated GPU does them fastest, without the wait of "
                 "sending each frame to a separate card.", "smart-city-monitor", "feed-1"),
            Chip(_NPU if stand.npu else _CPU, "Camera 2: the same detector, on OpenVINO.",
                 "The NPU is built for exactly this: one small model running all day at a few watts. A second "
                 "camera costs the first one nothing." if stand.npu else
                 "This machine has no NPU, so the second camera runs on the processor.", "smart-city-monitor", "feed-2"),
        ),
        look_at=(
            "Every box is one detection: people, cars, buses, bicycles, each followed from frame to frame.",
            "The counts go up as things cross the picture; nothing is counted twice.",
            "Two frame rates, one per chip. Neither drops because the other is working.",
        ),
        steps=(
            Start("/api/smart-city-monitor/start", {
                "engine": "openvino", "loop": True,
                "feeds": [{"path": first["feeds"], "compute_device": stand.igpu},
                          {"path": second["feeds"], "compute_device": other}],
            }),
            Wait(75.0),
        ),
        stop=("/api/smart-city-monitor/stop",),
        hold=0.0,
        at_most=150.0,
    )


def seeing_and_answering(stand: Stand, loop: int) -> Scene | Skip:
    title = "Seeing and answering at once"
    if "seeing-and-answering" in HELD_BACK:
        return Skip(title, HELD_BACK["seeing-and-answering"])
    if not (stand.igpu and stand.npu):
        return Skip(title, "needs a GPU and an NPU")
    sample = _sample(stand, "doc-qa", loop - 1, lambda s: s.get("folder") and s.get("question"))
    if sample is None:
        return Skip(title, "its sample documents could not be found")
    camera = bool(stand.cameras)
    source = {"source": "camera", "camera_index": stand.cameras[0]} if camera else {"source": "screen"}
    return Scene(
        id="seeing-and-answering",
        title=title,
        demo="doc-qa",
        happening=(
            ("A camera on the stand is watched by an object detector" if camera else "The screen is watched by an object detector")
            + " while, on another chip, a language model reads a folder of business documents and answers a question "
            f"about them: {sample['name']}. Two unrelated jobs, at the same time."
        ),
        chips=(
            Chip(_IGPU, "Detection: YOLO11s on OpenVINO, every frame.",
                 "Video is many small jobs a second, and the integrated GPU does them fastest."
                 + (" Nothing the camera sees is recorded." if camera else ""), "object-detection"),
            Chip(_NPU, "Answering: Qwen2.5 1.5B on OpenVINO, with the passages it found in the documents.",
                 "The NPU answers at the same speed whether the GPU is busy or not: it was measured at 54 tokens a "
                 "second both ways.", "doc-qa"),
        ),
        look_at=(
            "The answer is written from the documents, and says which file each part came from.",
            "Its tokens per second, with the detector's frames per second beside it: neither slows the other.",
        ),
        steps=(
            Start("/api/object-detection/start", {**source, "engine": "openvino", "compute_device": stand.igpu}),
            Ask("/api/doc-qa/ingest", {"folder": sample["folder"], "engine": "openvino", "compute_device": "NPU"}),
            Ask("/api/doc-qa/ask", {"question": sample["question"]}, cancel="/api/bricks/doc-qa/stop"),
        ),
        stop=("/api/object-detection/stop",),
        hold=30.0,
        at_most=240.0,
    )


# In the order they play. Heavy and light alternate: the 30B model lost a
# fifth of its speed after many runs in a row (docs/AUTO_DEMO.md).
PLAYLIST: list[Builder] = [page_agent, expense_extraction, smart_city, seeing_and_answering]
