"""Static registry describing every demo the launcher can show -- whether
or not it's implemented yet. Planned entries exist so the showcase reads as
a suite from day one; flip `status` to "available" and wire up the launcher
UI/API for a demo once its brick actually exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Demo:
    id: str
    name: str
    category: str
    tagline: str
    description: str
    engines: list[str] = field(default_factory=list)
    status: str = "planned"  # "available" | "planned"


REGISTRY: list[Demo] = [
    Demo(
        id="live-translation",
        name="Live Speech Translation",
        category="Speech",
        tagline="Any spoken language, live, straight to English text.",
        description=(
            "Capture the microphone or the device's own audio output (e.g. a "
            "playing video) and translate speech to English in real time, "
            "fully on-device."
        ),
        engines=["portable", "openvino"],
        status="available",
    ),
    Demo(
        id="voice-assistant",
        name="Local Voice Assistant",
        category="Speech",
        tagline="Wake word, local LLM, local text-to-speech -- no cloud round trip.",
        description=(
            "A hands-free assistant that listens, reasons with a small local "
            "LLM, and talks back, entirely on-device."
        ),
        engines=["portable", "openvino"],
        status="available",
    ),
    Demo(
        id="meeting-notes",
        name="Live Meeting Notes",
        category="Speech",
        tagline="Transcribe and summarize a call as it happens.",
        description=(
            "Live transcription (via the live-translation brick) feeding a "
            "local LLM (via the doc-qa brick) that generates a running "
            "summary and action items on demand -- nothing here is a new "
            "model, it's two bricks composed together."
        ),
        engines=["portable", "openvino"],
        status="available",
    ),
    Demo(
        id="voice-clone-studio",
        name="Voice Clone Studio",
        category="Speech",
        tagline="Record a short sample, then generate speech in your own voice.",
        description=(
            "Enrolls a short voice sample into a local tone-cloning model, "
            "then synthesizes any typed text in that voice -- zero-shot "
            "cloning, not a training run: no gradient descent and no wait, "
            "just inference on the reference clip you just recorded."
        ),
        engines=["portable", "openvino"],
        status="available",
    ),
    Demo(
        id="webcam-effects",
        name="Webcam Background Effects",
        category="Vision",
        tagline="Real-time background blur/replacement, no cloud video upload.",
        description=(
            "Person segmentation on the webcam feed at interactive frame "
            "rates -- the identical model runs on either engine, so "
            "switching is purely a CPU-vs-NPU/iGPU comparison."
        ),
        engines=["portable", "openvino"],
        status="available",
    ),
    Demo(
        id="object-detection",
        name="Object Detection Overlay",
        category="Vision",
        tagline="Live bounding boxes from a webcam or screen capture.",
        description=(
            "Runs a local object-detection model over a live webcam or screen "
            "feed and overlays labeled, confidence-scored boxes in real time."
        ),
        engines=["portable", "openvino"],
        status="available",
    ),
    Demo(
        id="screen-ocr",
        name="Screen / Image Text Extraction",
        category="Vision",
        tagline="Pull text out of a screenshot or photo, locally.",
        description=(
            "Capture the screen or a webcam frame, or upload a photo, and "
            "read the text in it -- optionally translated to English on the "
            "OpenVINO engine's vision-language model."
        ),
        engines=["portable", "openvino"],
        status="available",
    ),
    Demo(
        id="doc-qa",
        name="Local Document Q&A",
        category="Text",
        tagline="Chat with your own files -- nothing leaves the device.",
        description=(
            "Point it at a folder of .txt/.md/.pdf files; it chunks and embeds "
            "them into a local index, then answers questions using only the "
            "retrieved excerpts, with a small local LLM."
        ),
        engines=["portable", "openvino"],
        status="available",
    ),
    Demo(
        id="expense-extract",
        name="Expense Report Extractor",
        category="Productivity",
        tagline="Photograph a receipt, get a structured expense line -- no retyping.",
        description=(
            "Runs OCR (the screen-ocr brick) and a local LLM (the doc-qa "
            "brick) over a folder of receipt photos, pulling out vendor, "
            "date, amount, and category into a CSV -- the two stages run "
            "concurrently on two independently chosen devices (e.g. NPU "
            "for OCR, GPU for structuring), not one after the other."
        ),
        engines=["portable", "openvino"],
        status="available",
    ),
    Demo(
        id="smart-inbox",
        name="Inbox Triage & Draft Assistant",
        category="Productivity",
        tagline="Summarize a folder of emails and draft replies, locally.",
        description=(
            "Feeds a folder of .eml/.txt emails through a local LLM (the "
            "doc-qa brick) to produce a per-thread summary, priority flag, "
            "and a draft reply -- inbox triage without a single message "
            "leaving the device."
        ),
        engines=["portable", "openvino"],
        status="planned",
    ),
    Demo(
        id="code-review-assist",
        name="Commit & Code Review Assistant",
        category="Productivity",
        tagline="Turn a git diff into a commit message and review notes in seconds.",
        description=(
            "Reads a working-tree diff and asks a local LLM (the doc-qa "
            "brick) to draft a commit message and flag obvious issues -- the "
            "same round trip a cloud coding assistant does, but no diff ever "
            "leaves the machine."
        ),
        engines=["portable", "openvino"],
        status="planned",
    ),
    Demo(
        id="smart-recall",
        name="Local Screen Memory",
        category="Productivity",
        tagline="Semantic search over everything you've seen on screen -- fully local.",
        description=(
            "Periodically OCRs the screen (the screen-ocr brick) and indexes "
            "it (the doc-qa brick's embedding pipeline) so you can ask for "
            "'that pricing page from yesterday' and get it back instantly -- "
            "the Windows Recall idea, minus sending your screen history "
            "anywhere."
        ),
        engines=["portable", "openvino"],
        status="planned",
    ),
    Demo(
        id="noise-suppression",
        name="Live Noise Suppression",
        category="Audio",
        tagline="Clean up mic audio in real time for calls.",
        description=(
            "Removes background noise from the microphone stream in real "
            "time using a local denoising model."
        ),
        engines=["openvino"],
        status="planned",
    ),
]


def get(demo_id: str) -> Demo | None:
    return next((d for d in REGISTRY if d.id == demo_id), None)
