"""Named example texts so the CLI/launcher can offer something ready to
synthesize once a voice is enrolled, instead of requiring hand-typed text."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    name: str
    description: str
    text: str


SAMPLES: list[Sample] = [
    Sample(
        name="Product intro",
        description="A short on-device-privacy pitch line.",
        text=(
            "Welcome to Panther Lake AI Studio. Everything you just heard was "
            "generated entirely on this device, with no audio ever leaving your "
            "machine."
        ),
    ),
    Sample(
        name="Weather announcement",
        description="A cheerful short weather blurb.",
        text=(
            "Good morning! Today's forecast is mostly sunny with a high of "
            "seventy-two degrees. Perfect weather for a walk this afternoon."
        ),
    ),
    Sample(
        name="Story snippet",
        description="A short narrative passage.",
        text=(
            "The old lighthouse keeper climbed the spiral stairs one last time, "
            "lantern in hand, watching the fog roll in from the north."
        ),
    ),
    Sample(
        name="Meeting reminder",
        description="A short workplace reminder message.",
        text=(
            "Hi team, just a reminder that our quarterly planning meeting starts "
            "at ten o'clock tomorrow morning in the main conference room."
        ),
    ),
]
