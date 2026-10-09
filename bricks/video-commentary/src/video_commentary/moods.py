"""The moods a comment can be said in.

A mood is an instruction to the small language model that rewrites the
vision model's plain line: nothing about the picture changes, only the
voice. That is why it can be changed while the video plays, and why the
plain line stays on screen beside the one in a mood -- the small model
embroiders (asked for "upbeat", it sent a herd "on their way to market"),
and what was actually seen should be one glance away.

Not here on purpose: any mood that passes judgement on the people in the
picture -- how they look, what they wear. A commentator that talks about
a street is one thing; one that rates the passers-by is another, to be
built differently if it is built (asked for first, about clothes only,
kind only) and not as one more line in this list.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mood:
    key: str
    name: str
    # What the language model is told. None: the line is shown as it was seen.
    instruction: str | None


_RULES = " One short sentence. Keep every fact; add none. Reply with the sentence only."

MOODS: tuple[Mood, ...] = (
    Mood("plain", "Just what it sees", None),
    Mood("upbeat", "Upbeat", "Rewrite the sentence as a warm, upbeat commentator would say it." + _RULES),
    Mood("sports", "Sports commentator", "Rewrite the sentence as an excited sports commentator would call it." + _RULES),
    Mood("documentary", "Nature documentary", "Rewrite the sentence as the calm narrator of a nature documentary would say it." + _RULES),
    Mood("deadpan", "Deadpan", "Rewrite the sentence in a dry, deadpan tone." + _RULES),
)
BY_KEY = {mood.key: mood for mood in MOODS}
DEFAULT = "upbeat"


def get(key: str) -> Mood:
    try:
        return BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown mood '{key}': one of {', '.join(BY_KEY)}.") from None
