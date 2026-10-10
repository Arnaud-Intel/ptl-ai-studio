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

# The rules above were written for Qwen2.5-1.5B, which needs holding back.
# Qwen3-8B (the "8b" of doc-qa's language_models) obeys "add none" to the
# letter: of 48 lines -- twelve sentences seen in the sample videos, each in
# the four moods, on the NPU, 2026-10-10 -- it gave 29 back as they were,
# word for word. Told instead to use new words, it repeated 12 and invented
# about as little: "parked cars" for cars in the background, "tall
# buildings". A third wording, the voice first and the facts after, repeated
# 5 and brought the embroidery back (a person "sprints", a storm drain
# "peeking through the cracks"). So the second is what the 8B is told.
#
# What the two models make of a mood, on those 48: the 1.5B has the voice
# and adds something that was not seen in about 18 lines ("Sealed 200
# bottles per minute", a parked car's "wheels spinning silently"); the 8B
# adds something in about 4 and has less of a voice -- a sports commentator
# who says "a herd of cattle charges down a road, guided by a rider on
# horseback". A line takes it 2.2 s against 0.8.
_RULES_BY_MODEL = {
    "8b": " One short sentence, in new words: do not repeat the sentence as it is. Say only what the sentence says; "
          "add no detail that is not in it. Reply with the sentence only.",
}


def worded_for(instruction: str, model: str) -> str:
    """`instruction` as the language model `model` should be given it."""
    return instruction.replace(_RULES, _RULES_BY_MODEL.get(model, _RULES))


def get(key: str) -> Mood:
    try:
        return BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown mood '{key}': one of {', '.join(BY_KEY)}.") from None
