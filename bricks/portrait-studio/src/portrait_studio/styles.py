"""The styles a portrait can be drawn in: a name to show, and what the model
is told.

What the model is told is an instruction about the photo it is given, in
two halves: what to make of it, and what to leave alone. The second half
names nothing a person may or may not have. The model takes every word as
something to draw (tried on four portraits, 2026-10-11 and -12):

- "keep the same glasses and beard" put glasses and a beard on a woman who
  had neither, and "the same facial hair" did it again;
- "the same dark or light skin colour" made a light-skinned woman dark;
- with none of that said, "a character from a family film" made a
  dark-skinned man light and took his beard.

What held on all four was to say that only the rendering changes and
everything else in the photo stays as it is. It still draws dark skin a
shade lighter than the photo in the cartoon style: the README says so.
"""
from __future__ import annotations

from dataclasses import dataclass

_KEEP = "Change only the rendering style: keep the person, their colours, their features and everything else in the photo exactly as they are."
# For a style without colour: the colours are the one thing that does change.
_KEEP_NO_COLOUR = "Change only the rendering style: keep the person, their features and everything else in the photo exactly as they are."
# For a style that dresses the person and moves them to a studio.
_KEEP_PERSON = "Change only the clothes, the background and the lighting: keep the person, their colours and their features exactly as they are."


@dataclass(frozen=True)
class Style:
    id: str
    name: str
    does: str  # what it makes of the photo
    hint: str = ""  # a few words for the card
    keeps: str = _KEEP  # what it leaves alone

    @property
    def prompt(self) -> str:
        return f"{self.does} {self.keeps}"


STYLES: tuple[Style, ...] = (
    Style("cartoon", "3D cartoon",
          "Restyle this photo as a frame from a modern 3D animated film.",
          "a frame from an animated film"),
    Style("corporate", "Corporate headshot",
          "Turn this photo into a professional corporate headshot: a dark business suit, a plain light grey studio "
          "background, soft studio lighting, sharp focus, head and shoulders.",
          "suit, studio light, grey backdrop", _KEEP_PERSON),
    Style("manga", "Manga",
          "Redraw this photo as a black and white manga illustration, with clean ink lines and screentone shading.",
          "ink lines and screentones", _KEEP_NO_COLOUR),
    Style("comic", "Comic book",
          "Redraw this photo as an American comic book illustration, with bold black outlines, flat vivid colours and "
          "halftone dots. No speech bubbles and no lettering.",
          "bold outlines, flat colours"),
    Style("oil", "Oil painting",
          "Repaint this photo as a classical oil portrait, with visible brush strokes and warm museum lighting.",
          "brush strokes, museum light"),
    Style("watercolour", "Watercolour",
          "Repaint this photo as a loose watercolour on textured paper, with soft washes of colour and white paper "
          "showing through.",
          "soft washes on paper"),
    Style("sketch", "Pencil sketch",
          "Redraw this photo as a pencil sketch on white paper, with fine graphite lines and light cross-hatching.",
          "graphite on white paper", _KEEP_NO_COLOUR),
    Style("clay", "Clay figure",
          # "A still", not "a frame": asked for a frame from such a film, it drew a wooden picture frame.
          "Restyle this photo as a still from a stop-motion film made with hand-modelled clay figures.",
          "stop-motion plasticine"),
)

_BY_ID = {style.id: style for style in STYLES}


def get(style_id: str) -> Style:
    try:
        return _BY_ID[style_id]
    except KeyError:
        raise ValueError(f"No style called '{style_id}'. Choose one of: {', '.join(_BY_ID)}.") from None
