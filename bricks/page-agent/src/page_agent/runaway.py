"""Stops a page that has started going round in circles.

A coding model writing a long stylesheet can fall into a loop: the same few
rules again and again until it runs out of tokens. Three pages in fifteen
did at the bricks' usual temperature of 0.2 (2026-10-08): the first was
36,000 characters, the last 28,000 of them five `.section:last-child .btn`
rules over and over, 150 seconds, and no `</html>`. Nothing in such a page
is usable, and every second spent on it is a second the second try starts
later. The page is now written at the coding model's own temperature, where
none of the next thirteen looped (see the conductor); this is what is left
for the day one does.

A loop is easy to tell from a page by how well its last few thousand
characters compress. Measured on what the models here wrote that day, three
loops and some thirty-five sound pages: the most repetitive 3,000
characters of a sound page -- three cards alike, a table -- compress to 19%
of their size at the very least, and those of a loop to between 4% and 6%.
The watch stops the page under 10%, which was 2,700 characters into the
first loop instead of 28,000.
"""
from __future__ import annotations

import zlib

from pantherlake_ai_core.types import GenerationControl

WINDOW = 3000  # characters looked at, at the end of what has been written
LIMIT = 0.10  # under this compression ratio they are one thing said many times
_EVERY = 400  # how many new characters between two looks


def repetitive(text: str) -> bool:
    """Whether the last WINDOW characters of `text` are a loop."""
    tail = text[-WINDOW:].encode("utf-8")
    return len(tail) >= WINDOW and len(zlib.compress(tail, 6)) / len(tail) < LIMIT


class Watch:
    """Sits between the model and whoever follows the page as it is written
    (`inner`, if anybody does): passes every piece on, and asks for the page
    to stop when the pieces have turned into a loop. `ran_away` says
    afterwards that this is why it stopped."""

    def __init__(self, inner: GenerationControl | None = None):
        self._inner = inner
        self._pieces: list[str] = []
        self._size = self._looked_at = 0
        self.ran_away = False
        self.control = GenerationControl(
            on_text=self._on_text,
            should_stop=self._should_stop,
            on_heading=inner.on_heading if inner else None,
            on_tokens=inner.on_tokens if inner else None,
        )

    def _on_text(self, piece: str) -> None:
        self._pieces.append(piece)
        self._size += len(piece)
        if self._inner is not None and self._inner.on_text is not None:
            self._inner.on_text(piece)

    def _should_stop(self) -> bool:
        if self._inner is not None and self._inner.should_stop is not None and self._inner.should_stop():
            return True
        if self._size - self._looked_at >= _EVERY:
            self._looked_at = self._size
            # Only the end is needed, and joining it is cheap next to a token.
            tail, held = [], 0
            for piece in reversed(self._pieces):
                tail.append(piece)
                held += len(piece)
                if held >= WINDOW:
                    break
            self.ran_away = repetitive("".join(reversed(tail)))
        return self.ran_away
