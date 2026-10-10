"""Saying the line aloud.

Two voices, both the studio's own (voice-clone-studio), neither new:

- **The studio's voice**: the base speaker the voice assistant answers
  with. It has deliveries -- cheerful, excited, sad -- and the mood picks
  one, so a sports line is not read like a documentary.
- **A cloned voice**: whichever voice was enrolled for it. The commentator
  does not enrol one itself: the launcher lends it the one enrolled in
  Voice Clone Studio, the command line clones the clip it is given.

Measured on the XPS 14 (2026-10-09). Alone, one comment of about fifteen
words, which is four and a half seconds of speech:

    the studio's voice, CPU       0.5 s
    the studio's voice, GPU       16 to 22 s: it compiles again for every
                                  sentence of another length
    the studio's voice, NPU       does not compile; the compiler takes the
                                  whole process down with it
    a cloned voice, OpenVoice     6 s (CPU)
    a cloned voice, Chatterbox    8 to 11 s (CPU; the model has no other chip)

In the launcher, while the video plays and the two other models work:

    the studio's voice            0.6 to 1.0 s for a line of 5 to 8 s
    a cloned voice, Chatterbox    18 to 30 s a line
    a cloned voice, OpenVoice     22 s a line, enrolled on the GPU

So the studio's voice runs on the CPU and nowhere else, and a comment is
about three seconds behind the picture when it starts to be heard: a line
every eight to ten seconds, since the next look waits for the last line to
have been said. A cloned voice is a different commentary: two lines a
minute, each some twenty seconds behind its picture, and a Stop that waits
for the line in hand. It is offered for what it is -- the enrolled voice,
saying what the machine saw -- not as a live commentary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import moods

OFF, STUDIO, CLONED = "", "studio", "cloned"
# Where the studio's voice runs: see the measurements above.
STUDIO_DEVICE = "CPU"

Speech = tuple[np.ndarray, int]  # the samples, and how many of them a second


@dataclass(frozen=True)
class Voice:
    key: str
    name: str


VOICES: list[Voice] = [
    Voice(STUDIO, "The studio's voice"),
    Voice(CLONED, "The cloned voice"),
]

# How the studio's voice delivers a line in each mood.
DELIVERY: dict[str, str] = {
    "plain": "default",
    "upbeat": "cheerful",
    "sports": "excited",
    "documentary": "default",
    "deadpan": "sad",
}


def get(key: str | None) -> str:
    """The voice's key, checked. Nothing, or "off", is silence."""
    key = (key or "").strip().lower()
    if key in (OFF, "off", "none"):
        return OFF
    if key not in {voice.key for voice in VOICES}:
        raise ValueError(f"Unknown voice '{key}'. Choices: {', '.join(voice.key for voice in VOICES)}, or nothing for silence.")
    return key


def spoken(text: str) -> str:
    """The line as it is handed to a voice: what a caption can carry and a
    voice cannot -- quotation marks, asterisks -- left out."""
    return " ".join(text.replace("*", " ").replace('"', " ").replace("“", " ").replace("”", " ").split())


class StudioVoice:
    """The base speaker, through OpenVINO on the CPU."""

    def __init__(self, on_downloading: Callable[[], None] | None = None) -> None:
        from voice_clone_studio import voice_model

        self._rate = voice_model.SAMPLE_RATE
        self._tts = voice_model.load_tts_only(on_downloading=on_downloading)
        voice_model.accelerate_tts_with_openvino(self._tts, device=STUDIO_DEVICE)

    def speak(self, text: str, mood_key: str) -> Speech:
        delivery = DELIVERY.get(mood_key, "default")
        return np.asarray(self._tts.tts(text, None, speaker=delivery, language="English"), dtype=np.float32), self._rate


class Speaker:
    """Says a line in the voice asked for. A voice is loaded the first time
    it is used, so that a commentary nobody asked to hear loads none."""

    def __init__(
        self,
        *,
        clone: Callable[[str], Speech] | None = None,
        studio: Callable[[], StudioVoice] | None = None,
        on_downloading: Callable[[], None] | None = None,
    ) -> None:
        self._clone = clone
        self._make_studio = studio or (lambda: StudioVoice(on_downloading=on_downloading))
        self._studio: StudioVoice | None = None

    def speak(self, voice_key: str, text: str, mood_key: str = moods.DEFAULT) -> Speech:
        line = spoken(text)
        if not line:
            raise ValueError("Nothing to say.")
        if get(voice_key) == CLONED:
            if self._clone is None:
                raise RuntimeError("No cloned voice to speak with: enrol one first.")
            audio, rate = self._clone(line)
            return np.asarray(audio, dtype=np.float32), int(rate)
        if self._studio is None:
            self._studio = self._make_studio()
        return self._studio.speak(line, mood_key)
