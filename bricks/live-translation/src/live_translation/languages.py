"""The spoken languages a session can be told to expect.

Whisper works out the language of each utterance on its own, from that
utterance alone. That is what lets a session follow a meeting where people
switch languages -- and what makes it guess wrong on a short or clipped
phrase, a name, a "merci" in an English sentence, with the small models
most of all. Someone who knows what will be spoken can say so, and nothing
is guessed.

A short list on purpose: the model knows 99 languages, and a menu of 99 is
no help to anyone. Codes are Whisper's own (ISO 639-1).
"""
from __future__ import annotations

SPOKEN_LANGUAGES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ru": "Russian",
    "ar": "Arabic",
    "he": "Hebrew",
    "hi": "Hindi",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
}


def spoken_language(choice: str | None) -> str | None:
    """A language code for the model, or None to let it detect each
    utterance's language: nothing chosen, "" and "auto" all mean detect.
    Anything else has to be one of SPOKEN_LANGUAGES."""
    code = (choice or "").strip().lower()
    if code in ("", "auto"):
        return None
    if code not in SPOKEN_LANGUAGES:
        raise ValueError(f"Unknown spoken language '{choice}'. Choose auto or one of: {', '.join(SPOKEN_LANGUAGES)}.")
    return code
