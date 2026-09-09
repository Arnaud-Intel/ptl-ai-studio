"""Enroll a reference voice once, synthesize text in it as many times as
you like -- shared by the CLI and the launcher so this logic lives in one
place.
"""
from __future__ import annotations

from typing import Callable

from pantherlake_ai_core.engine import Engine

from .engine_factory import DEFAULT_MODEL, create_cloner
from .voice_model import STYLES


class VoiceCloneSession:
    """Holds one loaded cloner. Enroll once, synthesize many times."""

    def __init__(
        self,
        engine: Engine,
        *,
        model: str = DEFAULT_MODEL,
        device: str = "CPU",
        model_path: str | None = None,
        on_downloading: Callable[[], None] | None = None,
    ):
        self.engine = engine
        self.model = model
        self.cloner = create_cloner(
            engine, model=model, device=device, model_path=model_path, on_downloading=on_downloading
        )
        self.target_se = None
        self.reference_path: str | None = None

    @property
    def supports_styles(self) -> bool:
        """Whether `style`/`tau` mean anything for the loaded model."""
        return getattr(self.cloner, "supports_styles", False)

    def enroll(self, reference_audio_path: str) -> None:
        self.target_se = self.cloner.enroll(reference_audio_path)
        self.reference_path = reference_audio_path

    def synthesize(self, text: str, *, style: str = "default", tau: float = 0.3):
        if self.target_se is None:
            raise RuntimeError("No voice enrolled yet -- call enroll() first.")
        if not self.supports_styles:
            # Not silently dropped: a caller asking for "angry" and getting
            # a flat reading back would look like the model ignoring them.
            if style != "default":
                raise ValueError(
                    f"The '{self.model}' model has no delivery styles -- it takes its delivery from the "
                    f"reference clip. Write a paralinguistic tag such as [laugh] into the text instead."
                )
            return self.cloner.synthesize(text, self.target_se)
        if style not in STYLES:
            raise ValueError(f"Unknown style '{style}'. Choices: {', '.join(STYLES)}")
        return self.cloner.synthesize(text, self.target_se, style=style, tau=tau)
