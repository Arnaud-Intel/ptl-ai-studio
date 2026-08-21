"""Enroll a reference voice once, synthesize text in it as many times as
you like -- shared by the CLI and the launcher so this logic lives in one
place.
"""
from __future__ import annotations

from pantherlake_ai_core.engine import Engine

from .engine_factory import create_cloner
from .voice_model import STYLES


class VoiceCloneSession:
    """Holds one loaded cloner. Enroll once, synthesize many times."""

    def __init__(self, engine: Engine, *, device: str = "CPU", model_path: str | None = None):
        self.engine = engine
        self.cloner = create_cloner(engine, device=device, model_path=model_path)
        self.target_se = None
        self.reference_path: str | None = None

    def enroll(self, reference_audio_path: str) -> None:
        self.target_se = self.cloner.enroll(reference_audio_path)
        self.reference_path = reference_audio_path

    def synthesize(self, text: str, *, style: str = "default", tau: float = 0.3):
        if self.target_se is None:
            raise RuntimeError("No voice enrolled yet -- call enroll() first.")
        if style not in STYLES:
            raise ValueError(f"Unknown style '{style}'. Choices: {', '.join(STYLES)}")
        return self.cloner.synthesize(text, self.target_se, style=style, tau=tau)
