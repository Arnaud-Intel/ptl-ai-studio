"""Picks a voice-cloning backend (engine) so the rest of the brick doesn't
need to know which one it's talking to -- both expose `.enroll(path) ->
target_se` and `.synthesize(text, target_se, style, tau) -> (audio, sr)`.
"""
from __future__ import annotations

from typing import Protocol

from pantherlake_ai_core.engine import Engine


class Cloner(Protocol):
    def enroll(self, reference_audio_path: str): ...
    def synthesize(self, text: str, target_se, style: str = "default", tau: float = 0.3): ...


def create_cloner(engine: Engine, *, device: str = "CPU", model_path: str | None = None) -> Cloner:
    if engine == Engine.PORTABLE:
        from .cloner_portable import PortableCloner

        return PortableCloner(model_path=model_path)

    if engine == Engine.OPENVINO:
        from .cloner_openvino import OpenVINOCloner

        return OpenVINOCloner(device=device, model_path=model_path)

    raise ValueError(f"Unknown engine '{engine}'.")
