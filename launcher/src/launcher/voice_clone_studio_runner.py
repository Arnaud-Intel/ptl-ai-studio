"""Holds the voice-clone-studio brick's session (cloner + enrolled voice)
and exposes its blocking calls for the web UI to run off the event loop
(via `fastapi.concurrency.run_in_threadpool`).
"""
from __future__ import annotations

import threading

from pantherlake_ai_core import audio
from pantherlake_ai_core.engine import Engine
from voice_clone_studio.pipeline import VoiceCloneSession

from . import activity, events

_DEMO_ID = "voice-clone-studio"


class VoiceCloneStudioRunner:
    def __init__(self) -> None:
        self._session: VoiceCloneSession | None = None
        self._engine: str | None = None
        self._device: str | None = None
        self._enrolled = False
        self._lock = threading.Lock()

    @property
    def enrolled(self) -> bool:
        return self._enrolled

    def record_reference(self, seconds: float) -> str:
        """Blocking -- records from this machine's own default microphone,
        the same way as every other capture in this launcher (the browser
        is a control surface for the local machine, not the mic source)."""
        import tempfile

        import numpy as np
        import soundfile as sf

        blocks = []
        captured = 0.0
        for block in audio.stream_blocks("mic", None):
            blocks.append(block)
            captured += len(block) / audio.SAMPLE_RATE
            if captured >= seconds:
                break
        clip = np.concatenate(blocks)

        fd, path = tempfile.mkstemp(suffix=".wav")
        import os

        os.close(fd)
        sf.write(path, clip, audio.SAMPLE_RATE)
        return path

    def enroll(self, *, reference_path: str, engine: str, device: str) -> None:
        """Blocking -- loads the cloner the first time or when the
        engine/device changes, then enrolls the reference clip."""
        with self._lock:
            activity.set_active(_DEMO_ID, engine=engine, device=device)
            try:
                if self._session is None or self._engine != engine or self._device != device:
                    events.set_phase(_DEMO_ID, "loading", f"Loading model (engine={engine}, device={device})...")
                    self._session = VoiceCloneSession(Engine(engine), device=device)
                    self._engine = engine
                    self._device = device
                events.set_phase(_DEMO_ID, "running", "Enrolling voice...")
                try:
                    self._session.enroll(reference_path)
                except Exception as exc:
                    events.set_phase(_DEMO_ID, "error", str(exc))
                    raise
                self._enrolled = True
                events.clear_phase(_DEMO_ID)
            finally:
                activity.clear_active(_DEMO_ID)

    def synthesize(self, *, text: str, style: str, tau: float):
        """Blocking. Returns (audio: np.ndarray, sample_rate)."""
        with self._lock:
            if self._session is None or not self._enrolled:
                raise RuntimeError("Enroll a voice first.")
            activity.set_active(_DEMO_ID, engine=self._engine, device=self._device)
            events.set_phase(_DEMO_ID, "running", "Synthesizing speech...")
            try:
                result = self._session.synthesize(text, style=style, tau=tau)
            except Exception as exc:
                events.set_phase(_DEMO_ID, "error", str(exc))
                raise
            finally:
                activity.clear_active(_DEMO_ID)
            events.clear_phase(_DEMO_ID)
            return result
