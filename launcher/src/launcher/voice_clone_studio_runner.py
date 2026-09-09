"""Holds the voice-clone-studio brick's session (cloner + enrolled voice)
and exposes its blocking calls for the web UI to run off the event loop
(via `fastapi.concurrency.run_in_threadpool`).
"""
from __future__ import annotations

import os
import tempfile
import threading

from pantherlake_ai_core import audio
from pantherlake_ai_core.engine import Engine
from voice_clone_studio.pipeline import VoiceCloneSession

from . import activity, events
from .errors import Conflict

_DEMO_ID = "voice-clone-studio"


class VoiceCloneStudioRunner:
    def __init__(self) -> None:
        self._session: VoiceCloneSession | None = None
        self._engine: str | None = None
        self._model: str | None = None
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
        os.close(fd)
        sf.write(path, clip, audio.SAMPLE_RATE)
        return path

    def enroll(self, *, reference_path: str, engine: str, device: str, model: str) -> None:
        """Blocking -- loads the cloner the first time or when the
        engine/device changes, then enrolls the reference clip."""

        def on_downloading() -> None:
            events.set_phase(_DEMO_ID, "loading", f"Downloading {model} (first run only)...")

        with self._lock:
            activity.set_active(_DEMO_ID, engine=engine, device=device)
            try:
                if self._session is None or self._engine != engine or self._device != device or self._model != model:
                    events.set_phase(_DEMO_ID, "loading", f"Loading {model} (engine={engine}, device={device})...")
                    self._session = VoiceCloneSession(
                        Engine(engine), model=model, device=device, on_downloading=on_downloading
                    )
                    self._engine = engine
                    self._device = device
                    self._model = model
                events.set_phase(_DEMO_ID, "running", "Enrolling voice...")
                self._session.enroll(reference_path)
                self._enrolled = True
                events.clear_phase(_DEMO_ID)
            except Exception as exc:  # covers the session build too, so a failed load can't stick at "loading"
                events.set_phase(_DEMO_ID, "error", str(exc))
                raise
            finally:
                activity.clear_active(_DEMO_ID)

    @property
    def supports_styles(self) -> bool:
        """Whether the loaded model has delivery styles at all -- the UI
        hides the style and tau controls when it doesn't."""
        return bool(self._session and self._session.supports_styles)

    @property
    def model(self) -> str | None:
        return self._model

    def synthesize(self, *, text: str, style: str, tau: float):
        """Blocking. Returns (audio: np.ndarray, sample_rate)."""
        with self._lock:
            if self._session is None or not self._enrolled:
                raise Conflict("Enroll a voice first.")
            activity.set_active(_DEMO_ID, engine=self._engine, device=self._device)
            events.set_phase(_DEMO_ID, "running", "Synthesizing speech...")
            try:
                # Passed through whatever the model is: the session owns
                # the rule, and refuses a style on a model that has none
                # rather than dropping it. Branching here instead would
                # silently ignore what the caller actually asked for.
                result = self._session.synthesize(text, style=style, tau=tau)
            except Exception as exc:
                events.set_phase(_DEMO_ID, "error", str(exc))
                raise
            finally:
                activity.clear_active(_DEMO_ID)
            events.clear_phase(_DEMO_ID)
            return result
