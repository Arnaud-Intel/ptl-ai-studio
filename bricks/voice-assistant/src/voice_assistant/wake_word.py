"""Always-listening wake-word detection via openWakeWord (ONNX Runtime).

Deliberately not split into portable/openvino engine variants like every
other model in this workspace: wake-word detection is a tiny classifier
meant to run continuously in the background at near-zero CPU cost -- it
is not the part of a voice assistant that benefits from NPU/iGPU offload.
The `--engine` flag on this brick instead selects the transcription,
reasoning, and speech-synthesis backend (see session.py) -- exactly where
OpenVINO acceleration actually matters, and where the heavy per-request
compute is spent.
"""
from __future__ import annotations

import numpy as np

AVAILABLE_WAKE_WORDS = ["hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy"]
DEFAULT_WAKE_WORD = "hey_jarvis"


class WakeWordDetector:
    def __init__(self, wake_word: str = DEFAULT_WAKE_WORD, threshold: float = 0.5):
        import openwakeword
        from openwakeword.model import Model

        if wake_word not in AVAILABLE_WAKE_WORDS:
            raise ValueError(f"Unknown wake word '{wake_word}'. Choices: {', '.join(AVAILABLE_WAKE_WORDS)}")

        openwakeword.utils.download_models([wake_word])
        self.model = Model(wakeword_models=[wake_word], inference_framework="onnx")
        self.wake_word = wake_word
        self.threshold = threshold

    def triggered(self, block: np.ndarray) -> bool:
        """`block`: mono float32 audio in [-1, 1] at 16kHz, any length --
        openWakeWord buffers internally, so this can be called with
        whatever chunk size the capture stream hands it. Returns True the
        instant the wake word's score crosses the threshold."""
        pcm16 = np.clip(block * 32767, -32768, 32767).astype(np.int16)
        scores = self.model.predict(pcm16)
        return scores.get(self.wake_word, 0.0) > self.threshold

    def reset(self) -> None:
        """Clears the model's internal audio-feature and prediction
        buffers. Call this after handling a wake -> command cycle, before
        resuming wake-word scanning -- otherwise the buffer's rolling
        window still partly reflects audio from before the gap where this
        detector wasn't being fed (command capture bypasses it entirely),
        which can produce a spurious trigger right as scanning resumes.
        Found via testing, not a hypothetical: a real run reliably
        produced exactly this false positive without the reset."""
        self.model.reset()
