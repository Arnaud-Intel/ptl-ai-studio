"""Wake -> listen -> think -> speak loop, shared by the CLI and the launcher.

Composes three other bricks rather than reimplementing any of them:
`live_translation.transcriber` for speech-to-text, `doc_qa.engine_factory`
for the LLM, and `voice_clone_studio.voice_model` for text-to-speech --
the same "brick composing bricks" shape `meeting-notes` uses, just with a
third brick added and a wake-word gate (this brick's own, genuinely new
piece) in front of it.
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from doc_qa.engine_factory import create_llm
from live_translation.transcriber import create_translator
from pantherlake_ai_core import audio
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.segmenter import VADConfig, segment_stream
from voice_clone_studio import voice_model as vc_voice_model

from .wake_word import DEFAULT_WAKE_WORD, WakeWordDetector

_SYSTEM_PROMPT = (
    "You are a helpful voice assistant running entirely on this device. "
    "Answer the user's spoken request in 1-3 short sentences of plain "
    "spoken language -- no markdown, no bullet points, no code blocks. "
    "If you don't know, say so briefly instead of guessing."
)


class VoiceAssistantSession:
    """Holds one loaded wake-word detector + transcriber + LLM + TTS voice."""

    def __init__(
        self,
        engine: Engine,
        *,
        whisper_model_size: str,
        device: str,
        wake_word: str = DEFAULT_WAKE_WORD,
        wake_threshold: float = 0.5,
    ):
        self.engine = engine
        self.device = device
        self.wake_detector = WakeWordDetector(wake_word=wake_word, threshold=wake_threshold)
        self.transcriber = create_translator(engine, whisper_model_size, device, task="transcribe")
        self.llm = create_llm(engine, device=device)

        self.tts = vc_voice_model.load_tts_only()
        if engine == Engine.OPENVINO:
            vc_voice_model.accelerate_tts_with_openvino(self.tts, device=device)

    def ask(self, question: str, *, max_tokens: int = 200) -> str:
        return self.llm.answer(_SYSTEM_PROMPT, question, max_tokens=max_tokens)

    def speak(self, text: str, style: str = "default") -> tuple[np.ndarray, int]:
        audio_out = self.tts.tts(text, None, speaker=style, language="English")
        return audio_out, vc_voice_model.SAMPLE_RATE


def run(
    *,
    audio_device: str | None,
    engine: Engine,
    whisper_model_size: str,
    compute_device: str,
    wake_word: str = DEFAULT_WAKE_WORD,
    wake_threshold: float = 0.5,
    on_wake: Callable[[], None] = lambda: None,
    on_heard: Callable[[str], None] = lambda text: None,
    on_reply: Callable[[str], None] = lambda text: None,
    speak_replies: bool = True,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread. Continuously scans the microphone for the
    wake word; once heard, captures and transcribes the command that
    follows (the same energy-based segmenter live-translation uses -- so
    pause briefly after the wake word, the same way you would with any
    wake-word assistant, rather than running straight into your request),
    asks the LLM, and (optionally) speaks the reply out loud before going
    back to listening.
    """
    session = VoiceAssistantSession(
        engine,
        whisper_model_size=whisper_model_size,
        device=compute_device,
        wake_word=wake_word,
        wake_threshold=wake_threshold,
    )

    blocks = audio.stream_blocks("mic", audio_device, stop_event=stop_event)

    for block in blocks:
        if not session.wake_detector.triggered(block):
            continue

        on_wake()
        command_audio = next(segment_stream(blocks, VADConfig()), None)
        if command_audio is None:
            session.wake_detector.reset()
            continue

        heard = session.transcriber.translate(command_audio)
        if heard is None or not heard.text.strip():
            session.wake_detector.reset()
            continue
        on_heard(heard.text)

        reply = session.ask(heard.text)
        on_reply(reply)

        if speak_replies:
            audio_out, sample_rate = session.speak(reply)
            audio.play(audio_out, sample_rate)

        # See WakeWordDetector.reset()'s docstring: command capture bypasses
        # the wake-word detector entirely, so its internal buffer needs a
        # clear before scanning resumes -- otherwise it can spuriously
        # re-trigger on the very next silent block.
        session.wake_detector.reset()
