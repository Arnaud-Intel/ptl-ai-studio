"""Listen -> think -> speak loop, shared by the CLI and the launcher.

Composes three other bricks rather than reimplementing any of them:
`live_translation.transcriber` for speech-to-text, `doc_qa.engine_factory`
for the LLM, and `voice_clone_studio.voice_model` for text-to-speech --
the same "brick composing bricks" shape `meeting-notes` uses, just with a
third brick added and, in front of it, this brick's own piece: what says a
question is being asked. That is a button held while speaking, unless a wake
word is asked for (see wake_word.py for why it is not the default).
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Iterator

import numpy as np
from doc_qa import language_models
from doc_qa.engine_factory import create_llm
from live_translation.transcriber import create_translator
from pantherlake_ai_core import audio
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.segmenter import VADConfig, segment_stream
from voice_clone_studio import voice_model as vc_voice_model

from .wake_word import WakeWordDetector

DEFAULT_MODEL = language_models.DEFAULT

# Hold to talk. What is kept from just before the button went down, so that a
# word begun as it is pressed is not cut; the shortest hold that is read (a
# click is not a question); and the longest, Whisper reading thirty seconds
# at a time.
_LEAD_SECONDS = 0.3
_SHORTEST_SECONDS = 0.4
_LONGEST_SECONDS = 28.0

TOO_SHORT = "Hold the button for as long as you speak."
NOTHING_HEARD = "Nothing was heard."

_SYSTEM_PROMPT = (
    "You are a helpful voice assistant running entirely on this device. "
    "Answer the user's spoken request in 1-3 short sentences of plain "
    "spoken language -- no markdown, no bullet points, no code blocks. "
    "If you don't know, say so briefly instead of guessing."
)


class VoiceAssistantSession:
    """Holds one loaded transcriber + LLM + TTS voice, and the wake-word
    detector when there is a wake word."""

    def __init__(
        self,
        engine: Engine,
        *,
        whisper_model_size: str,
        device: str,
        wake_word: str | None = None,
        wake_threshold: float = 0.5,
        model: str | None = None,
    ):
        self.engine = engine
        self.device = device
        self.wake_detector = WakeWordDetector(wake_word=wake_word, threshold=wake_threshold) if wake_word else None
        self.transcriber = create_translator(engine, whisper_model_size, device, task="transcribe")
        # Which language model answers, by its key in doc-qa's language_models.
        self.model = language_models.get(model or DEFAULT_MODEL).key
        self.llm = create_llm(engine, device=device, model_repo=language_models.repo_for(self.model, engine, device))

        self.tts = vc_voice_model.load_tts_only()
        # The voice is made on the CPU whatever chip the rest is on: its
        # model does not compile for the NPU (giving it the NPU ended the
        # whole program), and on a GPU it is compiled again for every
        # sentence of a new length, 16 to 22 s each. On the CPU a reply is
        # ready in half a second. Listening and answering stay on `device`.
        self.voice_device = vc_voice_model.TTS_DEVICE if engine == Engine.OPENVINO else "CPU"
        if engine == Engine.OPENVINO:
            vc_voice_model.accelerate_tts_with_openvino(self.tts, device=self.voice_device)

    def ask(self, question: str, *, max_tokens: int = 200) -> str:
        return self.llm.answer(_SYSTEM_PROMPT, question, max_tokens=max_tokens)

    def speak(self, text: str, style: str = "default") -> tuple[np.ndarray, int]:
        audio_out = self.tts.tts(text, None, speaker=style, language="English")
        return audio_out, vc_voice_model.SAMPLE_RATE


def _after_the_wake_word(
    detector: WakeWordDetector, blocks: Iterator[np.ndarray], on_wake: Callable[[], None]
) -> Iterator[np.ndarray | None]:
    """What is said after each wake word: the same energy-based segmenter
    live-translation uses cuts it -- so pause briefly after the wake word,
    the same way you would with any wake-word assistant, rather than running
    straight into your request."""
    for block in blocks:
        if not detector.triggered(block):
            continue
        on_wake()
        yield next(segment_stream(blocks, VADConfig()), None)


def _while_held(
    blocks: Iterator[np.ndarray],
    talk: threading.Event,
    on_wake: Callable[[], None],
    on_missed: Callable[[str], None],
) -> Iterator[np.ndarray | None]:
    """What the microphone gives while `talk` is set, each time it is: the
    button says where a question starts and ends, so nothing has to guess it
    from the sound -- which is what fails in a loud hall. The microphone is
    read all along and what it gives between two questions is dropped, but
    for the last third of a second."""
    rate = audio.SAMPLE_RATE
    lead: deque[np.ndarray] = deque()
    lead_samples = 0
    for block in blocks:
        if not talk.is_set():
            lead.append(block)
            lead_samples += len(block)
            while lead_samples - len(lead[0]) >= _LEAD_SECONDS * rate:
                lead_samples -= len(lead.popleft())
            continue
        on_wake()
        held, spoken = [*lead, block], len(block)
        lead.clear()
        lead_samples = 0
        for block in blocks:
            held.append(block)
            spoken += len(block)
            if not talk.is_set() or spoken >= _LONGEST_SECONDS * rate:
                break
        if spoken < _SHORTEST_SECONDS * rate:
            on_missed(TOO_SHORT)
            yield None
        else:
            yield np.concatenate(held)


def run(
    *,
    audio_device: str | None,
    engine: Engine,
    whisper_model_size: str,
    compute_device: str,
    wake_word: str | None = None,
    wake_threshold: float = 0.5,
    talk: threading.Event | None = None,
    llm_model: str | None = None,
    on_wake: Callable[[], None] = lambda: None,
    on_missed: Callable[[str], None] = lambda reason: None,
    on_heard: Callable[[str], None] = lambda text: None,
    on_reply: Callable[[str], None] = lambda text: None,
    on_idle: Callable[[], None] = lambda: None,
    on_ready: Callable[[], None] | None = None,
    on_stats: Callable[[object], None] | None = None,
    speak_replies: bool = True,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks the calling thread. A question is what is said while `talk` is
    set -- a button held, the caller's to set and clear -- or, when
    `wake_word` names one, what follows that word. It is transcribed, the LLM
    is asked, and the reply is (optionally) spoken out loud before listening
    again.

    `on_wake` fires when a question starts to be listened to, `on_missed`
    with the reason when one came to nothing, `on_idle` when the assistant is
    ready for the next. `on_ready`, if given, fires once the models (Whisper,
    LLM, TTS, and the wake word's if any) are loaded and the mic is about to
    be opened -- the real "loading -> listening" boundary, which can be a
    minute or more on a cold cache.
    """
    if not wake_word and talk is None:
        raise ValueError("With no wake word the assistant listens while `talk` is set: pass that event.")
    session = VoiceAssistantSession(
        engine,
        whisper_model_size=whisper_model_size,
        device=compute_device,
        wake_word=wake_word or None,
        wake_threshold=wake_threshold,
        model=llm_model,
    )
    if on_ready is not None:
        on_ready()

    blocks = audio.stream_blocks("mic", audio_device, stop_event=stop_event)
    detector = session.wake_detector
    questions = _after_the_wake_word(detector, blocks, on_wake) if detector else _while_held(blocks, talk, on_wake, on_missed)

    def rest() -> None:
        if detector is not None:
            # See WakeWordDetector.reset()'s docstring: command capture
            # bypasses the wake-word detector entirely, so its internal
            # buffer needs a clear before scanning resumes -- otherwise it
            # can spuriously re-trigger on the very next silent block.
            detector.reset()
        else:
            # A button pressed while the answer was being written or said is
            # not a question: what the microphone gave then is the answer.
            talk.clear()
        on_idle()

    for command_audio in questions:
        if command_audio is None:
            rest()
            continue

        heard = session.transcriber.translate(command_audio)
        if heard is None or not heard.text.strip():
            on_missed(NOTHING_HEARD)
            rest()
            continue
        on_heard(heard.text)

        reply = session.ask(heard.text)
        on_reply(reply)
        # How fast the reply came, for a caller showing it (the launcher's panel).
        stats = getattr(session.llm, "last_stats", None)
        if on_stats is not None and stats is not None:
            on_stats(stats)

        if speak_replies:
            audio_out, sample_rate = session.speak(reply)
            audio.play(audio_out, sample_rate)

        rest()
