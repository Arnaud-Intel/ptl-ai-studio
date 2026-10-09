"""A video watched and commented on, a line every few seconds.

Two models, each on its own chip:

- a vision-language model looks at one frame and says in one plain
  sentence what is happening (screen-ocr's model, asked another question);
- a small language model says that sentence again in the mood chosen
  (doc-qa's model).

Measured on the XPS 14 (2026-10-09), the vision model on the integrated GPU
and the language model on the NPU: a plain line in 0.8 to 1.0 s for a frame
672 pixels wide (25 tok/s, the first word after 0.2 s), its mood in 0.5 to
0.8 s more. So a comment is about a second and a half behind the picture,
which a line every four seconds leaves time to read.

Why two models and not the mood asked of the vision model directly: asked
for "upbeat" or "a nature documentary", it wrote the same plain sentence
with "bustling" in it. The small model changes the voice for real -- and
the mood can then change without the picture being looked at again.

The video plays at its own pace on one thread; the commentary takes the
newest frame whenever it is ready for one. Frames between two comments are
shown and never looked at: this is a commentator, not a detector.

The line can be said aloud as well (`voices.py`): then a comment is handed
over with its speech, once the voice has it ready, and the next look at the
picture waits for the voice to have finished -- two lines are never spoken
over each other, and a slow voice slows the commentary rather than fall
behind it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterator

import cv2
import numpy as np
from pantherlake_ai_core import video

from . import moods, voices

SEE_PROMPT = "In one short sentence of at most 20 words, say what is happening in this picture. Only what can be seen."
# The frame as the vision model is shown it. Wider reads no better for this
# question and costs time: 0.8 s at 448, 0.9 at 672, 1.3 at 1280.
PICTURE_WIDTH = 672
EVERY_SECONDS = 4.0
# A picture that has not changed is not commented on again -- until this
# long has passed, when a still scene gets a fresh look all the same.
STILL_SECONDS = 20.0
_CHANGE = 4.0  # mean difference, on 0-255, between two thumbnails that counts as "something moved"


@dataclass(frozen=True)
class Comment:
    seen: str  # what the vision model said, plainly
    said: str  # the same in the mood chosen (equal to `seen` in the plain mood)
    mood: str
    seeing_seconds: float
    saying_seconds: float
    at: float  # wall-clock time it was ready
    voice: str = ""  # the voice it is spoken in ("" when it is only written)
    voicing_seconds: float = 0.0  # what the voice took to have it ready
    speech_seconds: float = 0.0  # how long it takes to say


def shown(frame: np.ndarray) -> np.ndarray:
    """The frame at the size the vision model is given."""
    if frame.shape[1] <= PICTURE_WIDTH:
        return frame
    height = int(frame.shape[0] * PICTURE_WIDTH / frame.shape[1])
    return cv2.resize(frame, (PICTURE_WIDTH, height), interpolation=cv2.INTER_AREA)


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def _tidy(line: str) -> str:
    """One sentence as a caption: the first line of the answer, without the
    quotation marks a model likes to put round a rewritten sentence."""
    first = next((part.strip() for part in line.splitlines() if part.strip()), "")
    return first.strip("\"'“” ")


class Commentator:
    """Decides, look after look, whether there is something new to say, and
    says it. The two models are handed in, so the rules can be tested
    without either: `see(frame) -> text`, `say(instruction, text) -> text`."""

    def __init__(
        self,
        see: Callable[[np.ndarray], str],
        say: Callable[[str, str], str],
        *,
        still_seconds: float = STILL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._see, self._say, self._still, self._clock = see, say, still_seconds, clock
        self._last_picture: np.ndarray | None = None
        self._last_seen = ""
        self._last_mood = ""
        self._last_at = float("-inf")

    @property
    def mood(self) -> str:
        """The mood of the last thing said ("" before anything was)."""
        return self._last_mood

    def revoice(self, mood_key: str) -> Comment | None:
        """The last thing seen, said again in another mood, without the
        picture being looked at: what a change of mood gets at once. None
        if nothing has been seen yet, or if that is the mood it was said in."""
        mood = moods.get(mood_key)
        if not self._last_seen or mood.key == self._last_mood:
            return None
        started = time.perf_counter()
        seen = self._last_seen
        said = seen if mood.instruction is None else (_tidy(self._say(mood.instruction, seen)) or seen)
        self._last_mood = mood.key
        return Comment(seen=seen, said=said, mood=mood.key, seeing_seconds=0.0,
                       saying_seconds=time.perf_counter() - started, at=time.time())

    def consider(self, frame: np.ndarray, mood_key: str) -> Comment | None:
        """One look at the newest frame. None when there is nothing new to
        say: the picture has not moved, or it has and the plain sentence
        comes out the same as last time."""
        mood = moods.get(mood_key)
        now = self._clock()
        picture = _thumbnail(frame)
        moved = self._last_picture is None or float(np.abs(picture - self._last_picture).mean()) >= _CHANGE
        if not moved and mood.key == self._last_mood and now - self._last_at < self._still:
            return None

        started = time.perf_counter()
        # A mood that changed over a picture that did not: the same sentence,
        # said the new way, without the picture being looked at again.
        only_the_voice = not moved and bool(self._last_seen) and mood.key != self._last_mood
        seen = self._last_seen if only_the_voice else _tidy(self._see(shown(frame)))
        seeing = time.perf_counter() - started
        if not seen:
            return None
        if seen == self._last_seen and mood.key == self._last_mood:
            self._last_picture, self._last_at = picture, now  # looked, and it is the same thing happening
            return None

        started = time.perf_counter()
        said = seen if mood.instruction is None else (_tidy(self._say(mood.instruction, seen)) or seen)
        saying = time.perf_counter() - started
        self._last_picture, self._last_seen, self._last_mood, self._last_at = picture, seen, mood.key, now
        return Comment(seen=seen, said=said, mood=mood.key, seeing_seconds=seeing, saying_seconds=saying, at=time.time())


class Voicing:
    """Gives a comment its speech, when a voice is on.

    `voice()` is asked for every line, so it can be switched on, off or
    changed while the video plays. A voice that fails is said once and not
    tried again until another is chosen: the commentary goes on in writing.
    `quiet_after` is when the line being spoken will have been said (on
    `clock`), which the next look at the picture waits for."""

    def __init__(
        self,
        speaker: voices.Speaker,
        voice: Callable[[], str],
        *,
        on_work: Callable[[str, bool, object], None] = lambda stage, working, stats: None,
        on_failed: Callable[[str], None] = lambda message: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._speaker, self._voice, self._on_work, self._on_failed, self._clock = speaker, voice, on_work, on_failed, clock
        self._failed = voices.OFF
        self.quiet_after = 0.0

    def __call__(self, comment: Comment) -> tuple[Comment, voices.Speech | None]:
        key = voices.get(self._voice())
        if key != self._failed:
            self._failed = voices.OFF
        if not key or key == self._failed:
            return comment, None
        self._on_work("voice", True, None)
        started = time.perf_counter()
        try:
            audio, rate = self._speaker.speak(key, comment.said, comment.mood)
        except Exception as exc:
            self._failed = key
            self._on_failed(str(exc))
            return comment, None
        finally:
            self._on_work("voice", False, None)
        seconds = len(audio) / rate if rate else 0.0
        self.quiet_after = self._clock() + seconds
        spoken = replace(comment, voice=key, voicing_seconds=time.perf_counter() - started, speech_seconds=seconds,
                         at=time.time())
        return spoken, (audio, rate)


def frames_from(
    source: str, *, path: str = "", camera_index: int = 0, screen_index: int = 1, loop: bool = True,
    stop_event: threading.Event | None = None,
) -> Iterator[np.ndarray]:
    if source == "file":
        if not path:
            raise ValueError("A video file is needed: give its path.")
        return video.stream_video_file_frames(path, loop=loop, stop_event=stop_event)
    if source == "webcam":
        return video.stream_camera_frames(camera_index, stop_event=stop_event)
    if source == "screen":
        return video.stream_screen_frames(screen_index, stop_event=stop_event)
    raise ValueError(f"Unknown source '{source}': one of file, webcam, screen.")


def run(
    *,
    source: str,
    path: str = "",
    camera_index: int = 0,
    screen_index: int = 1,
    loop: bool = True,
    vision_device: str,
    mood_device: str,
    mood: Callable[[], str],
    voice: Callable[[], str] = lambda: voices.OFF,
    clone: Callable[[str], voices.Speech] | None = None,
    every: float = EVERY_SECONDS,
    on_frame: Callable[[np.ndarray], None] = lambda frame: None,
    on_comment: Callable[[Comment, voices.Speech | None], None] = lambda comment, speech: None,
    on_work: Callable[[str, bool, object], None] = lambda stage, working, stats: None,
    on_ready: Callable[[], None] = lambda: None,
    on_downloading: Callable[[], None] | None = None,
    on_voice_failed: Callable[[str], None] = lambda message: None,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocks until the video ends or `stop_event` is set.

    `mood()` is asked at every look, so the voice can change while the
    video plays. `on_work(stage, working, stats)` says when the vision
    model ("vision"), the language model ("mood") and the voice ("voice")
    start and finish a piece of work, with how fast the finished one went.

    `voice()` is asked for every line: "" and the line is only written,
    `voices.STUDIO` or `voices.CLONED` and `on_comment` is handed its speech
    with it -- the samples and their rate, for whoever is listening to play.
    `clone(text)` is the cloned voice, lent by whoever enrolled one. A voice
    that fails is said once (`on_voice_failed`) and the commentary goes on
    in writing: it is the line that matters."""
    from doc_qa.engine_factory import create_llm
    from pantherlake_ai_core.engine import Engine
    from screen_ocr.extractor_openvino import OpenVINOExtractor

    asked_to_stop = stop_event or threading.Event()
    # The player's own signal: set when this function leaves, however it
    # leaves, so that a video is never left playing to nobody.
    stop = threading.Event()
    moods.get(mood())  # an unknown mood is refused before anything loads
    frames = frames_from(source, path=path, camera_index=camera_index, screen_index=screen_index, loop=loop, stop_event=stop)
    eyes = OpenVINOExtractor(device=vision_device, on_downloading=on_downloading)
    wording = create_llm(Engine.OPENVINO, device=mood_device, on_downloading=on_downloading)

    def see(picture: np.ndarray) -> str:
        on_work("vision", True, None)
        try:
            text, stats = eyes.ask(picture, SEE_PROMPT, max_new_tokens=40)
        except Exception:
            on_work("vision", False, None)
            raise
        on_work("vision", False, stats)
        return text

    def say(instruction: str, line: str) -> str:
        on_work("mood", True, None)
        try:
            # The likeliest words, not drawn ones: drawn, the same model put a
            # herd under a clear sky "at night", called its rider Buffalo Bill
            # and had stars twinkle above. It still embroiders; less.
            text = wording.answer(instruction, line, max_tokens=48, sample=False)
        except Exception:
            on_work("mood", False, None)
            raise
        on_work("mood", False, getattr(wording, "last_stats", None))
        return text

    commentator = Commentator(see, say)
    voiced = Voicing(
        voices.Speaker(clone=clone, on_downloading=on_downloading), voice, on_work=on_work, on_failed=on_voice_failed,
    )
    newest: list[np.ndarray | None] = [None]
    ended = threading.Event()
    failed: list[BaseException] = []

    def play() -> None:
        try:
            for frame in frames:
                if stop.is_set():
                    break
                newest[0] = frame
                on_frame(frame)
        except BaseException as exc:  # the video could not be opened, or stopped being readable
            failed.append(exc)
        finally:
            ended.set()

    player = threading.Thread(target=play, daemon=True, name="video-commentary-play")
    player.start()
    ready = False
    try:
        while not asked_to_stop.is_set() and not ended.is_set():
            frame = newest[0]
            if frame is None:
                asked_to_stop.wait(0.05)
                continue
            if not ready:
                ready = True
                on_ready()
            started = time.monotonic()
            comment = commentator.consider(frame, mood())
            if comment is not None:
                on_comment(*voiced(comment))
            # Until the next look, a change of mood is answered at once: the
            # same sentence in the new voice, which only the small model has
            # to say again. And the next look waits for a line being spoken
            # to have been said.
            while not asked_to_stop.is_set() and not ended.is_set():
                if time.monotonic() >= max(started + max(every, 0.1), voiced.quiet_after):
                    break
                if commentator.mood and mood() != commentator.mood:
                    again = commentator.revoice(mood())
                    if again is not None:
                        on_comment(*voiced(again))
                asked_to_stop.wait(0.2)
    finally:
        stop.set()
        player.join(timeout=2.0)
    if failed:
        raise failed[0]
