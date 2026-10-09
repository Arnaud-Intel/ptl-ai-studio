"""Runs the video-commentary brick on a background thread: the video as an
MJPEG stream (only the newest frame matters, as for object detection), and
the comments as a short list the page asks for.

Two models on two chips, so two rows in the hardware panel: the vision
model under the chip that sees ("vision"), the language model under the one
that gives the line its mood ("mood"), each with how fast its last line
went. The mood can be changed while the video plays.

The line can be said aloud. The speech is made here and played by the page:
each spoken comment keeps its sound for the page to ask for, the way the
Voice Clone Studio hands its own to the page. A third row then shows the
chip the voice is made on ("voice"). The cloned voice is not the
commentator's: it is the one enrolled in the Voice Clone Studio, lent.
"""
from __future__ import annotations

import io
import threading
from collections import deque
from dataclasses import asdict
from typing import Protocol

import cv2
import numpy as np
from pantherlake_ai_core.engine import Engine
from video_commentary import moods, pipeline, voices

from . import activity, events, metrics, worker

_DEMO_ID = "video-commentary"
_ENGINE = Engine.OPENVINO.value
_JPEG_QUALITY = 80
# The picture as the page shows it: wide enough for a panel, small enough
# that encoding thirty of them a second is not the busiest thing here.
_STREAM_WIDTH = 960
_KEPT = 30  # comments the page can still ask for
_SPOKEN_KEPT = 6  # of which this many still have their sound
VISION, MOOD, VOICE = "vision", "mood", "voice"
_STAGES = (VISION, MOOD)
_LABELS = {VISION: "Sees", MOOD: "Says", VOICE: "Speaks"}
_NO_CLONE = "No voice has been enrolled yet: enrol one in Voice Clone Studio, and the commentator can speak with it."


class ClonedVoice(Protocol):
    """What the Voice Clone Studio's runner lends: whether a voice is
    enrolled there, on which chip it is made, and a line said in it."""

    @property
    def enrolled(self) -> bool: ...
    @property
    def device(self) -> str | None: ...
    def speak(self, text: str): ...


def _wav(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()


class VideoCommentaryRunner:
    def __init__(self, cloned: ClonedVoice | None = None) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._comments: deque[dict] = deque(maxlen=_KEPT)
        self._speech: dict[int, bytes] = {}
        self._count = 0
        self._mood = moods.DEFAULT
        self._voice = voices.OFF
        self._cloned = cloned
        self._devices: dict[str, str] = {}
        self.error: str | None = None
        self.notice: str | None = None  # a voice that failed: the commentary went on in writing

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def mood(self) -> str:
        return self._mood

    def set_mood(self, key: str) -> str:
        """The voice of the next comment on. An unknown one is a ValueError."""
        self._mood = moods.get(key).key
        return self._mood

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def clone_ready(self) -> bool:
        return bool(self._cloned is not None and self._cloned.enrolled)

    def voice_device(self, key: str) -> str:
        """The chip a voice is made on."""
        if key == voices.CLONED:
            return (self._cloned.device if self._cloned is not None else None) or "CPU"
        return voices.STUDIO_DEVICE

    def set_voice(self, key: str | None) -> str:
        """The voice of the next line on: "" for silence. An unknown one is
        a ValueError, and so is the cloned voice before one is enrolled."""
        key = voices.get(key)
        if key == voices.CLONED and not self.clone_ready:
            raise ValueError(_NO_CLONE)
        self._voice = key
        self.notice = None
        events.clear_phase(_DEMO_ID, stage=VOICE)
        if self.running:
            self._show_voice()
        return self._voice

    def _show_voice(self) -> None:
        """The voice's row in the hardware panel: there while a voice is on."""
        if self._voice:
            activity.set_active(_DEMO_ID, engine=_ENGINE, device=self.voice_device(self._voice), stage=VOICE, stage_label=_LABELS[VOICE])
        else:
            activity.clear_active(_DEMO_ID, stage=VOICE)

    def start(
        self, *, source: str, path: str, camera_index: int, screen_index: int, loop: bool,
        vision_device: str, mood_device: str, mood: str, every: float, voice: str | None = None,
        frames=None, people: bool = False,
    ) -> None:
        worker.refuse_if_busy(_DEMO_ID, self._thread, self._stop_event)
        self.set_mood(mood)
        if voice is not None:
            self.set_voice(voice)
        elif self._voice == voices.CLONED and not self.clone_ready:
            self._voice = voices.OFF  # kept from an earlier run, and the voice is no longer enrolled
        if source == "file" and not path.strip() and frames is None:
            raise ValueError("Choose a video file, or one of the samples.")
        self.error = None
        self.notice = None
        events.clear_phase(_DEMO_ID, stage=VOICE)
        with self._lock:
            self._latest_jpeg = None
            self._comments.clear()
            self._speech.clear()
            self._count = 0
        self._devices = {VISION: vision_device, MOOD: mood_device}
        self._stop_event = threading.Event()
        stop_event = self._stop_event

        def on_frame(frame: np.ndarray) -> None:
            if frame.shape[1] > _STREAM_WIDTH:
                height = int(frame.shape[0] * _STREAM_WIDTH / frame.shape[1])
                frame = cv2.resize(frame, (_STREAM_WIDTH, height), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

        def on_comment(comment: pipeline.Comment, speech) -> None:
            if stop_event.is_set():
                return  # a line a slow voice finished after Stop: nobody is there to hear it
            sound = _wav(*speech) if speech is not None else None
            with self._lock:
                self._count += 1
                self._comments.append({"number": self._count, **asdict(comment), "speech": sound is not None})
                if sound is not None:
                    self._speech[self._count] = sound
                    for number in sorted(self._speech)[:-_SPOKEN_KEPT]:
                        del self._speech[number]
            if sound is not None and comment.voicing_seconds > 0:
                # Seconds of speech made per second of work, as the Voice Clone Studio reports its own.
                metrics.report(_DEMO_ID, comment.speech_seconds / comment.voicing_seconds, "x real time", stage=VOICE)

        def clone(text: str):
            if not self.clone_ready:
                raise RuntimeError(_NO_CLONE)
            return self._cloned.speak(text)

        def on_voice_failed(message: str) -> None:
            self.notice = f"The voice could not speak, and the commentary goes on in writing: {message}"
            events.set_phase(_DEMO_ID, "error", self.notice, stage=VOICE)  # in the activity log, too

        def on_work(stage: str, working: bool, stats) -> None:
            if stats is not None and getattr(stats, "tokens_per_second", 0):
                # The figure of the line just written: it stays on the chip's
                # row between two lines, which is most of the time. A live
                # figure, not a "last result": the stage is at work for as
                # long as the video plays, and the Auto Demo's stage shows a
                # chip at work its live figure and nothing else.
                metrics.report(_DEMO_ID, stats.tokens_per_second, "tok/s", stage=stage)

        def on_ready() -> None:
            events.set_phase(_DEMO_ID, "running", "Watching and commenting...")

        def on_downloading() -> None:
            events.set_phase(_DEMO_ID, "loading", "Downloading a model (first run only)...")

        def target() -> None:
            for stage in _STAGES:
                activity.set_active(_DEMO_ID, engine=_ENGINE, device=self._devices[stage], stage=stage, stage_label=_LABELS[stage])
            self._show_voice()
            events.set_phase(
                _DEMO_ID, "loading", f"Loading the vision model on {vision_device} and the language model on {mood_device}..."
            )
            try:
                pipeline.run(
                    source=source, path=path.strip(), camera_index=camera_index, screen_index=screen_index, loop=loop,
                    vision_device=vision_device, mood_device=mood_device, mood=lambda: self._mood, every=every,
                    voice=lambda: self._voice, clone=clone, frames=frames, people=people,
                    on_frame=on_frame, on_comment=on_comment, on_work=on_work, on_ready=on_ready,
                    on_downloading=on_downloading, on_voice_failed=on_voice_failed, stop_event=stop_event,
                )
            except Exception as exc:  # surfaced to the UI, not silently dropped
                self.error = str(exc)
                events.set_phase(_DEMO_ID, "error", str(exc))
            else:
                events.clear_phase(_DEMO_ID)
            finally:
                for stage in (*_STAGES, VOICE):
                    activity.clear_active(_DEMO_ID, stage=stage)
                metrics.clear(_DEMO_ID)

        self._thread = threading.Thread(target=target, daemon=True, name="video-commentary")
        self._thread.start()

    def stop(self) -> None:
        if not worker.request_stop(_DEMO_ID, self._thread, self._stop_event):
            return
        self._thread = None
        with self._lock:
            self._latest_jpeg = None
            self._speech.clear()

    def speech(self, number: int) -> bytes | None:
        """The sound of one spoken comment (a WAV file), while it is kept."""
        with self._lock:
            return self._speech.get(number)

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def comments(self, after: int = 0) -> list[dict]:
        """The comments numbered above `after`, oldest first."""
        with self._lock:
            return [comment for comment in self._comments if comment["number"] > after]

    def state(self) -> dict:
        return {
            "running": self.running, "mood": self._mood, "voice": self._voice, "error": self.error, "notice": self.notice,
            "devices": dict(self._devices) if self.running else {},
            # Whether it has said anything yet: what a scene of the Auto Demo waits for.
            "commenting": self.running and self._count > 0,
            "clone_ready": self.clone_ready,
        }
