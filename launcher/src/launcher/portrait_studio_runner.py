"""Holds the portrait-studio brick's image model and exposes its one
blocking call for the web UI to run off the event loop.

The photo comes with the request and goes back out as a picture: nothing of
it is kept here between two requests, and nothing is written to disk. The
page holds the photo and sends it again with each style.
"""
from __future__ import annotations

import base64
import io
import threading

from pantherlake_ai_core.engine import Engine
from portrait_studio import photo as photo_kit
from portrait_studio import styles
from portrait_studio.stylist import Stylist

from . import activity, energy, events, generation, metrics
from .errors import Conflict

_DEMO_ID = "portrait-studio"
_ENGINE = Engine.OPENVINO.value
_JPEG_QUALITY = 92


class PortraitStudioRunner:
    def __init__(self) -> None:
        # The names `loaded.info` / `loaded.unload` look for.
        self._session: Stylist | None = None
        self._engine: str | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def _use(self, device: str) -> Stylist:
        if self._session is None or self._device != device:
            self._session = None  # the old one goes before the new one loads: both would not fit a small GPU twice
            self._session = Stylist(
                device,
                on_downloading=lambda: events.set_phase(
                    _DEMO_ID, "loading", "Downloading the image model (first run only, about 4.6 GB)..."),
                on_preparing=lambda: events.set_phase(
                    _DEMO_ID, "loading", "Preparing the image model (once, a few seconds)..."),
            )
            self._engine, self._device = _ENGINE, device
        return self._session

    def restyle(self, *, device: str, data: bytes, style_id: str, size: int, seed: int) -> dict:
        """Blocking. Draws the person in the photo `data` in the style
        asked for; loads the model the first time, or when the chip changes.
        What cannot be drawn is refused before the model is waited for."""
        style = styles.get(style_id)
        picture = photo_kit.square(photo_kit.read(data), size)
        if not self._lock.acquire(blocking=False):
            raise Conflict("A portrait is being drawn -- wait for it, or stop it first.")
        try:
            activity.set_active(_DEMO_ID, engine=_ENGINE, device=device)
            events.set_phase(_DEMO_ID, "loading", f"Loading the image model on {device}...")
            live = generation.get(_DEMO_ID)
            control = live.begin()
            try:
                stylist = self._use(device)
                events.set_phase(_DEMO_ID, "running", f"Drawing: {style.name}...")
                # The picture's energy window opens once the model is loaded:
                # loading it is not what a picture costs.
                started = energy.mark()
                drawn = stylist.restyle(
                    picture, style, size=size, seed=seed,
                    on_step=lambda done, of: events.set_phase(_DEMO_ID, "running", f"Drawing: {style.name}, step {done} of {of}..."),
                    stopped=control.should_stop,
                )
            except Exception as exc:
                events.set_phase(_DEMO_ID, "error", str(exc))
                raise
            finally:
                live.end()
            spent = energy.since(started, _DEMO_ID)
            if drawn.steps_per_second:
                metrics.report(_DEMO_ID, drawn.steps_per_second, "steps/s", sticky=True)
            events.clear_phase(_DEMO_ID)
            return {
                "style": style.id, "style_name": style.name, "size": drawn.size, "seed": drawn.seed,
                "image": None if drawn.image is None else _data_url(drawn.image),
                "seconds": round(drawn.seconds, 2), "steps": drawn.steps,
                "step_seconds": [round(step, 2) for step in drawn.step_seconds],
                "steps_per_second": round(drawn.steps_per_second, 2) if drawn.steps_per_second else None,
                "cancelled": drawn.cancelled, "device": stylist.device, "energy": spent,
            }
        finally:
            activity.clear_active(_DEMO_ID)
            self._lock.release()


def _data_url(image) -> str:
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=_JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


def photo_bytes(data_url: str) -> bytes:
    """The picture inside what a page sends: `data:image/...;base64,....`.
    ValueError if it is not that."""
    head, comma, body = (data_url or "").partition(",")
    if not comma or not head.startswith("data:image/") or not head.endswith(";base64"):
        raise ValueError("The photo did not arrive as a picture: take it or choose it again.")
    try:
        return base64.b64decode(body, validate=True)
    except ValueError as exc:
        raise ValueError("The photo did not arrive whole: take it or choose it again.") from exc
