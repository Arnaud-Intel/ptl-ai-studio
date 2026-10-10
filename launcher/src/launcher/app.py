"""FastAPI app: serves the Panther Lake AI Studio UI and drives every
available brick through one runner each (see the *_runner.py modules).

Run with `uv run panther-lake-launcher` from the workspace root
(`--host`, `--port`, `--no-browser` to taste).

Conventions every route follows, so the front end can treat them alike:

- Engine/device selection goes through `resolve()` -- the same rule the
  CLIs use (an explicit engine, else the best available; the engine's
  default device unless one is given), so the UI and the command line
  agree on what "no choice" means.
- Errors are `{"error": message}` under one status policy (see
  `error_response`): 400 for a bad input, 409 when the brick isn't in a
  state to do that, 500 for a failure inside a model -- always carrying
  the message, so the UI can show the real reason.
- A live video feed's `/stream` is an MJPEG response that 404s when the
  brick isn't running (`mjpeg_stream`); a message stream's `/ws/<id>`
  socket drains that brick's queue (`ws_drain`).
- `GET /api/<id>/devices` is one registry-driven route: each demo declares
  which hardware lists (and which samples) its controls need.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import io
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import uuid
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import uvicorn
from fastapi import FastAPI, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from live_translation.languages import SPOKEN_LANGUAGES, spoken_language
from page_agent import conductor as page_agent_conductor
from pantherlake_ai_core import audio, npu, video
from pantherlake_ai_core.engine import (
    Engine,
    default_device,
    list_gpu_devices,
    list_openvino_devices,
    preferred_device,
    preferred_large_model_device,
    preferred_realtime_vision_device,
    resolve_engine,
)
from pydantic import BaseModel, Field
from pantherlake_ai_core import sample_videos
from object_detection.pipeline import SOURCES as object_detection_sources
from smart_city_monitor import sources as smart_city_sources
from smart_city_monitor.types import COUNTING as SMART_CITY_COUNTING
from smart_city_monitor.types import FeedSpec as SmartCityFeedSpec
from voice_clone_studio import engine_factory as voice_clone_models

from . import activity, autodemo, autodemo_scenes, events, generation, loaded, metrics, registry, updates
from . import demo_assets
from pantherlake_ai_core.demo_samples import SAMPLE_ROOT
from .code_review_assist_runner import CodeReviewAssistRunner
from .doc_qa_runner import DocQARunner
from .errors import Conflict
from .expense_extract_runner import ExpenseExtractRunner
from .html_creator_runner import HtmlCreatorRunner
from .page_agent_runner import PageAgentRunner
from .live_translation_runner import LiveTranslationRunner
from .model_routes import router as model_router
from .meeting_notes_runner import MeetingNotesRunner
from .object_detection_runner import ObjectDetectionRunner
from .screen_ocr_runner import ScreenOcrRunner
from .smart_city_monitor_runner import SmartCityMonitorRunner
from .smart_recall_runner import SmartRecallRunner
from .telemetry_poller import TelemetryPoller
from .voice_assistant_runner import VoiceAssistantRunner
from .voice_clone_studio_runner import VoiceCloneStudioRunner
from .video_commentary_runner import VideoCommentaryRunner
from .webcam_effects_runner import WebcamEffectsRunner

STATIC_DIR = Path(__file__).parent / "static"
VERSION_FILE = Path(__file__).resolve().parents[3] / "VERSION"

# Whisper size defaults. The portable engine (faster-whisper) is comfortable
# with "small" on CPU. On OpenVINO the voice assistant, which hears a short
# question in one language, keeps "base"; translating speech is another job.
_WHISPER_SIZE_DEFAULTS = {Engine.PORTABLE: "small", Engine.OPENVINO: "base"}
# Live translation and meeting notes turn any language into English, and
# there "base" is not enough: on 14 French sentences (FLEURS, 149 s,
# 2026-10-07, XPS 14) its English scored 45 chrF against the sentences'
# English originals, "medium" 66 -- the difference between a gist and a
# translation -- while still running 17 times faster than real time on the
# NPU and 22 on the integrated GPU ("base": 100). "large-v3" adds nothing
# measurable (66) at 10 times real time, and on the NPU it names the wrong
# language for what it hears.
_TRANSLATION_SIZE_DEFAULTS = {Engine.PORTABLE: "small", Engine.OPENVINO: "medium"}


def read_version_file() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


# Snapshotted at import, deliberately. Read fresh on every request, this
# reports whatever is on disk -- which after a `git pull` is the version you
# have *not* started yet, so a launcher left running for hours cheerfully
# claims to be the newest code while serving the oldest. The static files
# make that worse by being read per request: the UI updates on reload, the
# Python behind it does not, and the two disagree silently.
RUNNING_VERSION = read_version_file()


def get_version() -> str:
    """The version this process actually started with."""
    return RUNNING_VERSION


# --- shared plumbing --------------------------------------------------------


def resolve(
    engine: str | None,
    device: str | None,
    *,
    large_model: bool = False,
    realtime_vision: bool = False,
    prefer_npu: bool = False,
) -> tuple[Engine, str]:
    """Engine + device for a request: `engine` if given (an unknown name is a
    ValueError, i.e. a 400), else the best available; `device` if given and
    available.

    A device left to the app -- nothing chosen, or "Auto" -- is resolved here
    to a real chip, never passed on as OpenVINO's "AUTO": the integrated GPU
    by default; for a brick with a large model (`large_model`) the fastest
    GPU, discrete if there is one; for a small model on live video
    (`realtime_vision`) the integrated GPU; for the work this app keeps on
    the NPU (`prefer_npu`: speech and meeting notes) the NPU, when the
    machine has one in working order. A brick that ran on "AUTO"
    reported "AUTO" as its device and so showed up under no chip at all,
    which is the one thing this app exists to show (see
    pantherlake_ai_core.engine.preferred_device)."""
    resolved = resolve_engine(engine)
    if resolved != Engine.OPENVINO:
        if not device:
            return resolved, default_device(resolved)
        if device.lower() != "cpu" and not (device.lower() == "cuda" or device.lower().startswith("cuda:")):
            raise ValueError("Portable engines support cpu or a compatible CUDA device, not OpenVINO device IDs")
        return resolved, device

    available = list_openvino_devices()
    asked = (device or "AUTO").upper()
    if npu.is_npu(asked) and npu.lost():
        # Windows reset the NPU earlier in this session and it is out of use
        # until the app restarts (pantherlake_ai_core.npu). A brick still set
        # to it runs where its models would have moved to; the hardware
        # panel says so under the NPU.
        return resolved, npu.fallback_device()
    if asked != "AUTO":
        if not available:
            raise ValueError("No OpenVINO devices are available; choose the portable engine")
        if asked not in available and not (asked == "GPU" and any(d.startswith("GPU.") for d in available)):
            raise ValueError(f"OpenVINO device {device!r} is unavailable; choose from {', '.join(available)}")
        return resolved, asked
    if prefer_npu and not npu.lost() and any(npu.is_npu(d) for d in available):
        return resolved, "NPU"
    if large_model:
        picked = preferred_large_model_device()
    elif realtime_vision:
        picked = preferred_realtime_vision_device()
    else:
        picked = preferred_device()
    # Those two answer "AUTO" on a machine with no GPU; here that means the CPU.
    return resolved, "CPU" if picked == "AUTO" else picked


def _was_stopped(result) -> bool:
    """True if an answer was stopped part-way: said by the result itself
    when it spans several model calls, else by its one call's stats."""
    stats = getattr(result, "stats", None)
    return bool(getattr(result, "cancelled", False) or (stats is not None and stats.cancelled))


def error_response(exc: BaseException) -> JSONResponse:
    """One error policy: what the caller sent was wrong (400), the brick
    isn't in a state to do that (409), or something failed while doing it
    (500). Every branch carries the message so the UI can show the real
    reason instead of a bare status code."""
    if isinstance(exc, Conflict):
        status = 409
    elif isinstance(exc, (ValueError, FileNotFoundError, NotADirectoryError)):
        status = 400
    else:
        status = 500
    return JSONResponse({"error": str(exc)}, status_code=status)


def mjpeg_stream(latest_jpeg: Callable[[], bytes | None], is_running: Callable[[], bool]) -> Response:
    """A multipart MJPEG response fed from a runner's single "latest frame"
    buffer -- polled, not queued: for video only the newest frame matters,
    so there's nothing to gain from buffering ones the client hasn't seen.
    404 when the brick isn't running, so an <img> doesn't hang on a stream
    that will never produce a frame."""
    if not is_running():
        return JSONResponse({"error": "not running"}, status_code=404)

    async def frames() -> AsyncIterator[bytes]:
        last_sent = None
        while is_running():
            jpeg = latest_jpeg()
            if jpeg is not None and jpeg is not last_sent:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                last_sent = jpeg
            # 20ms, not 50: this poll is the ceiling on delivered frame
            # rate, and at 50ms it capped every video brick at 20fps no
            # matter how fast detection ran. Detection on an iGPU is ~8ms.
            await asyncio.sleep(0.02)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


async def ws_drain(websocket: WebSocket, queue: asyncio.Queue) -> None:
    """Forward a brick's queue to one connected socket until the client
    goes away. The disconnect is watched *concurrently* with waiting on
    the queue, so a closed tab is noticed at once rather than on the next
    message -- which a stale handler would otherwise swallow, losing it
    for the tab that reconnected. (One shared queue per brick: fine for
    this launcher's one-operator-one-tab use; a second tab on the same
    brick would only get every other message.)"""
    await websocket.accept()

    async def until_disconnect() -> None:
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass

    watcher = asyncio.create_task(until_disconnect())
    try:
        while not watcher.done():
            getter = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({getter, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if getter not in done:
                getter.cancel()  # asyncio.Queue leaves the item queued for the next consumer
                break
            try:
                await websocket.send_json(getter.result())
            except (WebSocketDisconnect, RuntimeError):
                break
    finally:
        watcher.cancel()


# --- app ----------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.live_translation_queue = asyncio.Queue()
    app.state.meeting_notes_queue = asyncio.Queue()
    app.state.voice_assistant_queue = asyncio.Queue()
    app.state.expense_extract_queue = asyncio.Queue()
    app.state.smart_recall_queue = asyncio.Queue()
    telemetry_poller.start()
    updates.check_in_background()
    yield
    telemetry_poller.stop()


app = FastAPI(title="Panther Lake AI Studio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/demo-assets", StaticFiles(directory=SAMPLE_ROOT, follow_symlink=False), name="demo-assets")
app.include_router(model_router)  # /api/models: what the demos need, and fetching it (R10)

live_translation_runner = LiveTranslationRunner()
doc_qa_runner = DocQARunner()
object_detection_runner = ObjectDetectionRunner()
smart_city_monitor_runner = SmartCityMonitorRunner()
screen_ocr_runner = ScreenOcrRunner()
meeting_notes_runner = MeetingNotesRunner()
webcam_effects_runner = WebcamEffectsRunner()
voice_clone_studio_runner = VoiceCloneStudioRunner()
# The commentator speaks, when asked to, with the voice enrolled in the Voice Clone Studio.
video_commentary_runner = VideoCommentaryRunner(cloned=voice_clone_studio_runner)
voice_assistant_runner = VoiceAssistantRunner()
expense_extract_runner = ExpenseExtractRunner()
smart_recall_runner = SmartRecallRunner()
code_review_assist_runner = CodeReviewAssistRunner()
html_creator_runner = HtmlCreatorRunner()
page_agent_runner = PageAgentRunner()
telemetry_poller = TelemetryPoller(is_idle=lambda: not activity.snapshot())


@app.exception_handler(RequestValidationError)
async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """A malformed body gets the same `{"error": ...}` shape as every other
    failure, not FastAPI's default 422 `detail` list the UI can't show."""
    problems = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'][1:]) or 'body'}: {err['msg']}" for err in exc.errors()
    )
    return JSONResponse({"error": f"invalid request: {problems}"}, status_code=400)


# A script or stylesheet the page loads: "/static/app.js", quotes included.
_PAGE_ASSET = re.compile(r'"/static/([\w.-]+\.(?:js|css))"')


def _stamped(match: re.Match) -> str:
    try:
        stamp = int((STATIC_DIR / match.group(1)).stat().st_mtime)
    except OSError:
        return match.group(0)
    return f'"/static/{match.group(1)}?v={stamp}"'


@app.get("/")
def index() -> HTMLResponse:
    """The page, with its script/stylesheet URLs stamped by their files'
    modification time -- so a browser that cached the previous version's
    app.js picks up the new one on a plain reload after an update, instead
    of running stale code against new markup. Every one the page loads, not
    a list of names: a script added later (expense-review.js was) otherwise
    stays cached while the code that calls it moves on."""
    html = _PAGE_ASSET.sub(_stamped, (STATIC_DIR / "index.html").read_text(encoding="utf-8"))
    # `no-store`, not `no-cache`: the page carries the asset stamps, so a
    # stored copy pins the whole UI to whatever app.js/style.css it was
    # built against. `no-cache` only asks for revalidation, and this
    # response has no ETag to revalidate against -- browsers were observed
    # serving an index.html nearly two hours stale, i.e. new markup running
    # old JS (or the reverse), which fails in confusing, partial ways. The
    # page is a few KB; re-fetching it every navigation costs nothing.
    return HTMLResponse(html, headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/api/demos")
def list_demos() -> JSONResponse:
    return JSONResponse([asdict(d) for d in registry.REGISTRY])


@app.get("/api/version")
def api_version() -> JSONResponse:
    """`version` is what's running; `on_disk` is what a restart would give
    you. When they differ the UI says so, because "am I running the latest?"
    is otherwise unanswerable from the page."""
    on_disk = read_version_file()
    return JSONResponse({
        "version": RUNNING_VERSION, "on_disk": on_disk, "restart_needed": on_disk != RUNNING_VERSION,
        # Which copy of the project this is: the setup assistant must not
        # take a Studio started from another folder for the one it installed.
        "root": str(updates.REPO_ROOT),
    })


@app.get("/api/telemetry")
def telemetry_snapshot() -> JSONResponse:
    """CPU/GPU/NPU utilization and processor power (from the background poller's cache -- see
    telemetry_poller.py for why this isn't queried fresh per request),
    plus which demo (if any) is currently driving each device."""
    payload = telemetry_poller.snapshot()
    # can_stop: a loop can be stopped, and so can an answer being written.
    # What can't is a one-shot brick still loading its model, or one that
    # doesn't generate text (voice cloning).
    payload["active"] = [
        {**entry, "can_stop": entry["demo_id"] in _STOPPABLE or generation.in_flight(entry["demo_id"])}
        for entry in activity.snapshot()
    ]
    # For the side panel: each brick's own number, and the bricks that are
    # idle but still holding a model on a chip.
    payload["metrics"] = metrics.snapshot()
    # Set once Windows has reset the NPU under this process: it stays out of
    # use until the app restarts, and the panel says where its work went.
    payload["npu_lost"] = {"at": npu.lost_at(), "moved_to": npu.fallback_device()} if npu.lost() else None
    payload["loaded"] = [
        {"demo_id": demo_id, **held}
        for demo_id, runner in _UNLOADABLE.items()
        # A runner that holds several models on several chips lists them itself.
        for held in (runner.held() if hasattr(runner, "held") else filter(None, [loaded.info(runner)]))
    ]
    return JSONResponse(payload)


# What the side panel's close button reaches. A brick with a loop is stopped;
# one that answers a request at a time has nothing to stop, so its model is
# unloaded instead -- which is what frees the chip's memory.
_STOPPABLE = {
    "live-translation": live_translation_runner,
    "object-detection": object_detection_runner,
    "smart-city-monitor": smart_city_monitor_runner,
    "meeting-notes": meeting_notes_runner,
    "webcam-effects": webcam_effects_runner,
    "voice-assistant": voice_assistant_runner,
    "expense-extract": expense_extract_runner,
    "smart-recall": smart_recall_runner,
    "video-commentary": video_commentary_runner,
}
_UNLOADABLE = {
    "doc-qa": doc_qa_runner,
    "screen-ocr": screen_ocr_runner,
    "voice-clone-studio": voice_clone_studio_runner,
    "code-review-assist": code_review_assist_runner,
    "html-creator": html_creator_runner,
    "page-agent": page_agent_runner,
}


@app.post("/api/bricks/{demo_id}/stop")
async def stop_brick(demo_id: str, stage: str | None = None) -> JSONResponse:
    """Stop a running brick, stop an answer being written, or unload an idle
    brick's model -- one route, so the panel doesn't need to know which kind
    each brick is. `stage` narrows it to one stage's answer: meeting notes'
    summary can be stopped without ending the transcription."""
    try:
        if stage and stage != "default" and generation.cancel(demo_id, stage):
            return JSONResponse({"status": "cancelling"})
        if demo_id in _STOPPABLE:
            generation.cancel(demo_id)  # so the stop doesn't wait for a summary to finish
            await run_in_threadpool(_STOPPABLE[demo_id].stop)
            return JSONResponse({"status": "stopped"})
        if demo_id in _UNLOADABLE:
            # Mid-answer, the close button stops the answer; the model stays
            # loaded, and a second press unloads it.
            if generation.cancel(demo_id):
                return JSONResponse({"status": "cancelling"})
            had_model = await run_in_threadpool(loaded.unload, _UNLOADABLE[demo_id], demo_id)
            return JSONResponse({"status": "unloaded" if had_model else "idle"})
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"error": f"No brick called '{demo_id}' can be stopped."}, status_code=404)


@app.get("/api/bricks/{demo_id}/partial")
def brick_partial(demo_id: str, stage: str = "default") -> JSONResponse:
    """The answer a brick is writing, so far. Asked a few times a second by
    the page while its request is pending (see generation.py for why it is
    polled rather than pushed)."""
    return JSONResponse(generation.snapshot(demo_id, stage))


@app.get("/api/status")
def status_snapshot() -> JSONResponse:
    """Per-demo lifecycle phase (loading/running/error) -- what's actually
    happening right now, for a UI indicator during a slow first-time model
    load. See events.py; a separate concern from /api/telemetry's activity
    (which device, for gauge labeling), not a replacement for it."""
    return JSONResponse(events.status_snapshot())


@app.get("/api/logs")
def logs(limit: int = 100) -> JSONResponse:
    """Recent lifecycle events (successes and errors) for the Activity Log
    viewer -- also persisted to logs/events.log at the repo root."""
    return JSONResponse(events.recent_events(limit))


@app.get("/api/system/gpu-devices")
def system_gpu_devices() -> JSONResponse:
    """Every OpenVINO-visible GPU on this machine, with a friendly name --
    machine-level (not per-brick), so the frontend fetches it once and uses
    it to label every brick's compute-device dropdown and to build one
    telemetry gauge per physical GPU."""
    return JSONResponse([{"id": gd.id, "full_name": gd.full_name} for gd in list_gpu_devices()])


# The bricks that answer with a small language model, and the model each
# takes when nobody chose (doc-qa's language_models has the two).
def _preferred_language_models() -> dict[str, str]:
    from doc_qa import pipeline as doc_qa_pipeline
    from expense_extract import pipeline as expense_pipeline
    from video_commentary import pipeline as commentary_pipeline
    from voice_assistant import session as voice_session

    return {
        "doc-qa": doc_qa_pipeline.PREFERRED_MODEL,
        "expense-extract": expense_pipeline.DEFAULT_MODEL,
        "voice-assistant": voice_session.DEFAULT_MODEL,
        "video-commentary": commentary_pipeline.DEFAULT_MOOD_MODEL,
    }


def _language_models(demo_id: str) -> dict[str, Any] | None:
    """What the brick's Model menu offers on the OpenVINO engine, or None
    for a brick that has no such choice. A model comes in two builds, one
    for the NPU and one for the other chips, so whether it is on this
    laptop -- and so which one "left to the app" means -- is said for each."""
    wanted = _preferred_language_models().get(demo_id)
    if wanted is None:
        return None
    from doc_qa import language_models

    kinds = {"npu": "NPU", "other": "GPU"}
    return {
        "models": [
            {
                "key": model.key, "name": model.name, "says": model.says,
                "on_disk": {kind: language_models.on_disk(model.key, Engine.OPENVINO, device) for kind, device in kinds.items()},
            }
            for model in language_models.choices(Engine.OPENVINO)
        ],
        "default": {kind: language_models.preferred(wanted, Engine.OPENVINO, device) for kind, device in kinds.items()},
        "portable": language_models.DEFAULT,
    }


def _language_model(key: str | None, engine: Engine, device: str) -> str | None:
    """`key` as a brick takes it: None when the choice is left to the brick,
    refused here (a ValueError, so a 400) when it names no model or one the
    engine does not have -- before a thread is started to find that out."""
    if not key:
        return None
    from doc_qa import language_models

    language_models.repo_for(key, engine, device)
    return language_models.get(key).key


def _wake_words() -> list[str]:
    from voice_assistant.wake_word import AVAILABLE_WAKE_WORDS

    return list(AVAILABLE_WAKE_WORDS)


# Looked up at call time (not bound here) so the probes can be swapped --
# the tests replace them with fakes; nothing should probe a camera on import.
_DEVICE_SOURCES: dict[str, Callable[[], Any]] = {
    "microphones": lambda: audio.list_microphones(),
    "speakers": lambda: audio.list_speakers(),
    "cameras": lambda: video.list_cameras(),
    "screens": lambda: video.list_screens(),
    "wake_words": _wake_words,
}


@app.get("/api/{demo_id}/devices")
def demo_devices(demo_id: str) -> JSONResponse:
    """What this demo's controls can pick from on this machine: the
    hardware lists its registry entry asks for, the OpenVINO devices, and
    its bundled samples (if any) -- one route for all thirteen bricks
    instead of one hand-written copy each."""
    demo = registry.get(demo_id)
    if demo is None or demo.status != "available":
        return JSONResponse({"error": f"unknown demo '{demo_id}'"}, status_code=404)
    payload: dict[str, Any] = {"openvino_devices": list_openvino_devices()}
    choice = _language_models(demo_id)
    if choice is not None:
        payload["language_models"] = choice
    if demo_id == "webcam-effects":
        from webcam_effects.capabilities import GPU_REASON

        payload["openvino_unsupported"] = {
            d: GPU_REASON for d in ["AUTO", *payload["openvino_devices"]] if d == "AUTO" or d.startswith("GPU")
        }
    if demo_id == "voice-clone-studio":
        from voice_clone_studio.voice_model import NPU_REASON, TTS_DEVICE

        # The voice models do not compile for the NPU, and trying took the
        # launcher down: it is shown greyed out with why. Left to the app,
        # the voice is made on the CPU, where it is fastest.
        payload["openvino_unsupported"] = {d: NPU_REASON for d in payload["openvino_devices"] if npu.is_npu(d)}
        if payload["openvino_devices"]:
            payload["auto_device"] = TTS_DEVICE
    if demo_id in ("live-translation", "meeting-notes"):
        # What the "Spoken language" menu offers after "Detect automatically".
        payload["spoken_languages"] = [{"code": code, "name": name} for code, name in SPOKEN_LANGUAGES.items()]
        if payload["openvino_devices"]:
            # What "Auto" means for these two, so the menu can say it: the NPU when there is one.
            payload["auto_device"] = resolve(Engine.OPENVINO.value, None, prefer_npu=True)[1]
    if demo_id == "video-commentary":
        from video_commentary import moods as commentary_moods

        from video_commentary import voices as commentary_voices

        payload["moods"] = [{"key": mood.key, "name": mood.name} for mood in commentary_moods.MOODS]
        payload["default_mood"] = commentary_moods.DEFAULT
        # The voices a line can be said aloud in, and whether each can be used now:
        # the cloned one is the voice enrolled in the Voice Clone Studio, if there is one.
        payload["voices"] = [
            {
                "key": voice.key, "name": voice.name,
                "ready": voice.key != commentary_voices.CLONED or video_commentary_runner.clone_ready,
                "device": video_commentary_runner.voice_device(voice.key),
            }
            for voice in commentary_voices.VOICES
        ]
        if payload["openvino_devices"]:
            # What "Auto" means for each of its two models, so the menus can say it.
            payload["auto_devices"] = dict(zip(("vision", "mood"), _video_commentary_devices(None, None)))
    if demo_id == "page-agent" and payload["openvino_devices"]:
        # Who does what when every chip is left to the conductor, so the menus can say it.
        payload["auto_assignment"] = asdict(_page_agent_assignment(None, None, None))
    for kind in demo.devices:
        payload[kind] = _DEVICE_SOURCES[kind]()
    if demo.samples:
        payload["samples"] = [demo_assets.enrich_sample(asdict(s)) for s in importlib.import_module(demo.samples).SAMPLES]
        for sample in payload["samples"]:
            # Asked now, not when the brick was imported: a sample whose video
            # is still to be fetched says so, until it has been.
            if sample.get("videos"):
                sample["ready"] = all(sample_videos.present(sample_videos.BY_KEY[key]) for key in sample["videos"])
    return JSONResponse(payload)


# --- live-translation -------------------------------------------------------------


class LiveTranslationStartRequest(BaseModel):
    source: str = "mic"
    audio_device: str | None = None
    engine: str | None = None
    model_size: str | None = None
    compute_device: str | None = None
    language: str | None = None  # the language being spoken; nothing or "auto" = detect it


@app.post("/api/live-translation/start")
async def start_live_translation(req: LiveTranslationStartRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device, prefer_npu=True)
        live_translation_runner.start(
            loop=asyncio.get_running_loop(),
            queue=app.state.live_translation_queue,
            source=req.source,
            audio_device=req.audio_device,
            engine=engine,
            model_size=req.model_size or _TRANSLATION_SIZE_DEFAULTS[engine],
            compute_device=device,
            language=spoken_language(req.language),
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started"})


@app.post("/api/live-translation/stop")
async def stop_live_translation() -> JSONResponse:
    live_translation_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.get("/api/live-translation/transcript")
def live_translation_transcript() -> JSONResponse:
    """Everything live translation has heard since its transcript was last
    cleared, across Stop and Start. The launcher keeps it, so a reloaded
    page gets its lines back and Meeting Notes can be handed the lot."""
    return JSONResponse(live_translation_runner.transcript())


@app.delete("/api/live-translation/transcript")
def clear_live_translation_transcript() -> JSONResponse:
    """Start a new transcript (the next meeting). Returns its `id`."""
    return JSONResponse({"id": live_translation_runner.clear_transcript()})


@app.websocket("/ws/live-translation")
async def ws_live_translation(websocket: WebSocket) -> None:
    await ws_drain(websocket, app.state.live_translation_queue)


# --- doc-qa -----------------------------------------------------------------------


class DocQAIngestRequest(BaseModel):
    folder: str
    engine: str | None = None
    compute_device: str | None = None
    reindex: bool = False
    # Which language model writes the answers: a key of doc-qa's
    # language_models, or nothing for the brick's own choice on that chip.
    model: str | None = None


@app.post("/api/doc-qa/ingest")
async def doc_qa_ingest(req: DocQAIngestRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device)
        count, folder = await run_in_threadpool(
            doc_qa_runner.ingest, folder=req.folder, engine=engine.value, device=device, reindex=req.reindex,
            model=_language_model(req.model, engine, device),
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({
        "chunks": count, "folder": folder, "files": doc_qa_runner.status()["files"], "model": doc_qa_runner.model,
    })


@app.get("/api/doc-qa/status")
def doc_qa_status() -> JSONResponse:
    """Whether an index is loaded (and from where) -- so reopening the panel
    or reloading the page picks up an index built earlier instead of
    asking to build it again."""
    return JSONResponse(doc_qa_runner.status())


class DocQAAskRequest(BaseModel):
    question: str
    top_k: int = 4
    # The question put to the language model with no document in the
    # conversation: what the documents change is what it answers then. No
    # folder need be indexed for it; `engine` and `compute_device` say what
    # to load when nothing is loaded yet.
    alone: bool = False
    engine: str | None = None
    compute_device: str | None = None
    # Another language model for this answer and the ones after it (the
    # index stays): a key of doc-qa's language_models. Nothing: as it is.
    model: str | None = None


@app.post("/api/doc-qa/ask")
async def doc_qa_ask(req: DocQAAskRequest) -> JSONResponse:
    try:
        if not req.question.strip():
            raise ValueError("Ask a question.")
        engine = device = None
        if req.alone and (req.engine or req.compute_device):
            resolved, device = resolve(req.engine, req.compute_device)
            engine = resolved.value
        model = None
        if req.model:
            from doc_qa import language_models

            # An unknown name stops here; whether the engine that is loaded
            # has the model is for the session to say.
            model = language_models.get(req.model).key
        answer = await run_in_threadpool(
            doc_qa_runner.ask, question=req.question, top_k=req.top_k, alone=req.alone, engine=engine, device=device,
            model=model,
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(
        {
            "alone": req.alone,
            "model": doc_qa_runner.model,
            "text": answer.text,
            "sources": [
                {"source": r.chunk.source, "chunk_index": r.chunk.chunk_index, "score": r.score}
                for r in answer.sources
            ],
            "stats": asdict(answer.stats) if answer.stats else None,
            "cancelled": _was_stopped(answer),
        }
    )


# --- object-detection -------------------------------------------------------------


class ObjectDetectionStartRequest(BaseModel):
    source: str = "screen"  # "webcam", "screen" or "file"
    camera_index: int = 0
    screen_index: int = 1
    path: str = ""  # the video, for "file"
    loop: bool = True
    engine: str | None = None
    compute_device: str | None = None


@app.post("/api/object-detection/start")
async def start_object_detection(req: ObjectDetectionStartRequest) -> JSONResponse:
    try:
        # Refused here, before a model is loaded for it: "started" is not an
        # answer to a source that does not exist.
        if req.source not in object_detection_sources:
            raise ValueError(f"unknown source '{req.source}': one of {', '.join(object_detection_sources)}")
        path = req.path.strip()
        if req.source == "file" and not path:
            raise ValueError("Choose a sample video, or give the path of a video file.")
        engine, device = resolve(req.engine, req.compute_device, realtime_vision=True)
        video_file = sample_videos.for_path(path) if req.source == "file" else None
        if video_file is not None and not sample_videos.present(video_file) and not object_detection_runner.running:
            await run_in_threadpool(_fetch_sample_videos, "object-detection", [(None, video_file)])
        object_detection_runner.start(
            source=req.source,
            camera_index=req.camera_index,
            screen_index=req.screen_index,
            path=path,
            loop=req.loop,
            engine=engine,
            compute_device=device,
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started"})


@app.post("/api/object-detection/stop")
async def stop_object_detection() -> JSONResponse:
    object_detection_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.get("/api/object-detection/detections")
def object_detection_detections() -> JSONResponse:
    detections = object_detection_runner.latest_detections()
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection["label"]] = counts.get(detection["label"], 0) + 1
    return JSONResponse({
        "detections": detections,
        # How many of each kind are in the picture now, most first.
        "counts": dict(sorted(counts.items(), key=lambda item: -item[1])),
        "running": object_detection_runner.running,
        # Whether it has a picture yet: what a demo that watches the same thing waits for.
        "watching": object_detection_runner.running and object_detection_runner.latest_jpeg() is not None,
        "error": object_detection_runner.error,
    })


@app.get("/api/object-detection/stream")
def object_detection_stream() -> Response:
    return mjpeg_stream(object_detection_runner.latest_jpeg, lambda: object_detection_runner.running)


# --- smart-city-monitor -----------------------------------------------------------


# Where a video dropped onto the UI lands. A browser can't tell a page the
# real path of a dropped file, so the only way "drag your clip here" can
# work at all is to copy the bytes over -- see the upload route below.
SMART_CITY_UPLOAD_DIR = Path.home() / ".cache" / "pantherlake-ai-studio" / "uploads"


class SmartCityFeedInput(BaseModel):
    path: str
    compute_device: str | None = None
    # Per feed, both optional: unset means "whatever the run was started
    # with". Set, they let one run mix backends -- one feed on the NPU with
    # YOLO11s, another on the CPU with DETR, at the same time.
    engine: str | None = None
    model_path: str | None = None
    # What to count in it: "street", "line" or "herd". Unset, one of the
    # sample videos is counted for what it shows and anything else as a street.
    counting: str | None = None


class SmartCityMonitorStartRequest(BaseModel):
    feeds: list[SmartCityFeedInput]
    engine: str | None = None
    compute_device: str | None = None
    loop: bool = True


def _fetch_sample_videos(demo_id: str, wanted: list[tuple[str | None, sample_videos.SampleVideo]]) -> None:
    """A demo pointed at one of the sample videos, on a machine that has not
    fetched it yet: fetched now, the way a model is at its first use, and
    said on the stage that waits for it (a feed, or the demo itself)."""
    for stage, video in wanted:
        megabytes = video.size_bytes / 1e6
        events.set_phase(demo_id, "loading", f"Downloading {video.name} (first use only, {megabytes:.0f} MB)...", stage=stage)
        try:
            sample_videos.download(video)
        finally:
            events.clear_phase(demo_id, stage=stage)


def _smart_city_feed_json(feed: SmartCityFeedSpec) -> dict[str, Any]:
    """What the UI needs to rebuild a feed's card after a reload: not just
    its name and chip, but the engine and source it was started with."""
    return {
        "feed_id": feed.feed_id,
        "name": feed.name,
        "compute_device": feed.compute_device,
        "engine": feed.engine.value if feed.engine else None,
        "model_path": feed.model_path,
        "path": feed.path,
        "source_type": "url" if smart_city_sources.is_url(feed.path) else "file",
        "counting": feed.counting,
    }


@app.post("/api/smart-city-monitor/start")
async def start_smart_city_monitor(req: SmartCityMonitorStartRequest) -> JSONResponse:
    try:
        if not req.feeds:
            raise ValueError("at least one feed is required")
        default_engine, default_device = resolve(req.engine, req.compute_device, realtime_vision=True)
        feeds = []
        for i, f in enumerate(req.feeds, start=1):
            if not f.path.strip():
                raise ValueError(f"feed {i} has no source -- give it a file path or a URL")
            # Resolved per feed, so an unknown engine name is a 400 naming
            # the feed rather than a failure deep inside the pipeline.
            feed_engine, feed_device = resolve(
                f.engine or req.engine, f.compute_device or req.compute_device, realtime_vision=True
            )
            known = sample_videos.for_path(f.path.strip())
            counting = f.counting or (known.counts if known else "street")
            if counting not in SMART_CITY_COUNTING:
                raise ValueError(f"feed {i}: unknown kind of counting '{counting}' -- one of {', '.join(SMART_CITY_COUNTING)}")
            feeds.append(
                SmartCityFeedSpec(
                    feed_id=f"feed-{i}",
                    path=f.path.strip(),
                    compute_device=feed_device,
                    name=smart_city_sources.display_name(f.path.strip()),
                    engine=feed_engine,
                    model_path=f.model_path or None,
                    counting=counting,
                )
            )
        missing = [
            (feed.feed_id, video) for feed in feeds
            if (video := sample_videos.for_path(feed.path)) is not None and not sample_videos.present(video)
        ]
        if missing:
            await run_in_threadpool(_fetch_sample_videos, "smart-city-monitor", missing)
        smart_city_monitor_runner.start(feeds=feeds, engine=default_engine, loop=req.loop)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started", "feeds": [_smart_city_feed_json(f) for f in feeds]})


@app.post("/api/smart-city-monitor/upload")
async def upload_smart_city_video(file: UploadFile) -> JSONResponse:
    """Stage a video dropped on the UI and hand back the path it landed at.

    Copied in chunks rather than read whole: these are video files, and
    `await file.read()` on a two-gigabyte clip would put all of it in
    memory. The copy is the price of drag-and-drop working at all -- typing
    a path into the feed's own box still reads the file where it already
    lives, with nothing duplicated.
    """
    try:
        if not file.filename:
            raise ValueError("the upload had no filename")
        # Only the basename, and prefixed: the client names this file, so it
        # must not be able to choose where it lands or overwrite a sibling.
        safe_name = Path(file.filename).name
        SMART_CITY_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        destination = SMART_CITY_UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}-{safe_name}"

        def save() -> int:
            with destination.open("wb") as out:
                shutil.copyfileobj(file.file, out, length=1024 * 1024)
            return destination.stat().st_size

        size = await run_in_threadpool(save)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"path": str(destination), "name": safe_name, "bytes": size})


@app.post("/api/smart-city-monitor/stop")
async def stop_smart_city_monitor() -> JSONResponse:
    smart_city_monitor_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.get("/api/smart-city-monitor/counts")
def smart_city_monitor_counts() -> JSONResponse:
    snapshot = smart_city_monitor_runner.latest_snapshot()
    return JSONResponse(
        {
            "snapshot": asdict(snapshot) if snapshot else None,
            # The feed list too, so a panel opened after the run started (or
            # after a reload) can rebuild every feed's card as it was set up.
            "feeds": [_smart_city_feed_json(f) for f in smart_city_monitor_runner.feeds()],
            "error": smart_city_monitor_runner.error,
            "health": smart_city_monitor_runner.health(),
            "running": smart_city_monitor_runner.running,
        }
    )


@app.get("/api/smart-city-monitor/stream")
def smart_city_monitor_stream(feed: str) -> Response:
    return mjpeg_stream(lambda: smart_city_monitor_runner.latest_jpeg(feed), lambda: smart_city_monitor_runner.running)


# --- video-commentary -------------------------------------------------------------


def _video_commentary_devices(vision: str | None, mood: str | None) -> tuple[str, str]:
    """Where its two models run: the vision model on a GPU -- the integrated
    one unless another is asked for, which is where a line was measured at
    0.9 s -- and the language model on the NPU when there is one. The vision
    model does not compile for the NPU on this hardware (screen-ocr's README)."""
    _, sees = resolve(Engine.OPENVINO.value, vision if vision and vision.upper() != "AUTO" else None)
    if npu.is_npu(sees):
        raise ValueError("The vision model does not run on the NPU: choose a GPU, or the CPU, for it.")
    _, says = resolve(Engine.OPENVINO.value, mood if mood and mood.upper() != "AUTO" else None, prefer_npu=True)
    return sees, says


class VideoCommentaryStartRequest(BaseModel):
    # "file", "webcam", "screen" -- or "detector": what Object Detection is
    # watching, which has to be running. A camera can be opened by one demo
    # only; this is how the two watch the same one.
    source: str = "file"
    # Whether people are likely in the picture: the vision model is then
    # asked what they are doing and never what they look like. Not said:
    # yes for a camera, no for the rest.
    people: bool | None = None
    path: str = ""
    camera_index: int = 0
    screen_index: int = 1
    loop: bool = True
    vision_device: str | None = None
    mood_device: str | None = None
    mood: str | None = None
    # The line said aloud: "studio", "cloned" (the voice enrolled in the Voice
    # Clone Studio), or "" for silence. Not said: as it was last left.
    voice: str | None = None
    every: float = 4.0  # seconds between two looks at the picture
    # Which language model says the line in a mood: a key of doc-qa's language_models.
    mood_model: str | None = None


class VideoCommentaryMoodRequest(BaseModel):
    mood: str


class VideoCommentaryVoiceRequest(BaseModel):
    voice: str = ""


@app.post("/api/video-commentary/start")
async def start_video_commentary(req: VideoCommentaryStartRequest) -> JSONResponse:
    try:
        if req.source not in ("file", "webcam", "screen", "detector"):
            raise ValueError(f"unknown source '{req.source}': one of file, webcam, screen, detector")
        frames, watched = None, req.source
        if req.source == "detector":
            if not object_detection_runner.running:
                raise Conflict("Object Detection is not running: start it first, and the commentator watches what it watches.")
            frames, watched = object_detection_runner.frames, object_detection_runner.source
        people = req.people if req.people is not None else watched == "webcam"
        sees, says = _video_commentary_devices(req.vision_device, req.mood_device)
        path = req.path.strip()
        video_file = sample_videos.for_path(path) if req.source == "file" and path else None
        if video_file is not None and not sample_videos.present(video_file) and not video_commentary_runner.running:
            await run_in_threadpool(_fetch_sample_videos, "video-commentary", [(None, video_file)])
        video_commentary_runner.start(
            source=req.source, path=path, camera_index=req.camera_index, screen_index=req.screen_index, loop=req.loop,
            vision_device=sees, mood_device=says, mood=req.mood or video_commentary_runner.mood,
            every=min(max(req.every, 2.0), 30.0), voice=req.voice, frames=frames, people=people,
            mood_model=_language_model(req.mood_model, Engine.OPENVINO, says),
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({
        "status": "started", "devices": {"vision": sees, "mood": says},
        "mood": video_commentary_runner.mood, "voice": video_commentary_runner.voice,
    })


@app.post("/api/video-commentary/stop")
async def stop_video_commentary() -> JSONResponse:
    await run_in_threadpool(video_commentary_runner.stop)
    return JSONResponse({"status": "stopped"})


@app.post("/api/video-commentary/mood")
def video_commentary_mood(req: VideoCommentaryMoodRequest) -> JSONResponse:
    """The voice of the next comment on: it can change while the video plays."""
    try:
        return JSONResponse({"mood": video_commentary_runner.set_mood(req.mood)})
    except Exception as exc:
        return error_response(exc)


@app.post("/api/video-commentary/voice")
def video_commentary_voice(req: VideoCommentaryVoiceRequest) -> JSONResponse:
    """Whether the next line on is said aloud, and in which voice: it can
    change while the video plays."""
    try:
        return JSONResponse({"voice": video_commentary_runner.set_voice(req.voice)})
    except Exception as exc:
        return error_response(exc)


@app.get("/api/video-commentary/speech/{number}")
def video_commentary_speech(number: int) -> Response:
    """The sound of one spoken comment, for the page to play. Only the
    last few are kept: a line that was not heard in time is not heard late."""
    sound = video_commentary_runner.speech(number)
    if sound is None:
        return JSONResponse({"error": "That line's sound is no longer kept."}, status_code=404)
    return Response(content=sound, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.get("/api/video-commentary/comments")
def video_commentary_comments(after: int = 0) -> JSONResponse:
    return JSONResponse({"comments": video_commentary_runner.comments(after), **video_commentary_runner.state()})


@app.get("/api/video-commentary/stream")
def video_commentary_stream() -> Response:
    return mjpeg_stream(video_commentary_runner.latest_jpeg, lambda: video_commentary_runner.running)


# --- screen-ocr -------------------------------------------------------------------


def _serialize_extraction(result) -> dict:
    return {
        "text": result.text,
        "translated_text": result.translated_text,
        "regions": [{"text": r.text, "confidence": r.confidence, "box": list(r.box)} for r in result.regions],
        "cancelled": _was_stopped(result),
        "stats": asdict(result.stats) if result.stats else None,
    }


class ScreenOcrExtractRequest(BaseModel):
    source: str = "screen"  # "screen" | "webcam"
    screen_index: int = 1
    camera_index: int = 0
    engine: str | None = None
    compute_device: str | None = None
    translate: bool = False


@app.post("/api/screen-ocr/extract")
async def screen_ocr_extract(req: ScreenOcrExtractRequest) -> JSONResponse:
    def work():
        if req.source == "webcam":
            image = video.capture_camera_frame(req.camera_index)
        elif req.source == "screen":
            image = video.capture_screen_frame(req.screen_index)
        else:
            raise ValueError(f"unknown source '{req.source}', expected 'screen' or 'webcam'")
        return screen_ocr_runner.extract(image=image, engine=engine.value, device=device, translate=req.translate)

    try:
        engine, device = resolve(req.engine, req.compute_device, large_model=True)
        result = await run_in_threadpool(work)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(_serialize_extraction(result))


@app.post("/api/screen-ocr/extract-upload")
async def screen_ocr_extract_upload(
    file: UploadFile,
    engine: str | None = Form(None),
    compute_device: str | None = Form(None),
    translate: bool = Form(False),
) -> JSONResponse:
    file_bytes = await file.read()

    def work():
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode the uploaded file as an image.")
        return screen_ocr_runner.extract(image=image, engine=resolved.value, device=device, translate=translate)

    try:
        resolved, device = resolve(engine, compute_device, large_model=True)
        result = await run_in_threadpool(work)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(_serialize_extraction(result))


# --- meeting-notes ----------------------------------------------------------------


def resolve_notes_device(engine: Engine, device: str | None) -> str:
    """The chip meeting notes are written on. Left to the app ("Auto"), that
    is the NPU when the machine has one in working order, as for speech, and
    otherwise what resolve() picks for any brick."""
    return resolve(engine.value, device, prefer_npu=True)[1]


class MeetingNotesStartRequest(BaseModel):
    source: str = "system"
    audio_device: str | None = None
    engine: str | None = None
    compute_device: str | None = None
    notes_device: str | None = None  # where the notes are written; nothing or "AUTO" = the NPU if there is one
    whisper_model: str | None = None
    language: str | None = None  # the language being spoken; nothing or "auto" = detect it


@app.post("/api/meeting-notes/start")
async def start_meeting_notes(req: MeetingNotesStartRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device, prefer_npu=True)
        meeting_notes_runner.start(
            loop=asyncio.get_running_loop(),
            queue=app.state.meeting_notes_queue,
            source=req.source,
            audio_device=req.audio_device,
            engine=engine,
            compute_device=device,
            whisper_model_size=req.whisper_model or _TRANSLATION_SIZE_DEFAULTS[engine],
            spoken_language=spoken_language(req.language),
            notes_device=resolve_notes_device(engine, req.notes_device),
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started"})


class MeetingNotesHandoverRequest(BaseModel):
    engine: str | None = None
    notes_device: str | None = None


@app.post("/api/meeting-notes/from-live-translation")
async def meeting_notes_from_live_translation(req: MeetingNotesHandoverRequest) -> JSONResponse:
    """Hand live translation's whole transcript to Meeting Notes, which then
    writes notes from it as from a meeting of its own (POST
    /api/meeting-notes/generate). `engine`/`notes_device` are where the
    notes will be written. Live translation is left running."""
    try:
        engine, _ = resolve(req.engine, None)
        taken = meeting_notes_runner.adopt(
            live_translation_runner.transcript()["lines"],
            engine=engine,
            notes_device=resolve_notes_device(engine, req.notes_device),
            whisper_model_size=_TRANSLATION_SIZE_DEFAULTS[engine],
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(
        {"lines": [asdict(line) for line in taken], "words": sum(len(line.text.split()) for line in taken)}
    )


@app.post("/api/meeting-notes/stop")
async def stop_meeting_notes() -> JSONResponse:
    meeting_notes_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws/meeting-notes")
async def ws_meeting_notes(websocket: WebSocket) -> None:
    await ws_drain(websocket, app.state.meeting_notes_queue)


class MeetingNotesGenerateRequest(BaseModel):
    notes_device: str | None = None  # the chip to write them on from now on; nothing = no change


@app.post("/api/meeting-notes/generate")
async def generate_meeting_notes(req: MeetingNotesGenerateRequest | None = None) -> JSONResponse:
    try:
        device = None
        engine = meeting_notes_runner.engine  # the meeting's own: None when there is none, and the runner says so
        if req is not None and req.notes_device and engine is not None:
            device = resolve_notes_device(engine, req.notes_device)
        notes = await run_in_threadpool(meeting_notes_runner.generate_notes, device)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(
        {
            "text": notes.text,
            "transcript_line_count": notes.transcript_line_count,
            "parts": notes.parts,
            "cancelled": _was_stopped(notes),
            # The chip that wrote them, which is not the one asked for when the NPU was lost on the way.
            "device": notes.stats.device if notes.stats else None,
        }
    )


# --- webcam-effects ---------------------------------------------------------------


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#RRGGBB' (what an <input type="color"> gives) -> OpenCV's BGR tuple.
    Raises ValueError on anything else, so a bad client value is a 400 at
    the route rather than an unhandled 500."""
    digits = hex_color.strip().lstrip("#")
    if len(digits) != 6 or any(c not in "0123456789abcdefABCDEF" for c in digits):
        raise ValueError(f"color must be '#RRGGBB', got '{hex_color}'")
    r, g, b = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


class WebcamEffectsStartRequest(BaseModel):
    camera_index: int = 0
    engine: str | None = None
    compute_device: str | None = None
    effect: str = "blur"
    color: str = "#0068B5"  # Intel blue, as an "#RRGGBB" hex string (what an <input type="color"> gives)


@app.post("/api/webcam-effects/start")
async def start_webcam_effects(req: WebcamEffectsStartRequest) -> JSONResponse:
    try:
        requested = req.compute_device or ("cpu" if req.engine == "portable" else "CPU")
        engine, device = resolve(req.engine, requested)
        if engine == Engine.OPENVINO:
            from webcam_effects.capabilities import validate_device

            # What was asked for, not what it resolved to: "Auto" stays
            # refused here, because it could land on the GPU this gate
            # exists to keep out. The default above is an explicit CPU.
            validate_device(requested)
        webcam_effects_runner.start(
            camera_index=req.camera_index,
            engine=engine,
            compute_device=device,
            effect=req.effect,
            color=_hex_to_bgr(req.color),
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started"})


@app.post("/api/webcam-effects/stop")
async def stop_webcam_effects() -> JSONResponse:
    webcam_effects_runner.stop()
    return JSONResponse({"status": "stopped"})


class WebcamEffectsEffectRequest(BaseModel):
    effect: str
    color: str | None = None


@app.post("/api/webcam-effects/effect")
async def set_webcam_effect(req: WebcamEffectsEffectRequest) -> JSONResponse:
    # Changes the blend live -- the capture/segmentation loop keeps running
    # untouched, only the per-frame effect render (done in the runner's
    # on_frame callback) picks this up on the next frame.
    try:
        webcam_effects_runner.set_effect(req.effect, _hex_to_bgr(req.color) if req.color else None)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "ok"})


@app.get("/api/webcam-effects/stats")
def webcam_effects_stats() -> JSONResponse:
    return JSONResponse({**webcam_effects_runner.latest_stats(), "error": webcam_effects_runner.error})


@app.get("/api/webcam-effects/stream")
def webcam_effects_stream() -> Response:
    return mjpeg_stream(webcam_effects_runner.latest_jpeg, lambda: webcam_effects_runner.running)


# --- voice-clone-studio -----------------------------------------------------------


def _voice_clone_engine(model: str, engine: str | None, device: str | None) -> tuple[Engine, str]:
    """Engine/device for a voice-clone request, honouring what the chosen
    model can actually run on. Without this, the usual "best available"
    rule hands Chatterbox the openvino engine it has no backend for, and
    the failure surfaces as a model error rather than a bad pairing."""
    if model not in voice_clone_models.MODELS:
        raise ValueError(f"Unknown model '{model}'. Choices: {', '.join(voice_clone_models.MODELS)}")
    allowed = voice_clone_models.MODEL_ENGINES[model]
    if engine is None and len(allowed) == 1:
        only = allowed[0]
        return only, device or default_device(only)
    from voice_clone_studio import voice_model

    chosen = resolve_engine(engine)
    if chosen == Engine.OPENVINO:
        if not device or device.upper() == "AUTO":
            # Left to the app: the CPU, which says a sentence in half a
            # second where a GPU compiles the models again for each one.
            return chosen, voice_model.TTS_DEVICE
        voice_model.validate_device(device)  # the NPU: refused here, never tried
    return resolve(engine, device)


class VoiceCloneStudioEnrollRecordRequest(BaseModel):
    seconds: float = 10.0
    engine: str | None = None
    compute_device: str | None = None
    model: str = voice_clone_models.DEFAULT_MODEL


@app.post("/api/voice-clone-studio/enroll-record")
async def voice_clone_studio_enroll_record(req: VoiceCloneStudioEnrollRecordRequest) -> JSONResponse:
    def work():
        reference_path = voice_clone_studio_runner.record_reference(req.seconds)
        voice_clone_studio_runner.enroll(
            reference_path=reference_path, engine=engine.value, device=device, model=req.model
        )

    try:
        engine, device = _voice_clone_engine(req.model, req.engine, req.compute_device)
        await run_in_threadpool(work)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "enrolled"})


@app.post("/api/voice-clone-studio/enroll-upload")
async def voice_clone_studio_enroll_upload(
    file: UploadFile,
    engine: str | None = Form(None),
    compute_device: str | None = Form(None),
    model: str = Form(voice_clone_models.DEFAULT_MODEL),
) -> JSONResponse:
    file_bytes = await file.read()
    suffix = Path(file.filename or "reference.wav").suffix or ".wav"

    def work():
        # Write and *close* the temp file before handing it to the cloner --
        # on Windows an open handle would make the unlink below fail and
        # mask the real error if enrolling itself raised.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            path = tmp.name
        try:
            voice_clone_studio_runner.enroll(
                reference_path=path, engine=resolved.value, device=device, model=model
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    try:
        resolved, device = _voice_clone_engine(model, engine, compute_device)
        await run_in_threadpool(work)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "enrolled"})


@app.get("/api/voice-clone-studio/status")
def voice_clone_studio_status() -> JSONResponse:
    """`supports_styles` tells the UI whether to offer the style and tau
    controls at all -- they belong to OpenVoice, not to every model."""
    return JSONResponse(
        {
            "enrolled": voice_clone_studio_runner.enrolled,
            "model": voice_clone_studio_runner.model,
            "supports_styles": voice_clone_studio_runner.supports_styles,
        }
    )


class VoiceCloneStudioSynthesizeRequest(BaseModel):
    text: str
    style: str = "default"
    tau: float = 0.3


@app.post("/api/voice-clone-studio/synthesize")
async def voice_clone_studio_synthesize(req: VoiceCloneStudioSynthesizeRequest) -> Response:
    def work():
        import soundfile as sf

        audio_out, sample_rate = voice_clone_studio_runner.synthesize(text=req.text, style=req.style, tau=req.tau)
        buffer = io.BytesIO()
        sf.write(buffer, audio_out, sample_rate, format="WAV")
        return buffer.getvalue()

    try:
        wav_bytes = await run_in_threadpool(work)
    except Exception as exc:
        return error_response(exc)
    return Response(content=wav_bytes, media_type="audio/wav")


# --- voice-assistant --------------------------------------------------------------


class VoiceAssistantStartRequest(BaseModel):
    audio_device: str | None = None
    engine: str | None = None
    whisper_model: str | None = None
    compute_device: str | None = None
    wake_word: str = "hey_jarvis"
    wake_threshold: float = 0.5
    speak_replies: bool = True
    # Which language model answers: a key of doc-qa's language_models.
    llm_model: str | None = None


@app.post("/api/voice-assistant/start")
async def start_voice_assistant(req: VoiceAssistantStartRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device)
        llm_model = _language_model(req.llm_model, engine, device)  # a wrong one stops here
        voice_assistant_runner.start(
            loop=asyncio.get_running_loop(),
            queue=app.state.voice_assistant_queue,
            audio_device=req.audio_device,
            engine=engine,
            whisper_model_size=req.whisper_model or _WHISPER_SIZE_DEFAULTS[engine],
            compute_device=device,
            wake_word=req.wake_word,
            wake_threshold=req.wake_threshold,
            speak_replies=req.speak_replies,
            llm_model=llm_model,
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started"})


@app.post("/api/voice-assistant/stop")
async def stop_voice_assistant() -> JSONResponse:
    voice_assistant_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws/voice-assistant")
async def ws_voice_assistant(websocket: WebSocket) -> None:
    await ws_drain(websocket, app.state.voice_assistant_queue)


# --- expense-extract --------------------------------------------------------------


class ExpenseExtractStartRequest(BaseModel):
    folder: str
    ocr_engine: str | None = None
    ocr_compute_device: str | None = None
    llm_engine: str | None = None
    llm_compute_device: str | None = None
    # Which language model makes the lines: a key of doc-qa's language_models.
    llm_model: str | None = None


@app.post("/api/expense-extract/start")
async def start_expense_extract(req: ExpenseExtractStartRequest) -> JSONResponse:
    try:
        ocr_engine, ocr_device = resolve(req.ocr_engine, req.ocr_compute_device, large_model=True)
        llm_engine, llm_device = resolve(req.llm_engine, req.llm_compute_device)
        llm_model = _language_model(req.llm_model, llm_engine, llm_device)  # a wrong one stops here
        expense_extract_runner.start(
            loop=asyncio.get_running_loop(),
            queue=app.state.expense_extract_queue,
            folder=req.folder,
            ocr_engine=ocr_engine,
            ocr_device=ocr_device,
            llm_engine=llm_engine,
            llm_device=llm_device,
            llm_model=llm_model,
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started", "report_id": expense_extract_runner.report_id})


@app.get("/api/expense-extract/report")
def expense_report(report_id: str | None = None):
    """The report to review: the one asked for, else the one this launcher
    has produced since it started (until it is cleared), else none -- the
    brick opens on an empty workspace, not on an old run. `reports` lists
    every saved one either way."""
    try:
        report = expense_extract_runner.reports.snapshot(report_id or expense_extract_runner.report_id)
        return JSONResponse({**report, "running": expense_extract_runner.running,
                             "active_report_id": expense_extract_runner.report_id})
    except Exception as exc:
        return error_response(exc)


@app.post("/api/expense-extract/report/close")
def close_expense_report() -> JSONResponse:
    """Empty the workspace for the next run. The report stays saved."""
    try:
        expense_extract_runner.close_report()
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "closed"})


@app.delete("/api/expense-extract/reports")
def delete_expense_reports() -> JSONResponse:
    """Delete every saved report and its expenses (not the receipt images)."""
    try:
        deleted = expense_extract_runner.delete_reports()
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "deleted", "reports": deleted})


class ExpenseReviewRequest(BaseModel):
    revision: int
    vendor: str = ""
    date: str = ""
    amount: str = ""
    currency: str = ""
    category: str = "Other"
    notes: str = ""
    validate_expense: bool = Field(False, alias="validate")


@app.put("/api/expense-extract/reports/{report_id}/expenses/{item_id}")
def update_expense(report_id: str, item_id: str, req: ExpenseReviewRequest):
    try:
        return JSONResponse(expense_extract_runner.reports.update(
            report_id, item_id, req.model_dump(), req.revision, req.validate_expense))
    except Exception as exc:
        return error_response(exc)


@app.get("/api/expense-extract/reports/{report_id}/expenses/{item_id}/receipt")
def expense_receipt(report_id: str, item_id: str):
    try:
        from PIL import Image, ImageOps

        # Convert TIFF/BMP as well as ordinary photos into a browser-readable preview.
        path = expense_extract_runner.reports.receipt(report_id, item_id)
        with Image.open(path) as receipt:
            preview = ImageOps.exif_transpose(receipt).convert("RGB")
            preview.thumbnail((1800, 2400))
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG")
        return Response(buffer.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except Exception as exc:
        return error_response(exc)


@app.get("/api/expense-extract/reports/{report_id}/export.xlsx")
def export_expense_report(report_id: str):
    try:
        if expense_extract_runner.running and expense_extract_runner.report_id == report_id:
            raise Conflict("Wait for extraction to finish before exporting the whole report.")
        return Response(expense_extract_runner.reports.export(report_id),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": 'attachment; filename="expense-report.xlsx"'})
    except Exception as exc:
        return error_response(exc)


@app.post("/api/expense-extract/stop")
async def stop_expense_extract() -> JSONResponse:
    expense_extract_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws/expense-extract")
async def ws_expense_extract(websocket: WebSocket) -> None:
    await ws_drain(websocket, app.state.expense_extract_queue)


# --- smart-recall -----------------------------------------------------------------


@app.get("/api/smart-recall/status")
def smart_recall_status() -> JSONResponse:
    from smart_recall.pipeline import index_status

    return JSONResponse({"running": smart_recall_runner.running, **index_status()})


class SmartRecallStartRequest(BaseModel):
    screen_index: int = 1
    interval_seconds: float = 5.0
    ocr_engine: str | None = None
    ocr_compute_device: str | None = None
    embed_engine: str | None = None
    embed_compute_device: str | None = None


@app.post("/api/smart-recall/start")
async def start_smart_recall(req: SmartRecallStartRequest) -> JSONResponse:
    try:
        ocr_engine, ocr_device = resolve(req.ocr_engine, req.ocr_compute_device, large_model=True)
        embed_engine, embed_device = resolve(req.embed_engine, req.embed_compute_device)
        smart_recall_runner.start(
            loop=asyncio.get_running_loop(),
            queue=app.state.smart_recall_queue,
            screen_index=req.screen_index,
            interval_seconds=req.interval_seconds,
            ocr_engine=ocr_engine,
            ocr_device=ocr_device,
            embed_engine=embed_engine,
            embed_device=embed_device,
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started"})


@app.post("/api/smart-recall/stop")
async def stop_smart_recall() -> JSONResponse:
    smart_recall_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.post("/api/smart-recall/reset")
async def reset_smart_recall() -> JSONResponse:
    try:
        await run_in_threadpool(smart_recall_runner.reset)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "reset"})


@app.websocket("/ws/smart-recall")
async def ws_smart_recall(websocket: WebSocket) -> None:
    await ws_drain(websocket, app.state.smart_recall_queue)


class SmartRecallSearchRequest(BaseModel):
    question: str
    top_k: int = 5
    compute_device: str | None = None  # None: the index's embedding engine's default device


@app.post("/api/smart-recall/search")
async def search_smart_recall(req: SmartRecallSearchRequest) -> JSONResponse:
    try:
        results = await run_in_threadpool(
            smart_recall_runner.search, question=req.question, top_k=req.top_k, device=req.compute_device
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(
        {
            "results": [
                {
                    "text": r.chunk.text,
                    "source": r.chunk.source,
                    "score": r.score,
                    "screenshot_url": f"/api/smart-recall/screenshot/{r.chunk.source}",
                }
                for r in results
            ]
        }
    )


@app.get("/api/smart-recall/screenshot/{filename}")
def smart_recall_screenshot(filename: str) -> Response:
    from smart_recall.pipeline import SCREENSHOTS_DIR

    # Strip any path components -- filenames come from chunk.source, which
    # this brick only ever generates itself, but a route parameter is
    # still untrusted input on principle.
    path = SCREENSHOTS_DIR / Path(filename).name
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


# --- code-review-assist -----------------------------------------------------------


class CodeReviewRequest(BaseModel):
    source: str = "worktree"  # "worktree" | "diff_text"
    folder: str | None = None
    against: str = "HEAD"
    diff_text: str | None = None
    engine: str | None = None
    compute_device: str | None = None


@app.post("/api/code-review-assist/review")
async def code_review_assist_review(req: CodeReviewRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device, large_model=True)
        result = await run_in_threadpool(
            code_review_assist_runner.review,
            engine=engine.value,
            device=device,
            folder=req.folder if req.source == "worktree" else None,
            against=req.against,
            diff_text=req.diff_text if req.source == "diff_text" else None,
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(
        {
            "commit_message": result.commit_message,
            "review_notes": result.review_notes,
            "diff_char_count": result.diff_char_count,
            "diff_truncated": result.diff_truncated,
            "cancelled": _was_stopped(result),
            "stats": asdict(result.stats) if result.stats else None,
        }
    )


# --- html-creator -----------------------------------------------------------------


class HtmlCreatorRequest(BaseModel):
    mode: str = "landing_page"  # "landing_page" | "document"
    prompt: str | None = None
    folder: str | None = None
    # A folder of images a landing page may place (html_creator/pictures.py).
    pictures: str | None = None
    # The same prompt gives the same page: a fresh model for each page, at
    # the price of its load time (html_creator/session.py says why).
    repeatable: bool = False
    engine: str | None = None
    compute_device: str | None = None


@app.post("/api/html-creator/generate")
async def html_creator_generate(req: HtmlCreatorRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device, large_model=True)
        result = await run_in_threadpool(
            html_creator_runner.generate,
            engine=engine.value,
            device=device,
            mode=req.mode,
            prompt=req.prompt,
            folder=req.folder,
            pictures=req.pictures,
            repeatable=req.repeatable,
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(
        {
            "html": result.html,
            # The page as written, before pictures were embedded: what a
            # person can read. Absent when nothing was embedded.
            "html_source": result.html_source,
            "pictures_offered": result.pictures_offered,
            "pictures_used": result.pictures_used,
            "picture_notes": result.picture_notes,
            "repeatable": result.repeatable,
            "mode": result.mode,
            "source_char_count": result.source_char_count,
            "source_truncated": result.source_truncated,
            "fence_stripped": result.fence_stripped,
            "html_truncated": result.html_truncated,
            "cancelled": _was_stopped(result),
            "stats": asdict(result.stats) if result.stats else None,
        }
    )


# --- page-agent (experimental) ------------------------------------------------------


class PageAgentRequest(BaseModel):
    request: str = ""
    # Each of these left out or "AUTO": the conductor decides from the hardware
    # (page_agent.conductor.assign).
    planner_device: str | None = None
    image_device: str | None = None
    page_device: str | None = None


def _page_agent_assignment(planner: str | None, images: str | None, page: str | None):
    """Which chip does which step: the ones asked for by name, checked
    against the machine, and the conductor's own choice for the rest."""
    available = list_openvino_devices()
    if not available:
        raise ValueError("The page agent needs the OpenVINO engine, and no OpenVINO device is available.")

    def chosen(step: str, device: str | None) -> str | None:
        asked = (device or "AUTO").upper()
        if asked == "AUTO":
            return None
        if asked not in available:
            raise ValueError(f"OpenVINO device {device!r} is unavailable for the {step}; choose from {', '.join(available)}")
        return asked

    planner, images, page = chosen("planner", planner), chosen("pictures", images), chosen("page", page)
    if planner and npu.is_npu(planner) and npu.lost():
        planner = None  # out of use until the app restarts: the conductor picks where its work goes
    if images and npu.is_npu(images):
        raise ValueError("The image model does not run on the NPU: choose a GPU, or the CPU.")
    return page_agent_conductor.assign(
        available, list_gpu_devices(), planner=planner, images=images, page=page, npu_usable=not npu.lost()
    )


@app.post("/api/page-agent/build")
async def page_agent_build(req: PageAgentRequest) -> JSONResponse:
    """Plan, draw, write and check one page. Takes the better part of a
    minute; GET /api/page-agent/progress says where it is meanwhile, and
    /api/bricks/page-agent/partial has the page as it is written."""
    try:
        if not req.request.strip():
            raise ValueError("Describe the page to build.")
        assignment = _page_agent_assignment(req.planner_device, req.image_device, req.page_device)
        result = await run_in_threadpool(page_agent_runner.build, request=req.request, assignment=assignment)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(
        {
            "html": result.html,
            "html_source": result.html_source,
            "assignment": asdict(result.assignment),
            "plan": {
                "title": result.plan.title,
                "headline": result.plan.headline,
                "style": result.plan.style,
                "sections": result.plan.sections,
                "offers": result.plan.offers,
                "notes": result.plan.notes,
            },
            "pictures": [
                {"name": p.name, "width": p.width, "height": p.height, "prompt": p.prompt, "seconds": p.seconds}
                for p in result.pictures
            ],
            "pictures_used": result.pictures_used,
            "checks": [asdict(check) for check in result.checks],
            "attempts": result.attempts,
            "seconds": result.seconds,
            "cancelled": result.cancelled,
            # How fast each model worked, each in its own unit, and what
            # loading each took in this build (nothing, once it is loaded).
            "stats": asdict(result.stats) if result.stats else None,
            "planner_stats": asdict(result.planner_stats) if result.planner_stats else None,
            "picture_stats": asdict(result.picture_stats) if result.picture_stats else None,
            "loads": result.loads,
        }
    )


@app.get("/api/page-agent/progress")
def page_agent_progress() -> JSONResponse:
    """The build in hand, or the last one: each step with its chip and
    state, the plan once written, each picture once drawn."""
    return JSONResponse(page_agent_runner.progress())


@app.get("/api/page-agent/picture/{name}")
def page_agent_picture(name: str) -> Response:
    path = page_agent_runner.picture(name)
    if path is None:
        return JSONResponse({"error": "No such picture in the current build."}, status_code=404)
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# --- auto demo ---------------------------------------------------------------------
# The app running itself on a stand (autodemo.py; docs/AUTO_DEMO.md). The
# director acts through the routes above, on this machine, the way a person
# at the page would.


def _own_route(method: str, path: str, body: dict | None, timeout: float):
    """Call one of this launcher's own routes and return its JSON answer.
    An error answer is raised with the message the route gave."""
    import json
    import urllib.error
    import urllib.request

    host, port = getattr(app.state, "bind", ("127.0.0.1", 8765))
    host = "127.0.0.1" if host in ("0.0.0.0", "::", "localhost") else host
    data = None if method == "GET" else json.dumps(body or {}).encode("utf-8")
    request = urllib.request.Request(
        f"http://{host}:{port}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read()).get("error") or str(exc)
        except Exception:
            message = str(exc)
        # 409 is a demo saying it is still at something: the director waits for it.
        raise (autodemo.Busy if exc.code == 409 else RuntimeError)(message) from None


def _street_cameras_reachable() -> bool:
    """Whether the one scene that needs the internet can have it: its
    traffic-camera clips are fetched from this host."""
    try:
        with socket.create_connection(("s3-eu-west-1.amazonaws.com", 443), timeout=3):
            return True
    except OSError:
        return False


autodemo_director = autodemo.Director(_own_route, autodemo_scenes.PLAYLIST, online=_street_cameras_reachable)


class AutoDemoRequest(BaseModel):
    # "auto": use the discrete GPU if it is plugged in. "off": play as if it
    # were not there -- for a stand that will lose it, or to rehearse one.
    dgpu: str = "auto"
    big_screen: bool = False
    lang: str = "en"  # the language the story is told in: "en" or "fr"
    # "auto": a scene that can show the camera does. "off": none switches it
    # on -- in a meeting, or anywhere people have not come to be on a screen.
    camera: str = "auto"
    # The scenes to play, by the keys /api/autodemo/check lists. Not said: all of them.
    scenes: list[str] | None = None


@app.get("/api/autodemo")
def autodemo_state() -> JSONResponse:
    """Where the loop is: its state, the scene in hand with what it says
    about itself, what this turn of the loop plays and skips, and why."""
    return JSONResponse(autodemo_director.snapshot())


@app.get("/api/autodemo/check")
def autodemo_check(dgpu: str = "auto", big_screen: bool = False, lang: str = "en", camera: str = "auto") -> JSONResponse:
    """What the loop would play on this stand, before starting it: every
    scene of the playlist under the key it is chosen by, whether it can play
    here and, if not, why."""
    try:
        return JSONResponse(autodemo_director.check(dgpu=dgpu, big_screen=big_screen, lang=lang, camera=camera))
    except Exception as exc:
        return error_response(exc)


@app.get("/api/autodemo/result")
def autodemo_result() -> JSONResponse:
    """What the scene in hand was answered, for the page to draw with the
    panel's own drawing code."""
    result = autodemo_director.result()
    if result is None:
        return JSONResponse({"error": "The scene in hand has no result yet."}, status_code=404)
    return JSONResponse(result)


@app.post("/api/autodemo/start")
def autodemo_start(req: AutoDemoRequest) -> JSONResponse:
    try:
        return JSONResponse(autodemo_director.start(
            dgpu=req.dgpu, big_screen=req.big_screen, lang=req.lang, camera=req.camera, scenes=req.scenes,
        ))
    except Exception as exc:
        return error_response(exc)


@app.post("/api/autodemo/stop")
def autodemo_stop() -> JSONResponse:
    """Answers within a few seconds either way: "idle" if the loop has let
    go of everything by then, "stopping" if the demo it was in is still
    finishing -- the page shows that on the start screen, where it goes."""
    return JSONResponse(autodemo_director.stop(wait=6.0))


@app.post("/api/autodemo/pause")
def autodemo_pause() -> JSONResponse:
    """Hold the loop where it is: the scene in hand finishes and stays on
    screen, and the next one waits for a resume -- or for five minutes
    with nobody asking for anything."""
    return JSONResponse(autodemo_director.pause())


@app.post("/api/autodemo/resume")
def autodemo_resume() -> JSONResponse:
    return JSONResponse(autodemo_director.resume())


@app.post("/api/autodemo/skip")
def autodemo_skip() -> JSONResponse:
    return JSONResponse(autodemo_director.skip())


# --- updates ----------------------------------------------------------------------

# What the launcher exits with when it stops to be upgraded, so
# start_launcher.bat can tell that apart from a crash (see updates.py).
UPGRADE_EXIT_CODE = 3
_upgrade_requested = threading.Event()


def _busy_demos() -> list[str]:
    """Demos loading or running right now, by name: upgrading restarts the
    launcher and would cut them off."""
    names = {demo.id: demo.name for demo in registry.REGISTRY}
    ids = {entry["demo_id"] for entry in activity.snapshot()}
    ids |= {
        key.split(":")[0]
        for key, state in events.status_snapshot().items()
        if state.get("phase") in ("loading", "running")
    }
    busy = sorted(names.get(demo_id, demo_id) for demo_id in ids)
    # Between two scenes nothing is running, and the loop is still on: an
    # upgrade must not slip into that gap on a stand nobody is watching.
    return [*busy, "Auto Demo"] if autodemo_director.running else busy


@app.get("/api/update")
def update_status() -> JSONResponse:
    """Whether GitHub has a newer version, whether this copy can upgrade
    itself, and whether the page should still offer the one-time prompt."""
    return JSONResponse({**updates.snapshot(), "running_demos": _busy_demos()})


@app.post("/api/update/check")
async def update_check() -> JSONResponse:
    await run_in_threadpool(updates.refresh)
    return update_status()


@app.post("/api/update/prompted")
def update_prompted() -> JSONResponse:
    updates.mark_prompted()
    return JSONResponse({"ok": True})


@app.post("/api/update/upgrade")
async def update_upgrade() -> JSONResponse:
    host, port = getattr(app.state, "bind", ("127.0.0.1", 8765))
    try:
        started = await run_in_threadpool(updates.start_upgrade, host=host, port=port, busy=_busy_demos())
    except Exception as exc:
        return error_response(exc)
    _upgrade_requested.set()
    server = getattr(app.state, "server", None)
    if server is not None:
        # A moment's grace so this response reaches the page before the server
        # stops; the helper waits for the process to be gone before it works.
        threading.Timer(1.0, lambda: setattr(server, "should_exit", True)).start()
    return JSONResponse(started, status_code=202)


@app.get("/api/update/last")
def update_last() -> JSONResponse:
    """What the helper recorded about the most recent upgrade, or null."""
    return JSONResponse(updates.last_result())


@app.get("/api/changelog")
def api_changelog() -> JSONResponse:
    """Every version this copy has been through and what each changed,
    newest first, from its own git history (no network). `running` is the
    one to mark as current; `on_disk` differs from it until a restart."""
    return JSONResponse({**updates.history(), "running": RUNNING_VERSION, "on_disk": read_version_file()})


# --- entry point --------------------------------------------------------------------


def is_already_serving(host: str, port: int) -> bool:
    """True if something already answers on that port -- i.e. a copy of this
    launcher is running. Asked by connecting rather than by trying to bind,
    because on Windows a bind test can succeed against a socket someone else
    is listening on and tell us the opposite of the truth."""
    target = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        try:
            return probe.connect_ex((target, port)) == 0
        except OSError:
            return False


def main() -> None:
    parser = argparse.ArgumentParser(prog="panther-lake-launcher", description="Serve the Panther Lake AI Studio UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind. Default: 127.0.0.1 (this machine only).")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on. Default: 8765")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the UI in a browser tab on start.")
    args = parser.parse_args()

    # Checked *before* the browser opens, which is the whole point. Started a
    # second time on a taken port, this used to open a tab -- pointed at the
    # copy already running -- and only then die on the bind. You would be
    # looking at the old instance, in a tab your restart had just opened, with
    # nothing on screen saying so. Worse, the page would look updated: static
    # files are read per request, so the UI refreshes while the Python behind
    # it stays as it was.
    if is_already_serving(args.host, args.port):
        print(f"Panther Lake AI Studio is already running on http://{args.host}:{args.port}.", file=sys.stderr)
        print("Close that window (or press Ctrl+C in it), then start this again.", file=sys.stderr)
        print("A running copy keeps serving the code it started with -- it will not pick up an update.", file=sys.stderr)
        print(f"To run a second copy alongside it instead: panther-lake-launcher --port {args.port + 1}", file=sys.stderr)
        raise SystemExit(1)

    if not args.no_browser:
        browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        webbrowser.open(f"http://{browse_host}:{args.port}")
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port, timeout_graceful_shutdown=5))
    # Where the upgrade route can reach it: an upgrade stops this server
    # gracefully, lifespan shutdown included, rather than killing the process.
    app.state.server = server
    app.state.bind = (args.host, args.port)
    server.run()
    code = UPGRADE_EXIT_CODE if _upgrade_requested.is_set() else 0
    if code:
        print("Upgrading: the new version starts in a new window once it's installed. This one can be closed.")
    if page_agent_runner.has_built:
        # A process that has built a page may never finish exiting on its own
        # (page_agent/leaving.py) -- and an upgrade waits for this one to end.
        from page_agent.leaving import leave_now

        leave_now(code)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
