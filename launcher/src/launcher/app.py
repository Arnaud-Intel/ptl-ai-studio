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
import shutil
import tempfile
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
from pantherlake_ai_core import audio, video
from pantherlake_ai_core.engine import (
    Engine,
    default_device,
    list_gpu_devices,
    list_openvino_devices,
    preferred_large_model_device,
    resolve_engine,
)
from pydantic import BaseModel
from smart_city_monitor import sources as smart_city_sources
from smart_city_monitor.types import FeedSpec as SmartCityFeedSpec

from . import activity, events, registry
from .code_review_assist_runner import CodeReviewAssistRunner
from .doc_qa_runner import DocQARunner
from .errors import Conflict
from .expense_extract_runner import ExpenseExtractRunner
from .html_creator_runner import HtmlCreatorRunner
from .live_translation_runner import LiveTranslationRunner
from .meeting_notes_runner import MeetingNotesRunner
from .object_detection_runner import ObjectDetectionRunner
from .screen_ocr_runner import ScreenOcrRunner
from .smart_city_monitor_runner import SmartCityMonitorRunner
from .smart_recall_runner import SmartRecallRunner
from .telemetry_poller import TelemetryPoller
from .voice_assistant_runner import VoiceAssistantRunner
from .voice_clone_studio_runner import VoiceCloneStudioRunner
from .webcam_effects_runner import WebcamEffectsRunner

STATIC_DIR = Path(__file__).parent / "static"
VERSION_FILE = Path(__file__).resolve().parents[3] / "VERSION"

# Whisper size defaults for the three speech bricks that expose one: the
# portable engine (faster-whisper) is comfortable with "small" on CPU;
# "base" is the largest multilingual size Intel pre-converts for OpenVINO
# short of large-v3.
_WHISPER_SIZE_DEFAULTS = {Engine.PORTABLE: "small", Engine.OPENVINO: "base"}


def get_version() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


# --- shared plumbing --------------------------------------------------------


def resolve(engine: str | None, device: str | None, *, large_model: bool = False) -> tuple[Engine, str]:
    """Engine + device for a request, by the rule the CLIs use: `engine` if
    given (an unknown name is a ValueError, i.e. a 400), else the best
    available; `device` if given, else the engine's default -- or, for a
    brick whose openvino model needs a discrete GPU's VRAM (`large_model`),
    the machine's discrete GPU when it has one."""
    resolved = resolve_engine(engine)
    if device:
        return resolved, device
    if large_model and resolved == Engine.OPENVINO:
        return resolved, preferred_large_model_device()
    return resolved, default_device(resolved)


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
            await asyncio.sleep(0.05)

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
    yield
    telemetry_poller.stop()


app = FastAPI(title="Panther Lake AI Studio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

live_translation_runner = LiveTranslationRunner()
doc_qa_runner = DocQARunner()
object_detection_runner = ObjectDetectionRunner()
smart_city_monitor_runner = SmartCityMonitorRunner()
screen_ocr_runner = ScreenOcrRunner()
meeting_notes_runner = MeetingNotesRunner()
webcam_effects_runner = WebcamEffectsRunner()
voice_clone_studio_runner = VoiceCloneStudioRunner()
voice_assistant_runner = VoiceAssistantRunner()
expense_extract_runner = ExpenseExtractRunner()
smart_recall_runner = SmartRecallRunner()
code_review_assist_runner = CodeReviewAssistRunner()
html_creator_runner = HtmlCreatorRunner()
telemetry_poller = TelemetryPoller()


@app.exception_handler(RequestValidationError)
async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """A malformed body gets the same `{"error": ...}` shape as every other
    failure, not FastAPI's default 422 `detail` list the UI can't show."""
    problems = "; ".join(
        f"{'.'.join(str(part) for part in err['loc'][1:]) or 'body'}: {err['msg']}" for err in exc.errors()
    )
    return JSONResponse({"error": f"invalid request: {problems}"}, status_code=400)


@app.get("/")
def index() -> HTMLResponse:
    """The page, with its script/stylesheet URLs stamped by their files'
    modification time -- so a browser that cached the previous version's
    app.js picks up the new one on a plain reload after an update, instead
    of running stale code against new markup."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for asset in ("style.css", "app.js"):
        try:
            stamp = int((STATIC_DIR / asset).stat().st_mtime)
        except OSError:
            continue
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={stamp}")
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
    return JSONResponse({"version": get_version()})


@app.get("/api/telemetry")
def telemetry_snapshot() -> JSONResponse:
    """CPU/GPU/NPU utilization (from the background poller's cache -- see
    telemetry_poller.py for why this isn't queried fresh per request),
    plus which demo (if any) is currently driving each device."""
    payload = telemetry_poller.snapshot()
    payload["active"] = activity.snapshot()
    return JSONResponse(payload)


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
    for kind in demo.devices:
        payload[kind] = _DEVICE_SOURCES[kind]()
    if demo.samples:
        payload["samples"] = [asdict(s) for s in importlib.import_module(demo.samples).SAMPLES]
    return JSONResponse(payload)


# --- live-translation -------------------------------------------------------------


class LiveTranslationStartRequest(BaseModel):
    source: str = "mic"
    audio_device: str | None = None
    engine: str | None = None
    model_size: str | None = None
    compute_device: str | None = None


@app.post("/api/live-translation/start")
async def start_live_translation(req: LiveTranslationStartRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device)
        live_translation_runner.start(
            loop=asyncio.get_running_loop(),
            queue=app.state.live_translation_queue,
            source=req.source,
            audio_device=req.audio_device,
            engine=engine,
            model_size=req.model_size or _WHISPER_SIZE_DEFAULTS[engine],
            compute_device=device,
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started"})


@app.post("/api/live-translation/stop")
async def stop_live_translation() -> JSONResponse:
    live_translation_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws/live-translation")
async def ws_live_translation(websocket: WebSocket) -> None:
    await ws_drain(websocket, app.state.live_translation_queue)


# --- doc-qa -----------------------------------------------------------------------


class DocQAIngestRequest(BaseModel):
    folder: str
    engine: str | None = None
    compute_device: str | None = None
    reindex: bool = False


@app.post("/api/doc-qa/ingest")
async def doc_qa_ingest(req: DocQAIngestRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device)
        count, folder = await run_in_threadpool(
            doc_qa_runner.ingest, folder=req.folder, engine=engine.value, device=device, reindex=req.reindex
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"chunks": count, "folder": folder})


@app.get("/api/doc-qa/status")
def doc_qa_status() -> JSONResponse:
    """Whether an index is loaded (and from where) -- so reopening the panel
    or reloading the page picks up an index built earlier instead of
    asking to build it again."""
    return JSONResponse(doc_qa_runner.status())


class DocQAAskRequest(BaseModel):
    question: str
    top_k: int = 4


@app.post("/api/doc-qa/ask")
async def doc_qa_ask(req: DocQAAskRequest) -> JSONResponse:
    try:
        answer = await run_in_threadpool(doc_qa_runner.ask, question=req.question, top_k=req.top_k)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(
        {
            "text": answer.text,
            "sources": [
                {"source": r.chunk.source, "chunk_index": r.chunk.chunk_index, "score": r.score}
                for r in answer.sources
            ],
        }
    )


# --- object-detection -------------------------------------------------------------


class ObjectDetectionStartRequest(BaseModel):
    source: str = "screen"
    camera_index: int = 0
    screen_index: int = 1
    engine: str | None = None
    compute_device: str | None = None


@app.post("/api/object-detection/start")
async def start_object_detection(req: ObjectDetectionStartRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device)
        object_detection_runner.start(
            source=req.source,
            camera_index=req.camera_index,
            screen_index=req.screen_index,
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
    return JSONResponse(
        {"detections": object_detection_runner.latest_detections(), "error": object_detection_runner.error}
    )


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
    # YOLO11n, another on the CPU with DETR, at the same time.
    engine: str | None = None
    model_path: str | None = None


class SmartCityMonitorStartRequest(BaseModel):
    feeds: list[SmartCityFeedInput]
    engine: str | None = None
    compute_device: str | None = None
    loop: bool = True


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
    }


@app.post("/api/smart-city-monitor/start")
async def start_smart_city_monitor(req: SmartCityMonitorStartRequest) -> JSONResponse:
    try:
        if not req.feeds:
            raise ValueError("at least one feed is required")
        default_engine, default_device = resolve(req.engine, req.compute_device)
        feeds = []
        for i, f in enumerate(req.feeds, start=1):
            if not f.path.strip():
                raise ValueError(f"feed {i} has no source -- give it a file path or a URL")
            # Resolved per feed, so an unknown engine name is a 400 naming
            # the feed rather than a failure deep inside the pipeline.
            feed_engine, feed_device = resolve(f.engine or req.engine, f.compute_device or req.compute_device)
            feeds.append(
                SmartCityFeedSpec(
                    feed_id=f"feed-{i}",
                    path=f.path.strip(),
                    compute_device=feed_device,
                    name=smart_city_sources.display_name(f.path.strip()),
                    engine=feed_engine,
                    model_path=f.model_path or None,
                )
            )
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
        }
    )


@app.get("/api/smart-city-monitor/stream")
def smart_city_monitor_stream(feed: str) -> Response:
    return mjpeg_stream(lambda: smart_city_monitor_runner.latest_jpeg(feed), lambda: smart_city_monitor_runner.running)


# --- screen-ocr -------------------------------------------------------------------


def _serialize_extraction(result) -> dict:
    return {
        "text": result.text,
        "translated_text": result.translated_text,
        "regions": [{"text": r.text, "confidence": r.confidence, "box": list(r.box)} for r in result.regions],
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
        engine, device = resolve(req.engine, req.compute_device)
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
        resolved, device = resolve(engine, compute_device)
        result = await run_in_threadpool(work)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(_serialize_extraction(result))


# --- meeting-notes ----------------------------------------------------------------


class MeetingNotesStartRequest(BaseModel):
    source: str = "system"
    audio_device: str | None = None
    engine: str | None = None
    compute_device: str | None = None
    whisper_model: str | None = None


@app.post("/api/meeting-notes/start")
async def start_meeting_notes(req: MeetingNotesStartRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device)
        meeting_notes_runner.start(
            loop=asyncio.get_running_loop(),
            queue=app.state.meeting_notes_queue,
            source=req.source,
            audio_device=req.audio_device,
            engine=engine,
            compute_device=device,
            whisper_model_size=req.whisper_model or _WHISPER_SIZE_DEFAULTS[engine],
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started"})


@app.post("/api/meeting-notes/stop")
async def stop_meeting_notes() -> JSONResponse:
    meeting_notes_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws/meeting-notes")
async def ws_meeting_notes(websocket: WebSocket) -> None:
    await ws_drain(websocket, app.state.meeting_notes_queue)


@app.post("/api/meeting-notes/generate")
async def generate_meeting_notes() -> JSONResponse:
    try:
        notes = await run_in_threadpool(meeting_notes_runner.generate_notes)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"text": notes.text, "transcript_line_count": notes.transcript_line_count})


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
        engine, device = resolve(req.engine, req.compute_device)
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


class VoiceCloneStudioEnrollRecordRequest(BaseModel):
    seconds: float = 10.0
    engine: str | None = None
    compute_device: str | None = None


@app.post("/api/voice-clone-studio/enroll-record")
async def voice_clone_studio_enroll_record(req: VoiceCloneStudioEnrollRecordRequest) -> JSONResponse:
    def work():
        reference_path = voice_clone_studio_runner.record_reference(req.seconds)
        voice_clone_studio_runner.enroll(reference_path=reference_path, engine=engine.value, device=device)

    try:
        engine, device = resolve(req.engine, req.compute_device)
        await run_in_threadpool(work)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "enrolled"})


@app.post("/api/voice-clone-studio/enroll-upload")
async def voice_clone_studio_enroll_upload(
    file: UploadFile,
    engine: str | None = Form(None),
    compute_device: str | None = Form(None),
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
            voice_clone_studio_runner.enroll(reference_path=path, engine=resolved.value, device=device)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    try:
        resolved, device = resolve(engine, compute_device)
        await run_in_threadpool(work)
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "enrolled"})


@app.get("/api/voice-clone-studio/status")
def voice_clone_studio_status() -> JSONResponse:
    return JSONResponse({"enrolled": voice_clone_studio_runner.enrolled})


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


@app.post("/api/voice-assistant/start")
async def start_voice_assistant(req: VoiceAssistantStartRequest) -> JSONResponse:
    try:
        engine, device = resolve(req.engine, req.compute_device)
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


@app.post("/api/expense-extract/start")
async def start_expense_extract(req: ExpenseExtractStartRequest) -> JSONResponse:
    try:
        ocr_engine, ocr_device = resolve(req.ocr_engine, req.ocr_compute_device)
        llm_engine, llm_device = resolve(req.llm_engine, req.llm_compute_device)
        expense_extract_runner.start(
            loop=asyncio.get_running_loop(),
            queue=app.state.expense_extract_queue,
            folder=req.folder,
            ocr_engine=ocr_engine,
            ocr_device=ocr_device,
            llm_engine=llm_engine,
            llm_device=llm_device,
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse({"status": "started"})


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
        ocr_engine, ocr_device = resolve(req.ocr_engine, req.ocr_compute_device)
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
        }
    )


# --- html-creator -----------------------------------------------------------------


class HtmlCreatorRequest(BaseModel):
    mode: str = "landing_page"  # "landing_page" | "document"
    prompt: str | None = None
    folder: str | None = None
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
        )
    except Exception as exc:
        return error_response(exc)
    return JSONResponse(
        {
            "html": result.html,
            "mode": result.mode,
            "source_char_count": result.source_char_count,
            "source_truncated": result.source_truncated,
            "fence_stripped": result.fence_stripped,
            "html_truncated": result.html_truncated,
        }
    )


# --- entry point --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(prog="panther-lake-launcher", description="Serve the Panther Lake AI Studio UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind. Default: 127.0.0.1 (this machine only).")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on. Default: 8765")
    parser.add_argument("--no-browser", action="store_true", help="Don't open the UI in a browser tab on start.")
    args = parser.parse_args()

    if not args.no_browser:
        browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        webbrowser.open(f"http://{browse_host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
