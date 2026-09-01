"""FastAPI app: serves the Panther Lake AI Studio UI and drives the
live-translation demo.

Run with `uv run panther-lake-launcher` from the workspace root.
"""
from __future__ import annotations

import asyncio
import time
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from code_review_assist.samples import SAMPLES as CODE_REVIEW_ASSIST_SAMPLES
from doc_qa.samples import SAMPLES as DOC_QA_SAMPLES
from html_creator.samples import SAMPLES as HTML_CREATOR_SAMPLES
from pantherlake_ai_core import audio, video
from pantherlake_ai_core.engine import Engine, list_gpu_devices, list_openvino_devices
from pydantic import BaseModel
from smart_recall.samples import SAMPLES as SMART_RECALL_SAMPLES
from voice_clone_studio.samples import SAMPLES as VOICE_CLONE_STUDIO_SAMPLES

from . import activity, events, registry
from .code_review_assist_runner import CodeReviewAssistRunner
from .doc_qa_runner import DocQARunner
from .html_creator_runner import HtmlCreatorRunner
from .live_translation_runner import LiveTranslationRunner
from .meeting_notes_runner import MeetingNotesRunner
from .object_detection_runner import ObjectDetectionRunner
from .screen_ocr_runner import ScreenOcrRunner
from .telemetry_poller import TelemetryPoller
from .expense_extract_runner import ExpenseExtractRunner
from .smart_recall_runner import SmartRecallRunner
from .voice_assistant_runner import VoiceAssistantRunner
from .voice_clone_studio_runner import VoiceCloneStudioRunner
from .webcam_effects_runner import WebcamEffectsRunner

STATIC_DIR = Path(__file__).parent / "static"
VERSION_FILE = Path(__file__).resolve().parents[3] / "VERSION"


def get_version() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "unknown"

_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"model_size": "small", "compute_device": "auto"},
    Engine.OPENVINO: {"model_size": "base", "compute_device": "AUTO"},
}

_DOC_QA_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"compute_device": "cpu"},
    Engine.OPENVINO: {"compute_device": "AUTO"},
}

_OBJECT_DETECTION_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"compute_device": "cpu"},
    Engine.OPENVINO: {"compute_device": "AUTO"},
}

_SCREEN_OCR_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"compute_device": "cpu"},
    Engine.OPENVINO: {"compute_device": "AUTO"},
}

_MEETING_NOTES_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"whisper_model": "small", "compute_device": "auto"},
    Engine.OPENVINO: {"whisper_model": "base", "compute_device": "AUTO"},
}

_WEBCAM_EFFECTS_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"compute_device": "cpu"},
    Engine.OPENVINO: {"compute_device": "AUTO"},
}

_VOICE_CLONE_STUDIO_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"compute_device": "CPU"},
    Engine.OPENVINO: {"compute_device": "AUTO"},
}

_VOICE_ASSISTANT_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"whisper_model": "small", "compute_device": "auto"},
    Engine.OPENVINO: {"whisper_model": "base", "compute_device": "AUTO"},
}

_EXPENSE_EXTRACT_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"compute_device": "cpu"},
    Engine.OPENVINO: {"compute_device": "AUTO"},
}

_SMART_RECALL_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"compute_device": "cpu"},
    Engine.OPENVINO: {"compute_device": "AUTO"},
}

_CODE_REVIEW_ASSIST_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"compute_device": "cpu"},
    # GPU.1 is this dev machine's Arc B60 card id, not a portable default the
    # way "AUTO" is for every other brick -- the coding model this brick
    # defaults to is picked to run well on that specific card.
    Engine.OPENVINO: {"compute_device": "GPU.1"},
}

_HTML_CREATOR_ENGINE_DEFAULTS = {
    Engine.PORTABLE: {"compute_device": "cpu"},
    Engine.OPENVINO: {"compute_device": "GPU.1"},
}


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

runner = LiveTranslationRunner()
doc_qa_runner = DocQARunner()
object_detection_runner = ObjectDetectionRunner()
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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


@app.get("/api/live-translation/devices")
def live_translation_devices() -> JSONResponse:
    return JSONResponse(
        {
            "microphones": audio.list_microphones(),
            "speakers": audio.list_speakers(),
            "openvino_devices": list_openvino_devices(),
        }
    )


class StartRequest(BaseModel):
    source: str = "mic"
    audio_device: str | None = None
    engine: str = "portable"
    model_size: str | None = None
    compute_device: str | None = None


@app.post("/api/live-translation/start")
async def start_live_translation(req: StartRequest) -> JSONResponse:
    if runner.running:
        return JSONResponse({"error": "live-translation is already running"}, status_code=409)

    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    defaults = _ENGINE_DEFAULTS[engine]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = app.state.live_translation_queue

    try:
        runner.start(
            loop=loop,
            queue=queue,
            source=req.source,
            audio_device=req.audio_device,
            engine=engine,
            model_size=req.model_size or defaults["model_size"],
            compute_device=req.compute_device or defaults["compute_device"],
        )
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)

    return JSONResponse({"status": "started"})


@app.post("/api/live-translation/stop")
async def stop_live_translation() -> JSONResponse:
    runner.stop()
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws/live-translation")
async def ws_live_translation(websocket: WebSocket) -> None:
    # Single shared queue: fine for this launcher's one-operator-one-tab
    # use case, but note a second concurrently connected tab would only
    # get every other message rather than a full duplicate stream.
    await websocket.accept()
    queue: asyncio.Queue = app.state.live_translation_queue
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass


@app.get("/api/doc-qa/devices")
def doc_qa_devices() -> JSONResponse:
    return JSONResponse(
        {"openvino_devices": list_openvino_devices(), "samples": [asdict(s) for s in DOC_QA_SAMPLES]}
    )


class DocQAIngestRequest(BaseModel):
    folder: str
    engine: str = "portable"
    compute_device: str | None = None
    reindex: bool = False


@app.post("/api/doc-qa/ingest")
async def doc_qa_ingest(req: DocQAIngestRequest) -> JSONResponse:
    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    device = req.compute_device or _DOC_QA_ENGINE_DEFAULTS[engine]["compute_device"]

    try:
        count, folder = await run_in_threadpool(
            doc_qa_runner.ingest,
            folder=req.folder,
            engine=req.engine,
            device=device,
            reindex=req.reindex,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse({"chunks": count, "folder": folder})


class DocQAAskRequest(BaseModel):
    question: str
    top_k: int = 4


@app.post("/api/doc-qa/ask")
async def doc_qa_ask(req: DocQAAskRequest) -> JSONResponse:
    try:
        answer = await run_in_threadpool(doc_qa_runner.ask, question=req.question, top_k=req.top_k)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse(
        {
            "text": answer.text,
            "sources": [
                {"source": r.chunk.source, "chunk_index": r.chunk.chunk_index, "score": r.score}
                for r in answer.sources
            ],
        }
    )


@app.get("/api/object-detection/devices")
def object_detection_devices() -> JSONResponse:
    return JSONResponse(
        {
            "cameras": video.list_cameras(),
            "screens": video.list_screens(),
            "openvino_devices": list_openvino_devices(),
        }
    )


class ObjectDetectionStartRequest(BaseModel):
    source: str = "screen"
    camera_index: int = 0
    screen_index: int = 1
    engine: str = "portable"
    compute_device: str | None = None


@app.post("/api/object-detection/start")
async def start_object_detection(req: ObjectDetectionStartRequest) -> JSONResponse:
    if object_detection_runner.running:
        return JSONResponse({"error": "object-detection is already running"}, status_code=409)

    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    compute_device = req.compute_device or _OBJECT_DETECTION_ENGINE_DEFAULTS[engine]["compute_device"]

    try:
        object_detection_runner.start(
            source=req.source,
            camera_index=req.camera_index,
            screen_index=req.screen_index,
            engine=engine,
            compute_device=compute_device,
        )
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)

    return JSONResponse({"status": "started"})


@app.post("/api/object-detection/stop")
async def stop_object_detection() -> JSONResponse:
    object_detection_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.get("/api/object-detection/detections")
def object_detection_detections() -> JSONResponse:
    return JSONResponse({"detections": object_detection_runner.latest_detections(), "error": object_detection_runner.error})


@app.get("/api/object-detection/stream")
def object_detection_stream() -> StreamingResponse:
    def generate():
        last_sent = None
        # Poll the runner's single "latest frame" buffer rather than a
        # queue: for video, only the newest frame matters, so there's
        # nothing to gain from buffering ones the client hasn't seen yet.
        while object_detection_runner.running:
            jpeg = object_detection_runner.latest_jpeg()
            if jpeg is not None and jpeg is not last_sent:
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
                last_sent = jpeg
            time.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/screen-ocr/devices")
def screen_ocr_devices() -> JSONResponse:
    return JSONResponse(
        {
            "cameras": video.list_cameras(),
            "screens": video.list_screens(),
            "openvino_devices": list_openvino_devices(),
        }
    )


def _serialize_extraction(result) -> dict:
    return {
        "text": result.text,
        "translated_text": result.translated_text,
        "regions": [
            {"text": r.text, "confidence": r.confidence, "box": list(r.box)} for r in result.regions
        ],
    }


class ScreenOcrExtractRequest(BaseModel):
    source: str = "screen"  # "screen" | "webcam"
    screen_index: int = 1
    camera_index: int = 0
    engine: str = "portable"
    compute_device: str | None = None
    translate: bool = False


@app.post("/api/screen-ocr/extract")
async def screen_ocr_extract(req: ScreenOcrExtractRequest) -> JSONResponse:
    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    compute_device = req.compute_device or _SCREEN_OCR_ENGINE_DEFAULTS[engine]["compute_device"]

    def work():
        if req.source == "webcam":
            image = video.capture_camera_frame(req.camera_index)
        else:
            image = video.capture_screen_frame(req.screen_index)
        return screen_ocr_runner.extract(
            image=image, engine=req.engine, device=compute_device, translate=req.translate
        )

    try:
        result = await run_in_threadpool(work)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse(_serialize_extraction(result))


@app.post("/api/screen-ocr/extract-upload")
async def screen_ocr_extract_upload(
    file: UploadFile,
    engine: str = Form("portable"),
    compute_device: str | None = Form(None),
    translate: bool = Form(False),
) -> JSONResponse:
    try:
        engine_enum = Engine(engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{engine}'"}, status_code=400)

    resolved_device = compute_device or _SCREEN_OCR_ENGINE_DEFAULTS[engine_enum]["compute_device"]
    file_bytes = await file.read()

    def work():
        import cv2
        import numpy as np

        array = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode the uploaded file as an image.")
        return screen_ocr_runner.extract(image=image, engine=engine, device=resolved_device, translate=translate)

    try:
        result = await run_in_threadpool(work)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse(_serialize_extraction(result))


@app.get("/api/meeting-notes/devices")
def meeting_notes_devices() -> JSONResponse:
    return JSONResponse(
        {
            "microphones": audio.list_microphones(),
            "speakers": audio.list_speakers(),
            "openvino_devices": list_openvino_devices(),
        }
    )


class MeetingNotesStartRequest(BaseModel):
    source: str = "system"
    audio_device: str | None = None
    engine: str = "portable"
    compute_device: str | None = None
    whisper_model: str | None = None


@app.post("/api/meeting-notes/start")
async def start_meeting_notes(req: MeetingNotesStartRequest) -> JSONResponse:
    if meeting_notes_runner.running:
        return JSONResponse({"error": "meeting-notes is already running"}, status_code=409)

    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    defaults = _MEETING_NOTES_ENGINE_DEFAULTS[engine]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = app.state.meeting_notes_queue

    try:
        meeting_notes_runner.start(
            loop=loop,
            queue=queue,
            source=req.source,
            audio_device=req.audio_device,
            engine=engine,
            compute_device=req.compute_device or defaults["compute_device"],
            whisper_model_size=req.whisper_model or defaults["whisper_model"],
        )
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)

    return JSONResponse({"status": "started"})


@app.post("/api/meeting-notes/stop")
async def stop_meeting_notes() -> JSONResponse:
    meeting_notes_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws/meeting-notes")
async def ws_meeting_notes(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = app.state.meeting_notes_queue
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass


@app.post("/api/meeting-notes/generate")
async def generate_meeting_notes() -> JSONResponse:
    try:
        notes = await run_in_threadpool(meeting_notes_runner.generate_notes)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse({"text": notes.text, "transcript_line_count": notes.transcript_line_count})


@app.get("/api/webcam-effects/devices")
def webcam_effects_devices() -> JSONResponse:
    return JSONResponse(
        {
            "cameras": video.list_cameras(),
            "openvino_devices": list_openvino_devices(),
        }
    )


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


class WebcamEffectsStartRequest(BaseModel):
    camera_index: int = 0
    engine: str = "portable"
    compute_device: str | None = None
    effect: str = "blur"
    color: str = "#0068B5"  # Intel blue, as an "#RRGGBB" hex string (what an <input type="color"> gives)


@app.post("/api/webcam-effects/start")
async def start_webcam_effects(req: WebcamEffectsStartRequest) -> JSONResponse:
    if webcam_effects_runner.running:
        return JSONResponse({"error": "webcam-effects is already running"}, status_code=409)

    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    compute_device = req.compute_device or _WEBCAM_EFFECTS_ENGINE_DEFAULTS[engine]["compute_device"]

    try:
        webcam_effects_runner.start(
            camera_index=req.camera_index,
            engine=engine,
            compute_device=compute_device,
            effect=req.effect,
            color=_hex_to_bgr(req.color),
        )
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)

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
    webcam_effects_runner.set_effect(req.effect, _hex_to_bgr(req.color) if req.color else None)
    return JSONResponse({"status": "ok"})


@app.get("/api/webcam-effects/stats")
def webcam_effects_stats() -> JSONResponse:
    return JSONResponse({**webcam_effects_runner.latest_stats(), "error": webcam_effects_runner.error})


@app.get("/api/webcam-effects/stream")
def webcam_effects_stream() -> StreamingResponse:
    def generate():
        last_sent = None
        while webcam_effects_runner.running:
            jpeg = webcam_effects_runner.latest_jpeg()
            if jpeg is not None and jpeg is not last_sent:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                last_sent = jpeg
            time.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/voice-clone-studio/devices")
def voice_clone_studio_devices() -> JSONResponse:
    return JSONResponse(
        {
            "microphones": audio.list_microphones(),
            "openvino_devices": list_openvino_devices(),
            "samples": [asdict(s) for s in VOICE_CLONE_STUDIO_SAMPLES],
        }
    )


class VoiceCloneStudioEnrollRecordRequest(BaseModel):
    seconds: float = 10.0
    engine: str = "portable"
    compute_device: str | None = None


@app.post("/api/voice-clone-studio/enroll-record")
async def voice_clone_studio_enroll_record(req: VoiceCloneStudioEnrollRecordRequest) -> JSONResponse:
    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    compute_device = req.compute_device or _VOICE_CLONE_STUDIO_ENGINE_DEFAULTS[engine]["compute_device"]

    def work():
        reference_path = voice_clone_studio_runner.record_reference(req.seconds)
        voice_clone_studio_runner.enroll(reference_path=reference_path, engine=req.engine, device=compute_device)

    try:
        await run_in_threadpool(work)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse({"status": "enrolled"})


@app.post("/api/voice-clone-studio/enroll-upload")
async def voice_clone_studio_enroll_upload(
    file: UploadFile,
    engine: str = Form("portable"),
    compute_device: str | None = Form(None),
) -> JSONResponse:
    try:
        engine_enum = Engine(engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{engine}'"}, status_code=400)

    resolved_device = compute_device or _VOICE_CLONE_STUDIO_ENGINE_DEFAULTS[engine_enum]["compute_device"]
    file_bytes = await file.read()
    suffix = Path(file.filename or "reference.wav").suffix or ".wav"

    def work():
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, file_bytes)
            os.close(fd)
            voice_clone_studio_runner.enroll(reference_path=path, engine=engine, device=resolved_device)
        finally:
            os.unlink(path)

    try:
        await run_in_threadpool(work)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

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
        import io

        import soundfile as sf

        audio_out, sample_rate = voice_clone_studio_runner.synthesize(text=req.text, style=req.style, tau=req.tau)
        buffer = io.BytesIO()
        sf.write(buffer, audio_out, sample_rate, format="WAV")
        return buffer.getvalue()

    try:
        wav_bytes = await run_in_threadpool(work)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/api/voice-assistant/devices")
def voice_assistant_devices() -> JSONResponse:
    from voice_assistant.wake_word import AVAILABLE_WAKE_WORDS

    return JSONResponse(
        {
            "microphones": audio.list_microphones(),
            "openvino_devices": list_openvino_devices(),
            "wake_words": AVAILABLE_WAKE_WORDS,
        }
    )


class VoiceAssistantStartRequest(BaseModel):
    audio_device: str | None = None
    engine: str = "portable"
    whisper_model: str | None = None
    compute_device: str | None = None
    wake_word: str = "hey_jarvis"
    wake_threshold: float = 0.5
    speak_replies: bool = True


@app.post("/api/voice-assistant/start")
async def start_voice_assistant(req: VoiceAssistantStartRequest) -> JSONResponse:
    if voice_assistant_runner.running:
        return JSONResponse({"error": "voice-assistant is already running"}, status_code=409)

    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    defaults = _VOICE_ASSISTANT_ENGINE_DEFAULTS[engine]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = app.state.voice_assistant_queue

    try:
        voice_assistant_runner.start(
            loop=loop,
            queue=queue,
            audio_device=req.audio_device,
            engine=engine,
            whisper_model_size=req.whisper_model or defaults["whisper_model"],
            compute_device=req.compute_device or defaults["compute_device"],
            wake_word=req.wake_word,
            wake_threshold=req.wake_threshold,
            speak_replies=req.speak_replies,
        )
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)

    return JSONResponse({"status": "started"})


@app.post("/api/voice-assistant/stop")
async def stop_voice_assistant() -> JSONResponse:
    voice_assistant_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws/voice-assistant")
async def ws_voice_assistant(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = app.state.voice_assistant_queue
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass


@app.get("/api/expense-extract/devices")
def expense_extract_devices() -> JSONResponse:
    return JSONResponse({"openvino_devices": list_openvino_devices()})


class ExpenseExtractStartRequest(BaseModel):
    folder: str
    ocr_engine: str = "portable"
    ocr_compute_device: str | None = None
    llm_engine: str = "portable"
    llm_compute_device: str | None = None


@app.post("/api/expense-extract/start")
async def start_expense_extract(req: ExpenseExtractStartRequest) -> JSONResponse:
    if expense_extract_runner.running:
        return JSONResponse({"error": "expense-extract is already running"}, status_code=409)

    try:
        ocr_engine = Engine(req.ocr_engine)
        llm_engine = Engine(req.llm_engine)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    ocr_device = req.ocr_compute_device or _EXPENSE_EXTRACT_ENGINE_DEFAULTS[ocr_engine]["compute_device"]
    llm_device = req.llm_compute_device or _EXPENSE_EXTRACT_ENGINE_DEFAULTS[llm_engine]["compute_device"]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = app.state.expense_extract_queue

    try:
        expense_extract_runner.start(
            loop=loop,
            queue=queue,
            folder=req.folder,
            ocr_engine=ocr_engine,
            ocr_device=ocr_device,
            llm_engine=llm_engine,
            llm_device=llm_device,
        )
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)

    return JSONResponse({"status": "started"})


@app.post("/api/expense-extract/stop")
async def stop_expense_extract() -> JSONResponse:
    expense_extract_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws/expense-extract")
async def ws_expense_extract(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = app.state.expense_extract_queue
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass


@app.get("/api/smart-recall/devices")
def smart_recall_devices() -> JSONResponse:
    return JSONResponse(
        {
            "screens": video.list_screens(),
            "openvino_devices": list_openvino_devices(),
            "samples": [asdict(s) for s in SMART_RECALL_SAMPLES],
        }
    )


@app.get("/api/smart-recall/status")
def smart_recall_status() -> JSONResponse:
    from smart_recall.pipeline import index_status

    return JSONResponse({"running": smart_recall_runner.running, **index_status()})


class SmartRecallStartRequest(BaseModel):
    screen_index: int = 1
    interval_seconds: float = 5.0
    ocr_engine: str = "portable"
    ocr_compute_device: str | None = None
    embed_engine: str = "portable"
    embed_compute_device: str | None = None


@app.post("/api/smart-recall/start")
async def start_smart_recall(req: SmartRecallStartRequest) -> JSONResponse:
    if smart_recall_runner.running:
        return JSONResponse({"error": "smart-recall is already running"}, status_code=409)

    try:
        ocr_engine = Engine(req.ocr_engine)
        embed_engine = Engine(req.embed_engine)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    ocr_device = req.ocr_compute_device or _SMART_RECALL_ENGINE_DEFAULTS[ocr_engine]["compute_device"]
    embed_device = req.embed_compute_device or _SMART_RECALL_ENGINE_DEFAULTS[embed_engine]["compute_device"]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = app.state.smart_recall_queue

    try:
        smart_recall_runner.start(
            loop=loop,
            queue=queue,
            screen_index=req.screen_index,
            interval_seconds=req.interval_seconds,
            ocr_engine=ocr_engine,
            ocr_device=ocr_device,
            embed_engine=embed_engine,
            embed_device=embed_device,
        )
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)

    return JSONResponse({"status": "started"})


@app.post("/api/smart-recall/stop")
async def stop_smart_recall() -> JSONResponse:
    smart_recall_runner.stop()
    return JSONResponse({"status": "stopped"})


@app.post("/api/smart-recall/reset")
async def reset_smart_recall() -> JSONResponse:
    try:
        await run_in_threadpool(smart_recall_runner.reset)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse({"status": "reset"})


@app.websocket("/ws/smart-recall")
async def ws_smart_recall(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue = app.state.smart_recall_queue
    try:
        while True:
            message = await queue.get()
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass


class SmartRecallSearchRequest(BaseModel):
    question: str
    top_k: int = 5
    compute_device: str = "AUTO"


@app.post("/api/smart-recall/search")
async def search_smart_recall(req: SmartRecallSearchRequest) -> JSONResponse:
    try:
        results = await run_in_threadpool(
            smart_recall_runner.search, question=req.question, top_k=req.top_k, device=req.compute_device
        )
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

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
def smart_recall_screenshot(filename: str) -> FileResponse:
    from smart_recall.pipeline import SCREENSHOTS_DIR

    # Strip any path components -- filenames come from chunk.source, which
    # this brick only ever generates itself, but a route parameter is
    # still untrusted input on principle.
    safe_name = Path(filename).name
    path = SCREENSHOTS_DIR / safe_name
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/code-review-assist/devices")
def code_review_assist_devices() -> JSONResponse:
    return JSONResponse(
        {"openvino_devices": list_openvino_devices(), "samples": [asdict(s) for s in CODE_REVIEW_ASSIST_SAMPLES]}
    )


class CodeReviewRequest(BaseModel):
    source: str = "worktree"  # "worktree" | "diff_text"
    folder: str | None = None
    against: str = "HEAD"
    diff_text: str | None = None
    engine: str = "portable"
    compute_device: str | None = None


@app.post("/api/code-review-assist/review")
async def code_review_assist_review(req: CodeReviewRequest) -> JSONResponse:
    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    device = req.compute_device or _CODE_REVIEW_ASSIST_ENGINE_DEFAULTS[engine]["compute_device"]

    try:
        result = await run_in_threadpool(
            code_review_assist_runner.review,
            engine=req.engine,
            device=device,
            folder=req.folder if req.source == "worktree" else None,
            against=req.against,
            diff_text=req.diff_text if req.source == "diff_text" else None,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse(
        {
            "commit_message": result.commit_message,
            "review_notes": result.review_notes,
            "diff_char_count": result.diff_char_count,
            "diff_truncated": result.diff_truncated,
        }
    )


@app.get("/api/html-creator/devices")
def html_creator_devices() -> JSONResponse:
    return JSONResponse(
        {"openvino_devices": list_openvino_devices(), "samples": [asdict(s) for s in HTML_CREATOR_SAMPLES]}
    )


class HtmlCreatorRequest(BaseModel):
    mode: str = "landing_page"  # "landing_page" | "document"
    prompt: str | None = None
    folder: str | None = None
    engine: str = "portable"
    compute_device: str | None = None


@app.post("/api/html-creator/generate")
async def html_creator_generate(req: HtmlCreatorRequest) -> JSONResponse:
    try:
        engine = Engine(req.engine)
    except ValueError:
        return JSONResponse({"error": f"unknown engine '{req.engine}'"}, status_code=400)

    device = req.compute_device or _HTML_CREATOR_ENGINE_DEFAULTS[engine]["compute_device"]

    try:
        result = await run_in_threadpool(
            html_creator_runner.generate,
            engine=req.engine,
            device=device,
            mode=req.mode,
            prompt=req.prompt,
            folder=req.folder,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

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


def main() -> None:
    url = "http://127.0.0.1:8765"
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
