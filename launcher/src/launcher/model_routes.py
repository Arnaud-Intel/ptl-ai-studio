"""Which models the demos need, fetching them before a show (BACKLOG R10),
and compiling them for this machine's chips (`compile_ahead`).

A router rather than more of app.py: the inventory, the sizes and the
progress all live in `pantherlake_ai_core.prefetch`, so the launcher only
has to expose them and keep one prefetcher for the process.
"""
from __future__ import annotations

from typing import Callable

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pantherlake_ai_core.prefetch import Prefetcher
from pydantic import BaseModel

from . import compile_ahead
from .errors import Conflict

router = APIRouter(prefix="/api/models")
prefetcher = Prefetcher()
compiler = compile_ahead.compiler
# The demos loading or running now, by name. app.py puts its own list here:
# a compile shares the chips with them, and waits until they have stopped.
busy_demos: Callable[[], list[str]] = lambda: []  # noqa: E731


def _compile_status() -> dict:
    # "Is it downloaded" as the prefetcher has it for a few seconds: asked
    # afresh for every model at every poll, it walks the Hub's cache each time.
    return compiler.status(is_cached=lambda spec: prefetcher._cache_state(spec)[0])


class PrefetchRequest(BaseModel):
    # Which models to fetch; empty means everything still missing.
    keys: list[str] | None = None


@router.get("")
def model_status() -> JSONResponse:
    """The inventory with what's cached, what's missing and -- once the Hub
    has answered -- what the missing ones weigh."""
    prefetcher.sizes_in_background()
    return JSONResponse({**prefetcher.status(), "compile": _compile_status()})


@router.post("/prefetch")
def start_prefetch(request: PrefetchRequest) -> JSONResponse:
    if compiler.running:
        return JSONResponse({"error": "Models are being compiled: download once that is done, or stop it."}, status_code=409)
    try:
        started = prefetcher.start(request.keys or None)
    except ValueError as exc:  # an unknown key: the caller sent something wrong
        return JSONResponse({"error": str(exc)}, status_code=400)
    except RuntimeError as exc:  # already running, or nothing to do
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse({"started": started}, status_code=202)


@router.post("/prefetch/stop")
def stop_prefetch() -> JSONResponse:
    prefetcher.stop()
    return JSONResponse({"stopping": True})


class CompileRequest(BaseModel):
    # The chips to compile for, by kind ("npu", "igpu", "dgpu"); none named means all of them.
    chips: list[str] | None = None


@router.post("/compile")
def start_compile(request: CompileRequest) -> JSONResponse:
    """Compile what is downloaded and not yet compiled, for the chips named."""
    if prefetcher.running:
        return JSONResponse({"error": "Models are being downloaded: compile once that is done."}, status_code=409)
    try:
        started = compiler.start(request.chips or None, busy=busy_demos())
    except ValueError as exc:  # a chip this machine does not have
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Conflict as exc:  # demos running, already compiling, or nothing to do
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse({"started": started}, status_code=202)


@router.post("/compile/stop")
def stop_compile() -> JSONResponse:
    compiler.stop()
    return JSONResponse({"stopping": True})
