"""Which models the demos need, and fetching them before a show (BACKLOG R10).

A router rather than more of app.py: the inventory, the sizes and the
progress all live in `pantherlake_ai_core.prefetch`, so the launcher only
has to expose them and keep one prefetcher for the process.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pantherlake_ai_core.prefetch import Prefetcher
from pydantic import BaseModel

router = APIRouter(prefix="/api/models")
prefetcher = Prefetcher()


class PrefetchRequest(BaseModel):
    # Which models to fetch; empty means everything still missing.
    keys: list[str] | None = None


@router.get("")
def model_status() -> JSONResponse:
    """The inventory with what's cached, what's missing and -- once the Hub
    has answered -- what the missing ones weigh."""
    prefetcher.sizes_in_background()
    return JSONResponse(prefetcher.status())


@router.post("/prefetch")
def start_prefetch(request: PrefetchRequest) -> JSONResponse:
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
