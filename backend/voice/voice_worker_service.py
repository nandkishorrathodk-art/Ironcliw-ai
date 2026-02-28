"""
Dedicated Python voice worker process for sidecar supervision.

This module intentionally delegates all model/inference logic to existing
voice modules. It only provides process lifecycle and health contract endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
import uvicorn

logger = logging.getLogger("voice_worker_service")
logging.basicConfig(
    level=os.getenv("Ironcliw_VOICE_WORKER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)


@dataclass
class WorkerState:
    started_at: float
    warmup_started_at: Optional[float] = None
    warmup_completed_at: Optional[float] = None
    warmup_error: Optional[str] = None
    warmup_attempts: int = 0
    ready: bool = False


app = FastAPI(title="Ironcliw Voice Worker", version="1.0")
_state = WorkerState(started_at=time.time())
_warmup_lock = asyncio.Lock()
_service_ref: Any = None


def _service_snapshot() -> Dict[str, Any]:
    svc = _service_ref
    if svc is None:
        return {
            "loaded": False,
            "initialized": False,
            "encoder_preloaded": False,
            "encoder_preloading": False,
        }
    return {
        "loaded": True,
        "initialized": bool(getattr(svc, "initialized", False)),
        "encoder_preloaded": bool(getattr(svc, "_encoder_preloaded", False)),
        "encoder_preloading": bool(getattr(svc, "_encoder_preloading", False)),
    }


async def _import_speaker_service_factory():
    try:
        from backend.voice.speaker_verification_service import (
            get_speaker_verification_service,
        )
    except Exception:
        from voice.speaker_verification_service import get_speaker_verification_service
    return get_speaker_verification_service


async def warmup_voice_service(force: bool = False) -> Dict[str, Any]:
    global _service_ref

    async with _warmup_lock:
        if _state.ready and not force:
            return {
                "status": "already_ready",
                "service": _service_snapshot(),
            }

        _state.warmup_attempts += 1
        _state.warmup_started_at = time.time()
        _state.warmup_error = None

        timeout_s = float(os.getenv("Ironcliw_VOICE_WORKER_WARMUP_TIMEOUT", "45"))

        try:
            factory = await _import_speaker_service_factory()
            _service_ref = await asyncio.wait_for(factory(), timeout=timeout_s)

            # initialize_fast is idempotent and keeps heavyweight encoder preload
            # in existing module logic.
            if hasattr(_service_ref, "initialize_fast"):
                await asyncio.wait_for(_service_ref.initialize_fast(), timeout=timeout_s)

            _state.ready = True
            _state.warmup_completed_at = time.time()

            return {
                "status": "ready",
                "service": _service_snapshot(),
            }
        except Exception as exc:
            _state.ready = False
            _state.warmup_error = str(exc)
            _state.warmup_completed_at = time.time()
            logger.exception("Voice worker warmup failed")
            raise


@app.on_event("startup")
async def _on_startup() -> None:
    auto_warmup = os.getenv("Ironcliw_VOICE_WORKER_AUTO_WARMUP", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not auto_warmup:
        return

    async def _background_warmup() -> None:
        try:
            await warmup_voice_service(force=False)
        except Exception:
            # Keep process alive for supervised recovery and explicit warmup retries.
            pass

    asyncio.create_task(_background_warmup(), name="voice_worker_auto_warmup")


@app.get("/health")
async def health() -> Dict[str, Any]:
    now = time.time()
    service = _service_snapshot()
    healthy = _state.ready or service["loaded"]

    return {
        "status": "ok" if healthy else "degraded",
        "ready": bool(_state.ready),
        "uptime_seconds": round(now - _state.started_at, 3),
        "state": asdict(_state),
        "service": service,
    }


@app.get("/ready")
async def ready() -> Dict[str, Any]:
    if not _state.ready:
        raise HTTPException(status_code=503, detail="voice_worker_not_ready")
    return {
        "ready": True,
        "service": _service_snapshot(),
    }


@app.post("/warmup")
async def warmup_endpoint(force: bool = False) -> Dict[str, Any]:
    try:
        result = await warmup_voice_service(force=force)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result


def main() -> None:
    host = os.getenv("Ironcliw_VOICE_WORKER_HOST", "127.0.0.1")
    port = int(os.getenv("Ironcliw_VOICE_WORKER_PORT", "8790"))
    log_level = os.getenv("Ironcliw_VOICE_WORKER_UVICORN_LOG_LEVEL", "warning")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
