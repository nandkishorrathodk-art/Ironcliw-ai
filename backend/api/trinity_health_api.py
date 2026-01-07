"""
Trinity Health API v1.0
========================

Health monitoring endpoints for the Trinity ecosystem.

Provides real-time status of:
- JARVIS-Prime (Mind) - Local AI inference
- Reactor-Core (Nerves) - Orchestration layer
- Graceful Degradation status
- Routing metrics
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["trinity"])


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class ComponentStatus(BaseModel):
    """Status of a Trinity component."""
    name: str
    status: str  # healthy, degraded, unhealthy, unknown
    latency_ms: Optional[float] = None
    last_check: Optional[float] = None
    details: Dict[str, Any] = {}


class TrinityHealthResponse(BaseModel):
    """Full Trinity health response."""
    overall_status: str
    components: Dict[str, ComponentStatus]
    routing: Dict[str, Any]
    graceful_degradation: Dict[str, Any]
    timestamp: float


class PrimeStatusResponse(BaseModel):
    """JARVIS-Prime specific status."""
    status: str
    available: bool
    circuit_breaker: Dict[str, Any]
    metrics: Dict[str, Any]
    config: Dict[str, Any]


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@router.get("/trinity", response_model=TrinityHealthResponse)
async def get_trinity_health() -> TrinityHealthResponse:
    """
    Get comprehensive health status of the Trinity ecosystem.

    Returns status of:
    - JARVIS-Prime (local AI)
    - Reactor-Core (orchestration)
    - Prime Router
    - Graceful Degradation
    """
    components = {}
    overall_healthy = True

    # Check JARVIS-Prime (Mind)
    prime_status = await _check_prime_health()
    components["jarvis_prime"] = prime_status
    if prime_status.status not in ("healthy", "unknown"):
        overall_healthy = False

    # Check Prime Router
    router_status = await _check_router_health()
    components["prime_router"] = router_status
    if router_status.status == "unhealthy":
        overall_healthy = False

    # Check Reactor-Core (Nerves) via heartbeat file
    reactor_status = await _check_reactor_health()
    components["reactor_core"] = reactor_status

    # Get routing metrics
    routing_info = await _get_routing_metrics()

    # Get graceful degradation status
    degradation_info = await _get_degradation_status()

    return TrinityHealthResponse(
        overall_status="healthy" if overall_healthy else "degraded",
        components=components,
        routing=routing_info,
        graceful_degradation=degradation_info,
        timestamp=time.time(),
    )


@router.get("/prime", response_model=PrimeStatusResponse)
async def get_prime_health() -> PrimeStatusResponse:
    """Get detailed JARVIS-Prime status."""
    try:
        try:
            from core.prime_router import get_prime_router
        except ImportError:
            from backend.core.prime_router import get_prime_router

        router = await get_prime_router()
        status = router.get_status()

        prime_available = (
            status.get("prime_client", {}).get("available", False) and
            status.get("prime_client", {}).get("status", {}).get("status") in ("available", "degraded")
        )

        return PrimeStatusResponse(
            status="available" if prime_available else "unavailable",
            available=prime_available,
            circuit_breaker=status.get("prime_client", {}).get("status", {}).get("circuit_breaker", {}),
            metrics=status.get("metrics", {}),
            config=status.get("config", {}),
        )

    except Exception as e:
        logger.error(f"[TrinityHealth] Error getting Prime status: {e}")
        return PrimeStatusResponse(
            status="error",
            available=False,
            circuit_breaker={},
            metrics={},
            config={"error": str(e)},
        )


@router.get("/routing")
async def get_routing_status() -> Dict[str, Any]:
    """Get current AI routing status and metrics."""
    return await _get_routing_metrics()


@router.get("/degradation")
async def get_degradation_status() -> Dict[str, Any]:
    """Get graceful degradation system status."""
    return await _get_degradation_status()


@router.post("/prime/reset-circuit")
async def reset_prime_circuit() -> Dict[str, Any]:
    """Reset the Prime client circuit breaker."""
    try:
        try:
            from core.prime_client import get_prime_client
        except ImportError:
            from backend.core.prime_client import get_prime_client

        client = await get_prime_client()
        # Reset is done via the circuit breaker
        if hasattr(client, '_circuit'):
            client._circuit._state.state = "closed"
            client._circuit._state.failures = 0
            client._circuit._state.circuit_open = False

        return {"success": True, "message": "Circuit breaker reset"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _check_prime_health() -> ComponentStatus:
    """Check JARVIS-Prime health."""
    try:
        try:
            from core.prime_client import get_prime_client
        except ImportError:
            from backend.core.prime_client import get_prime_client

        start = time.time()
        client = await get_prime_client()
        latency = (time.time() - start) * 1000

        status = client.get_status()
        prime_status = status.get("status", "unknown")

        return ComponentStatus(
            name="JARVIS-Prime (Mind)",
            status=prime_status if prime_status in ("available", "degraded") else "unhealthy",
            latency_ms=latency,
            last_check=time.time(),
            details=status,
        )

    except Exception as e:
        return ComponentStatus(
            name="JARVIS-Prime (Mind)",
            status="unknown",
            details={"error": str(e)},
        )


async def _check_router_health() -> ComponentStatus:
    """Check Prime Router health."""
    try:
        try:
            from core.prime_router import get_prime_router
        except ImportError:
            from backend.core.prime_router import get_prime_router

        start = time.time()
        router = await get_prime_router()
        latency = (time.time() - start) * 1000

        status = router.get_status()
        is_healthy = status.get("initialized", False)

        return ComponentStatus(
            name="Prime Router",
            status="healthy" if is_healthy else "unhealthy",
            latency_ms=latency,
            last_check=time.time(),
            details=status,
        )

    except Exception as e:
        return ComponentStatus(
            name="Prime Router",
            status="unhealthy",
            details={"error": str(e)},
        )


async def _check_reactor_health() -> ComponentStatus:
    """Check Reactor-Core health via heartbeat file."""
    try:
        import json

        trinity_dir = os.getenv("TRINITY_DIR", os.path.expanduser("~/.jarvis/trinity"))
        heartbeat_file = os.path.join(trinity_dir, "components", "reactor_core.json")

        if os.path.exists(heartbeat_file):
            with open(heartbeat_file, 'r') as f:
                data = json.load(f)

            last_heartbeat = data.get("timestamp", 0)
            age = time.time() - last_heartbeat

            if age < 30:
                status = "healthy"
            elif age < 60:
                status = "degraded"
            else:
                status = "unhealthy"

            return ComponentStatus(
                name="Reactor-Core (Nerves)",
                status=status,
                last_check=last_heartbeat,
                details={
                    "heartbeat_age_seconds": round(age, 1),
                    "version": data.get("version", "unknown"),
                    "orchestrator_state": data.get("state", "unknown"),
                },
            )
        else:
            return ComponentStatus(
                name="Reactor-Core (Nerves)",
                status="unknown",
                details={"error": "Heartbeat file not found"},
            )

    except Exception as e:
        return ComponentStatus(
            name="Reactor-Core (Nerves)",
            status="unknown",
            details={"error": str(e)},
        )


async def _get_routing_metrics() -> Dict[str, Any]:
    """Get routing metrics from Prime Router."""
    try:
        try:
            from core.prime_router import get_prime_router
        except ImportError:
            from backend.core.prime_router import get_prime_router

        router = await get_prime_router()
        status = router.get_status()

        return {
            "initialized": status.get("initialized", False),
            "metrics": status.get("metrics", {}),
            "config": status.get("config", {}),
            "prime_available": status.get("prime_client", {}).get("available", False),
            "cloud_available": status.get("cloud_client", {}).get("available", False),
        }

    except Exception as e:
        return {"error": str(e)}


async def _get_degradation_status() -> Dict[str, Any]:
    """Get graceful degradation status."""
    try:
        try:
            from core.graceful_degradation import get_degradation
        except ImportError:
            from backend.core.graceful_degradation import get_degradation

        degradation = get_degradation()
        return degradation.get_status()

    except Exception as e:
        return {"error": str(e), "available": False}
