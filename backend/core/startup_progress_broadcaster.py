"""
JARVIS Startup Progress Broadcaster v1.0.0
==========================================

Real-time WebSocket broadcaster for startup progress.
Sends detailed updates to the frontend loading page.

Features:
- WebSocket broadcast to all connected clients
- HTTP polling endpoint for fallback
- Detailed component-level progress
- Memory status and mode information
- Live log streaming
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set
from weakref import WeakSet

logger = logging.getLogger(__name__)


@dataclass
class StartupEvent:
    """A single startup event for the progress stream"""
    timestamp: float
    event_type: str  # 'component_start', 'component_complete', 'component_fail', 'log', 'progress', 'complete'
    component: Optional[str] = None
    message: str = ""
    progress: float = 0.0
    phase: str = "initializing"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "component": self.component,
            "message": self.message,
            "progress": self.progress,
            "phase": self.phase,
            "metadata": self.metadata or {}
        }


class StartupProgressBroadcaster:
    """
    Broadcasts startup progress to WebSocket clients in real-time.

    Usage:
        broadcaster = get_startup_broadcaster()
        await broadcaster.broadcast_component_start("cloud_sql_proxy", "Connecting to Cloud SQL...")
        await broadcaster.broadcast_progress(25, "Loading databases...")
    """

    _instance: Optional["StartupProgressBroadcaster"] = None

    def __init__(self):
        self.websocket_clients: Set[Any] = set()
        self._events: List[StartupEvent] = []
        self._max_events = 500  # Keep last 500 events
        self._lock = asyncio.Lock()
        self._started_at: float = time.time()

        # Current state
        self._current_progress = 0.0
        self._current_phase = "initializing"
        self._current_message = "Starting JARVIS..."
        self._current_component: Optional[str] = None
        self._components_status: Dict[str, Dict[str, Any]] = {}
        self._memory_info: Optional[Dict[str, Any]] = None
        self._startup_mode: Optional[str] = None

        # Component definitions with weights
        self._component_weights = {
            "config": 2,
            "cloud_sql_proxy": 5,
            "learning_database": 10,
            "memory_aware_startup": 5,
            "cloud_ml_router": 5,
            "cloud_ecapa_client": 8,
            "vbi_prewarm": 10,
            "ml_engine_registry": 8,
            "speaker_verification": 8,
            "voice_unlock_api": 5,
            "jarvis_voice_api": 5,
            "unified_websocket": 5,
            "neural_mesh": 8,
            "goal_inference": 3,
            "uae_engine": 5,
            "hybrid_orchestrator": 5,
            "vision_analyzer": 5,
            "display_monitor": 3,
            "dynamic_components": 5,
        }
        self._total_weight = sum(self._component_weights.values())
        self._completed_weight = 0

    @classmethod
    def get_instance(cls) -> "StartupProgressBroadcaster":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_websocket(self, websocket):
        """Register a WebSocket client for progress updates"""
        self.websocket_clients.add(websocket)
        logger.debug(f"[StartupBroadcaster] WebSocket client registered ({len(self.websocket_clients)} total)")

    def unregister_websocket(self, websocket):
        """Unregister a WebSocket client"""
        self.websocket_clients.discard(websocket)
        logger.debug(f"[StartupBroadcaster] WebSocket client unregistered ({len(self.websocket_clients)} total)")

    async def _broadcast(self, event: StartupEvent):
        """Broadcast an event to all connected WebSocket clients"""
        async with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

        # Prepare message
        message = json.dumps(event.to_dict())

        # Send to all clients
        dead_clients = set()
        for ws in self.websocket_clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead_clients.add(ws)

        # Clean up dead clients
        self.websocket_clients -= dead_clients

    def _calculate_progress(self) -> float:
        """Calculate actual progress based on component weights"""
        if self._total_weight == 0:
            return 0.0
        return min(100.0, (self._completed_weight / self._total_weight) * 100)

    async def broadcast_component_start(
        self,
        component: str,
        message: str,
        substep: int = 0,
        total_substeps: int = 1
    ):
        """Broadcast that a component is starting"""
        self._current_component = component
        self._current_message = message
        self._components_status[component] = {
            "status": "running",
            "started_at": time.time(),
            "substep": substep,
            "total_substeps": total_substeps,
            "message": message
        }

        event = StartupEvent(
            timestamp=time.time(),
            event_type="component_start",
            component=component,
            message=message,
            progress=self._calculate_progress(),
            phase=self._current_phase,
            metadata={
                "substep": substep,
                "total_substeps": total_substeps,
                "elapsed_ms": (time.time() - self._started_at) * 1000
            }
        )
        await self._broadcast(event)

    async def broadcast_component_progress(
        self,
        component: str,
        message: str,
        substep: int,
        total_substeps: int = 1
    ):
        """Broadcast progress within a component"""
        self._current_message = message
        if component in self._components_status:
            self._components_status[component]["substep"] = substep
            self._components_status[component]["message"] = message

        event = StartupEvent(
            timestamp=time.time(),
            event_type="component_progress",
            component=component,
            message=message,
            progress=self._calculate_progress(),
            phase=self._current_phase,
            metadata={
                "substep": substep,
                "total_substeps": total_substeps
            }
        )
        await self._broadcast(event)

    async def broadcast_component_complete(
        self,
        component: str,
        message: str = "",
        duration_ms: Optional[float] = None
    ):
        """Broadcast that a component completed successfully"""
        weight = self._component_weights.get(component, 5)
        self._completed_weight += weight

        self._components_status[component] = {
            "status": "complete",
            "completed_at": time.time(),
            "duration_ms": duration_ms,
            "message": message or f"{component} ready"
        }

        self._current_progress = self._calculate_progress()
        self._current_message = message or f"{component} initialized"

        event = StartupEvent(
            timestamp=time.time(),
            event_type="component_complete",
            component=component,
            message=message or f"{component} ready",
            progress=self._current_progress,
            phase=self._current_phase,
            metadata={
                "duration_ms": duration_ms,
                "components_ready": sum(
                    1 for c in self._components_status.values()
                    if c.get("status") == "complete"
                ),
                "total_components": len(self._component_weights)
            }
        )
        await self._broadcast(event)

    async def broadcast_component_failed(
        self,
        component: str,
        error: str,
        is_critical: bool = False
    ):
        """Broadcast that a component failed"""
        # Still count partial progress for failed components
        weight = self._component_weights.get(component, 5) * 0.5  # Half credit for attempted
        self._completed_weight += weight

        self._components_status[component] = {
            "status": "failed",
            "error": error,
            "is_critical": is_critical,
            "failed_at": time.time()
        }

        event = StartupEvent(
            timestamp=time.time(),
            event_type="component_fail",
            component=component,
            message=f"Failed: {error}",
            progress=self._calculate_progress(),
            phase=self._current_phase,
            metadata={
                "error": error,
                "is_critical": is_critical
            }
        )
        await self._broadcast(event)

    async def broadcast_log(self, message: str, level: str = "info"):
        """Broadcast a log message"""
        event = StartupEvent(
            timestamp=time.time(),
            event_type="log",
            message=message,
            progress=self._current_progress,
            phase=self._current_phase,
            metadata={"level": level}
        )
        await self._broadcast(event)

    async def broadcast_memory_info(
        self,
        available_gb: float,
        used_gb: float,
        total_gb: float,
        pressure_percent: float,
        startup_mode: str
    ):
        """Broadcast memory status update"""
        self._memory_info = {
            "available_gb": available_gb,
            "used_gb": used_gb,
            "total_gb": total_gb,
            "pressure_percent": pressure_percent
        }
        self._startup_mode = startup_mode

        event = StartupEvent(
            timestamp=time.time(),
            event_type="memory_update",
            message=f"Memory: {available_gb:.1f}GB available, Mode: {startup_mode}",
            progress=self._current_progress,
            phase=self._current_phase,
            metadata={
                "memory_available_gb": available_gb,
                "memory_used_gb": used_gb,
                "memory_total_gb": total_gb,
                "memory_pressure": pressure_percent,
                "startup_mode": startup_mode
            }
        )
        await self._broadcast(event)

    async def broadcast_phase_change(self, phase: str, message: str):
        """Broadcast a phase change"""
        self._current_phase = phase
        self._current_message = message

        event = StartupEvent(
            timestamp=time.time(),
            event_type="phase_change",
            message=message,
            progress=self._current_progress,
            phase=phase,
            metadata={"previous_phase": self._current_phase}
        )
        await self._broadcast(event)

    async def broadcast_complete(self, success: bool = True, message: str = ""):
        """Broadcast startup complete"""
        self._current_progress = 100.0 if success else self._current_progress
        self._current_phase = "complete" if success else "failed"
        self._current_message = message or ("JARVIS Online" if success else "Startup failed")

        elapsed_seconds = time.time() - self._started_at

        event = StartupEvent(
            timestamp=time.time(),
            event_type="complete",
            message=self._current_message,
            progress=self._current_progress,
            phase=self._current_phase,
            metadata={
                "success": success,
                "elapsed_seconds": elapsed_seconds,
                "components_ready": sum(
                    1 for c in self._components_status.values()
                    if c.get("status") == "complete"
                ),
                "components_failed": sum(
                    1 for c in self._components_status.values()
                    if c.get("status") == "failed"
                ),
                "redirect_url": "http://localhost:3000"
            }
        )
        await self._broadcast(event)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current startup state for HTTP polling"""
        return {
            "progress": self._current_progress,
            "phase": self._current_phase,
            "message": self._current_message,
            "current_component": self._current_component,
            "components": self._components_status,
            "memory": self._memory_info,
            "startup_mode": self._startup_mode,
            "elapsed_seconds": time.time() - self._started_at,
            "events_count": len(self._events)
        }

    def get_recent_events(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent events for clients that just connected"""
        return [e.to_dict() for e in self._events[-count:]]


# Global singleton getter
def get_startup_broadcaster() -> StartupProgressBroadcaster:
    """Get the global startup progress broadcaster"""
    return StartupProgressBroadcaster.get_instance()
