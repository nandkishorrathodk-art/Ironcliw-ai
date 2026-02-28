"""
Cross-Repository Intelligence Bridge for Ironcliw
================================================

Integrates Repository Intelligence with the cross-repo event system,
enabling seamless communication of codebase context between:
- Ironcliw-AI-Agent (main orchestrator)
- Ironcliw Prime (hybrid inference)
- Reactor Core (training pipeline)

This bridge:
- Emits repository intelligence events to the event bus
- Provides repository context to Ironcliw Prime for better inference
- Notifies Reactor Core of codebase changes for adaptive training

Author: Ironcliw AI System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration (Environment-Driven)
# ============================================================================

def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_env_path(key: str, default: str) -> Path:
    return Path(os.path.expanduser(_get_env(key, default)))


def _get_env_bool(key: str, default: bool = False) -> bool:
    return _get_env(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class CrossRepoBridgeConfig:
    """Configuration for the cross-repo intelligence bridge."""
    enabled: bool = field(
        default_factory=lambda: _get_env_bool("Ironcliw_CROSS_REPO_BRIDGE_ENABLED", True)
    )
    state_dir: Path = field(
        default_factory=lambda: _get_env_path("Ironcliw_CROSS_REPO_DIR", "~/.jarvis/cross_repo")
    )
    events_file: str = field(
        default_factory=lambda: _get_env("Ironcliw_REPO_EVENTS_FILE", "repo_intelligence_events.json")
    )
    max_events_stored: int = field(
        default_factory=lambda: int(_get_env("Ironcliw_MAX_REPO_EVENTS", "100"))
    )
    emit_to_prime: bool = field(
        default_factory=lambda: _get_env_bool("Ironcliw_EMIT_TO_PRIME", True)
    )
    emit_to_reactor: bool = field(
        default_factory=lambda: _get_env_bool("Ironcliw_EMIT_TO_REACTOR", True)
    )


# ============================================================================
# Event Types
# ============================================================================

class RepoIntelligenceEventType(str, Enum):
    """Types of repository intelligence events."""
    REPO_MAP_GENERATED = "repo_map_generated"
    REPO_MAP_CACHED = "repo_map_cached"
    REPO_ANALYSIS_COMPLETE = "repo_analysis_complete"
    REPO_SYMBOL_FOUND = "repo_symbol_found"
    REPO_DEPENDENCY_DETECTED = "repo_dependency_detected"
    REPO_CONTEXT_ENRICHED = "repo_context_enriched"


@dataclass
class RepoIntelligenceEvent:
    """An event from the Repository Intelligence system."""
    event_id: str
    event_type: RepoIntelligenceEventType
    timestamp: datetime
    repository: str
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "repository": self.repository,
            "payload": self.payload,
            "metadata": self.metadata,
            "source": "jarvis_agent",
        }


# ============================================================================
# Cross-Repo Intelligence Bridge
# ============================================================================

class CrossRepoIntelligenceBridge:
    """
    Bridge between Repository Intelligence and cross-repo communication.

    This class:
    1. Listens for repository intelligence events
    2. Writes them to shared state files for Ironcliw Prime and Reactor Core
    3. Optionally sends via WebSocket if available
    """

    def __init__(self, config: Optional[CrossRepoBridgeConfig] = None):
        self.config = config or CrossRepoBridgeConfig()
        self._events_queue: List[RepoIntelligenceEvent] = []
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the bridge."""
        async with self._lock:
            if self._initialized:
                return

            # Ensure state directory exists
            self.config.state_dir.mkdir(parents=True, exist_ok=True)

            # Load existing events
            events_file = self.config.state_dir / self.config.events_file
            if events_file.exists():
                try:
                    data = json.loads(events_file.read_text())
                    self._events_queue = [
                        RepoIntelligenceEvent(
                            event_id=e["event_id"],
                            event_type=RepoIntelligenceEventType(e["event_type"]),
                            timestamp=datetime.fromisoformat(e["timestamp"]),
                            repository=e["repository"],
                            payload=e.get("payload", {}),
                            metadata=e.get("metadata", {}),
                        )
                        for e in data.get("events", [])
                    ]
                except Exception as e:
                    logger.warning(f"Failed to load existing events: {e}")

            self._initialized = True
            logger.info("Cross-Repo Intelligence Bridge initialized")

    async def emit_event(
        self,
        event_type: RepoIntelligenceEventType,
        repository: str,
        payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Emit a repository intelligence event.

        Args:
            event_type: Type of event
            repository: Repository name (jarvis, jarvis_prime, reactor_core)
            payload: Event-specific data
            metadata: Additional metadata

        Returns:
            Event ID
        """
        await self.initialize()

        event = RepoIntelligenceEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            repository=repository,
            payload=payload or {},
            metadata=metadata or {},
        )

        # Add to queue
        self._events_queue.append(event)

        # Trim queue if needed
        if len(self._events_queue) > self.config.max_events_stored:
            self._events_queue = self._events_queue[-self.config.max_events_stored:]

        # Persist to file
        await self._persist_events()

        # Notify consumers
        await self._notify_consumers(event)

        logger.debug(f"Emitted repo intelligence event: {event_type.value}")
        return event.event_id

    async def _persist_events(self) -> None:
        """Persist events to the shared state file."""
        events_file = self.config.state_dir / self.config.events_file
        try:
            data = {
                "events": [e.to_dict() for e in self._events_queue],
                "last_updated": datetime.utcnow().isoformat(),
                "count": len(self._events_queue),
            }
            events_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to persist events: {e}")

    async def _notify_consumers(self, event: RepoIntelligenceEvent) -> None:
        """Notify Ironcliw Prime and Reactor Core of the event."""
        # Write to Ironcliw Prime state if enabled
        if self.config.emit_to_prime:
            await self._write_prime_state(event)

        # Write to Reactor Core state if enabled
        if self.config.emit_to_reactor:
            await self._write_reactor_state(event)

    async def _write_prime_state(self, event: RepoIntelligenceEvent) -> None:
        """Write event to Ironcliw Prime shared state."""
        try:
            prime_state_file = self.config.state_dir / "repo_intel_for_prime.json"

            # Read existing state
            existing = {}
            if prime_state_file.exists():
                existing = json.loads(prime_state_file.read_text())

            # Add/update repo context
            if "repo_context" not in existing:
                existing["repo_context"] = {}

            existing["repo_context"][event.repository] = {
                "last_event": event.event_type.value,
                "last_update": event.timestamp.isoformat(),
                "payload_summary": str(event.payload)[:200],
            }

            # Track recent events
            if "recent_events" not in existing:
                existing["recent_events"] = []
            existing["recent_events"].append(event.to_dict())
            existing["recent_events"] = existing["recent_events"][-10:]

            prime_state_file.write_text(json.dumps(existing, indent=2))

        except Exception as e:
            logger.debug(f"Failed to write Prime state: {e}")

    async def _write_reactor_state(self, event: RepoIntelligenceEvent) -> None:
        """Write event to Reactor Core shared state."""
        try:
            reactor_state_file = self.config.state_dir / "repo_intel_for_reactor.json"

            # Read existing state
            existing = {}
            if reactor_state_file.exists():
                existing = json.loads(reactor_state_file.read_text())

            # Reactor Core uses events for training decisions
            if "codebase_events" not in existing:
                existing["codebase_events"] = []

            existing["codebase_events"].append({
                "event": event.to_dict(),
                "training_relevant": event.event_type in [
                    RepoIntelligenceEventType.REPO_ANALYSIS_COMPLETE,
                    RepoIntelligenceEventType.REPO_DEPENDENCY_DETECTED,
                ],
            })
            existing["codebase_events"] = existing["codebase_events"][-50:]

            existing["last_update"] = datetime.utcnow().isoformat()

            reactor_state_file.write_text(json.dumps(existing, indent=2))

        except Exception as e:
            logger.debug(f"Failed to write Reactor state: {e}")

    async def get_recent_events(
        self,
        count: int = 10,
        event_type: Optional[RepoIntelligenceEventType] = None,
    ) -> List[RepoIntelligenceEvent]:
        """Get recent events, optionally filtered by type."""
        await self.initialize()

        events = self._events_queue
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-count:]

    async def get_repo_context_for_prime(self) -> Dict[str, Any]:
        """
        Get repository context formatted for Ironcliw Prime.

        This provides the context that Prime can use for better routing
        and inference decisions.
        """
        await self.initialize()

        try:
            # Lazy import to avoid circular dependencies
            from backend.intelligence.repository_intelligence import get_repository_mapper

            mapper = await get_repository_mapper()

            # Get cross-repo analysis
            analysis = await mapper.analyze_cross_repo_dependencies()

            return {
                "repositories": analysis.repositories,
                "shared_symbols": analysis.shared_symbols[:20],
                "dependency_graph": analysis.dependency_graph,
                "integration_points": analysis.integration_points[:10],
                "recommendations": analysis.recommendations,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Failed to get repo context for Prime: {e}")
            return {}


# ============================================================================
# Singleton and Convenience Functions
# ============================================================================

_bridge_instance: Optional[CrossRepoIntelligenceBridge] = None


async def get_cross_repo_bridge() -> CrossRepoIntelligenceBridge:
    """Get the singleton bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = CrossRepoIntelligenceBridge()
        await _bridge_instance.initialize()
    return _bridge_instance


async def emit_repo_event(
    event_type: RepoIntelligenceEventType,
    repository: str,
    payload: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience function to emit a repository intelligence event."""
    bridge = await get_cross_repo_bridge()
    return await bridge.emit_event(event_type, repository, payload)


# ============================================================================
# Integration with Repository Intelligence
# ============================================================================

async def hook_repository_intelligence() -> None:
    """
    Hook into the Repository Intelligence system to emit events automatically.

    Call this during Ironcliw startup to enable automatic event emission.
    """
    try:
        from backend.intelligence.repository_intelligence import (
            get_repository_mapper,
            RepositoryMapper,
        )

        mapper = await get_repository_mapper()
        bridge = await get_cross_repo_bridge()

        # Store original method
        original_get_repo_map = mapper.get_repo_map

        # Wrap to emit events
        async def wrapped_get_repo_map(*args, **kwargs):
            result = await original_get_repo_map(*args, **kwargs)

            # Emit event
            event_type = (
                RepoIntelligenceEventType.REPO_MAP_CACHED
                if result.cache_hit
                else RepoIntelligenceEventType.REPO_MAP_GENERATED
            )

            await bridge.emit_event(
                event_type=event_type,
                repository=result.repository,
                payload={
                    "token_count": result.token_count,
                    "files_included": result.files_included,
                    "symbols_extracted": result.symbols_extracted,
                    "generation_time_ms": result.generation_time_ms,
                    "cache_hit": result.cache_hit,
                },
            )

            return result

        # Replace method
        mapper.get_repo_map = wrapped_get_repo_map
        logger.info("Repository Intelligence hooked to cross-repo bridge")

    except Exception as e:
        logger.warning(f"Failed to hook Repository Intelligence: {e}")


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "CrossRepoBridgeConfig",
    "RepoIntelligenceEventType",
    "RepoIntelligenceEvent",
    "CrossRepoIntelligenceBridge",
    "get_cross_repo_bridge",
    "emit_repo_event",
    "hook_repository_intelligence",
]
