"""
Cross-Repo Neural Mesh Bridge v1.0
==================================

Integrates JARVIS Prime and Reactor Core as Neural Mesh agents.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                       Neural Mesh                                │
    │  ┌──────────────────────────────────────────────────────────┐  │
    │  │              CrossRepoNeuralMeshBridge                   │  │
    │  └──────────────────────────────────────────────────────────┘  │
    │           │                    │                    │           │
    │           ▼                    ▼                    ▼           │
    │  ┌────────────┐       ┌────────────┐       ┌────────────┐      │
    │  │ JARVIS     │       │ JARVIS     │       │ Reactor    │      │
    │  │ Agents     │       │ Prime      │       │ Core       │      │
    │  │ (local)    │       │ (remote)   │       │ (remote)   │      │
    │  └────────────┘       └────────────┘       └────────────┘      │
    └─────────────────────────────────────────────────────────────────┘

JARVIS Prime provides:
- Local model inference (Llama, Mistral, etc.)
- GPU-accelerated processing
- Cost-effective local alternatives to cloud APIs

Reactor Core provides:
- Model fine-tuning
- Experience ingestion
- Continuous learning pipeline

This bridge:
1. Registers external repos as agents in Neural Mesh
2. Monitors their health via heartbeat files
3. Routes tasks to appropriate repos
4. Handles failover when repos are unavailable

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("CrossRepoNeuralMesh")

# =============================================================================
# Configuration
# =============================================================================

CROSS_REPO_DIR = Path(os.getenv(
    "CROSS_REPO_DIR",
    str(Path.home() / ".jarvis" / "cross_repo")
))

TRINITY_COMPONENTS_DIR = Path(os.getenv(
    "TRINITY_COMPONENTS_DIR",
    str(Path.home() / ".jarvis" / "trinity" / "components")
))

# Heartbeat monitoring
HEARTBEAT_TIMEOUT = float(os.getenv("CROSS_REPO_HEARTBEAT_TIMEOUT", "30.0"))
HEALTH_CHECK_INTERVAL = float(os.getenv("CROSS_REPO_HEALTH_INTERVAL", "10.0"))

# External repo endpoints
JARVIS_PRIME_ENDPOINT = os.getenv("JARVIS_PRIME_ENDPOINT", "http://localhost:8001")
REACTOR_CORE_ENDPOINT = os.getenv("REACTOR_CORE_ENDPOINT", "http://localhost:8002")


class ExternalRepoStatus(Enum):
    """Status of external repo."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    STARTING = "starting"
    UNKNOWN = "unknown"


class ExternalRepoCapability(Enum):
    """Capabilities provided by external repos."""
    # JARVIS Prime capabilities
    LOCAL_INFERENCE = "local_inference"
    LOCAL_LLM = "local_llm"
    GPU_COMPUTE = "gpu_compute"
    EMBEDDINGS = "embeddings"
    COST_FREE_INFERENCE = "cost_free_inference"

    # Reactor Core capabilities
    MODEL_TRAINING = "model_training"
    EXPERIENCE_INGESTION = "experience_ingestion"
    FINE_TUNING = "fine_tuning"
    MODEL_VALIDATION = "model_validation"
    CONTINUOUS_LEARNING = "continuous_learning"


@dataclass
class ExternalRepoAgent:
    """Represents an external repo as a Neural Mesh agent."""
    repo_name: str
    agent_id: str
    endpoint: str
    capabilities: List[str]
    status: ExternalRepoStatus = ExternalRepoStatus.UNKNOWN
    last_heartbeat: float = 0.0
    health_score: float = 0.0
    load: float = 0.0
    version: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if repo is healthy."""
        now = time.time()
        heartbeat_fresh = (now - self.last_heartbeat) < HEARTBEAT_TIMEOUT
        return heartbeat_fresh and self.status == ExternalRepoStatus.ONLINE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_name": self.repo_name,
            "agent_id": self.agent_id,
            "endpoint": self.endpoint,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "health_score": self.health_score,
            "load": self.load,
            "version": self.version,
            "metadata": self.metadata,
            "is_healthy": self.is_healthy(),
        }


@dataclass
class CrossRepoMetrics:
    """Metrics for cross-repo bridge."""
    health_checks: int = 0
    tasks_routed_to_prime: int = 0
    tasks_routed_to_reactor: int = 0
    prime_failures: int = 0
    reactor_failures: int = 0
    heartbeat_timeouts: int = 0
    capabilities_queries: int = 0
    last_health_check: float = 0.0


# =============================================================================
# Cross-Repo Neural Mesh Bridge
# =============================================================================

class CrossRepoNeuralMeshBridge:
    """
    Bridges external repos (JARVIS Prime, Reactor Core) with Neural Mesh.

    Features:
    - Registers external repos as Neural Mesh agents
    - Monitors health via heartbeat files
    - Routes tasks to appropriate repos
    - Handles failover when repos unavailable
    - Provides unified capability lookup
    """

    def __init__(self):
        self.logger = logging.getLogger("CrossRepoNeuralMesh")

        # External repo agents
        self._prime_agent = ExternalRepoAgent(
            repo_name="jarvis_prime",
            agent_id="jarvis_prime_inference",
            endpoint=JARVIS_PRIME_ENDPOINT,
            capabilities=[
                ExternalRepoCapability.LOCAL_INFERENCE.value,
                ExternalRepoCapability.LOCAL_LLM.value,
                ExternalRepoCapability.GPU_COMPUTE.value,
                ExternalRepoCapability.EMBEDDINGS.value,
                ExternalRepoCapability.COST_FREE_INFERENCE.value,
            ],
        )

        self._reactor_agent = ExternalRepoAgent(
            repo_name="reactor_core",
            agent_id="reactor_core_training",
            endpoint=REACTOR_CORE_ENDPOINT,
            capabilities=[
                ExternalRepoCapability.MODEL_TRAINING.value,
                ExternalRepoCapability.EXPERIENCE_INGESTION.value,
                ExternalRepoCapability.FINE_TUNING.value,
                ExternalRepoCapability.MODEL_VALIDATION.value,
                ExternalRepoCapability.CONTINUOUS_LEARNING.value,
            ],
        )

        # State
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._mesh_registered = False

        # Metrics
        self._metrics = CrossRepoMetrics()

        # Neural Mesh reference (lazy)
        self._neural_mesh = None
        self._unified_registry = None

        # Callbacks
        self._status_change_callbacks: Set[Callable] = set()

        # Ensure directories exist
        CROSS_REPO_DIR.mkdir(parents=True, exist_ok=True)
        TRINITY_COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)

    async def start(self) -> bool:
        """Start the cross-repo bridge."""
        if self._running:
            return True

        self._running = True
        self.logger.info("CrossRepoNeuralMeshBridge starting...")

        # Connect to Neural Mesh
        await self._connect_neural_mesh()

        # Initial health check
        await self._check_all_repos()

        # Register agents with Neural Mesh
        await self._register_agents_with_mesh()

        # Start health monitoring
        self._health_task = asyncio.create_task(
            self._health_loop(),
            name="cross_repo_health_monitor",
        )

        self.logger.info(
            f"CrossRepoNeuralMeshBridge ready "
            f"(Prime: {self._prime_agent.status.value}, "
            f"Reactor: {self._reactor_agent.status.value})"
        )
        return True

    async def stop(self) -> None:
        """Stop the cross-repo bridge."""
        self._running = False

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Deregister from Neural Mesh
        await self._deregister_from_mesh()

        self.logger.info("CrossRepoNeuralMeshBridge stopped")

    async def _connect_neural_mesh(self) -> bool:
        """Connect to Neural Mesh coordinator."""
        try:
            from backend.neural_mesh.neural_mesh_coordinator import get_neural_mesh
            self._neural_mesh = await get_neural_mesh()
            self.logger.debug("Connected to Neural Mesh")
            return True
        except ImportError:
            self.logger.warning("Neural Mesh not available")
            return False
        except Exception as e:
            self.logger.warning(f"Neural Mesh connection failed: {e}")
            return False

    async def _register_agents_with_mesh(self) -> None:
        """Register external repos as Neural Mesh agents."""
        if not self._neural_mesh:
            return

        try:
            # Register JARVIS Prime if healthy
            if self._prime_agent.is_healthy():
                await self._register_single_agent(self._prime_agent)

            # Register Reactor Core if healthy
            if self._reactor_agent.is_healthy():
                await self._register_single_agent(self._reactor_agent)

            self._mesh_registered = True
            self.logger.info("Registered external repos with Neural Mesh")

        except Exception as e:
            self.logger.warning(f"Failed to register with Neural Mesh: {e}")

    async def _register_single_agent(self, agent: ExternalRepoAgent) -> None:
        """Register a single external agent."""
        if not self._neural_mesh:
            return

        try:
            # Create registration data
            agent_data = {
                "agent_name": agent.agent_id,
                "agent_type": "external",
                "capabilities": set(agent.capabilities),
                "backend": "cross_repo",
                "version": agent.version,
                "status": "online" if agent.is_healthy() else "offline",
                "load": agent.load,
                "health": "healthy" if agent.health_score >= 0.8 else "degraded",
                "metadata": {
                    **agent.metadata,
                    "repo_name": agent.repo_name,
                    "endpoint": agent.endpoint,
                    "is_external": True,
                    "cross_repo_bridge": True,
                },
            }

            # Register with Neural Mesh
            if hasattr(self._neural_mesh, 'registry'):
                await self._neural_mesh.registry.register(**agent_data)
                self.logger.debug(f"Registered {agent.agent_id} with Neural Mesh")

        except Exception as e:
            self.logger.warning(f"Failed to register {agent.agent_id}: {e}")

    async def _deregister_from_mesh(self) -> None:
        """Deregister external repos from Neural Mesh."""
        if not self._neural_mesh or not self._mesh_registered:
            return

        try:
            if hasattr(self._neural_mesh, 'registry'):
                await self._neural_mesh.registry.deregister(self._prime_agent.agent_id)
                await self._neural_mesh.registry.deregister(self._reactor_agent.agent_id)
        except Exception as e:
            self.logger.debug(f"Deregistration error: {e}")

    async def _health_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

                if not self._running:
                    break

                await self._check_all_repos()
                await self._update_mesh_status()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(5.0)

    async def _check_all_repos(self) -> None:
        """Check health of all external repos."""
        self._metrics.health_checks += 1
        self._metrics.last_health_check = time.time()

        await self._check_repo_health(self._prime_agent, "jarvis_prime.json")
        await self._check_repo_health(self._reactor_agent, "reactor_core.json")

    async def _check_repo_health(
        self,
        agent: ExternalRepoAgent,
        heartbeat_filename: str,
    ) -> None:
        """Check health of a single repo."""
        old_status = agent.status

        # Try multiple heartbeat locations
        heartbeat_paths = [
            TRINITY_COMPONENTS_DIR / heartbeat_filename,
            CROSS_REPO_DIR / f"{agent.repo_name}_heartbeat.json",
            CROSS_REPO_DIR / "heartbeat.json",
        ]

        heartbeat_found = False
        for heartbeat_file in heartbeat_paths:
            if heartbeat_file.exists():
                try:
                    data = json.loads(heartbeat_file.read_text())

                    # Update agent from heartbeat
                    timestamp = data.get("timestamp", 0)
                    if time.time() - timestamp < HEARTBEAT_TIMEOUT:
                        agent.last_heartbeat = timestamp
                        agent.status = ExternalRepoStatus.ONLINE
                        agent.health_score = data.get("health_score", 1.0)
                        agent.load = data.get("load", data.get("cpu_percent", 0.0)) / 100
                        agent.version = data.get("version", "unknown")
                        agent.metadata.update({
                            "pid": data.get("pid"),
                            "uptime": data.get("uptime_seconds"),
                            "memory_percent": data.get("memory_percent"),
                        })
                        heartbeat_found = True
                        break
                    else:
                        self._metrics.heartbeat_timeouts += 1

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.logger.debug(f"Heartbeat read error: {e}")
                    continue

        if not heartbeat_found:
            agent.status = ExternalRepoStatus.OFFLINE
            agent.health_score = 0.0

        # Notify callbacks on status change
        if old_status != agent.status:
            self.logger.info(
                f"{agent.repo_name} status changed: {old_status.value} -> {agent.status.value}"
            )
            await self._notify_status_change(agent)

    async def _update_mesh_status(self) -> None:
        """Update status of external agents in Neural Mesh."""
        if not self._neural_mesh or not hasattr(self._neural_mesh, 'registry'):
            return

        for agent in [self._prime_agent, self._reactor_agent]:
            try:
                # Update status in mesh
                new_status = "online" if agent.is_healthy() else "offline"
                await self._neural_mesh.registry.update_status(
                    agent.agent_id,
                    new_status,
                )

                # Update health/load
                await self._neural_mesh.registry.update_health(
                    agent.agent_id,
                    "healthy" if agent.health_score >= 0.8 else "degraded",
                )

            except Exception as e:
                self.logger.debug(f"Mesh status update error: {e}")

    async def _notify_status_change(self, agent: ExternalRepoAgent) -> None:
        """Notify callbacks of status change."""
        for callback in self._status_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(agent)
                else:
                    callback(agent)
            except Exception as e:
                self.logger.debug(f"Status callback error: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def get_prime_agent(self) -> ExternalRepoAgent:
        """Get JARVIS Prime agent."""
        return self._prime_agent

    def get_reactor_agent(self) -> ExternalRepoAgent:
        """Get Reactor Core agent."""
        return self._reactor_agent

    def is_prime_available(self) -> bool:
        """Check if JARVIS Prime is available."""
        return self._prime_agent.is_healthy()

    def is_reactor_available(self) -> bool:
        """Check if Reactor Core is available."""
        return self._reactor_agent.is_healthy()

    def has_capability(self, capability: str) -> bool:
        """Check if any external repo has capability."""
        if capability in self._prime_agent.capabilities and self._prime_agent.is_healthy():
            return True
        if capability in self._reactor_agent.capabilities and self._reactor_agent.is_healthy():
            return True
        return False

    def get_agent_for_capability(
        self,
        capability: str,
    ) -> Optional[ExternalRepoAgent]:
        """Get best external agent for capability."""
        self._metrics.capabilities_queries += 1

        # Check JARVIS Prime first (usually preferred for inference)
        if capability in self._prime_agent.capabilities and self._prime_agent.is_healthy():
            return self._prime_agent

        # Check Reactor Core
        if capability in self._reactor_agent.capabilities and self._reactor_agent.is_healthy():
            return self._reactor_agent

        return None

    async def route_to_prime(
        self,
        task_type: str,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Route task to JARVIS Prime.

        Uses file-based RPC for cross-repo communication.
        """
        if not self._prime_agent.is_healthy():
            self._metrics.prime_failures += 1
            return None

        self._metrics.tasks_routed_to_prime += 1

        # Write request to cross-repo directory
        request_file = CROSS_REPO_DIR / "prime_requests" / f"{task_type}_{int(time.time()*1000)}.json"
        request_file.parent.mkdir(parents=True, exist_ok=True)

        request_data = {
            "task_type": task_type,
            "payload": payload,
            "timestamp": time.time(),
            "source": "jarvis_neural_mesh",
        }

        try:
            request_file.write_text(json.dumps(request_data))
            self.logger.debug(f"Routed task to Prime: {task_type}")
            return {"status": "routed", "request_file": str(request_file)}
        except Exception as e:
            self.logger.error(f"Failed to route to Prime: {e}")
            self._metrics.prime_failures += 1
            return None

    async def route_to_reactor(
        self,
        task_type: str,
        payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Route task to Reactor Core.

        Uses file-based RPC for cross-repo communication.
        """
        if not self._reactor_agent.is_healthy():
            self._metrics.reactor_failures += 1
            return None

        self._metrics.tasks_routed_to_reactor += 1

        # Write request to cross-repo directory
        request_file = CROSS_REPO_DIR / "reactor_requests" / f"{task_type}_{int(time.time()*1000)}.json"
        request_file.parent.mkdir(parents=True, exist_ok=True)

        request_data = {
            "task_type": task_type,
            "payload": payload,
            "timestamp": time.time(),
            "source": "jarvis_neural_mesh",
        }

        try:
            request_file.write_text(json.dumps(request_data))
            self.logger.debug(f"Routed task to Reactor: {task_type}")
            return {"status": "routed", "request_file": str(request_file)}
        except Exception as e:
            self.logger.error(f"Failed to route to Reactor: {e}")
            self._metrics.reactor_failures += 1
            return None

    def on_status_change(self, callback: Callable) -> None:
        """Register callback for status changes."""
        self._status_change_callbacks.add(callback)

    def get_metrics(self) -> Dict[str, Any]:
        """Get bridge metrics."""
        return {
            "running": self._running,
            "mesh_registered": self._mesh_registered,
            "health_checks": self._metrics.health_checks,
            "tasks_routed_to_prime": self._metrics.tasks_routed_to_prime,
            "tasks_routed_to_reactor": self._metrics.tasks_routed_to_reactor,
            "prime_failures": self._metrics.prime_failures,
            "reactor_failures": self._metrics.reactor_failures,
            "heartbeat_timeouts": self._metrics.heartbeat_timeouts,
            "capabilities_queries": self._metrics.capabilities_queries,
            "last_health_check": self._metrics.last_health_check,
            "prime": self._prime_agent.to_dict(),
            "reactor": self._reactor_agent.to_dict(),
        }


# =============================================================================
# Global Instance Management
# =============================================================================

_cross_repo_mesh: Optional[CrossRepoNeuralMeshBridge] = None
_mesh_lock: Optional[asyncio.Lock] = None


def _get_mesh_lock() -> asyncio.Lock:
    """Get or create the mesh lock."""
    global _mesh_lock
    if _mesh_lock is None:
        _mesh_lock = asyncio.Lock()
    return _mesh_lock


async def get_cross_repo_neural_mesh() -> CrossRepoNeuralMeshBridge:
    """Get the global CrossRepoNeuralMeshBridge instance."""
    global _cross_repo_mesh

    lock = _get_mesh_lock()
    async with lock:
        if _cross_repo_mesh is None:
            _cross_repo_mesh = CrossRepoNeuralMeshBridge()
            await _cross_repo_mesh.start()

        return _cross_repo_mesh


async def shutdown_cross_repo_neural_mesh() -> None:
    """Shutdown the global CrossRepoNeuralMeshBridge."""
    global _cross_repo_mesh

    if _cross_repo_mesh:
        await _cross_repo_mesh.stop()
        _cross_repo_mesh = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CrossRepoNeuralMeshBridge",
    "ExternalRepoAgent",
    "ExternalRepoStatus",
    "ExternalRepoCapability",
    "get_cross_repo_neural_mesh",
    "shutdown_cross_repo_neural_mesh",
]
