"""
Supervisor Resilience Integration v1.0
======================================

Integrates the Unified Resilience Engine with the Ironcliw Supervisor.
This module is the single entry point for initializing all resilience
components when running `python3 run_supervisor.py`.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SUPERVISOR RESILIENCE INTEGRATION                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   run_supervisor.py                                                      │
    │         │                                                                │
    │         ▼                                                                │
    │   ┌─────────────────────────────────────────────────────────────┐       │
    │   │         SupervisorResilienceCoordinator                       │       │
    │   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │       │
    │   │  │ Unified     │  │ Neural Mesh │  │ Cross-Repo  │          │       │
    │   │  │ Resilience  │──│ Resilience  │──│ Circuit     │          │       │
    │   │  │ Engine      │  │ Bridge      │  │ Breakers    │          │       │
    │   │  └─────────────┘  └─────────────┘  └─────────────┘          │       │
    │   └─────────────────────────────────────────────────────────────┘       │
    │                             │                                           │
    │                             ▼                                           │
    │   ┌─────────────────────────────────────────────────────────────┐       │
    │   │                    TRINITY ECOSYSTEM                          │       │
    │   │  ┌───────────┐    ┌───────────┐    ┌───────────┐            │       │
    │   │  │  Ironcliw   │◀══▶│  Ironcliw   │◀══▶│  REACTOR  │            │       │
    │   │  │  (Body)   │    │  PRIME    │    │  CORE     │            │       │
    │   │  └───────────┘    └───────────┘    └───────────┘            │       │
    │   └─────────────────────────────────────────────────────────────┘       │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Usage:
    # In run_supervisor.py:
    from backend.core.resilience.supervisor_integration import (
        initialize_supervisor_resilience,
        shutdown_supervisor_resilience,
        get_supervisor_resilience_status,
    )

    # During startup:
    success = await initialize_supervisor_resilience()

    # During shutdown:
    await shutdown_supervisor_resilience()

    # For health checks:
    status = await get_supervisor_resilience_status()

Author: Trinity Resilience System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.core.resilience.unified_resilience_engine import (
    UnifiedResilienceEngine,
    get_resilience_engine,
    initialize_resilience,
    shutdown_resilience,
    ResilienceConfig,
    ServiceHealth,
)

from backend.core.resilience.neural_mesh_resilience import (
    NeuralMeshResilienceBridge,
    get_mesh_resilience_bridge,
    initialize_mesh_resilience,
    shutdown_mesh_resilience,
    MeshHealthSnapshot,
)

from backend.core.resilience.cross_repo_circuit_breaker import (
    get_all_breaker_status,
)

logger = logging.getLogger("Supervisor.ResilienceIntegration")


# =============================================================================
# CONFIGURATION
# =============================================================================

class SupervisorResilienceConfig:
    """Configuration for supervisor resilience integration."""

    @staticmethod
    def is_resilience_enabled() -> bool:
        return os.getenv("Ironcliw_RESILIENCE_ENABLED", "true").lower() == "true"

    @staticmethod
    def is_chaos_allowed() -> bool:
        """Chaos engineering should only be enabled in non-production environments."""
        env = os.getenv("Ironcliw_ENVIRONMENT", "development")
        explicitly_enabled = os.getenv("RESILIENCE_CHAOS_ENABLED", "false").lower() == "true"
        return env != "production" or explicitly_enabled

    @staticmethod
    def get_health_check_interval() -> float:
        return float(os.getenv("RESILIENCE_HEALTH_INTERVAL", "30.0"))

    @staticmethod
    def get_startup_timeout() -> float:
        return float(os.getenv("RESILIENCE_STARTUP_TIMEOUT", "60.0"))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ResilienceInitializationResult:
    """Result of resilience initialization."""
    success: bool
    engine_initialized: bool = False
    mesh_bridge_initialized: bool = False
    initialization_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "engine_initialized": self.engine_initialized,
            "mesh_bridge_initialized": self.mesh_bridge_initialized,
            "initialization_time_ms": round(self.initialization_time_ms, 2),
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class ResilienceHealthReport:
    """Comprehensive health report of all resilience components."""
    timestamp: float = field(default_factory=time.time)
    overall_health: ServiceHealth = ServiceHealth.UNKNOWN
    engine_status: Dict[str, Any] = field(default_factory=dict)
    mesh_health: Optional[MeshHealthSnapshot] = None
    circuit_breaker_status: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "overall_health": self.overall_health.value,
            "engine_status": self.engine_status,
            "mesh_health": self.mesh_health.to_dict() if self.mesh_health else None,
            "circuit_breaker_status": self.circuit_breaker_status,
            "warnings": self.warnings,
        }


# =============================================================================
# SUPERVISOR RESILIENCE COORDINATOR
# =============================================================================

class SupervisorResilienceCoordinator:
    """
    Coordinates all resilience components for the Ironcliw Supervisor.

    This is the main entry point for initializing and managing resilience
    across the entire Ironcliw ecosystem.
    """

    _instance: Optional["SupervisorResilienceCoordinator"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._engine: Optional[UnifiedResilienceEngine] = None
        self._mesh_bridge: Optional[NeuralMeshResilienceBridge] = None
        self._initialized = False
        self._running = False
        self._health_task: Optional[asyncio.Task] = None

        # Metrics
        self._initialization_time_ms: float = 0.0
        self._last_health_check: float = 0.0
        self._health_check_count: int = 0

        self.logger = logging.getLogger("Supervisor.ResilienceCoordinator")

    @classmethod
    async def get_instance(cls) -> "SupervisorResilienceCoordinator":
        """Get or create the singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    async def initialize(self, neural_mesh=None) -> ResilienceInitializationResult:
        """
        Initialize all resilience components.

        Args:
            neural_mesh: Optional NeuralMesh instance for cross-repo communication

        Returns:
            ResilienceInitializationResult with success status and details
        """
        if self._initialized:
            return ResilienceInitializationResult(
                success=True,
                engine_initialized=True,
                mesh_bridge_initialized=True,
                initialization_time_ms=self._initialization_time_ms,
            )

        result = ResilienceInitializationResult(success=False)
        start_time = time.time()

        self.logger.info("=" * 70)
        self.logger.info("   INITIALIZING SUPERVISOR RESILIENCE COORDINATOR")
        self.logger.info("=" * 70)

        # Check if resilience is enabled
        if not SupervisorResilienceConfig.is_resilience_enabled():
            result.warnings.append("Resilience is disabled via Ironcliw_RESILIENCE_ENABLED=false")
            result.success = True
            self.logger.warning("Resilience is DISABLED - skipping initialization")
            return result

        try:
            # 1. Initialize Unified Resilience Engine
            self.logger.info("[1/3] Initializing Unified Resilience Engine...")
            try:
                self._engine = await get_resilience_engine()
                engine_success = await self._engine.initialize()

                if engine_success:
                    result.engine_initialized = True
                    self.logger.info("      Unified Resilience Engine: OK")
                else:
                    result.errors.append("Failed to initialize Unified Resilience Engine")
                    self.logger.error("      Unified Resilience Engine: FAILED")

            except Exception as e:
                result.errors.append(f"Engine initialization error: {e}")
                self.logger.error(f"      Unified Resilience Engine: ERROR - {e}")

            # 2. Initialize Neural Mesh Resilience Bridge
            self.logger.info("[2/3] Initializing Neural Mesh Resilience Bridge...")
            try:
                self._mesh_bridge = await get_mesh_resilience_bridge()
                bridge_success = await self._mesh_bridge.initialize(neural_mesh)

                if bridge_success:
                    result.mesh_bridge_initialized = True
                    self.logger.info("      Neural Mesh Resilience Bridge: OK")
                else:
                    result.warnings.append("Neural Mesh Resilience Bridge initialization incomplete")
                    self.logger.warning("      Neural Mesh Resilience Bridge: INCOMPLETE")

            except Exception as e:
                result.warnings.append(f"Mesh bridge initialization error: {e}")
                self.logger.warning(f"      Neural Mesh Resilience Bridge: WARNING - {e}")

            # 3. Start health monitoring
            self.logger.info("[3/3] Starting resilience health monitoring...")
            self._running = True
            self._health_task = asyncio.create_task(self._health_monitor_loop())
            self.logger.info("      Health Monitoring: STARTED")

            # Check chaos engineering status
            if ResilienceConfig.is_chaos_enabled():
                if SupervisorResilienceConfig.is_chaos_allowed():
                    result.warnings.append("Chaos engineering is ENABLED")
                    self.logger.warning("      Chaos Engineering: ENABLED (non-production)")
                else:
                    result.warnings.append("Chaos engineering enabled in production - disabling")
                    self.logger.warning("      Chaos Engineering: DISABLED (production safety)")
                    if self._engine:
                        self._engine._chaos_controller.disable()

            # Calculate initialization time
            result.initialization_time_ms = (time.time() - start_time) * 1000
            self._initialization_time_ms = result.initialization_time_ms

            # Determine overall success
            result.success = result.engine_initialized
            self._initialized = result.success

            if result.success:
                self.logger.info("-" * 70)
                self.logger.info(f"   RESILIENCE INITIALIZED in {result.initialization_time_ms:.0f}ms")
                self.logger.info("-" * 70)
            else:
                self.logger.error("-" * 70)
                self.logger.error(f"   RESILIENCE INITIALIZATION FAILED")
                self.logger.error(f"   Errors: {result.errors}")
                self.logger.error("-" * 70)

            return result

        except Exception as e:
            result.errors.append(f"Critical initialization error: {e}")
            result.initialization_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Critical error during resilience initialization: {e}")
            return result

    async def shutdown(self) -> None:
        """Shutdown all resilience components."""
        self.logger.info("Shutting down Supervisor Resilience Coordinator...")
        self._running = False

        # Stop health monitoring
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Shutdown mesh bridge
        if self._mesh_bridge:
            await self._mesh_bridge.shutdown()

        # Shutdown resilience engine
        if self._engine:
            await self._engine.shutdown()

        self._initialized = False
        self.logger.info("Supervisor Resilience Coordinator shutdown complete")

    async def get_health_report(self) -> ResilienceHealthReport:
        """Generate a comprehensive health report."""
        report = ResilienceHealthReport()

        try:
            # Get engine status
            if self._engine:
                report.engine_status = self._engine.get_status()
            else:
                report.warnings.append("Resilience engine not initialized")

            # Get mesh health
            if self._mesh_bridge:
                report.mesh_health = await self._mesh_bridge.get_health_snapshot()
            else:
                report.warnings.append("Mesh resilience bridge not initialized")

            # Get circuit breaker status
            report.circuit_breaker_status = get_all_breaker_status()

            # Determine overall health
            report.overall_health = self._calculate_overall_health(report)

        except Exception as e:
            report.warnings.append(f"Error generating health report: {e}")
            self.logger.error(f"Error generating health report: {e}")

        return report

    def _calculate_overall_health(self, report: ResilienceHealthReport) -> ServiceHealth:
        """Calculate overall health from component health."""
        health_indicators = []

        # Check engine
        if report.engine_status:
            if report.engine_status.get("engine", {}).get("running", False):
                health_indicators.append(ServiceHealth.HEALTHY)
            else:
                health_indicators.append(ServiceHealth.UNHEALTHY)
        else:
            health_indicators.append(ServiceHealth.UNKNOWN)

        # Check mesh health
        if report.mesh_health:
            health_indicators.append(report.mesh_health.overall_health)
        else:
            health_indicators.append(ServiceHealth.UNKNOWN)

        # Calculate overall
        if all(h == ServiceHealth.HEALTHY for h in health_indicators):
            return ServiceHealth.HEALTHY
        if all(h == ServiceHealth.UNHEALTHY for h in health_indicators):
            return ServiceHealth.UNHEALTHY
        if any(h == ServiceHealth.UNHEALTHY for h in health_indicators):
            return ServiceHealth.DEGRADED
        return ServiceHealth.UNKNOWN

    async def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        interval = SupervisorResilienceConfig.get_health_check_interval()

        while self._running:
            try:
                await asyncio.sleep(interval)

                self._health_check_count += 1
                self._last_health_check = time.time()

                # Get health report
                report = await self.get_health_report()

                # Log if degraded or unhealthy
                if report.overall_health == ServiceHealth.UNHEALTHY:
                    self.logger.error(
                        f"Resilience health check #{self._health_check_count}: UNHEALTHY"
                    )
                elif report.overall_health == ServiceHealth.DEGRADED:
                    self.logger.warning(
                        f"Resilience health check #{self._health_check_count}: DEGRADED"
                    )
                else:
                    self.logger.debug(
                        f"Resilience health check #{self._health_check_count}: {report.overall_health.value}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor loop error: {e}")
                await asyncio.sleep(5.0)

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "initialization_time_ms": self._initialization_time_ms,
            "health_check_count": self._health_check_count,
            "last_health_check": self._last_health_check,
            "engine_initialized": self._engine is not None and self._engine._initialized,
            "mesh_bridge_initialized": self._mesh_bridge is not None and self._mesh_bridge._initialized,
        }


# =============================================================================
# GLOBAL FUNCTIONS
# =============================================================================

_coordinator: Optional[SupervisorResilienceCoordinator] = None


async def get_supervisor_resilience_coordinator() -> SupervisorResilienceCoordinator:
    """Get the global supervisor resilience coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = await SupervisorResilienceCoordinator.get_instance()
    return _coordinator


async def initialize_supervisor_resilience(neural_mesh=None) -> ResilienceInitializationResult:
    """
    Initialize all supervisor resilience components.

    This is the main entry point called from run_supervisor.py.

    Args:
        neural_mesh: Optional NeuralMesh instance

    Returns:
        ResilienceInitializationResult
    """
    coordinator = await get_supervisor_resilience_coordinator()
    return await coordinator.initialize(neural_mesh)


async def shutdown_supervisor_resilience() -> None:
    """Shutdown all supervisor resilience components."""
    global _coordinator
    if _coordinator:
        await _coordinator.shutdown()
        _coordinator = None


async def get_supervisor_resilience_status() -> Dict[str, Any]:
    """Get comprehensive status of supervisor resilience."""
    coordinator = await get_supervisor_resilience_coordinator()
    health_report = await coordinator.get_health_report()

    return {
        "coordinator": coordinator.get_status(),
        "health_report": health_report.to_dict(),
    }


async def get_supervisor_resilience_health() -> ResilienceHealthReport:
    """Get health report for supervisor resilience."""
    coordinator = await get_supervisor_resilience_coordinator()
    return await coordinator.get_health_report()


# =============================================================================
# CONVENIENCE CONTEXT MANAGER
# =============================================================================

class SupervisorResilienceContext:
    """
    Context manager for supervisor resilience lifecycle.

    Usage:
        async with SupervisorResilienceContext() as resilience:
            # Resilience is initialized
            status = await resilience.get_status()
            ...
        # Resilience is automatically shutdown
    """

    def __init__(self, neural_mesh=None):
        self._neural_mesh = neural_mesh
        self._coordinator: Optional[SupervisorResilienceCoordinator] = None

    async def __aenter__(self) -> SupervisorResilienceCoordinator:
        self._coordinator = await get_supervisor_resilience_coordinator()
        await self._coordinator.initialize(self._neural_mesh)
        return self._coordinator

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._coordinator:
            await self._coordinator.shutdown()
        return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "SupervisorResilienceConfig",
    # Data Structures
    "ResilienceInitializationResult",
    "ResilienceHealthReport",
    # Main Coordinator
    "SupervisorResilienceCoordinator",
    # Global Functions
    "get_supervisor_resilience_coordinator",
    "initialize_supervisor_resilience",
    "shutdown_supervisor_resilience",
    "get_supervisor_resilience_status",
    "get_supervisor_resilience_health",
    # Context Manager
    "SupervisorResilienceContext",
]
