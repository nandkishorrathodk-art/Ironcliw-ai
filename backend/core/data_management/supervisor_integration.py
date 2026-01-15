"""
Data Management Supervisor Integration v1.0
============================================

Provides seamless integration of the Unified Data Management System with the
JARVIS Trinity Supervisor. Enables single-command initialization and shutdown
of all data management capabilities.

Features:
- Single entry point for all data management initialization
- Automatic component startup ordering with dependency resolution
- Health monitoring and status reporting
- Graceful shutdown with data flushing
- Integration with resilience systems
- Cross-repo data synchronization

Author: Trinity Data System
Version: 1.0.0
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class DataSystemState(Enum):
    """State of the data management system."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    DEGRADED = auto()
    SHUTTING_DOWN = auto()
    STOPPED = auto()


class DataComponentState(Enum):
    """State of individual data components."""
    PENDING = auto()
    STARTING = auto()
    RUNNING = auto()
    UNHEALTHY = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DataManagementSupervisorConfig:
    """Configuration for data management supervisor integration."""

    # Component toggles
    enable_training_collector: bool = os.getenv("DATA_ENABLE_TRAINING_COLLECTOR", "true").lower() == "true"
    enable_versioning: bool = os.getenv("DATA_ENABLE_VERSIONING", "true").lower() == "true"
    enable_validation: bool = os.getenv("DATA_ENABLE_VALIDATION", "true").lower() == "true"
    enable_privacy: bool = os.getenv("DATA_ENABLE_PRIVACY", "true").lower() == "true"
    enable_retention: bool = os.getenv("DATA_ENABLE_RETENTION", "true").lower() == "true"
    enable_deduplication: bool = os.getenv("DATA_ENABLE_DEDUPLICATION", "true").lower() == "true"
    enable_sampling: bool = os.getenv("DATA_ENABLE_SAMPLING", "true").lower() == "true"
    enable_lineage: bool = os.getenv("DATA_ENABLE_LINEAGE", "true").lower() == "true"
    enable_cross_repo: bool = os.getenv("DATA_ENABLE_CROSS_REPO", "true").lower() == "true"

    # Initialization settings
    startup_timeout_seconds: float = float(os.getenv("DATA_STARTUP_TIMEOUT", "30.0"))
    shutdown_timeout_seconds: float = float(os.getenv("DATA_SHUTDOWN_TIMEOUT", "30.0"))

    # Health monitoring
    health_check_interval: float = float(os.getenv("DATA_HEALTH_CHECK_INTERVAL", "10.0"))
    unhealthy_threshold: int = int(os.getenv("DATA_UNHEALTHY_THRESHOLD", "3"))

    # Data paths
    data_base_path: str = os.getenv("JARVIS_DATA_PATH", "/tmp/jarvis_data")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class DataComponentHealth:
    """Health status of a data component."""
    component_name: str
    state: DataComponentState = DataComponentState.PENDING
    last_check: Optional[datetime] = None
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataManagementInitResult:
    """Result of data management initialization."""
    success: bool
    state: DataSystemState
    components: Dict[str, DataComponentHealth]
    errors: List[str]
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataManagementHealthReport:
    """Health report for the data management system."""
    system_state: DataSystemState
    components: Dict[str, DataComponentHealth]
    overall_health_score: float  # 0.0 to 1.0
    issues: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# SUPERVISOR COORDINATOR
# =============================================================================


class DataManagementSupervisorCoordinator:
    """Coordinates data management system with the supervisor."""

    def __init__(self, config: Optional[DataManagementSupervisorConfig] = None):
        self.config = config or DataManagementSupervisorConfig()

        # State
        self._state = DataSystemState.UNINITIALIZED
        self._components: Dict[str, DataComponentHealth] = {}
        self._initialized = False

        # Component instances
        self._data_engine = None
        self._cross_repo_bridge = None

        # Background tasks
        self._health_task: Optional[asyncio.Task] = None

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info("DataManagementSupervisorCoordinator created")

    @property
    def state(self) -> DataSystemState:
        """Get current system state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._state == DataSystemState.RUNNING

    async def initialize(self) -> DataManagementInitResult:
        """Initialize all data management components."""
        async with self._lock:
            if self._initialized:
                return DataManagementInitResult(
                    success=True,
                    state=self._state,
                    components=self._components,
                    errors=[],
                    duration_seconds=0.0
                )

            start_time = datetime.utcnow()
            errors: List[str] = []
            self._state = DataSystemState.INITIALIZING

            logger.info("Initializing Data Management System...")

            # Initialize components in order
            component_order = [
                ("validation", self._init_validation),
                ("privacy", self._init_privacy),
                ("versioning", self._init_versioning),
                ("deduplication", self._init_deduplication),
                ("retention", self._init_retention),
                ("sampling", self._init_sampling),
                ("lineage", self._init_lineage),
                ("training_collector", self._init_training_collector),
                ("data_engine", self._init_data_engine),
                ("cross_repo", self._init_cross_repo),
            ]

            for component_name, init_fn in component_order:
                # Check if component is enabled
                if not self._is_component_enabled(component_name):
                    logger.info(f"Component {component_name} is disabled, skipping")
                    continue

                # Initialize component
                health = DataComponentHealth(
                    component_name=component_name,
                    state=DataComponentState.STARTING
                )
                self._components[component_name] = health

                try:
                    await asyncio.wait_for(
                        init_fn(),
                        timeout=self.config.startup_timeout_seconds
                    )
                    health.state = DataComponentState.RUNNING
                    health.last_check = datetime.utcnow()
                    logger.info(f"Component {component_name} initialized successfully")

                except asyncio.TimeoutError:
                    error = f"Component {component_name} initialization timed out"
                    errors.append(error)
                    health.state = DataComponentState.FAILED
                    health.error_message = error
                    logger.error(error)

                except Exception as e:
                    error = f"Component {component_name} initialization failed: {e}"
                    errors.append(error)
                    health.state = DataComponentState.FAILED
                    health.error_message = str(e)
                    logger.error(error)

            # Start health monitoring
            self._health_task = asyncio.create_task(self._health_monitor())

            # Determine final state
            failed_count = sum(
                1 for h in self._components.values()
                if h.state == DataComponentState.FAILED
            )

            if failed_count == 0:
                self._state = DataSystemState.RUNNING
                self._initialized = True
            elif failed_count < len(self._components):
                self._state = DataSystemState.DEGRADED
                self._initialized = True
            else:
                self._state = DataSystemState.STOPPED

            duration = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                f"Data Management initialization complete: "
                f"state={self._state.name}, duration={duration:.2f}s, errors={len(errors)}"
            )

            return DataManagementInitResult(
                success=len(errors) == 0,
                state=self._state,
                components=self._components.copy(),
                errors=errors,
                duration_seconds=duration
            )

    async def shutdown(self):
        """Shutdown all data management components."""
        async with self._lock:
            if not self._initialized:
                return

            self._state = DataSystemState.SHUTTING_DOWN
            logger.info("Shutting down Data Management System...")

            # Cancel health task
            if self._health_task:
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass

            # Shutdown components in reverse order
            for component_name, health in reversed(list(self._components.items())):
                if health.state not in (DataComponentState.RUNNING, DataComponentState.UNHEALTHY):
                    continue

                health.state = DataComponentState.STOPPING

                try:
                    await self._shutdown_component(component_name)
                    health.state = DataComponentState.STOPPED
                    logger.info(f"Component {component_name} stopped")

                except Exception as e:
                    logger.error(f"Error stopping {component_name}: {e}")
                    health.error_message = str(e)

            # Shutdown main components
            if self._cross_repo_bridge:
                try:
                    from backend.core.data_management.cross_repo_data import shutdown_cross_repo_data
                    await shutdown_cross_repo_data()
                except Exception as e:
                    logger.error(f"Error shutting down cross-repo bridge: {e}")

            if self._data_engine:
                try:
                    from backend.core.data_management.unified_engine import shutdown_data_management
                    await shutdown_data_management()
                except Exception as e:
                    logger.error(f"Error shutting down data engine: {e}")

            self._state = DataSystemState.STOPPED
            self._initialized = False
            logger.info("Data Management System shutdown complete")

    async def get_health(self) -> DataManagementHealthReport:
        """Get health report for the data management system."""
        issues: List[str] = []
        metrics: Dict[str, Any] = {}

        # Calculate health score
        running_count = 0
        total_count = 0

        for name, health in self._components.items():
            total_count += 1
            if health.state == DataComponentState.RUNNING:
                running_count += 1
            elif health.state == DataComponentState.UNHEALTHY:
                issues.append(f"{name} is unhealthy: {health.error_message}")
            elif health.state == DataComponentState.FAILED:
                issues.append(f"{name} has failed: {health.error_message}")

            if health.metrics:
                metrics[name] = health.metrics

        health_score = running_count / total_count if total_count > 0 else 0.0

        # Get engine metrics if available
        if self._data_engine:
            try:
                engine_status = await self._data_engine.get_status()
                metrics["engine"] = engine_status
            except Exception:
                pass

        # Get cross-repo metrics if available
        if self._cross_repo_bridge:
            try:
                bridge_metrics = self._cross_repo_bridge.get_metrics()
                metrics["cross_repo"] = {
                    "packets_sent": bridge_metrics.packets_sent,
                    "packets_received": bridge_metrics.packets_received,
                    "packets_failed": bridge_metrics.packets_failed,
                    "success_rate": bridge_metrics.success_rate,
                }
            except Exception:
                pass

        return DataManagementHealthReport(
            system_state=self._state,
            components=self._components.copy(),
            overall_health_score=health_score,
            issues=issues,
            metrics=metrics
        )

    async def get_status(self) -> Dict[str, Any]:
        """Get status summary."""
        health = await self.get_health()
        return {
            "state": self._state.name,
            "initialized": self._initialized,
            "health_score": health.overall_health_score,
            "components": {
                name: h.state.name
                for name, h in self._components.items()
            },
            "issues": health.issues,
        }

    # -------------------------------------------------------------------------
    # Component Initialization
    # -------------------------------------------------------------------------

    def _is_component_enabled(self, component_name: str) -> bool:
        """Check if a component is enabled."""
        mapping = {
            "training_collector": self.config.enable_training_collector,
            "versioning": self.config.enable_versioning,
            "validation": self.config.enable_validation,
            "privacy": self.config.enable_privacy,
            "retention": self.config.enable_retention,
            "deduplication": self.config.enable_deduplication,
            "sampling": self.config.enable_sampling,
            "lineage": self.config.enable_lineage,
            "cross_repo": self.config.enable_cross_repo,
            "data_engine": True,  # Always enabled
        }
        return mapping.get(component_name, True)

    async def _init_validation(self):
        """Initialize validation component."""
        # Validation is part of the unified engine
        pass

    async def _init_privacy(self):
        """Initialize privacy component."""
        # Privacy is part of the unified engine
        pass

    async def _init_versioning(self):
        """Initialize versioning component."""
        # Versioning is part of the unified engine
        pass

    async def _init_deduplication(self):
        """Initialize deduplication component."""
        # Deduplication is part of the unified engine
        pass

    async def _init_retention(self):
        """Initialize retention component."""
        # Retention is part of the unified engine
        pass

    async def _init_sampling(self):
        """Initialize sampling component."""
        # Sampling is part of the unified engine
        pass

    async def _init_lineage(self):
        """Initialize lineage component."""
        # Lineage is part of the unified engine
        pass

    async def _init_training_collector(self):
        """Initialize training collector component."""
        # Training collector is part of the unified engine
        pass

    async def _init_data_engine(self):
        """Initialize the main data engine."""
        from backend.core.data_management.unified_engine import (
            initialize_data_management,
            get_data_management_engine,
        )

        # Initialize the engine (no config param - uses defaults)
        await initialize_data_management()
        # Get reference to the engine
        self._data_engine = await get_data_management_engine()

    async def _init_cross_repo(self):
        """Initialize cross-repo data bridge."""
        from backend.core.data_management.cross_repo_data import (
            initialize_cross_repo_data,
            CrossRepoDataConfig,
        )

        config = CrossRepoDataConfig()
        self._cross_repo_bridge = await initialize_cross_repo_data(config)

    async def _shutdown_component(self, component_name: str):
        """Shutdown a specific component."""
        # Components are shut down via the main engine and bridge
        pass

    # -------------------------------------------------------------------------
    # Health Monitoring
    # -------------------------------------------------------------------------

    async def _health_monitor(self):
        """Background task to monitor component health."""
        while self._state in (DataSystemState.RUNNING, DataSystemState.DEGRADED):
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._check_component_health()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")

    async def _check_component_health(self):
        """Check health of all components."""
        for name, health in self._components.items():
            if health.state not in (DataComponentState.RUNNING, DataComponentState.UNHEALTHY):
                continue

            try:
                # Check component health
                is_healthy = await self._probe_component(name)

                if is_healthy:
                    health.state = DataComponentState.RUNNING
                    health.consecutive_failures = 0
                    health.error_message = None
                else:
                    health.consecutive_failures += 1
                    if health.consecutive_failures >= self.config.unhealthy_threshold:
                        health.state = DataComponentState.UNHEALTHY
                        health.error_message = "Consecutive health checks failed"

                health.last_check = datetime.utcnow()

            except Exception as e:
                health.consecutive_failures += 1
                health.error_message = str(e)
                if health.consecutive_failures >= self.config.unhealthy_threshold:
                    health.state = DataComponentState.UNHEALTHY

        # Update system state based on component health
        unhealthy_count = sum(
            1 for h in self._components.values()
            if h.state in (DataComponentState.UNHEALTHY, DataComponentState.FAILED)
        )

        if unhealthy_count == 0:
            self._state = DataSystemState.RUNNING
        elif unhealthy_count < len(self._components):
            self._state = DataSystemState.DEGRADED
        else:
            self._state = DataSystemState.DEGRADED

    async def _probe_component(self, component_name: str) -> bool:
        """Probe a component for health."""
        try:
            if component_name == "data_engine" and self._data_engine:
                status = await self._data_engine.get_status()
                return status.get("running", False)

            elif component_name == "cross_repo" and self._cross_repo_bridge:
                metrics = self._cross_repo_bridge.get_metrics()
                return metrics.success_rate > 0.5

            # Other components are considered healthy if data engine is running
            return self._data_engine is not None

        except Exception:
            return False


# =============================================================================
# GLOBAL INSTANCE MANAGEMENT
# =============================================================================

_data_supervisor: Optional[DataManagementSupervisorCoordinator] = None
_supervisor_lock = asyncio.Lock()


async def get_data_management_supervisor() -> DataManagementSupervisorCoordinator:
    """Get or create the global data management supervisor."""
    global _data_supervisor

    async with _supervisor_lock:
        if _data_supervisor is None:
            _data_supervisor = DataManagementSupervisorCoordinator()
        return _data_supervisor


async def initialize_data_management_supervisor(
    config: Optional[DataManagementSupervisorConfig] = None
) -> DataManagementInitResult:
    """Initialize the data management supervisor and all components."""
    global _data_supervisor

    async with _supervisor_lock:
        if _data_supervisor is None:
            _data_supervisor = DataManagementSupervisorCoordinator(config)

        return await _data_supervisor.initialize()


async def shutdown_data_management_supervisor():
    """Shutdown the data management supervisor and all components."""
    global _data_supervisor

    async with _supervisor_lock:
        if _data_supervisor is not None:
            await _data_supervisor.shutdown()
            _data_supervisor = None


async def get_data_management_status() -> Dict[str, Any]:
    """Get data management status."""
    supervisor = await get_data_management_supervisor()
    return await supervisor.get_status()


async def get_data_management_health() -> DataManagementHealthReport:
    """Get data management health report."""
    supervisor = await get_data_management_supervisor()
    return await supervisor.get_health()


# =============================================================================
# CONTEXT MANAGER
# =============================================================================


class DataManagementContext:
    """Context manager for data management initialization."""

    def __init__(self, config: Optional[DataManagementSupervisorConfig] = None):
        self.config = config
        self._result: Optional[DataManagementInitResult] = None

    async def __aenter__(self) -> DataManagementSupervisorCoordinator:
        """Enter context and initialize data management."""
        self._result = await initialize_data_management_supervisor(self.config)
        return await get_data_management_supervisor()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and shutdown data management."""
        await shutdown_data_management_supervisor()
        return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "DataSystemState",
    "DataComponentState",
    # Configuration
    "DataManagementSupervisorConfig",
    # Data Structures
    "DataComponentHealth",
    "DataManagementInitResult",
    "DataManagementHealthReport",
    # Coordinator
    "DataManagementSupervisorCoordinator",
    # Global Functions
    "get_data_management_supervisor",
    "initialize_data_management_supervisor",
    "shutdown_data_management_supervisor",
    "get_data_management_status",
    "get_data_management_health",
    # Context Manager
    "DataManagementContext",
]
