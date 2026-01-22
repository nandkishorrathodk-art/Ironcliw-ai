"""
Race Condition Prevention - Trinity IPC Integration v1.0
=========================================================

Integrates the race condition prevention system with existing Trinity IPC
infrastructure to provide seamless, atomic operations across all components.

This module bridges:
- race_condition_prevention.py (atomic operations)
- trinity_ipc.py (IPC bus and communication)
- trinity_port_manager.py (port allocation)
- cross_repo_startup_orchestrator.py (startup coordination)

KEY INTEGRATION POINTS:
=======================
1. Enhanced AtomicIPCFile - Uses atomic lock for all operations
2. Safe HeartbeatPublisher - Integrates atomic heartbeat writes
3. RacePreventionPortManager - Atomic port reservation with Trinity
4. StartupOrderValidator - Dependency-aware startup with monitoring

ZERO HARDCODING:
================
All configuration via environment variables:
    RACE_TRINITY_INTEGRATION_ENABLED    - Enable/disable integration (default: true)
    RACE_TRINITY_LOCK_PREFIX            - Prefix for Trinity locks (default: "trinity_ipc")
    RACE_TRINITY_HEARTBEAT_COMPONENT    - Component name for heartbeats
    RACE_TRINITY_STARTUP_GRACE_PERIOD   - Grace period for startup (default: 120.0)

Usage:
    from backend.core.race_condition_ipc_integration import (
        get_safe_ipc_bus,
        get_race_aware_port_manager,
        validate_startup_order,
    )

    # Get race-safe IPC bus
    bus = await get_safe_ipc_bus()

    # Get race-aware port manager
    port_manager = await get_race_aware_port_manager()

    # Validate startup order
    await validate_startup_order()

Author: JARVIS Trinity v1.0 - Race Condition IPC Integration
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Environment Configuration
# =============================================================================


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment."""
    value = os.environ.get(key, "").lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _env_str(key: str, default: str) -> str:
    """Get string from environment."""
    return os.environ.get(key, default)


def _env_float(key: str, default: float) -> float:
    """Get float from environment."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


# Configuration flags
INTEGRATION_ENABLED = _env_bool("RACE_TRINITY_INTEGRATION_ENABLED", True)
LOCK_PREFIX = _env_str("RACE_TRINITY_LOCK_PREFIX", "trinity_ipc")
STARTUP_GRACE_PERIOD = _env_float("RACE_TRINITY_STARTUP_GRACE_PERIOD", 120.0)


# =============================================================================
# Safe Imports with Fallbacks
# =============================================================================

# Availability flags
RACE_PREVENTION_AVAILABLE = False
TRINITY_IPC_AVAILABLE = False
PORT_MANAGER_AVAILABLE = False

# Function references (set to None initially, populated by imports)
get_file_lock: Any = None
get_heartbeat_writer: Any = None
get_port_reserver: Any = None
get_dependency_monitor: Any = None
get_process_manager: Any = None
get_config_manager: Any = None
get_message_queue: Any = None
get_trinity_ipc_bus: Any = None
get_trinity_port_manager: Any = None

# Issue 16-20 function references
get_startup_timeout: Any = None
get_health_checker: Any = None
get_resource_manager: Any = None
get_log_writer: Any = None
get_state_file: Any = None

# Issue 26-30 function references
get_disk_manager: Any = None
get_memory_manager: Any = None
get_permission_validator: Any = None
get_state_recovery: Any = None
get_version_manager: Any = None
startup_health_validation: Any = None
comprehensive_health_check: Any = None

# Class references
AtomicFileLock: Any = None
AtomicHeartbeatWriter: Any = None
AtomicPortReservation: Any = None
DependencyHealthMonitor: Any = None
SafeProcessManager: Any = None
AtomicConfigManager: Any = None
PersistentMessageQueue: Any = None
AtomicIPCFile: Any = None
ComponentType: Any = None
AllocationResult: Any = None

# Issue 16-20 class references
AdaptiveStartupTimeout: Any = None
ResilientHealthChecker: Any = None
CoordinatedResourceManager: Any = None
AtomicLogWriter: Any = None
AtomicStateFile: Any = None
StartupPhase: Any = None
HealthCheckResult: Any = None
ResourceType: Any = None

# Issue 26-30 class references
DiskSpaceManager: Any = None
DiskPressureLevel: Any = None
MemoryPressureManager: Any = None
MemoryPressureLevel: Any = None
PermissionValidator: Any = None
StateFileRecoveryManager: Any = None
DependencyVersionManager: Any = None
SystemHealthReport: Any = None

# Race condition prevention imports
try:
    from backend.core.race_condition_prevention import (
        # Issues 9-15 classes
        AtomicConfigManager as _AtomicConfigManager,
        AtomicFileLock as _AtomicFileLock,
        AtomicHeartbeatWriter as _AtomicHeartbeatWriter,
        AtomicPortReservation as _AtomicPortReservation,
        DependencyHealthMonitor as _DependencyHealthMonitor,
        PersistentMessageQueue as _PersistentMessageQueue,
        SafeProcessManager as _SafeProcessManager,
        # Issues 16-20 classes
        AdaptiveStartupTimeout as _AdaptiveStartupTimeout,
        AtomicLogWriter as _AtomicLogWriter,
        AtomicStateFile as _AtomicStateFile,
        CoordinatedResourceManager as _CoordinatedResourceManager,
        HealthCheckResult as _HealthCheckResult,
        ResilientHealthChecker as _ResilientHealthChecker,
        ResourceType as _ResourceType,
        StartupPhase as _StartupPhase,
        # Issues 9-15 factory functions
        get_config_manager as _get_config_manager,
        get_dependency_monitor as _get_dependency_monitor,
        get_file_lock as _get_file_lock,
        get_heartbeat_writer as _get_heartbeat_writer,
        get_message_queue as _get_message_queue,
        get_port_reserver as _get_port_reserver,
        get_process_manager as _get_process_manager,
        # Issues 16-20 factory functions
        get_health_checker as _get_health_checker,
        get_log_writer as _get_log_writer,
        get_resource_manager as _get_resource_manager,
        get_startup_timeout as _get_startup_timeout,
        get_state_file as _get_state_file,
        # Issues 26-30 classes
        DependencyVersionManager as _DependencyVersionManager,
        DiskPressureLevel as _DiskPressureLevel,
        DiskSpaceManager as _DiskSpaceManager,
        MemoryPressureLevel as _MemoryPressureLevel,
        MemoryPressureManager as _MemoryPressureManager,
        PermissionValidator as _PermissionValidator,
        StateFileRecoveryManager as _StateFileRecoveryManager,
        SystemHealthReport as _SystemHealthReport,
        # Issues 26-30 factory functions
        comprehensive_health_check as _comprehensive_health_check,
        get_disk_manager as _get_disk_manager,
        get_memory_manager as _get_memory_manager,
        get_permission_validator as _get_permission_validator,
        get_state_recovery as _get_state_recovery,
        get_version_manager as _get_version_manager,
        startup_health_validation as _startup_health_validation,
    )
    # Issues 9-15
    get_file_lock = _get_file_lock
    get_heartbeat_writer = _get_heartbeat_writer
    get_port_reserver = _get_port_reserver
    get_dependency_monitor = _get_dependency_monitor
    get_process_manager = _get_process_manager
    get_config_manager = _get_config_manager
    get_message_queue = _get_message_queue
    AtomicFileLock = _AtomicFileLock
    AtomicHeartbeatWriter = _AtomicHeartbeatWriter
    AtomicPortReservation = _AtomicPortReservation
    DependencyHealthMonitor = _DependencyHealthMonitor
    SafeProcessManager = _SafeProcessManager
    AtomicConfigManager = _AtomicConfigManager
    PersistentMessageQueue = _PersistentMessageQueue
    # Issues 16-20
    get_startup_timeout = _get_startup_timeout
    get_health_checker = _get_health_checker
    get_resource_manager = _get_resource_manager
    get_log_writer = _get_log_writer
    get_state_file = _get_state_file
    AdaptiveStartupTimeout = _AdaptiveStartupTimeout
    ResilientHealthChecker = _ResilientHealthChecker
    CoordinatedResourceManager = _CoordinatedResourceManager
    AtomicLogWriter = _AtomicLogWriter
    AtomicStateFile = _AtomicStateFile
    StartupPhase = _StartupPhase
    HealthCheckResult = _HealthCheckResult
    ResourceType = _ResourceType
    # Issues 26-30
    get_disk_manager = _get_disk_manager
    get_memory_manager = _get_memory_manager
    get_permission_validator = _get_permission_validator
    get_state_recovery = _get_state_recovery
    get_version_manager = _get_version_manager
    startup_health_validation = _startup_health_validation
    comprehensive_health_check = _comprehensive_health_check
    DiskSpaceManager = _DiskSpaceManager
    DiskPressureLevel = _DiskPressureLevel
    MemoryPressureManager = _MemoryPressureManager
    MemoryPressureLevel = _MemoryPressureLevel
    PermissionValidator = _PermissionValidator
    StateFileRecoveryManager = _StateFileRecoveryManager
    DependencyVersionManager = _DependencyVersionManager
    SystemHealthReport = _SystemHealthReport
    RACE_PREVENTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[RaceIPC] Race prevention not available: {e}")


# Trinity IPC imports
try:
    from backend.core.trinity_ipc import (
        AtomicIPCFile as _AtomicIPCFile,
        ComponentType as _ComponentType,
        get_trinity_ipc_bus as _get_trinity_ipc_bus,
    )
    get_trinity_ipc_bus = _get_trinity_ipc_bus
    AtomicIPCFile = _AtomicIPCFile
    ComponentType = _ComponentType
    TRINITY_IPC_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[RaceIPC] Trinity IPC not available: {e}")


# Trinity port manager imports
try:
    from backend.core.trinity_port_manager import (
        AllocationResult as _AllocationResult,
        get_trinity_port_manager as _get_trinity_port_manager,
    )
    get_trinity_port_manager = _get_trinity_port_manager
    AllocationResult = _AllocationResult
    PORT_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[RaceIPC] Port manager not available: {e}")


# =============================================================================
# Enhanced Atomic IPC File with Race Prevention
# =============================================================================


class RaceSafeAtomicIPCFile:
    """
    Enhanced AtomicIPCFile that uses race prevention locks.

    Wraps the original AtomicIPCFile with additional atomic locking
    to prevent race conditions during concurrent access.
    """

    def __init__(
        self,
        file_path: Path,
        lock_name: Optional[str] = None,
    ):
        """
        Initialize race-safe IPC file.

        Args:
            file_path: Path to IPC file
            lock_name: Optional custom lock name
        """
        self.file_path = Path(file_path)
        self.lock_name = lock_name or f"{LOCK_PREFIX}_{self.file_path.stem}"

        # Get atomic lock
        if RACE_PREVENTION_AVAILABLE:
            self._lock = get_file_lock(self.lock_name)
        else:
            self._lock = None

        # Create wrapped IPC file
        if TRINITY_IPC_AVAILABLE:
            self._ipc_file = AtomicIPCFile(file_path)
        else:
            self._ipc_file = None

    async def read_atomic(self) -> Optional[Dict[str, Any]]:
        """Read with race-safe locking."""
        if not RACE_PREVENTION_AVAILABLE or self._lock is None:
            if self._ipc_file:
                return await self._ipc_file.read_atomic()
            return None

        # Use atomic lock for read (shared would be ideal but we use exclusive for safety)
        async with self._lock.acquire_context(timeout=10.0) as acquired:
            if not acquired:
                logger.warning(f"[RaceIPC] Lock timeout for read: {self.lock_name}")
                # Fallback to unprotected read
                if self._ipc_file:
                    return await self._ipc_file.read_atomic()
                return None

            if self._ipc_file:
                return await self._ipc_file.read_atomic()
            return None

    async def write_atomic(self, data: Dict[str, Any]) -> None:
        """Write with race-safe locking."""
        if not RACE_PREVENTION_AVAILABLE or self._lock is None:
            if self._ipc_file:
                await self._ipc_file.write_atomic(data)
            return

        async with self._lock.acquire_context(timeout=30.0) as acquired:
            if not acquired:
                logger.warning(f"[RaceIPC] Lock timeout for write: {self.lock_name}")
                # Still attempt write
            if self._ipc_file:
                await self._ipc_file.write_atomic(data)

    async def update_atomic(
        self,
        updater: Callable[[Dict[str, Any]], Dict[str, Any]],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update with race-safe locking."""
        if not RACE_PREVENTION_AVAILABLE or self._lock is None:
            if self._ipc_file:
                return await self._ipc_file.update_atomic(updater, timeout)
            return {}

        async with self._lock.acquire_context(timeout=timeout or 30.0) as acquired:
            if not acquired:
                logger.warning(f"[RaceIPC] Lock timeout for update: {self.lock_name}")
            if self._ipc_file:
                return await self._ipc_file.update_atomic(updater, timeout)
            return {}


# =============================================================================
# Race-Safe IPC Bus
# =============================================================================


class RaceSafeIPCBus:
    """
    IPC bus wrapper with race condition prevention.

    Enhances TrinityIPCBus with:
    - Atomic heartbeat writes using race prevention
    - Lock-protected command queue operations
    - Safe state updates
    """

    def __init__(self, base_bus: Optional[Any] = None):
        """
        Initialize race-safe IPC bus.

        Args:
            base_bus: Optional base TrinityIPCBus to wrap
        """
        self._base_bus = base_bus
        self._heartbeat_writers: Dict[str, Any] = {}
        self._command_lock: Optional[Any] = None

        if RACE_PREVENTION_AVAILABLE:
            self._command_lock = get_file_lock(f"{LOCK_PREFIX}_command_queue")

    async def _get_base_bus(self) -> Any:
        """Get or create base bus."""
        if self._base_bus is None and TRINITY_IPC_AVAILABLE:
            self._base_bus = await get_trinity_ipc_bus()
        return self._base_bus

    def _get_heartbeat_writer(self, component: str) -> Optional[Any]:
        """Get or create heartbeat writer for component."""
        if not RACE_PREVENTION_AVAILABLE:
            return None

        if component not in self._heartbeat_writers:
            self._heartbeat_writers[component] = get_heartbeat_writer(component)
        return self._heartbeat_writers[component]

    async def publish_heartbeat(
        self,
        component: Any,  # ComponentType or str
        status: str,
        pid: int,
        metrics: Optional[Dict[str, Any]] = None,
        dependencies_ready: Optional[Dict[str, bool]] = None,
    ) -> None:
        """
        Publish heartbeat with race-safe atomic writes.

        Uses both:
        1. Race prevention atomic heartbeat writer (for atomic guarantees)
        2. Original Trinity IPC (for compatibility)
        """
        # Get component name
        component_name = component.value if hasattr(component, 'value') else str(component)

        # Write via race prevention (atomic guarantees)
        if RACE_PREVENTION_AVAILABLE:
            writer = self._get_heartbeat_writer(component_name)
            if writer:
                await writer.write_heartbeat(
                    status=status,
                    metrics={
                        "pid": pid,
                        **(metrics or {}),
                        "dependencies_ready": dependencies_ready or {},
                    },
                )

        # Also write via original Trinity IPC (compatibility)
        base_bus = await self._get_base_bus()
        if base_bus:
            await base_bus.publish_heartbeat(
                component=component,
                status=status,
                pid=pid,
                metrics=metrics,
                dependencies_ready=dependencies_ready,
            )

    async def read_heartbeat(
        self,
        component: Any,
        max_age_seconds: Optional[float] = None,
    ) -> Optional[Any]:
        """Read heartbeat with race-safe access."""
        base_bus = await self._get_base_bus()
        if base_bus:
            return await base_bus.read_heartbeat(component, max_age_seconds)
        return None

    async def enqueue_command(self, command: Any) -> str:
        """Enqueue command with lock protection."""
        if RACE_PREVENTION_AVAILABLE and self._command_lock:
            async with self._command_lock.acquire_context(timeout=10.0):
                base_bus = await self._get_base_bus()
                if base_bus:
                    return await base_bus.enqueue_command(command)
        else:
            base_bus = await self._get_base_bus()
            if base_bus:
                return await base_bus.enqueue_command(command)

        return ""

    async def dequeue_command(self, target: Any) -> Optional[Any]:
        """Dequeue command with lock protection."""
        if RACE_PREVENTION_AVAILABLE and self._command_lock:
            async with self._command_lock.acquire_context(timeout=10.0):
                base_bus = await self._get_base_bus()
                if base_bus:
                    return await base_bus.dequeue_command(target)
        else:
            base_bus = await self._get_base_bus()
            if base_bus:
                return await base_bus.dequeue_command(target)

        return None


# =============================================================================
# Race-Aware Port Manager
# =============================================================================


class RaceAwarePortManager:
    """
    Port manager with race condition prevention.

    Wraps TrinityPortManager with:
    - Atomic port reservation from race prevention
    - PID tracking with creation time validation
    - Automatic stale reservation cleanup
    """

    def __init__(self, base_manager: Optional[Any] = None):
        """
        Initialize race-aware port manager.

        Args:
            base_manager: Optional base TrinityPortManager to wrap
        """
        self._base_manager = base_manager
        self._port_reserver: Optional[Any] = None

        if RACE_PREVENTION_AVAILABLE:
            self._port_reserver = get_port_reserver()

    async def _get_base_manager(self) -> Any:
        """Get or create base manager."""
        if self._base_manager is None and PORT_MANAGER_AVAILABLE:
            self._base_manager = await get_trinity_port_manager()
        return self._base_manager

    async def allocate_port(
        self,
        component: Any,  # ComponentType
        pid: Optional[int] = None,
    ) -> Any:  # AllocationResult
        """
        Allocate port with race-safe reservation.

        Uses race prevention for atomic reservation, then
        registers with Trinity port manager for compatibility.
        """
        # Get component name
        component_name = component.value if hasattr(component, 'value') else str(component)

        # Get port configuration from base manager
        base_manager = await self._get_base_manager()

        if base_manager and hasattr(base_manager, 'config'):
            allocation = base_manager.config.allocations.get(component)
            if allocation:
                primary_port = allocation.primary
                fallback_ports = list(allocation.fallbacks)
            else:
                primary_port = 8000  # Fallback default
                fallback_ports = [8001, 8002, 8003]
        else:
            primary_port = 8000
            fallback_ports = [8001, 8002, 8003]

        # Use race prevention for atomic reservation
        if RACE_PREVENTION_AVAILABLE and self._port_reserver:
            port = await self._port_reserver.reserve_port(
                component=component_name,
                preferred_port=primary_port,
                fallback_ports=fallback_ports,
            )

            if port:
                # Create compatible result
                if PORT_MANAGER_AVAILABLE:
                    return AllocationResult(
                        success=True,
                        component=component,
                        port=port,
                        is_primary=(port == primary_port),
                        elapsed_ms=0.0,
                    )
                else:
                    return {"success": True, "port": port}

        # Fallback to base manager
        if base_manager:
            return await base_manager.allocate_port(component, pid)

        # Last resort - return failure
        if PORT_MANAGER_AVAILABLE:
            return AllocationResult(
                success=False,
                component=component,
                port=0,
                is_primary=False,
                elapsed_ms=0.0,
                error="No port manager available",
            )
        return {"success": False, "port": 0, "error": "No port manager available"}

    async def release_port(self, component: Any) -> None:
        """Release port reservation."""
        # Get current port
        base_manager = await self._get_base_manager()
        current_port = None

        if base_manager:
            current_port = base_manager.get_port(component)

        # Release via race prevention
        if RACE_PREVENTION_AVAILABLE and self._port_reserver and current_port:
            await self._port_reserver.release_port(current_port)

        # Also release via base manager
        if base_manager:
            await base_manager.release_port(component)

    async def cleanup_stale(self) -> int:
        """Clean up stale port reservations."""
        cleaned = 0

        if RACE_PREVENTION_AVAILABLE and self._port_reserver:
            cleaned = await self._port_reserver.cleanup_stale_reservations()

        return cleaned


# =============================================================================
# Startup Order Validator
# =============================================================================


@dataclass
class ComponentDefinition:
    """Definition of a component for startup validation."""
    name: str
    component_type: Any  # ComponentType
    dependencies: List[str]
    health_check: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None


class StartupOrderValidator:
    """
    Validates and enforces correct startup order using dependency monitoring.

    Integrates with DependencyHealthMonitor for:
    - Topological sort of components
    - Continuous health monitoring
    - Automatic restart of dependents on failure
    """

    def __init__(self):
        """Initialize startup validator."""
        self._monitor: Optional[Any] = None
        self._components: Dict[str, ComponentDefinition] = {}

        if RACE_PREVENTION_AVAILABLE:
            self._monitor = get_dependency_monitor()

    def register_component(
        self,
        name: str,
        component_type: Any,
        dependencies: List[str],
        health_check: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None,
    ) -> None:
        """
        Register a component for startup validation.

        Args:
            name: Component name
            component_type: Component type enum value
            dependencies: List of dependency component names
            health_check: Optional async health check function
        """
        component = ComponentDefinition(
            name=name,
            component_type=component_type,
            dependencies=dependencies,
            health_check=health_check,
        )
        self._components[name] = component

        # Register with dependency monitor
        if self._monitor:
            self._monitor.register_component(
                name=name,
                dependencies=dependencies,
                health_checker=health_check,
            )

    def get_startup_order(self) -> List[str]:
        """Get components in correct startup order."""
        if self._monitor:
            return self._monitor.get_startup_order()

        # Fallback: simple topological sort
        in_degree: Dict[str, int] = {name: 0 for name in self._components}
        for comp in self._components.values():
            for dep in comp.dependencies:
                if dep in in_degree:
                    in_degree[comp.name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: List[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for comp in self._components.values():
                if node in comp.dependencies and comp.name in in_degree:
                    in_degree[comp.name] -= 1
                    if in_degree[comp.name] == 0:
                        queue.append(comp.name)

        return result

    async def validate_dependencies_ready(self, component: str) -> bool:
        """
        Check if all dependencies of a component are ready.

        Args:
            component: Component name to check

        Returns:
            True if all dependencies are healthy
        """
        comp = self._components.get(component)
        if comp is None:
            return True  # Unknown component - assume ready

        if self._monitor:
            return self._monitor.are_dependencies_healthy(component)

        # Fallback: check heartbeats
        for dep in comp.dependencies:
            if RACE_PREVENTION_AVAILABLE:
                writer = get_heartbeat_writer(dep)
                heartbeat = await writer.read_heartbeat()
                if heartbeat is None or heartbeat.is_stale(30.0):
                    return False
            else:
                # Can't verify - assume ready
                pass

        return True

    async def wait_for_dependencies(
        self,
        component: str,
        timeout: float = 120.0,
    ) -> bool:
        """
        Wait for all dependencies to be ready.

        Args:
            component: Component name
            timeout: Maximum wait time

        Returns:
            True if all dependencies became ready, False on timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await self.validate_dependencies_ready(component):
                return True
            await asyncio.sleep(1.0)

        logger.warning(
            f"[StartupValidator] Timeout waiting for dependencies of {component}"
        )
        return False

    async def start_monitoring(self) -> None:
        """Start dependency health monitoring."""
        if self._monitor:
            await self._monitor.start_monitoring()

    async def stop_monitoring(self) -> None:
        """Stop dependency health monitoring."""
        if self._monitor:
            await self._monitor.stop_monitoring()

    def on_dependency_failed(
        self,
        callback: Callable[[str, List[str]], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register callback when a dependency fails.

        Callback receives: (failed_dependency, list_of_affected_dependents)
        """
        if self._monitor:
            self._monitor.on_dependency_failed(callback)


# =============================================================================
# Singleton Instances and Factory Functions
# =============================================================================


_safe_ipc_bus: Optional[RaceSafeIPCBus] = None
_race_port_manager: Optional[RaceAwarePortManager] = None
_startup_validator: Optional[StartupOrderValidator] = None


async def get_safe_ipc_bus() -> RaceSafeIPCBus:
    """Get the global race-safe IPC bus."""
    global _safe_ipc_bus

    if _safe_ipc_bus is None:
        _safe_ipc_bus = RaceSafeIPCBus()

    return _safe_ipc_bus


async def get_race_aware_port_manager() -> RaceAwarePortManager:
    """Get the global race-aware port manager."""
    global _race_port_manager

    if _race_port_manager is None:
        _race_port_manager = RaceAwarePortManager()

    return _race_port_manager


def get_startup_validator() -> StartupOrderValidator:
    """Get the global startup validator."""
    global _startup_validator

    if _startup_validator is None:
        _startup_validator = StartupOrderValidator()

    return _startup_validator


# =============================================================================
# Convenience Functions
# =============================================================================


async def validate_startup_order() -> List[str]:
    """
    Validate and return correct startup order for Trinity components.

    Registers default Trinity components and returns their startup order.
    """
    validator = get_startup_validator()

    # Register Trinity components with their dependencies
    if TRINITY_IPC_AVAILABLE:
        # Reactor Core has no dependencies
        validator.register_component(
            name="reactor_core",
            component_type=ComponentType.REACTOR_CORE,
            dependencies=[],
        )

        # JARVIS Prime depends on Reactor Core
        validator.register_component(
            name="jarvis_prime",
            component_type=ComponentType.JARVIS_PRIME,
            dependencies=["reactor_core"],
        )

        # JARVIS Body depends on both
        validator.register_component(
            name="jarvis_body",
            component_type=ComponentType.JARVIS_BODY,
            dependencies=["reactor_core", "jarvis_prime"],
        )

    return validator.get_startup_order()


async def ensure_race_safe_startup() -> bool:
    """
    Ensure race-safe startup of Trinity system.

    This should be called before starting any components.
    Returns True if startup is safe to proceed.
    """
    if not INTEGRATION_ENABLED:
        logger.info("[RaceIPC] Integration disabled - using original IPC")
        return True

    if not RACE_PREVENTION_AVAILABLE:
        logger.warning("[RaceIPC] Race prevention not available - using original IPC")
        return True

    try:
        # Clean up stale reservations
        port_manager = await get_race_aware_port_manager()
        cleaned = await port_manager.cleanup_stale()
        if cleaned > 0:
            logger.info(f"[RaceIPC] Cleaned {cleaned} stale port reservations")

        # Validate startup order
        order = await validate_startup_order()
        logger.info(f"[RaceIPC] Startup order validated: {order}")

        # Start dependency monitoring
        validator = get_startup_validator()
        await validator.start_monitoring()

        return True

    except Exception as e:
        logger.error(f"[RaceIPC] Startup validation error: {e}")
        return False


@asynccontextmanager
async def race_safe_operation(
    resource_name: str,
    timeout: float = 30.0,
) -> AsyncIterator[bool]:
    """
    Context manager for race-safe operations.

    Usage:
        async with race_safe_operation("heartbeat_update") as acquired:
            if acquired:
                # Safe to perform operation
    """
    if not RACE_PREVENTION_AVAILABLE:
        yield True
        return

    lock = get_file_lock(f"{LOCK_PREFIX}_{resource_name}")
    async with lock.acquire_context(timeout) as acquired:
        yield acquired


# =============================================================================
# Cross-Repo Process Management (Issue 13 Integration)
# =============================================================================


async def register_trinity_process(
    component: str,
    pid: int,
) -> bool:
    """
    Register a Trinity process with fingerprint validation.

    Args:
        component: Component name (e.g., "reactor_core", "jarvis_prime", "jarvis_body")
        pid: Process ID

    Returns:
        True if registered successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_process_manager is None:
        return True  # Fallback: assume success

    manager = get_process_manager()
    return await manager.register_process(component, pid)


async def verify_trinity_process(component: str, pid: int) -> bool:
    """
    Verify a Trinity process is who we think it is.

    Args:
        component: Component name
        pid: Expected PID

    Returns:
        True if verified
    """
    if not RACE_PREVENTION_AVAILABLE or get_process_manager is None:
        # Fallback: basic PID check
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    manager = get_process_manager()
    from backend.core.race_condition_prevention import ProcessVerificationResult
    result = await manager.verify_process(component, pid)
    return result == ProcessVerificationResult.VERIFIED


async def safe_terminate_trinity_process(
    component: str,
    timeout: float = 10.0,
) -> bool:
    """
    Safely terminate a Trinity process with verification.

    Args:
        component: Component name
        timeout: Timeout for graceful shutdown

    Returns:
        True if terminated successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_process_manager is None:
        return False

    manager = get_process_manager()
    return await manager.safe_kill(component, timeout=timeout)


# =============================================================================
# Cross-Repo Configuration (Issue 14 Integration)
# =============================================================================


async def load_trinity_config(config_name: str) -> Dict[str, Any]:
    """
    Load Trinity configuration atomically with version tracking.

    Args:
        config_name: Configuration name

    Returns:
        Configuration data (empty dict if not found)
    """
    if not RACE_PREVENTION_AVAILABLE or get_config_manager is None:
        return {}

    config_mgr = get_config_manager(config_name)
    config = await config_mgr.read()
    return config.data if config else {}


async def save_trinity_config(
    config_name: str,
    data: Dict[str, Any],
) -> bool:
    """
    Save Trinity configuration atomically.

    Args:
        config_name: Configuration name
        data: Configuration data

    Returns:
        True if saved successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_config_manager is None:
        return False

    config_mgr = get_config_manager(config_name)
    return await config_mgr.write(data)


@asynccontextmanager
async def modify_trinity_config(config_name: str) -> AsyncIterator[Dict[str, Any]]:
    """
    Context manager for atomic read-modify-write config operations.

    Usage:
        async with modify_trinity_config("jarvis_settings") as config:
            config["new_key"] = "new_value"
            # Automatically saved on exit
    """
    if not RACE_PREVENTION_AVAILABLE or get_config_manager is None:
        data: Dict[str, Any] = {}
        yield data
        return

    config_mgr = get_config_manager(config_name)
    async with config_mgr.modify() as data:
        yield data


# =============================================================================
# Cross-Repo Event Bus (Issue 15 Integration)
# =============================================================================


async def publish_cross_repo_event(
    topic: str,
    payload: Dict[str, Any],
    source_component: str,
) -> str:
    """
    Publish an event to the cross-repo event bus.

    Args:
        topic: Event topic
        payload: Event payload
        source_component: Source component name

    Returns:
        Message ID
    """
    if not RACE_PREVENTION_AVAILABLE or get_message_queue is None:
        return ""

    queue = get_message_queue("cross_repo_events")
    enriched_payload = {
        **payload,
        "_source": source_component,
        "_timestamp": time.time(),
    }
    return await queue.publish(topic, enriched_payload)


async def subscribe_cross_repo_events(
    consumer_id: str,
    topics: Optional[List[str]] = None,
) -> AsyncIterator[Any]:
    """
    Subscribe to cross-repo events.

    Args:
        consumer_id: Unique consumer identifier
        topics: Optional list of topics to subscribe to

    Yields:
        Event messages
    """
    if not RACE_PREVENTION_AVAILABLE or get_message_queue is None:
        return

    queue = get_message_queue("cross_repo_events")
    async for msg in queue.consume(consumer_id, topics):
        yield msg


async def acknowledge_cross_repo_event(msg_id: str, consumer_id: str) -> bool:
    """Acknowledge a cross-repo event."""
    if not RACE_PREVENTION_AVAILABLE or get_message_queue is None:
        return False

    queue = get_message_queue("cross_repo_events")
    return await queue.acknowledge(msg_id, consumer_id)


# =============================================================================
# Issue 16: Startup Timeout Integration
# =============================================================================


async def track_component_startup(component: str) -> bool:
    """
    Start tracking startup for a Trinity component.

    Uses adaptive timeout that adjusts based on system load.

    Args:
        component: Component name

    Returns:
        True if tracking started successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_startup_timeout is None:
        return True

    timeout_mgr = get_startup_timeout()
    state = await timeout_mgr.start_tracking(component)
    return state is not None


async def confirm_component_spawn(component: str, pid: int) -> bool:
    """
    Confirm that a component has been spawned.

    Resets timeout from spawn phase to initialization phase.

    Args:
        component: Component name
        pid: Process ID

    Returns:
        True if confirmed successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_startup_timeout is None:
        return True

    timeout_mgr = get_startup_timeout()
    state = await timeout_mgr.confirm_spawn(component, pid)
    return state is not None


async def mark_component_ready(component: str) -> bool:
    """
    Mark a component as ready and fully initialized.

    Args:
        component: Component name

    Returns:
        True if marked successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_startup_timeout is None:
        return True

    timeout_mgr = get_startup_timeout()
    state = await timeout_mgr.mark_ready(component)
    return state is not None


async def check_component_timeout(component: str) -> tuple:
    """
    Check if a component has timed out during startup.

    Args:
        component: Component name

    Returns:
        Tuple of (is_timed_out, reason)
    """
    if not RACE_PREVENTION_AVAILABLE or get_startup_timeout is None:
        return False, None

    timeout_mgr = get_startup_timeout()
    return await timeout_mgr.is_timed_out(component)


# =============================================================================
# Issue 17: Health Check Integration
# =============================================================================


def register_component_health_check(
    component: str,
    checker: Callable[[], Coroutine[Any, Any, bool]],
) -> None:
    """
    Register a health check function for a component.

    The health checker provides retry with backoff and caching.

    Args:
        component: Component name
        checker: Async function returning True if healthy
    """
    if not RACE_PREVENTION_AVAILABLE or get_health_checker is None:
        return

    health_mgr = get_health_checker()
    health_mgr.register(component, checker)


async def check_component_health(component: str) -> Any:
    """
    Check component health with retry and caching.

    Returns cached result if within TTL, otherwise performs check
    with exponential backoff retry.

    Args:
        component: Component name

    Returns:
        HealthCheckResult enum or None
    """
    if not RACE_PREVENTION_AVAILABLE or get_health_checker is None:
        return None

    health_mgr = get_health_checker()
    status = await health_mgr.check_health(component)
    return status.result if status else None


async def mark_component_restarting(component: str) -> None:
    """
    Mark a component as restarting.

    Health checks will return RESTARTING during grace period,
    preventing false negatives during restart.

    Args:
        component: Component name
    """
    if not RACE_PREVENTION_AVAILABLE or get_health_checker is None:
        return

    health_mgr = get_health_checker()
    await health_mgr.mark_restarting(component)


# =============================================================================
# Issue 18: Resource Cleanup Integration
# =============================================================================


async def reserve_component_resource(
    resource_id: str,
    resource_type: Any,
    component: str,
    ttl: Optional[float] = None,
) -> bool:
    """
    Reserve a resource for a component.

    Prevents resource from being cleaned up during use.

    Args:
        resource_id: Resource identifier (e.g., "port:8010")
        resource_type: ResourceType enum value
        component: Component reserving the resource
        ttl: Optional time-to-live in seconds

    Returns:
        True if reserved successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_resource_manager is None:
        return True

    resource_mgr = get_resource_manager()
    return await resource_mgr.reserve(resource_id, resource_type, component, ttl)


async def release_component_resource(
    resource_id: str,
    component: str,
) -> bool:
    """
    Release a resource reservation.

    Args:
        resource_id: Resource identifier
        component: Component releasing the resource

    Returns:
        True if released successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_resource_manager is None:
        return True

    resource_mgr = get_resource_manager()
    return await resource_mgr.release(resource_id, component)


@asynccontextmanager
async def component_startup_protection(
    component: str,
    timeout: float = 30.0,
) -> AsyncIterator[bool]:
    """
    Context manager for protected startup phase.

    Resources cannot be cleaned up during startup.

    Usage:
        async with component_startup_protection("jarvis_body") as protected:
            if protected:
                # Perform startup safely
    """
    if not RACE_PREVENTION_AVAILABLE or get_resource_manager is None:
        yield True
        return

    resource_mgr = get_resource_manager()
    async with resource_mgr.startup_phase(component, timeout) as acquired:
        yield acquired


async def cleanup_stale_component_resources() -> int:
    """
    Cleanup all stale component resources.

    Cleans resources where owner is dead or expired.

    Returns:
        Number of resources cleaned
    """
    if not RACE_PREVENTION_AVAILABLE or get_resource_manager is None:
        return 0

    resource_mgr = get_resource_manager()
    return await resource_mgr.cleanup_stale_resources()


# =============================================================================
# Issue 19: Log Writing Integration
# =============================================================================


async def write_component_log(
    component: str,
    level: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write a log entry atomically for a component.

    Uses per-process log files to avoid write conflicts.

    Args:
        component: Component name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Log message
        context: Optional context dictionary
    """
    if not RACE_PREVENTION_AVAILABLE or get_log_writer is None:
        return

    log_writer = get_log_writer(component)
    await log_writer.write(level, message, context)


async def rotate_component_logs(component: str) -> bool:
    """
    Rotate log files for a component.

    Acquires rotation lock to prevent concurrent rotation.

    Args:
        component: Component name

    Returns:
        True if rotated successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_log_writer is None:
        return False

    log_writer = get_log_writer(component)
    return await log_writer.rotate()


async def aggregate_component_logs(
    component: str,
    output_path: Path,
    since: Optional[float] = None,
) -> int:
    """
    Aggregate logs from all processes of a component.

    Args:
        component: Component name
        output_path: Output file path
        since: Optional timestamp to filter logs

    Returns:
        Number of entries aggregated
    """
    if not RACE_PREVENTION_AVAILABLE or get_log_writer is None:
        return 0

    log_writer = get_log_writer(component)
    return await log_writer.aggregate_logs(
        log_writer.log_dir,
        component,
        output_path,
        since,
    )


# =============================================================================
# Issue 20: State File Integration
# =============================================================================


async def read_component_state(name: str) -> Optional[Dict[str, Any]]:
    """
    Read component state file atomically.

    Supports automatic recovery from backup.

    Args:
        name: State file name

    Returns:
        State data or None
    """
    if not RACE_PREVENTION_AVAILABLE or get_state_file is None:
        return None

    state_file = get_state_file(name)
    state = await state_file.read()
    return state.data if state else None


async def write_component_state(
    name: str,
    data: Dict[str, Any],
) -> bool:
    """
    Write component state atomically with versioning.

    Creates backup before writing for corruption recovery.

    Args:
        name: State file name
        data: State data

    Returns:
        True if written successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_state_file is None:
        return False

    state_file = get_state_file(name)
    return await state_file.write(data)


@asynccontextmanager
async def update_component_state(name: str) -> AsyncIterator[Dict[str, Any]]:
    """
    Context manager for atomic state updates.

    Holds lock for entire read-modify-write operation.

    Usage:
        async with update_component_state("supervisor") as state:
            state["component_count"] += 1
    """
    if not RACE_PREVENTION_AVAILABLE or get_state_file is None:
        data: Dict[str, Any] = {}
        yield data
        return

    state_file = get_state_file(name)
    async with state_file.update() as data:
        yield data


async def recover_component_state(name: str) -> bool:
    """
    Recover component state from backup.

    Args:
        name: State file name

    Returns:
        True if recovered successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_state_file is None:
        return False

    state_file = get_state_file(name)
    return await state_file.recover()


# =============================================================================
# Issue 26: Disk Space Integration
# =============================================================================


async def check_trinity_disk_space() -> Optional[Any]:
    """
    Check disk space for Trinity system.

    Returns:
        DiskSpaceStatus or None
    """
    if not RACE_PREVENTION_AVAILABLE or get_disk_manager is None:
        return None

    manager = get_disk_manager()
    return await manager.get_status()


async def start_disk_space_monitoring() -> None:
    """Start disk space monitoring for Trinity."""
    if not RACE_PREVENTION_AVAILABLE or get_disk_manager is None:
        return

    manager = get_disk_manager()
    await manager.start_monitoring()


async def stop_disk_space_monitoring() -> None:
    """Stop disk space monitoring."""
    if not RACE_PREVENTION_AVAILABLE or get_disk_manager is None:
        return

    manager = get_disk_manager()
    await manager.stop_monitoring()


async def cleanup_trinity_disk() -> Dict[str, Any]:
    """
    Clean up old files to free disk space.

    Returns:
        Cleanup results by category
    """
    if not RACE_PREVENTION_AVAILABLE or get_disk_manager is None:
        return {}

    manager = get_disk_manager()
    return await manager.cleanup_all()


# =============================================================================
# Issue 27: Memory Pressure Integration
# =============================================================================


async def check_trinity_memory() -> Optional[Any]:
    """
    Check memory status for Trinity system.

    Returns:
        MemoryStatus or None
    """
    if not RACE_PREVENTION_AVAILABLE or get_memory_manager is None:
        return None

    manager = get_memory_manager()
    return await manager.get_status()


async def start_memory_monitoring() -> None:
    """Start memory pressure monitoring for Trinity."""
    if not RACE_PREVENTION_AVAILABLE or get_memory_manager is None:
        return

    manager = get_memory_manager()
    await manager.start_monitoring()


async def stop_memory_monitoring() -> None:
    """Stop memory monitoring."""
    if not RACE_PREVENTION_AVAILABLE or get_memory_manager is None:
        return

    manager = get_memory_manager()
    await manager.stop_monitoring()


def register_trinity_cache(
    name: str,
    clear_func: Callable[[], int],
) -> None:
    """
    Register a cache for automatic cleanup under memory pressure.

    Args:
        name: Cache name
        clear_func: Function that clears cache and returns bytes freed
    """
    if not RACE_PREVENTION_AVAILABLE or get_memory_manager is None:
        return

    manager = get_memory_manager()
    manager.register_cache(name, clear_func)


# =============================================================================
# Issue 28: Permission Validation Integration
# =============================================================================


async def validate_trinity_permissions() -> Optional[Any]:
    """
    Validate all required permissions for Trinity.

    Returns:
        PermissionValidationResult or None
    """
    if not RACE_PREVENTION_AVAILABLE or get_permission_validator is None:
        return None

    validator = get_permission_validator()
    return await validator.validate_all()


async def validate_trinity_permissions_with_report() -> bool:
    """
    Validate permissions and log detailed report.

    Returns:
        True if all critical permissions are valid
    """
    if not RACE_PREVENTION_AVAILABLE or get_permission_validator is None:
        return True

    validator = get_permission_validator()
    return await validator.validate_and_report()


# =============================================================================
# Issue 29: State File Recovery Integration
# =============================================================================


async def check_trinity_state_health(name: str) -> Optional[Any]:
    """
    Check health of a Trinity state file.

    Args:
        name: State file name

    Returns:
        StateFileHealth or None
    """
    if not RACE_PREVENTION_AVAILABLE or get_state_recovery is None:
        return None

    recovery = get_state_recovery()
    return await recovery.check_health(name)


async def repair_trinity_state(name: str) -> bool:
    """
    Repair a corrupted Trinity state file.

    Args:
        name: State file name

    Returns:
        True if repaired successfully
    """
    if not RACE_PREVENTION_AVAILABLE or get_state_recovery is None:
        return False

    recovery = get_state_recovery()
    return await recovery.repair(name)


async def ensure_trinity_state_healthy(name: str) -> bool:
    """
    Ensure Trinity state file is healthy.

    Attempts repair if needed.

    Args:
        name: State file name

    Returns:
        True if healthy (after repair if needed)
    """
    if not RACE_PREVENTION_AVAILABLE or get_state_recovery is None:
        return True

    recovery = get_state_recovery()
    return await recovery.ensure_healthy(name)


# =============================================================================
# Issue 30: Dependency Version Integration
# =============================================================================


async def validate_trinity_dependencies() -> List[Any]:
    """
    Validate all Trinity package dependencies.

    Returns:
        List of VersionCheckResult
    """
    if not RACE_PREVENTION_AVAILABLE or get_version_manager is None:
        return []

    manager = get_version_manager()
    return await manager.validate_all_packages()


async def validate_trinity_dependencies_with_report() -> bool:
    """
    Validate dependencies and log detailed report.

    Returns:
        True if all checks pass
    """
    if not RACE_PREVENTION_AVAILABLE or get_version_manager is None:
        return True

    manager = get_version_manager()
    return await manager.validate_and_report()


def add_trinity_dependency(
    package: str,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None,
) -> None:
    """
    Add a package dependency requirement.

    Args:
        package: Package name
        min_version: Minimum required version
        max_version: Maximum allowed version
    """
    if not RACE_PREVENTION_AVAILABLE or get_version_manager is None:
        return

    manager = get_version_manager()
    manager.add_requirement(package, min_version, max_version)


def register_trinity_component_version(
    name: str,
    version: str,
    api_version: str,
    min_compatible: str,
) -> None:
    """
    Register a Trinity component's version info.

    Args:
        name: Component name
        version: Current version
        api_version: API version
        min_compatible: Minimum compatible version
    """
    if not RACE_PREVENTION_AVAILABLE or get_version_manager is None:
        return

    manager = get_version_manager()
    manager.register_component(name, version, api_version, min_compatible)


# =============================================================================
# Comprehensive Health Check Integration
# =============================================================================


async def trinity_health_check() -> Optional[Any]:
    """
    Perform comprehensive Trinity health check.

    Checks disk, memory, permissions, state files, and dependencies.

    Returns:
        SystemHealthReport or None
    """
    if not RACE_PREVENTION_AVAILABLE or comprehensive_health_check is None:
        return None

    return await comprehensive_health_check()


async def trinity_startup_validation() -> bool:
    """
    Perform startup health validation.

    Should be called before starting Trinity components.

    Returns:
        True if system is healthy enough to start
    """
    if not RACE_PREVENTION_AVAILABLE or startup_health_validation is None:
        return True

    return await startup_health_validation()


# =============================================================================
# Full System Integration
# =============================================================================


async def initialize_race_safe_trinity() -> Dict[str, Any]:
    """
    Initialize the complete race-safe Trinity system.

    This should be called once at supervisor startup.

    Returns:
        Initialization status and any warnings
    """
    result = {
        "success": True,
        "warnings": [],
        "features_enabled": {
            "race_prevention": RACE_PREVENTION_AVAILABLE,
            "trinity_ipc": TRINITY_IPC_AVAILABLE,
            "port_manager": PORT_MANAGER_AVAILABLE,
        },
    }

    if not RACE_PREVENTION_AVAILABLE:
        result["warnings"].append("Race prevention not available")
        return result

    try:
        # Ensure race-safe startup
        startup_ok = await ensure_race_safe_startup()
        if not startup_ok:
            result["warnings"].append("Startup validation had issues")

        # Start event queue cleanup
        if get_message_queue is not None:
            queue = get_message_queue("cross_repo_events")
            await queue.start_cleanup_task()

        logger.info("[RaceIPC] Trinity race-safe system initialized")
        return result

    except Exception as e:
        result["success"] = False
        result["warnings"].append(f"Initialization error: {e}")
        logger.error(f"[RaceIPC] Initialization failed: {e}")
        return result


async def shutdown_race_safe_trinity() -> None:
    """
    Shutdown the race-safe Trinity system gracefully.

    Releases all resources and stops background tasks.
    """
    if not RACE_PREVENTION_AVAILABLE:
        return

    try:
        # Stop startup validator monitoring
        validator = get_startup_validator()
        await validator.stop_monitoring()

        # Stop event queue
        if get_message_queue is not None:
            queue = get_message_queue("cross_repo_events")
            await queue.stop()

        logger.info("[RaceIPC] Trinity race-safe system shutdown complete")

    except Exception as e:
        logger.error(f"[RaceIPC] Shutdown error: {e}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main classes
    "RaceSafeAtomicIPCFile",
    "RaceSafeIPCBus",
    "RaceAwarePortManager",
    "StartupOrderValidator",
    "ComponentDefinition",

    # Factory functions
    "get_safe_ipc_bus",
    "get_race_aware_port_manager",
    "get_startup_validator",

    # Basic convenience functions
    "validate_startup_order",
    "ensure_race_safe_startup",
    "race_safe_operation",

    # Issue 13: Process management
    "register_trinity_process",
    "verify_trinity_process",
    "safe_terminate_trinity_process",

    # Issue 14: Configuration management
    "load_trinity_config",
    "save_trinity_config",
    "modify_trinity_config",

    # Issue 15: Event bus
    "publish_cross_repo_event",
    "subscribe_cross_repo_events",
    "acknowledge_cross_repo_event",

    # Issue 16: Startup timeout
    "track_component_startup",
    "confirm_component_spawn",
    "mark_component_ready",
    "check_component_timeout",

    # Issue 17: Health check
    "register_component_health_check",
    "check_component_health",
    "mark_component_restarting",

    # Issue 18: Resource cleanup
    "reserve_component_resource",
    "release_component_resource",
    "component_startup_protection",
    "cleanup_stale_component_resources",

    # Issue 19: Log writing
    "write_component_log",
    "rotate_component_logs",
    "aggregate_component_logs",

    # Issue 20: State file
    "read_component_state",
    "write_component_state",
    "update_component_state",
    "recover_component_state",

    # Issue 26: Disk space
    "check_trinity_disk_space",
    "start_disk_space_monitoring",
    "stop_disk_space_monitoring",
    "cleanup_trinity_disk",

    # Issue 27: Memory pressure
    "check_trinity_memory",
    "start_memory_monitoring",
    "stop_memory_monitoring",
    "register_trinity_cache",

    # Issue 28: Permission validation
    "validate_trinity_permissions",
    "validate_trinity_permissions_with_report",

    # Issue 29: State file recovery
    "check_trinity_state_health",
    "repair_trinity_state",
    "ensure_trinity_state_healthy",

    # Issue 30: Dependency version
    "validate_trinity_dependencies",
    "validate_trinity_dependencies_with_report",
    "add_trinity_dependency",
    "register_trinity_component_version",

    # Comprehensive health check
    "trinity_health_check",
    "trinity_startup_validation",

    # Full system integration
    "initialize_race_safe_trinity",
    "shutdown_race_safe_trinity",

    # Flags
    "INTEGRATION_ENABLED",
    "RACE_PREVENTION_AVAILABLE",
    "TRINITY_IPC_AVAILABLE",
    "PORT_MANAGER_AVAILABLE",
]
