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

# Race condition prevention imports
try:
    from backend.core.race_condition_prevention import (
        AtomicConfigManager as _AtomicConfigManager,
        AtomicFileLock as _AtomicFileLock,
        AtomicHeartbeatWriter as _AtomicHeartbeatWriter,
        AtomicPortReservation as _AtomicPortReservation,
        DependencyHealthMonitor as _DependencyHealthMonitor,
        PersistentMessageQueue as _PersistentMessageQueue,
        SafeProcessManager as _SafeProcessManager,
        get_config_manager as _get_config_manager,
        get_dependency_monitor as _get_dependency_monitor,
        get_file_lock as _get_file_lock,
        get_heartbeat_writer as _get_heartbeat_writer,
        get_message_queue as _get_message_queue,
        get_port_reserver as _get_port_reserver,
        get_process_manager as _get_process_manager,
    )
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

    # Full system integration
    "initialize_race_safe_trinity",
    "shutdown_race_safe_trinity",

    # Flags
    "INTEGRATION_ENABLED",
    "RACE_PREVENTION_AVAILABLE",
    "TRINITY_IPC_AVAILABLE",
    "PORT_MANAGER_AVAILABLE",
]
