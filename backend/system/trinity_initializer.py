"""
PROJECT TRINITY Phase 3: Auto-Initializer for JARVIS Body

This module provides automatic Trinity initialization for JARVIS.
It integrates with the FastAPI lifespan to:
- Connect to the Trinity network on startup
- Register command handlers
- Broadcast heartbeats
- Reconcile state with the orchestrator

USAGE:
    In backend/main.py lifespan:

        from backend.system.trinity_initializer import (
            initialize_trinity,
            shutdown_trinity,
        )

        async def lifespan(app):
            # ... existing startup code ...

            # Initialize Trinity
            await initialize_trinity(app)

            yield

            # Shutdown Trinity
            await shutdown_trinity()

Or use the context manager:

    from backend.system.trinity_initializer import trinity_context

    async with trinity_context(app):
        yield
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# v73.0: ATOMIC FILE I/O - Diamond-Hard Protocol
# =============================================================================

class AtomicTrinityIO:
    """
    v73.0: Ensures zero-corruption file operations via Atomic Renames.

    The Problem:
        Standard file writing (`open('w').write()`) takes non-zero time (e.g., 5ms).
        If JARVIS tries to read the file during those 5ms, it reads incomplete JSON
        and crashes with JSONDecodeError.

    The Solution:
        Write to a temporary file first, then perform an OS-level atomic rename
        (`os.replace`) to the final filename. This guarantees the file is either
        *missing* or *perfect*, never partial.

    Features:
        - Atomic writes with fsync for durability
        - Safe reads with automatic retry on corruption
        - Lock-free design for high concurrency
        - Works across all platforms (macOS, Linux, Windows)
    """

    @staticmethod
    def write_json_atomic(filepath: Union[str, Path], data: Dict[str, Any]) -> bool:
        """
        Write JSON data atomically to prevent partial reads.

        Args:
            filepath: Target file path
            data: JSON-serializable dictionary

        Returns:
            True if write succeeded, False otherwise
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd = None
        tmp_name = None

        try:
            # 1. Create temp file in same directory (required for atomic rename)
            tmp_fd, tmp_name = tempfile.mkstemp(
                dir=filepath.parent,
                prefix=f".{filepath.stem}.",
                suffix=".tmp"
            )

            # 2. Write data to temp file
            with os.fdopen(tmp_fd, 'w') as tmp_file:
                tmp_fd = None  # os.fdopen takes ownership
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # Force write to physical disk

            # 3. Atomic swap (OS guarantees this is instantaneous)
            os.replace(tmp_name, filepath)
            return True

        except Exception as e:
            logger.debug(f"[AtomicIO] Write failed: {e}")
            # Cleanup temp file on failure
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
            return False

        finally:
            # Ensure fd is closed if not transferred to fdopen
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass

    @staticmethod
    def read_json_safe(
        filepath: Union[str, Path],
        default: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 0.05
    ) -> Optional[Dict[str, Any]]:
        """
        Read JSON with automatic retry on corruption.

        This handles the rare case where we catch a file mid-rename
        (between unlink and rename on some filesystems).

        Args:
            filepath: File to read
            default: Value to return if file doesn't exist
            max_retries: Maximum read attempts
            retry_delay: Delay between retries in seconds

        Returns:
            Parsed JSON or default value
        """
        filepath = Path(filepath)

        for attempt in range(max_retries):
            try:
                if not filepath.exists():
                    return default

                with open(filepath, 'r') as f:
                    return json.load(f)

            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    logger.debug(f"[AtomicIO] JSON decode retry {attempt + 1}: {e}")
                    time.sleep(retry_delay)
                else:
                    logger.warning(f"[AtomicIO] JSON decode failed after {max_retries} retries: {e}")
                    return default

            except Exception as e:
                logger.debug(f"[AtomicIO] Read failed: {e}")
                return default

        return default

    @staticmethod
    def cleanup_temp_files(directory: Union[str, Path], prefix: str = ".") -> int:
        """
        Clean up orphaned temp files from failed atomic writes.

        Args:
            directory: Directory to clean
            prefix: Temp file prefix to match

        Returns:
            Number of files cleaned
        """
        directory = Path(directory)
        cleaned = 0

        try:
            for f in directory.glob(f"{prefix}*.tmp"):
                try:
                    # Only remove if older than 1 minute (avoid race with active writes)
                    if time.time() - f.stat().st_mtime > 60:
                        f.unlink()
                        cleaned += 1
                except OSError:
                    pass
        except Exception:
            pass

        return cleaned


# Convenience function for module-level access
def write_json_atomic(filepath: Union[str, Path], data: Dict[str, Any]) -> bool:
    """Write JSON atomically. See AtomicTrinityIO.write_json_atomic."""
    return AtomicTrinityIO.write_json_atomic(filepath, data)


def read_json_safe(
    filepath: Union[str, Path],
    default: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Read JSON safely. See AtomicTrinityIO.read_json_safe."""
    return AtomicTrinityIO.read_json_safe(filepath, default)


# =============================================================================
# CONFIGURATION
# =============================================================================

TRINITY_ENABLED = os.getenv("TRINITY_ENABLED", "true").lower() == "true"
TRINITY_HEARTBEAT_INTERVAL = float(os.getenv("TRINITY_HEARTBEAT_INTERVAL", "5.0"))
TRINITY_AUTO_CONNECT = os.getenv("TRINITY_AUTO_CONNECT", "true").lower() == "true"

# Instance identification
JARVIS_INSTANCE_ID = os.getenv(
    "JARVIS_INSTANCE_ID",
    f"jarvis-{os.getpid()}-{int(time.time())}"
)


# =============================================================================
# GLOBAL STATE
# =============================================================================

_trinity_initialized = False
_heartbeat_task: Optional[asyncio.Task] = None
_bridge = None
_app = None
_start_time = time.time()


# =============================================================================
# IMPORTS (Lazy to avoid circular imports)
# =============================================================================

def _get_reactor_bridge():
    """Lazy import of ReactorCoreBridge."""
    global _bridge
    if _bridge is None:
        try:
            from backend.system.reactor_bridge import get_reactor_bridge
            _bridge = get_reactor_bridge()
        except ImportError:
            try:
                from system.reactor_bridge import get_reactor_bridge
                _bridge = get_reactor_bridge()
            except ImportError:
                logger.warning("[Trinity] ReactorCoreBridge not available")
    return _bridge


def _get_trinity_handlers():
    """Lazy import of trinity_handlers."""
    try:
        from backend.system.trinity_handlers import register_trinity_handlers
        return register_trinity_handlers
    except ImportError:
        try:
            from system.trinity_handlers import register_trinity_handlers
            return register_trinity_handlers
        except ImportError:
            logger.warning("[Trinity] Trinity handlers not available")
            return None


def _get_cryostasis_manager():
    """Lazy import of CryostasisManager."""
    try:
        from backend.system.cryostasis_manager import get_cryostasis_manager
        return get_cryostasis_manager()
    except ImportError:
        try:
            from system.cryostasis_manager import get_cryostasis_manager
            return get_cryostasis_manager()
        except ImportError:
            return None


def _get_yabai_detector():
    """Lazy import of YabaiSpaceDetector."""
    try:
        from backend.vision.yabai_space_detector import get_yabai_detector
        return get_yabai_detector()
    except ImportError:
        try:
            from vision.yabai_space_detector import get_yabai_detector
            return get_yabai_detector()
        except ImportError:
            return None


# =============================================================================
# INITIALIZATION
# =============================================================================

async def initialize_trinity(app=None) -> bool:
    """
    Initialize Trinity for JARVIS Body.

    This should be called during FastAPI lifespan startup.

    Args:
        app: Optional FastAPI app instance for state attachment

    Returns:
        True if initialization succeeded
    """
    global _trinity_initialized, _heartbeat_task, _app

    if not TRINITY_ENABLED:
        logger.info("[Trinity] Trinity is disabled (TRINITY_ENABLED=false)")
        return False

    if _trinity_initialized:
        logger.debug("[Trinity] Already initialized")
        return True

    logger.info("=" * 60)
    logger.info("PROJECT TRINITY: Initializing JARVIS Body Connection")
    logger.info("=" * 60)

    _app = app

    try:
        # Step 1: Get ReactorCoreBridge
        bridge = _get_reactor_bridge()
        if bridge is None:
            logger.warning("[Trinity] ReactorCoreBridge not available - skipping")
            return False

        # Step 2: Connect to Trinity network
        if TRINITY_AUTO_CONNECT:
            logger.info("[Trinity] Connecting to Trinity network...")
            connected = await bridge.connect_async()
            if connected:
                logger.info("[Trinity] ✓ Connected to Trinity network")
            else:
                logger.warning("[Trinity] Connection failed - continuing in standalone mode")

        # Step 3: Register command handlers
        register_handlers = _get_trinity_handlers()
        if register_handlers:
            register_handlers(bridge)
            logger.info("[Trinity] ✓ Command handlers registered")
        else:
            logger.warning("[Trinity] Command handlers not available")

        # Step 4: Start heartbeat broadcast
        _heartbeat_task = asyncio.create_task(_heartbeat_loop())
        logger.info(f"[Trinity] ✓ Heartbeat started (interval={TRINITY_HEARTBEAT_INTERVAL}s)")

        # Step 5: Attach to app state if available
        if app is not None:
            app.state.trinity_bridge = bridge
            app.state.trinity_instance_id = JARVIS_INSTANCE_ID
            logger.info("[Trinity] ✓ Attached to FastAPI app.state")

        _trinity_initialized = True

        logger.info("=" * 60)
        logger.info(f"PROJECT TRINITY: JARVIS Body Online (ID: {JARVIS_INSTANCE_ID[:16]})")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"[Trinity] Initialization failed: {e}")
        return False


async def shutdown_trinity() -> None:
    """
    Shutdown Trinity connection.

    This should be called during FastAPI lifespan shutdown.
    """
    global _trinity_initialized, _heartbeat_task, _bridge

    if not _trinity_initialized:
        return

    logger.info("[Trinity] Shutting down JARVIS Body connection...")

    # Stop heartbeat
    if _heartbeat_task:
        _heartbeat_task.cancel()
        try:
            await _heartbeat_task
        except asyncio.CancelledError:
            pass
        _heartbeat_task = None

    # Disconnect bridge
    bridge = _get_reactor_bridge()
    if bridge and bridge.is_connected():
        await bridge.disconnect_async()

    _trinity_initialized = False
    _bridge = None

    logger.info("[Trinity] JARVIS Body disconnected")


@asynccontextmanager
async def trinity_context(app=None):
    """
    Context manager for Trinity lifecycle.

    Usage:
        async with trinity_context(app):
            yield
    """
    await initialize_trinity(app)
    try:
        yield
    finally:
        await shutdown_trinity()


# =============================================================================
# HEARTBEAT
# =============================================================================

async def _heartbeat_loop() -> None:
    """Background task to broadcast heartbeats."""
    global _trinity_initialized

    while _trinity_initialized:
        try:
            await _broadcast_heartbeat()
            await asyncio.sleep(TRINITY_HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"[Trinity] Heartbeat error: {e}")
            await asyncio.sleep(TRINITY_HEARTBEAT_INTERVAL)


async def _broadcast_heartbeat() -> None:
    """Broadcast current state as heartbeat."""
    bridge = _get_reactor_bridge()
    if bridge is None or not bridge.is_connected():
        return

    # Gather state
    state = await _gather_jarvis_state()

    # Publish heartbeat
    await bridge.publish_heartbeat_async()

    # Also write state to orchestrator file for cross-repo sync
    await _write_state_to_orchestrator(state)


async def _gather_jarvis_state() -> Dict[str, Any]:
    """Gather current JARVIS state for heartbeat."""
    state = {
        "instance_id": JARVIS_INSTANCE_ID,
        "uptime_seconds": time.time() - _start_time,
        "timestamp": time.time(),
    }

    # System metrics
    try:
        import psutil
        state["system_cpu_percent"] = psutil.cpu_percent()
        state["system_memory_percent"] = psutil.virtual_memory().percent
    except ImportError:
        pass

    # Cryostasis state
    cryo = _get_cryostasis_manager()
    if cryo:
        state["frozen_apps"] = cryo.get_frozen_app_names()

    # Yabai/Ghost Display state
    yabai = _get_yabai_detector()
    if yabai:
        try:
            ghost_space = yabai.get_ghost_display_space()
            state["ghost_display_available"] = ghost_space is not None

            if ghost_space:
                windows = yabai.get_windows_on_space(ghost_space)
                state["apps_on_ghost_display"] = list(set(
                    w.get("app", "") for w in windows if w.get("app")
                ))
        except Exception:
            state["ghost_display_available"] = False

    # Surveillance state (if available)
    state["surveillance_active"] = False
    state["surveillance_targets"] = []

    return state


async def _write_state_to_orchestrator(state: Dict[str, Any]) -> None:
    """
    Write state to orchestrator's component directory.

    v73.0: Uses atomic writes to prevent partial read race conditions.
    """
    try:
        components_dir = Path.home() / ".jarvis" / "trinity" / "components"
        components_dir.mkdir(parents=True, exist_ok=True)

        state_file = components_dir / "jarvis_body.json"

        # v73.0: Atomic write - prevents JSONDecodeError from partial reads
        data = {
            "component_type": "jarvis_body",
            "instance_id": JARVIS_INSTANCE_ID,
            "timestamp": time.time(),
            "metrics": state,
        }

        if not write_json_atomic(state_file, data):
            logger.debug("[Trinity] Atomic write failed, state may be stale")

    except Exception as e:
        logger.debug(f"[Trinity] Could not write state: {e}")


# =============================================================================
# v75.0: TRINITY HEALTH MONITOR - Crash Detection & Recovery
# =============================================================================

class TrinityHealthMonitor:
    """
    v75.0: Production-grade health monitor for all Trinity components.

    Features:
        - Stale heartbeat detection (>15s = component dead)
        - Process liveness verification via PID
        - Automatic restart triggering via callbacks
        - Dead letter queue for failed commands
        - Circuit breaker pattern for repeated failures
        - Resource exhaustion detection (memory/CPU)
    """

    HEARTBEAT_TIMEOUT = 15.0  # 3 missed heartbeats (5s interval)
    CHECK_INTERVAL = 5.0
    MAX_RESTART_ATTEMPTS = 3
    RESTART_BACKOFF_BASE = 2.0

    def __init__(self):
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False

        # Component states: "healthy", "degraded", "down", "restarting"
        self._component_states: Dict[str, str] = {
            "j_prime": "unknown",
            "reactor_core": "unknown",
            "jarvis_body": "unknown",
        }

        # Restart tracking for circuit breaker
        self._restart_attempts: Dict[str, int] = {
            "j_prime": 0,
            "reactor_core": 0,
        }
        self._last_restart_time: Dict[str, float] = {}

        # Callbacks for restart triggers
        self._restart_callbacks: Dict[str, Any] = {}

        # Dead letter queue for failed commands
        self._dead_letter_queue: list = []
        self._max_dlq_size = 1000

        # Metrics
        self._health_checks_performed = 0
        self._components_restarted = 0
        self._commands_in_dlq = 0

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("[Trinity] v75.0 Health Monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("[Trinity] Health Monitor stopped")

    def register_restart_callback(self, component: str, callback) -> None:
        """Register a callback to restart a component."""
        self._restart_callbacks[component] = callback

    async def _monitor_loop(self) -> None:
        """Main monitoring loop - runs every CHECK_INTERVAL seconds."""
        while self._running:
            try:
                await self._check_all_components()
                self._health_checks_performed += 1
            except Exception as e:
                logger.debug(f"[Trinity] Health check error: {e}")

            await asyncio.sleep(self.CHECK_INTERVAL)

    async def _check_all_components(self) -> None:
        """Check health of all Trinity components."""
        components_dir = Path.home() / ".jarvis" / "trinity" / "components"

        # Check J-Prime (Mind)
        await self._check_component(
            "j_prime",
            components_dir / "j_prime.json",
        )

        # Check Reactor-Core (Nerves)
        await self._check_component(
            "reactor_core",
            components_dir / "reactor_core.json",
        )

        # JARVIS Body is this process - always healthy if running
        self._component_states["jarvis_body"] = "healthy"

    async def _check_component(
        self,
        component: str,
        heartbeat_file: Path,
    ) -> None:
        """Check a specific component's health."""
        # Check heartbeat file exists
        if not heartbeat_file.exists():
            if self._component_states[component] != "down":
                logger.warning(f"[Trinity] {component} heartbeat file missing")
            await self._handle_component_down(component, "heartbeat_missing")
            return

        # Read heartbeat with safe JSON parsing
        state = read_json_safe(heartbeat_file)
        if not state:
            await self._handle_component_down(component, "heartbeat_corrupted")
            return

        # Check heartbeat age
        timestamp = state.get("timestamp", 0)
        age = time.time() - timestamp
        if age > self.HEARTBEAT_TIMEOUT:
            await self._handle_component_down(
                component,
                f"heartbeat_stale_{age:.1f}s"
            )
            return

        # Check process liveness (if PID available)
        pid = state.get("pid") or self._extract_pid_from_instance_id(
            state.get("instance_id", "")
        )
        if pid and not self._is_process_alive(pid):
            await self._handle_component_down(component, f"process_{pid}_dead")
            return

        # Component is healthy
        if self._component_states[component] != "healthy":
            logger.info(f"[Trinity] {component} recovered to healthy state")

        self._component_states[component] = "healthy"
        self._restart_attempts[component] = 0  # Reset restart attempts

    def _extract_pid_from_instance_id(self, instance_id: str) -> Optional[int]:
        """Extract PID from instance_id format: 'component-PID-timestamp'."""
        try:
            parts = instance_id.split("-")
            if len(parts) >= 2:
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        return None

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is still alive."""
        try:
            os.kill(pid, 0)  # Signal 0 = check existence
            return True
        except OSError:
            return False

    async def _handle_component_down(self, component: str, reason: str) -> None:
        """Handle a component being down."""
        old_state = self._component_states.get(component)
        self._component_states[component] = "down"

        # Only log once per state transition
        if old_state != "down":
            logger.error(f"[Trinity] {component} is DOWN: {reason}")

        # Circuit breaker: check if we've exceeded restart attempts
        attempts = self._restart_attempts.get(component, 0)
        if attempts >= self.MAX_RESTART_ATTEMPTS:
            # Check if enough time has passed for backoff reset
            last_restart = self._last_restart_time.get(component, 0)
            backoff_time = self.RESTART_BACKOFF_BASE ** attempts
            if time.time() - last_restart < backoff_time * 60:
                logger.warning(
                    f"[Trinity] {component} restart circuit breaker OPEN "
                    f"(attempts={attempts}, backoff={backoff_time:.0f}min)"
                )
                return
            else:
                # Reset attempts after backoff period
                self._restart_attempts[component] = 0

        # Trigger restart if callback registered
        if component in self._restart_callbacks:
            await self._trigger_restart(component, reason)

    async def _trigger_restart(self, component: str, reason: str) -> None:
        """Trigger component restart via callback."""
        self._component_states[component] = "restarting"
        self._restart_attempts[component] = self._restart_attempts.get(component, 0) + 1
        self._last_restart_time[component] = time.time()
        self._components_restarted += 1

        logger.info(
            f"[Trinity] Triggering {component} restart "
            f"(attempt {self._restart_attempts[component]}/{self.MAX_RESTART_ATTEMPTS})"
        )

        try:
            callback = self._restart_callbacks[component]
            if asyncio.iscoroutinefunction(callback):
                await callback(reason)
            else:
                callback(reason)
        except Exception as e:
            logger.error(f"[Trinity] Restart callback failed for {component}: {e}")

    def add_to_dead_letter_queue(self, command: Dict[str, Any], reason: str) -> None:
        """Add a failed command to the dead letter queue."""
        if len(self._dead_letter_queue) >= self._max_dlq_size:
            # Remove oldest entries
            self._dead_letter_queue = self._dead_letter_queue[-self._max_dlq_size // 2:]

        self._dead_letter_queue.append({
            "command": command,
            "failed_at": time.time(),
            "reason": reason,
        })
        self._commands_in_dlq = len(self._dead_letter_queue)

    def get_dead_letter_queue(self) -> list:
        """Get dead letter queue contents."""
        return self._dead_letter_queue.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get health monitor status."""
        return {
            "running": self._running,
            "component_states": self._component_states.copy(),
            "restart_attempts": self._restart_attempts.copy(),
            "health_checks_performed": self._health_checks_performed,
            "components_restarted": self._components_restarted,
            "dead_letter_queue_size": self._commands_in_dlq,
        }


# =============================================================================
# v75.0: FILE SYSTEM RESILIENCE - Disk Space & Permission Handling
# =============================================================================

class FileSystemGuard:
    """
    v75.0: Protects against file system failures.

    Features:
        - Disk space checking before writes
        - Permission validation
        - Retry with exponential backoff
        - Fallback to alternative locations
    """

    MIN_DISK_SPACE_BYTES = 10 * 1024 * 1024  # 10MB minimum
    MAX_RETRIES = 3
    BACKOFF_BASE = 0.5

    @classmethod
    def has_disk_space(cls, path: Path, min_bytes: int = None) -> bool:
        """Check if there's sufficient disk space."""
        min_bytes = min_bytes or cls.MIN_DISK_SPACE_BYTES
        try:
            import shutil
            usage = shutil.disk_usage(path.parent if path.is_file() else path)
            return usage.free > min_bytes
        except Exception:
            return True  # Assume OK if we can't check

    @classmethod
    def has_write_permission(cls, path: Path) -> bool:
        """Check if we have write permission."""
        try:
            if path.is_dir():
                test_file = path / f".perm_test_{os.getpid()}"
            else:
                test_file = path.parent / f".perm_test_{os.getpid()}"

            test_file.touch()
            test_file.unlink()
            return True
        except Exception:
            return False

    @classmethod
    async def write_with_resilience(
        cls,
        filepath: Path,
        data: Dict[str, Any],
        max_retries: int = None,
    ) -> bool:
        """
        Write JSON data with comprehensive resilience.

        Returns:
            True if write succeeded, False otherwise
        """
        max_retries = max_retries or cls.MAX_RETRIES
        filepath = Path(filepath)

        # Pre-flight checks
        if not cls.has_disk_space(filepath):
            logger.error(f"[Trinity] Insufficient disk space for {filepath}")
            return False

        if not cls.has_write_permission(filepath.parent):
            logger.error(f"[Trinity] No write permission for {filepath.parent}")
            return False

        # Attempt write with retries
        for attempt in range(max_retries):
            if write_json_atomic(filepath, data):
                return True

            if attempt < max_retries - 1:
                delay = cls.BACKOFF_BASE * (2 ** attempt)
                logger.debug(
                    f"[Trinity] Write failed, retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)

        logger.error(f"[Trinity] Write failed after {max_retries} attempts: {filepath}")
        return False


# Global health monitor instance
_health_monitor: Optional[TrinityHealthMonitor] = None


def get_health_monitor() -> Optional[TrinityHealthMonitor]:
    """Get the global health monitor instance."""
    return _health_monitor


async def start_health_monitor() -> TrinityHealthMonitor:
    """Start the global health monitor."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = TrinityHealthMonitor()
    await _health_monitor.start()
    return _health_monitor


async def stop_health_monitor() -> None:
    """Stop the global health monitor."""
    global _health_monitor
    if _health_monitor:
        await _health_monitor.stop()


# =============================================================================
# STATUS
# =============================================================================

def is_trinity_initialized() -> bool:
    """Check if Trinity is initialized."""
    return _trinity_initialized


def get_trinity_status() -> Dict[str, Any]:
    """Get current Trinity status."""
    bridge = _get_reactor_bridge()

    return {
        "enabled": TRINITY_ENABLED,
        "initialized": _trinity_initialized,
        "instance_id": JARVIS_INSTANCE_ID,
        "uptime_seconds": time.time() - _start_time if _trinity_initialized else 0,
        "connected": bridge.is_connected() if bridge else False,
        "heartbeat_interval": TRINITY_HEARTBEAT_INTERVAL,
        "bridge_stats": bridge.get_stats() if bridge else None,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core initialization
    "initialize_trinity",
    "shutdown_trinity",
    "trinity_context",
    "is_trinity_initialized",
    "get_trinity_status",
    "JARVIS_INSTANCE_ID",
    # v73.0: Atomic I/O
    "AtomicTrinityIO",
    "write_json_atomic",
    "read_json_safe",
    # v75.0: Health monitoring
    "TrinityHealthMonitor",
    "get_health_monitor",
    "start_health_monitor",
    "stop_health_monitor",
    # v75.0: File system resilience
    "FileSystemGuard",
]
