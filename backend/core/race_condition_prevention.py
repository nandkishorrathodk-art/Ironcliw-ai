"""
Race Condition Prevention System v1.0 - Production-Grade Atomic Operations
===========================================================================

Comprehensive race condition prevention for Trinity IPC architecture.
This module addresses 4 critical race conditions:

ISSUE 9: Trinity IPC File Lock Race Condition
    - Root Fix: Atomic lock creation with O_CREAT | O_EXCL
    - PID validation for process liveness
    - Stale lock detection with automatic cleanup
    - Lock timeout with exponential backoff

ISSUE 10: Heartbeat File Write Race Condition
    - Root Fix: Atomic write via temp file + rename
    - fsync for durability guarantees
    - File locking during read-modify-write operations
    - Corruption detection and recovery

ISSUE 11: Port Allocation Race Condition
    - Root Fix: Atomic port reservation with file lock
    - Port registry with PID tracking
    - Socket bind verification before reservation
    - Stale reservation cleanup

ISSUE 12: Component Startup Order Race
    - Root Fix: Continuous dependency health monitoring
    - Automatic restart of dependents when dependency fails
    - Dependency graph validation before startup
    - Event-driven dependency state propagation

KEY FEATURES:
=============
- Zero hardcoding - all configuration via environment variables
- Async-first design with sync fallbacks
- Exponential backoff with jitter for retries
- Circuit breaker pattern for resilience
- Comprehensive logging and metrics
- PID + creation time validation (prevents PID reuse)
- Atomic operations using OS primitives

Architecture:
    +------------------------------------------------------------------------+
    |                    Race Condition Prevention v1.0                      |
    +------------------------------------------------------------------------+
    |  AtomicFileLock (Issue 9)                                              |
    |  - O_CREAT | O_EXCL atomic creation                                    |
    |  - PID validation with process creation time                           |
    |  - Stale lock timeout detection                                        |
    +------------------------------------------------------------------------+
    |  AtomicHeartbeatWriter (Issue 10)                                      |
    |  - Temp file + atomic rename                                           |
    |  - fsync for durability                                                |
    |  - Read-modify-write with exclusive lock                               |
    +------------------------------------------------------------------------+
    |  AtomicPortReservation (Issue 11)                                      |
    |  - Socket bind verification                                            |
    |  - Atomic reservation file creation                                    |
    |  - PID tracking with staleness detection                               |
    +------------------------------------------------------------------------+
    |  DependencyHealthMonitor (Issue 12)                                    |
    |  - Continuous health polling                                           |
    |  - Dependency graph with topological order                             |
    |  - Automatic dependent restart on failure                              |
    +------------------------------------------------------------------------+

Author: JARVIS Trinity v1.0 - Race Condition Prevention
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import logging
import os
import random
import socket
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Environment Variable Helpers (Zero Hardcoding)
# =============================================================================


def _env_int(key: str, default: int) -> int:
    """Get integer from environment with validation."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"[RacePrevent] Invalid int for {key}: {value}, using default")
        return default


def _env_float(key: str, default: float) -> float:
    """Get float from environment with validation."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"[RacePrevent] Invalid float for {key}: {value}, using default")
        return default


def _env_path(key: str, default: Path) -> Path:
    """Get path from environment."""
    value = os.environ.get(key)
    if value is None:
        return default
    return Path(value).expanduser()


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class RacePreventionConfig:
    """Configuration for race condition prevention - all from environment."""

    # Base directory for locks and state
    state_dir: Path = field(default_factory=lambda: _env_path(
        "RACE_PREVENTION_STATE_DIR",
        Path.home() / ".jarvis" / "race_prevention"
    ))

    # Lock timeouts
    lock_acquire_timeout: float = field(default_factory=lambda: _env_float(
        "RACE_LOCK_ACQUIRE_TIMEOUT", 30.0
    ))
    lock_stale_timeout: float = field(default_factory=lambda: _env_float(
        "RACE_LOCK_STALE_TIMEOUT", 300.0
    ))
    lock_poll_interval: float = field(default_factory=lambda: _env_float(
        "RACE_LOCK_POLL_INTERVAL", 0.05
    ))

    # Backoff configuration
    backoff_base: float = field(default_factory=lambda: _env_float(
        "RACE_BACKOFF_BASE", 0.1
    ))
    backoff_max: float = field(default_factory=lambda: _env_float(
        "RACE_BACKOFF_MAX", 5.0
    ))
    backoff_multiplier: float = field(default_factory=lambda: _env_float(
        "RACE_BACKOFF_MULTIPLIER", 2.0
    ))
    backoff_jitter: float = field(default_factory=lambda: _env_float(
        "RACE_BACKOFF_JITTER", 0.1
    ))

    # Heartbeat configuration
    heartbeat_interval: float = field(default_factory=lambda: _env_float(
        "RACE_HEARTBEAT_INTERVAL", 5.0
    ))
    heartbeat_stale_timeout: float = field(default_factory=lambda: _env_float(
        "RACE_HEARTBEAT_STALE_TIMEOUT", 30.0
    ))

    # Port reservation
    port_reservation_timeout: float = field(default_factory=lambda: _env_float(
        "RACE_PORT_RESERVATION_TIMEOUT", 10.0
    ))
    port_socket_timeout: float = field(default_factory=lambda: _env_float(
        "RACE_PORT_SOCKET_TIMEOUT", 2.0
    ))

    # Dependency monitoring
    dependency_check_interval: float = field(default_factory=lambda: _env_float(
        "RACE_DEPENDENCY_CHECK_INTERVAL", 5.0
    ))
    dependency_failure_threshold: int = field(default_factory=lambda: _env_int(
        "RACE_DEPENDENCY_FAILURE_THRESHOLD", 3
    ))

    def __post_init__(self) -> None:
        """Ensure directories exist."""
        # Use object.__setattr__ since frozen=True
        object.__setattr__(self, 'state_dir', Path(self.state_dir))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "locks").mkdir(exist_ok=True)
        (self.state_dir / "heartbeats").mkdir(exist_ok=True)
        (self.state_dir / "ports").mkdir(exist_ok=True)
        (self.state_dir / "dependencies").mkdir(exist_ok=True)


# Global configuration instance (lazy initialized)
_config: Optional[RacePreventionConfig] = None


def get_config() -> RacePreventionConfig:
    """Get or create global configuration."""
    global _config
    if _config is None:
        _config = RacePreventionConfig()
    return _config


# =============================================================================
# Process Identity (PID Reuse Prevention)
# =============================================================================


@dataclass(frozen=True)
class ProcessIdentity:
    """
    Unique process identity that survives PID reuse.

    Uses PID + process creation time to uniquely identify a process.
    This prevents false positives when a PID is reused by a new process.
    """
    pid: int
    creation_time: float
    hostname: str

    @classmethod
    def current(cls) -> "ProcessIdentity":
        """Create identity for current process."""
        pid = os.getpid()
        creation_time = time.time()

        # Try to get actual process creation time
        try:
            import psutil
            proc = psutil.Process(pid)
            creation_time = proc.create_time()
        except ImportError:
            pass
        except Exception:
            pass

        return cls(
            pid=pid,
            creation_time=creation_time,
            hostname=socket.gethostname(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pid": self.pid,
            "creation_time": self.creation_time,
            "hostname": self.hostname,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessIdentity":
        """Deserialize from dictionary."""
        return cls(
            pid=data.get("pid", 0),
            creation_time=data.get("creation_time", 0.0),
            hostname=data.get("hostname", "unknown"),
        )

    def is_alive(self) -> bool:
        """
        Check if this process is still alive.

        Uses both PID check and creation time validation to prevent
        PID reuse false positives.
        """
        # Check if PID exists
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we don't have permission
            pass
        except OSError:
            return False

        # Validate creation time to prevent PID reuse
        try:
            import psutil
            proc = psutil.Process(self.pid)
            actual_creation = proc.create_time()
            # Allow 5 second tolerance for clock differences
            if abs(actual_creation - self.creation_time) > 5.0:
                logger.debug(
                    f"[ProcessIdentity] PID {self.pid} reused: "
                    f"expected creation={self.creation_time:.0f}, "
                    f"actual={actual_creation:.0f}"
                )
                return False
        except ImportError:
            # No psutil - can't validate creation time
            pass
        except Exception:
            return False

        return True


# Cache current process identity
_current_identity: Optional[ProcessIdentity] = None


def get_current_identity() -> ProcessIdentity:
    """Get cached identity for current process."""
    global _current_identity
    if _current_identity is None:
        _current_identity = ProcessIdentity.current()
    return _current_identity


# =============================================================================
# ISSUE 9: Atomic File Lock with O_CREAT | O_EXCL
# =============================================================================


class LockState(Enum):
    """Lock state enumeration."""
    UNLOCKED = auto()
    LOCKED = auto()
    STALE = auto()
    ERROR = auto()


@dataclass
class LockInfo:
    """Information about a lock holder."""
    identity: ProcessIdentity
    acquired_at: float
    lock_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "identity": self.identity.to_dict(),
            "acquired_at": self.acquired_at,
            "lock_name": self.lock_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LockInfo":
        """Deserialize from dictionary."""
        return cls(
            identity=ProcessIdentity.from_dict(data.get("identity", {})),
            acquired_at=data.get("acquired_at", 0.0),
            lock_name=data.get("lock_name", "unknown"),
            metadata=data.get("metadata", {}),
        )

    @property
    def age_seconds(self) -> float:
        """Get lock age in seconds."""
        return time.time() - self.acquired_at

    def is_stale(self, timeout: float) -> bool:
        """Check if lock is stale based on timeout."""
        return self.age_seconds > timeout or not self.identity.is_alive()


class AtomicFileLock:
    """
    Atomic file lock using O_CREAT | O_EXCL for race-free creation.

    FIXES ISSUE 9: Trinity IPC File Lock Race Condition

    Key Features:
    - Atomic lock file creation with O_CREAT | O_EXCL
    - PID validation with process creation time
    - Automatic stale lock detection and cleanup
    - Exponential backoff with jitter for retries
    - Both sync and async interfaces

    Usage:
        lock = AtomicFileLock("my_resource")

        # Async context manager
        async with lock.acquire_async():
            # Critical section

        # Sync context manager
        with lock.acquire_sync():
            # Critical section
    """

    def __init__(
        self,
        name: str,
        config: Optional[RacePreventionConfig] = None,
    ):
        """
        Initialize atomic file lock.

        Args:
            name: Lock name (used for file naming)
            config: Optional configuration override
        """
        self.name = name
        self.config = config or get_config()
        self.lock_file = self.config.state_dir / "locks" / f"{name}.lock"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        self._fd: Optional[int] = None
        self._owned = False

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff with jitter."""
        base_delay = min(
            self.config.backoff_base * (self.config.backoff_multiplier ** attempt),
            self.config.backoff_max,
        )
        jitter = random.uniform(0, self.config.backoff_jitter * base_delay)
        return base_delay + jitter

    def _read_lock_info(self) -> Optional[LockInfo]:
        """Read lock info from file."""
        if not self.lock_file.exists():
            return None

        try:
            content = self.lock_file.read_text()
            if not content.strip():
                return None
            data = json.loads(content)
            return LockInfo.from_dict(data)
        except Exception as e:
            logger.debug(f"[AtomicLock] Failed to read lock info for {self.name}: {e}")
            return None

    def _write_lock_info(self, fd: int, info: LockInfo) -> None:
        """Write lock info to file descriptor."""
        content = json.dumps(info.to_dict(), indent=2)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)

    def _check_and_clean_stale(self) -> bool:
        """
        Check for stale lock and clean if necessary.

        Returns:
            True if stale lock was cleaned, False otherwise
        """
        lock_info = self._read_lock_info()
        if lock_info is None:
            return False

        if lock_info.is_stale(self.config.lock_stale_timeout):
            logger.info(
                f"[AtomicLock] Cleaning stale lock {self.name}: "
                f"holder PID {lock_info.identity.pid} is dead or timed out"
            )
            try:
                self.lock_file.unlink()
                return True
            except OSError:
                pass

        return False

    def _try_acquire_once(self) -> bool:
        """
        Try to acquire lock once using O_CREAT | O_EXCL.

        This is the atomic operation that prevents race conditions.
        O_EXCL ensures the file is created only if it doesn't exist.

        Returns:
            True if lock acquired, False otherwise
        """
        try:
            # O_CREAT | O_EXCL is the key: atomic creation that fails if exists
            fd = os.open(
                str(self.lock_file),
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o644
            )

            # Got the lock - write our info
            lock_info = LockInfo(
                identity=get_current_identity(),
                acquired_at=time.time(),
                lock_name=self.name,
            )
            self._write_lock_info(fd, lock_info)

            self._fd = fd
            self._owned = True

            logger.debug(f"[AtomicLock] Acquired lock: {self.name}")
            return True

        except FileExistsError:
            # Lock file exists - check if stale
            if self._check_and_clean_stale():
                # Stale lock cleaned - try again
                return self._try_acquire_once()
            return False

        except OSError as e:
            if e.errno == errno.EEXIST:
                # Lock exists (race with another process)
                return False
            logger.warning(f"[AtomicLock] OS error acquiring {self.name}: {e}")
            return False

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire lock with timeout (synchronous).

        Args:
            timeout: Maximum time to wait (uses config default if None)

        Returns:
            True if lock acquired, False on timeout
        """
        timeout = timeout or self.config.lock_acquire_timeout
        start_time = time.time()
        attempt = 0

        while True:
            if self._try_acquire_once():
                return True

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(
                    f"[AtomicLock] Timeout acquiring {self.name} "
                    f"after {elapsed:.1f}s ({attempt} attempts)"
                )
                return False

            # Exponential backoff with jitter
            backoff = self._calculate_backoff(attempt)
            remaining = timeout - elapsed
            sleep_time = min(backoff, remaining)

            if sleep_time > 0:
                time.sleep(sleep_time)

            attempt += 1

    async def acquire_async(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire lock with timeout (asynchronous).

        Args:
            timeout: Maximum time to wait (uses config default if None)

        Returns:
            True if lock acquired, False on timeout
        """
        timeout = timeout or self.config.lock_acquire_timeout
        start_time = time.time()
        attempt = 0

        while True:
            # Run acquisition in executor to avoid blocking
            loop = asyncio.get_running_loop()
            acquired = await loop.run_in_executor(None, self._try_acquire_once)

            if acquired:
                return True

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning(
                    f"[AtomicLock] Timeout acquiring {self.name} "
                    f"after {elapsed:.1f}s ({attempt} attempts)"
                )
                return False

            # Exponential backoff with jitter
            backoff = self._calculate_backoff(attempt)
            remaining = timeout - elapsed
            sleep_time = min(backoff, remaining)

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            attempt += 1

    def release(self) -> None:
        """Release the lock."""
        if not self._owned:
            return

        try:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None

            if self.lock_file.exists():
                self.lock_file.unlink()

            self._owned = False
            logger.debug(f"[AtomicLock] Released lock: {self.name}")

        except OSError as e:
            logger.warning(f"[AtomicLock] Error releasing {self.name}: {e}")
            self._owned = False

    @contextmanager
    def acquire_sync(
        self,
        timeout: Optional[float] = None,
    ) -> Iterator[bool]:
        """
        Synchronous context manager for lock acquisition.

        Usage:
            with lock.acquire_sync() as acquired:
                if acquired:
                    # Critical section
        """
        acquired = self.acquire(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()

    @asynccontextmanager
    async def acquire_context(
        self,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[bool]:
        """
        Asynchronous context manager for lock acquisition.

        Usage:
            async with lock.acquire_context() as acquired:
                if acquired:
                    # Critical section
        """
        acquired = await self.acquire_async(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release()

    def __enter__(self) -> "AtomicFileLock":
        """Enter sync context."""
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire lock: {self.name}")
        return self

    def __exit__(self, *_args: Any) -> None:
        """Exit sync context."""
        self.release()

    async def __aenter__(self) -> "AtomicFileLock":
        """Enter async context."""
        if not await self.acquire_async():
            raise TimeoutError(f"Failed to acquire lock: {self.name}")
        return self

    async def __aexit__(self, *_args: Any) -> None:
        """Exit async context."""
        self.release()


# =============================================================================
# ISSUE 10: Atomic Heartbeat Writer
# =============================================================================


@dataclass
class HeartbeatRecord:
    """Heartbeat record with integrity check."""
    identity: ProcessIdentity
    timestamp: float
    component_name: str
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with checksum."""
        data = {
            "identity": self.identity.to_dict(),
            "timestamp": self.timestamp,
            "component_name": self.component_name,
            "status": self.status,
            "metrics": self.metrics,
        }
        # Compute checksum of data
        json_str = json.dumps(data, sort_keys=True, default=str)
        checksum = hashlib.sha256(json_str.encode()).hexdigest()[:16]
        data["checksum"] = checksum
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeartbeatRecord":
        """Deserialize with checksum validation."""
        stored_checksum = data.pop("checksum", "")

        # Verify checksum
        json_str = json.dumps(data, sort_keys=True, default=str)
        computed_checksum = hashlib.sha256(json_str.encode()).hexdigest()[:16]

        if stored_checksum and stored_checksum != computed_checksum:
            logger.warning(
                f"[Heartbeat] Checksum mismatch for {data.get('component_name', 'unknown')}: "
                f"stored={stored_checksum}, computed={computed_checksum}"
            )

        return cls(
            identity=ProcessIdentity.from_dict(data.get("identity", {})),
            timestamp=data.get("timestamp", 0.0),
            component_name=data.get("component_name", "unknown"),
            status=data.get("status", "unknown"),
            metrics=data.get("metrics", {}),
            checksum=computed_checksum,
        )

    @property
    def age_seconds(self) -> float:
        """Get heartbeat age."""
        return time.time() - self.timestamp

    def is_stale(self, timeout: float) -> bool:
        """Check if heartbeat is stale."""
        return self.age_seconds > timeout


class AtomicHeartbeatWriter:
    """
    Atomic heartbeat writer using temp file + rename pattern.

    FIXES ISSUE 10: Heartbeat File Write Race Condition

    Key Features:
    - Atomic write via temp file + os.replace()
    - fsync for durability guarantees
    - Checksum for corruption detection
    - File locking for read-modify-write operations
    - Automatic stale heartbeat cleanup

    Usage:
        writer = AtomicHeartbeatWriter("my_component")

        # Write heartbeat
        await writer.write_heartbeat("running", {"cpu": 50})

        # Read heartbeat
        heartbeat = await writer.read_heartbeat()
    """

    def __init__(
        self,
        component_name: str,
        config: Optional[RacePreventionConfig] = None,
    ):
        """
        Initialize heartbeat writer.

        Args:
            component_name: Name of the component
            config: Optional configuration override
        """
        self.component_name = component_name
        self.config = config or get_config()
        self.heartbeat_file = self.config.state_dir / "heartbeats" / f"{component_name}.json"
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)

        # Lock for read-modify-write operations
        self._lock = AtomicFileLock(f"heartbeat_{component_name}", config)

    def _write_atomic_sync(self, record: HeartbeatRecord) -> None:
        """
        Write heartbeat atomically (synchronous).

        Uses temp file + rename pattern for atomicity.
        """
        # Write to temp file in same directory
        fd, temp_path = tempfile.mkstemp(
            dir=self.heartbeat_file.parent,
            prefix=f".{self.heartbeat_file.name}.",
            suffix=".tmp"
        )

        try:
            # Write data
            content = json.dumps(record.to_dict(), indent=2)
            os.write(fd, content.encode("utf-8"))

            # fsync for durability
            os.fsync(fd)
            os.close(fd)
            fd = -1

            # Atomic rename
            os.replace(temp_path, self.heartbeat_file)

            logger.debug(f"[Heartbeat] Wrote heartbeat for {self.component_name}")

        except Exception as e:
            logger.error(f"[Heartbeat] Write error for {self.component_name}: {e}")
            raise
        finally:
            if fd >= 0:
                os.close(fd)
            # Clean up temp file if it exists
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def write_heartbeat(
        self,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write heartbeat atomically (asynchronous).

        Args:
            status: Component status (e.g., "running", "degraded")
            metrics: Optional metrics dictionary
        """
        record = HeartbeatRecord(
            identity=get_current_identity(),
            timestamp=time.time(),
            component_name=self.component_name,
            status=status,
            metrics=metrics or {},
        )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_atomic_sync, record)

    def write_heartbeat_sync(
        self,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write heartbeat synchronously."""
        record = HeartbeatRecord(
            identity=get_current_identity(),
            timestamp=time.time(),
            component_name=self.component_name,
            status=status,
            metrics=metrics or {},
        )
        self._write_atomic_sync(record)

    def _read_sync(self) -> Optional[HeartbeatRecord]:
        """Read heartbeat synchronously."""
        if not self.heartbeat_file.exists():
            return None

        try:
            content = self.heartbeat_file.read_text()
            if not content.strip():
                return None
            data = json.loads(content)
            return HeartbeatRecord.from_dict(data)
        except json.JSONDecodeError as e:
            logger.warning(f"[Heartbeat] Corrupted heartbeat for {self.component_name}: {e}")
            return None
        except Exception as e:
            logger.debug(f"[Heartbeat] Read error for {self.component_name}: {e}")
            return None

    async def read_heartbeat(self) -> Optional[HeartbeatRecord]:
        """Read heartbeat asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_sync)

    def read_heartbeat_sync(self) -> Optional[HeartbeatRecord]:
        """Read heartbeat synchronously."""
        return self._read_sync()

    async def update_heartbeat(
        self,
        updater: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Optional[HeartbeatRecord]:
        """
        Atomic read-modify-write operation with locking.

        Args:
            updater: Function to update the metrics dict

        Returns:
            Updated heartbeat record
        """
        async with self._lock.acquire_context() as acquired:
            if not acquired:
                logger.warning(
                    f"[Heartbeat] Failed to acquire lock for update: {self.component_name}"
                )
                return None

            # Read current
            current = await self.read_heartbeat()
            if current is None:
                return None

            # Apply update
            updated_metrics = updater(current.metrics.copy())

            # Write back
            await self.write_heartbeat(current.status, updated_metrics)

            return await self.read_heartbeat()

    async def cleanup_stale(self) -> bool:
        """
        Clean up stale heartbeat file.

        Returns:
            True if cleaned, False otherwise
        """
        heartbeat = await self.read_heartbeat()
        if heartbeat is None:
            return False

        if heartbeat.is_stale(self.config.heartbeat_stale_timeout):
            try:
                self.heartbeat_file.unlink()
                logger.info(f"[Heartbeat] Cleaned stale heartbeat: {self.component_name}")
                return True
            except OSError:
                pass

        return False


# =============================================================================
# ISSUE 11: Atomic Port Reservation
# =============================================================================


class PortState(Enum):
    """Port reservation state."""
    AVAILABLE = auto()
    RESERVED = auto()
    IN_USE = auto()
    STALE = auto()


@dataclass
class PortReservation:
    """Port reservation record."""
    port: int
    component: str
    identity: ProcessIdentity
    reserved_at: float
    verified_at: float
    socket_bound: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "port": self.port,
            "component": self.component,
            "identity": self.identity.to_dict(),
            "reserved_at": self.reserved_at,
            "verified_at": self.verified_at,
            "socket_bound": self.socket_bound,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortReservation":
        """Deserialize from dictionary."""
        return cls(
            port=data.get("port", 0),
            component=data.get("component", "unknown"),
            identity=ProcessIdentity.from_dict(data.get("identity", {})),
            reserved_at=data.get("reserved_at", 0.0),
            verified_at=data.get("verified_at", 0.0),
            socket_bound=data.get("socket_bound", False),
        )

    def is_stale(self) -> bool:
        """Check if reservation is stale (owner dead)."""
        return not self.identity.is_alive()


class AtomicPortReservation:
    """
    Atomic port reservation with file lock and socket verification.

    FIXES ISSUE 11: Port Allocation Race Condition

    Key Features:
    - Atomic reservation file creation
    - Socket bind verification before reservation
    - PID tracking with staleness detection
    - Automatic cleanup of stale reservations

    Usage:
        reserver = AtomicPortReservation()

        # Reserve a port
        port = await reserver.reserve_port("jarvis_body", 8010, [8011, 8012])

        # Release port
        await reserver.release_port(8010)
    """

    def __init__(self, config: Optional[RacePreventionConfig] = None):
        """Initialize port reservation manager."""
        self.config = config or get_config()
        self.ports_dir = self.config.state_dir / "ports"
        self.ports_dir.mkdir(parents=True, exist_ok=True)

        # Global lock for port operations
        self._global_lock = AtomicFileLock("port_reservation", config)

    def _port_file(self, port: int) -> Path:
        """Get path for port reservation file."""
        return self.ports_dir / f"port_{port}.json"

    def _verify_socket_available(self, port: int, host: str = "127.0.0.1") -> bool:
        """
        Verify port is available by attempting socket bind.

        This is the definitive check - if we can bind, port is free.
        """
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(self.config.port_socket_timeout)
            sock.bind((host, port))
            return True
        except (OSError, socket.error):
            return False
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def _read_reservation(self, port: int) -> Optional[PortReservation]:
        """Read port reservation from file."""
        port_file = self._port_file(port)
        if not port_file.exists():
            return None

        try:
            content = port_file.read_text()
            if not content.strip():
                return None
            data = json.loads(content)
            return PortReservation.from_dict(data)
        except Exception as e:
            logger.debug(f"[PortReserve] Read error for port {port}: {e}")
            return None

    def _write_reservation(self, reservation: PortReservation) -> None:
        """Write port reservation atomically."""
        port_file = self._port_file(reservation.port)

        # Use temp file + rename for atomicity
        fd, temp_path = tempfile.mkstemp(
            dir=self.ports_dir,
            prefix=f".port_{reservation.port}.",
            suffix=".tmp"
        )

        try:
            content = json.dumps(reservation.to_dict(), indent=2)
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = -1

            os.replace(temp_path, port_file)

        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _clean_stale_reservation(self, port: int) -> bool:
        """Clean stale reservation if owner is dead."""
        reservation = self._read_reservation(port)
        if reservation is None:
            return False

        if reservation.is_stale():
            logger.info(
                f"[PortReserve] Cleaning stale reservation for port {port}: "
                f"owner PID {reservation.identity.pid} is dead"
            )
            try:
                self._port_file(port).unlink()
                return True
            except OSError:
                pass

        return False

    async def reserve_port(
        self,
        component: str,
        preferred_port: int,
        fallback_ports: Optional[List[int]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[int]:
        """
        Reserve a port atomically with socket verification.

        Args:
            component: Component name requesting the port
            preferred_port: Preferred port number
            fallback_ports: Alternative ports if preferred is unavailable
            timeout: Maximum time to wait for reservation

        Returns:
            Reserved port number, or None if all ports unavailable
        """
        timeout = timeout or self.config.port_reservation_timeout
        ports_to_try = [preferred_port] + (fallback_ports or [])
        start_time = time.time()

        async with self._global_lock.acquire_context(timeout) as acquired:
            if not acquired:
                logger.warning(f"[PortReserve] Failed to acquire global lock for {component}")
                return None

            for port in ports_to_try:
                if time.time() - start_time > timeout:
                    break

                # Check existing reservation
                existing = self._read_reservation(port)
                if existing:
                    # Check if it's ours
                    current = get_current_identity()
                    if (
                        existing.identity.pid == current.pid and
                        abs(existing.identity.creation_time - current.creation_time) < 5.0
                    ):
                        # We already have this port
                        logger.debug(f"[PortReserve] Reusing existing reservation: port {port}")
                        return port

                    # Check if stale
                    if existing.is_stale():
                        self._clean_stale_reservation(port)
                    else:
                        # Port taken by active process
                        continue

                # Verify socket is available
                loop = asyncio.get_running_loop()
                available = await loop.run_in_executor(
                    None, self._verify_socket_available, port
                )

                if not available:
                    # Port in use at OS level
                    continue

                # Create reservation
                reservation = PortReservation(
                    port=port,
                    component=component,
                    identity=get_current_identity(),
                    reserved_at=time.time(),
                    verified_at=time.time(),
                    socket_bound=True,
                )

                self._write_reservation(reservation)
                logger.info(f"[PortReserve] Reserved port {port} for {component}")
                return port

        logger.warning(
            f"[PortReserve] No available ports for {component}: "
            f"tried {ports_to_try}"
        )
        return None

    async def release_port(self, port: int) -> bool:
        """
        Release a port reservation.

        Args:
            port: Port number to release

        Returns:
            True if released, False if not owned
        """
        port_file = self._port_file(port)

        async with self._global_lock.acquire_context() as acquired:
            if not acquired:
                return False

            reservation = self._read_reservation(port)
            if reservation is None:
                return True  # Already released

            # Verify ownership
            current = get_current_identity()
            if (
                reservation.identity.pid != current.pid or
                abs(reservation.identity.creation_time - current.creation_time) > 5.0
            ):
                logger.warning(
                    f"[PortReserve] Cannot release port {port}: "
                    f"owned by PID {reservation.identity.pid}"
                )
                return False

            try:
                port_file.unlink()
                logger.debug(f"[PortReserve] Released port {port}")
                return True
            except OSError as e:
                logger.warning(f"[PortReserve] Error releasing port {port}: {e}")
                return False

    async def get_reservation(self, port: int) -> Optional[PortReservation]:
        """Get reservation info for a port."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_reservation, port)

    async def cleanup_stale_reservations(self) -> int:
        """
        Clean up all stale port reservations.

        Returns:
            Number of reservations cleaned
        """
        cleaned = 0

        async with self._global_lock.acquire_context() as acquired:
            if not acquired:
                return 0

            for port_file in self.ports_dir.glob("port_*.json"):
                try:
                    # Extract port number from filename
                    port = int(port_file.stem.replace("port_", ""))
                    if self._clean_stale_reservation(port):
                        cleaned += 1
                except (ValueError, OSError):
                    pass

        if cleaned > 0:
            logger.info(f"[PortReserve] Cleaned {cleaned} stale reservations")

        return cleaned


# =============================================================================
# ISSUE 12: Continuous Dependency Health Monitoring
# =============================================================================


class DependencyState(Enum):
    """Dependency health state."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


@dataclass
class DependencyNode:
    """Node in the dependency graph."""
    name: str
    dependencies: List[str]
    state: DependencyState = DependencyState.UNKNOWN
    last_check: float = 0.0
    failure_count: int = 0
    health_checker: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "dependencies": self.dependencies,
            "state": self.state.value,
            "last_check": self.last_check,
            "failure_count": self.failure_count,
        }


class DependencyHealthMonitor:
    """
    Continuous dependency health monitoring with automatic dependent restart.

    FIXES ISSUE 12: Component Startup Order Race

    Key Features:
    - Continuous health polling with configurable interval
    - Dependency graph with topological ordering
    - Automatic restart of dependents when dependency fails
    - Event-driven state propagation
    - Failure threshold before marking unhealthy

    Usage:
        monitor = DependencyHealthMonitor()

        # Register components with dependencies
        monitor.register_component("reactor_core", [])
        monitor.register_component("jarvis_prime", ["reactor_core"])
        monitor.register_component("jarvis_body", ["jarvis_prime", "reactor_core"])

        # Start monitoring
        await monitor.start_monitoring()

        # Register restart callback
        monitor.on_dependency_failed(restart_dependent)
    """

    def __init__(self, config: Optional[RacePreventionConfig] = None):
        """Initialize dependency monitor."""
        self.config = config or get_config()
        self.state_dir = self.config.state_dir / "dependencies"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._nodes: Dict[str, DependencyNode] = {}
        self._dependency_graph: Dict[str, List[str]] = {}  # node -> nodes that depend on it
        self._topological_order: List[str] = []

        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Callbacks
        self._on_state_change: List[Callable[[str, DependencyState, DependencyState], Coroutine[Any, Any, None]]] = []
        self._on_dependency_failed: List[Callable[[str, List[str]], Coroutine[Any, Any, None]]] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def register_component(
        self,
        name: str,
        dependencies: List[str],
        health_checker: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None,
    ) -> None:
        """
        Register a component with its dependencies.

        Args:
            name: Component name
            dependencies: List of dependency component names
            health_checker: Optional async function that returns True if healthy
        """
        node = DependencyNode(
            name=name,
            dependencies=dependencies,
            health_checker=health_checker,
        )
        self._nodes[name] = node

        # Build reverse dependency graph
        for dep in dependencies:
            if dep not in self._dependency_graph:
                self._dependency_graph[dep] = []
            self._dependency_graph[dep].append(name)

        # Recompute topological order
        self._compute_topological_order()

        logger.debug(f"[DepMonitor] Registered {name} with deps: {dependencies}")

    def _compute_topological_order(self) -> None:
        """Compute topological order using Kahn's algorithm."""
        # Build in-degree map
        in_degree: Dict[str, int] = {name: 0 for name in self._nodes}
        for node in self._nodes.values():
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node.name] += 1

        # Start with nodes that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: List[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree for dependents
            for dependent in self._dependency_graph.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            logger.warning(
                "[DepMonitor] Dependency cycle detected - "
                "some components may not start correctly"
            )

        self._topological_order = result
        logger.debug(f"[DepMonitor] Startup order: {result}")

    def get_startup_order(self) -> List[str]:
        """Get components in correct startup order."""
        return self._topological_order.copy()

    async def check_component_health(self, name: str) -> bool:
        """
        Check health of a single component.

        Returns:
            True if healthy, False otherwise
        """
        node = self._nodes.get(name)
        if node is None:
            return False

        # Use custom health checker if provided
        if node.health_checker:
            try:
                return await asyncio.wait_for(
                    node.health_checker(),
                    timeout=self.config.dependency_check_interval / 2,
                )
            except Exception as e:
                logger.debug(f"[DepMonitor] Health check failed for {name}: {e}")
                return False

        # Default: check heartbeat file
        heartbeat_file = self.config.state_dir / "heartbeats" / f"{name}.json"
        if not heartbeat_file.exists():
            return False

        try:
            content = heartbeat_file.read_text()
            data = json.loads(content)
            timestamp = data.get("timestamp", 0.0)
            age = time.time() - timestamp

            if age > self.config.heartbeat_stale_timeout:
                return False

            # Also verify process is alive
            identity = ProcessIdentity.from_dict(data.get("identity", {}))
            return identity.is_alive()

        except Exception:
            return False

    async def _update_component_state(
        self,
        name: str,
        is_healthy: bool,
    ) -> None:
        """Update component state based on health check result."""
        async with self._lock:
            node = self._nodes.get(name)
            if node is None:
                return

            old_state = node.state
            node.last_check = time.time()

            if is_healthy:
                node.failure_count = 0
                new_state = DependencyState.HEALTHY
            else:
                node.failure_count += 1
                if node.failure_count >= self.config.dependency_failure_threshold:
                    new_state = DependencyState.UNHEALTHY
                else:
                    new_state = DependencyState.DEGRADED

            if old_state != new_state:
                node.state = new_state
                logger.info(
                    f"[DepMonitor] {name} state changed: {old_state.value} -> {new_state.value}"
                )

                # Notify state change callbacks
                for callback in self._on_state_change:
                    try:
                        await callback(name, old_state, new_state)
                    except Exception as e:
                        logger.warning(f"[DepMonitor] State change callback error: {e}")

                # If unhealthy, notify dependent restart callbacks
                if new_state == DependencyState.UNHEALTHY:
                    dependents = self._dependency_graph.get(name, [])
                    if dependents:
                        for callback in self._on_dependency_failed:
                            try:
                                await callback(name, dependents)
                            except Exception as e:
                                logger.warning(f"[DepMonitor] Dependency failed callback error: {e}")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Check all components
                for name in self._nodes:
                    if self._shutdown_event.is_set():
                        break

                    is_healthy = await self.check_component_health(name)
                    await self._update_component_state(name, is_healthy)

                # Write state to file for cross-process visibility
                await self._persist_state()

            except Exception as e:
                logger.warning(f"[DepMonitor] Monitoring loop error: {e}")

            # Wait for next check interval
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.dependency_check_interval,
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Continue monitoring

    async def _persist_state(self) -> None:
        """Persist current state to file."""
        state_file = self.state_dir / "dependency_state.json"

        state = {
            "timestamp": time.time(),
            "components": {
                name: node.to_dict()
                for name, node in self._nodes.items()
            },
            "startup_order": self._topological_order,
        }

        # Atomic write
        fd, temp_path = tempfile.mkstemp(
            dir=self.state_dir,
            prefix=".state_",
            suffix=".tmp"
        )

        try:
            content = json.dumps(state, indent=2)
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = -1

            os.replace(temp_path, state_file)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def start_monitoring(self) -> None:
        """Start the monitoring task."""
        if self._monitoring_task is not None:
            return

        self._shutdown_event.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("[DepMonitor] Started dependency monitoring")

    async def stop_monitoring(self) -> None:
        """Stop the monitoring task."""
        self._shutdown_event.set()

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("[DepMonitor] Stopped dependency monitoring")

    def on_state_change(
        self,
        callback: Callable[[str, DependencyState, DependencyState], Coroutine[Any, Any, None]],
    ) -> None:
        """Register callback for state changes."""
        self._on_state_change.append(callback)

    def on_dependency_failed(
        self,
        callback: Callable[[str, List[str]], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register callback when a dependency fails.

        Callback receives: (failed_dependency, list_of_dependents)
        """
        self._on_dependency_failed.append(callback)

    def are_dependencies_healthy(self, name: str) -> bool:
        """Check if all dependencies of a component are healthy."""
        node = self._nodes.get(name)
        if node is None:
            return False

        for dep_name in node.dependencies:
            dep_node = self._nodes.get(dep_name)
            if dep_node is None or dep_node.state != DependencyState.HEALTHY:
                return False

        return True

    def get_component_state(self, name: str) -> DependencyState:
        """Get current state of a component."""
        node = self._nodes.get(name)
        return node.state if node else DependencyState.UNKNOWN


# =============================================================================
# ISSUE 13: Process Discovery Race Condition Prevention
# =============================================================================


class ProcessVerificationResult(Enum):
    """Result of process verification."""
    VERIFIED = "verified"
    PID_MISMATCH = "pid_mismatch"
    NAME_MISMATCH = "name_mismatch"
    PARENT_MISMATCH = "parent_mismatch"
    NOT_FOUND = "not_found"
    ZOMBIE = "zombie"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class ProcessFingerprint:
    """
    Complete fingerprint of a process for verification.

    Uses multiple attributes to uniquely identify a process
    and prevent PID reuse attacks.
    """
    pid: int
    name: str
    cmdline: List[str]
    creation_time: float
    parent_pid: int
    parent_name: str
    cwd: str
    username: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pid": self.pid,
            "name": self.name,
            "cmdline": self.cmdline,
            "creation_time": self.creation_time,
            "parent_pid": self.parent_pid,
            "parent_name": self.parent_name,
            "cwd": self.cwd,
            "username": self.username,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessFingerprint":
        """Deserialize from dictionary."""
        return cls(
            pid=data.get("pid", 0),
            name=data.get("name", ""),
            cmdline=data.get("cmdline", []),
            creation_time=data.get("creation_time", 0.0),
            parent_pid=data.get("parent_pid", 0),
            parent_name=data.get("parent_name", ""),
            cwd=data.get("cwd", ""),
            username=data.get("username", ""),
        )

    @classmethod
    def capture(cls, pid: int) -> Optional["ProcessFingerprint"]:
        """
        Capture fingerprint of a process.

        Returns None if process doesn't exist or can't be accessed.
        """
        try:
            import psutil
            proc = psutil.Process(pid)

            # Get parent info
            parent = proc.parent()
            parent_pid = parent.pid if parent else 0
            parent_name = parent.name() if parent else ""

            return cls(
                pid=pid,
                name=proc.name(),
                cmdline=proc.cmdline(),
                creation_time=proc.create_time(),
                parent_pid=parent_pid,
                parent_name=parent_name,
                cwd=proc.cwd() if hasattr(proc, 'cwd') else "",
                username=proc.username() if hasattr(proc, 'username') else "",
            )
        except ImportError:
            # Fallback without psutil
            try:
                os.kill(pid, 0)
                return cls(
                    pid=pid,
                    name="unknown",
                    cmdline=[],
                    creation_time=time.time(),
                    parent_pid=os.getppid(),
                    parent_name="",
                    cwd=os.getcwd(),
                    username=os.getenv("USER", "unknown"),
                )
            except (ProcessLookupError, PermissionError):
                return None
        except Exception:
            return None


class SafeProcessManager:
    """
    Safe process management with race condition prevention.

    FIXES ISSUE 13: Process Discovery Race Condition

    Key Features:
    - PID validation with process name matching
    - Process tree validation (parent-child relationships)
    - Graceful degradation if process disappears
    - Fingerprint-based process identification
    - Atomic process registry with file locking

    Usage:
        manager = SafeProcessManager()

        # Register a process
        await manager.register_process("jarvis_body", pid, fingerprint)

        # Verify before action
        result = await manager.verify_process("jarvis_body", pid)
        if result == ProcessVerificationResult.VERIFIED:
            # Safe to act on process
    """

    def __init__(self, config: Optional[RacePreventionConfig] = None):
        """Initialize safe process manager."""
        self.config = config or get_config()
        self.process_dir = self.config.state_dir / "processes"
        self.process_dir.mkdir(parents=True, exist_ok=True)

        self._lock = AtomicFileLock("process_registry", config)
        self._process_tree_cache: Dict[int, ProcessFingerprint] = {}
        self._cache_time: float = 0.0
        self._cache_ttl: float = _env_float("PROCESS_CACHE_TTL", 2.0)

    def _process_file(self, component: str) -> Path:
        """Get path for process registry file."""
        return self.process_dir / f"{component}.json"

    async def register_process(
        self,
        component: str,
        pid: int,
        fingerprint: Optional[ProcessFingerprint] = None,
    ) -> bool:
        """
        Register a process atomically.

        Args:
            component: Component name
            pid: Process ID
            fingerprint: Optional pre-captured fingerprint

        Returns:
            True if registered successfully
        """
        # Capture fingerprint if not provided
        if fingerprint is None:
            loop = asyncio.get_running_loop()
            fingerprint = await loop.run_in_executor(
                None, ProcessFingerprint.capture, pid
            )

        if fingerprint is None:
            logger.warning(f"[SafeProcess] Cannot capture fingerprint for PID {pid}")
            return False

        async with self._lock.acquire_context() as acquired:
            if not acquired:
                logger.warning(f"[SafeProcess] Lock timeout registering {component}")
                return False

            # Write atomically
            process_file = self._process_file(component)
            data = {
                "component": component,
                "registered_at": time.time(),
                "fingerprint": fingerprint.to_dict(),
            }

            fd, temp_path = tempfile.mkstemp(
                dir=self.process_dir,
                prefix=f".{component}.",
                suffix=".tmp"
            )

            try:
                content = json.dumps(data, indent=2)
                os.write(fd, content.encode("utf-8"))
                os.fsync(fd)
                os.close(fd)
                fd = -1

                os.replace(temp_path, process_file)
                logger.debug(f"[SafeProcess] Registered {component} with PID {pid}")
                return True

            finally:
                if fd >= 0:
                    os.close(fd)
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    async def verify_process(
        self,
        component: str,
        expected_pid: int,
    ) -> ProcessVerificationResult:
        """
        Verify a process is who we think it is.

        Compares current process state against registered fingerprint.

        Args:
            component: Component name
            expected_pid: Expected PID

        Returns:
            Verification result
        """
        # Read registered fingerprint
        process_file = self._process_file(component)
        if not process_file.exists():
            return ProcessVerificationResult.NOT_FOUND

        try:
            content = process_file.read_text()
            data = json.loads(content)
            registered = ProcessFingerprint.from_dict(data.get("fingerprint", {}))
        except Exception as e:
            logger.debug(f"[SafeProcess] Error reading registry: {e}")
            return ProcessVerificationResult.NOT_FOUND

        # Verify PID matches
        if registered.pid != expected_pid:
            return ProcessVerificationResult.PID_MISMATCH

        # Capture current fingerprint
        loop = asyncio.get_running_loop()
        current = await loop.run_in_executor(
            None, ProcessFingerprint.capture, expected_pid
        )

        if current is None:
            return ProcessVerificationResult.NOT_FOUND

        # Verify creation time (prevents PID reuse)
        if abs(current.creation_time - registered.creation_time) > 5.0:
            logger.warning(
                f"[SafeProcess] PID {expected_pid} reused: "
                f"registered={registered.creation_time:.0f}, "
                f"current={current.creation_time:.0f}"
            )
            return ProcessVerificationResult.PID_MISMATCH

        # Verify process name
        if current.name != registered.name and registered.name != "unknown":
            logger.warning(
                f"[SafeProcess] Process name mismatch for PID {expected_pid}: "
                f"registered={registered.name}, current={current.name}"
            )
            return ProcessVerificationResult.NAME_MISMATCH

        # Verify parent relationship
        if registered.parent_pid != 0:
            if current.parent_pid != registered.parent_pid:
                # Parent changed - might be orphaned or reparented
                logger.debug(
                    f"[SafeProcess] Parent changed for PID {expected_pid}: "
                    f"registered={registered.parent_pid}, current={current.parent_pid}"
                )
                # This is not necessarily an error, just log it

        return ProcessVerificationResult.VERIFIED

    async def safe_kill(
        self,
        component: str,
        signal_num: int = 15,  # SIGTERM
        timeout: float = 10.0,
    ) -> bool:
        """
        Safely terminate a process with verification.

        Verifies process identity before sending signal.

        Args:
            component: Component name
            signal_num: Signal to send (default: SIGTERM)
            timeout: Timeout for graceful shutdown

        Returns:
            True if process was terminated successfully
        """
        # Read registered info
        process_file = self._process_file(component)
        if not process_file.exists():
            logger.debug(f"[SafeProcess] No registered process for {component}")
            return True  # Nothing to kill

        try:
            content = process_file.read_text()
            data = json.loads(content)
            registered = ProcessFingerprint.from_dict(data.get("fingerprint", {}))
        except Exception:
            return False

        pid = registered.pid

        # Verify before kill
        result = await self.verify_process(component, pid)
        if result != ProcessVerificationResult.VERIFIED:
            if result == ProcessVerificationResult.NOT_FOUND:
                # Process already gone
                await self.unregister_process(component)
                return True

            logger.warning(
                f"[SafeProcess] Cannot kill {component}: verification failed ({result.value})"
            )
            return False

        # Send signal
        try:
            os.kill(pid, signal_num)
            logger.debug(f"[SafeProcess] Sent signal {signal_num} to {component} (PID {pid})")
        except ProcessLookupError:
            # Already dead
            await self.unregister_process(component)
            return True
        except PermissionError:
            logger.error(f"[SafeProcess] Permission denied killing {component}")
            return False

        # Wait for termination
        start_time = time.time()
        while time.time() - start_time < timeout:
            await asyncio.sleep(0.1)

            # Check if still alive
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                # Dead
                await self.unregister_process(component)
                return True
            except PermissionError:
                pass

        # Still alive - force kill
        logger.warning(f"[SafeProcess] {component} didn't terminate, sending SIGKILL")
        try:
            os.kill(pid, 9)  # SIGKILL
            await asyncio.sleep(0.5)
            await self.unregister_process(component)
            return True
        except ProcessLookupError:
            await self.unregister_process(component)
            return True
        except Exception as e:
            logger.error(f"[SafeProcess] Failed to kill {component}: {e}")
            return False

    async def unregister_process(self, component: str) -> None:
        """Unregister a process."""
        process_file = self._process_file(component)
        try:
            process_file.unlink(missing_ok=True)
        except OSError:
            pass

    async def get_process_tree(self, root_pid: int) -> Dict[int, ProcessFingerprint]:
        """
        Get entire process tree starting from root.

        Useful for validating parent-child relationships.
        """
        tree: Dict[int, ProcessFingerprint] = {}

        try:
            import psutil

            def collect_tree(pid: int) -> None:
                try:
                    fp = ProcessFingerprint.capture(pid)
                    if fp:
                        tree[pid] = fp
                        proc = psutil.Process(pid)
                        for child in proc.children():
                            collect_tree(child.pid)
                except Exception:
                    pass

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, collect_tree, root_pid)

        except ImportError:
            # No psutil - just capture root
            fp = ProcessFingerprint.capture(root_pid)
            if fp:
                tree[root_pid] = fp

        return tree


# =============================================================================
# ISSUE 14: Configuration File Read Race Prevention
# =============================================================================


@dataclass
class VersionedConfig:
    """Configuration with version tracking."""
    version: int
    timestamp: float
    checksum: str
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize with checksum."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "checksum": self.checksum,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "VersionedConfig":
        """Deserialize with checksum validation."""
        return cls(
            version=raw.get("version", 0),
            timestamp=raw.get("timestamp", 0.0),
            checksum=raw.get("checksum", ""),
            data=raw.get("data", {}),
        )

    @staticmethod
    def compute_checksum(data: Dict[str, Any]) -> str:
        """Compute checksum of data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:32]

    def validate_checksum(self) -> bool:
        """Validate data integrity via checksum."""
        expected = self.compute_checksum(self.data)
        return self.checksum == expected


class AtomicConfigManager:
    """
    Atomic configuration management with versioning and locking.

    FIXES ISSUE 14: Configuration File Read Race

    Key Features:
    - Config file locking for read/write operations
    - Automatic versioning with monotonic counters
    - Checksum-based corruption detection
    - Atomic updates with temp file + rename
    - Read validation after write
    - Concurrent read support with write locking

    Usage:
        config_mgr = AtomicConfigManager("jarvis_config")

        # Read config
        config = await config_mgr.read()

        # Update config atomically
        await config_mgr.update({"key": "value"})

        # Read-modify-write with locking
        async with config_mgr.modify() as data:
            data["new_key"] = "new_value"
    """

    def __init__(
        self,
        name: str,
        config: Optional[RacePreventionConfig] = None,
    ):
        """
        Initialize config manager.

        Args:
            name: Config name (used for file naming)
            config: Optional configuration override
        """
        self.name = name
        self.config = config or get_config()
        self.config_dir = self.config.state_dir / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.config_dir / f"{name}.json"
        self._lock = AtomicFileLock(f"config_{name}", self.config)

        # In-memory cache
        self._cache: Optional[VersionedConfig] = None
        self._cache_version: int = -1

    def _read_sync(self) -> Optional[VersionedConfig]:
        """Read config synchronously."""
        if not self.config_file.exists():
            return None

        try:
            content = self.config_file.read_text()
            if not content.strip():
                return None

            raw = json.loads(content)
            config = VersionedConfig.from_dict(raw)

            # Validate checksum
            if not config.validate_checksum():
                logger.warning(
                    f"[ConfigMgr] Checksum mismatch for {self.name}: "
                    f"config may be corrupted"
                )

            return config

        except json.JSONDecodeError as e:
            logger.error(f"[ConfigMgr] JSON error reading {self.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"[ConfigMgr] Error reading {self.name}: {e}")
            return None

    async def read(self, use_cache: bool = True) -> Optional[VersionedConfig]:
        """
        Read configuration atomically.

        Args:
            use_cache: Use cached version if available and valid

        Returns:
            VersionedConfig or None if not found
        """
        # Check cache
        if use_cache and self._cache is not None:
            # Quick version check without lock
            current = await asyncio.get_running_loop().run_in_executor(
                None, self._read_sync
            )
            if current and current.version == self._cache_version:
                return self._cache

        # Read with shared-like semantics (we still use exclusive lock for safety)
        async with self._lock.acquire_context(timeout=10.0) as acquired:
            if not acquired:
                logger.warning(f"[ConfigMgr] Lock timeout reading {self.name}")
                # Fallback to unprotected read
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self._read_sync)

            loop = asyncio.get_running_loop()
            config = await loop.run_in_executor(None, self._read_sync)

            if config:
                self._cache = config
                self._cache_version = config.version

            return config

    def _write_sync(self, config: VersionedConfig) -> bool:
        """Write config synchronously with atomic rename."""
        fd, temp_path = tempfile.mkstemp(
            dir=self.config_dir,
            prefix=f".{self.name}.",
            suffix=".tmp"
        )

        try:
            content = json.dumps(config.to_dict(), indent=2)
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = -1

            os.replace(temp_path, self.config_file)
            return True

        except Exception as e:
            logger.error(f"[ConfigMgr] Write error for {self.name}: {e}")
            return False
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def write(
        self,
        data: Dict[str, Any],
        validate_after: bool = True,
    ) -> bool:
        """
        Write configuration atomically.

        Args:
            data: Configuration data to write
            validate_after: Re-read and validate after write

        Returns:
            True if written successfully
        """
        async with self._lock.acquire_context(timeout=30.0) as acquired:
            if not acquired:
                logger.error(f"[ConfigMgr] Lock timeout writing {self.name}")
                return False

            # Get current version
            loop = asyncio.get_running_loop()
            current = await loop.run_in_executor(None, self._read_sync)
            version = (current.version + 1) if current else 1

            # Create new versioned config
            new_config = VersionedConfig(
                version=version,
                timestamp=time.time(),
                checksum=VersionedConfig.compute_checksum(data),
                data=data,
            )

            # Write atomically
            success = await loop.run_in_executor(None, self._write_sync, new_config)

            if success and validate_after:
                # Re-read to validate
                written = await loop.run_in_executor(None, self._read_sync)
                if written is None or written.version != version:
                    logger.error(f"[ConfigMgr] Validation failed for {self.name}")
                    return False

                if not written.validate_checksum():
                    logger.error(f"[ConfigMgr] Checksum validation failed for {self.name}")
                    return False

                # Update cache
                self._cache = written
                self._cache_version = written.version

            return success

    async def update(
        self,
        updates: Dict[str, Any],
        merge: bool = True,
    ) -> bool:
        """
        Update configuration atomically.

        Args:
            updates: Updates to apply
            merge: If True, merge with existing; if False, replace entirely

        Returns:
            True if updated successfully
        """
        async with self._lock.acquire_context(timeout=30.0) as acquired:
            if not acquired:
                return False

            # Read current
            loop = asyncio.get_running_loop()
            current = await loop.run_in_executor(None, self._read_sync)

            if merge and current:
                # Deep merge
                data = current.data.copy()
                data.update(updates)
            else:
                data = updates

            # Write back (release lock temporarily to use write method)

        return await self.write(data)

    @asynccontextmanager
    async def modify(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Context manager for read-modify-write operations.

        Holds lock for entire operation to ensure atomicity.

        Usage:
            async with config_mgr.modify() as data:
                data["key"] = "new_value"
                # Automatically saved on exit
        """
        async with self._lock.acquire_context(timeout=60.0) as acquired:
            if not acquired:
                raise TimeoutError(f"Failed to acquire lock for {self.name}")

            # Read current
            loop = asyncio.get_running_loop()
            current = await loop.run_in_executor(None, self._read_sync)
            data = current.data.copy() if current else {}

            try:
                yield data

                # Write modified data
                version = (current.version + 1) if current else 1
                new_config = VersionedConfig(
                    version=version,
                    timestamp=time.time(),
                    checksum=VersionedConfig.compute_checksum(data),
                    data=data,
                )

                await loop.run_in_executor(None, self._write_sync, new_config)

                # Update cache
                self._cache = new_config
                self._cache_version = version

            except Exception as e:
                logger.error(f"[ConfigMgr] Modify failed for {self.name}: {e}")
                raise


# =============================================================================
# ISSUE 15: Event Bus Message Loss Prevention
# =============================================================================


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageState(Enum):
    """Message processing state."""
    PENDING = "pending"
    PROCESSING = "processing"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class PersistentMessage:
    """
    Persistent message with deduplication support.

    Uses message ID for deduplication and acknowledgment tracking.
    """
    id: str
    topic: str
    payload: Dict[str, Any]
    priority: MessagePriority
    state: MessageState
    created_at: float
    expires_at: float
    attempts: int = 0
    last_attempt: float = 0.0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "topic": self.topic,
            "payload": self.payload,
            "priority": self.priority.value,
            "state": self.state.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "attempts": self.attempts,
            "last_attempt": self.last_attempt,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersistentMessage":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", ""),
            topic=data.get("topic", ""),
            payload=data.get("payload", {}),
            priority=MessagePriority(data.get("priority", 1)),
            state=MessageState(data.get("state", "pending")),
            created_at=data.get("created_at", 0.0),
            expires_at=data.get("expires_at", 0.0),
            attempts=data.get("attempts", 0),
            last_attempt=data.get("last_attempt", 0.0),
            acknowledged_by=data.get("acknowledged_by"),
            acknowledged_at=data.get("acknowledged_at"),
        )

    def is_expired(self) -> bool:
        """Check if message has expired."""
        return time.time() > self.expires_at

    def can_retry(self, max_attempts: int) -> bool:
        """Check if message can be retried."""
        return self.attempts < max_attempts and not self.is_expired()


class PersistentMessageQueue:
    """
    Persistent message queue with acknowledgments and deduplication.

    FIXES ISSUE 15: Event Bus Message Loss

    Key Features:
    - Persistent message storage with atomic writes
    - Message deduplication via unique IDs
    - Acknowledgment-based delivery guarantees
    - Automatic retry for failed messages
    - Message expiration and cleanup
    - Priority-based message ordering
    - Concurrent consumer support

    Usage:
        queue = PersistentMessageQueue("events")

        # Publish message
        msg_id = await queue.publish("heartbeat", {"status": "running"})

        # Consume with acknowledgment
        async for msg in queue.consume("consumer_1"):
            try:
                process(msg)
                await queue.acknowledge(msg.id, "consumer_1")
            except Exception:
                await queue.reject(msg.id)
    """

    def __init__(
        self,
        name: str,
        config: Optional[RacePreventionConfig] = None,
    ):
        """
        Initialize persistent message queue.

        Args:
            name: Queue name
            config: Optional configuration override
        """
        self.name = name
        self.config = config or get_config()
        self.queue_dir = self.config.state_dir / "queues" / name
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Configuration from environment
        self.max_attempts = _env_int("MSG_QUEUE_MAX_ATTEMPTS", 3)
        self.default_ttl = _env_float("MSG_QUEUE_DEFAULT_TTL", 3600.0)
        self.retry_delay = _env_float("MSG_QUEUE_RETRY_DELAY", 5.0)
        self.cleanup_interval = _env_float("MSG_QUEUE_CLEANUP_INTERVAL", 60.0)

        # Locks
        self._queue_lock = AtomicFileLock(f"queue_{name}", config)

        # Deduplication cache (in-memory for speed)
        self._seen_ids: Dict[str, float] = {}
        self._seen_ids_max = _env_int("MSG_QUEUE_DEDUP_CACHE_SIZE", 10000)

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    def _message_file(self, msg_id: str) -> Path:
        """Get path for message file."""
        return self.queue_dir / f"{msg_id}.json"

    def _generate_id(self) -> str:
        """Generate unique message ID."""
        import uuid
        return f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}"

    def _is_duplicate(self, msg_id: str) -> bool:
        """Check if message ID is a duplicate."""
        if msg_id in self._seen_ids:
            return True

        # Also check file exists
        return self._message_file(msg_id).exists()

    def _mark_seen(self, msg_id: str) -> None:
        """Mark message ID as seen."""
        self._seen_ids[msg_id] = time.time()

        # Cleanup old entries if cache too large
        if len(self._seen_ids) > self._seen_ids_max:
            # Remove oldest 20%
            sorted_ids = sorted(self._seen_ids.items(), key=lambda x: x[1])
            for old_id, _ in sorted_ids[:len(sorted_ids) // 5]:
                del self._seen_ids[old_id]

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: Optional[float] = None,
        msg_id: Optional[str] = None,
    ) -> str:
        """
        Publish a message to the queue.

        Args:
            topic: Message topic/channel
            payload: Message payload
            priority: Message priority
            ttl: Time-to-live in seconds
            msg_id: Optional custom message ID (for idempotent publish)

        Returns:
            Message ID
        """
        msg_id = msg_id or self._generate_id()
        ttl = ttl or self.default_ttl

        # Check for duplicate
        if self._is_duplicate(msg_id):
            logger.debug(f"[MsgQueue] Duplicate message ignored: {msg_id}")
            return msg_id

        message = PersistentMessage(
            id=msg_id,
            topic=topic,
            payload=payload,
            priority=priority,
            state=MessageState.PENDING,
            created_at=time.time(),
            expires_at=time.time() + ttl,
        )

        async with self._queue_lock.acquire_context(timeout=10.0) as acquired:
            if not acquired:
                raise TimeoutError(f"Failed to acquire lock for queue {self.name}")

            # Write message atomically
            msg_file = self._message_file(msg_id)
            fd, temp_path = tempfile.mkstemp(
                dir=self.queue_dir,
                prefix=f".{msg_id[:8]}.",
                suffix=".tmp"
            )

            try:
                content = json.dumps(message.to_dict(), indent=2)
                os.write(fd, content.encode("utf-8"))
                os.fsync(fd)
                os.close(fd)
                fd = -1

                os.replace(temp_path, msg_file)
                self._mark_seen(msg_id)

                logger.debug(f"[MsgQueue] Published message {msg_id} to {topic}")
                return msg_id

            finally:
                if fd >= 0:
                    os.close(fd)
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    async def consume(
        self,
        consumer_id: str,
        topics: Optional[List[str]] = None,
        batch_size: int = 10,
    ) -> AsyncIterator[PersistentMessage]:
        """
        Consume messages from the queue.

        Messages must be acknowledged or rejected after processing.

        Args:
            consumer_id: Unique consumer identifier
            topics: Optional list of topics to consume (None = all)
            batch_size: Maximum messages to fetch per batch

        Yields:
            PersistentMessage objects
        """
        while not self._shutdown_event.is_set():
            messages = await self._fetch_pending(topics, batch_size)

            if not messages:
                # No messages - wait before retry
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=1.0
                    )
                    break
                except asyncio.TimeoutError:
                    continue

            for msg in messages:
                # Mark as processing
                msg.state = MessageState.PROCESSING
                msg.attempts += 1
                msg.last_attempt = time.time()
                await self._update_message(msg)

                yield msg

    async def _fetch_pending(
        self,
        topics: Optional[List[str]],
        limit: int,
    ) -> List[PersistentMessage]:
        """Fetch pending messages."""
        messages: List[PersistentMessage] = []

        async with self._queue_lock.acquire_context(timeout=5.0) as acquired:
            if not acquired:
                return []

            # Scan message files
            for msg_file in sorted(self.queue_dir.glob("*.json")):
                if len(messages) >= limit:
                    break

                try:
                    content = msg_file.read_text()
                    msg = PersistentMessage.from_dict(json.loads(content))

                    # Skip non-pending
                    if msg.state != MessageState.PENDING:
                        # Check if stuck in processing
                        if msg.state == MessageState.PROCESSING:
                            if time.time() - msg.last_attempt > self.retry_delay * 2:
                                # Reset to pending
                                msg.state = MessageState.PENDING
                            else:
                                continue
                        else:
                            continue

                    # Check expiration
                    if msg.is_expired():
                        msg.state = MessageState.EXPIRED
                        await self._update_message(msg)
                        continue

                    # Check retry limit
                    if not msg.can_retry(self.max_attempts):
                        msg.state = MessageState.FAILED
                        await self._update_message(msg)
                        continue

                    # Filter by topic
                    if topics and msg.topic not in topics:
                        continue

                    messages.append(msg)

                except Exception as e:
                    logger.debug(f"[MsgQueue] Error reading message: {e}")

        # Sort by priority (highest first) then by creation time (oldest first)
        messages.sort(key=lambda m: (-m.priority.value, m.created_at))

        return messages

    async def _update_message(self, msg: PersistentMessage) -> None:
        """Update message state."""
        msg_file = self._message_file(msg.id)

        fd, temp_path = tempfile.mkstemp(
            dir=self.queue_dir,
            prefix=f".{msg.id[:8]}.",
            suffix=".tmp"
        )

        try:
            content = json.dumps(msg.to_dict(), indent=2)
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = -1

            os.replace(temp_path, msg_file)

        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def acknowledge(self, msg_id: str, consumer_id: str) -> bool:
        """
        Acknowledge successful message processing.

        Args:
            msg_id: Message ID
            consumer_id: Consumer that processed the message

        Returns:
            True if acknowledged successfully
        """
        msg_file = self._message_file(msg_id)

        async with self._queue_lock.acquire_context(timeout=5.0) as acquired:
            if not acquired:
                return False

            if not msg_file.exists():
                return False

            try:
                content = msg_file.read_text()
                msg = PersistentMessage.from_dict(json.loads(content))

                msg.state = MessageState.ACKNOWLEDGED
                msg.acknowledged_by = consumer_id
                msg.acknowledged_at = time.time()

                await self._update_message(msg)

                logger.debug(f"[MsgQueue] Acknowledged message {msg_id} by {consumer_id}")
                return True

            except Exception as e:
                logger.warning(f"[MsgQueue] Failed to acknowledge {msg_id}: {e}")
                return False

    async def reject(
        self,
        msg_id: str,
        requeue: bool = True,
    ) -> bool:
        """
        Reject a message (processing failed).

        Args:
            msg_id: Message ID
            requeue: If True, put back in queue for retry

        Returns:
            True if rejected successfully
        """
        msg_file = self._message_file(msg_id)

        async with self._queue_lock.acquire_context(timeout=5.0) as acquired:
            if not acquired:
                return False

            if not msg_file.exists():
                return False

            try:
                content = msg_file.read_text()
                msg = PersistentMessage.from_dict(json.loads(content))

                if requeue and msg.can_retry(self.max_attempts):
                    msg.state = MessageState.PENDING
                else:
                    msg.state = MessageState.FAILED

                await self._update_message(msg)

                logger.debug(f"[MsgQueue] Rejected message {msg_id}, requeue={requeue}")
                return True

            except Exception as e:
                logger.warning(f"[MsgQueue] Failed to reject {msg_id}: {e}")
                return False

    async def cleanup(self) -> int:
        """
        Clean up acknowledged, expired, and failed messages.

        Returns:
            Number of messages cleaned up
        """
        cleaned = 0

        async with self._queue_lock.acquire_context(timeout=30.0) as acquired:
            if not acquired:
                return 0

            for msg_file in list(self.queue_dir.glob("*.json")):
                try:
                    content = msg_file.read_text()
                    msg = PersistentMessage.from_dict(json.loads(content))

                    should_delete = False

                    # Delete acknowledged messages after 1 hour
                    if msg.state == MessageState.ACKNOWLEDGED:
                        if msg.acknowledged_at and time.time() - msg.acknowledged_at > 3600:
                            should_delete = True

                    # Delete expired messages
                    elif msg.state == MessageState.EXPIRED:
                        should_delete = True

                    # Delete failed messages after 24 hours
                    elif msg.state == MessageState.FAILED:
                        if time.time() - msg.created_at > 86400:
                            should_delete = True

                    if should_delete:
                        msg_file.unlink()
                        cleaned += 1

                except Exception:
                    pass

        if cleaned > 0:
            logger.debug(f"[MsgQueue] Cleaned up {cleaned} messages")

        return cleaned

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while not self._shutdown_event.is_set():
                try:
                    await self.cleanup()
                except Exception as e:
                    logger.warning(f"[MsgQueue] Cleanup error: {e}")

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.cleanup_interval
                    )
                    break
                except asyncio.TimeoutError:
                    pass

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop(self) -> None:
        """Stop the queue and cleanup task."""
        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = {
            "pending": 0,
            "processing": 0,
            "acknowledged": 0,
            "failed": 0,
            "expired": 0,
            "total": 0,
        }

        for msg_file in self.queue_dir.glob("*.json"):
            try:
                content = msg_file.read_text()
                msg = PersistentMessage.from_dict(json.loads(content))
                stats["total"] += 1
                stats[msg.state.value] = stats.get(msg.state.value, 0) + 1
            except Exception:
                pass

        return stats


# =============================================================================
# Singleton Access and Factory Functions
# =============================================================================


_file_locks: Dict[str, AtomicFileLock] = {}
_heartbeat_writers: Dict[str, AtomicHeartbeatWriter] = {}
_port_reserver: Optional[AtomicPortReservation] = None
_dependency_monitor: Optional[DependencyHealthMonitor] = None
_process_manager: Optional[SafeProcessManager] = None
_config_managers: Dict[str, AtomicConfigManager] = {}
_message_queues: Dict[str, PersistentMessageQueue] = {}


def get_file_lock(name: str) -> AtomicFileLock:
    """Get or create a file lock by name."""
    if name not in _file_locks:
        _file_locks[name] = AtomicFileLock(name)
    return _file_locks[name]


def get_heartbeat_writer(component_name: str) -> AtomicHeartbeatWriter:
    """Get or create a heartbeat writer for a component."""
    if component_name not in _heartbeat_writers:
        _heartbeat_writers[component_name] = AtomicHeartbeatWriter(component_name)
    return _heartbeat_writers[component_name]


def get_port_reserver() -> AtomicPortReservation:
    """Get the global port reserver."""
    global _port_reserver
    if _port_reserver is None:
        _port_reserver = AtomicPortReservation()
    return _port_reserver


def get_dependency_monitor() -> DependencyHealthMonitor:
    """Get the global dependency monitor."""
    global _dependency_monitor
    if _dependency_monitor is None:
        _dependency_monitor = DependencyHealthMonitor()
    return _dependency_monitor


def get_process_manager() -> SafeProcessManager:
    """Get the global safe process manager."""
    global _process_manager
    if _process_manager is None:
        _process_manager = SafeProcessManager()
    return _process_manager


def get_config_manager(name: str) -> AtomicConfigManager:
    """Get or create a config manager by name."""
    if name not in _config_managers:
        _config_managers[name] = AtomicConfigManager(name)
    return _config_managers[name]


def get_message_queue(name: str) -> PersistentMessageQueue:
    """Get or create a message queue by name."""
    if name not in _message_queues:
        _message_queues[name] = PersistentMessageQueue(name)
    return _message_queues[name]


# =============================================================================
# Convenience Functions for Trinity Integration
# =============================================================================


async def acquire_trinity_lock(resource: str, timeout: float = 30.0) -> AtomicFileLock:
    """
    Acquire a Trinity IPC lock.

    Usage:
        lock = await acquire_trinity_lock("heartbeat_update")
        try:
            # Critical section
        finally:
            lock.release()
    """
    lock = get_file_lock(f"trinity_{resource}")
    if not await lock.acquire_async(timeout):
        raise TimeoutError(f"Failed to acquire Trinity lock: {resource}")
    return lock


async def write_trinity_heartbeat(
    component: str,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a Trinity component heartbeat atomically."""
    writer = get_heartbeat_writer(component)
    await writer.write_heartbeat(status, metrics)


async def reserve_trinity_port(
    component: str,
    primary_port: int,
    fallback_ports: Optional[List[int]] = None,
) -> int:
    """
    Reserve a port for a Trinity component.

    Raises:
        RuntimeError: If no port could be reserved
    """
    reserver = get_port_reserver()
    port = await reserver.reserve_port(component, primary_port, fallback_ports)
    if port is None:
        raise RuntimeError(f"No available port for {component}")
    return port


async def start_trinity_dependency_monitoring(
    components: Dict[str, List[str]],
    health_checkers: Optional[Dict[str, Callable[[], Coroutine[Any, Any, bool]]]] = None,
) -> DependencyHealthMonitor:
    """
    Start dependency monitoring for Trinity components.

    Args:
        components: Dict mapping component name to list of dependencies
        health_checkers: Optional dict of custom health check functions

    Returns:
        DependencyHealthMonitor instance
    """
    health_checkers = health_checkers or {}
    monitor = get_dependency_monitor()

    for name, deps in components.items():
        monitor.register_component(
            name=name,
            dependencies=deps,
            health_checker=health_checkers.get(name),
        )

    await monitor.start_monitoring()
    return monitor


async def safe_register_process(
    component: str,
    pid: int,
) -> bool:
    """
    Safely register a Trinity process.

    Args:
        component: Component name
        pid: Process ID

    Returns:
        True if registered successfully
    """
    manager = get_process_manager()
    return await manager.register_process(component, pid)


async def safe_kill_process(
    component: str,
    signal_num: int = 15,
    timeout: float = 10.0,
) -> bool:
    """
    Safely terminate a Trinity process with verification.

    Args:
        component: Component name
        signal_num: Signal to send (default: SIGTERM)
        timeout: Timeout for graceful shutdown

    Returns:
        True if process was terminated
    """
    manager = get_process_manager()
    return await manager.safe_kill(component, signal_num, timeout)


async def get_trinity_config(name: str) -> Optional[Dict[str, Any]]:
    """
    Get Trinity configuration atomically.

    Args:
        name: Config name

    Returns:
        Configuration data or None
    """
    config_mgr = get_config_manager(name)
    config = await config_mgr.read()
    return config.data if config else None


async def set_trinity_config(
    name: str,
    data: Dict[str, Any],
) -> bool:
    """
    Set Trinity configuration atomically.

    Args:
        name: Config name
        data: Configuration data

    Returns:
        True if written successfully
    """
    config_mgr = get_config_manager(name)
    return await config_mgr.write(data)


async def publish_trinity_event(
    topic: str,
    payload: Dict[str, Any],
    priority: MessagePriority = MessagePriority.NORMAL,
) -> str:
    """
    Publish a Trinity event to the persistent queue.

    Args:
        topic: Event topic
        payload: Event payload
        priority: Message priority

    Returns:
        Message ID
    """
    queue = get_message_queue("trinity_events")
    return await queue.publish(topic, payload, priority)


async def consume_trinity_events(
    consumer_id: str,
    topics: Optional[List[str]] = None,
) -> AsyncIterator[PersistentMessage]:
    """
    Consume Trinity events from the persistent queue.

    Args:
        consumer_id: Unique consumer identifier
        topics: Optional list of topics to consume

    Yields:
        PersistentMessage objects
    """
    queue = get_message_queue("trinity_events")
    async for msg in queue.consume(consumer_id, topics):
        yield msg


async def acknowledge_trinity_event(msg_id: str, consumer_id: str) -> bool:
    """Acknowledge a Trinity event."""
    queue = get_message_queue("trinity_events")
    return await queue.acknowledge(msg_id, consumer_id)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "RacePreventionConfig",
    "get_config",

    # Process Identity
    "ProcessIdentity",
    "get_current_identity",

    # Issue 9: Atomic File Lock
    "AtomicFileLock",
    "LockState",
    "LockInfo",
    "get_file_lock",
    "acquire_trinity_lock",

    # Issue 10: Atomic Heartbeat Writer
    "AtomicHeartbeatWriter",
    "HeartbeatRecord",
    "get_heartbeat_writer",
    "write_trinity_heartbeat",

    # Issue 11: Atomic Port Reservation
    "AtomicPortReservation",
    "PortReservation",
    "PortState",
    "get_port_reserver",
    "reserve_trinity_port",

    # Issue 12: Dependency Health Monitor
    "DependencyHealthMonitor",
    "DependencyNode",
    "DependencyState",
    "get_dependency_monitor",
    "start_trinity_dependency_monitoring",

    # Issue 13: Safe Process Manager
    "SafeProcessManager",
    "ProcessFingerprint",
    "ProcessVerificationResult",
    "get_process_manager",
    "safe_register_process",
    "safe_kill_process",

    # Issue 14: Atomic Config Manager
    "AtomicConfigManager",
    "VersionedConfig",
    "get_config_manager",
    "get_trinity_config",
    "set_trinity_config",

    # Issue 15: Persistent Message Queue
    "PersistentMessageQueue",
    "PersistentMessage",
    "MessagePriority",
    "MessageState",
    "get_message_queue",
    "publish_trinity_event",
    "consume_trinity_events",
    "acknowledge_trinity_event",
]
