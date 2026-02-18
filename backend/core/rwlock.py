"""
Read-Write Lock v2.0 -- Concurrent readers, exclusive writers.
=============================================================

Production-grade RWLock for the JARVIS ecosystem supporting both async
and synchronous contexts. Designed for read-heavy shared state (config,
health status, metrics) where writes are infrequent.

COMPONENTS:
    1. AsyncRWLock   - asyncio-based RWLock with Condition variable pattern
    2. SyncRWLock    - threading-based RWLock for synchronous code
    3. RWLockStats   - Observable statistics dataclass
    4. read_locked / write_locked - Standalone context manager helpers
    5. create_async_rwlock / create_sync_rwlock - Factory functions

FEATURES:
    - Writer-preference (configurable): pending writers block new readers
    - Timeout support: TimeoutError on acquire expiry
    - Reentrant-safe: detects and warns on re-entrant reads from same task/thread
    - FIFO ordering for writers via internal queue
    - Contention logging: warns if write lock waits > threshold
    - Full statistics for monitoring dashboards
    - Zero hardcoded values: all thresholds from env vars or constructor args

DESIGN PRINCIPLES:
    - Zero hardcoding: All thresholds configurable via env or constructor
    - Graceful degradation: Never crash on edge cases
    - Observable: Full metrics and logging
    - Non-blocking: All operations have optional timeouts
    - Backward compatible: original RWLock class preserved as alias

Usage:
    from backend.core.rwlock import AsyncRWLock, SyncRWLock, create_async_rwlock

    # --- Async ---
    lock = create_async_rwlock(name="config")

    async with lock.read_lock():
        value = shared_state["key"]

    async with lock.write_lock():
        shared_state["key"] = new_value

    # --- Sync ---
    slock = create_sync_rwlock(name="metrics")

    with slock.read_lock():
        value = shared_state["key"]

    with slock.write_lock():
        shared_state["key"] = new_value

Author: JARVIS System
Version: 2.0.0 (February 2026)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import traceback
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Deque,
    Dict,
    Iterator,
    Optional,
    Set,
)

logger = logging.getLogger("jarvis.rwlock")

__all__ = [
    "AsyncRWLock",
    "SyncRWLock",
    "RWLockStats",
    "RWLock",
    "read_locked",
    "write_locked",
    "sync_read_locked",
    "sync_write_locked",
    "create_async_rwlock",
    "create_sync_rwlock",
]


# =============================================================================
# CONFIGURATION FROM ENVIRONMENT
# =============================================================================

def _env_float(key: str, default: float) -> float:
    """Get float from environment with default."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Get bool from environment with default."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


# =============================================================================
# STATISTICS
# =============================================================================

@dataclass
class RWLockStats:
    """Observable statistics for an RWLock instance.

    Attributes:
        name: Human-readable lock name for dashboards.
        total_read_acquisitions: Cumulative successful read lock acquisitions.
        total_write_acquisitions: Cumulative successful write lock acquisitions.
        total_read_wait_time_ms: Cumulative wall-clock time spent waiting for read locks.
        total_write_wait_time_ms: Cumulative wall-clock time spent waiting for write locks.
        total_read_timeouts: Number of read acquisitions that timed out.
        total_write_timeouts: Number of write acquisitions that timed out.
        peak_concurrent_readers: High-water mark of simultaneous readers.
        current_readers: Number of readers currently holding the lock.
        current_writer: Whether a writer currently holds the lock.
        pending_writers: Number of writers waiting to acquire.
        reentrant_warnings: Number of reentrant read attempts detected.
    """
    name: str = ""
    total_read_acquisitions: int = 0
    total_write_acquisitions: int = 0
    total_read_wait_time_ms: float = 0.0
    total_write_wait_time_ms: float = 0.0
    total_read_timeouts: int = 0
    total_write_timeouts: int = 0
    peak_concurrent_readers: int = 0
    current_readers: int = 0
    current_writer: bool = False
    pending_writers: int = 0
    reentrant_warnings: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON dashboards / health endpoints."""
        return {
            "name": self.name,
            "total_read_acquisitions": self.total_read_acquisitions,
            "total_write_acquisitions": self.total_write_acquisitions,
            "total_read_wait_time_ms": round(self.total_read_wait_time_ms, 3),
            "total_write_wait_time_ms": round(self.total_write_wait_time_ms, 3),
            "total_read_timeouts": self.total_read_timeouts,
            "total_write_timeouts": self.total_write_timeouts,
            "peak_concurrent_readers": self.peak_concurrent_readers,
            "current_readers": self.current_readers,
            "current_writer": self.current_writer,
            "pending_writers": self.pending_writers,
            "reentrant_warnings": self.reentrant_warnings,
        }


# =============================================================================
# ASYNC READ-WRITE LOCK
# =============================================================================

class AsyncRWLock:
    """Async read-write lock with configurable writer preference.

    Uses ``asyncio.Condition`` internally.  Writer preference means that when
    one or more writers are waiting, new ``acquire_read()`` calls will block
    until all pending writers have been served. This prevents writer starvation
    in read-heavy workloads.

    Writers are served in FIFO order via an internal event queue.

    Parameters:
        name: Human-readable name for logging and stats.
        writer_priority: When True (default), pending writers block new readers.
        contention_warn_threshold_s: Log a warning with stack trace if a write
            acquire waits longer than this many seconds.  Sourced from
            ``JARVIS_RWLOCK_CONTENTION_WARN_S`` env var when not provided.
    """

    def __init__(
        self,
        name: str = "",
        writer_priority: bool = True,
        contention_warn_threshold_s: Optional[float] = None,
    ) -> None:
        self._name = name or f"AsyncRWLock-{id(self):x}"
        self._writer_priority = writer_priority
        self._contention_warn_s = (
            contention_warn_threshold_s
            if contention_warn_threshold_s is not None
            else _env_float("JARVIS_RWLOCK_CONTENTION_WARN_S", 1.0)
        )

        # Internal state -- protected by _cond
        self._readers: int = 0
        self._writer_active: bool = False

        # FIFO writer queue: each waiting writer gets its own Event
        self._writer_queue: Deque[asyncio.Event] = deque()

        # Track which asyncio tasks currently hold a read lock (for re-entrancy detection)
        self._reader_tasks: Set[int] = set()

        # Condition variable for reader/writer coordination
        self._cond: Optional[asyncio.Condition] = None
        self._cond_lock = threading.Lock()  # guards lazy creation

        # Stats
        self._stats = RWLockStats(name=self._name)

    # -- Lazy condition creation (safe at module-level instantiation) ---------

    def _get_cond(self) -> asyncio.Condition:
        """Lazily create the asyncio.Condition, safe even if no event loop at init."""
        if self._cond is None:
            with self._cond_lock:
                if self._cond is None:
                    self._cond = asyncio.Condition()
        return self._cond

    # -- Core API: acquire / release -----------------------------------------

    async def acquire_read(self, timeout: Optional[float] = None) -> None:
        """Acquire a shared (read) lock.

        Multiple readers can hold the lock simultaneously.  If ``writer_priority``
        is enabled, this call will block while any writers are waiting.

        Args:
            timeout: Maximum seconds to wait. ``None`` means wait forever.
                Raises ``asyncio.TimeoutError`` on expiry.

        Raises:
            asyncio.TimeoutError: If *timeout* expires before the lock is acquired.
        """
        # Reentrant detection
        task_id = id(asyncio.current_task()) if asyncio.current_task() else 0
        if task_id and task_id in self._reader_tasks:
            self._stats.reentrant_warnings += 1
            logger.warning(
                "[%s] Reentrant read lock detected from task %s. "
                "This may indicate a logic error.\n%s",
                self._name, task_id, "".join(traceback.format_stack()),
            )
            # Allow it to proceed (no deadlock since we use Condition, not Lock)
            # but warn loudly.

        cond = self._get_cond()
        t0 = time.monotonic()
        deadline = (t0 + timeout) if timeout is not None else None

        async with cond:
            while True:
                can_read = (
                    not self._writer_active
                    and (not self._writer_priority or len(self._writer_queue) == 0)
                )
                if can_read:
                    break

                # Compute remaining time for this wait cycle
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._stats.total_read_timeouts += 1
                        raise asyncio.TimeoutError(
                            f"[{self._name}] Read lock acquire timed out after {timeout}s"
                        )

                try:
                    await asyncio.wait_for(cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    self._stats.total_read_timeouts += 1
                    raise asyncio.TimeoutError(
                        f"[{self._name}] Read lock acquire timed out after {timeout}s"
                    ) from None

            self._readers += 1
            if task_id:
                self._reader_tasks.add(task_id)

        # Stats
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._stats.total_read_acquisitions += 1
        self._stats.total_read_wait_time_ms += elapsed_ms
        self._stats.current_readers = self._readers
        if self._readers > self._stats.peak_concurrent_readers:
            self._stats.peak_concurrent_readers = self._readers

    async def release_read(self) -> None:
        """Release a shared (read) lock."""
        cond = self._get_cond()
        async with cond:
            if self._readers <= 0:
                logger.error("[%s] release_read called with no active readers!", self._name)
                return
            self._readers -= 1
            self._stats.current_readers = self._readers

            task_id = id(asyncio.current_task()) if asyncio.current_task() else 0
            self._reader_tasks.discard(task_id)

            if self._readers == 0:
                # Wake all waiters -- writers check their own condition
                cond.notify_all()

    async def acquire_write(self, timeout: Optional[float] = None) -> None:
        """Acquire an exclusive (write) lock.

        Blocks until all readers have released and no other writer is active.
        Writers are served in FIFO order.

        Args:
            timeout: Maximum seconds to wait. ``None`` means wait forever.
                Raises ``asyncio.TimeoutError`` on expiry.

        Raises:
            asyncio.TimeoutError: If *timeout* expires before the lock is acquired.
        """
        cond = self._get_cond()
        t0 = time.monotonic()
        deadline = (t0 + timeout) if timeout is not None else None

        # Create a per-writer event for FIFO ordering
        my_turn = asyncio.Event()

        async with cond:
            self._writer_queue.append(my_turn)
            self._stats.pending_writers = len(self._writer_queue)

            try:
                while True:
                    # I can write when: I'm first in queue, no readers, no active writer
                    can_write = (
                        len(self._writer_queue) > 0
                        and self._writer_queue[0] is my_turn
                        and self._readers == 0
                        and not self._writer_active
                    )
                    if can_write:
                        break

                    remaining = None
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self._stats.total_write_timeouts += 1
                            raise asyncio.TimeoutError(
                                f"[{self._name}] Write lock acquire timed out after {timeout}s"
                            )

                    try:
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        self._stats.total_write_timeouts += 1
                        raise asyncio.TimeoutError(
                            f"[{self._name}] Write lock acquire timed out after {timeout}s"
                        ) from None

                # Got the lock
                self._writer_queue.popleft()
                self._writer_active = True
                self._stats.pending_writers = len(self._writer_queue)

            except BaseException:
                # Clean up our entry on failure (timeout, cancellation, etc.)
                try:
                    self._writer_queue.remove(my_turn)
                except ValueError:
                    pass  # already popped
                self._stats.pending_writers = len(self._writer_queue)
                # Wake others so they can re-evaluate
                cond.notify_all()
                raise

        # Stats and contention warning
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._stats.total_write_acquisitions += 1
        self._stats.total_write_wait_time_ms += elapsed_ms
        self._stats.current_writer = True

        if elapsed_ms > self._contention_warn_s * 1000.0:
            logger.warning(
                "[%s] Write lock acquisition took %.1fms (threshold: %.0fms). "
                "High contention detected.\n%s",
                self._name,
                elapsed_ms,
                self._contention_warn_s * 1000.0,
                "".join(traceback.format_stack()),
            )

    async def release_write(self) -> None:
        """Release an exclusive (write) lock."""
        cond = self._get_cond()
        async with cond:
            if not self._writer_active:
                logger.error("[%s] release_write called with no active writer!", self._name)
                return
            self._writer_active = False
            self._stats.current_writer = False
            # Wake all so readers and next queued writer can proceed
            cond.notify_all()

    # -- Context manager helpers ----------------------------------------------

    @asynccontextmanager
    async def read_lock(self, timeout: Optional[float] = None) -> AsyncIterator[None]:
        """Async context manager for read (shared) access.

        Usage::

            async with lock.read_lock():
                value = shared_state["key"]
        """
        await self.acquire_read(timeout=timeout)
        try:
            yield
        finally:
            await self.release_read()

    @asynccontextmanager
    async def write_lock(self, timeout: Optional[float] = None) -> AsyncIterator[None]:
        """Async context manager for write (exclusive) access.

        Usage::

            async with lock.write_lock():
                shared_state["key"] = new_value
        """
        await self.acquire_write(timeout=timeout)
        try:
            yield
        finally:
            await self.release_write()

    # -- Backward-compatible aliases (original API) ---------------------------

    @asynccontextmanager
    async def read(self, timeout: Optional[float] = None) -> AsyncIterator[None]:
        """Alias for :meth:`read_lock` (backward compatibility)."""
        async with self.read_lock(timeout=timeout):
            yield

    @asynccontextmanager
    async def write(self, timeout: Optional[float] = None) -> AsyncIterator[None]:
        """Alias for :meth:`write_lock` (backward compatibility)."""
        async with self.write_lock(timeout=timeout):
            yield

    # -- Stats / properties ---------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics as a dict."""
        return self._stats.to_dict()

    @property
    def reader_count(self) -> int:
        """Number of active readers."""
        return self._readers

    @property
    def writer_active(self) -> bool:
        """Whether a writer currently holds the lock."""
        return self._writer_active

    @property
    def pending_writers(self) -> int:
        """Number of writers waiting in the queue."""
        return len(self._writer_queue)

    # Backward-compatible aliases
    @property
    def readers(self) -> int:
        """Number of active readers (alias for reader_count)."""
        return self._readers

    @property
    def writers_waiting(self) -> int:
        """Number of writers waiting (alias for pending_writers)."""
        return len(self._writer_queue)

    def __repr__(self) -> str:
        return (
            f"<AsyncRWLock '{self._name}' readers={self._readers} "
            f"writer_active={self._writer_active} "
            f"pending_writers={len(self._writer_queue)}>"
        )


# =============================================================================
# SYNC READ-WRITE LOCK
# =============================================================================

class SyncRWLock:
    """Threading-based read-write lock with configurable writer preference.

    Uses ``threading.Condition`` internally.  Same semantics as :class:`AsyncRWLock`
    but for synchronous (threaded) contexts.

    Parameters:
        name: Human-readable name for logging and stats.
        writer_priority: When True (default), pending writers block new readers.
        contention_warn_threshold_s: Log a warning if write acquire exceeds this.
    """

    def __init__(
        self,
        name: str = "",
        writer_priority: bool = True,
        contention_warn_threshold_s: Optional[float] = None,
    ) -> None:
        self._name = name or f"SyncRWLock-{id(self):x}"
        self._writer_priority = writer_priority
        self._contention_warn_s = (
            contention_warn_threshold_s
            if contention_warn_threshold_s is not None
            else _env_float("JARVIS_RWLOCK_CONTENTION_WARN_S", 1.0)
        )

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Internal state
        self._readers: int = 0
        self._writer_active: bool = False
        self._writer_queue: Deque[threading.Event] = deque()
        self._reader_threads: Set[int] = set()

        # Stats (protected by _lock since all access is under _cond)
        self._stats = RWLockStats(name=self._name)

    def acquire_read(self, timeout: Optional[float] = None) -> None:
        """Acquire a shared (read) lock.

        Args:
            timeout: Maximum seconds to wait. Raises ``TimeoutError`` on expiry.

        Raises:
            TimeoutError: If *timeout* expires before the lock is acquired.
        """
        tid = threading.current_thread().ident or 0
        if tid and tid in self._reader_threads:
            self._stats.reentrant_warnings += 1
            logger.warning(
                "[%s] Reentrant read lock from thread %s.\n%s",
                self._name, tid, "".join(traceback.format_stack()),
            )

        t0 = time.monotonic()
        deadline = (t0 + timeout) if timeout is not None else None

        with self._cond:
            while True:
                can_read = (
                    not self._writer_active
                    and (not self._writer_priority or len(self._writer_queue) == 0)
                )
                if can_read:
                    break

                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._stats.total_read_timeouts += 1
                        raise TimeoutError(
                            f"[{self._name}] Read lock acquire timed out after {timeout}s"
                        )

                if not self._cond.wait(timeout=remaining):
                    # wait() returned False => timeout expired
                    # Re-check condition in case of spurious wakeup timing
                    can_read = (
                        not self._writer_active
                        and (not self._writer_priority or len(self._writer_queue) == 0)
                    )
                    if not can_read:
                        self._stats.total_read_timeouts += 1
                        raise TimeoutError(
                            f"[{self._name}] Read lock acquire timed out after {timeout}s"
                        )
                    break  # spurious timing, but condition met

            self._readers += 1
            if tid:
                self._reader_threads.add(tid)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._stats.total_read_acquisitions += 1
        self._stats.total_read_wait_time_ms += elapsed_ms
        self._stats.current_readers = self._readers
        if self._readers > self._stats.peak_concurrent_readers:
            self._stats.peak_concurrent_readers = self._readers

    def release_read(self) -> None:
        """Release a shared (read) lock."""
        with self._cond:
            if self._readers <= 0:
                logger.error("[%s] release_read called with no active readers!", self._name)
                return
            self._readers -= 1
            self._stats.current_readers = self._readers

            tid = threading.current_thread().ident or 0
            self._reader_threads.discard(tid)

            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self, timeout: Optional[float] = None) -> None:
        """Acquire an exclusive (write) lock.

        Args:
            timeout: Maximum seconds to wait. Raises ``TimeoutError`` on expiry.

        Raises:
            TimeoutError: If *timeout* expires before the lock is acquired.
        """
        t0 = time.monotonic()
        deadline = (t0 + timeout) if timeout is not None else None

        my_turn = threading.Event()

        with self._cond:
            self._writer_queue.append(my_turn)
            self._stats.pending_writers = len(self._writer_queue)

            try:
                while True:
                    can_write = (
                        len(self._writer_queue) > 0
                        and self._writer_queue[0] is my_turn
                        and self._readers == 0
                        and not self._writer_active
                    )
                    if can_write:
                        break

                    remaining = None
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self._stats.total_write_timeouts += 1
                            raise TimeoutError(
                                f"[{self._name}] Write lock acquire timed out after {timeout}s"
                            )

                    if not self._cond.wait(timeout=remaining):
                        can_write = (
                            len(self._writer_queue) > 0
                            and self._writer_queue[0] is my_turn
                            and self._readers == 0
                            and not self._writer_active
                        )
                        if not can_write:
                            self._stats.total_write_timeouts += 1
                            raise TimeoutError(
                                f"[{self._name}] Write lock acquire timed out after {timeout}s"
                            )
                        break

                self._writer_queue.popleft()
                self._writer_active = True
                self._stats.pending_writers = len(self._writer_queue)

            except BaseException:
                try:
                    self._writer_queue.remove(my_turn)
                except ValueError:
                    pass
                self._stats.pending_writers = len(self._writer_queue)
                self._cond.notify_all()
                raise

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._stats.total_write_acquisitions += 1
        self._stats.total_write_wait_time_ms += elapsed_ms
        self._stats.current_writer = True

        if elapsed_ms > self._contention_warn_s * 1000.0:
            logger.warning(
                "[%s] Write lock acquisition took %.1fms (threshold: %.0fms). "
                "High contention detected.\n%s",
                self._name,
                elapsed_ms,
                self._contention_warn_s * 1000.0,
                "".join(traceback.format_stack()),
            )

    def release_write(self) -> None:
        """Release an exclusive (write) lock."""
        with self._cond:
            if not self._writer_active:
                logger.error("[%s] release_write called with no active writer!", self._name)
                return
            self._writer_active = False
            self._stats.current_writer = False
            self._cond.notify_all()

    # -- Context managers -----------------------------------------------------

    @contextmanager
    def read_lock(self, timeout: Optional[float] = None) -> Iterator[None]:
        """Sync context manager for read (shared) access.

        Usage::

            with lock.read_lock():
                value = shared_state["key"]
        """
        self.acquire_read(timeout=timeout)
        try:
            yield
        finally:
            self.release_read()

    @contextmanager
    def write_lock(self, timeout: Optional[float] = None) -> Iterator[None]:
        """Sync context manager for write (exclusive) access.

        Usage::

            with lock.write_lock():
                shared_state["key"] = new_value
        """
        self.acquire_write(timeout=timeout)
        try:
            yield
        finally:
            self.release_write()

    # -- Stats / properties ---------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics as a dict."""
        return self._stats.to_dict()

    @property
    def reader_count(self) -> int:
        return self._readers

    @property
    def writer_active(self) -> bool:
        return self._writer_active

    @property
    def pending_writers(self) -> int:
        return len(self._writer_queue)

    def __repr__(self) -> str:
        return (
            f"<SyncRWLock '{self._name}' readers={self._readers} "
            f"writer_active={self._writer_active} "
            f"pending_writers={len(self._writer_queue)}>"
        )


# =============================================================================
# STANDALONE CONTEXT MANAGER HELPERS
# =============================================================================

@asynccontextmanager
async def read_locked(
    lock: AsyncRWLock,
    timeout: Optional[float] = None,
) -> AsyncIterator[None]:
    """Standalone async context manager for read access.

    Usage::

        async with read_locked(my_lock):
            data = shared_state["key"]
    """
    await lock.acquire_read(timeout=timeout)
    try:
        yield
    finally:
        await lock.release_read()


@asynccontextmanager
async def write_locked(
    lock: AsyncRWLock,
    timeout: Optional[float] = None,
) -> AsyncIterator[None]:
    """Standalone async context manager for write access.

    Usage::

        async with write_locked(my_lock):
            shared_state["key"] = value
    """
    await lock.acquire_write(timeout=timeout)
    try:
        yield
    finally:
        await lock.release_write()


@contextmanager
def sync_read_locked(
    lock: SyncRWLock,
    timeout: Optional[float] = None,
) -> Iterator[None]:
    """Standalone sync context manager for read access."""
    lock.acquire_read(timeout=timeout)
    try:
        yield
    finally:
        lock.release_read()


@contextmanager
def sync_write_locked(
    lock: SyncRWLock,
    timeout: Optional[float] = None,
) -> Iterator[None]:
    """Standalone sync context manager for write access."""
    lock.acquire_write(timeout=timeout)
    try:
        yield
    finally:
        lock.release_write()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_async_rwlock(
    name: str = "",
    writer_priority: bool = True,
    contention_warn_threshold_s: Optional[float] = None,
) -> AsyncRWLock:
    """Create a new async RWLock with sensible defaults.

    Args:
        name: Human-readable name for logging/stats.
        writer_priority: Pending writers block new readers (default True).
        contention_warn_threshold_s: Seconds before logging a contention warning.

    Returns:
        A configured :class:`AsyncRWLock` instance.
    """
    wp = writer_priority if not _env_bool("JARVIS_RWLOCK_DISABLE_WRITER_PRIORITY", False) else False
    return AsyncRWLock(
        name=name,
        writer_priority=wp,
        contention_warn_threshold_s=contention_warn_threshold_s,
    )


def create_sync_rwlock(
    name: str = "",
    writer_priority: bool = True,
    contention_warn_threshold_s: Optional[float] = None,
) -> SyncRWLock:
    """Create a new sync RWLock with sensible defaults.

    Args:
        name: Human-readable name for logging/stats.
        writer_priority: Pending writers block new readers (default True).
        contention_warn_threshold_s: Seconds before logging a contention warning.

    Returns:
        A configured :class:`SyncRWLock` instance.
    """
    wp = writer_priority if not _env_bool("JARVIS_RWLOCK_DISABLE_WRITER_PRIORITY", False) else False
    return SyncRWLock(
        name=name,
        writer_priority=wp,
        contention_warn_threshold_s=contention_warn_threshold_s,
    )


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# The original v1.0 class was named ``RWLock``.  Preserve it as an alias.
RWLock = AsyncRWLock
