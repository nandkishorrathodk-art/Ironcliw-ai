"""
Advanced Async Primitives v1.0
==============================

Production-grade async patterns for the Trinity Ecosystem.
These primitives prevent deadlocks, manage backpressure, isolate failures,
and provide intelligent resource management across JARVIS, J-Prime, and Reactor-Core.

COMPONENTS:
    1. TimeoutProtectedLock      - Deadlock-proof async lock with timeout and tracking
    2. AdaptiveBackpressure      - Memory-aware request throttling
    3. ResourceBulkhead          - Failure isolation between components
    4. TrinityRateLimiter        - Token bucket rate limiter for message floods
    5. AdaptiveResourceManager   - Dynamic concurrency based on system capacity
    6. DeepHealthVerifier        - Functional verification beyond HTTP 200
    7. PriorityMessageQueue      - Priority-based message ordering
    8. AtomicFileIPC             - Race-condition-free file-based IPC

DESIGN PRINCIPLES:
    - Zero hardcoding: All thresholds from environment variables
    - Graceful degradation: Never crash, always fall back
    - Observable: Full metrics and logging
    - Non-blocking: All operations have timeouts
    - Self-healing: Automatic recovery from failures

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import fcntl
import gc
import hashlib
import json
import logging
import os
import platform
import struct
import sys
import tempfile
import time
import uuid
import weakref
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# CONFIGURATION FROM ENVIRONMENT
# =============================================================================


def _env_float(key: str, default: float) -> float:
    """Get float from environment with default."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """Get int from environment with default."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Get bool from environment with default."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


# =============================================================================
# 1. TIMEOUT PROTECTED LOCK
# =============================================================================


@dataclass
class LockAcquisitionMetrics:
    """Metrics for lock acquisition."""
    total_acquisitions: int = 0
    total_timeouts: int = 0
    total_wait_time_ms: float = 0.0
    max_hold_time_ms: float = 0.0
    current_holder: Optional[str] = None
    acquire_time: Optional[float] = None

    def record_acquisition(self, wait_time_ms: float) -> None:
        self.total_acquisitions += 1
        self.total_wait_time_ms += wait_time_ms

    def record_timeout(self) -> None:
        self.total_timeouts += 1

    def record_release(self, hold_time_ms: float) -> None:
        if hold_time_ms > self.max_hold_time_ms:
            self.max_hold_time_ms = hold_time_ms


class TimeoutProtectedLock:
    """
    Async lock with timeout protection to prevent infinite deadlocks.

    Features:
        - Configurable acquisition timeout
        - Caller tracking for debugging
        - Hold time monitoring
        - Metrics collection
        - Automatic stale lock detection

    Environment Variables:
        ASYNC_LOCK_TIMEOUT: Default acquisition timeout (seconds)
        ASYNC_LOCK_WARN_HOLD_TIME: Warn if lock held longer than this (seconds)
        ASYNC_LOCK_STALE_THRESHOLD: Consider lock stale after this time (seconds)

    Example:
        lock = TimeoutProtectedLock(timeout=10.0)
        async with lock.acquire_context("my_operation"):
            # Critical section
            pass
    """

    __slots__ = (
        "_lock", "_timeout", "_warn_hold_time", "_stale_threshold",
        "_held_by", "_acquire_time", "_acquire_stack", "_metrics",
        "_lock_id", "_contention_waiters"
    )

    def __init__(
        self,
        timeout: Optional[float] = None,
        warn_hold_time: Optional[float] = None,
        lock_id: Optional[str] = None,
    ):
        self._lock = asyncio.Lock()
        self._timeout = timeout or _env_float("ASYNC_LOCK_TIMEOUT", 30.0)
        self._warn_hold_time = warn_hold_time or _env_float("ASYNC_LOCK_WARN_HOLD_TIME", 10.0)
        self._stale_threshold = _env_float("ASYNC_LOCK_STALE_THRESHOLD", 60.0)
        self._held_by: Optional[str] = None
        self._acquire_time: Optional[float] = None
        self._acquire_stack: Optional[str] = None
        self._metrics = LockAcquisitionMetrics()
        self._lock_id = lock_id or f"lock_{id(self):x}"
        self._contention_waiters: int = 0

    @property
    def is_locked(self) -> bool:
        return self._lock.locked()

    @property
    def is_stale(self) -> bool:
        """Check if lock appears to be stale (held too long)."""
        if self._acquire_time is None:
            return False
        return (time.time() - self._acquire_time) > self._stale_threshold

    @property
    def metrics(self) -> LockAcquisitionMetrics:
        return self._metrics

    @property
    def contention_level(self) -> int:
        """Number of coroutines waiting for this lock."""
        return self._contention_waiters

    async def acquire(
        self,
        caller: str = "unknown",
        timeout: Optional[float] = None,
        capture_stack: bool = False,
    ) -> bool:
        """
        Acquire lock with timeout and caller tracking.

        Args:
            caller: Identifier for the caller (for debugging)
            timeout: Override default timeout for this acquisition
            capture_stack: Whether to capture stack trace (expensive)

        Returns:
            True if lock acquired, False if timeout

        Raises:
            TimeoutError: If acquisition times out (when timeout > 0)
        """
        effective_timeout = timeout if timeout is not None else self._timeout
        start_time = time.time()

        # Track contention
        self._contention_waiters += 1

        try:
            try:
                await asyncio.wait_for(
                    self._lock.acquire(),
                    timeout=effective_timeout if effective_timeout > 0 else None
                )
            except asyncio.TimeoutError:
                self._metrics.record_timeout()
                wait_time = (time.time() - start_time) * 1000
                logger.error(
                    f"[{self._lock_id}] Lock timeout after {effective_timeout}s "
                    f"(caller={caller}, held_by={self._held_by}, "
                    f"hold_time={(time.time() - self._acquire_time):.1f}s)"
                )
                raise TimeoutError(
                    f"Lock acquisition timeout for {caller} after {effective_timeout}s. "
                    f"Lock held by: {self._held_by}"
                )

            # Record acquisition
            wait_time_ms = (time.time() - start_time) * 1000
            self._metrics.record_acquisition(wait_time_ms)
            self._held_by = caller
            self._acquire_time = time.time()
            self._metrics.current_holder = caller
            self._metrics.acquire_time = self._acquire_time

            if capture_stack:
                import traceback
                self._acquire_stack = "".join(traceback.format_stack())

            if wait_time_ms > 1000:
                logger.warning(
                    f"[{self._lock_id}] High contention - waited {wait_time_ms:.0f}ms "
                    f"(caller={caller})"
                )

            return True

        finally:
            self._contention_waiters -= 1

    def release(self) -> None:
        """Release lock with duration tracking."""
        if not self._lock.locked():
            logger.warning(f"[{self._lock_id}] Attempted to release unlocked lock")
            return

        hold_time_ms = 0.0
        if self._acquire_time:
            hold_time_ms = (time.time() - self._acquire_time) * 1000
            self._metrics.record_release(hold_time_ms)

            if hold_time_ms > self._warn_hold_time * 1000:
                logger.warning(
                    f"[{self._lock_id}] Lock held for {hold_time_ms:.0f}ms by {self._held_by}"
                )

        # Clear tracking
        self._held_by = None
        self._acquire_time = None
        self._acquire_stack = None
        self._metrics.current_holder = None
        self._metrics.acquire_time = None

        self._lock.release()

    @asynccontextmanager
    async def acquire_context(
        self,
        caller: str = "unknown",
        timeout: Optional[float] = None,
    ):
        """
        Context manager for lock acquisition with automatic release.

        Example:
            async with lock.acquire_context("my_operation"):
                # Critical section
                pass
        """
        await self.acquire(caller=caller, timeout=timeout)
        try:
            yield
        finally:
            self.release()

    def force_release(self) -> bool:
        """
        Force release a stale lock. Use with caution!

        Returns:
            True if lock was released, False if not locked
        """
        if not self._lock.locked():
            return False

        logger.warning(
            f"[{self._lock_id}] FORCE RELEASE - was held by {self._held_by} "
            f"for {(time.time() - (self._acquire_time or 0)):.1f}s"
        )

        self._held_by = None
        self._acquire_time = None
        self._acquire_stack = None
        self._metrics.current_holder = None

        # Force unlock by directly manipulating internal state
        # This is a recovery mechanism for truly stuck locks
        try:
            self._lock.release()
            return True
        except RuntimeError:
            # Already unlocked somehow
            return False

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the lock."""
        return {
            "lock_id": self._lock_id,
            "is_locked": self.is_locked,
            "is_stale": self.is_stale,
            "held_by": self._held_by,
            "hold_time_s": (
                (time.time() - self._acquire_time) if self._acquire_time else None
            ),
            "contention_waiters": self._contention_waiters,
            "metrics": {
                "total_acquisitions": self._metrics.total_acquisitions,
                "total_timeouts": self._metrics.total_timeouts,
                "avg_wait_time_ms": (
                    self._metrics.total_wait_time_ms / max(1, self._metrics.total_acquisitions)
                ),
                "max_hold_time_ms": self._metrics.max_hold_time_ms,
            },
            "config": {
                "timeout": self._timeout,
                "warn_hold_time": self._warn_hold_time,
                "stale_threshold": self._stale_threshold,
            },
        }


# =============================================================================
# 2. ADAPTIVE BACKPRESSURE
# =============================================================================


class BackpressureState(Enum):
    """Current backpressure state."""
    NORMAL = "normal"           # Accept all requests
    ELEVATED = "elevated"       # Slightly throttled
    HIGH = "high"               # Significant throttling
    CRITICAL = "critical"       # Rejecting most requests
    OVERLOADED = "overloaded"   # Rejecting all new requests


@dataclass
class BackpressureMetrics:
    """Metrics for backpressure monitoring."""
    requests_accepted: int = 0
    requests_rejected: int = 0
    current_queue_size: int = 0
    peak_queue_size: int = 0
    memory_pressure: float = 0.0
    cpu_pressure: float = 0.0
    state: BackpressureState = BackpressureState.NORMAL
    last_state_change: float = field(default_factory=time.time)


class AdaptiveBackpressure:
    """
    Adaptive backpressure control based on system resources.

    Prevents memory exhaustion and maintains system stability under load
    by dynamically adjusting admission control based on:
        - Memory pressure
        - CPU usage
        - Queue depth
        - Request rate

    Environment Variables:
        MAX_REQUEST_QUEUE: Maximum queue size
        BACKPRESSURE_MEMORY_THRESHOLD: Memory usage threshold (0.0-1.0)
        BACKPRESSURE_CPU_THRESHOLD: CPU usage threshold (0.0-1.0)
        BACKPRESSURE_RATE_LIMIT: Max requests per second
        BACKPRESSURE_SAMPLE_INTERVAL: Sampling interval in seconds

    Example:
        backpressure = AdaptiveBackpressure()

        if await backpressure.try_acquire():
            try:
                await process_request()
            finally:
                await backpressure.release()
        else:
            raise ServiceOverloadedError()
    """

    def __init__(self):
        self._max_queue = _env_int("MAX_REQUEST_QUEUE", 1000)
        self._memory_threshold = _env_float("BACKPRESSURE_MEMORY_THRESHOLD", 0.85)
        self._cpu_threshold = _env_float("BACKPRESSURE_CPU_THRESHOLD", 0.90)
        self._rate_limit = _env_float("BACKPRESSURE_RATE_LIMIT", 100.0)
        self._sample_interval = _env_float("BACKPRESSURE_SAMPLE_INTERVAL", 1.0)

        self._current_queue = 0
        self._lock = asyncio.Lock()
        self._metrics = BackpressureMetrics()

        # Rate limiting using sliding window
        self._request_times: Deque[float] = deque(maxlen=1000)

        # State thresholds (fraction of max_queue)
        self._elevated_threshold = 0.5
        self._high_threshold = 0.75
        self._critical_threshold = 0.90

        # Cached pressure values
        self._last_memory_check = 0.0
        self._cached_memory_pressure = 0.0
        self._last_cpu_check = 0.0
        self._cached_cpu_pressure = 0.0

        # Monitor task
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background monitoring."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("[Backpressure] Started adaptive monitoring")

    async def stop(self) -> None:
        """Stop background monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._sample_interval)
                await self._update_pressure_readings()
                self._update_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Backpressure] Monitor error: {e}")

    async def _update_pressure_readings(self) -> None:
        """Update cached pressure readings."""
        now = time.time()

        # Memory pressure
        if now - self._last_memory_check > self._sample_interval:
            self._cached_memory_pressure = self._get_memory_pressure()
            self._last_memory_check = now
            self._metrics.memory_pressure = self._cached_memory_pressure

        # CPU pressure (only if psutil available)
        if PSUTIL_AVAILABLE and now - self._last_cpu_check > self._sample_interval:
            try:
                # Non-blocking CPU check (returns immediately with last reading)
                self._cached_cpu_pressure = psutil.cpu_percent(interval=None) / 100.0
                self._last_cpu_check = now
                self._metrics.cpu_pressure = self._cached_cpu_pressure
            except Exception:
                pass

    def _get_memory_pressure(self) -> float:
        """Get current memory pressure (0.0 to 1.0)."""
        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                return mem.percent / 100.0
            except Exception:
                pass

        # Fallback: Use gc stats for approximation
        gc_stats = gc.get_stats()
        if gc_stats:
            # Rough heuristic based on GC pressure
            collections = sum(s.get("collections", 0) for s in gc_stats)
            # If lots of GC happening, assume memory pressure
            if collections > 100:
                return 0.7

        return 0.5  # Unknown, assume moderate

    def _get_current_rate(self) -> float:
        """Get current request rate (requests per second)."""
        now = time.time()
        # Remove old entries
        while self._request_times and now - self._request_times[0] > 1.0:
            self._request_times.popleft()
        return len(self._request_times)

    def _update_state(self) -> None:
        """Update backpressure state based on current conditions."""
        old_state = self._metrics.state

        # Check memory first (most critical)
        if self._cached_memory_pressure > 0.95:
            new_state = BackpressureState.OVERLOADED
        elif self._cached_memory_pressure > self._memory_threshold:
            new_state = BackpressureState.CRITICAL
        elif self._cached_cpu_pressure > self._cpu_threshold:
            new_state = BackpressureState.HIGH
        else:
            # Check queue-based state
            queue_ratio = self._current_queue / max(1, self._max_queue)
            if queue_ratio >= self._critical_threshold:
                new_state = BackpressureState.CRITICAL
            elif queue_ratio >= self._high_threshold:
                new_state = BackpressureState.HIGH
            elif queue_ratio >= self._elevated_threshold:
                new_state = BackpressureState.ELEVATED
            else:
                new_state = BackpressureState.NORMAL

        if new_state != old_state:
            self._metrics.state = new_state
            self._metrics.last_state_change = time.time()
            logger.info(f"[Backpressure] State changed: {old_state.value} → {new_state.value}")

    def _calculate_admission_probability(self) -> float:
        """Calculate probability of admitting a new request."""
        state = self._metrics.state

        if state == BackpressureState.NORMAL:
            return 1.0
        elif state == BackpressureState.ELEVATED:
            return 0.9
        elif state == BackpressureState.HIGH:
            return 0.7
        elif state == BackpressureState.CRITICAL:
            return 0.3
        else:  # OVERLOADED
            return 0.0

    async def try_acquire(self, request_weight: int = 1) -> bool:
        """
        Try to acquire capacity for a request.

        Args:
            request_weight: Weight of this request (default 1)

        Returns:
            True if request admitted, False if rejected
        """
        async with self._lock:
            # Update state
            await self._update_pressure_readings()
            self._update_state()

            # Check rate limit
            current_rate = self._get_current_rate()
            if current_rate >= self._rate_limit:
                self._metrics.requests_rejected += 1
                return False

            # Check queue capacity
            if self._current_queue + request_weight > self._max_queue:
                self._metrics.requests_rejected += 1
                return False

            # Probabilistic admission based on state
            probability = self._calculate_admission_probability()
            if probability < 1.0:
                import random
                if random.random() > probability:
                    self._metrics.requests_rejected += 1
                    return False

            # Admit request
            self._current_queue += request_weight
            self._request_times.append(time.time())
            self._metrics.requests_accepted += 1
            self._metrics.current_queue_size = self._current_queue

            if self._current_queue > self._metrics.peak_queue_size:
                self._metrics.peak_queue_size = self._current_queue

            return True

    async def release(self, request_weight: int = 1) -> None:
        """Release capacity."""
        async with self._lock:
            self._current_queue = max(0, self._current_queue - request_weight)
            self._metrics.current_queue_size = self._current_queue

    @asynccontextmanager
    async def acquire_context(self, request_weight: int = 1):
        """Context manager for backpressure acquisition."""
        if not await self.try_acquire(request_weight):
            raise BackpressureRejection(
                f"Request rejected due to backpressure (state={self._metrics.state.value})"
            )
        try:
            yield
        finally:
            await self.release(request_weight)

    def get_state(self) -> BackpressureState:
        """Get current backpressure state."""
        return self._metrics.state

    def get_metrics(self) -> Dict[str, Any]:
        """Get backpressure metrics."""
        return {
            "state": self._metrics.state.value,
            "current_queue": self._metrics.current_queue_size,
            "max_queue": self._max_queue,
            "peak_queue": self._metrics.peak_queue_size,
            "requests_accepted": self._metrics.requests_accepted,
            "requests_rejected": self._metrics.requests_rejected,
            "rejection_rate": (
                self._metrics.requests_rejected /
                max(1, self._metrics.requests_accepted + self._metrics.requests_rejected)
            ),
            "memory_pressure": self._metrics.memory_pressure,
            "cpu_pressure": self._metrics.cpu_pressure,
            "current_rate": self._get_current_rate(),
            "rate_limit": self._rate_limit,
        }


class BackpressureRejection(Exception):
    """Exception raised when request is rejected due to backpressure."""
    pass


# =============================================================================
# 3. RESOURCE BULKHEAD
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)


class PoolCircuitBreaker:
    """
    Circuit breaker for a resource pool.

    Protects against cascading failures by temporarily stopping
    requests to failing resources.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Resource failing, requests rejected immediately
        HALF_OPEN: Testing recovery, limited requests allowed

    Environment Variables:
        CIRCUIT_FAILURE_THRESHOLD: Failures before opening
        CIRCUIT_SUCCESS_THRESHOLD: Successes to close
        CIRCUIT_RESET_TIMEOUT: Seconds before half-open
    """

    def __init__(
        self,
        failure_threshold: Optional[int] = None,
        success_threshold: Optional[int] = None,
        reset_timeout: Optional[float] = None,
        name: str = "default",
    ):
        self._failure_threshold = failure_threshold or _env_int("CIRCUIT_FAILURE_THRESHOLD", 5)
        self._success_threshold = success_threshold or _env_int("CIRCUIT_SUCCESS_THRESHOLD", 3)
        self._reset_timeout = reset_timeout or _env_float("CIRCUIT_RESET_TIMEOUT", 30.0)
        self._name = name
        self._metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        self._half_open_permits = 1  # Limit concurrent half-open tests
        self._half_open_in_use = 0

    @property
    def state(self) -> CircuitState:
        return self._metrics.state

    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._metrics.state == CircuitState.OPEN

    async def can_execute(self) -> bool:
        """
        Check if execution is allowed and acquire permit if half-open.

        Returns:
            True if execution allowed, False otherwise
        """
        async with self._lock:
            if self._metrics.state == CircuitState.CLOSED:
                return True

            if self._metrics.state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if self._metrics.last_failure_time:
                    elapsed = time.time() - self._metrics.last_failure_time
                    if elapsed >= self._reset_timeout:
                        self._transition_to(CircuitState.HALF_OPEN)
                        return await self._try_acquire_half_open_permit()
                return False

            if self._metrics.state == CircuitState.HALF_OPEN:
                return await self._try_acquire_half_open_permit()

            return False

    async def _try_acquire_half_open_permit(self) -> bool:
        """Try to acquire a half-open permit for testing recovery."""
        if self._half_open_in_use < self._half_open_permits:
            self._half_open_in_use += 1
            return True
        return False

    async def record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            self._metrics.successes += 1
            self._metrics.consecutive_successes += 1
            self._metrics.consecutive_failures = 0
            self._metrics.last_success_time = time.time()

            if self._metrics.state == CircuitState.HALF_OPEN:
                self._half_open_in_use = max(0, self._half_open_in_use - 1)

                if self._metrics.consecutive_successes >= self._success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed execution."""
        async with self._lock:
            self._metrics.failures += 1
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
            self._metrics.last_failure_time = time.time()

            if self._metrics.state == CircuitState.HALF_OPEN:
                self._half_open_in_use = max(0, self._half_open_in_use - 1)
                self._transition_to(CircuitState.OPEN)

            elif self._metrics.state == CircuitState.CLOSED:
                if self._metrics.consecutive_failures >= self._failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._metrics.state
        if old_state != new_state:
            self._metrics.state = new_state
            self._metrics.last_state_change = time.time()

            if new_state == CircuitState.CLOSED:
                self._metrics.consecutive_failures = 0
                self._half_open_in_use = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._metrics.consecutive_successes = 0

            logger.info(
                f"[Circuit:{self._name}] State transition: {old_state.value} → {new_state.value}"
            )

    async def force_reset(self) -> None:
        """Force reset the circuit breaker to closed state."""
        async with self._lock:
            self._metrics = CircuitBreakerMetrics()
            self._half_open_in_use = 0
            logger.info(f"[Circuit:{self._name}] Force reset to CLOSED")

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self._name,
            "state": self._metrics.state.value,
            "failures": self._metrics.failures,
            "successes": self._metrics.successes,
            "consecutive_failures": self._metrics.consecutive_failures,
            "consecutive_successes": self._metrics.consecutive_successes,
            "failure_threshold": self._failure_threshold,
            "success_threshold": self._success_threshold,
            "reset_timeout": self._reset_timeout,
            "half_open_in_use": self._half_open_in_use,
        }


class ResourceBulkhead:
    """
    Isolates resource pools to prevent cascading failures.

    Each pool has its own semaphore and circuit breaker, ensuring that
    failure in one pool doesn't affect others.

    Environment Variables:
        BULKHEAD_POOL_SIZE_{POOL_NAME}: Size for specific pool
        BULKHEAD_DEFAULT_POOL_SIZE: Default pool size
        BULKHEAD_TIMEOUT: Default execution timeout

    Example:
        bulkhead = ResourceBulkhead({
            "prime": 10,      # 10 concurrent Prime requests
            "claude": 5,      # 5 concurrent Claude requests
            "reactor": 3,     # 3 concurrent Reactor requests
        })

        async with bulkhead.execute("prime") as permit:
            result = await call_prime()
    """

    def __init__(
        self,
        pools: Optional[Dict[str, int]] = None,
        default_timeout: Optional[float] = None,
    ):
        default_size = _env_int("BULKHEAD_DEFAULT_POOL_SIZE", 10)
        self._default_timeout = default_timeout or _env_float("BULKHEAD_TIMEOUT", 30.0)

        # Initialize pools from config or environment
        self._pool_sizes: Dict[str, int] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._circuit_breakers: Dict[str, PoolCircuitBreaker] = {}
        self._active_counts: Dict[str, int] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

        if pools:
            for name, size in pools.items():
                # Check for environment override
                env_size = _env_int(f"BULKHEAD_POOL_SIZE_{name.upper()}", size)
                self._add_pool(name, env_size)

        # Add default pools for Trinity
        for name in ["prime", "claude", "reactor"]:
            if name not in self._pool_sizes:
                env_size = _env_int(f"BULKHEAD_POOL_SIZE_{name.upper()}", default_size)
                self._add_pool(name, env_size)

    def _add_pool(self, name: str, size: int) -> None:
        """Add a resource pool."""
        self._pool_sizes[name] = size
        self._semaphores[name] = asyncio.Semaphore(size)
        self._circuit_breakers[name] = PoolCircuitBreaker(name=name)
        self._active_counts[name] = 0
        self._locks[name] = asyncio.Lock()

    def get_pool_names(self) -> List[str]:
        """Get list of pool names."""
        return list(self._pool_sizes.keys())

    def get_pool_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific pool."""
        if name not in self._pool_sizes:
            return None

        return {
            "size": self._pool_sizes[name],
            "active": self._active_counts[name],
            "available": self._pool_sizes[name] - self._active_counts[name],
            "circuit": self._circuit_breakers[name].get_metrics(),
        }

    @asynccontextmanager
    async def execute(
        self,
        pool_name: str,
        timeout: Optional[float] = None,
    ):
        """
        Execute within a bulkhead pool.

        Args:
            pool_name: Name of the pool
            timeout: Execution timeout (overrides default)

        Yields:
            Permit for execution

        Raises:
            BulkheadRejection: If pool circuit is open
            BulkheadTimeout: If semaphore acquisition times out
        """
        if pool_name not in self._semaphores:
            # Create pool on demand
            self._add_pool(pool_name, _env_int("BULKHEAD_DEFAULT_POOL_SIZE", 10))

        circuit = self._circuit_breakers[pool_name]
        semaphore = self._semaphores[pool_name]
        effective_timeout = timeout if timeout is not None else self._default_timeout

        # Check circuit breaker first
        if not await circuit.can_execute():
            raise BulkheadCircuitOpen(
                f"Pool '{pool_name}' circuit is open: "
                f"{circuit.get_metrics()}"
            )

        # Try to acquire semaphore
        try:
            acquired = await asyncio.wait_for(
                semaphore.acquire(),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            await circuit.record_failure()
            raise BulkheadTimeout(
                f"Pool '{pool_name}' acquisition timeout after {effective_timeout}s"
            )

        # Track active count
        async with self._locks[pool_name]:
            self._active_counts[pool_name] += 1

        try:
            yield BulkheadPermit(pool_name)
            await circuit.record_success()
        except Exception as e:
            await circuit.record_failure(e)
            raise
        finally:
            semaphore.release()
            async with self._locks[pool_name]:
                self._active_counts[pool_name] = max(0, self._active_counts[pool_name] - 1)

    async def execute_with_fallback(
        self,
        pool_name: str,
        primary_fn: Callable[[], Awaitable[T]],
        fallback_fn: Optional[Callable[[], Awaitable[T]]] = None,
        fallback_pool: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> T:
        """
        Execute with automatic fallback on bulkhead rejection.

        Args:
            pool_name: Primary pool name
            primary_fn: Primary function to execute
            fallback_fn: Fallback function if primary fails
            fallback_pool: Pool for fallback execution
            timeout: Execution timeout

        Returns:
            Result from primary or fallback function
        """
        try:
            async with self.execute(pool_name, timeout):
                return await primary_fn()
        except (BulkheadCircuitOpen, BulkheadTimeout) as e:
            if fallback_fn is None:
                raise

            logger.warning(f"[Bulkhead] Primary pool '{pool_name}' unavailable: {e}")

            if fallback_pool:
                async with self.execute(fallback_pool, timeout):
                    return await fallback_fn()
            else:
                return await fallback_fn()

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all pools."""
        return {
            name: self.get_pool_status(name)
            for name in self._pool_sizes
        }

    async def reset_circuit(self, pool_name: str) -> bool:
        """Reset circuit breaker for a pool."""
        if pool_name not in self._circuit_breakers:
            return False
        await self._circuit_breakers[pool_name].force_reset()
        return True


@dataclass
class BulkheadPermit:
    """Permit for bulkhead execution."""
    pool_name: str


class BulkheadCircuitOpen(Exception):
    """Exception raised when bulkhead circuit is open."""
    pass


class BulkheadTimeout(Exception):
    """Exception raised when bulkhead acquisition times out."""
    pass


# =============================================================================
# 4. TRINITY RATE LIMITER
# =============================================================================


class TrinityRateLimiter:
    """
    Token bucket rate limiter for Trinity message flows.

    Prevents message floods from overwhelming components while allowing
    short bursts of activity.

    Environment Variables:
        TRINITY_RATE_LIMIT: Messages per second
        TRINITY_RATE_BURST: Maximum burst size
        TRINITY_RATE_REFILL_INTERVAL: Token refill interval in seconds

    Example:
        limiter = TrinityRateLimiter(rate=100.0, burst=50)

        if await limiter.acquire():
            await send_message()
        else:
            # Rate limited, wait or reject
            pass
    """

    def __init__(
        self,
        rate: Optional[float] = None,
        burst: Optional[int] = None,
        name: str = "default",
    ):
        self._rate = rate or _env_float("TRINITY_RATE_LIMIT", 100.0)
        self._burst = burst or _env_int("TRINITY_RATE_BURST", 50)
        self._name = name

        self._tokens = float(self._burst)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

        # Metrics
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_waited = 0

    async def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(
            float(self._burst),
            self._tokens + elapsed * self._rate
        )
        self._last_update = now

    async def acquire(self, tokens: int = 1, wait: bool = False) -> bool:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire
            wait: If True, wait until tokens available

        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            await self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += tokens
                return True

            if not wait:
                self._total_rejected += tokens
                return False

            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self._rate

        # Wait outside lock
        await asyncio.sleep(wait_time)
        self._total_waited += 1

        # Retry acquisition
        async with self._lock:
            await self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += tokens
                return True

            self._total_rejected += tokens
            return False

    async def acquire_or_wait(self, tokens: int = 1, max_wait: float = 10.0) -> bool:
        """
        Acquire tokens, waiting up to max_wait seconds.

        Args:
            tokens: Number of tokens to acquire
            max_wait: Maximum time to wait

        Returns:
            True if acquired, False if timeout
        """
        try:
            return await asyncio.wait_for(
                self._acquire_with_wait(tokens),
                timeout=max_wait
            )
        except asyncio.TimeoutError:
            self._total_rejected += tokens
            return False

    async def _acquire_with_wait(self, tokens: int) -> bool:
        """Internal acquire with wait."""
        while True:
            async with self._lock:
                await self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._total_acquired += tokens
                    return True

                # Calculate wait time
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._rate

            await asyncio.sleep(min(wait_time, 0.1))

    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        return {
            "name": self._name,
            "rate": self._rate,
            "burst": self._burst,
            "current_tokens": self._tokens,
            "total_acquired": self._total_acquired,
            "total_rejected": self._total_rejected,
            "total_waited": self._total_waited,
            "rejection_rate": (
                self._total_rejected /
                max(1, self._total_acquired + self._total_rejected)
            ),
        }


# =============================================================================
# 5. ADAPTIVE RESOURCE MANAGER
# =============================================================================


class AdaptiveResourceManager:
    """
    Dynamically adjusts resource limits based on system capacity.

    Features:
        - CPU-based concurrency scaling
        - Memory-aware resource allocation
        - Dynamic semaphore resizing
        - Load-based optimization

    Environment Variables:
        ADAPTIVE_MIN_CONCURRENCY: Minimum concurrent operations
        ADAPTIVE_MAX_CONCURRENCY: Maximum concurrent operations
        ADAPTIVE_CPU_SCALE_FACTOR: CPU cores to concurrency ratio
        ADAPTIVE_MEMORY_THRESHOLD: Memory threshold for scaling
        ADAPTIVE_SAMPLE_INTERVAL: Sampling interval in seconds

    Example:
        manager = AdaptiveResourceManager()
        await manager.start()

        concurrency = manager.get_optimal_concurrency()
        async with manager.acquire_resource("compute"):
            await heavy_computation()
    """

    def __init__(self):
        self._min_concurrency = _env_int("ADAPTIVE_MIN_CONCURRENCY", 2)
        self._max_concurrency = _env_int("ADAPTIVE_MAX_CONCURRENCY", 100)
        self._cpu_scale_factor = _env_float("ADAPTIVE_CPU_SCALE_FACTOR", 2.0)
        self._memory_threshold = _env_float("ADAPTIVE_MEMORY_THRESHOLD", 0.85)
        self._sample_interval = _env_float("ADAPTIVE_SAMPLE_INTERVAL", 5.0)

        self._cpu_cores = os.cpu_count() or 4
        self._base_concurrency = int(self._cpu_cores * self._cpu_scale_factor)

        # Dynamic semaphores
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._semaphore_sizes: Dict[str, int] = {}
        self._active_counts: Dict[str, int] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

        # Current optimal concurrency
        self._current_concurrency = min(
            self._max_concurrency,
            max(self._min_concurrency, self._base_concurrency)
        )

        # Monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_cpu_usage = 0.0
        self._last_memory_usage = 0.0

        # History for trend analysis
        self._cpu_history: Deque[float] = deque(maxlen=12)
        self._memory_history: Deque[float] = deque(maxlen=12)

    async def start(self) -> None:
        """Start background monitoring."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info(
                f"[AdaptiveRM] Started with base concurrency {self._base_concurrency} "
                f"(cores={self._cpu_cores}, scale={self._cpu_scale_factor})"
            )

    async def stop(self) -> None:
        """Stop background monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self._sample_interval)
                await self._update_metrics()
                self._adjust_concurrency()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AdaptiveRM] Monitor error: {e}")

    async def _update_metrics(self) -> None:
        """Update system metrics."""
        if PSUTIL_AVAILABLE:
            try:
                self._last_cpu_usage = psutil.cpu_percent(interval=None) / 100.0
                self._last_memory_usage = psutil.virtual_memory().percent / 100.0

                self._cpu_history.append(self._last_cpu_usage)
                self._memory_history.append(self._last_memory_usage)
            except Exception:
                pass

    def _adjust_concurrency(self) -> None:
        """Adjust concurrency based on system load."""
        old_concurrency = self._current_concurrency

        # Calculate optimal concurrency
        if self._last_memory_usage > self._memory_threshold:
            # Memory pressure - reduce concurrency
            factor = max(0.5, 1.0 - (self._last_memory_usage - self._memory_threshold) * 2)
            new_concurrency = int(self._base_concurrency * factor)
        elif self._last_cpu_usage > 0.9:
            # High CPU - reduce concurrency
            factor = max(0.5, 1.0 - (self._last_cpu_usage - 0.8))
            new_concurrency = int(self._base_concurrency * factor)
        elif self._last_cpu_usage < 0.3 and self._last_memory_usage < 0.5:
            # Low load - can increase concurrency
            new_concurrency = int(self._base_concurrency * 1.5)
        else:
            # Normal operation
            new_concurrency = self._base_concurrency

        # Apply limits
        self._current_concurrency = min(
            self._max_concurrency,
            max(self._min_concurrency, new_concurrency)
        )

        if self._current_concurrency != old_concurrency:
            logger.info(
                f"[AdaptiveRM] Concurrency adjusted: {old_concurrency} → "
                f"{self._current_concurrency} (cpu={self._last_cpu_usage:.1%}, "
                f"mem={self._last_memory_usage:.1%})"
            )

    def get_optimal_concurrency(self) -> int:
        """Get current optimal concurrency level."""
        return self._current_concurrency

    def get_semaphore(self, name: str, size: Optional[int] = None) -> asyncio.Semaphore:
        """
        Get or create a named semaphore.

        Args:
            name: Semaphore name
            size: Optional specific size (otherwise uses optimal concurrency)

        Returns:
            Semaphore for the resource
        """
        if name not in self._semaphores:
            effective_size = size or self._current_concurrency
            self._semaphores[name] = asyncio.Semaphore(effective_size)
            self._semaphore_sizes[name] = effective_size
            self._active_counts[name] = 0
            self._locks[name] = asyncio.Lock()
        return self._semaphores[name]

    @asynccontextmanager
    async def acquire_resource(
        self,
        name: str,
        timeout: Optional[float] = None,
    ):
        """
        Acquire a resource with adaptive concurrency.

        Args:
            name: Resource name
            timeout: Acquisition timeout

        Yields:
            Resource permit
        """
        semaphore = self.get_semaphore(name)
        effective_timeout = timeout or 30.0

        try:
            acquired = await asyncio.wait_for(
                semaphore.acquire(),
                timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Resource '{name}' acquisition timeout after {effective_timeout}s")

        async with self._locks[name]:
            self._active_counts[name] = self._active_counts.get(name, 0) + 1

        try:
            yield ResourcePermit(name)
        finally:
            semaphore.release()
            async with self._locks[name]:
                self._active_counts[name] = max(0, self._active_counts.get(name, 1) - 1)

    def get_status(self) -> Dict[str, Any]:
        """Get resource manager status."""
        return {
            "optimal_concurrency": self._current_concurrency,
            "base_concurrency": self._base_concurrency,
            "cpu_cores": self._cpu_cores,
            "cpu_scale_factor": self._cpu_scale_factor,
            "last_cpu_usage": self._last_cpu_usage,
            "last_memory_usage": self._last_memory_usage,
            "min_concurrency": self._min_concurrency,
            "max_concurrency": self._max_concurrency,
            "resources": {
                name: {
                    "size": self._semaphore_sizes.get(name, 0),
                    "active": self._active_counts.get(name, 0),
                }
                for name in self._semaphores
            },
        }


@dataclass
class ResourcePermit:
    """Permit for resource acquisition."""
    resource_name: str


# =============================================================================
# 6. DEEP HEALTH VERIFIER
# =============================================================================


class HealthCheckResult(Enum):
    """Health check result."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component."""
    name: str
    result: HealthCheckResult
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class DeepHealthVerifier:
    """
    Verifies actual functionality beyond HTTP 200 responses.

    Features:
        - Functional verification (can component actually do work)
        - Latency measurement
        - Capability testing
        - Trend analysis

    Environment Variables:
        HEALTH_CHECK_TIMEOUT: Default timeout for health checks
        HEALTH_FUNCTIONAL_TEST: Whether to run functional tests

    Example:
        verifier = DeepHealthVerifier()

        health = await verifier.verify_component(
            "prime",
            url="http://localhost:8000",
            functional_test=async () => await prime.generate("test")
        )
    """

    def __init__(self):
        self._default_timeout = _env_float("HEALTH_CHECK_TIMEOUT", 10.0)
        self._functional_test_enabled = _env_bool("HEALTH_FUNCTIONAL_TEST", True)

        # Cache session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None

        # Health history for trend analysis
        self._health_history: Dict[str, Deque[ComponentHealth]] = {}

    async def _get_session(self) -> Optional[aiohttp.ClientSession]:
        """Get or create aiohttp session."""
        if not AIOHTTP_AVAILABLE:
            return None
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._default_timeout)
            )
        return self._session

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def verify_http(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        timeout: Optional[float] = None,
    ) -> ComponentHealth:
        """
        Verify HTTP endpoint is responding.

        Args:
            name: Component name
            url: Health check URL
            expected_status: Expected HTTP status code
            timeout: Request timeout

        Returns:
            ComponentHealth with result
        """
        start_time = time.time()
        effective_timeout = timeout or self._default_timeout

        try:
            session = await self._get_session()
            if session is None:
                return ComponentHealth(
                    name=name,
                    result=HealthCheckResult.UNKNOWN,
                    latency_ms=0,
                    error="aiohttp not available",
                )

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=effective_timeout)) as response:
                latency_ms = (time.time() - start_time) * 1000

                if response.status == expected_status:
                    result = HealthCheckResult.HEALTHY
                elif response.status < 500:
                    result = HealthCheckResult.DEGRADED
                else:
                    result = HealthCheckResult.UNHEALTHY

                return ComponentHealth(
                    name=name,
                    result=result,
                    latency_ms=latency_ms,
                    details={"status_code": response.status},
                )

        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                result=HealthCheckResult.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"Timeout after {effective_timeout}s",
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                result=HealthCheckResult.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def verify_functional(
        self,
        name: str,
        test_fn: Callable[[], Awaitable[Any]],
        validate_fn: Optional[Callable[[Any], bool]] = None,
        timeout: Optional[float] = None,
    ) -> ComponentHealth:
        """
        Verify component can actually perform work.

        Args:
            name: Component name
            test_fn: Async function to test functionality
            validate_fn: Optional function to validate result
            timeout: Test timeout

        Returns:
            ComponentHealth with result
        """
        if not self._functional_test_enabled:
            return ComponentHealth(
                name=name,
                result=HealthCheckResult.UNKNOWN,
                latency_ms=0,
                details={"reason": "functional tests disabled"},
            )

        start_time = time.time()
        effective_timeout = timeout or self._default_timeout

        try:
            result = await asyncio.wait_for(test_fn(), timeout=effective_timeout)
            latency_ms = (time.time() - start_time) * 1000

            # Validate result if validator provided
            if validate_fn:
                is_valid = validate_fn(result)
                if not is_valid:
                    return ComponentHealth(
                        name=name,
                        result=HealthCheckResult.DEGRADED,
                        latency_ms=latency_ms,
                        details={"reason": "validation failed"},
                    )

            return ComponentHealth(
                name=name,
                result=HealthCheckResult.HEALTHY,
                latency_ms=latency_ms,
                details={"functional_test": "passed"},
            )

        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                result=HealthCheckResult.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"Functional test timeout after {effective_timeout}s",
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                result=HealthCheckResult.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                error=f"Functional test failed: {e}",
            )

    async def verify_component(
        self,
        name: str,
        url: Optional[str] = None,
        functional_test: Optional[Callable[[], Awaitable[Any]]] = None,
        validate_fn: Optional[Callable[[Any], bool]] = None,
        timeout: Optional[float] = None,
    ) -> ComponentHealth:
        """
        Comprehensive component verification.

        Args:
            name: Component name
            url: Optional HTTP health URL
            functional_test: Optional functional test
            validate_fn: Optional result validator
            timeout: Verification timeout

        Returns:
            ComponentHealth with combined result
        """
        results: List[ComponentHealth] = []

        # HTTP check
        if url:
            http_health = await self.verify_http(name, url, timeout=timeout)
            results.append(http_health)

            # If HTTP fails, skip functional test
            if http_health.result == HealthCheckResult.UNHEALTHY:
                self._record_health(name, http_health)
                return http_health

        # Functional test
        if functional_test:
            func_health = await self.verify_functional(
                name, functional_test, validate_fn, timeout
            )
            results.append(func_health)

        # Combine results
        if not results:
            combined = ComponentHealth(
                name=name,
                result=HealthCheckResult.UNKNOWN,
                latency_ms=0,
                details={"reason": "no checks performed"},
            )
        else:
            # Use worst result
            worst = min(results, key=lambda h: list(HealthCheckResult).index(h.result))
            total_latency = sum(r.latency_ms for r in results)
            combined = ComponentHealth(
                name=name,
                result=worst.result,
                latency_ms=total_latency,
                details={
                    "checks": len(results),
                    "results": [
                        {"check": r.name, "result": r.result.value, "latency_ms": r.latency_ms}
                        for r in results
                    ],
                },
                error=worst.error,
            )

        self._record_health(name, combined)
        return combined

    def _record_health(self, name: str, health: ComponentHealth) -> None:
        """Record health for trend analysis."""
        if name not in self._health_history:
            self._health_history[name] = deque(maxlen=100)
        self._health_history[name].append(health)

    def get_health_trend(self, name: str) -> Dict[str, Any]:
        """Get health trend for a component."""
        if name not in self._health_history:
            return {"error": "no history"}

        history = list(self._health_history[name])
        if not history:
            return {"error": "empty history"}

        # Calculate stats
        healthy_count = sum(1 for h in history if h.result == HealthCheckResult.HEALTHY)
        avg_latency = sum(h.latency_ms for h in history) / len(history)

        return {
            "samples": len(history),
            "healthy_percentage": healthy_count / len(history) * 100,
            "avg_latency_ms": avg_latency,
            "latest": {
                "result": history[-1].result.value,
                "latency_ms": history[-1].latency_ms,
                "timestamp": history[-1].timestamp,
            },
        }


# =============================================================================
# 7. PRIORITY MESSAGE QUEUE
# =============================================================================


class MessagePriority(IntEnum):
    """Message priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class PriorityMessage(Generic[T]):
    """Message with priority."""
    data: T
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class PriorityMessageQueue(Generic[T]):
    """
    Priority-based message queue with fair scheduling.

    Features:
        - Multiple priority levels
        - Fair scheduling (prevent starvation)
        - Bounded queues per priority
        - Metrics and monitoring

    Environment Variables:
        PRIORITY_QUEUE_MAX_SIZE: Maximum total queue size
        PRIORITY_QUEUE_FAIR_INTERVAL: Interval for fairness adjustment

    Example:
        queue = PriorityMessageQueue[str]()

        await queue.put("urgent", MessagePriority.CRITICAL)
        await queue.put("normal", MessagePriority.NORMAL)

        msg = await queue.get()  # Returns "urgent" first
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        fair_interval: Optional[int] = None,
    ):
        self._max_size = max_size or _env_int("PRIORITY_QUEUE_MAX_SIZE", 10000)
        self._fair_interval = fair_interval or _env_int("PRIORITY_QUEUE_FAIR_INTERVAL", 10)

        # One queue per priority level
        self._queues: Dict[MessagePriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=self._max_size // 4)
            for priority in MessagePriority
        }

        self._total_size = 0
        self._lock = asyncio.Lock()

        # Fairness: track low-priority starvation
        self._gets_since_low_priority = 0

        # Metrics
        self._total_put = 0
        self._total_get = 0
        self._total_dropped = 0

    async def put(
        self,
        data: T,
        priority: MessagePriority = MessagePriority.NORMAL,
        nowait: bool = False,
    ) -> bool:
        """
        Put an item in the queue.

        Args:
            data: Data to queue
            priority: Message priority
            nowait: If True, don't wait if full

        Returns:
            True if queued, False if dropped
        """
        message = PriorityMessage(data=data, priority=priority)
        queue = self._queues[priority]

        try:
            if nowait:
                queue.put_nowait(message)
            else:
                await queue.put(message)

            async with self._lock:
                self._total_size += 1
                self._total_put += 1

            return True

        except asyncio.QueueFull:
            self._total_dropped += 1
            return False

    async def get(self, timeout: Optional[float] = None) -> T:
        """
        Get the highest priority message.

        Implements fair scheduling to prevent low-priority starvation.

        Args:
            timeout: Maximum wait time

        Returns:
            Message data

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        async with self._lock:
            # Fairness check: occasionally check low priority
            if self._gets_since_low_priority >= self._fair_interval:
                for priority in reversed(list(MessagePriority)):
                    queue = self._queues[priority]
                    if not queue.empty():
                        try:
                            message = queue.get_nowait()
                            self._total_size -= 1
                            self._total_get += 1
                            self._gets_since_low_priority = 0
                            return message.data
                        except asyncio.QueueEmpty:
                            continue

        # Normal priority-based get
        while True:
            for priority in MessagePriority:
                queue = self._queues[priority]
                if not queue.empty():
                    try:
                        message = queue.get_nowait()
                        async with self._lock:
                            self._total_size -= 1
                            self._total_get += 1
                            self._gets_since_low_priority += 1
                        return message.data
                    except asyncio.QueueEmpty:
                        continue

            # All empty, wait on critical queue
            try:
                if timeout:
                    message = await asyncio.wait_for(
                        self._queues[MessagePriority.CRITICAL].get(),
                        timeout=timeout
                    )
                else:
                    message = await self._queues[MessagePriority.CRITICAL].get()

                async with self._lock:
                    self._total_size -= 1
                    self._total_get += 1
                    self._gets_since_low_priority += 1

                return message.data

            except asyncio.TimeoutError:
                raise

    def size(self) -> int:
        """Get total queue size."""
        return self._total_size

    def size_by_priority(self) -> Dict[str, int]:
        """Get size by priority level."""
        return {
            priority.name: queue.qsize()
            for priority, queue in self._queues.items()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        return {
            "total_size": self._total_size,
            "max_size": self._max_size,
            "by_priority": self.size_by_priority(),
            "total_put": self._total_put,
            "total_get": self._total_get,
            "total_dropped": self._total_dropped,
            "drop_rate": (
                self._total_dropped / max(1, self._total_put)
            ),
        }


# =============================================================================
# 8. ATOMIC FILE IPC
# =============================================================================


class AtomicFileIPC:
    """
    Race-condition-free file-based IPC for Trinity Protocol.

    Uses fcntl file locking with atomic operations to prevent
    TOCTOU (time-of-check-time-of-use) race conditions.

    Features:
        - Atomic read-modify-write operations
        - fcntl-based file locking
        - Automatic lock cleanup
        - Corruption detection

    Environment Variables:
        TRINITY_IPC_DIR: Base directory for IPC files
        TRINITY_IPC_TIMEOUT: Lock acquisition timeout
        TRINITY_IPC_MAX_SIZE: Maximum message size

    Example:
        ipc = AtomicFileIPC("/path/to/trinity")

        # Atomic write
        await ipc.write_message("component", {"type": "heartbeat"})

        # Atomic read
        messages = await ipc.read_messages("component")
    """

    def __init__(self, base_dir: Optional[str] = None):
        self._base_dir = Path(
            base_dir or os.getenv("TRINITY_IPC_DIR", str(Path.home() / ".jarvis" / "trinity"))
        )
        self._timeout = _env_float("TRINITY_IPC_TIMEOUT", 5.0)
        self._max_size = _env_int("TRINITY_IPC_MAX_SIZE", 1024 * 1024)  # 1MB

        # Ensure directory exists
        self._base_dir.mkdir(parents=True, exist_ok=True)

        # Lock tracking
        self._held_locks: Dict[str, int] = {}  # path -> fd

    def _get_queue_path(self, component: str) -> Path:
        """Get queue file path for component."""
        return self._base_dir / f"{component}.queue.json"

    def _get_lock_path(self, component: str) -> Path:
        """Get lock file path for component."""
        return self._base_dir / f"{component}.lock"

    @asynccontextmanager
    async def _acquire_lock(self, component: str):
        """
        Acquire exclusive lock for component with atomic operation.

        This eliminates TOCTOU by acquiring lock BEFORE any file operations.
        """
        lock_path = self._get_lock_path(component)

        # Create lock file if needed (atomic operation)
        lock_path.touch(exist_ok=True)

        # Open for locking
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)

        try:
            # Acquire exclusive lock with timeout
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (IOError, OSError) as e:
                    if time.time() - start_time > self._timeout:
                        raise TimeoutError(
                            f"Lock acquisition timeout for {component} after {self._timeout}s"
                        )
                    await asyncio.sleep(0.01)

            self._held_locks[component] = fd

            yield

        finally:
            # Release lock
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except Exception:
                pass

            os.close(fd)
            self._held_locks.pop(component, None)

    async def write_message(
        self,
        component: str,
        message: Dict[str, Any],
        append: bool = True,
    ) -> bool:
        """
        Write message atomically.

        Args:
            component: Target component
            message: Message to write
            append: If True, append to queue; if False, replace

        Returns:
            True if successful
        """
        async with self._acquire_lock(component):
            queue_path = self._get_queue_path(component)

            # Read existing messages
            messages = []
            if append and queue_path.exists():
                try:
                    content = queue_path.read_text()
                    if content.strip():
                        messages = json.loads(content)
                except (json.JSONDecodeError, IOError):
                    # Corrupted, start fresh
                    messages = []

            # Add new message with metadata
            messages.append({
                **message,
                "_id": str(uuid.uuid4()),
                "_timestamp": time.time(),
            })

            # Check size limit
            content = json.dumps(messages, default=str)
            if len(content) > self._max_size:
                # Trim oldest messages
                while len(content) > self._max_size and len(messages) > 1:
                    messages.pop(0)
                    content = json.dumps(messages, default=str)

            # Write atomically using temp file
            temp_path = queue_path.with_suffix(".tmp")
            temp_path.write_text(content)
            temp_path.rename(queue_path)

            return True

    async def read_messages(
        self,
        component: str,
        clear: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Read messages atomically.

        Args:
            component: Source component
            clear: If True, clear after reading

        Returns:
            List of messages
        """
        async with self._acquire_lock(component):
            queue_path = self._get_queue_path(component)

            if not queue_path.exists():
                return []

            try:
                content = queue_path.read_text()
                if not content.strip():
                    return []

                messages = json.loads(content)

                if clear:
                    queue_path.write_text("[]")

                return messages

            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"[AtomicIPC] Read error for {component}: {e}")
                # Clear corrupted file
                queue_path.write_text("[]")
                return []

    async def write_heartbeat(
        self,
        component: str,
        status: str = "healthy",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Write heartbeat for component.

        Args:
            component: Component name
            status: Health status
            metadata: Optional additional metadata

        Returns:
            True if successful
        """
        heartbeat_path = self._base_dir / f"{component}.heartbeat.json"

        heartbeat = {
            "component": component,
            "status": status,
            "timestamp": time.time(),
            "pid": os.getpid(),
            **(metadata or {}),
        }

        # Write atomically
        temp_path = heartbeat_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(heartbeat))
        temp_path.rename(heartbeat_path)

        return True

    async def read_heartbeat(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Read heartbeat for component.

        Args:
            component: Component name

        Returns:
            Heartbeat data or None if not found/stale
        """
        heartbeat_path = self._base_dir / f"{component}.heartbeat.json"

        if not heartbeat_path.exists():
            return None

        try:
            content = heartbeat_path.read_text()
            return json.loads(content)
        except (json.JSONDecodeError, IOError):
            return None

    async def is_component_alive(
        self,
        component: str,
        stale_threshold: float = 30.0,
    ) -> bool:
        """
        Check if component is alive based on heartbeat.

        Args:
            component: Component name
            stale_threshold: Seconds before heartbeat considered stale

        Returns:
            True if component is alive
        """
        heartbeat = await self.read_heartbeat(component)
        if heartbeat is None:
            return False

        age = time.time() - heartbeat.get("timestamp", 0)
        return age < stale_threshold


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_backpressure: Optional[AdaptiveBackpressure] = None
_bulkhead: Optional[ResourceBulkhead] = None
_resource_manager: Optional[AdaptiveResourceManager] = None
_health_verifier: Optional[DeepHealthVerifier] = None
_rate_limiter: Optional[TrinityRateLimiter] = None
_atomic_ipc: Optional[AtomicFileIPC] = None

_init_lock = asyncio.Lock()


async def get_backpressure() -> AdaptiveBackpressure:
    """Get singleton AdaptiveBackpressure instance."""
    global _backpressure
    if _backpressure is None:
        async with _init_lock:
            if _backpressure is None:
                _backpressure = AdaptiveBackpressure()
                await _backpressure.start()
    return _backpressure


async def get_bulkhead(pools: Optional[Dict[str, int]] = None) -> ResourceBulkhead:
    """Get singleton ResourceBulkhead instance."""
    global _bulkhead
    if _bulkhead is None:
        async with _init_lock:
            if _bulkhead is None:
                _bulkhead = ResourceBulkhead(pools)
    return _bulkhead


async def get_resource_manager() -> AdaptiveResourceManager:
    """Get singleton AdaptiveResourceManager instance."""
    global _resource_manager
    if _resource_manager is None:
        async with _init_lock:
            if _resource_manager is None:
                _resource_manager = AdaptiveResourceManager()
                await _resource_manager.start()
    return _resource_manager


async def get_health_verifier() -> DeepHealthVerifier:
    """Get singleton DeepHealthVerifier instance."""
    global _health_verifier
    if _health_verifier is None:
        async with _init_lock:
            if _health_verifier is None:
                _health_verifier = DeepHealthVerifier()
    return _health_verifier


def get_rate_limiter(
    name: str = "default",
    rate: Optional[float] = None,
    burst: Optional[int] = None,
) -> TrinityRateLimiter:
    """Get or create a TrinityRateLimiter instance."""
    global _rate_limiter
    if _rate_limiter is None or name != "default":
        return TrinityRateLimiter(rate=rate, burst=burst, name=name)
    if _rate_limiter is None:
        _rate_limiter = TrinityRateLimiter(rate=rate, burst=burst, name=name)
    return _rate_limiter


async def get_atomic_ipc(base_dir: Optional[str] = None) -> AtomicFileIPC:
    """Get singleton AtomicFileIPC instance."""
    global _atomic_ipc
    if _atomic_ipc is None:
        async with _init_lock:
            if _atomic_ipc is None:
                _atomic_ipc = AtomicFileIPC(base_dir)
    return _atomic_ipc


async def shutdown_all() -> None:
    """Shutdown all singleton instances."""
    global _backpressure, _resource_manager, _health_verifier

    if _backpressure:
        await _backpressure.stop()
        _backpressure = None

    if _resource_manager:
        await _resource_manager.stop()
        _resource_manager = None

    if _health_verifier:
        await _health_verifier.close()
        _health_verifier = None


# =============================================================================
# DECORATORS
# =============================================================================


def with_timeout(
    timeout: Optional[float] = None,
    timeout_error_message: str = "Operation timed out",
):
    """
    Decorator to add timeout to async function.

    Args:
        timeout: Timeout in seconds (from env if None)
        timeout_error_message: Error message on timeout
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            effective_timeout = timeout or _env_float("DEFAULT_ASYNC_TIMEOUT", 30.0)
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"{timeout_error_message} after {effective_timeout}s")
        return wrapper
    return decorator


def with_bulkhead(
    pool_name: str,
    fallback: Optional[Callable[..., Awaitable[T]]] = None,
):
    """
    Decorator to execute function within a bulkhead pool.

    Args:
        pool_name: Name of the bulkhead pool
        fallback: Optional fallback function
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            bulkhead = await get_bulkhead()
            try:
                async with bulkhead.execute(pool_name):
                    return await func(*args, **kwargs)
            except (BulkheadCircuitOpen, BulkheadTimeout):
                if fallback:
                    return await fallback(*args, **kwargs)
                raise
        return wrapper
    return decorator


def with_backpressure():
    """Decorator to apply backpressure to async function."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            bp = await get_backpressure()
            async with bp.acquire_context():
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def with_rate_limit(
    rate: Optional[float] = None,
    burst: Optional[int] = None,
):
    """
    Decorator to rate limit async function.

    Args:
        rate: Max calls per second
        burst: Maximum burst size
    """
    limiter = TrinityRateLimiter(rate=rate, burst=burst)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not await limiter.acquire_or_wait():
                raise BackpressureRejection("Rate limit exceeded")
            return await func(*args, **kwargs)
        return wrapper
    return decorator
