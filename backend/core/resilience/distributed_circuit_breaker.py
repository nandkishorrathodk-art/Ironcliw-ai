"""
Distributed Circuit Breaker with Cross-Instance Coordination
=============================================================

Provides circuit breaker functionality that coordinates across multiple
instances via Redis for consistent failure handling.

Features:
    - Redis-backed state synchronization across instances
    - Exponential backoff with jitter for recovery attempts
    - Half-open state with probe requests
    - Sliding window failure tracking
    - Per-service and per-tier circuit breakers
    - Health aggregation across instances
    - Automatic recovery with gradual traffic ramp-up
    - Circuit breaker groups for related services

Theory:
    Circuit breakers prevent cascade failures by temporarily stopping
    requests to failing services. Distributed coordination ensures all
    instances agree on circuit state, preventing one instance from
    overwhelming a recovering service.

Usage:
    cb = await get_distributed_circuit_breaker()

    async with cb.call("service-name"):
        result = await external_service.call()

    # Or manual control
    if cb.allow_request("service-name"):
        try:
            result = await external_service.call()
            cb.record_success("service-name")
        except Exception as e:
            cb.record_failure("service-name", e)

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple, Type

logger = logging.getLogger("DistributedCircuitBreaker")


# =============================================================================
# Configuration
# =============================================================================

# Circuit breaker parameters
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
CB_SUCCESS_THRESHOLD = int(os.getenv("CB_SUCCESS_THRESHOLD", "3"))
CB_TIMEOUT_SECONDS = float(os.getenv("CB_TIMEOUT_SECONDS", "60.0"))
CB_HALF_OPEN_MAX_CALLS = int(os.getenv("CB_HALF_OPEN_MAX_CALLS", "3"))
CB_SLIDING_WINDOW_SIZE = int(os.getenv("CB_SLIDING_WINDOW_SIZE", "60"))  # seconds

# Recovery parameters
CB_MIN_RECOVERY_TIME = float(os.getenv("CB_MIN_RECOVERY_TIME", "5.0"))
CB_MAX_RECOVERY_TIME = float(os.getenv("CB_MAX_RECOVERY_TIME", "300.0"))
CB_RECOVERY_MULTIPLIER = float(os.getenv("CB_RECOVERY_MULTIPLIER", "2.0"))
CB_JITTER_FACTOR = float(os.getenv("CB_JITTER_FACTOR", "0.1"))

# Redis configuration
CB_REDIS_PREFIX = os.getenv("CB_REDIS_PREFIX", "circuit_breaker:")
CB_SYNC_INTERVAL = float(os.getenv("CB_SYNC_INTERVAL", "1.0"))
CB_STATE_TTL = int(os.getenv("CB_STATE_TTL", "600"))  # 10 minutes

# Instance identification
INSTANCE_ID = os.getenv("INSTANCE_ID", f"instance-{os.getpid()}-{random.randint(1000, 9999)}")


# =============================================================================
# Enums and Data Classes
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation, requests allowed
    OPEN = "open"           # Failures exceeded threshold, requests blocked
    HALF_OPEN = "half_open"  # Recovery testing, limited requests allowed


class FailureCategory(Enum):
    """Categories of failures for intelligent handling."""
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN = "unknown"


@dataclass
class CircuitConfig:
    """Configuration for a circuit breaker."""
    failure_threshold: int = CB_FAILURE_THRESHOLD
    success_threshold: int = CB_SUCCESS_THRESHOLD
    timeout_seconds: float = CB_TIMEOUT_SECONDS
    half_open_max_calls: int = CB_HALF_OPEN_MAX_CALLS
    sliding_window_size: int = CB_SLIDING_WINDOW_SIZE
    min_recovery_time: float = CB_MIN_RECOVERY_TIME
    max_recovery_time: float = CB_MAX_RECOVERY_TIME
    recovery_multiplier: float = CB_RECOVERY_MULTIPLIER
    excluded_exceptions: Set[Type[Exception]] = field(default_factory=set)


@dataclass
class CircuitMetrics:
    """Metrics for a circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    last_state_change: float = 0.0
    current_recovery_time: float = CB_MIN_RECOVERY_TIME
    consecutive_successes: int = 0
    consecutive_failures: int = 0


@dataclass
class DistributedState:
    """Distributed state shared via Redis."""
    service_name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure: float = 0.0
    last_success: float = 0.0
    opened_at: float = 0.0
    recovery_deadline: float = 0.0
    half_open_calls: int = 0
    instance_states: Dict[str, str] = field(default_factory=dict)
    version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure,
            "last_success": self.last_success,
            "opened_at": self.opened_at,
            "recovery_deadline": self.recovery_deadline,
            "half_open_calls": self.half_open_calls,
            "instance_states": self.instance_states,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DistributedState":
        """Create from dictionary."""
        return cls(
            service_name=data["service_name"],
            state=CircuitState(data["state"]),
            failure_count=data.get("failure_count", 0),
            success_count=data.get("success_count", 0),
            last_failure=data.get("last_failure", 0.0),
            last_success=data.get("last_success", 0.0),
            opened_at=data.get("opened_at", 0.0),
            recovery_deadline=data.get("recovery_deadline", 0.0),
            half_open_calls=data.get("half_open_calls", 0),
            instance_states=data.get("instance_states", {}),
            version=data.get("version", 0),
        )


# =============================================================================
# Sliding Window Failure Tracker
# =============================================================================

class SlidingWindowTracker:
    """
    Tracks failures in a sliding time window.

    More accurate than simple counters as it only counts recent failures.
    """

    def __init__(self, window_size: int = CB_SLIDING_WINDOW_SIZE):
        self._window_size = window_size
        self._failures: List[Tuple[float, FailureCategory]] = []
        self._successes: List[float] = []
        self._lock = asyncio.Lock()

    async def record_failure(self, category: FailureCategory = FailureCategory.UNKNOWN) -> None:
        """Record a failure."""
        async with self._lock:
            now = time.time()
            self._failures.append((now, category))
            self._cleanup()

    async def record_success(self) -> None:
        """Record a success."""
        async with self._lock:
            now = time.time()
            self._successes.append(now)
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove old entries outside the window."""
        cutoff = time.time() - self._window_size
        self._failures = [(t, c) for t, c in self._failures if t > cutoff]
        self._successes = [t for t in self._successes if t > cutoff]

    async def get_failure_count(self) -> int:
        """Get failure count in the current window."""
        async with self._lock:
            self._cleanup()
            return len(self._failures)

    async def get_success_count(self) -> int:
        """Get success count in the current window."""
        async with self._lock:
            self._cleanup()
            return len(self._successes)

    async def get_failure_rate(self) -> float:
        """Get failure rate (failures / total calls)."""
        async with self._lock:
            self._cleanup()
            total = len(self._failures) + len(self._successes)
            if total == 0:
                return 0.0
            return len(self._failures) / total

    async def get_failure_categories(self) -> Dict[FailureCategory, int]:
        """Get breakdown of failures by category."""
        async with self._lock:
            self._cleanup()
            categories: Dict[FailureCategory, int] = {}
            for _, cat in self._failures:
                categories[cat] = categories.get(cat, 0) + 1
            return categories


# =============================================================================
# Circuit Breaker Instance
# =============================================================================

class CircuitBreaker:
    """
    Individual circuit breaker for a service.

    Manages local state and coordinates with distributed state.
    """

    def __init__(
        self,
        service_name: str,
        config: Optional[CircuitConfig] = None,
        redis_client: Optional[Any] = None,
    ):
        self._service_name = service_name
        self._config = config or CircuitConfig()
        self._redis = redis_client

        # Local state
        self._state = CircuitState.CLOSED
        self._tracker = SlidingWindowTracker(self._config.sliding_window_size)
        self._metrics = CircuitMetrics()

        # Recovery management
        self._recovery_deadline = 0.0
        self._current_recovery_time = self._config.min_recovery_time
        self._half_open_permits = 0

        # Synchronization
        self._state_lock = asyncio.Lock()
        self._distributed_state: Optional[DistributedState] = None

        logger.debug(f"CircuitBreaker created for {service_name}")

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def service_name(self) -> str:
        """Service name."""
        return self._service_name

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    async def allow_request(self) -> bool:
        """
        Check if a request should be allowed.

        Returns True if request should proceed, False if blocked.
        """
        async with self._state_lock:
            await self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                self._metrics.rejected_calls += 1
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_permits < self._config.half_open_max_calls:
                    self._half_open_permits += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._state_lock:
            self._metrics.total_calls += 1
            self._metrics.successful_calls += 1
            self._metrics.consecutive_successes += 1
            self._metrics.consecutive_failures = 0
            self._metrics.last_success_time = time.time()

            await self._tracker.record_success()

            if self._state == CircuitState.HALF_OPEN:
                if self._metrics.consecutive_successes >= self._config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    self._current_recovery_time = self._config.min_recovery_time

    async def record_failure(
        self,
        exception: Optional[Exception] = None,
        category: FailureCategory = FailureCategory.UNKNOWN,
    ) -> None:
        """Record a failed call."""
        # Check if exception should be excluded
        if exception and type(exception) in self._config.excluded_exceptions:
            return

        # Categorize the failure if exception provided
        if exception and category == FailureCategory.UNKNOWN:
            category = self._categorize_exception(exception)

        async with self._state_lock:
            self._metrics.total_calls += 1
            self._metrics.failed_calls += 1
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
            self._metrics.last_failure_time = time.time()

            await self._tracker.record_failure(category)

            failure_count = await self._tracker.get_failure_count()

            if self._state == CircuitState.CLOSED:
                if failure_count >= self._config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                await self._transition_to(CircuitState.OPEN)
                # Increase recovery time (exponential backoff)
                self._current_recovery_time = min(
                    self._current_recovery_time * self._config.recovery_multiplier,
                    self._config.max_recovery_time,
                )

    def _categorize_exception(self, exception: Exception) -> FailureCategory:
        """Categorize an exception into a failure category."""
        exc_name = type(exception).__name__.lower()

        if "timeout" in exc_name or "timedout" in exc_name:
            return FailureCategory.TIMEOUT
        if "connection" in exc_name or "connect" in exc_name:
            return FailureCategory.CONNECTION
        if "429" in str(exception) or "rate" in exc_name:
            return FailureCategory.RESOURCE_EXHAUSTED
        if "500" in str(exception) or "502" in str(exception) or "503" in str(exception):
            return FailureCategory.SERVER_ERROR
        if "400" in str(exception) or "401" in str(exception) or "403" in str(exception):
            return FailureCategory.CLIENT_ERROR

        return FailureCategory.UNKNOWN

    async def _check_state_transition(self) -> None:
        """Check if state should transition based on time."""
        if self._state == CircuitState.OPEN:
            if time.time() >= self._recovery_deadline:
                await self._transition_to(CircuitState.HALF_OPEN)

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._metrics.state_changes += 1
        self._metrics.last_state_change = time.time()

        if new_state == CircuitState.OPEN:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(-CB_JITTER_FACTOR, CB_JITTER_FACTOR)
            recovery_time = self._current_recovery_time * (1 + jitter)
            self._recovery_deadline = time.time() + recovery_time
            self._metrics.current_recovery_time = recovery_time

        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_permits = 0
            self._metrics.consecutive_successes = 0

        elif new_state == CircuitState.CLOSED:
            self._half_open_permits = 0

        logger.info(
            f"Circuit '{self._service_name}' transitioned: {old_state.value} -> {new_state.value}"
        )

        # Sync to Redis if available
        await self._sync_to_redis()

    async def _sync_to_redis(self) -> None:
        """Sync local state to Redis."""
        if not self._redis:
            return

        try:
            key = f"{CB_REDIS_PREFIX}{self._service_name}"

            # Get current distributed state
            state_json = await self._redis.get(key)
            if state_json:
                dist_state = DistributedState.from_dict(json.loads(state_json))
            else:
                dist_state = DistributedState(service_name=self._service_name)

            # Update with our instance state
            dist_state.instance_states[INSTANCE_ID] = self._state.value
            dist_state.version += 1

            # Consensus: if majority of instances are open, keep open
            open_count = sum(1 for s in dist_state.instance_states.values() if s == "open")
            total_instances = len(dist_state.instance_states)

            if total_instances > 1 and open_count > total_instances / 2:
                if self._state != CircuitState.OPEN:
                    dist_state.state = CircuitState.OPEN
            else:
                dist_state.state = self._state

            await self._redis.set(key, json.dumps(dist_state.to_dict()), ex=CB_STATE_TTL)
            self._distributed_state = dist_state

        except Exception as e:
            logger.warning(f"Failed to sync circuit state to Redis: {e}")

    async def sync_from_redis(self) -> None:
        """Sync distributed state from Redis."""
        if not self._redis:
            return

        try:
            key = f"{CB_REDIS_PREFIX}{self._service_name}"
            state_json = await self._redis.get(key)

            if state_json:
                dist_state = DistributedState.from_dict(json.loads(state_json))
                self._distributed_state = dist_state

                # If distributed state is open and we're not, respect majority
                open_count = sum(1 for s in dist_state.instance_states.values() if s == "open")
                total_instances = len(dist_state.instance_states)

                if total_instances > 1 and open_count > total_instances / 2:
                    if self._state != CircuitState.OPEN:
                        async with self._state_lock:
                            await self._transition_to(CircuitState.OPEN)

        except Exception as e:
            logger.warning(f"Failed to sync circuit state from Redis: {e}")

    @asynccontextmanager
    async def call(self) -> AsyncIterator[None]:
        """
        Context manager for circuit-protected calls.

        Raises CircuitOpenError if circuit is open.
        """
        if not await self.allow_request():
            raise CircuitOpenError(
                f"Circuit '{self._service_name}' is open",
                service=self._service_name,
                recovery_deadline=self._recovery_deadline,
            )

        try:
            yield
            await self.record_success()
        except Exception as e:
            await self.record_failure(e)
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "service_name": self._service_name,
            "state": self._state.value,
            "total_calls": self._metrics.total_calls,
            "successful_calls": self._metrics.successful_calls,
            "failed_calls": self._metrics.failed_calls,
            "rejected_calls": self._metrics.rejected_calls,
            "state_changes": self._metrics.state_changes,
            "consecutive_successes": self._metrics.consecutive_successes,
            "consecutive_failures": self._metrics.consecutive_failures,
            "current_recovery_time": self._metrics.current_recovery_time,
            "recovery_deadline": self._recovery_deadline,
            "last_failure": self._metrics.last_failure_time,
            "last_success": self._metrics.last_success_time,
        }


# =============================================================================
# Circuit Open Exception
# =============================================================================

class CircuitOpenError(Exception):
    """Exception raised when circuit is open."""

    def __init__(
        self,
        message: str,
        service: str,
        recovery_deadline: float,
    ):
        super().__init__(message)
        self.service = service
        self.recovery_deadline = recovery_deadline

    @property
    def seconds_until_retry(self) -> float:
        """Seconds until circuit might be half-open."""
        return max(0.0, self.recovery_deadline - time.time())


# =============================================================================
# Circuit Breaker Group
# =============================================================================

class CircuitBreakerGroup:
    """
    Group of related circuit breakers.

    Useful for treating a group of services as a single unit
    (e.g., all GCP services, all database connections).
    """

    def __init__(
        self,
        name: str,
        breakers: Optional[List[CircuitBreaker]] = None,
    ):
        self._name = name
        self._breakers = breakers or []

    def add(self, breaker: CircuitBreaker) -> None:
        """Add a circuit breaker to the group."""
        self._breakers.append(breaker)

    def remove(self, service_name: str) -> bool:
        """Remove a circuit breaker from the group."""
        for i, cb in enumerate(self._breakers):
            if cb.service_name == service_name:
                del self._breakers[i]
                return True
        return False

    @property
    def all_closed(self) -> bool:
        """Check if all circuits in the group are closed."""
        return all(cb.is_closed for cb in self._breakers)

    @property
    def any_open(self) -> bool:
        """Check if any circuit in the group is open."""
        return any(cb.is_open for cb in self._breakers)

    @property
    def health_percentage(self) -> float:
        """Get health percentage (closed circuits / total)."""
        if not self._breakers:
            return 100.0
        closed = sum(1 for cb in self._breakers if cb.is_closed)
        return (closed / len(self._breakers)) * 100

    async def record_success(self, service_name: Optional[str] = None) -> None:
        """Record success for a specific service or all."""
        for cb in self._breakers:
            if service_name is None or cb.service_name == service_name:
                await cb.record_success()

    async def record_failure(
        self,
        exception: Optional[Exception] = None,
        service_name: Optional[str] = None,
    ) -> None:
        """Record failure for a specific service or all."""
        for cb in self._breakers:
            if service_name is None or cb.service_name == service_name:
                await cb.record_failure(exception)

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all breakers in the group."""
        return {
            "name": self._name,
            "health_percentage": self.health_percentage,
            "all_closed": self.all_closed,
            "any_open": self.any_open,
            "breakers": [cb.get_metrics() for cb in self._breakers],
        }


# =============================================================================
# Distributed Circuit Breaker Manager
# =============================================================================

class DistributedCircuitBreakerManager:
    """
    Manager for distributed circuit breakers.

    Handles creation, coordination, and synchronization of circuit breakers.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        default_config: Optional[CircuitConfig] = None,
    ):
        self._redis = redis_client
        self._default_config = default_config or CircuitConfig()

        # Circuit breakers by service
        self._breakers: Dict[str, CircuitBreaker] = {}

        # Groups
        self._groups: Dict[str, CircuitBreakerGroup] = {}

        # Synchronization
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._breaker_lock = asyncio.Lock()

        # Metrics
        self._metrics = {
            "total_breakers": 0,
            "open_breakers": 0,
            "half_open_breakers": 0,
            "closed_breakers": 0,
            "sync_operations": 0,
            "sync_failures": 0,
        }

        logger.info("DistributedCircuitBreakerManager initialized")

    async def start(self) -> None:
        """Start the manager and sync task."""
        self._running = True

        if self._redis:
            self._sync_task = asyncio.create_task(
                self._sync_loop(),
                name="circuit_breaker_sync",
            )

        logger.info("DistributedCircuitBreakerManager started")

    async def stop(self) -> None:
        """Stop the manager."""
        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info("DistributedCircuitBreakerManager stopped")

    async def get_or_create(
        self,
        service_name: str,
        config: Optional[CircuitConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        async with self._breaker_lock:
            if service_name not in self._breakers:
                self._breakers[service_name] = CircuitBreaker(
                    service_name=service_name,
                    config=config or self._default_config,
                    redis_client=self._redis,
                )
                self._metrics["total_breakers"] += 1

            return self._breakers[service_name]

    async def get(self, service_name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker if it exists."""
        return self._breakers.get(service_name)

    async def remove(self, service_name: str) -> bool:
        """Remove a circuit breaker."""
        async with self._breaker_lock:
            if service_name in self._breakers:
                del self._breakers[service_name]
                self._metrics["total_breakers"] -= 1
                return True
            return False

    def create_group(self, name: str, services: List[str]) -> CircuitBreakerGroup:
        """Create a group of circuit breakers."""
        breakers = [self._breakers[s] for s in services if s in self._breakers]
        group = CircuitBreakerGroup(name, breakers)
        self._groups[name] = group
        return group

    def get_group(self, name: str) -> Optional[CircuitBreakerGroup]:
        """Get a circuit breaker group."""
        return self._groups.get(name)

    async def allow_request(self, service_name: str) -> bool:
        """Check if a request to a service should be allowed."""
        cb = await self.get_or_create(service_name)
        return await cb.allow_request()

    async def record_success(self, service_name: str) -> None:
        """Record a successful call to a service."""
        cb = await self.get_or_create(service_name)
        await cb.record_success()

    async def record_failure(
        self,
        service_name: str,
        exception: Optional[Exception] = None,
    ) -> None:
        """Record a failed call to a service."""
        cb = await self.get_or_create(service_name)
        await cb.record_failure(exception)

    @asynccontextmanager
    async def call(self, service_name: str) -> AsyncIterator[None]:
        """Context manager for circuit-protected calls."""
        cb = await self.get_or_create(service_name)
        async with cb.call():
            yield

    async def _sync_loop(self) -> None:
        """Background loop to sync with Redis."""
        while self._running:
            try:
                await asyncio.sleep(CB_SYNC_INTERVAL)

                # Sync all breakers from Redis
                for cb in self._breakers.values():
                    await cb.sync_from_redis()

                self._metrics["sync_operations"] += 1

                # Update aggregate metrics
                self._metrics["open_breakers"] = sum(
                    1 for cb in self._breakers.values() if cb.is_open
                )
                self._metrics["half_open_breakers"] = sum(
                    1 for cb in self._breakers.values() if cb.is_half_open
                )
                self._metrics["closed_breakers"] = sum(
                    1 for cb in self._breakers.values() if cb.is_closed
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Circuit breaker sync error: {e}")
                self._metrics["sync_failures"] += 1
                await asyncio.sleep(5)

    def get_metrics(self) -> Dict[str, Any]:
        """Get manager metrics."""
        return {
            **self._metrics,
            "breakers": {
                name: cb.get_metrics()
                for name, cb in self._breakers.items()
            },
            "groups": {
                name: group.get_metrics()
                for name, group in self._groups.items()
            },
        }

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all circuits."""
        total = len(self._breakers)
        open_count = sum(1 for cb in self._breakers.values() if cb.is_open)
        half_open_count = sum(1 for cb in self._breakers.values() if cb.is_half_open)
        closed_count = sum(1 for cb in self._breakers.values() if cb.is_closed)

        health_percentage = (closed_count / total * 100) if total > 0 else 100.0

        return {
            "total": total,
            "open": open_count,
            "half_open": half_open_count,
            "closed": closed_count,
            "health_percentage": health_percentage,
            "status": "healthy" if health_percentage > 80 else "degraded" if health_percentage > 50 else "critical",
        }


# =============================================================================
# Global Factory
# =============================================================================

_manager_instance: Optional[DistributedCircuitBreakerManager] = None
_manager_lock = asyncio.Lock()


async def get_distributed_circuit_breaker(
    redis_client: Optional[Any] = None,
    config: Optional[CircuitConfig] = None,
) -> DistributedCircuitBreakerManager:
    """Get or create the global DistributedCircuitBreakerManager instance."""
    global _manager_instance

    async with _manager_lock:
        if _manager_instance is None:
            _manager_instance = DistributedCircuitBreakerManager(
                redis_client=redis_client,
                default_config=config,
            )
            await _manager_instance.start()

        return _manager_instance


async def shutdown_distributed_circuit_breaker() -> None:
    """Shutdown the global circuit breaker manager."""
    global _manager_instance

    if _manager_instance:
        await _manager_instance.stop()
        _manager_instance = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerGroup",
    "DistributedCircuitBreakerManager",
    "CircuitConfig",
    "CircuitState",
    "FailureCategory",
    "CircuitOpenError",
    "SlidingWindowTracker",
    "get_distributed_circuit_breaker",
    "shutdown_distributed_circuit_breaker",
]
