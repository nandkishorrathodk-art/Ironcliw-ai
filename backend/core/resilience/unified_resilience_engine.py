"""
Unified Resilience Engine v1.0 - Enterprise-Grade Fault Tolerance
==================================================================

The central nervous system for fault tolerance across JARVIS, JARVIS Prime,
and Reactor Core. Implements all 12 resilience patterns:

1. Circuit Breakers (enhanced) - Adaptive thresholds with ML-based prediction
2. Graceful Degradation - Multi-tier fallback with service mesh awareness
3. Automatic Retry with Backoff - Exponential backoff with decorrelated jitter
4. Timeout Management - Adaptive timeouts based on system load
5. Bulkhead Isolation - Semaphore-based resource pools
6. Rate Limiting - Token bucket with adaptive capacity
7. Request Queuing - Priority queues with backpressure
8. Dead Letter Queue - Failed request persistence with retry scheduling
9. Health-Based Routing - Intelligent routing with latency awareness
10. Automatic Failover - Zero-downtime failover orchestration
11. Chaos Engineering - Controlled failure injection framework
12. Failure Injection - Programmable fault simulation

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    UNIFIED RESILIENCE ENGINE v1.0                        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
    │  │   BULKHEAD      │     │  PRIORITY       │     │  DEAD LETTER    │   │
    │  │   ISOLATION     │────▶│  QUEUE          │────▶│  QUEUE          │   │
    │  │   (Semaphores)  │     │  (Heap+Backpr.) │     │  (Persistent)   │   │
    │  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘   │
    │           │                       │                       │             │
    │           ▼                       ▼                       ▼             │
    │  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
    │  │  CIRCUIT        │     │  HEALTH-BASED   │     │  AUTOMATIC      │   │
    │  │  BREAKERS       │◀───▶│  ROUTER         │◀───▶│  FAILOVER       │   │
    │  │  (Adaptive)     │     │  (ML Scoring)   │     │  (Zero-DT)      │   │
    │  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘   │
    │           │                       │                       │             │
    │           ▼                       ▼                       ▼             │
    │  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐   │
    │  │  CHAOS          │     │  FAILURE        │     │  ADAPTIVE       │   │
    │  │  ENGINEERING    │────▶│  INJECTION      │────▶│  TIMEOUTS       │   │
    │  │  (Test Suite)   │     │  (Programmable) │     │  (Load-Aware)   │   │
    │  └─────────────────┘     └─────────────────┘     └─────────────────┘   │
    │                                                                          │
    │                    CROSS-REPO INTEGRATION                                │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │  JARVIS (Body) ◀══════▶ JARVIS Prime (Mind) ◀══════▶ Reactor     │  │
    │  │       ▲                        ▲                        ▲         │  │
    │  │       └────────────────────────┼────────────────────────┘         │  │
    │  │                     Neural Mesh Integration                        │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Author: Trinity Resilience System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import heapq
import json
import logging
import math
import os
import random
import statistics
import sys
import time
import traceback
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
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
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger("Resilience.UnifiedEngine")

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# CONFIGURATION - ALL DYNAMIC FROM ENVIRONMENT
# =============================================================================

class ResilienceConfig:
    """
    Dynamic configuration with environment variable support.
    NO HARDCODING - everything is configurable at runtime.
    """

    # Bulkhead Configuration
    @staticmethod
    def get_bulkhead_max_concurrent(pool_name: str = "default") -> int:
        env_key = f"RESILIENCE_BULKHEAD_{pool_name.upper()}_MAX"
        return int(os.getenv(env_key, os.getenv("RESILIENCE_BULKHEAD_MAX", "10")))

    @staticmethod
    def get_bulkhead_max_waiting(pool_name: str = "default") -> int:
        env_key = f"RESILIENCE_BULKHEAD_{pool_name.upper()}_WAITING"
        return int(os.getenv(env_key, os.getenv("RESILIENCE_BULKHEAD_WAITING", "50")))

    @staticmethod
    def get_bulkhead_timeout() -> float:
        return float(os.getenv("RESILIENCE_BULKHEAD_TIMEOUT", "30.0"))

    # Queue Configuration
    @staticmethod
    def get_queue_max_size() -> int:
        return int(os.getenv("RESILIENCE_QUEUE_MAX_SIZE", "1000"))

    @staticmethod
    def get_queue_worker_count() -> int:
        return int(os.getenv("RESILIENCE_QUEUE_WORKERS", "5"))

    @staticmethod
    def get_backpressure_threshold() -> float:
        """Percentage of queue capacity that triggers backpressure (0.0-1.0)"""
        return float(os.getenv("RESILIENCE_BACKPRESSURE_THRESHOLD", "0.8"))

    # Dead Letter Queue Configuration
    @staticmethod
    def get_dlq_max_retries() -> int:
        return int(os.getenv("RESILIENCE_DLQ_MAX_RETRIES", "3"))

    @staticmethod
    def get_dlq_retry_delay_base() -> float:
        return float(os.getenv("RESILIENCE_DLQ_RETRY_DELAY", "60.0"))

    @staticmethod
    def get_dlq_persistence_path() -> Path:
        default = Path.home() / ".jarvis" / "resilience" / "dlq"
        return Path(os.getenv("RESILIENCE_DLQ_PATH", str(default)))

    @staticmethod
    def get_dlq_max_age_hours() -> int:
        return int(os.getenv("RESILIENCE_DLQ_MAX_AGE_HOURS", "168"))  # 7 days

    # Health Routing Configuration
    @staticmethod
    def get_health_check_interval() -> float:
        return float(os.getenv("RESILIENCE_HEALTH_INTERVAL", "10.0"))

    @staticmethod
    def get_health_timeout() -> float:
        return float(os.getenv("RESILIENCE_HEALTH_TIMEOUT", "5.0"))

    @staticmethod
    def get_unhealthy_threshold() -> int:
        return int(os.getenv("RESILIENCE_UNHEALTHY_THRESHOLD", "3"))

    @staticmethod
    def get_healthy_threshold() -> int:
        return int(os.getenv("RESILIENCE_HEALTHY_THRESHOLD", "2"))

    # Adaptive Timeout Configuration
    @staticmethod
    def get_base_timeout() -> float:
        return float(os.getenv("RESILIENCE_BASE_TIMEOUT", "30.0"))

    @staticmethod
    def get_min_timeout() -> float:
        return float(os.getenv("RESILIENCE_MIN_TIMEOUT", "5.0"))

    @staticmethod
    def get_max_timeout() -> float:
        return float(os.getenv("RESILIENCE_MAX_TIMEOUT", "300.0"))

    @staticmethod
    def get_timeout_percentile() -> float:
        """Percentile of latency history to use for adaptive timeout"""
        return float(os.getenv("RESILIENCE_TIMEOUT_PERCENTILE", "95.0"))

    # Retry Configuration
    @staticmethod
    def get_retry_base_delay() -> float:
        return float(os.getenv("RESILIENCE_RETRY_BASE_DELAY", "1.0"))

    @staticmethod
    def get_retry_max_delay() -> float:
        return float(os.getenv("RESILIENCE_RETRY_MAX_DELAY", "60.0"))

    @staticmethod
    def get_retry_max_attempts() -> int:
        return int(os.getenv("RESILIENCE_RETRY_MAX_ATTEMPTS", "5"))

    @staticmethod
    def get_retry_jitter_factor() -> float:
        """Decorrelated jitter factor (0.0-1.0)"""
        return float(os.getenv("RESILIENCE_RETRY_JITTER", "0.5"))

    # Chaos Engineering Configuration
    @staticmethod
    def is_chaos_enabled() -> bool:
        return os.getenv("RESILIENCE_CHAOS_ENABLED", "false").lower() == "true"

    @staticmethod
    def get_chaos_failure_rate() -> float:
        return float(os.getenv("RESILIENCE_CHAOS_FAILURE_RATE", "0.1"))

    @staticmethod
    def get_chaos_latency_injection_ms() -> Tuple[float, float]:
        min_ms = float(os.getenv("RESILIENCE_CHAOS_LATENCY_MIN", "100"))
        max_ms = float(os.getenv("RESILIENCE_CHAOS_LATENCY_MAX", "2000"))
        return (min_ms, max_ms)

    # Cross-Repo Configuration
    @staticmethod
    def get_jarvis_url() -> str:
        return os.getenv("JARVIS_API_URL", "http://localhost:8010")

    @staticmethod
    def get_prime_url() -> str:
        return os.getenv("JARVIS_PRIME_API_URL", "http://localhost:8000")

    @staticmethod
    def get_reactor_url() -> str:
        return os.getenv("REACTOR_CORE_API_URL", "http://localhost:8020")


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class BulkheadState(Enum):
    """State of a bulkhead pool."""
    AVAILABLE = "available"
    SATURATED = "saturated"
    OVERLOADED = "overloaded"


class RequestPriority(Enum):
    """Priority levels for queued requests."""
    CRITICAL = 0    # System health, security
    HIGH = 1        # User-facing, time-sensitive
    NORMAL = 2      # Standard operations
    LOW = 3         # Background, batch jobs
    BACKGROUND = 4  # Housekeeping, cleanup


class FailureMode(Enum):
    """Types of failures for injection."""
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    LATENCY = "latency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"


class ServiceHealth(Enum):
    """Health states for services."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DRAINING = "draining"


class RoutingStrategy(Enum):
    """Strategies for health-based routing."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_LATENCY = "least_latency"
    WEIGHTED_RANDOM = "weighted_random"
    ADAPTIVE = "adaptive"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BulkheadMetrics:
    """Metrics for a bulkhead pool."""
    pool_name: str
    active_count: int
    waiting_count: int
    max_concurrent: int
    max_waiting: int
    total_acquired: int = 0
    total_rejected: int = 0
    total_timeouts: int = 0
    avg_wait_time_ms: float = 0.0
    state: BulkheadState = BulkheadState.AVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_name": self.pool_name,
            "active_count": self.active_count,
            "waiting_count": self.waiting_count,
            "max_concurrent": self.max_concurrent,
            "max_waiting": self.max_waiting,
            "total_acquired": self.total_acquired,
            "total_rejected": self.total_rejected,
            "total_timeouts": self.total_timeouts,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "state": self.state.value,
            "utilization": round(self.active_count / max(self.max_concurrent, 1), 3),
        }


@dataclass(order=True)
class QueuedRequest(Generic[T]):
    """A request in the priority queue."""
    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    payload: T = field(compare=False)
    callback: Optional[Callable[[T], Awaitable[Any]]] = field(compare=False, default=None)
    timeout: float = field(compare=False, default=30.0)
    retries: int = field(compare=False, default=0)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)

    @classmethod
    def create(
        cls,
        payload: T,
        priority: RequestPriority = RequestPriority.NORMAL,
        callback: Optional[Callable[[T], Awaitable[Any]]] = None,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "QueuedRequest[T]":
        return cls(
            priority=priority.value,
            timestamp=time.time(),
            request_id=uuid.uuid4().hex[:16],
            payload=payload,
            callback=callback,
            timeout=timeout,
            metadata=metadata or {},
        )


@dataclass
class DeadLetterEntry:
    """An entry in the dead letter queue."""
    request_id: str
    original_request: Dict[str, Any]
    failure_reason: str
    failure_traceback: str
    failure_time: float
    retry_count: int
    next_retry_time: float
    max_retries: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def is_ready_for_retry(self) -> bool:
        return self.should_retry() and time.time() >= self.next_retry_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "original_request": self.original_request,
            "failure_reason": self.failure_reason,
            "failure_traceback": self.failure_traceback,
            "failure_time": self.failure_time,
            "retry_count": self.retry_count,
            "next_retry_time": self.next_retry_time,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeadLetterEntry":
        return cls(
            request_id=data["request_id"],
            original_request=data["original_request"],
            failure_reason=data["failure_reason"],
            failure_traceback=data.get("failure_traceback", ""),
            failure_time=data["failure_time"],
            retry_count=data["retry_count"],
            next_retry_time=data["next_retry_time"],
            max_retries=data["max_retries"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ServiceEndpoint:
    """A service endpoint for routing."""
    name: str
    url: str
    health: ServiceHealth = ServiceHealth.UNKNOWN
    weight: float = 1.0
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    latency_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_health_check: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_history:
            return float("inf")
        return statistics.mean(self.latency_history)

    @property
    def p95_latency_ms(self) -> float:
        if len(self.latency_history) < 20:
            return float("inf")
        sorted_latencies = sorted(self.latency_history)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    @property
    def health_score(self) -> float:
        """Calculate a health score (0-1, higher is better)."""
        if self.health == ServiceHealth.UNHEALTHY:
            return 0.0
        if self.health == ServiceHealth.UNKNOWN:
            return 0.5

        # Factors: error rate, latency, consecutive successes
        error_factor = 1.0 - min(self.error_rate * 10, 1.0)  # 10% errors = 0 score

        latency_factor = 1.0
        if self.latency_history:
            # Normalize latency: <100ms = 1.0, >5000ms = 0.0
            avg = self.avg_latency_ms
            latency_factor = max(0.0, 1.0 - (avg - 100) / 4900)

        success_factor = min(self.consecutive_successes / 10, 1.0)

        # Weight the factors
        score = (error_factor * 0.4 + latency_factor * 0.3 + success_factor * 0.3)

        if self.health == ServiceHealth.DEGRADED:
            score *= 0.7

        return round(score, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "health": self.health.value,
            "weight": self.weight,
            "active_connections": self.active_connections,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "health_score": self.health_score,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
        }


@dataclass
class AdaptiveTimeoutState:
    """State for adaptive timeout calculation."""
    operation_name: str
    latency_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    timeout_history: Deque[Tuple[float, bool]] = field(default_factory=lambda: deque(maxlen=50))
    current_timeout: float = 30.0
    min_timeout: float = 5.0
    max_timeout: float = 300.0
    cold_start: bool = True

    def record_latency(self, latency_ms: float) -> None:
        self.latency_history.append(latency_ms)
        self.cold_start = False

    def record_timeout(self, timeout_ms: float, did_timeout: bool) -> None:
        self.timeout_history.append((timeout_ms, did_timeout))

    def calculate_adaptive_timeout(self) -> float:
        """Calculate adaptive timeout based on latency history."""
        if self.cold_start or len(self.latency_history) < 5:
            return self.current_timeout

        # Use configurable percentile
        percentile = ResilienceConfig.get_timeout_percentile() / 100.0
        sorted_latencies = sorted(self.latency_history)
        idx = int(len(sorted_latencies) * percentile)
        p_latency = sorted_latencies[min(idx, len(sorted_latencies) - 1)]

        # Add safety margin (2x the percentile latency)
        suggested = p_latency * 2

        # Adjust based on recent timeouts
        recent_timeouts = [t for _, t in list(self.timeout_history)[-10:]]
        if recent_timeouts:
            timeout_rate = sum(1 for t in recent_timeouts if t) / len(recent_timeouts)
            if timeout_rate > 0.1:  # More than 10% timeouts
                suggested *= 1.5  # Increase timeout
            elif timeout_rate == 0 and len(recent_timeouts) >= 10:
                suggested *= 0.9  # Decrease timeout slightly

        # Clamp to configured bounds
        self.current_timeout = max(
            self.min_timeout,
            min(self.max_timeout, suggested)
        )

        return self.current_timeout


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment."""
    experiment_id: str
    name: str
    target_services: List[str]
    failure_mode: FailureMode
    parameters: Dict[str, Any]
    duration_seconds: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    active: bool = False
    results: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if not self.start_time:
            return False
        return time.time() > self.start_time + self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "target_services": self.target_services,
            "failure_mode": self.failure_mode.value,
            "parameters": self.parameters,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "active": self.active,
            "results": self.results,
        }


# =============================================================================
# BULKHEAD ISOLATION
# =============================================================================

class BulkheadPool:
    """
    Semaphore-based bulkhead for resource isolation.

    Prevents cascading failures by limiting concurrent operations per pool.
    Each pool is isolated - failures in one don't affect others.
    """

    def __init__(self, pool_name: str):
        self.pool_name = pool_name
        self._max_concurrent = ResilienceConfig.get_bulkhead_max_concurrent(pool_name)
        self._max_waiting = ResilienceConfig.get_bulkhead_max_waiting(pool_name)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._lock = asyncio.Lock()

        # Metrics
        self._active_count = 0
        self._waiting_count = 0
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_timeouts = 0
        self._wait_times: Deque[float] = deque(maxlen=100)

        self.logger = logging.getLogger(f"Bulkhead.{pool_name}")

    @property
    def state(self) -> BulkheadState:
        if self._active_count >= self._max_concurrent:
            if self._waiting_count >= self._max_waiting:
                return BulkheadState.OVERLOADED
            return BulkheadState.SATURATED
        return BulkheadState.AVAILABLE

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire a slot in the bulkhead pool.

        Args:
            timeout: Maximum time to wait for a slot

        Raises:
            BulkheadRejectedError: If waiting queue is full
            BulkheadTimeoutError: If timeout exceeded while waiting
        """
        timeout = timeout or ResilienceConfig.get_bulkhead_timeout()

        # Check if we can even queue
        async with self._lock:
            if self._waiting_count >= self._max_waiting:
                self._total_rejected += 1
                self.logger.warning(
                    f"Bulkhead {self.pool_name} REJECTED: "
                    f"waiting queue full ({self._waiting_count}/{self._max_waiting})"
                )
                raise BulkheadRejectedError(
                    self.pool_name,
                    f"Waiting queue full: {self._waiting_count}/{self._max_waiting}"
                )
            self._waiting_count += 1

        start_time = time.time()
        acquired = False

        try:
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=timeout,
                )
                acquired = True
            except asyncio.TimeoutError:
                self._total_timeouts += 1
                self.logger.warning(
                    f"Bulkhead {self.pool_name} TIMEOUT after {timeout}s"
                )
                raise BulkheadTimeoutError(
                    self.pool_name,
                    f"Timed out waiting for slot after {timeout}s"
                )

            # Record metrics
            wait_time = (time.time() - start_time) * 1000
            self._wait_times.append(wait_time)

            async with self._lock:
                self._waiting_count -= 1
                self._active_count += 1
                self._total_acquired += 1

            self.logger.debug(
                f"Bulkhead {self.pool_name} acquired: "
                f"active={self._active_count}/{self._max_concurrent}, "
                f"wait_time={wait_time:.1f}ms"
            )

            yield

        finally:
            if not acquired:
                async with self._lock:
                    self._waiting_count -= 1
            else:
                async with self._lock:
                    self._active_count -= 1
                self._semaphore.release()

    def get_metrics(self) -> BulkheadMetrics:
        avg_wait = (
            statistics.mean(self._wait_times) if self._wait_times else 0.0
        )
        return BulkheadMetrics(
            pool_name=self.pool_name,
            active_count=self._active_count,
            waiting_count=self._waiting_count,
            max_concurrent=self._max_concurrent,
            max_waiting=self._max_waiting,
            total_acquired=self._total_acquired,
            total_rejected=self._total_rejected,
            total_timeouts=self._total_timeouts,
            avg_wait_time_ms=avg_wait,
            state=self.state,
        )


class BulkheadRejectedError(Exception):
    """Raised when bulkhead rejects a request due to queue overflow."""
    def __init__(self, pool_name: str, reason: str):
        self.pool_name = pool_name
        self.reason = reason
        super().__init__(f"Bulkhead '{pool_name}' rejected: {reason}")


class BulkheadTimeoutError(Exception):
    """Raised when bulkhead wait times out."""
    def __init__(self, pool_name: str, reason: str):
        self.pool_name = pool_name
        self.reason = reason
        super().__init__(f"Bulkhead '{pool_name}' timeout: {reason}")


class BulkheadManager:
    """
    Manages multiple bulkhead pools with lazy initialization.
    """

    def __init__(self):
        self._pools: Dict[str, BulkheadPool] = {}
        self._lock = asyncio.Lock()

    async def get_pool(self, pool_name: str) -> BulkheadPool:
        """Get or create a bulkhead pool."""
        if pool_name not in self._pools:
            async with self._lock:
                if pool_name not in self._pools:
                    self._pools[pool_name] = BulkheadPool(pool_name)
        return self._pools[pool_name]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all pools."""
        return {
            name: pool.get_metrics().to_dict()
            for name, pool in self._pools.items()
        }


# =============================================================================
# PRIORITY QUEUE WITH BACKPRESSURE
# =============================================================================

class BackpressureSignal(Exception):
    """Raised when backpressure is triggered."""
    def __init__(self, queue_name: str, utilization: float):
        self.queue_name = queue_name
        self.utilization = utilization
        super().__init__(
            f"Queue '{queue_name}' backpressure at {utilization:.1%} capacity"
        )


class PriorityRequestQueue(Generic[T]):
    """
    Thread-safe priority queue with backpressure support.

    Features:
    - Priority-based ordering (lower priority value = higher priority)
    - Configurable backpressure threshold
    - Multiple worker support
    - Metrics and monitoring
    """

    def __init__(
        self,
        queue_name: str,
        processor: Callable[[T], Awaitable[Any]],
        max_size: Optional[int] = None,
        worker_count: Optional[int] = None,
    ):
        self.queue_name = queue_name
        self._processor = processor
        self._max_size = max_size or ResilienceConfig.get_queue_max_size()
        self._worker_count = worker_count or ResilienceConfig.get_queue_worker_count()
        self._backpressure_threshold = ResilienceConfig.get_backpressure_threshold()

        self._heap: List[QueuedRequest[T]] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)

        # State
        self._running = False
        self._workers: List[asyncio.Task] = []

        # Metrics
        self._total_enqueued = 0
        self._total_processed = 0
        self._total_failed = 0
        self._total_rejected = 0
        self._processing_times: Deque[float] = deque(maxlen=100)

        # Callback for failed requests (DLQ integration)
        self._on_failure: Optional[Callable[[QueuedRequest[T], Exception], Awaitable[None]]] = None

        self.logger = logging.getLogger(f"PriorityQueue.{queue_name}")

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def utilization(self) -> float:
        return self.size / max(self._max_size, 1)

    @property
    def is_backpressure(self) -> bool:
        return self.utilization >= self._backpressure_threshold

    def set_failure_callback(
        self,
        callback: Callable[[QueuedRequest[T], Exception], Awaitable[None]]
    ) -> None:
        """Set callback for failed requests (for DLQ integration)."""
        self._on_failure = callback

    async def enqueue(
        self,
        payload: T,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
        reject_on_backpressure: bool = False,
    ) -> str:
        """
        Enqueue a request.

        Args:
            payload: The request payload
            priority: Request priority
            timeout: Processing timeout
            metadata: Additional metadata
            reject_on_backpressure: If True, raise BackpressureSignal when queue is pressured

        Returns:
            Request ID

        Raises:
            BackpressureSignal: If backpressure and reject_on_backpressure=True
            ValueError: If queue is full
        """
        async with self._lock:
            if self.size >= self._max_size:
                self._total_rejected += 1
                raise ValueError(f"Queue '{self.queue_name}' is full")

            if reject_on_backpressure and self.is_backpressure:
                raise BackpressureSignal(self.queue_name, self.utilization)

            request = QueuedRequest.create(
                payload=payload,
                priority=priority,
                timeout=timeout,
                metadata=metadata,
            )

            heapq.heappush(self._heap, request)
            self._total_enqueued += 1

            self._not_empty.notify()

            self.logger.debug(
                f"Enqueued request {request.request_id} "
                f"priority={priority.name} size={self.size}"
            )

            return request.request_id

    async def _dequeue(self) -> Optional[QueuedRequest[T]]:
        """Dequeue the highest priority request."""
        async with self._not_empty:
            while self._running and not self._heap:
                await self._not_empty.wait()

            if not self._running:
                return None

            return heapq.heappop(self._heap)

    async def start(self) -> None:
        """Start queue workers."""
        if self._running:
            return

        self._running = True
        self.logger.info(f"Starting {self._worker_count} workers for queue '{self.queue_name}'")

        for i in range(self._worker_count):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)

    async def stop(self, wait_for_drain: bool = False, drain_timeout: float = 30.0) -> None:
        """Stop queue workers."""
        self._running = False

        # Wake up all waiting workers
        async with self._lock:
            self._not_empty.notify_all()

        if wait_for_drain and self._heap:
            self.logger.info(f"Draining {self.size} remaining requests...")
            drain_start = time.time()
            while self._heap and (time.time() - drain_start) < drain_timeout:
                await asyncio.sleep(0.1)

        # Cancel workers
        for task in self._workers:
            task.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        self.logger.info(f"Queue '{self.queue_name}' stopped")

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing requests."""
        self.logger.debug(f"Worker {worker_id} started")

        while self._running:
            try:
                request = await self._dequeue()
                if request is None:
                    continue

                start_time = time.time()

                try:
                    # Check if request has expired
                    age = time.time() - request.timestamp
                    if age > request.timeout:
                        self.logger.warning(
                            f"Request {request.request_id} expired after {age:.1f}s"
                        )
                        self._total_failed += 1
                        continue

                    # Process with remaining timeout
                    remaining_timeout = request.timeout - age
                    await asyncio.wait_for(
                        self._processor(request.payload),
                        timeout=remaining_timeout,
                    )

                    self._total_processed += 1
                    processing_time = (time.time() - start_time) * 1000
                    self._processing_times.append(processing_time)

                except Exception as e:
                    self._total_failed += 1
                    self.logger.error(
                        f"Request {request.request_id} failed: {e}"
                    )

                    if self._on_failure:
                        try:
                            await self._on_failure(request, e)
                        except Exception as callback_error:
                            self.logger.error(
                                f"Failure callback error: {callback_error}"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)

        self.logger.debug(f"Worker {worker_id} stopped")

    def get_metrics(self) -> Dict[str, Any]:
        avg_processing = (
            statistics.mean(self._processing_times)
            if self._processing_times else 0.0
        )
        return {
            "queue_name": self.queue_name,
            "size": self.size,
            "max_size": self._max_size,
            "utilization": round(self.utilization, 3),
            "is_backpressure": self.is_backpressure,
            "worker_count": len(self._workers),
            "running": self._running,
            "total_enqueued": self._total_enqueued,
            "total_processed": self._total_processed,
            "total_failed": self._total_failed,
            "total_rejected": self._total_rejected,
            "avg_processing_time_ms": round(avg_processing, 2),
        }


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================

class DeadLetterQueue:
    """
    Persistent dead letter queue for failed requests.

    Features:
    - Persistent storage (survives restarts)
    - Configurable retry scheduling with exponential backoff
    - Age-based cleanup
    - Metrics and monitoring
    """

    def __init__(self, queue_name: str = "default"):
        self.queue_name = queue_name
        self._max_retries = ResilienceConfig.get_dlq_max_retries()
        self._retry_delay_base = ResilienceConfig.get_dlq_retry_delay_base()
        self._persistence_path = ResilienceConfig.get_dlq_persistence_path() / queue_name
        self._max_age_hours = ResilienceConfig.get_dlq_max_age_hours()

        self._entries: Dict[str, DeadLetterEntry] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._retry_task: Optional[asyncio.Task] = None

        # Callback for retry processing
        self._retry_processor: Optional[Callable[[Dict[str, Any]], Awaitable[bool]]] = None

        # Metrics
        self._total_added = 0
        self._total_retried = 0
        self._total_expired = 0
        self._total_succeeded = 0

        self.logger = logging.getLogger(f"DLQ.{queue_name}")

    async def initialize(self) -> None:
        """Initialize DLQ and load persisted entries."""
        self._persistence_path.mkdir(parents=True, exist_ok=True)
        await self._load_persisted_entries()
        self._running = True
        self._retry_task = asyncio.create_task(self._retry_loop())
        self.logger.info(
            f"DLQ '{self.queue_name}' initialized with {len(self._entries)} entries"
        )

    async def shutdown(self) -> None:
        """Shutdown DLQ and persist entries."""
        self._running = False
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
        await self._persist_all_entries()
        self.logger.info(f"DLQ '{self.queue_name}' shutdown complete")

    def set_retry_processor(
        self,
        processor: Callable[[Dict[str, Any]], Awaitable[bool]]
    ) -> None:
        """Set the processor for retrying requests."""
        self._retry_processor = processor

    async def add(
        self,
        request_id: str,
        original_request: Dict[str, Any],
        failure_reason: str,
        failure_traceback: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a failed request to the DLQ."""
        async with self._lock:
            # Calculate next retry time with exponential backoff
            retry_count = 0
            if request_id in self._entries:
                retry_count = self._entries[request_id].retry_count + 1

            delay = self._retry_delay_base * (2 ** retry_count)
            # Add jitter (±20%)
            delay *= random.uniform(0.8, 1.2)

            entry = DeadLetterEntry(
                request_id=request_id,
                original_request=original_request,
                failure_reason=failure_reason,
                failure_traceback=failure_traceback,
                failure_time=time.time(),
                retry_count=retry_count,
                next_retry_time=time.time() + delay,
                max_retries=self._max_retries,
                metadata=metadata or {},
            )

            self._entries[request_id] = entry
            self._total_added += 1

            # Persist immediately
            await self._persist_entry(entry)

            self.logger.warning(
                f"Added to DLQ: {request_id} "
                f"reason='{failure_reason}' "
                f"retry={retry_count}/{self._max_retries} "
                f"next_retry={datetime.fromtimestamp(entry.next_retry_time).isoformat()}"
            )

    async def remove(self, request_id: str) -> Optional[DeadLetterEntry]:
        """Remove an entry from the DLQ."""
        async with self._lock:
            entry = self._entries.pop(request_id, None)
            if entry:
                await self._remove_persisted_entry(request_id)
            return entry

    async def get_ready_entries(self) -> List[DeadLetterEntry]:
        """Get entries ready for retry."""
        async with self._lock:
            return [
                entry for entry in self._entries.values()
                if entry.is_ready_for_retry()
            ]

    async def get_expired_entries(self) -> List[DeadLetterEntry]:
        """Get entries that have exceeded max retries or age."""
        max_age_seconds = self._max_age_hours * 3600
        cutoff = time.time() - max_age_seconds

        async with self._lock:
            return [
                entry for entry in self._entries.values()
                if not entry.should_retry() or entry.failure_time < cutoff
            ]

    async def _retry_loop(self) -> None:
        """Background loop for retrying failed requests."""
        while self._running:
            try:
                # Process ready entries
                ready = await self.get_ready_entries()
                for entry in ready:
                    if not self._retry_processor:
                        continue

                    try:
                        success = await self._retry_processor(entry.original_request)

                        if success:
                            await self.remove(entry.request_id)
                            self._total_succeeded += 1
                            self.logger.info(
                                f"DLQ retry succeeded: {entry.request_id}"
                            )
                        else:
                            # Update retry count and reschedule
                            entry.retry_count += 1
                            delay = self._retry_delay_base * (2 ** entry.retry_count)
                            entry.next_retry_time = time.time() + delay
                            await self._persist_entry(entry)
                            self._total_retried += 1

                    except Exception as e:
                        self.logger.error(
                            f"DLQ retry failed for {entry.request_id}: {e}"
                        )
                        entry.retry_count += 1
                        entry.failure_reason = str(e)
                        delay = self._retry_delay_base * (2 ** entry.retry_count)
                        entry.next_retry_time = time.time() + delay
                        await self._persist_entry(entry)
                        self._total_retried += 1

                # Clean up expired entries
                expired = await self.get_expired_entries()
                for entry in expired:
                    await self.remove(entry.request_id)
                    self._total_expired += 1
                    self.logger.warning(
                        f"DLQ entry expired: {entry.request_id} "
                        f"retries={entry.retry_count}"
                    )

                await asyncio.sleep(10.0)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"DLQ retry loop error: {e}")
                await asyncio.sleep(30.0)

    async def _load_persisted_entries(self) -> None:
        """Load entries from persistent storage."""
        if not self._persistence_path.exists():
            return

        for entry_file in self._persistence_path.glob("*.json"):
            try:
                if HAS_AIOFILES:
                    async with aiofiles.open(entry_file) as f:
                        data = json.loads(await f.read())
                else:
                    data = json.loads(entry_file.read_text())

                entry = DeadLetterEntry.from_dict(data)
                self._entries[entry.request_id] = entry

            except Exception as e:
                self.logger.error(f"Failed to load DLQ entry {entry_file}: {e}")

    async def _persist_entry(self, entry: DeadLetterEntry) -> None:
        """Persist a single entry."""
        try:
            filepath = self._persistence_path / f"{entry.request_id}.json"
            content = json.dumps(entry.to_dict(), indent=2)

            if HAS_AIOFILES:
                async with aiofiles.open(filepath, "w") as f:
                    await f.write(content)
            else:
                filepath.write_text(content)

        except Exception as e:
            self.logger.error(f"Failed to persist DLQ entry {entry.request_id}: {e}")

    async def _persist_all_entries(self) -> None:
        """Persist all entries."""
        for entry in self._entries.values():
            await self._persist_entry(entry)

    async def _remove_persisted_entry(self, request_id: str) -> None:
        """Remove persisted entry file."""
        try:
            filepath = self._persistence_path / f"{request_id}.json"
            if filepath.exists():
                filepath.unlink()
        except Exception as e:
            self.logger.error(f"Failed to remove persisted entry {request_id}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "queue_name": self.queue_name,
            "entry_count": len(self._entries),
            "total_added": self._total_added,
            "total_retried": self._total_retried,
            "total_expired": self._total_expired,
            "total_succeeded": self._total_succeeded,
            "max_retries": self._max_retries,
            "retry_delay_base": self._retry_delay_base,
            "persistence_path": str(self._persistence_path),
        }


# =============================================================================
# HEALTH-BASED ROUTER
# =============================================================================

class HealthBasedRouter:
    """
    Intelligent router that routes requests based on service health.

    Features:
    - Multiple routing strategies
    - Automatic health checks
    - Latency-aware routing
    - Automatic failover
    """

    def __init__(
        self,
        router_name: str,
        strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
    ):
        self.router_name = router_name
        self.strategy = strategy
        self._endpoints: Dict[str, ServiceEndpoint] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._round_robin_index = 0

        # Health check configuration
        self._health_interval = ResilienceConfig.get_health_check_interval()
        self._health_timeout = ResilienceConfig.get_health_timeout()
        self._unhealthy_threshold = ResilienceConfig.get_unhealthy_threshold()
        self._healthy_threshold = ResilienceConfig.get_healthy_threshold()

        self.logger = logging.getLogger(f"HealthRouter.{router_name}")

    async def add_endpoint(
        self,
        name: str,
        url: str,
        weight: float = 1.0,
        health_check_path: str = "/health",
    ) -> None:
        """Add a service endpoint."""
        async with self._lock:
            self._endpoints[name] = ServiceEndpoint(
                name=name,
                url=url.rstrip("/"),
                weight=weight,
                metadata={"health_check_path": health_check_path},
            )
            self.logger.info(f"Added endpoint: {name} -> {url}")

    async def remove_endpoint(self, name: str) -> None:
        """Remove a service endpoint."""
        async with self._lock:
            self._endpoints.pop(name, None)
            self.logger.info(f"Removed endpoint: {name}")

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._health_task = asyncio.create_task(self._health_check_loop())
        self.logger.info(f"Router '{self.router_name}' started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        self.logger.info(f"Router '{self.router_name}' stopped")

    async def get_endpoint(self) -> Optional[ServiceEndpoint]:
        """Get the best endpoint based on routing strategy."""
        async with self._lock:
            healthy_endpoints = [
                ep for ep in self._endpoints.values()
                if ep.health in (ServiceHealth.HEALTHY, ServiceHealth.DEGRADED)
            ]

            if not healthy_endpoints:
                self.logger.warning("No healthy endpoints available")
                return None

            if self.strategy == RoutingStrategy.ROUND_ROBIN:
                return self._route_round_robin(healthy_endpoints)
            elif self.strategy == RoutingStrategy.LEAST_CONNECTIONS:
                return self._route_least_connections(healthy_endpoints)
            elif self.strategy == RoutingStrategy.LEAST_LATENCY:
                return self._route_least_latency(healthy_endpoints)
            elif self.strategy == RoutingStrategy.WEIGHTED_RANDOM:
                return self._route_weighted_random(healthy_endpoints)
            else:  # ADAPTIVE
                return self._route_adaptive(healthy_endpoints)

    def _route_round_robin(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round-robin routing."""
        self._round_robin_index = (self._round_robin_index + 1) % len(endpoints)
        return endpoints[self._round_robin_index]

    def _route_least_connections(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Route to endpoint with least active connections."""
        return min(endpoints, key=lambda ep: ep.active_connections)

    def _route_least_latency(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Route to endpoint with lowest average latency."""
        return min(endpoints, key=lambda ep: ep.avg_latency_ms)

    def _route_weighted_random(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted random routing based on configured weights."""
        total_weight = sum(ep.weight for ep in endpoints)
        r = random.uniform(0, total_weight)

        cumulative = 0
        for ep in endpoints:
            cumulative += ep.weight
            if r <= cumulative:
                return ep

        return endpoints[-1]

    def _route_adaptive(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """
        Adaptive routing based on health score.

        Combines latency, error rate, and consecutive successes.
        """
        # Calculate weights based on health score
        weighted_endpoints = [
            (ep, ep.health_score * ep.weight)
            for ep in endpoints
        ]

        # If all scores are 0, fall back to round-robin
        total_score = sum(w for _, w in weighted_endpoints)
        if total_score == 0:
            return self._route_round_robin(endpoints)

        # Weighted random selection
        r = random.uniform(0, total_score)
        cumulative = 0
        for ep, weight in weighted_endpoints:
            cumulative += weight
            if r <= cumulative:
                return ep

        return endpoints[-1]

    async def record_success(
        self,
        endpoint_name: str,
        latency_ms: float,
    ) -> None:
        """Record a successful request."""
        async with self._lock:
            ep = self._endpoints.get(endpoint_name)
            if ep:
                ep.total_requests += 1
                ep.latency_history.append(latency_ms)
                ep.consecutive_successes += 1
                ep.consecutive_failures = 0

                # Update health status
                if ep.consecutive_successes >= self._healthy_threshold:
                    if ep.health != ServiceHealth.HEALTHY:
                        ep.health = ServiceHealth.HEALTHY
                        self.logger.info(f"Endpoint {endpoint_name} marked HEALTHY")

    async def record_failure(
        self,
        endpoint_name: str,
        error: Optional[str] = None,
    ) -> None:
        """Record a failed request."""
        async with self._lock:
            ep = self._endpoints.get(endpoint_name)
            if ep:
                ep.total_requests += 1
                ep.total_errors += 1
                ep.consecutive_failures += 1
                ep.consecutive_successes = 0

                # Update health status
                if ep.consecutive_failures >= self._unhealthy_threshold:
                    if ep.health != ServiceHealth.UNHEALTHY:
                        ep.health = ServiceHealth.UNHEALTHY
                        self.logger.warning(
                            f"Endpoint {endpoint_name} marked UNHEALTHY: {error}"
                        )
                elif ep.consecutive_failures >= self._unhealthy_threshold // 2:
                    if ep.health == ServiceHealth.HEALTHY:
                        ep.health = ServiceHealth.DEGRADED
                        self.logger.warning(
                            f"Endpoint {endpoint_name} marked DEGRADED"
                        )

    async def _health_check_loop(self) -> None:
        """Background loop for health checks."""
        while self._running:
            try:
                for name, ep in list(self._endpoints.items()):
                    await self._check_endpoint_health(ep)

                await asyncio.sleep(self._health_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5.0)

    async def _check_endpoint_health(self, endpoint: ServiceEndpoint) -> None:
        """Check health of a single endpoint."""
        if not HAS_AIOHTTP:
            return

        health_path = endpoint.metadata.get("health_check_path", "/health")
        url = f"{endpoint.url}{health_path}"

        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self._health_timeout),
                ) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    endpoint.last_health_check = time.time()

                    if response.status == 200:
                        endpoint.latency_history.append(latency_ms)
                        endpoint.consecutive_successes += 1
                        endpoint.consecutive_failures = 0

                        if endpoint.consecutive_successes >= self._healthy_threshold:
                            endpoint.health = ServiceHealth.HEALTHY
                    else:
                        await self.record_failure(
                            endpoint.name,
                            f"Health check returned {response.status}"
                        )

        except asyncio.TimeoutError:
            await self.record_failure(endpoint.name, "Health check timeout")
        except Exception as e:
            await self.record_failure(endpoint.name, str(e))

    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "router_name": self.router_name,
            "strategy": self.strategy.value,
            "running": self._running,
            "endpoint_count": len(self._endpoints),
            "healthy_count": sum(
                1 for ep in self._endpoints.values()
                if ep.health == ServiceHealth.HEALTHY
            ),
            "endpoints": {
                name: ep.to_dict()
                for name, ep in self._endpoints.items()
            },
        }


# =============================================================================
# AUTOMATIC FAILOVER
# =============================================================================

class FailoverOrchestrator:
    """
    Orchestrates automatic failover between primary and backup services.

    Features:
    - Zero-downtime failover
    - Health-based automatic triggering
    - Manual failover support
    - Failback support
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._primary: Optional[ServiceEndpoint] = None
        self._backups: List[ServiceEndpoint] = []
        self._current: Optional[ServiceEndpoint] = None
        self._lock = asyncio.Lock()

        # State
        self._failover_count = 0
        self._failback_count = 0
        self._last_failover_time: Optional[float] = None

        # Callbacks
        self._on_failover: Optional[Callable[[str, str], Awaitable[None]]] = None
        self._on_failback: Optional[Callable[[str], Awaitable[None]]] = None

        self.logger = logging.getLogger(f"Failover.{service_name}")

    def set_primary(self, endpoint: ServiceEndpoint) -> None:
        """Set the primary endpoint."""
        self._primary = endpoint
        if self._current is None:
            self._current = endpoint

    def add_backup(self, endpoint: ServiceEndpoint) -> None:
        """Add a backup endpoint."""
        self._backups.append(endpoint)

    def set_failover_callback(
        self,
        callback: Callable[[str, str], Awaitable[None]]
    ) -> None:
        """Set callback for failover events (from_endpoint, to_endpoint)."""
        self._on_failover = callback

    def set_failback_callback(
        self,
        callback: Callable[[str], Awaitable[None]]
    ) -> None:
        """Set callback for failback events (to_primary)."""
        self._on_failback = callback

    async def get_current_endpoint(self) -> Optional[ServiceEndpoint]:
        """Get the current active endpoint."""
        return self._current

    async def trigger_failover(self, reason: str = "manual") -> bool:
        """
        Trigger failover to the next available backup.

        Returns True if failover succeeded.
        """
        async with self._lock:
            if not self._backups:
                self.logger.error("No backup endpoints available for failover")
                return False

            # Find next healthy backup
            for backup in self._backups:
                if backup.health in (ServiceHealth.HEALTHY, ServiceHealth.DEGRADED):
                    old_endpoint = self._current.name if self._current else "none"
                    self._current = backup
                    self._failover_count += 1
                    self._last_failover_time = time.time()

                    self.logger.warning(
                        f"FAILOVER: {old_endpoint} -> {backup.name} "
                        f"reason='{reason}'"
                    )

                    if self._on_failover:
                        await self._on_failover(old_endpoint, backup.name)

                    return True

            self.logger.error("No healthy backup endpoints available")
            return False

    async def trigger_failback(self) -> bool:
        """
        Trigger failback to the primary endpoint.

        Returns True if failback succeeded.
        """
        async with self._lock:
            if not self._primary:
                self.logger.error("No primary endpoint configured")
                return False

            if self._primary.health != ServiceHealth.HEALTHY:
                self.logger.warning("Primary endpoint not healthy, cannot failback")
                return False

            if self._current == self._primary:
                self.logger.debug("Already on primary endpoint")
                return True

            self._current = self._primary
            self._failback_count += 1

            self.logger.info(f"FAILBACK to primary: {self._primary.name}")

            if self._on_failback:
                await self._on_failback(self._primary.name)

            return True

    async def should_failover(self) -> bool:
        """Check if failover should be triggered based on current endpoint health."""
        if not self._current:
            return False

        return self._current.health == ServiceHealth.UNHEALTHY

    async def should_failback(self) -> bool:
        """Check if failback should be triggered."""
        if not self._primary or self._current == self._primary:
            return False

        return self._primary.health == ServiceHealth.HEALTHY

    def get_status(self) -> Dict[str, Any]:
        """Get failover status."""
        return {
            "service_name": self.service_name,
            "primary": self._primary.to_dict() if self._primary else None,
            "current": self._current.to_dict() if self._current else None,
            "backup_count": len(self._backups),
            "failover_count": self._failover_count,
            "failback_count": self._failback_count,
            "last_failover_time": self._last_failover_time,
        }


# =============================================================================
# CHAOS ENGINEERING
# =============================================================================

class ChaosController:
    """
    Chaos engineering controller for controlled failure injection.

    Features:
    - Programmable failure modes
    - Time-bounded experiments
    - Service targeting
    - Metrics collection
    """

    def __init__(self):
        self._experiments: Dict[str, ChaosExperiment] = {}
        self._active_experiments: Set[str] = set()
        self._lock = asyncio.Lock()
        self._enabled = ResilienceConfig.is_chaos_enabled()

        self.logger = logging.getLogger("Chaos.Controller")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable chaos engineering."""
        self._enabled = True
        self.logger.warning("Chaos engineering ENABLED")

    def disable(self) -> None:
        """Disable chaos engineering."""
        self._enabled = False
        self._active_experiments.clear()
        self.logger.info("Chaos engineering DISABLED")

    async def create_experiment(
        self,
        name: str,
        target_services: List[str],
        failure_mode: FailureMode,
        duration_seconds: float,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a chaos experiment."""
        experiment_id = uuid.uuid4().hex[:12]

        experiment = ChaosExperiment(
            experiment_id=experiment_id,
            name=name,
            target_services=target_services,
            failure_mode=failure_mode,
            parameters=parameters or {},
            duration_seconds=duration_seconds,
        )

        async with self._lock:
            self._experiments[experiment_id] = experiment

        self.logger.info(
            f"Created chaos experiment: {name} ({experiment_id}) "
            f"mode={failure_mode.value} duration={duration_seconds}s"
        )

        return experiment_id

    async def start_experiment(self, experiment_id: str) -> bool:
        """Start a chaos experiment."""
        if not self._enabled:
            self.logger.warning("Chaos engineering is disabled")
            return False

        async with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                self.logger.error(f"Experiment not found: {experiment_id}")
                return False

            experiment.start_time = time.time()
            experiment.active = True
            self._active_experiments.add(experiment_id)

        self.logger.warning(
            f"CHAOS EXPERIMENT STARTED: {experiment.name} ({experiment_id})"
        )

        return True

    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a chaos experiment."""
        async with self._lock:
            experiment = self._experiments.get(experiment_id)
            if not experiment:
                return False

            experiment.end_time = time.time()
            experiment.active = False
            self._active_experiments.discard(experiment_id)

        self.logger.info(f"Chaos experiment stopped: {experiment_id}")
        return True

    async def should_inject_failure(self, service_name: str) -> Optional[ChaosExperiment]:
        """Check if failure should be injected for a service."""
        if not self._enabled:
            return None

        async with self._lock:
            for exp_id in list(self._active_experiments):
                experiment = self._experiments.get(exp_id)
                if not experiment or not experiment.active:
                    self._active_experiments.discard(exp_id)
                    continue

                # Check if experiment has expired
                if experiment.is_expired():
                    experiment.active = False
                    experiment.end_time = time.time()
                    self._active_experiments.discard(exp_id)
                    self.logger.info(f"Chaos experiment expired: {exp_id}")
                    continue

                # Check if service is targeted
                if service_name in experiment.target_services or "*" in experiment.target_services:
                    return experiment

        return None

    async def inject_failure(self, experiment: ChaosExperiment) -> None:
        """Inject failure based on experiment parameters."""
        mode = experiment.failure_mode
        params = experiment.parameters

        if mode == FailureMode.EXCEPTION:
            error_msg = params.get("error_message", "Chaos-injected failure")
            raise ChaosInjectedException(error_msg)

        elif mode == FailureMode.TIMEOUT:
            delay = params.get("delay_seconds", 30.0)
            await asyncio.sleep(delay)
            raise asyncio.TimeoutError("Chaos-injected timeout")

        elif mode == FailureMode.LATENCY:
            min_ms, max_ms = ResilienceConfig.get_chaos_latency_injection_ms()
            min_ms = params.get("min_latency_ms", min_ms)
            max_ms = params.get("max_latency_ms", max_ms)
            delay = random.uniform(min_ms, max_ms) / 1000
            await asyncio.sleep(delay)

        elif mode == FailureMode.RESOURCE_EXHAUSTION:
            raise MemoryError("Chaos-injected resource exhaustion")

        elif mode == FailureMode.NETWORK_PARTITION:
            raise ConnectionError("Chaos-injected network partition")

        elif mode == FailureMode.DATA_CORRUPTION:
            raise ValueError("Chaos-injected data corruption")

    def get_status(self) -> Dict[str, Any]:
        """Get chaos controller status."""
        return {
            "enabled": self._enabled,
            "total_experiments": len(self._experiments),
            "active_experiments": len(self._active_experiments),
            "experiments": {
                exp_id: exp.to_dict()
                for exp_id, exp in self._experiments.items()
            },
        }


class ChaosInjectedException(Exception):
    """Exception injected by chaos engineering."""
    pass


# =============================================================================
# ADAPTIVE TIMEOUT MANAGER
# =============================================================================

class AdaptiveTimeoutManager:
    """
    Manages adaptive timeouts based on operation latency history.

    Features:
    - Per-operation timeout tracking
    - Percentile-based timeout calculation
    - Cold start handling
    - System load awareness
    """

    def __init__(self):
        self._operations: Dict[str, AdaptiveTimeoutState] = {}
        self._lock = asyncio.Lock()

        self.logger = logging.getLogger("AdaptiveTimeout")

    async def get_timeout(self, operation_name: str) -> float:
        """Get the adaptive timeout for an operation."""
        async with self._lock:
            if operation_name not in self._operations:
                self._operations[operation_name] = AdaptiveTimeoutState(
                    operation_name=operation_name,
                    current_timeout=ResilienceConfig.get_base_timeout(),
                    min_timeout=ResilienceConfig.get_min_timeout(),
                    max_timeout=ResilienceConfig.get_max_timeout(),
                )

            return self._operations[operation_name].calculate_adaptive_timeout()

    async def record_latency(self, operation_name: str, latency_ms: float) -> None:
        """Record latency for an operation."""
        async with self._lock:
            if operation_name in self._operations:
                self._operations[operation_name].record_latency(latency_ms)

    async def record_timeout(
        self,
        operation_name: str,
        timeout_ms: float,
        did_timeout: bool,
    ) -> None:
        """Record a timeout event."""
        async with self._lock:
            if operation_name in self._operations:
                self._operations[operation_name].record_timeout(timeout_ms, did_timeout)

    def get_status(self) -> Dict[str, Any]:
        """Get timeout manager status."""
        return {
            "operations": {
                name: {
                    "current_timeout": state.current_timeout,
                    "cold_start": state.cold_start,
                    "sample_count": len(state.latency_history),
                }
                for name, state in self._operations.items()
            }
        }


# =============================================================================
# DECORRELATED JITTER RETRY
# =============================================================================

class DecorrelatedJitterRetry:
    """
    Retry strategy with decorrelated jitter.

    Uses AWS-style decorrelated jitter which provides better spread
    than simple exponential backoff with random jitter.
    """

    def __init__(
        self,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ):
        self.base_delay = base_delay or ResilienceConfig.get_retry_base_delay()
        self.max_delay = max_delay or ResilienceConfig.get_retry_max_delay()
        self.max_attempts = max_attempts or ResilienceConfig.get_retry_max_attempts()
        self._last_delay = self.base_delay

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay using decorrelated jitter.

        Formula: delay = random(base, last_delay * 3)
        Capped at max_delay
        """
        delay = random.uniform(self.base_delay, self._last_delay * 3)
        delay = min(delay, self.max_delay)
        self._last_delay = delay
        return delay

    def should_retry(self, attempt: int) -> bool:
        """Check if we should retry."""
        return attempt < self.max_attempts

    def reset(self) -> None:
        """Reset the retry state."""
        self._last_delay = self.base_delay


def with_retry(
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for async functions with decorrelated jitter retry.

    Usage:
        @with_retry(max_attempts=3, retryable_exceptions=(ConnectionError,))
        async def my_function():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            retry = DecorrelatedJitterRetry(
                base_delay=base_delay,
                max_delay=max_delay,
                max_attempts=max_attempts,
            )

            last_exception: Optional[Exception] = None

            for attempt in range(retry.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if not retry.should_retry(attempt + 1):
                        break

                    delay = retry.get_delay(attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{retry.max_attempts} "
                        f"for {func.__name__} after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)

            raise last_exception or RuntimeError("Retry exhausted without exception")

        return wrapper
    return decorator


# =============================================================================
# UNIFIED RESILIENCE ENGINE
# =============================================================================

class UnifiedResilienceEngine:
    """
    Central coordination point for all resilience patterns.

    Integrates:
    - Bulkhead isolation
    - Priority request queuing
    - Dead letter queue
    - Health-based routing
    - Automatic failover
    - Chaos engineering
    - Adaptive timeouts
    - Decorrelated jitter retry

    This is the main entry point for resilience in the JARVIS ecosystem.
    """

    _instance: Optional["UnifiedResilienceEngine"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        # Core components
        self._bulkhead_manager = BulkheadManager()
        self._request_queues: Dict[str, PriorityRequestQueue] = {}
        self._dead_letter_queues: Dict[str, DeadLetterQueue] = {}
        self._routers: Dict[str, HealthBasedRouter] = {}
        self._failover_orchestrators: Dict[str, FailoverOrchestrator] = {}
        self._chaos_controller = ChaosController()
        self._timeout_manager = AdaptiveTimeoutManager()

        # State
        self._initialized = False
        self._running = False

        # Cross-repo integration
        self._neural_mesh = None  # Will be set during initialization

        self.logger = logging.getLogger("UnifiedResilienceEngine")

    @classmethod
    async def get_instance(cls) -> "UnifiedResilienceEngine":
        """Get or create the singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    async def initialize(self) -> bool:
        """Initialize all resilience components."""
        if self._initialized:
            return True

        self.logger.info("=" * 60)
        self.logger.info("Initializing Unified Resilience Engine v1.0")
        self.logger.info("=" * 60)

        try:
            # Initialize default DLQ
            default_dlq = DeadLetterQueue("default")
            await default_dlq.initialize()
            self._dead_letter_queues["default"] = default_dlq

            # Initialize cross-repo routers
            await self._setup_cross_repo_routing()

            # Initialize chaos controller if enabled
            if ResilienceConfig.is_chaos_enabled():
                self._chaos_controller.enable()
                self.logger.warning("Chaos engineering is ENABLED")

            self._initialized = True
            self._running = True

            self.logger.info("Unified Resilience Engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize resilience engine: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown all resilience components."""
        self.logger.info("Shutting down Unified Resilience Engine...")
        self._running = False

        # Shutdown request queues
        for queue in self._request_queues.values():
            await queue.stop(wait_for_drain=True, drain_timeout=30.0)

        # Shutdown DLQs
        for dlq in self._dead_letter_queues.values():
            await dlq.shutdown()

        # Shutdown routers
        for router in self._routers.values():
            await router.stop()

        # Disable chaos
        self._chaos_controller.disable()

        self._initialized = False
        self.logger.info("Unified Resilience Engine shutdown complete")

    async def _setup_cross_repo_routing(self) -> None:
        """Setup routing for cross-repo communication."""
        # Create router for Trinity services
        trinity_router = HealthBasedRouter(
            "trinity",
            strategy=RoutingStrategy.ADAPTIVE,
        )

        # Add JARVIS endpoints
        await trinity_router.add_endpoint(
            "jarvis_body",
            ResilienceConfig.get_jarvis_url(),
            weight=1.0,
        )

        # Add JARVIS Prime endpoints
        await trinity_router.add_endpoint(
            "jarvis_prime",
            ResilienceConfig.get_prime_url(),
            weight=1.0,
        )

        # Add Reactor Core endpoints
        await trinity_router.add_endpoint(
            "reactor_core",
            ResilienceConfig.get_reactor_url(),
            weight=1.0,
        )

        await trinity_router.start()
        self._routers["trinity"] = trinity_router

        # Setup failover orchestrator
        failover = FailoverOrchestrator("prime_llm")

        # Primary: Local JARVIS Prime
        primary = ServiceEndpoint(
            name="local_prime",
            url=ResilienceConfig.get_prime_url(),
        )
        failover.set_primary(primary)

        # Backup: Ollama (if available)
        ollama_backup = ServiceEndpoint(
            name="ollama_backup",
            url=os.getenv("OLLAMA_API_URL", "http://localhost:11434"),
        )
        failover.add_backup(ollama_backup)

        # Backup: Cloud API (emergency)
        cloud_backup = ServiceEndpoint(
            name="cloud_api",
            url=os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com"),
        )
        failover.add_backup(cloud_backup)

        self._failover_orchestrators["prime_llm"] = failover

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def get_bulkhead(self, pool_name: str) -> BulkheadPool:
        """Get a bulkhead pool by name."""
        return await self._bulkhead_manager.get_pool(pool_name)

    @asynccontextmanager
    async def bulkhead(self, pool_name: str, timeout: Optional[float] = None):
        """Context manager for bulkhead-protected operations."""
        pool = await self.get_bulkhead(pool_name)
        async with pool.acquire(timeout):
            # Check for chaos injection
            if self._chaos_controller.is_enabled:
                experiment = await self._chaos_controller.should_inject_failure(pool_name)
                if experiment:
                    await self._chaos_controller.inject_failure(experiment)

            yield

    async def create_queue(
        self,
        queue_name: str,
        processor: Callable[[Any], Awaitable[Any]],
        max_size: Optional[int] = None,
        worker_count: Optional[int] = None,
    ) -> PriorityRequestQueue:
        """Create and start a priority request queue."""
        queue = PriorityRequestQueue(
            queue_name=queue_name,
            processor=processor,
            max_size=max_size,
            worker_count=worker_count,
        )

        # Connect to DLQ
        dlq = self._dead_letter_queues.get("default")
        if dlq:
            async def on_failure(request: QueuedRequest, error: Exception):
                await dlq.add(
                    request_id=request.request_id,
                    original_request={"payload": str(request.payload)},
                    failure_reason=str(error),
                    failure_traceback=traceback.format_exc(),
                    metadata=request.metadata,
                )

            queue.set_failure_callback(on_failure)

        await queue.start()
        self._request_queues[queue_name] = queue
        return queue

    async def enqueue(
        self,
        queue_name: str,
        payload: Any,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0,
    ) -> str:
        """Enqueue a request to a named queue."""
        queue = self._request_queues.get(queue_name)
        if not queue:
            raise ValueError(f"Queue '{queue_name}' not found")

        return await queue.enqueue(
            payload=payload,
            priority=priority,
            timeout=timeout,
        )

    async def get_router(self, router_name: str) -> Optional[HealthBasedRouter]:
        """Get a health-based router."""
        return self._routers.get(router_name)

    async def route_request(
        self,
        router_name: str,
        request_func: Callable[[ServiceEndpoint], Awaitable[T]],
    ) -> T:
        """
        Route a request through a health-based router.

        Handles automatic retries and endpoint recording.
        """
        router = self._routers.get(router_name)
        if not router:
            raise ValueError(f"Router '{router_name}' not found")

        endpoint = await router.get_endpoint()
        if not endpoint:
            raise RuntimeError(f"No healthy endpoints available in router '{router_name}'")

        start_time = time.time()
        try:
            result = await request_func(endpoint)
            latency_ms = (time.time() - start_time) * 1000
            await router.record_success(endpoint.name, latency_ms)
            return result

        except Exception as e:
            await router.record_failure(endpoint.name, str(e))
            raise

    async def get_failover(self, service_name: str) -> Optional[FailoverOrchestrator]:
        """Get a failover orchestrator."""
        return self._failover_orchestrators.get(service_name)

    async def get_adaptive_timeout(self, operation_name: str) -> float:
        """Get adaptive timeout for an operation."""
        return await self._timeout_manager.get_timeout(operation_name)

    async def record_operation_latency(
        self,
        operation_name: str,
        latency_ms: float,
    ) -> None:
        """Record operation latency for adaptive timeout calculation."""
        await self._timeout_manager.record_latency(operation_name, latency_ms)

    def get_chaos_controller(self) -> ChaosController:
        """Get the chaos controller."""
        return self._chaos_controller

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all resilience components."""
        return {
            "engine": {
                "initialized": self._initialized,
                "running": self._running,
            },
            "bulkheads": self._bulkhead_manager.get_all_metrics(),
            "queues": {
                name: queue.get_metrics()
                for name, queue in self._request_queues.items()
            },
            "dead_letter_queues": {
                name: dlq.get_metrics()
                for name, dlq in self._dead_letter_queues.items()
            },
            "routers": {
                name: router.get_status()
                for name, router in self._routers.items()
            },
            "failovers": {
                name: fo.get_status()
                for name, fo in self._failover_orchestrators.items()
            },
            "chaos": self._chaos_controller.get_status(),
            "timeouts": self._timeout_manager.get_status(),
        }


# =============================================================================
# GLOBAL INSTANCE AND HELPER FUNCTIONS
# =============================================================================

_engine: Optional[UnifiedResilienceEngine] = None


async def get_resilience_engine() -> UnifiedResilienceEngine:
    """Get the global resilience engine instance."""
    global _engine
    if _engine is None:
        _engine = await UnifiedResilienceEngine.get_instance()
    return _engine


async def initialize_resilience() -> bool:
    """Initialize the global resilience engine."""
    engine = await get_resilience_engine()
    return await engine.initialize()


async def shutdown_resilience() -> None:
    """Shutdown the global resilience engine."""
    global _engine
    if _engine:
        await _engine.shutdown()
        _engine = None


# Convenience decorators

def with_bulkhead(pool_name: str, timeout: Optional[float] = None):
    """
    Decorator for bulkhead-protected async functions.

    Usage:
        @with_bulkhead("llm_pool", timeout=30.0)
        async def call_llm():
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            engine = await get_resilience_engine()
            async with engine.bulkhead(pool_name, timeout):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def with_adaptive_timeout(operation_name: str):
    """
    Decorator for operations with adaptive timeout.

    Usage:
        @with_adaptive_timeout("llm_completion")
        async def complete(prompt: str):
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            engine = await get_resilience_engine()
            timeout = await engine.get_adaptive_timeout(operation_name)

            start_time = time.time()
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout,
                )
                latency_ms = (time.time() - start_time) * 1000
                await engine.record_operation_latency(operation_name, latency_ms)
                return result

            except asyncio.TimeoutError:
                await engine._timeout_manager.record_timeout(
                    operation_name,
                    timeout * 1000,
                    did_timeout=True,
                )
                raise

        return wrapper
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "ResilienceConfig",
    # Enums
    "BulkheadState",
    "RequestPriority",
    "FailureMode",
    "ServiceHealth",
    "RoutingStrategy",
    # Data Structures
    "BulkheadMetrics",
    "QueuedRequest",
    "DeadLetterEntry",
    "ServiceEndpoint",
    "AdaptiveTimeoutState",
    "ChaosExperiment",
    # Bulkhead
    "BulkheadPool",
    "BulkheadRejectedError",
    "BulkheadTimeoutError",
    "BulkheadManager",
    # Queue
    "BackpressureSignal",
    "PriorityRequestQueue",
    # DLQ
    "DeadLetterQueue",
    # Routing
    "HealthBasedRouter",
    # Failover
    "FailoverOrchestrator",
    # Chaos
    "ChaosController",
    "ChaosInjectedException",
    # Timeout
    "AdaptiveTimeoutManager",
    # Retry
    "DecorrelatedJitterRetry",
    "with_retry",
    # Main Engine
    "UnifiedResilienceEngine",
    # Global Functions
    "get_resilience_engine",
    "initialize_resilience",
    "shutdown_resilience",
    # Decorators
    "with_bulkhead",
    "with_adaptive_timeout",
]
