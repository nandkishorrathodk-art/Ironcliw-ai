"""
Cross-Repo Circuit Breaker with Failure Classification
======================================================

Enhanced circuit breaker for cross-repo communication with:
- Failure type classification (network, timeout, resource, etc.)
- Per-tier health tracking
- Adaptive thresholds based on failure patterns
- Comprehensive health status for monitoring

Author: Ironcliw Cross-Repo Resilience
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Classification of failure types."""

    NETWORK = "network"  # Connection refused, reset, timeout
    TIMEOUT = "timeout"  # Operation timeout
    RESOURCE = "resource"  # Resource exhausted (memory, disk, etc.)
    RATE_LIMIT = "rate_limit"  # Too many requests
    AUTH = "auth"  # Authentication/authorization failure
    VALIDATION = "validation"  # Invalid input/response
    INTERNAL = "internal"  # Internal server error
    UNKNOWN = "unknown"  # Unclassified


class CircuitState(Enum):
    """State of circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class TierHealth:
    """Health status for a single tier."""

    tier_name: str
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    opened_at: Optional[float] = None
    half_open_calls: int = 0

    # Failure breakdown
    failures_by_type: Dict[FailureType, int] = field(default_factory=dict)

    # Performance
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    recent_latencies: List[float] = field(default_factory=list)

    def record_success(self, latency_ms: float) -> None:
        """Record a successful operation."""
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.total_successes += 1
        self.last_success_time = time.time()
        self._record_latency(latency_ms)

    def record_failure(self, failure_type: FailureType) -> None:
        """Record a failed operation."""
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.total_failures += 1
        self.last_failure_time = time.time()
        self.failures_by_type[failure_type] = self.failures_by_type.get(failure_type, 0) + 1

    def _record_latency(self, latency_ms: float) -> None:
        """Record latency for percentile calculation."""
        self.recent_latencies.append(latency_ms)
        # Keep last 100 latencies
        if len(self.recent_latencies) > 100:
            self.recent_latencies = self.recent_latencies[-100:]

        # Update averages
        self.avg_latency_ms = sum(self.recent_latencies) / len(self.recent_latencies)

        # Calculate P95
        if len(self.recent_latencies) >= 20:
            sorted_latencies = sorted(self.recent_latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            self.p95_latency_ms = sorted_latencies[p95_index]

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 1.0
        return self.total_successes / total

    @property
    def dominant_failure_type(self) -> Optional[FailureType]:
        """Get the most common failure type."""
        if not self.failures_by_type:
            return None
        return max(self.failures_by_type.items(), key=lambda x: x[1])[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_name": self.tier_name,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "success_rate": round(self.success_rate, 3),
            "dominant_failure_type": self.dominant_failure_type.value if self.dominant_failure_type else None,
            "failures_by_type": {k.value: v for k, v in self.failures_by_type.items()},
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Thresholds
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: float = 30.0  # Time before half-open
    half_open_max_calls: int = 3  # Allowed calls in half-open

    # Adaptive thresholds based on failure type
    adaptive_thresholds: bool = True
    network_failure_threshold: int = 3  # Network issues open faster
    timeout_failure_threshold: int = 4
    rate_limit_failure_threshold: int = 2  # Rate limits open very fast

    # Recovery
    recovery_factor: float = 1.5  # Increase timeout on repeated opens
    max_timeout_seconds: float = 300.0  # Max timeout after repeated failures

    # v93.0: Startup grace period - more lenient during service initialization
    startup_grace_period_seconds: float = 180.0  # 3 minutes for ML model loading
    startup_failure_threshold: int = 30  # Much higher threshold during startup
    startup_network_failure_threshold: int = 20  # Network errors common during startup


class FailureClassifier:
    """Classifies exceptions into failure types."""

    # Exception type to failure type mapping
    TYPE_MAP: Dict[Type[Exception], FailureType] = {
        ConnectionRefusedError: FailureType.NETWORK,
        ConnectionResetError: FailureType.NETWORK,
        ConnectionAbortedError: FailureType.NETWORK,
        BrokenPipeError: FailureType.NETWORK,
        TimeoutError: FailureType.TIMEOUT,
        asyncio.TimeoutError: FailureType.TIMEOUT,
        MemoryError: FailureType.RESOURCE,
        PermissionError: FailureType.AUTH,
        ValueError: FailureType.VALIDATION,
        TypeError: FailureType.VALIDATION,
        KeyError: FailureType.VALIDATION,
    }

    # Pattern-based classification
    PATTERNS: Dict[str, FailureType] = {
        "connection": FailureType.NETWORK,
        "refused": FailureType.NETWORK,
        "reset": FailureType.NETWORK,
        "timeout": FailureType.TIMEOUT,
        "timed out": FailureType.TIMEOUT,
        "too many requests": FailureType.RATE_LIMIT,
        "rate limit": FailureType.RATE_LIMIT,
        "429": FailureType.RATE_LIMIT,
        "503": FailureType.RESOURCE,
        "502": FailureType.NETWORK,
        "504": FailureType.TIMEOUT,
        "unauthorized": FailureType.AUTH,
        "forbidden": FailureType.AUTH,
        "401": FailureType.AUTH,
        "403": FailureType.AUTH,
        "invalid": FailureType.VALIDATION,
        "not found": FailureType.VALIDATION,
        "internal server error": FailureType.INTERNAL,
        "500": FailureType.INTERNAL,
    }

    @classmethod
    def classify(cls, error: Exception) -> FailureType:
        """Classify an exception into a failure type."""
        # Check type mapping
        for exc_type, failure_type in cls.TYPE_MAP.items():
            if isinstance(error, exc_type):
                return failure_type

        # Check patterns
        error_str = str(error).lower()
        for pattern, failure_type in cls.PATTERNS.items():
            if pattern in error_str:
                return failure_type

        return FailureType.UNKNOWN

    @classmethod
    def is_retriable(cls, failure_type: FailureType) -> bool:
        """Check if failure type is worth retrying."""
        retriable = {
            FailureType.NETWORK,
            FailureType.TIMEOUT,
            FailureType.RESOURCE,
            FailureType.RATE_LIMIT,
            FailureType.INTERNAL,
        }
        return failure_type in retriable


class CircuitOpenError(Exception):
    """Raised when circuit is open."""

    def __init__(
        self,
        tier: str,
        state: CircuitState,
        retry_after: float,
        reason: Optional[str] = None,
    ):
        self.tier = tier
        self.state = state
        self.retry_after = retry_after
        self.reason = reason
        super().__init__(
            f"Circuit for '{tier}' is {state.value}. "
            f"Retry after {retry_after:.1f}s"
            + (f" ({reason})" if reason else "")
        )


class CrossRepoCircuitBreaker:
    """
    Circuit breaker with failure classification for cross-repo communication.

    Provides per-tier health tracking and adaptive thresholds based on
    failure patterns.

    Usage:
        breaker = CrossRepoCircuitBreaker("prime_router")

        # Execute with circuit protection
        result = await breaker.execute(
            tier="gcp_vm",
            func=call_gcp_vm,
            args=(task,),
        )

        # Check health
        health = breaker.get_tier_health("gcp_vm")
        if health.state == CircuitState.OPEN:
            # Use fallback

        # Get overall status
        status = breaker.get_status()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], Any]] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change

        self._tiers: Dict[str, TierHealth] = {}
        self._lock = asyncio.Lock()
        self._open_count: Dict[str, int] = {}  # Track repeated opens for adaptive timeout

        # v93.0: Startup grace period tracking
        self._creation_time = time.time()
        self._startup_logged = False

    def _get_tier(self, tier_name: str) -> TierHealth:
        """Get or create tier health tracker."""
        if tier_name not in self._tiers:
            self._tiers[tier_name] = TierHealth(tier_name=tier_name)
        return self._tiers[tier_name]

    def _is_in_startup_grace_period(self) -> bool:
        """v93.0: Check if we're still within the startup grace period."""
        elapsed = time.time() - self._creation_time
        return elapsed < self.config.startup_grace_period_seconds

    def _get_failure_threshold(self, failure_type: FailureType) -> int:
        """Get failure threshold based on failure type and startup state."""
        # v93.0: Use much higher thresholds during startup
        in_startup = self._is_in_startup_grace_period()

        if in_startup:
            # Log startup state once
            if not self._startup_logged:
                elapsed = time.time() - self._creation_time
                logger.info(
                    f"[{self.name}] Circuit breaker in startup grace period "
                    f"({elapsed:.0f}s / {self.config.startup_grace_period_seconds}s), "
                    f"using higher thresholds"
                )
                self._startup_logged = True

            # Higher thresholds during startup
            if failure_type == FailureType.NETWORK:
                return self.config.startup_network_failure_threshold
            return self.config.startup_failure_threshold

        if not self.config.adaptive_thresholds:
            return self.config.failure_threshold

        thresholds = {
            FailureType.NETWORK: self.config.network_failure_threshold,
            FailureType.TIMEOUT: self.config.timeout_failure_threshold,
            FailureType.RATE_LIMIT: self.config.rate_limit_failure_threshold,
        }
        return thresholds.get(failure_type, self.config.failure_threshold)

    def _get_timeout(self, tier_name: str) -> float:
        """Get timeout with adaptive increase for repeated failures."""
        open_count = self._open_count.get(tier_name, 0)
        timeout = self.config.timeout_seconds * (self.config.recovery_factor ** open_count)
        return min(timeout, self.config.max_timeout_seconds)

    async def execute(
        self,
        tier: str,
        func: Callable[..., Any],
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        fallback: Optional[Callable[..., Any]] = None,
    ) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            tier: Tier name (e.g., "local_prime", "gcp_vm")
            func: Async function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            fallback: Fallback function if circuit is open

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open and no fallback
            Exception: Original exception if function fails
        """
        kwargs = kwargs or {}

        async with self._lock:
            health = self._get_tier(tier)
            await self._check_state(health)

            if health.state == CircuitState.OPEN:
                timeout = self._get_timeout(tier)
                reason = health.dominant_failure_type.value if health.dominant_failure_type else None

                # v93.0: During startup grace period, allow request to proceed (service may be starting)
                if self._is_in_startup_grace_period():
                    elapsed = time.time() - self._creation_time
                    logger.debug(
                        f"[CircuitBreaker:{self.name}] {tier} is OPEN but in startup grace period "
                        f"({elapsed:.0f}s / {self.config.startup_grace_period_seconds}s), allowing request"
                    )
                    # Don't block - let the request try (fall through to execute)
                elif fallback:
                    logger.debug(
                        f"[CircuitBreaker:{self.name}] {tier} is OPEN, using fallback"
                    )
                    result = fallback(*args, **kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    return result
                else:
                    raise CircuitOpenError(tier, health.state, timeout, reason)

            if health.state == CircuitState.HALF_OPEN:
                if health.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitOpenError(
                        tier,
                        health.state,
                        1.0,
                        "Half-open max calls reached",
                    )
                health.half_open_calls += 1

        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            latency_ms = (time.time() - start_time) * 1000
            await self._on_success(tier, latency_ms)
            return result

        except Exception as e:
            failure_type = FailureClassifier.classify(e)
            await self._on_failure(tier, failure_type)
            raise

    async def _check_state(self, health: TierHealth) -> None:
        """Check and potentially update circuit state."""
        if health.state == CircuitState.OPEN:
            if health.opened_at:
                timeout = self._get_timeout(health.tier_name)
                elapsed = time.time() - health.opened_at

                if elapsed >= timeout:
                    await self._transition_to(health, CircuitState.HALF_OPEN)

    async def _on_success(self, tier: str, latency_ms: float) -> None:
        """Handle successful operation."""
        async with self._lock:
            health = self._get_tier(tier)
            health.record_success(latency_ms)

            if health.state == CircuitState.HALF_OPEN:
                if health.consecutive_successes >= self.config.success_threshold:
                    await self._transition_to(health, CircuitState.CLOSED)
                    # Reset open count on successful recovery
                    self._open_count[tier] = 0

    async def _on_failure(self, tier: str, failure_type: FailureType) -> None:
        """Handle failed operation."""
        async with self._lock:
            health = self._get_tier(tier)
            health.record_failure(failure_type)

            threshold = self._get_failure_threshold(failure_type)

            if health.state == CircuitState.CLOSED:
                if health.consecutive_failures >= threshold:
                    await self._transition_to(health, CircuitState.OPEN)
                    self._open_count[tier] = self._open_count.get(tier, 0) + 1

            elif health.state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens
                await self._transition_to(health, CircuitState.OPEN)
                self._open_count[tier] = self._open_count.get(tier, 0) + 1

    async def _transition_to(self, health: TierHealth, new_state: CircuitState) -> None:
        """Transition tier to new state."""
        old_state = health.state
        if old_state == new_state:
            return

        health.state = new_state

        if new_state == CircuitState.OPEN:
            health.opened_at = time.time()
            health.half_open_calls = 0
            logger.warning(
                f"[CircuitBreaker:{self.name}:{health.tier_name}] "
                f"OPENED after {health.consecutive_failures} failures "
                f"(dominant: {health.dominant_failure_type.value if health.dominant_failure_type else 'unknown'})"
            )

        elif new_state == CircuitState.HALF_OPEN:
            health.half_open_calls = 0
            logger.info(
                f"[CircuitBreaker:{self.name}:{health.tier_name}] "
                f"Entering HALF_OPEN for recovery test"
            )

        elif new_state == CircuitState.CLOSED:
            health.opened_at = None
            health.consecutive_failures = 0
            logger.info(
                f"[CircuitBreaker:{self.name}:{health.tier_name}] "
                f"CLOSED - recovered successfully"
            )

        # Callback
        if self._on_state_change:
            try:
                result = self._on_state_change(health.tier_name, old_state, new_state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[CircuitBreaker] State change callback error: {e}")

    def get_tier_health(self, tier: str) -> TierHealth:
        """Get health status for a tier."""
        return self._get_tier(tier)

    def get_healthy_tiers(self) -> List[str]:
        """Get list of healthy tiers (CLOSED state)."""
        return [
            name for name, health in self._tiers.items()
            if health.state == CircuitState.CLOSED
        ]

    def get_available_tiers(self) -> List[str]:
        """Get list of available tiers (CLOSED or HALF_OPEN)."""
        return [
            name for name, health in self._tiers.items()
            if health.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
        ]

    async def force_open(self, tier: str) -> None:
        """Manually open a tier's circuit."""
        async with self._lock:
            health = self._get_tier(tier)
            await self._transition_to(health, CircuitState.OPEN)

    async def force_close(self, tier: str) -> None:
        """Manually close a tier's circuit."""
        async with self._lock:
            health = self._get_tier(tier)
            await self._transition_to(health, CircuitState.CLOSED)
            self._open_count[tier] = 0

    async def reset(self, tier: Optional[str] = None) -> None:
        """Reset circuit breaker state."""
        async with self._lock:
            if tier:
                if tier in self._tiers:
                    self._tiers[tier] = TierHealth(tier_name=tier)
                    self._open_count.pop(tier, None)
            else:
                self._tiers.clear()
                self._open_count.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get overall circuit breaker status."""
        all_healthy = all(
            h.state == CircuitState.CLOSED for h in self._tiers.values()
        )
        any_open = any(
            h.state == CircuitState.OPEN for h in self._tiers.values()
        )
        in_startup = self._is_in_startup_grace_period()
        elapsed = time.time() - self._creation_time

        return {
            "name": self.name,
            "overall_healthy": all_healthy,
            "any_circuit_open": any_open,
            "total_tiers": len(self._tiers),
            "healthy_tiers": len(self.get_healthy_tiers()),
            "available_tiers": len(self.get_available_tiers()),
            # v93.0: Include startup status
            "in_startup_grace_period": in_startup,
            "startup_elapsed_seconds": round(elapsed, 1),
            "startup_grace_period_seconds": self.config.startup_grace_period_seconds,
            "tiers": {
                name: health.to_dict()
                for name, health in self._tiers.items()
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "adaptive_thresholds": self.config.adaptive_thresholds,
                # v93.0: Include startup config
                "startup_failure_threshold": self.config.startup_failure_threshold,
                "startup_network_failure_threshold": self.config.startup_network_failure_threshold,
            },
        }


# Global registry
_breakers: Dict[str, CrossRepoCircuitBreaker] = {}


def get_cross_repo_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CrossRepoCircuitBreaker:
    """Get or create a cross-repo circuit breaker."""
    if name not in _breakers:
        _breakers[name] = CrossRepoCircuitBreaker(name, config)
    return _breakers[name]


def get_all_breaker_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers."""
    return {name: breaker.get_status() for name, breaker in _breakers.items()}
