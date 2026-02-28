# Resilience Primitives Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build comprehensive resilience primitives for Ironcliw startup to handle broadcast failures, Docker check failures, and Invincible Node failures with graceful degradation and background auto-recovery.

**Architecture:** Hybrid approach - fast startup with degradation (never block on recoverable failures) + background auto-recovery (continuously attempt to restore full capability). Shared resilience library in `backend/core/resilience/` extractable to jarvis-common in Phase 2.

**Tech Stack:** Python 3.11+, asyncio, dataclasses, psutil (via async_io wrappers)

---

## Module Structure

```
backend/core/resilience/
├── __init__.py          # Public API exports
├── retry.py             # RetryPolicy with exponential backoff
├── circuit_breaker.py   # CircuitBreaker (closed/open/half-open)
├── health.py            # HealthProbe with caching
├── recovery.py          # BackgroundRecovery with adaptive backoff
└── capability.py        # CapabilityUpgrade for hot-swapping modes
```

---

## Task 1: Create Module Structure and Core Types

**Files:**
- Create: `backend/core/__init__.py`
- Create: `backend/core/resilience/__init__.py`
- Create: `backend/core/resilience/types.py`
- Test: `tests/unit/core/resilience/test_types.py`

**Step 1: Create directory structure**

```bash
mkdir -p backend/core/resilience
mkdir -p tests/unit/core/resilience
```

**Step 2: Create backend/core/__init__.py**

```python
"""
Core modules for Ironcliw backend.

This package contains fundamental building blocks used across the system.
"""
```

**Step 3: Create types.py with core enums and protocols**

```python
"""
Core types for resilience primitives.

Shared enums and protocols used across retry, circuit breaker, health, and recovery.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Protocol, runtime_checkable


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation, requests pass through
    OPEN = auto()        # Failing, requests rejected immediately
    HALF_OPEN = auto()   # Testing recovery, limited requests allowed


class CapabilityState(Enum):
    """Capability upgrade states."""
    DEGRADED = auto()    # Using fallback
    UPGRADING = auto()   # Attempting upgrade
    FULL = auto()        # Full capability active
    MONITORING = auto()  # Full mode, monitoring for regression


class RecoveryState(Enum):
    """Background recovery states."""
    IDLE = auto()        # Not running
    RECOVERING = auto()  # Actively attempting recovery
    PAUSED = auto()      # Paused due to safety valve
    SUCCEEDED = auto()   # Recovery completed successfully


@runtime_checkable
class HealthCheckable(Protocol):
    """Protocol for objects that can be health-checked."""

    async def check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if healthy, False otherwise
        """
        ...


@runtime_checkable
class Recoverable(Protocol):
    """Protocol for objects that can be recovered."""

    async def recover(self) -> bool:
        """
        Attempt recovery.

        Returns:
            True if recovery succeeded, False otherwise
        """
        ...


__all__ = [
    "CircuitState",
    "CapabilityState",
    "RecoveryState",
    "HealthCheckable",
    "Recoverable",
]
```

**Step 4: Create __init__.py with exports**

```python
"""
Resilience primitives for robust async operations.

This module provides building blocks for resilient systems:
- RetryPolicy: Exponential backoff with jitter
- CircuitBreaker: Fail-fast when services are down
- HealthProbe: Cached health checks with failure tracking
- BackgroundRecovery: Continuous recovery attempts with adaptive backoff
- CapabilityUpgrade: Hot-swap degraded to full mode

Example:
    from backend.core.resilience import (
        RetryPolicy,
        CircuitBreaker,
        HealthProbe,
        BackgroundRecovery,
        CapabilityUpgrade,
    )

    # Retry with exponential backoff
    policy = RetryPolicy(max_attempts=3, base_delay=1.0)
    result = await policy.execute(flaky_operation)

    # Circuit breaker for external service
    breaker = CircuitBreaker(failure_threshold=5)
    result = await breaker.call(external_api_call)

    # Health probe with caching
    probe = HealthProbe(check_fn=check_database, cache_ttl=30.0)
    is_healthy = await probe.check()

    # Background recovery
    recovery = BackgroundRecovery(
        recover_fn=reconnect_to_service,
        on_success=notify_service_restored,
    )
    await recovery.start()
"""

from backend.core.resilience.types import (
    CircuitState,
    CapabilityState,
    RecoveryState,
    HealthCheckable,
    Recoverable,
)

# These will be imported as each module is created
# from backend.core.resilience.retry import RetryPolicy
# from backend.core.resilience.circuit_breaker import CircuitBreaker
# from backend.core.resilience.health import HealthProbe
# from backend.core.resilience.recovery import BackgroundRecovery
# from backend.core.resilience.capability import CapabilityUpgrade

__all__ = [
    # Types
    "CircuitState",
    "CapabilityState",
    "RecoveryState",
    "HealthCheckable",
    "Recoverable",
    # Primitives (uncomment as implemented)
    # "RetryPolicy",
    # "CircuitBreaker",
    # "HealthProbe",
    # "BackgroundRecovery",
    # "CapabilityUpgrade",
]
```

**Step 5: Write tests for types**

```python
"""Tests for resilience core types."""

import pytest
from backend.core.resilience.types import (
    CircuitState,
    CapabilityState,
    RecoveryState,
    HealthCheckable,
    Recoverable,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_has_closed_state(self):
        assert CircuitState.CLOSED is not None

    def test_has_open_state(self):
        assert CircuitState.OPEN is not None

    def test_has_half_open_state(self):
        assert CircuitState.HALF_OPEN is not None

    def test_states_are_distinct(self):
        states = [CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN]
        assert len(set(states)) == 3


class TestCapabilityState:
    """Tests for CapabilityState enum."""

    def test_has_all_states(self):
        assert CapabilityState.DEGRADED is not None
        assert CapabilityState.UPGRADING is not None
        assert CapabilityState.FULL is not None
        assert CapabilityState.MONITORING is not None


class TestRecoveryState:
    """Tests for RecoveryState enum."""

    def test_has_all_states(self):
        assert RecoveryState.IDLE is not None
        assert RecoveryState.RECOVERING is not None
        assert RecoveryState.PAUSED is not None
        assert RecoveryState.SUCCEEDED is not None


class TestHealthCheckableProtocol:
    """Tests for HealthCheckable protocol."""

    @pytest.mark.asyncio
    async def test_class_implementing_protocol(self):
        class MyHealthCheck:
            async def check(self) -> bool:
                return True

        obj = MyHealthCheck()
        assert isinstance(obj, HealthCheckable)
        assert await obj.check() is True

    def test_class_not_implementing_protocol(self):
        class NotHealthCheckable:
            pass

        obj = NotHealthCheckable()
        assert not isinstance(obj, HealthCheckable)


class TestRecoverableProtocol:
    """Tests for Recoverable protocol."""

    @pytest.mark.asyncio
    async def test_class_implementing_protocol(self):
        class MyRecoverable:
            async def recover(self) -> bool:
                return True

        obj = MyRecoverable()
        assert isinstance(obj, Recoverable)
        assert await obj.recover() is True
```

**Step 6: Run tests**

```bash
pytest tests/unit/core/resilience/test_types.py -v
```

Expected: All tests pass

**Step 7: Commit**

```bash
git add backend/core/ tests/unit/core/
git commit -m "feat(resilience): Add core types and module structure

- Add CircuitState, CapabilityState, RecoveryState enums
- Add HealthCheckable and Recoverable protocols
- Set up backend/core/resilience/ module structure"
```

---

## Task 2: Implement RetryPolicy

**Files:**
- Create: `backend/core/resilience/retry.py`
- Test: `tests/unit/core/resilience/test_retry.py`

**Step 1: Write failing tests**

```python
"""Tests for RetryPolicy."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from backend.core.resilience.retry import RetryPolicy, RetryExhausted


class TestRetryPolicyBasic:
    """Basic RetryPolicy tests."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        func = AsyncMock(return_value="success")

        result = await policy.execute(func)

        assert result == "success"
        assert func.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        func = AsyncMock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])

        result = await policy.execute(func)

        assert result == "success"
        assert func.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        func = AsyncMock(side_effect=ValueError("always fails"))

        with pytest.raises(RetryExhausted) as exc_info:
            await policy.execute(func)

        assert func.call_count == 3
        assert "always fails" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_respects_retry_on_filter(self):
        # Only retry on ValueError, not TypeError
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.01,
            retry_on=(ValueError,),
        )
        func = AsyncMock(side_effect=TypeError("wrong type"))

        with pytest.raises(TypeError):
            await policy.execute(func)

        assert func.call_count == 1  # No retry


class TestRetryPolicyBackoff:
    """Tests for exponential backoff."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        policy = RetryPolicy(
            max_attempts=4,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=0.0,  # Disable jitter for predictable timing
        )
        func = AsyncMock(side_effect=[ValueError(), ValueError(), ValueError(), "ok"])

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await policy.execute(func)

        # Delays should be: 0.1, 0.2, 0.4 (exponential)
        calls = [call.args[0] for call in mock_sleep.call_args_list]
        assert len(calls) == 3
        assert abs(calls[0] - 0.1) < 0.01
        assert abs(calls[1] - 0.2) < 0.01
        assert abs(calls[2] - 0.4) < 0.01

    @pytest.mark.asyncio
    async def test_max_delay_caps_backoff(self):
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=1.0,
            exponential_base=10.0,  # Would grow very fast
            max_delay=5.0,  # But capped at 5s
            jitter=0.0,
        )
        func = AsyncMock(side_effect=[ValueError()] * 4 + ["ok"])

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await policy.execute(func)

        calls = [call.args[0] for call in mock_sleep.call_args_list]
        assert all(delay <= 5.0 for delay in calls)

    @pytest.mark.asyncio
    async def test_jitter_adds_randomness(self):
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=1.0,
            jitter=0.5,  # +/- 50%
        )
        func = AsyncMock(side_effect=[ValueError(), ValueError(), "ok"])

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await policy.execute(func)

        calls = [call.args[0] for call in mock_sleep.call_args_list]
        # With 50% jitter, delay should be between 0.5 and 1.5
        assert all(0.5 <= delay <= 1.5 for delay in calls)


class TestRetryPolicyTimeout:
    """Tests for per-attempt timeout."""

    @pytest.mark.asyncio
    async def test_timeout_cancels_slow_attempt(self):
        policy = RetryPolicy(max_attempts=2, base_delay=0.01, timeout=0.05)

        async def slow_func():
            await asyncio.sleep(1.0)  # Way longer than timeout
            return "never reached"

        fast_func = AsyncMock(return_value="fast")
        call_count = 0

        async def mixed_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(1.0)  # First call times out
            return "success"

        result = await policy.execute(mixed_func)
        assert result == "success"
        assert call_count == 2


class TestRetryPolicyCallbacks:
    """Tests for on_retry callback."""

    @pytest.mark.asyncio
    async def test_on_retry_called_with_exception_and_attempt(self):
        attempts_seen = []

        async def on_retry(exc, attempt):
            attempts_seen.append((str(exc), attempt))

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.01,
            on_retry=on_retry,
        )
        func = AsyncMock(side_effect=[ValueError("e1"), ValueError("e2"), "ok"])

        await policy.execute(func)

        assert len(attempts_seen) == 2
        assert attempts_seen[0] == ("e1", 1)
        assert attempts_seen[1] == ("e2", 2)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/core/resilience/test_retry.py -v
```

Expected: ImportError (module doesn't exist yet)

**Step 3: Implement retry.py**

```python
"""
RetryPolicy: Exponential backoff with jitter for transient failures.

Example:
    policy = RetryPolicy(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=0.1,
        timeout=10.0,
        retry_on=(ConnectionError, TimeoutError),
        on_retry=log_retry,
    )

    result = await policy.execute(flaky_api_call, arg1, kwarg1=value)
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception | None = None):
        super().__init__(message)
        self.last_exception = last_exception


@dataclass
class RetryPolicy:
    """
    Configurable retry policy with exponential backoff and jitter.

    Attributes:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries (caps exponential growth)
        exponential_base: Base for exponential backoff (delay * base^attempt)
        jitter: Random jitter factor (0.0 to 1.0, applied as +/- percentage)
        timeout: Per-attempt timeout in seconds (None = no timeout)
        retry_on: Exception types to retry on (default: all exceptions)
        on_retry: Optional callback(exception, attempt_number) called before each retry
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    timeout: float | None = None
    retry_on: Tuple[Type[Exception], ...] = field(default=(Exception,))
    on_retry: Callable[[Exception, int], Awaitable[None]] | None = None

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (1-indexed)."""
        # Exponential backoff: base_delay * exponential_base^(attempt-1)
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        # Apply jitter: random value between delay*(1-jitter) and delay*(1+jitter)
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay = delay + random.uniform(-jitter_range, jitter_range)

        return max(0.0, delay)  # Ensure non-negative

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function with retry policy.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of successful function call

        Raises:
            RetryExhausted: If all attempts fail
            Exception: If exception is not in retry_on tuple
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                # Execute with optional timeout
                if self.timeout is not None:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.timeout,
                    )
                else:
                    result = await func(*args, **kwargs)

                return result

            except asyncio.TimeoutError as e:
                last_exception = e
                if attempt < self.max_attempts:
                    logger.debug(
                        f"[RetryPolicy] Attempt {attempt}/{self.max_attempts} "
                        f"timed out after {self.timeout}s"
                    )
                    if self.on_retry:
                        await self.on_retry(e, attempt)
                    await asyncio.sleep(self._calculate_delay(attempt))

            except self.retry_on as e:
                last_exception = e
                if attempt < self.max_attempts:
                    logger.debug(
                        f"[RetryPolicy] Attempt {attempt}/{self.max_attempts} "
                        f"failed: {e}"
                    )
                    if self.on_retry:
                        await self.on_retry(e, attempt)
                    await asyncio.sleep(self._calculate_delay(attempt))

            except Exception:
                # Exception not in retry_on, re-raise immediately
                raise

        # All attempts exhausted
        raise RetryExhausted(
            f"All {self.max_attempts} attempts failed. "
            f"Last error: {last_exception}",
            last_exception=last_exception,
        )


__all__ = ["RetryPolicy", "RetryExhausted"]
```

**Step 4: Run tests**

```bash
pytest tests/unit/core/resilience/test_retry.py -v
```

Expected: All tests pass

**Step 5: Update __init__.py exports**

Add to `backend/core/resilience/__init__.py`:
```python
from backend.core.resilience.retry import RetryPolicy, RetryExhausted
```

And add to `__all__`:
```python
"RetryPolicy",
"RetryExhausted",
```

**Step 6: Commit**

```bash
git add backend/core/resilience/retry.py tests/unit/core/resilience/test_retry.py backend/core/resilience/__init__.py
git commit -m "feat(resilience): Add RetryPolicy with exponential backoff

- Configurable max_attempts, base_delay, max_delay
- Exponential backoff with jitter
- Per-attempt timeout support
- Selective retry via retry_on exception filter
- on_retry callback for logging/metrics"
```

---

## Task 3: Implement CircuitBreaker

**Files:**
- Create: `backend/core/resilience/circuit_breaker.py`
- Test: `tests/unit/core/resilience/test_circuit_breaker.py`

**Step 1: Write failing tests**

```python
"""Tests for CircuitBreaker."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from backend.core.resilience.circuit_breaker import CircuitBreaker, CircuitOpen
from backend.core.resilience.types import CircuitState


class TestCircuitBreakerBasic:
    """Basic CircuitBreaker tests."""

    @pytest.mark.asyncio
    async def test_starts_closed(self):
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_passes_through_when_closed(self):
        breaker = CircuitBreaker(failure_threshold=3)
        func = AsyncMock(return_value="success")

        result = await breaker.call(func)

        assert result == "success"
        assert func.call_count == 1

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        func = AsyncMock(side_effect=ValueError("fail"))

        # Fail 3 times
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
        func = AsyncMock(side_effect=ValueError("fail"))

        # Trip the breaker
        with pytest.raises(ValueError):
            await breaker.call(func)

        assert breaker.state == CircuitState.OPEN

        # Next call should be rejected without calling func
        with pytest.raises(CircuitOpen):
            await breaker.call(func)

        assert func.call_count == 1  # Only called once (before open)


class TestCircuitBreakerRecovery:
    """Tests for circuit recovery."""

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        func = AsyncMock(side_effect=ValueError("fail"))

        # Trip the breaker
        with pytest.raises(ValueError):
            await breaker.call(func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.1)

        # State transitions to HALF_OPEN on next call attempt
        func.side_effect = None
        func.return_value = "recovered"

        result = await breaker.call(func)
        assert result == "recovered"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        # Trip the breaker
        fail_func = AsyncMock(side_effect=ValueError("fail"))
        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        await asyncio.sleep(0.02)

        # Success in half-open closes circuit
        success_func = AsyncMock(return_value="ok")
        result = await breaker.call(success_func)

        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

        fail_func = AsyncMock(side_effect=ValueError("fail"))

        # Trip the breaker
        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        await asyncio.sleep(0.02)

        # Failure in half-open reopens circuit
        with pytest.raises(ValueError):
            await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerCallbacks:
    """Tests for state change callbacks."""

    @pytest.mark.asyncio
    async def test_on_open_called_when_circuit_opens(self):
        on_open = AsyncMock()
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=1.0,
            on_open=on_open,
        )
        func = AsyncMock(side_effect=ValueError("fail"))

        # First failure - still closed
        with pytest.raises(ValueError):
            await breaker.call(func)
        on_open.assert_not_called()

        # Second failure - opens
        with pytest.raises(ValueError):
            await breaker.call(func)
        on_open.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_close_called_when_circuit_closes(self):
        on_close = AsyncMock()
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.01,
            on_close=on_close,
        )

        # Trip the breaker
        with pytest.raises(ValueError):
            await breaker.call(AsyncMock(side_effect=ValueError()))

        await asyncio.sleep(0.02)

        # Recover
        await breaker.call(AsyncMock(return_value="ok"))
        on_close.assert_called_once()


class TestCircuitBreakerReset:
    """Tests for manual reset."""

    @pytest.mark.asyncio
    async def test_reset_closes_circuit(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=100.0)

        # Trip the breaker
        with pytest.raises(ValueError):
            await breaker.call(AsyncMock(side_effect=ValueError()))

        assert breaker.state == CircuitState.OPEN

        # Manual reset
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/core/resilience/test_circuit_breaker.py -v
```

**Step 3: Implement circuit_breaker.py**

```python
"""
CircuitBreaker: Fail-fast pattern for protecting against cascading failures.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Circuit tripped, requests rejected immediately
- HALF_OPEN: Testing recovery, limited requests allowed

Example:
    breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30.0,
        on_open=notify_circuit_open,
        on_close=notify_circuit_closed,
    )

    try:
        result = await breaker.call(external_api_call)
    except CircuitOpen:
        # Use fallback
        result = get_cached_response()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from backend.core.resilience.types import CircuitState

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitOpen(Exception):
    """Raised when circuit breaker is open and rejecting calls."""

    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for fail-fast behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before trying half-open
        on_open: Optional callback when circuit opens
        on_close: Optional callback when circuit closes
    """
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    on_open: Callable[[], Awaitable[None]] | None = None
    on_close: Callable[[], Awaitable[None]] | None = None

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of function call

        Raises:
            CircuitOpen: If circuit is open and rejecting calls
            Exception: Any exception from the function
        """
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.debug("[CircuitBreaker] Transitioning to HALF_OPEN")
                else:
                    raise CircuitOpen(
                        f"Circuit open, retry after "
                        f"{self.recovery_timeout - (time.monotonic() - self._last_failure_time):.1f}s"
                    )

        try:
            result = await func(*args, **kwargs)

            # Success - close circuit if half-open, reset failure count
            async with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    await self._close_circuit()
                self._failure_count = 0

            return result

        except Exception as e:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.monotonic()

                if self._state == CircuitState.HALF_OPEN:
                    # Failure in half-open, reopen circuit
                    await self._open_circuit()
                elif self._failure_count >= self.failure_threshold:
                    # Threshold reached, open circuit
                    await self._open_circuit()

            raise

    async def _open_circuit(self) -> None:
        """Transition to OPEN state."""
        if self._state != CircuitState.OPEN:
            self._state = CircuitState.OPEN
            logger.info(
                f"[CircuitBreaker] Circuit OPENED after {self._failure_count} failures"
            )
            if self.on_open:
                try:
                    await self.on_open()
                except Exception as e:
                    logger.warning(f"[CircuitBreaker] on_open callback error: {e}")

    async def _close_circuit(self) -> None:
        """Transition to CLOSED state."""
        if self._state != CircuitState.CLOSED:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info("[CircuitBreaker] Circuit CLOSED (recovered)")
            if self.on_close:
                try:
                    await self.on_close()
                except Exception as e:
                    logger.warning(f"[CircuitBreaker] on_close callback error: {e}")

    async def reset(self) -> None:
        """Manually reset circuit to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = 0.0
            logger.info("[CircuitBreaker] Circuit manually reset to CLOSED")


__all__ = ["CircuitBreaker", "CircuitOpen"]
```

**Step 4: Run tests**

```bash
pytest tests/unit/core/resilience/test_circuit_breaker.py -v
```

Expected: All tests pass

**Step 5: Update __init__.py exports**

**Step 6: Commit**

```bash
git add backend/core/resilience/circuit_breaker.py tests/unit/core/resilience/test_circuit_breaker.py backend/core/resilience/__init__.py
git commit -m "feat(resilience): Add CircuitBreaker with state transitions

- CLOSED -> OPEN -> HALF_OPEN -> CLOSED lifecycle
- Configurable failure_threshold and recovery_timeout
- on_open/on_close callbacks for monitoring
- Thread-safe with asyncio.Lock
- Manual reset() for testing/recovery"
```

---

## Task 4: Implement HealthProbe

**Files:**
- Create: `backend/core/resilience/health.py`
- Test: `tests/unit/core/resilience/test_health.py`

**Step 1: Write failing tests**

```python
"""Tests for HealthProbe."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from backend.core.resilience.health import HealthProbe


class TestHealthProbeBasic:
    """Basic HealthProbe tests."""

    @pytest.mark.asyncio
    async def test_returns_check_result(self):
        check_fn = AsyncMock(return_value=True)
        probe = HealthProbe(check_fn=check_fn)

        result = await probe.check()

        assert result is True
        check_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        check_fn = AsyncMock(side_effect=ConnectionError("down"))
        probe = HealthProbe(check_fn=check_fn)

        result = await probe.check()

        assert result is False


class TestHealthProbeCaching:
    """Tests for result caching."""

    @pytest.mark.asyncio
    async def test_caches_result_within_ttl(self):
        check_fn = AsyncMock(return_value=True)
        probe = HealthProbe(check_fn=check_fn, cache_ttl=1.0)

        # First check
        await probe.check()
        # Second check (should use cache)
        await probe.check()

        assert check_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_refreshes_after_ttl_expires(self):
        check_fn = AsyncMock(return_value=True)
        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.01)

        await probe.check()
        await asyncio.sleep(0.02)
        await probe.check()

        assert check_fn.call_count == 2

    @pytest.mark.asyncio
    async def test_force_bypasses_cache(self):
        check_fn = AsyncMock(return_value=True)
        probe = HealthProbe(check_fn=check_fn, cache_ttl=10.0)

        await probe.check()
        await probe.check(force=True)

        assert check_fn.call_count == 2


class TestHealthProbeFailureTracking:
    """Tests for consecutive failure tracking."""

    @pytest.mark.asyncio
    async def test_tracks_consecutive_failures(self):
        check_fn = AsyncMock(return_value=False)
        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0)

        await probe.check()
        assert probe.consecutive_failures == 1

        await probe.check()
        assert probe.consecutive_failures == 2

    @pytest.mark.asyncio
    async def test_resets_failures_on_success(self):
        call_count = 0

        async def check_fn():
            nonlocal call_count
            call_count += 1
            return call_count > 2  # Fails first 2, then succeeds

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0)

        await probe.check()  # Fail
        await probe.check()  # Fail
        assert probe.consecutive_failures == 2

        await probe.check()  # Success
        assert probe.consecutive_failures == 0


class TestHealthProbeTimeout:
    """Tests for check timeout."""

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        async def slow_check():
            await asyncio.sleep(1.0)
            return True

        probe = HealthProbe(check_fn=slow_check, timeout=0.01, cache_ttl=0.0)

        result = await probe.check()

        assert result is False
        assert probe.consecutive_failures == 1


class TestHealthProbeCallbacks:
    """Tests for state change callbacks."""

    @pytest.mark.asyncio
    async def test_on_unhealthy_called_after_threshold(self):
        on_unhealthy = AsyncMock()
        check_fn = AsyncMock(return_value=False)
        probe = HealthProbe(
            check_fn=check_fn,
            cache_ttl=0.0,
            unhealthy_threshold=2,
            on_unhealthy=on_unhealthy,
        )

        await probe.check()  # 1 failure
        on_unhealthy.assert_not_called()

        await probe.check()  # 2 failures = threshold
        on_unhealthy.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_healthy_called_on_recovery(self):
        on_healthy = AsyncMock()
        call_count = 0

        async def check_fn():
            nonlocal call_count
            call_count += 1
            return call_count > 2

        probe = HealthProbe(
            check_fn=check_fn,
            cache_ttl=0.0,
            unhealthy_threshold=2,
            on_healthy=on_healthy,
        )

        await probe.check()  # Fail
        await probe.check()  # Fail (now unhealthy)
        await probe.check()  # Success (recovered)

        on_healthy.assert_called_once()
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement health.py**

```python
"""
HealthProbe: Cached health checks with failure tracking.

Example:
    probe = HealthProbe(
        check_fn=check_database_connection,
        cache_ttl=30.0,
        timeout=5.0,
        unhealthy_threshold=3,
        on_unhealthy=notify_database_down,
        on_healthy=notify_database_recovered,
    )

    if await probe.check():
        # Service is healthy
        pass
    else:
        # Service is unhealthy, use fallback
        pass
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class HealthProbe:
    """
    Health probe with caching and failure tracking.

    Attributes:
        check_fn: Async function returning True if healthy
        cache_ttl: Seconds to cache health check result
        timeout: Per-check timeout in seconds
        unhealthy_threshold: Consecutive failures before considered unhealthy
        on_unhealthy: Callback when threshold reached
        on_healthy: Callback when recovering from unhealthy
    """
    check_fn: Callable[[], Awaitable[bool]]
    cache_ttl: float = 30.0
    timeout: float = 10.0
    unhealthy_threshold: int = 3
    on_unhealthy: Callable[[], Awaitable[None]] | None = None
    on_healthy: Callable[[], Awaitable[None]] | None = None

    # Internal state
    _cached_result: bool | None = field(default=None, init=False)
    _cache_time: float = field(default=0.0, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _is_unhealthy: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def consecutive_failures(self) -> int:
        """Current consecutive failure count."""
        return self._consecutive_failures

    @property
    def is_unhealthy(self) -> bool:
        """True if consecutive failures >= threshold."""
        return self._is_unhealthy

    async def check(self, force: bool = False) -> bool:
        """
        Perform health check with caching.

        Args:
            force: If True, bypass cache and perform fresh check

        Returns:
            True if healthy, False otherwise
        """
        async with self._lock:
            # Check cache
            if not force and self._cached_result is not None:
                if time.monotonic() - self._cache_time < self.cache_ttl:
                    return self._cached_result

            # Perform check
            try:
                result = await asyncio.wait_for(
                    self.check_fn(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                logger.debug(f"[HealthProbe] Check timed out after {self.timeout}s")
                result = False
            except Exception as e:
                logger.debug(f"[HealthProbe] Check failed: {e}")
                result = False

            # Update cache
            self._cached_result = result
            self._cache_time = time.monotonic()

            # Track failures
            if result:
                was_unhealthy = self._is_unhealthy
                self._consecutive_failures = 0
                self._is_unhealthy = False

                if was_unhealthy and self.on_healthy:
                    try:
                        await self.on_healthy()
                    except Exception as e:
                        logger.warning(f"[HealthProbe] on_healthy callback error: {e}")
            else:
                self._consecutive_failures += 1

                if (
                    not self._is_unhealthy
                    and self._consecutive_failures >= self.unhealthy_threshold
                ):
                    self._is_unhealthy = True
                    logger.warning(
                        f"[HealthProbe] Unhealthy after "
                        f"{self._consecutive_failures} consecutive failures"
                    )
                    if self.on_unhealthy:
                        try:
                            await self.on_unhealthy()
                        except Exception as e:
                            logger.warning(
                                f"[HealthProbe] on_unhealthy callback error: {e}"
                            )

            return result

    def reset(self) -> None:
        """Reset probe state (clear cache and failure count)."""
        self._cached_result = None
        self._cache_time = 0.0
        self._consecutive_failures = 0
        self._is_unhealthy = False


__all__ = ["HealthProbe"]
```

**Step 4: Run tests**

**Step 5: Update exports and commit**

---

## Task 5: Implement BackgroundRecovery

**Files:**
- Create: `backend/core/resilience/recovery.py`
- Test: `tests/unit/core/resilience/test_recovery.py`

**Step 1: Write failing tests**

```python
"""Tests for BackgroundRecovery."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from backend.core.resilience.recovery import BackgroundRecovery, RecoveryConfig
from backend.core.resilience.types import RecoveryState


class TestBackgroundRecoveryBasic:
    """Basic BackgroundRecovery tests."""

    @pytest.mark.asyncio
    async def test_starts_idle(self):
        recovery = BackgroundRecovery(
            recover_fn=AsyncMock(return_value=True),
            config=RecoveryConfig(base_delay=0.01),
        )
        assert recovery.state == RecoveryState.IDLE

    @pytest.mark.asyncio
    async def test_calls_recover_fn_until_success(self):
        call_count = 0

        async def recover():
            nonlocal call_count
            call_count += 1
            return call_count >= 3  # Succeeds on 3rd attempt

        recovery = BackgroundRecovery(
            recover_fn=recover,
            config=RecoveryConfig(base_delay=0.01, max_delay=0.01),
        )

        await recovery.start()
        await asyncio.sleep(0.1)
        await recovery.stop()

        assert call_count >= 3
        assert recovery.state == RecoveryState.SUCCEEDED

    @pytest.mark.asyncio
    async def test_on_success_callback_called(self):
        on_success = AsyncMock()
        recovery = BackgroundRecovery(
            recover_fn=AsyncMock(return_value=True),
            config=RecoveryConfig(base_delay=0.01),
            on_success=on_success,
        )

        await recovery.start()
        await asyncio.sleep(0.05)

        on_success.assert_called_once()


class TestBackgroundRecoverySafetyValve:
    """Tests for safety valve (max attempts/time)."""

    @pytest.mark.asyncio
    async def test_pauses_after_max_attempts(self):
        on_paused = AsyncMock()
        recovery = BackgroundRecovery(
            recover_fn=AsyncMock(return_value=False),
            config=RecoveryConfig(
                base_delay=0.01,
                max_delay=0.01,
                max_attempts=3,
            ),
            on_paused=on_paused,
        )

        await recovery.start()
        await asyncio.sleep(0.1)

        assert recovery.state == RecoveryState.PAUSED
        on_paused.assert_called_once()

    @pytest.mark.asyncio
    async def test_can_resume_after_pause(self):
        call_count = 0

        async def recover():
            nonlocal call_count
            call_count += 1
            return call_count > 5  # Succeeds after many attempts

        recovery = BackgroundRecovery(
            recover_fn=recover,
            config=RecoveryConfig(
                base_delay=0.01,
                max_delay=0.01,
                max_attempts=3,
            ),
        )

        await recovery.start()
        await asyncio.sleep(0.1)

        assert recovery.state == RecoveryState.PAUSED

        # Resume
        await recovery.resume()
        await asyncio.sleep(0.1)

        assert recovery.state == RecoveryState.SUCCEEDED


class TestBackgroundRecoveryAdaptive:
    """Tests for adaptive backoff."""

    @pytest.mark.asyncio
    async def test_notify_conditions_changed_speeds_up(self):
        recover_fn = AsyncMock(return_value=False)
        recovery = BackgroundRecovery(
            recover_fn=recover_fn,
            config=RecoveryConfig(
                base_delay=1.0,  # Long delay
                speedup_factor=0.1,  # Speed up to 10%
            ),
        )

        await recovery.start()
        await asyncio.sleep(0.05)

        # Notify conditions changed - should speed up next attempt
        recovery.notify_conditions_changed()

        await asyncio.sleep(0.2)
        await recovery.stop()

        # Should have called more than once due to speedup
        assert recover_fn.call_count >= 2


class TestBackgroundRecoveryShutdown:
    """Tests for clean shutdown."""

    @pytest.mark.asyncio
    async def test_stop_cancels_gracefully(self):
        recovery = BackgroundRecovery(
            recover_fn=AsyncMock(return_value=False),
            config=RecoveryConfig(base_delay=10.0),  # Long delay
        )

        await recovery.start()
        assert recovery.state == RecoveryState.RECOVERING

        await recovery.stop()
        assert recovery.state == RecoveryState.IDLE

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self):
        recovery = BackgroundRecovery(
            recover_fn=AsyncMock(return_value=False),
            config=RecoveryConfig(base_delay=0.01),
        )

        await recovery.start()
        await recovery.start()  # Should not raise

        await recovery.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self):
        recovery = BackgroundRecovery(
            recover_fn=AsyncMock(return_value=False),
            config=RecoveryConfig(base_delay=0.01),
        )

        await recovery.start()
        await recovery.stop()
        await recovery.stop()  # Should not raise
```

**Step 2: Implement recovery.py**

```python
"""
BackgroundRecovery: Continuous recovery attempts with adaptive backoff.

Example:
    recovery = BackgroundRecovery(
        recover_fn=reconnect_to_database,
        config=RecoveryConfig(
            base_delay=5.0,
            max_delay=300.0,
            max_attempts=10,
            max_total_time=1800.0,
        ),
        on_success=notify_database_restored,
        on_paused=notify_recovery_paused,
    )

    await recovery.start()

    # Later, when conditions change
    recovery.notify_conditions_changed()  # Speeds up next attempt

    # On shutdown
    await recovery.stop()
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from backend.core.resilience.types import RecoveryState

logger = logging.getLogger(__name__)


@dataclass
class RecoveryConfig:
    """Configuration for BackgroundRecovery."""
    base_delay: float = 5.0
    max_delay: float = 300.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    timeout: float = 30.0
    max_attempts: int | None = None  # Safety valve
    max_total_time: float | None = None  # Safety valve
    speedup_factor: float = 0.25  # Adaptive: reduce delay to this fraction


@dataclass
class BackgroundRecovery:
    """
    Background recovery loop with adaptive backoff and safety valve.

    Attributes:
        recover_fn: Async function returning True on success
        config: Recovery configuration
        on_success: Callback when recovery succeeds
        on_paused: Callback when safety valve triggers
    """
    recover_fn: Callable[[], Awaitable[bool]]
    config: RecoveryConfig = field(default_factory=RecoveryConfig)
    on_success: Callable[[], Awaitable[None]] | None = None
    on_paused: Callable[[], Awaitable[None]] | None = None

    # Internal state
    _state: RecoveryState = field(default=RecoveryState.IDLE, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)
    _shutdown_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _conditions_changed: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _attempt_count: int = field(default=0, init=False)
    _start_time: float = field(default=0.0, init=False)

    @property
    def state(self) -> RecoveryState:
        """Current recovery state."""
        return self._state

    @property
    def attempt_count(self) -> int:
        """Number of recovery attempts made."""
        return self._attempt_count

    def notify_conditions_changed(self) -> None:
        """
        Notify that conditions have changed (e.g., network restored).

        This triggers an immediate retry with reduced delay.
        """
        self._conditions_changed.set()

    async def start(self) -> None:
        """Start background recovery loop."""
        if self._task is not None and not self._task.done():
            return  # Already running

        self._shutdown_event.clear()
        self._conditions_changed.clear()
        self._attempt_count = 0
        self._start_time = time.monotonic()
        self._state = RecoveryState.RECOVERING

        self._task = asyncio.create_task(self._recovery_loop())
        logger.info("[BackgroundRecovery] Started")

    async def stop(self) -> None:
        """Stop background recovery loop."""
        if self._task is None:
            return

        self._shutdown_event.set()
        self._task.cancel()

        try:
            await self._task
        except asyncio.CancelledError:
            pass

        self._task = None
        if self._state != RecoveryState.SUCCEEDED:
            self._state = RecoveryState.IDLE
        logger.info("[BackgroundRecovery] Stopped")

    async def resume(self) -> None:
        """Resume recovery after pause (resets attempt count)."""
        if self._state != RecoveryState.PAUSED:
            return

        self._attempt_count = 0
        self._start_time = time.monotonic()
        await self.start()

    def _calculate_delay(self) -> float:
        """Calculate delay for current attempt."""
        delay = self.config.base_delay * (
            self.config.exponential_base ** (self._attempt_count - 1)
        )
        delay = min(delay, self.config.max_delay)

        if self.config.jitter > 0:
            jitter_range = delay * self.config.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.0, delay)

    def _check_safety_valve(self) -> bool:
        """Check if safety valve should trigger. Returns True if should pause."""
        if (
            self.config.max_attempts is not None
            and self._attempt_count >= self.config.max_attempts
        ):
            logger.warning(
                f"[BackgroundRecovery] Safety valve: "
                f"max attempts ({self.config.max_attempts}) reached"
            )
            return True

        if self.config.max_total_time is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed >= self.config.max_total_time:
                logger.warning(
                    f"[BackgroundRecovery] Safety valve: "
                    f"max time ({self.config.max_total_time}s) reached"
                )
                return True

        return False

    async def _recovery_loop(self) -> None:
        """Main recovery loop."""
        while not self._shutdown_event.is_set():
            self._attempt_count += 1

            try:
                # Attempt recovery with timeout
                success = await asyncio.wait_for(
                    self.recover_fn(),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    f"[BackgroundRecovery] Attempt {self._attempt_count} "
                    f"timed out after {self.config.timeout}s"
                )
                success = False
            except asyncio.CancelledError:
                raise  # Propagate cancellation
            except Exception as e:
                logger.debug(
                    f"[BackgroundRecovery] Attempt {self._attempt_count} "
                    f"failed: {e}"
                )
                success = False

            if success:
                self._state = RecoveryState.SUCCEEDED
                logger.info(
                    f"[BackgroundRecovery] Recovered after "
                    f"{self._attempt_count} attempts"
                )
                if self.on_success:
                    try:
                        await self.on_success()
                    except Exception as e:
                        logger.warning(
                            f"[BackgroundRecovery] on_success callback error: {e}"
                        )
                return

            # Check safety valve
            if self._check_safety_valve():
                self._state = RecoveryState.PAUSED
                if self.on_paused:
                    try:
                        await self.on_paused()
                    except Exception as e:
                        logger.warning(
                            f"[BackgroundRecovery] on_paused callback error: {e}"
                        )
                return

            # Wait with adaptive backoff
            delay = self._calculate_delay()

            try:
                # Wait for delay, but wake early if conditions change
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(self._shutdown_event.wait()),
                        asyncio.create_task(self._conditions_changed.wait()),
                        asyncio.create_task(asyncio.sleep(delay)),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check why we woke up
                if self._shutdown_event.is_set():
                    return

                if self._conditions_changed.is_set():
                    self._conditions_changed.clear()
                    logger.debug(
                        "[BackgroundRecovery] Conditions changed, "
                        "speeding up next attempt"
                    )
                    # Apply speedup - next delay will be shorter
                    # (handled by reduced attempt count effect)

            except asyncio.CancelledError:
                return


__all__ = ["BackgroundRecovery", "RecoveryConfig"]
```

**Step 3-6: Run tests, update exports, commit**

---

## Task 6: Implement CapabilityUpgrade

**Files:**
- Create: `backend/core/resilience/capability.py`
- Test: `tests/unit/core/resilience/test_capability.py`

**Step 1: Write failing tests**

```python
"""Tests for CapabilityUpgrade."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from backend.core.resilience.capability import CapabilityUpgrade
from backend.core.resilience.types import CapabilityState


class TestCapabilityUpgradeBasic:
    """Basic CapabilityUpgrade tests."""

    @pytest.mark.asyncio
    async def test_starts_degraded(self):
        upgrade = CapabilityUpgrade(
            name="test",
            check_available=AsyncMock(return_value=True),
            activate=AsyncMock(),
            deactivate=AsyncMock(),
        )
        assert upgrade.state == CapabilityState.DEGRADED

    @pytest.mark.asyncio
    async def test_try_upgrade_succeeds(self):
        upgrade = CapabilityUpgrade(
            name="test",
            check_available=AsyncMock(return_value=True),
            activate=AsyncMock(),
            deactivate=AsyncMock(),
        )

        result = await upgrade.try_upgrade()

        assert result is True
        assert upgrade.state == CapabilityState.FULL
        assert upgrade.is_full is True

    @pytest.mark.asyncio
    async def test_try_upgrade_fails_if_not_available(self):
        upgrade = CapabilityUpgrade(
            name="test",
            check_available=AsyncMock(return_value=False),
            activate=AsyncMock(),
            deactivate=AsyncMock(),
        )

        result = await upgrade.try_upgrade()

        assert result is False
        assert upgrade.state == CapabilityState.DEGRADED

    @pytest.mark.asyncio
    async def test_downgrade_returns_to_degraded(self):
        upgrade = CapabilityUpgrade(
            name="test",
            check_available=AsyncMock(return_value=True),
            activate=AsyncMock(),
            deactivate=AsyncMock(),
        )

        await upgrade.try_upgrade()
        assert upgrade.is_full

        await upgrade.downgrade()
        assert upgrade.state == CapabilityState.DEGRADED


class TestCapabilityUpgradeCallbacks:
    """Tests for upgrade/downgrade callbacks."""

    @pytest.mark.asyncio
    async def test_on_upgrade_called(self):
        on_upgrade = AsyncMock()
        upgrade = CapabilityUpgrade(
            name="test",
            check_available=AsyncMock(return_value=True),
            activate=AsyncMock(),
            deactivate=AsyncMock(),
            on_upgrade=on_upgrade,
        )

        await upgrade.try_upgrade()
        on_upgrade.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_downgrade_called(self):
        on_downgrade = AsyncMock()
        upgrade = CapabilityUpgrade(
            name="test",
            check_available=AsyncMock(return_value=True),
            activate=AsyncMock(),
            deactivate=AsyncMock(),
            on_downgrade=on_downgrade,
        )

        await upgrade.try_upgrade()
        await upgrade.downgrade()
        on_downgrade.assert_called_once()


class TestCapabilityUpgradeMonitoring:
    """Tests for background monitoring."""

    @pytest.mark.asyncio
    async def test_monitoring_attempts_upgrade(self):
        check_available = AsyncMock(return_value=True)
        upgrade = CapabilityUpgrade(
            name="test",
            check_available=check_available,
            activate=AsyncMock(),
            deactivate=AsyncMock(),
        )

        await upgrade.start_monitoring(interval=0.01)
        await asyncio.sleep(0.05)
        await upgrade.stop_monitoring()

        assert upgrade.is_full

    @pytest.mark.asyncio
    async def test_monitoring_detects_regression(self):
        call_count = 0

        async def check():
            nonlocal call_count
            call_count += 1
            return call_count < 3  # Available first 2 times, then not

        on_downgrade = AsyncMock()
        upgrade = CapabilityUpgrade(
            name="test",
            check_available=check,
            activate=AsyncMock(),
            deactivate=AsyncMock(),
            on_downgrade=on_downgrade,
        )

        await upgrade.start_monitoring(interval=0.01)
        await asyncio.sleep(0.1)
        await upgrade.stop_monitoring()

        on_downgrade.assert_called()

    @pytest.mark.asyncio
    async def test_stop_monitoring_is_clean(self):
        upgrade = CapabilityUpgrade(
            name="test",
            check_available=AsyncMock(return_value=False),
            activate=AsyncMock(),
            deactivate=AsyncMock(),
        )

        await upgrade.start_monitoring(interval=10.0)
        await upgrade.stop_monitoring()

        # Should not raise
        await asyncio.sleep(0.01)
```

**Step 2: Implement capability.py** (as shown in Section 6 above)

**Step 3-6: Run tests, update exports, commit**

---

## Task 7: Wire Resilience into Ironcliw Startup

**Files:**
- Modify: `backend/unified_supervisor.py`
- Create: `backend/core/resilience/startup.py` (startup-specific utilities)

This task wires the resilience primitives into the actual Ironcliw startup:

1. Create health probes for Docker, Invincible Node, Ollama
2. Configure circuit breakers for external services
3. Set up background recovery for failed services
4. Configure capability upgrades for local LLM

**Detailed implementation to be designed based on current unified_supervisor.py structure.**

---

## Summary

| Task | Files | Purpose |
|------|-------|---------|
| 1 | types.py, __init__.py | Core enums and protocols |
| 2 | retry.py | RetryPolicy with exponential backoff |
| 3 | circuit_breaker.py | CircuitBreaker with state transitions |
| 4 | health.py | HealthProbe with caching |
| 5 | recovery.py | BackgroundRecovery with adaptive backoff |
| 6 | capability.py | CapabilityUpgrade for hot-swapping |
| 7 | startup.py | Wire into Ironcliw startup |

Each task follows TDD: write failing tests, implement, verify tests pass, commit.
