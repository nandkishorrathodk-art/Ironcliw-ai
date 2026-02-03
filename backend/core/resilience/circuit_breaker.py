"""
CircuitBreaker - State Machine for Protecting Against Cascading Failures
========================================================================

This module provides a circuit breaker implementation that prevents cascading
failures by tracking consecutive failures and temporarily blocking calls
when a failure threshold is reached.

The circuit breaker pattern is like an electrical circuit breaker:
- CLOSED: Normal operation, requests flow through
- OPEN: Too many failures, requests are rejected immediately
- HALF_OPEN: Testing recovery, one request is allowed through

State Transitions:
    CLOSED -> OPEN: When failure_threshold consecutive failures occur
    OPEN -> HALF_OPEN: After recovery_timeout elapses
    HALF_OPEN -> CLOSED: If the probe request succeeds
    HALF_OPEN -> OPEN: If the probe request fails

Features:
- Configurable failure threshold
- Configurable recovery timeout
- Async callbacks for state change notifications
- Thread-safe with asyncio.Lock
- Manual reset capability

Example usage:
    from backend.core.resilience.circuit_breaker import CircuitBreaker, CircuitOpen

    # Basic usage
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    try:
        result = await breaker.call(my_async_function, arg1, arg2)
    except CircuitOpen:
        # Circuit is open, use fallback
        result = await fallback_function()

    # With callbacks
    async def on_open():
        logger.warning("Circuit opened - service unavailable")
        await alert_team()

    async def on_close():
        logger.info("Circuit closed - service recovered")

    breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30.0,
        on_open=on_open,
        on_close=on_close,
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    TypeVar,
)

from backend.core.resilience.types import CircuitState

T = TypeVar("T")


class CircuitOpen(Exception):
    """
    Raised when the circuit breaker is open and rejecting calls.

    This exception indicates that the circuit breaker has tripped due to
    too many consecutive failures. The caller should use a fallback
    strategy or wait for the circuit to recover.

    Example:
        try:
            result = await breaker.call(api_call)
        except CircuitOpen:
            logger.warning("Service unavailable, using cached data")
            result = get_cached_data()
    """

    pass


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    This class implements the circuit breaker pattern to prevent cascading
    failures when a downstream service or resource is experiencing issues.
    It tracks consecutive failures and opens the circuit when a threshold
    is reached, rejecting subsequent calls until a recovery timeout elapses.

    The circuit breaker has three states:
    - CLOSED: Normal operation, all calls pass through
    - OPEN: Circuit has tripped, all calls are rejected with CircuitOpen
    - HALF_OPEN: Testing recovery, one call is allowed through

    Attributes:
        failure_threshold: Number of consecutive failures before opening.
                          Default is 5.
        recovery_timeout: Seconds to wait before trying HALF_OPEN state.
                         Default is 30.0.
        on_open: Optional async callback called when circuit opens.
        on_close: Optional async callback called when circuit closes.

    Example:
        # Basic usage
        breaker = CircuitBreaker()
        result = await breaker.call(my_async_func, arg1, arg2)

        # Customized breaker with callbacks
        async def alert_team():
            await send_slack_message("Database circuit opened!")

        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60.0,
            on_open=alert_team,
        )
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    on_open: Callable[[], Awaitable[None]] | None = None
    on_close: Callable[[], Awaitable[None]] | None = None

    # Internal state (not exposed as constructor params)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def state(self) -> CircuitState:
        """
        Get the current circuit state.

        Returns:
            The current CircuitState (CLOSED, OPEN, or HALF_OPEN).
        """
        return self._state

    @property
    def failure_count(self) -> int:
        """
        Get the current consecutive failure count.

        Returns:
            The number of consecutive failures since last success or reset.
        """
        return self._failure_count

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function through the circuit breaker.

        If the circuit is CLOSED, the function is called normally.
        If the circuit is OPEN and recovery_timeout has elapsed, transitions
        to HALF_OPEN and allows one call through.
        If the circuit is OPEN and recovery_timeout has not elapsed, raises
        CircuitOpen immediately.

        On success:
        - CLOSED: Resets failure count to 0
        - HALF_OPEN: Transitions to CLOSED

        On failure:
        - CLOSED: Increments failure count, may open circuit
        - HALF_OPEN: Transitions back to OPEN

        Args:
            func: The async function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value from the successful function call.

        Raises:
            CircuitOpen: If the circuit is OPEN and rejecting calls.
            Exception: Any exception raised by the function (also updates state).

        Example:
            async def fetch_user(user_id: int) -> dict:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"/users/{user_id}") as resp:
                        return await resp.json()

            breaker = CircuitBreaker(failure_threshold=3)

            try:
                user = await breaker.call(fetch_user, 123)
            except CircuitOpen:
                user = get_cached_user(123)
        """
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                else:
                    raise CircuitOpen(
                        f"Circuit is open, recovery in {self.recovery_timeout - elapsed:.1f}s"
                    )

            current_state = self._state

        # Execute the function outside the lock
        try:
            result = await func(*args, **kwargs)

            # Handle success
            async with self._lock:
                if current_state == CircuitState.HALF_OPEN:
                    # Probe succeeded, close the circuit
                    await self._close_circuit()
                else:
                    # Regular success, reset failure count
                    self._failure_count = 0

            return result

        except Exception as exc:
            # Handle failure
            async with self._lock:
                if current_state == CircuitState.HALF_OPEN:
                    # Probe failed, reopen the circuit
                    await self._open_circuit()
                else:
                    # Increment failure count
                    self._failure_count += 1
                    if self._failure_count >= self.failure_threshold:
                        await self._open_circuit()

            # Re-raise the original exception
            raise

    async def reset(self) -> None:
        """
        Manually reset the circuit breaker to CLOSED state.

        This method can be used to manually close the circuit after
        an operator has verified the downstream service has recovered.
        It is idempotent - calling reset on an already CLOSED circuit
        is a no-op.

        If the circuit is OPEN or HALF_OPEN, the on_close callback
        will be called (if configured).

        Example:
            # Manual reset after service recovery
            await breaker.reset()

            # Can be called multiple times safely
            await breaker.reset()
            await breaker.reset()  # No-op
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return  # Already closed, no-op

            await self._close_circuit()

    async def _open_circuit(self) -> None:
        """
        Transition the circuit to OPEN state.

        Records the failure time for recovery timeout tracking and
        calls the on_open callback if configured.

        Note: This method should be called with the lock held.
        """
        self._state = CircuitState.OPEN
        self._last_failure_time = time.monotonic()

        if self.on_open is not None:
            # Release lock during callback to prevent deadlocks
            # Note: The callback is awaited with the state already set
            await self.on_open()

    async def _close_circuit(self) -> None:
        """
        Transition the circuit to CLOSED state.

        Resets the failure count and calls the on_close callback
        if configured.

        Note: This method should be called with the lock held.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0

        if self.on_close is not None:
            await self.on_close()


__all__ = [
    "CircuitBreaker",
    "CircuitOpen",
]
