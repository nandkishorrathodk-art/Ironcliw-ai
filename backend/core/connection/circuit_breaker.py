"""
Atomic Circuit Breaker with Thundering Herd Prevention
=======================================================

Enterprise-grade circuit breaker that:
- Uses CAS pattern for atomic state transitions
- Limits HALF_OPEN test requests to prevent thundering herd
- Provides full observability via event emission
- Integrates with DI container for cross-service coordination

The key innovation is limiting HALF_OPEN requests:
- When transitioning from OPEN to HALF_OPEN, only ONE coroutine wins
- That coroutine gets to make the test request
- Other coroutines are blocked, preventing thundering herd

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from backend.core.connection.state_machine import (
    AtomicStateMachine,
    CircuitState,
)

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for atomic circuit breaker.

    All values can be overridden via environment variables.
    """
    failure_threshold: int = 5
    success_threshold: int = 2
    recovery_timeout_seconds: float = 30.0
    half_open_max_requests: int = 1  # Limit test requests to prevent thundering herd

    def __post_init__(self):
        """Load from environment if available."""
        self.failure_threshold = int(os.getenv(
            'CIRCUIT_FAILURE_THRESHOLD', str(self.failure_threshold)
        ))
        self.success_threshold = int(os.getenv(
            'CIRCUIT_SUCCESS_THRESHOLD', str(self.success_threshold)
        ))
        self.recovery_timeout_seconds = float(os.getenv(
            'CIRCUIT_RECOVERY_TIMEOUT', str(self.recovery_timeout_seconds)
        ))
        self.half_open_max_requests = int(os.getenv(
            'CIRCUIT_HALF_OPEN_MAX_REQUESTS', str(self.half_open_max_requests)
        ))


class AtomicCircuitBreaker:
    """
    Atomic circuit breaker with thundering herd prevention.

    Key improvements over traditional circuit breakers:
    1. CAS pattern prevents multiple coroutines from racing to HALF_OPEN
    2. Limited test requests in HALF_OPEN state
    3. Atomic counters for success/failure tracking
    4. Full observability via events

    State Machine:
    ```
    CLOSED --[failures >= threshold]--> OPEN
    OPEN --[timeout elapsed]--> HALF_OPEN (only 1 coroutine wins!)
    HALF_OPEN --[success >= threshold]--> CLOSED
    HALF_OPEN --[any failure]--> OPEN
    ```

    Usage:
        breaker = AtomicCircuitBreaker()

        if await breaker.can_execute():
            try:
                result = await make_request()
                await breaker.record_success()
            except Exception as e:
                await breaker.record_failure(str(e))
        else:
            # Circuit is open, use fallback
            result = fallback_response()
    """

    __slots__ = (
        '_config', '_state_machine', '_failure_count', '_success_count',
        '_half_open_request_count', '_last_failure_time', '_last_success_time',
        '_lock', '_emitter', '_connection_refused_count',
    )

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        emitter: Optional[Any] = None,
    ):
        """
        Initialize atomic circuit breaker.

        Args:
            config: Configuration (uses defaults + env vars if None)
            emitter: Optional event emitter for observability
        """
        self._config = config or CircuitBreakerConfig()
        self._state_machine = AtomicStateMachine(
            initial_state=CircuitState.CLOSED,
            emitter=emitter,
        )
        self._emitter = emitter

        # Counters (protected by lock)
        self._lock = threading.Lock()
        self._failure_count = 0
        self._success_count = 0
        self._half_open_request_count = 0
        self._connection_refused_count = 0

        # Timestamps
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None

        logger.debug(
            f"AtomicCircuitBreaker initialized: "
            f"failure_threshold={self._config.failure_threshold}, "
            f"half_open_max={self._config.half_open_max_requests}"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state_machine.current_state

    @property
    def config(self) -> CircuitBreakerConfig:
        """Get configuration."""
        return self._config

    async def can_execute(self) -> bool:
        """
        Check if requests are allowed through the circuit.

        Returns True if:
        - CLOSED: always allowed
        - HALF_OPEN: allowed if under max test requests
        - OPEN: allowed if recovery timeout elapsed (transitions to HALF_OPEN)

        This method is the primary entry point for circuit breaker usage.
        It handles the OPEN -> HALF_OPEN transition atomically.

        Returns:
            True if request is allowed, False if circuit is tripped
        """
        current = self._state_machine.current_state

        if current == CircuitState.CLOSED:
            return True

        if current == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self._last_failure_time:
                elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                if elapsed >= self._config.recovery_timeout_seconds:
                    # Attempt atomic transition to HALF_OPEN
                    # Only ONE coroutine wins this transition!
                    won = await self._state_machine.try_transition(
                        from_state=CircuitState.OPEN,
                        to_state=CircuitState.HALF_OPEN,
                        reason=f"Recovery timeout elapsed ({elapsed:.1f}s)",
                    )
                    if won:
                        # We won! Reset and count this as first test request
                        with self._lock:
                            self._half_open_request_count = 1  # Count this as request
                            self._success_count = 0

                        logger.info(
                            f"Circuit breaker HALF-OPEN: testing after {elapsed:.1f}s"
                        )
                        return True
                    else:
                        # Someone else won the transition
                        # Recursively check what state we're in now
                        return await self.can_execute()
            return False

        if current == CircuitState.HALF_OPEN:
            # Limit test requests in HALF_OPEN to prevent thundering herd
            with self._lock:
                if self._half_open_request_count >= self._config.half_open_max_requests:
                    logger.debug(
                        f"HALF_OPEN request limit reached "
                        f"({self._half_open_request_count}/{self._config.half_open_max_requests})"
                    )
                    return False
                self._half_open_request_count += 1
                return True

        return False

    async def record_success(self) -> None:
        """
        Record a successful operation.

        In HALF_OPEN state, enough successes will close the circuit.
        In CLOSED state, this resets the failure count.
        """
        with self._lock:
            self._success_count += 1
            self._last_success_time = datetime.now()
            self._connection_refused_count = 0  # Reset on success

        current = self._state_machine.current_state

        if current == CircuitState.HALF_OPEN:
            # Check if enough successes to close circuit
            with self._lock:
                should_close = self._success_count >= self._config.success_threshold
                success_count = self._success_count

            if should_close:
                # Attempt atomic transition to CLOSED
                transitioned = await self._state_machine.try_transition(
                    from_state=CircuitState.HALF_OPEN,
                    to_state=CircuitState.CLOSED,
                    reason=f"Success threshold reached ({success_count})",
                )
                if transitioned:
                    with self._lock:
                        self._failure_count = 0
                        self._success_count = 0
                    logger.info(
                        f"Circuit breaker CLOSED after {success_count} successes"
                    )

        elif current == CircuitState.CLOSED:
            # In CLOSED state, success resets failure count
            with self._lock:
                self._failure_count = 0

    async def record_failure(self, error: str = "") -> None:
        """
        Record a failed operation.

        In HALF_OPEN state, any failure immediately reopens the circuit.
        In CLOSED state, failures accumulate toward the threshold.

        Args:
            error: Error message for logging and tracking
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            failure_count = self._failure_count

            # Track connection refused separately
            if "connection refused" in error.lower():
                self._connection_refused_count += 1

        current = self._state_machine.current_state

        if current == CircuitState.HALF_OPEN:
            # Any failure in HALF_OPEN immediately opens circuit
            transitioned = await self._state_machine.try_transition(
                from_state=CircuitState.HALF_OPEN,
                to_state=CircuitState.OPEN,
                reason=f"Test request failed: {error}",
            )
            if transitioned:
                with self._lock:
                    self._success_count = 0
                logger.warning(f"Circuit breaker OPEN: test failed ({error})")

        elif current == CircuitState.CLOSED:
            # Check if threshold reached
            if failure_count >= self._config.failure_threshold:
                transitioned = await self._state_machine.try_transition(
                    from_state=CircuitState.CLOSED,
                    to_state=CircuitState.OPEN,
                    reason=f"Failure threshold reached ({failure_count})",
                )
                if transitioned:
                    logger.warning(
                        f"Circuit breaker OPEN: {failure_count} failures"
                    )

    def get_state_info(self) -> Dict[str, Any]:
        """
        Get detailed state information for observability.

        Returns:
            Dict with state, counters, timestamps, and transition count
        """
        with self._lock:
            return {
                'state': self._state_machine.current_state.name,
                'failure_count': self._failure_count,
                'success_count': self._success_count,
                'half_open_requests': self._half_open_request_count,
                'connection_refused_count': self._connection_refused_count,
                'last_failure': (
                    self._last_failure_time.isoformat()
                    if self._last_failure_time else None
                ),
                'last_success': (
                    self._last_success_time.isoformat()
                    if self._last_success_time else None
                ),
                'transition_count': self._state_machine.transition_count,
                'config': {
                    'failure_threshold': self._config.failure_threshold,
                    'success_threshold': self._config.success_threshold,
                    'recovery_timeout': self._config.recovery_timeout_seconds,
                    'half_open_max_requests': self._config.half_open_max_requests,
                },
            }

    def reset(self) -> None:
        """
        Reset circuit breaker to initial CLOSED state.

        Use this for manual recovery or testing.
        """
        current = self._state_machine.current_state
        self._state_machine.try_transition_sync(
            from_state=current,
            to_state=CircuitState.CLOSED,
            reason="Manual reset",
        )
        with self._lock:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_request_count = 0
            self._connection_refused_count = 0

        logger.info("Circuit breaker reset to CLOSED")

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AtomicCircuitBreaker(state={self.state.name}, "
            f"failures={self._failure_count}, "
            f"config=threshold:{self._config.failure_threshold})"
        )
