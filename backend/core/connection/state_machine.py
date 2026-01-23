"""
Atomic State Machine with Compare-And-Swap (CAS) Pattern
=========================================================

Provides thread-safe and async-safe state transitions using the CAS pattern.
Only one coroutine can win a contested state transition.

The CAS pattern ensures atomicity:
1. Compare: Check if current state matches expected state
2. Swap: If match, atomically update to new state
3. Return: True if this caller won, False if someone else did

This prevents the thundering herd problem in circuit breakers where
multiple coroutines race to transition from OPEN to HALF_OPEN.

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation - requests flow through
    OPEN = auto()        # Circuit tripped - requests rejected
    HALF_OPEN = auto()   # Testing recovery - limited requests allowed


@dataclass(frozen=True)
class StateTransition:
    """
    Immutable record of a state transition.

    Frozen dataclass ensures thread-safety when passing between contexts.
    """
    from_state: CircuitState
    to_state: CircuitState
    timestamp: datetime
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateTransitionError(Exception):
    """Raised when a state transition is invalid or contested."""
    pass


class AtomicStateMachine:
    """
    Atomic state machine with CAS pattern for concurrent safety.

    Uses asyncio.Lock for async contexts and threading.Lock for sync.
    State transitions are atomic - only one caller wins contested transitions.

    Thread-safety guarantees:
    - All state reads are protected by threading.Lock
    - All async operations use asyncio.Lock per event loop
    - Observers are called outside locks to prevent deadlock

    Event loop awareness:
    - Creates separate asyncio.Lock per event loop
    - Handles multi-loop scenarios (tests, different threads)
    """

    __slots__ = (
        '_state', '_lock', '_async_lock', '_loop_id',
        '_transition_history', '_observers', '_emitter',
        '_max_history', '_transition_count',
    )

    def __init__(
        self,
        initial_state: CircuitState = CircuitState.CLOSED,
        max_history: int = 100,
        emitter: Optional[Any] = None,
    ):
        """
        Initialize state machine.

        Args:
            initial_state: Starting state (default CLOSED)
            max_history: Maximum transitions to keep in history
            emitter: Optional event emitter for observability
        """
        self._state = initial_state
        self._lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._loop_id: Optional[int] = None
        self._transition_history: List[StateTransition] = []
        self._observers: List[Callable[[StateTransition], None]] = []
        self._emitter = emitter
        self._max_history = max_history
        self._transition_count = 0

    def _get_async_lock(self) -> asyncio.Lock:
        """
        Get or create async lock for the current event loop.

        Creates a new lock if:
        - No lock exists yet
        - We're in a different event loop than before

        This handles the case where the same state machine is accessed
        from different event loops (e.g., in tests or multi-threaded apps).
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            # No running loop - return a fresh lock
            # Caller is responsible for proper usage
            return asyncio.Lock()

        with self._lock:
            if self._loop_id != loop_id or self._async_lock is None:
                # Different loop or no lock - create new one
                self._async_lock = asyncio.Lock()
                self._loop_id = loop_id
            return self._async_lock

    @property
    def current_state(self) -> CircuitState:
        """Get current state (atomic read)."""
        with self._lock:
            return self._state

    async def try_transition(
        self,
        from_state: CircuitState,
        to_state: CircuitState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Attempt atomic state transition using CAS pattern.

        This is the core method for concurrent state management.
        Only one concurrent caller can win a contested transition.

        Args:
            from_state: Expected current state (compare)
            to_state: Desired new state (swap)
            reason: Human-readable reason for transition
            metadata: Additional data to record with transition

        Returns:
            True if this caller won the transition, False otherwise.
            False means another caller already changed the state.

        Example:
            # Multiple coroutines racing to transition
            success = await machine.try_transition(
                from_state=CircuitState.OPEN,
                to_state=CircuitState.HALF_OPEN,
                reason="Recovery timeout elapsed",
            )
            if success:
                # We won! We're the only one allowed to test
                await test_connection()
            else:
                # Someone else won, check what state we're in now
                pass
        """
        async_lock = self._get_async_lock()

        async with async_lock:
            with self._lock:
                # Compare: is state what we expect?
                if self._state != from_state:
                    return False  # Someone else already transitioned

                # Swap: atomically update state
                old_state = self._state
                self._state = to_state
                self._transition_count += 1

                # Record transition
                transition = StateTransition(
                    from_state=from_state,
                    to_state=to_state,
                    timestamp=datetime.now(),
                    reason=reason,
                    metadata=metadata or {},
                )
                self._transition_history.append(transition)

                # Trim history to prevent unbounded growth
                if len(self._transition_history) > self._max_history:
                    self._transition_history = self._transition_history[-self._max_history:]

        # Notify observers OUTSIDE lock to prevent deadlock
        # Observers might call back into the state machine
        for observer in self._observers:
            try:
                observer(transition)
            except Exception as e:
                # Don't let observer errors break state machine
                logger.warning(f"Observer error: {e}")

        # Emit event if emitter available
        if self._emitter is not None:
            try:
                # Try async emit first
                if hasattr(self._emitter, 'emit_async'):
                    await self._emitter.emit_async(
                        "STATE_CHANGED",
                        service_name="CircuitBreaker",
                        data={
                            "from_state": from_state.name,
                            "to_state": to_state.name,
                            "reason": reason,
                        },
                    )
                elif hasattr(self._emitter, 'emit'):
                    self._emitter.emit(
                        "STATE_CHANGED",
                        service_name="CircuitBreaker",
                        data={
                            "from_state": from_state.name,
                            "to_state": to_state.name,
                            "reason": reason,
                        },
                    )
            except Exception as e:
                logger.debug(f"Event emission error: {e}")

        logger.debug(f"State transition: {from_state.name} -> {to_state.name} ({reason})")
        return True

    def try_transition_sync(
        self,
        from_state: CircuitState,
        to_state: CircuitState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Synchronous version of try_transition.

        Use this when not in an async context. Provides the same
        CAS guarantees but without async lock acquisition.

        Args:
            from_state: Expected current state
            to_state: Desired new state
            reason: Human-readable reason for transition
            metadata: Additional data to record

        Returns:
            True if transition succeeded, False otherwise.
        """
        with self._lock:
            if self._state != from_state:
                return False

            self._state = to_state
            self._transition_count += 1

            transition = StateTransition(
                from_state=from_state,
                to_state=to_state,
                timestamp=datetime.now(),
                reason=reason,
                metadata=metadata or {},
            )
            self._transition_history.append(transition)

            if len(self._transition_history) > self._max_history:
                self._transition_history = self._transition_history[-self._max_history:]

        # Notify observers outside lock
        for observer in self._observers:
            try:
                observer(transition)
            except Exception as e:
                logger.warning(f"Observer error: {e}")

        logger.debug(f"Sync state transition: {from_state.name} -> {to_state.name} ({reason})")
        return True

    def add_observer(self, callback: Callable[[StateTransition], None]) -> None:
        """
        Add observer for state transitions.

        Observers are called synchronously after each transition,
        outside the lock to prevent deadlock.

        Args:
            callback: Function called with StateTransition on each change
        """
        self._observers.append(callback)

    def remove_observer(self, callback: Callable[[StateTransition], None]) -> bool:
        """
        Remove an observer.

        Args:
            callback: The observer to remove

        Returns:
            True if observer was found and removed, False otherwise
        """
        try:
            self._observers.remove(callback)
            return True
        except ValueError:
            return False

    def get_history(self, limit: int = 10) -> List[StateTransition]:
        """
        Get recent transition history.

        Args:
            limit: Maximum number of transitions to return

        Returns:
            List of recent StateTransition objects (newest last)
        """
        with self._lock:
            return list(self._transition_history[-limit:])

    @property
    def transition_count(self) -> int:
        """Total number of transitions since creation."""
        with self._lock:
            return self._transition_count

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AtomicStateMachine(state={self.current_state.name}, "
            f"transitions={self.transition_count})"
        )
