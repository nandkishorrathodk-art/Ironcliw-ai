"""
Finite State Machine Framework with Compare-And-Swap (CAS) Pattern
===================================================================

Provides a reusable, generic FiniteStateMachine base class with:
- Generic state type support (any Enum or Hashable type)
- Explicit transition table with validation
- CAS-based atomic transitions (thread-safe + async-safe)
- Rich callback system: on_enter, on_exit, on_transition
- Bounded transition history (deque)
- Serialization support (to_dict / from_dict)
- Factory functions for common FSM patterns

The original AtomicStateMachine (circuit breaker CAS pattern) inherits from
FiniteStateMachine for full backward compatibility.

Author: Ironcliw System
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Hashable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Generic state type — must be hashable (Enum members, strings, ints, etc.)
S = TypeVar("S", bound=Hashable)


# ---------------------------------------------------------------------------
# Domain-specific enums
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation - requests flow through
    OPEN = auto()        # Circuit tripped - requests rejected
    HALF_OPEN = auto()   # Testing recovery - limited requests allowed


class ConnectionState(Enum):
    """Standard connection lifecycle states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTING = auto()
    ERROR = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateTransition:
    """
    Immutable record of a state transition.

    Frozen dataclass ensures thread-safety when passing between contexts.
    The ``from_state`` / ``to_state`` fields are intentionally typed as
    ``Any`` so a single dataclass works for every FSM regardless of its
    state type.
    """
    from_state: Any
    to_state: Any
    timestamp: datetime
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StateTransitionError(Exception):
    """Raised when a state transition is invalid or contested."""

    def __init__(
        self,
        message: str = "",
        from_state: Any = None,
        to_state: Any = None,
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(message or f"Invalid transition: {from_state} -> {to_state}")


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

# Called with (transition: StateTransition)
TransitionCallback = Callable[[StateTransition], None]
# Called with (state, transition: StateTransition)
StateCallback = Callable[[Any, StateTransition], None]


# ---------------------------------------------------------------------------
# FiniteStateMachine — the generic, reusable base class
# ---------------------------------------------------------------------------

class FiniteStateMachine(Generic[S]):
    """
    Generic finite state machine with CAS-based atomic transitions.

    Features:
    - Works with **any** hashable state type (Enum members, strings, ints).
    - Explicit transition table (``Dict[S, Set[S]]``).  Pass ``None`` for an
      open FSM that allows every transition.
    - Thread-safe via ``threading.Lock``.
    - Async-safe via lazily-created ``asyncio.Lock`` (one per event loop).
    - Rich callback system: ``on_enter``, ``on_exit``, ``on_transition``.
    - Bounded transition history (``collections.deque``).
    - Serialization helpers: ``to_dict()`` / ``from_dict()``.

    Minimal overhead: the hot path (``transition`` / ``transition_sync``) only
    acquires locks and does a dict/set lookup.
    """

    def __init__(
        self,
        initial_state: S,
        transitions: Optional[Dict[S, Set[S]]] = None,
        max_history: int = 100,
    ) -> None:
        """
        Args:
            initial_state: The starting state.
            transitions: Allowed transitions as ``{from_state: {to_state, ...}}``.
                         Pass ``None`` to allow **all** transitions (open FSM).
            max_history: Maximum number of transition records to retain.
        """
        self._state: S = initial_state
        # None means "open FSM — every transition is allowed"
        self._transitions: Optional[Dict[S, FrozenSet[S]]] = None
        if transitions is not None:
            self._transitions = {
                src: frozenset(dsts) for src, dsts in transitions.items()
            }
        self._max_history: int = max_history
        self._history: Deque[StateTransition] = deque(maxlen=max_history)
        self._transition_count: int = 0

        # Threading / async locks
        self._lock: threading.Lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._loop_id: Optional[int] = None

        # Callback registries
        self._on_enter_callbacks: Dict[S, List[StateCallback]] = {}
        self._on_exit_callbacks: Dict[S, List[StateCallback]] = {}
        self._on_transition_callbacks: Dict[
            Tuple[S, S], List[TransitionCallback]
        ] = {}
        # Global observers (legacy compat + general purpose)
        self._observers: List[TransitionCallback] = []

    # ------------------------------------------------------------------
    # Async lock management (event-loop aware)
    # ------------------------------------------------------------------

    def _get_async_lock(self) -> asyncio.Lock:
        """
        Return an ``asyncio.Lock`` bound to the current event loop.

        Creates a new lock when the event loop changes (e.g. tests that
        create fresh loops).
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            # No running loop — return an ephemeral lock.
            return asyncio.Lock()

        with self._lock:
            if self._loop_id != loop_id or self._async_lock is None:
                self._async_lock = asyncio.Lock()
                self._loop_id = loop_id
            return self._async_lock

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> S:
        """Atomic read of the current state."""
        with self._lock:
            return self._state

    # ------------------------------------------------------------------
    # Transition validation
    # ------------------------------------------------------------------

    def can_transition(self, from_state: S, to_state: S) -> bool:
        """
        Check whether the transition ``from_state -> to_state`` is permitted
        by the transition table **without** executing it.

        Returns ``True`` for any pair when the FSM was created with
        ``transitions=None`` (open FSM).
        """
        if self._transitions is None:
            return True
        allowed = self._transitions.get(from_state)
        if allowed is None:
            return False
        return to_state in allowed

    # ------------------------------------------------------------------
    # Core CAS transition — async
    # ------------------------------------------------------------------

    async def transition(
        self,
        from_state: S,
        to_state: S,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Attempt an atomic state transition (async CAS).

        Args:
            from_state: Expected current state (compare phase).
            to_state:   Desired new state (swap phase).
            reason:     Human-readable reason for the transition.
            metadata:   Arbitrary data attached to the transition record.

        Returns:
            ``True`` if this caller won the transition; ``False`` if the
            current state did not match ``from_state`` (someone else already
            transitioned).

        Raises:
            StateTransitionError: If the transition is not in the allowed
                transition table (and the FSM is not open).
        """
        if not self.can_transition(from_state, to_state):
            raise StateTransitionError(
                f"Transition {from_state} -> {to_state} is not allowed",
                from_state=from_state,
                to_state=to_state,
            )

        async_lock = self._get_async_lock()
        transition_record: Optional[StateTransition] = None
        old_state: Optional[S] = None

        async with async_lock:
            with self._lock:
                if self._state != from_state:
                    return False

                old_state = self._state
                self._state = to_state
                self._transition_count += 1

                transition_record = StateTransition(
                    from_state=from_state,
                    to_state=to_state,
                    timestamp=datetime.now(),
                    reason=reason,
                    metadata=metadata or {},
                )
                self._history.append(transition_record)

        # Fire callbacks OUTSIDE locks to prevent deadlock
        assert transition_record is not None
        self._fire_callbacks(old_state, to_state, transition_record)

        logger.debug(
            "State transition: %s -> %s (%s)",
            self._state_name(from_state),
            self._state_name(to_state),
            reason,
        )
        return True

    # ------------------------------------------------------------------
    # Core CAS transition — sync
    # ------------------------------------------------------------------

    def transition_sync(
        self,
        from_state: S,
        to_state: S,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Synchronous CAS transition.  Same semantics as :meth:`transition`
        but without ``asyncio.Lock`` acquisition.

        Raises:
            StateTransitionError: If the transition is not in the allowed
                transition table.
        """
        if not self.can_transition(from_state, to_state):
            raise StateTransitionError(
                f"Transition {from_state} -> {to_state} is not allowed",
                from_state=from_state,
                to_state=to_state,
            )

        transition_record: Optional[StateTransition] = None
        old_state: Optional[S] = None

        with self._lock:
            if self._state != from_state:
                return False

            old_state = self._state
            self._state = to_state
            self._transition_count += 1

            transition_record = StateTransition(
                from_state=from_state,
                to_state=to_state,
                timestamp=datetime.now(),
                reason=reason,
                metadata=metadata or {},
            )
            self._history.append(transition_record)

        assert transition_record is not None
        self._fire_callbacks(old_state, to_state, transition_record)

        logger.debug(
            "Sync state transition: %s -> %s (%s)",
            self._state_name(from_state),
            self._state_name(to_state),
            reason,
        )
        return True

    # ------------------------------------------------------------------
    # Force state (bypass transition table — error recovery only)
    # ------------------------------------------------------------------

    def force_state(
        self,
        state: S,
        reason: str = "force_state",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Forcefully set the state, bypassing the transition table.

        **Use only for error recovery.**  A warning is always logged.

        Args:
            state:    The state to force.
            reason:   Why the force was necessary.
            metadata: Additional context.
        """
        with self._lock:
            old_state = self._state
            self._state = state
            self._transition_count += 1

            transition_record = StateTransition(
                from_state=old_state,
                to_state=state,
                timestamp=datetime.now(),
                reason=f"[FORCED] {reason}",
                metadata=metadata or {},
            )
            self._history.append(transition_record)

        logger.warning(
            "Forced state: %s -> %s (reason: %s)",
            self._state_name(old_state),
            self._state_name(state),
            reason,
        )
        self._fire_callbacks(old_state, state, transition_record)

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_enter(self, state: S, callback: StateCallback) -> None:
        """
        Register a callback invoked when the FSM **enters** ``state``.

        The callback signature is ``(state, transition_record)``.
        """
        self._on_enter_callbacks.setdefault(state, []).append(callback)

    def on_exit(self, state: S, callback: StateCallback) -> None:
        """
        Register a callback invoked when the FSM **exits** ``state``.

        The callback signature is ``(state, transition_record)``.
        """
        self._on_exit_callbacks.setdefault(state, []).append(callback)

    def on_transition(
        self, from_state: S, to_state: S, callback: TransitionCallback
    ) -> None:
        """
        Register a callback for a **specific** transition edge.

        The callback signature is ``(transition_record,)``.
        """
        key = (from_state, to_state)
        self._on_transition_callbacks.setdefault(key, []).append(callback)

    def add_observer(self, callback: TransitionCallback) -> None:
        """
        Add a **global** observer called on every transition.

        Observers are invoked outside locks to prevent deadlock.
        """
        self._observers.append(callback)

    def remove_observer(self, callback: TransitionCallback) -> bool:
        """
        Remove a global observer.

        Returns ``True`` if found and removed, ``False`` otherwise.
        """
        try:
            self._observers.remove(callback)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 10) -> List[StateTransition]:
        """
        Return the most recent transition records (newest last).

        Args:
            limit: Maximum records to return.
        """
        with self._lock:
            items = list(self._history)
        return items[-limit:]

    def get_history_tuples(
        self, limit: int = 10,
    ) -> List[Tuple[Any, Any, float]]:
        """
        Return history as ``(from_state, to_state, timestamp_epoch)`` tuples.

        Useful for lightweight inspection without full dataclass overhead.
        """
        records = self.get_history(limit)
        return [
            (r.from_state, r.to_state, r.timestamp.timestamp())
            for r in records
        ]

    @property
    def transition_count(self) -> int:
        """Total transitions since creation."""
        with self._lock:
            return self._transition_count

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize FSM state to a plain dict suitable for JSON / persistence.

        The transition table and callbacks are **not** serialized (they are
        code, not data).  Only runtime state is captured.
        """
        with self._lock:
            history_list = [
                {
                    "from_state": self._state_name(r.from_state),
                    "to_state": self._state_name(r.to_state),
                    "timestamp": r.timestamp.isoformat(),
                    "reason": r.reason,
                    "metadata": r.metadata,
                }
                for r in self._history
            ]
            return {
                "current_state": self._state_name(self._state),
                "transition_count": self._transition_count,
                "max_history": self._max_history,
                "history": history_list,
            }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        state_resolver: Callable[[str], S],
        transitions: Optional[Dict[S, Set[S]]] = None,
    ) -> "FiniteStateMachine[S]":
        """
        Reconstruct an FSM from a serialized dict.

        Args:
            data:           The dict produced by :meth:`to_dict`.
            state_resolver: A callable that maps a state name (string) back
                            to the actual state value.  For Enums use
                            ``MyEnum.__members__.__getitem__``.
            transitions:    The transition table (must be supplied again;
                            it is not serialized).

        Returns:
            A new ``FiniteStateMachine`` instance with restored state and
            history.
        """
        current = state_resolver(data["current_state"])
        max_history = data.get("max_history", 100)
        fsm: FiniteStateMachine[S] = cls(
            initial_state=current,
            transitions=transitions,
            max_history=max_history,
        )
        fsm._transition_count = data.get("transition_count", 0)

        for entry in data.get("history", []):
            rec = StateTransition(
                from_state=state_resolver(entry["from_state"]),
                to_state=state_resolver(entry["to_state"]),
                timestamp=datetime.fromisoformat(entry["timestamp"]),
                reason=entry.get("reason", ""),
                metadata=entry.get("metadata", {}),
            )
            fsm._history.append(rec)

        return fsm

    # ------------------------------------------------------------------
    # Transition table introspection
    # ------------------------------------------------------------------

    def get_allowed_transitions(self, from_state: S) -> FrozenSet[S]:
        """
        Return the set of states reachable from ``from_state``.

        Returns an empty frozenset if ``from_state`` has no outgoing edges.
        For an open FSM (no transition table), raises ``ValueError`` because
        there is no finite set to return.
        """
        if self._transitions is None:
            raise ValueError(
                "Open FSM has no explicit transition table; "
                "all transitions are allowed."
            )
        return self._transitions.get(from_state, frozenset())

    @property
    def is_open_fsm(self) -> bool:
        """``True`` if this FSM has no transition table (all transitions allowed)."""
        return self._transitions is None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire_callbacks(
        self,
        old_state: Optional[S],
        new_state: S,
        record: StateTransition,
    ) -> None:
        """
        Invoke registered callbacks.  All exceptions are caught and logged
        so that a misbehaving callback never breaks the FSM.
        """
        # on_exit callbacks for old_state
        if old_state is not None:
            for cb in self._on_exit_callbacks.get(old_state, ()):
                try:
                    cb(old_state, record)
                except Exception as exc:
                    logger.warning("on_exit callback error (%s): %s", old_state, exc)

        # on_enter callbacks for new_state
        for cb in self._on_enter_callbacks.get(new_state, ()):
            try:
                cb(new_state, record)
            except Exception as exc:
                logger.warning("on_enter callback error (%s): %s", new_state, exc)

        # Edge-specific on_transition callbacks
        if old_state is not None:
            key = (old_state, new_state)
            for cb in self._on_transition_callbacks.get(key, ()):
                try:
                    cb(record)
                except Exception as exc:
                    logger.warning("on_transition callback error (%s): %s", key, exc)

        # Global observers
        for cb in self._observers:
            try:
                cb(record)
            except Exception as exc:
                logger.warning("Observer error: %s", exc)

    @staticmethod
    def _state_name(state: Any) -> str:
        """Return a human-readable name for a state value."""
        if isinstance(state, Enum):
            return state.name
        return str(state)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        kind = "open" if self.is_open_fsm else "constrained"
        return (
            f"FiniteStateMachine(state={self._state_name(self.current_state)}, "
            f"transitions={self.transition_count}, type={kind})"
        )


# ---------------------------------------------------------------------------
# AtomicStateMachine — backward-compatible subclass
# ---------------------------------------------------------------------------

class AtomicStateMachine(FiniteStateMachine[CircuitState]):
    """
    Atomic state machine with CAS pattern for concurrent safety.

    This is the original Ironcliw circuit-breaker state machine, now
    implemented as a thin subclass of :class:`FiniteStateMachine`.

    **Full backward compatibility** is preserved:

    - ``try_transition()`` and ``try_transition_sync()`` work identically.
    - Default ``initial_state`` is ``CircuitState.CLOSED``.
    - The FSM is **open** (no transition table constraint) by default,
      matching the original behavior where any ``CircuitState`` transition
      was permitted.
    - The optional ``emitter`` integration is preserved.
    - Observer and history APIs are unchanged.

    Thread-safety guarantees:
    - All state reads are protected by threading.Lock
    - All async operations use asyncio.Lock per event loop
    - Observers are called outside locks to prevent deadlock

    Event loop awareness:
    - Creates separate asyncio.Lock per event loop
    - Handles multi-loop scenarios (tests, different threads)
    """

    def __init__(
        self,
        initial_state: CircuitState = CircuitState.CLOSED,
        max_history: int = 100,
        emitter: Optional[Any] = None,
        transitions: Optional[Dict[CircuitState, Set[CircuitState]]] = None,
    ) -> None:
        """
        Initialize state machine.

        Args:
            initial_state: Starting state (default CLOSED)
            max_history:   Maximum transitions to keep in history
            emitter:       Optional event emitter for observability
            transitions:   Optional transition table.  ``None`` = open FSM
                           (all transitions allowed), which is the legacy
                           default.
        """
        super().__init__(
            initial_state=initial_state,
            transitions=transitions,
            max_history=max_history,
        )
        self._emitter = emitter

    # ------------------------------------------------------------------
    # Legacy async transition (delegates to base, adds emitter support)
    # ------------------------------------------------------------------

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
            to_state:   Desired new state (swap)
            reason:     Human-readable reason for transition
            metadata:   Additional data to record with transition

        Returns:
            True if this caller won the transition, False otherwise.
            False means another caller already changed the state.

        Example:
            success = await machine.try_transition(
                from_state=CircuitState.OPEN,
                to_state=CircuitState.HALF_OPEN,
                reason="Recovery timeout elapsed",
            )
            if success:
                await test_connection()
        """
        success = await self.transition(
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            metadata=metadata,
        )

        if success and self._emitter is not None:
            await self._emit_state_changed(from_state, to_state, reason)

        return success

    # ------------------------------------------------------------------
    # Legacy sync transition
    # ------------------------------------------------------------------

    def try_transition_sync(
        self,
        from_state: CircuitState,
        to_state: CircuitState,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Synchronous version of try_transition.

        Use this when not in an async context.  Provides the same CAS
        guarantees but without async lock acquisition.

        Args:
            from_state: Expected current state
            to_state:   Desired new state
            reason:     Human-readable reason for transition
            metadata:   Additional data to record

        Returns:
            True if transition succeeded, False otherwise.
        """
        success = self.transition_sync(
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            metadata=metadata,
        )

        if success and self._emitter is not None:
            try:
                if hasattr(self._emitter, 'emit'):
                    self._emitter.emit(
                        "STATE_CHANGED",
                        service_name="CircuitBreaker",
                        data={
                            "from_state": from_state.name,
                            "to_state": to_state.name,
                            "reason": reason,
                        },
                    )
            except Exception as exc:
                logger.debug("Event emission error: %s", exc)

        return success

    # ------------------------------------------------------------------
    # Emitter helper
    # ------------------------------------------------------------------

    async def _emit_state_changed(
        self,
        from_state: CircuitState,
        to_state: CircuitState,
        reason: str,
    ) -> None:
        """Emit a STATE_CHANGED event via the configured emitter."""
        try:
            payload = {
                "from_state": from_state.name,
                "to_state": to_state.name,
                "reason": reason,
            }
            if hasattr(self._emitter, 'emit_async'):
                await self._emitter.emit_async(
                    "STATE_CHANGED",
                    service_name="CircuitBreaker",
                    data=payload,
                )
            elif hasattr(self._emitter, 'emit'):
                self._emitter.emit(
                    "STATE_CHANGED",
                    service_name="CircuitBreaker",
                    data=payload,
                )
        except Exception as exc:
            logger.debug("Event emission error: %s", exc)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AtomicStateMachine(state={self.current_state.name}, "
            f"transitions={self.transition_count})"
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_lifecycle_fsm(
    states_enum: Type[Enum],
    error_state_name: str = "ERROR",
    max_history: int = 100,
) -> FiniteStateMachine:
    """
    Create a **linear lifecycle** FSM from an Enum.

    Each state can transition to the next state in declaration order, and
    every state can transition to the error state (if one exists in the
    Enum with name matching ``error_state_name``).

    Example::

        class DeployPhase(Enum):
            INIT = auto()
            BUILD = auto()
            TEST = auto()
            DEPLOY = auto()
            DONE = auto()
            ERROR = auto()

        fsm = create_lifecycle_fsm(DeployPhase)
        # INIT -> BUILD -> TEST -> DEPLOY -> DONE
        # Any state -> ERROR

    Args:
        states_enum:     An Enum class whose members define the lifecycle.
        error_state_name: Name of the error member (case-sensitive).
                          If not found, no universal error edge is added.
        max_history:     Passed to the FSM constructor.

    Returns:
        A ``FiniteStateMachine`` instance with constrained transitions.
    """
    members = list(states_enum)
    if not members:
        raise ValueError("states_enum must have at least one member")

    transitions: Dict[Any, Set[Any]] = {}

    # Identify error state if present
    error_state: Optional[Any] = None
    for member in members:
        if member.name == error_state_name:
            error_state = member
            break

    # Build linear chain: each state -> next state
    for idx, member in enumerate(members):
        allowed: Set[Any] = set()
        # Forward edge to next state in declaration order
        if idx + 1 < len(members):
            allowed.add(members[idx + 1])
        # Universal edge to ERROR (if error state exists and this isn't ERROR)
        if error_state is not None and member != error_state:
            allowed.add(error_state)
        transitions[member] = allowed

    # Error state gets no outgoing edges by default (terminal)
    # unless it already got some from the loop above
    if error_state is not None and error_state not in transitions:
        transitions[error_state] = set()

    initial = members[0]
    return FiniteStateMachine(
        initial_state=initial,
        transitions=transitions,
        max_history=max_history,
    )


def create_connection_fsm(
    initial_state: ConnectionState = ConnectionState.DISCONNECTED,
    max_history: int = 100,
) -> FiniteStateMachine[ConnectionState]:
    """
    Create a standard **connection lifecycle** FSM::

        DISCONNECTED <-> CONNECTING <-> CONNECTED <-> DISCONNECTING
                                                          |
                                                          v
                                                     DISCONNECTED
        Any state -> ERROR
        ERROR -> DISCONNECTED  (recovery)

    Args:
        initial_state: Starting state (default ``DISCONNECTED``).
        max_history:   Passed to the FSM constructor.

    Returns:
        A ``FiniteStateMachine[ConnectionState]`` with the standard
        connection state graph.
    """
    D = ConnectionState.DISCONNECTED
    Ci = ConnectionState.CONNECTING
    Co = ConnectionState.CONNECTED
    Di = ConnectionState.DISCONNECTING
    E = ConnectionState.ERROR

    transitions: Dict[ConnectionState, Set[ConnectionState]] = {
        D:  {Ci, E},
        Ci: {Co, D, E},
        Co: {Di, E},
        Di: {D, E},
        E:  {D},
    }

    return FiniteStateMachine(
        initial_state=initial_state,
        transitions=transitions,
        max_history=max_history,
    )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Base class
    "FiniteStateMachine",
    # Legacy / circuit breaker
    "AtomicStateMachine",
    "CircuitState",
    "StateTransition",
    "StateTransitionError",
    # Connection states
    "ConnectionState",
    # Factory functions
    "create_lifecycle_fsm",
    "create_connection_fsm",
    # Callback types
    "TransitionCallback",
    "StateCallback",
]
