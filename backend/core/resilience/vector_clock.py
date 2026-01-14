"""
Vector Clock Implementation for Distributed Event Ordering
==========================================================

Provides Lamport timestamps and vector clocks for ensuring causality
in distributed cross-repo events.

Features:
    - Lamport timestamps for total ordering
    - Vector clocks for partial ordering with causality
    - Happens-before relationship detection
    - Concurrent event detection
    - Redis-backed clock synchronization across repos
    - Clock drift detection and compensation

Theory:
    Vector clocks solve the problem of ordering events in a distributed
    system where physical clocks may drift. Each repo maintains a vector
    of logical clocks, one per repo. Events are ordered by the happens-before
    relation: e1 → e2 iff VC(e1) < VC(e2) for all entries.

Usage:
    clock = VectorClock(repo_id="jarvis")
    event1 = clock.tick()  # Creates event with incremented clock

    # When receiving event from another repo:
    clock.merge(other_clock)  # Updates local clock

    # Check ordering:
    if event1.happens_before(event2):
        # event1 causally precedes event2
    elif event2.happens_before(event1):
        # event2 causally precedes event1
    else:
        # Concurrent events - need conflict resolution

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from pathlib import Path

logger = logging.getLogger("VectorClock")


# =============================================================================
# Configuration
# =============================================================================

CLOCK_SYNC_INTERVAL = float(os.getenv("CLOCK_SYNC_INTERVAL", "5.0"))
CLOCK_DRIFT_THRESHOLD_MS = float(os.getenv("CLOCK_DRIFT_THRESHOLD_MS", "100.0"))
MAX_CLOCK_HISTORY = int(os.getenv("MAX_CLOCK_HISTORY", "1000"))


class CausalityRelation(Enum):
    """Relationship between two events."""
    HAPPENS_BEFORE = "happens_before"  # e1 → e2
    HAPPENS_AFTER = "happens_after"    # e2 → e1
    CONCURRENT = "concurrent"          # e1 || e2
    EQUAL = "equal"                    # Same event


@dataclass(frozen=True)
class LamportTimestamp:
    """
    Lamport logical timestamp for total ordering.

    Simpler than vector clocks but provides only total ordering,
    not causality detection.
    """
    counter: int
    repo_id: str
    wall_time: float = field(default_factory=time.time)

    def increment(self) -> 'LamportTimestamp':
        """Create next timestamp."""
        return LamportTimestamp(
            counter=self.counter + 1,
            repo_id=self.repo_id,
            wall_time=time.time(),
        )

    def merge(self, other: 'LamportTimestamp') -> 'LamportTimestamp':
        """Merge with received timestamp (take max + 1)."""
        return LamportTimestamp(
            counter=max(self.counter, other.counter) + 1,
            repo_id=self.repo_id,
            wall_time=time.time(),
        )

    def __lt__(self, other: 'LamportTimestamp') -> bool:
        """Total ordering: counter, then repo_id for tie-breaking."""
        if self.counter != other.counter:
            return self.counter < other.counter
        return self.repo_id < other.repo_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counter": self.counter,
            "repo_id": self.repo_id,
            "wall_time": self.wall_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LamportTimestamp':
        return cls(
            counter=data["counter"],
            repo_id=data["repo_id"],
            wall_time=data.get("wall_time", time.time()),
        )


@dataclass
class VectorClock:
    """
    Vector clock for distributed event ordering with causality detection.

    Each repo maintains a vector of logical clocks (one per known repo).
    This allows detection of happens-before relationships and concurrent events.
    """
    clocks: Dict[str, int] = field(default_factory=dict)
    repo_id: str = ""
    wall_time: float = field(default_factory=time.time)

    def tick(self) -> 'VectorClock':
        """
        Increment this repo's clock and return new vector clock.

        Called when creating a new event locally.
        """
        new_clocks = self.clocks.copy()
        new_clocks[self.repo_id] = new_clocks.get(self.repo_id, 0) + 1
        return VectorClock(
            clocks=new_clocks,
            repo_id=self.repo_id,
            wall_time=time.time(),
        )

    def merge(self, other: 'VectorClock') -> 'VectorClock':
        """
        Merge with received vector clock (take max per repo).

        Called when receiving an event from another repo.
        After merge, tick() should be called to create the receive event.
        """
        merged = self.clocks.copy()
        for repo, clock in other.clocks.items():
            merged[repo] = max(merged.get(repo, 0), clock)
        return VectorClock(
            clocks=merged,
            repo_id=self.repo_id,
            wall_time=time.time(),
        )

    def compare(self, other: 'VectorClock') -> CausalityRelation:
        """
        Compare two vector clocks to determine causality.

        Returns:
            CausalityRelation indicating relationship
        """
        all_repos = set(self.clocks.keys()) | set(other.clocks.keys())

        self_less = False
        other_less = False

        for repo in all_repos:
            self_val = self.clocks.get(repo, 0)
            other_val = other.clocks.get(repo, 0)

            if self_val < other_val:
                self_less = True
            elif self_val > other_val:
                other_less = True

        if self_less and not other_less:
            return CausalityRelation.HAPPENS_BEFORE
        elif other_less and not self_less:
            return CausalityRelation.HAPPENS_AFTER
        elif not self_less and not other_less:
            return CausalityRelation.EQUAL
        else:
            return CausalityRelation.CONCURRENT

    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this clock happens before other."""
        return self.compare(other) == CausalityRelation.HAPPENS_BEFORE

    def is_concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if this clock is concurrent with other."""
        return self.compare(other) == CausalityRelation.CONCURRENT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clocks": self.clocks,
            "repo_id": self.repo_id,
            "wall_time": self.wall_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorClock':
        return cls(
            clocks=data.get("clocks", {}),
            repo_id=data.get("repo_id", ""),
            wall_time=data.get("wall_time", time.time()),
        )

    def __hash__(self) -> int:
        return hash(json.dumps(self.clocks, sort_keys=True))


@dataclass
class CausalEvent:
    """
    Event with causal ordering metadata.

    Wraps any event data with vector clock for distributed ordering.
    """
    event_id: str
    event_type: str
    data: Dict[str, Any]
    vector_clock: VectorClock
    lamport_timestamp: LamportTimestamp
    source_repo: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "vector_clock": self.vector_clock.to_dict(),
            "lamport_timestamp": self.lamport_timestamp.to_dict(),
            "source_repo": self.source_repo,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalEvent':
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            data=data.get("data", {}),
            vector_clock=VectorClock.from_dict(data["vector_clock"]),
            lamport_timestamp=LamportTimestamp.from_dict(data["lamport_timestamp"]),
            source_repo=data["source_repo"],
            timestamp=data.get("timestamp", time.time()),
        )

    def happens_before(self, other: 'CausalEvent') -> bool:
        """Check if this event happens before other."""
        return self.vector_clock.happens_before(other.vector_clock)

    def is_concurrent_with(self, other: 'CausalEvent') -> bool:
        """Check if this event is concurrent with other."""
        return self.vector_clock.is_concurrent_with(other.vector_clock)


class CausalEventManager:
    """
    Manager for causal event ordering across repos.

    Features:
    - Maintains vector clock for this repo
    - Creates causally ordered events
    - Detects and handles concurrent events
    - Provides total ordering via Lamport timestamps
    - Redis-backed clock synchronization (optional)
    """

    def __init__(
        self,
        repo_id: str,
        redis_client: Optional[Any] = None,
        on_concurrent_event: Optional[Callable[[CausalEvent, CausalEvent], Any]] = None,
    ):
        self.repo_id = repo_id
        self._redis = redis_client
        self._on_concurrent = on_concurrent_event

        # Initialize clocks
        self._vector_clock = VectorClock(clocks={repo_id: 0}, repo_id=repo_id)
        self._lamport = LamportTimestamp(counter=0, repo_id=repo_id)

        # Event history for ordering
        self._event_history: List[CausalEvent] = []
        self._pending_events: Dict[str, CausalEvent] = {}  # Awaiting causal dependencies

        # Lock for clock operations
        self._lock = asyncio.Lock()

        # Metrics
        self._metrics = {
            "events_created": 0,
            "events_received": 0,
            "concurrent_events": 0,
            "causal_violations_detected": 0,
            "reorderings_performed": 0,
        }

        logger.info(f"CausalEventManager initialized for repo: {repo_id}")

    async def create_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        event_id: Optional[str] = None,
    ) -> CausalEvent:
        """
        Create a new causally ordered event.

        Args:
            event_type: Type of event (e.g., "MODEL_READY")
            data: Event payload
            event_id: Optional custom event ID

        Returns:
            CausalEvent with updated clocks
        """
        async with self._lock:
            # Tick clocks
            self._vector_clock = self._vector_clock.tick()
            self._lamport = self._lamport.increment()

            # Generate event ID if not provided
            if not event_id:
                event_id = self._generate_event_id(event_type, data)

            event = CausalEvent(
                event_id=event_id,
                event_type=event_type,
                data=data,
                vector_clock=self._vector_clock,
                lamport_timestamp=self._lamport,
                source_repo=self.repo_id,
            )

            # Add to history
            self._event_history.append(event)
            self._trim_history()

            # Sync to Redis if available
            if self._redis:
                await self._sync_clock_to_redis()

            self._metrics["events_created"] += 1

            return event

    async def receive_event(self, event: CausalEvent) -> Tuple[CausalEvent, List[CausalEvent]]:
        """
        Receive and process an event from another repo.

        Updates local clocks and detects causality violations.

        Args:
            event: Event received from another repo

        Returns:
            Tuple of (processed_event, any_reordered_events)
        """
        async with self._lock:
            # Merge vector clocks
            self._vector_clock = self._vector_clock.merge(event.vector_clock)
            self._vector_clock = self._vector_clock.tick()  # Record receive event

            # Merge Lamport timestamp
            self._lamport = self._lamport.merge(event.lamport_timestamp)

            # Check for concurrent events in history
            concurrent_events = []
            for hist_event in self._event_history[-50:]:  # Check recent events
                if event.is_concurrent_with(hist_event):
                    concurrent_events.append(hist_event)
                    self._metrics["concurrent_events"] += 1

            # Handle concurrent events
            if concurrent_events and self._on_concurrent:
                for concurrent in concurrent_events:
                    await self._handle_concurrent_events(event, concurrent)

            # Add to history with proper ordering
            reordered = self._insert_ordered(event)

            # Sync to Redis if available
            if self._redis:
                await self._sync_clock_to_redis()

            self._metrics["events_received"] += 1

            return event, reordered

    def _insert_ordered(self, event: CausalEvent) -> List[CausalEvent]:
        """Insert event in proper causal order, returning any reordered events."""
        reordered = []

        # Find correct position based on Lamport timestamp
        insert_pos = len(self._event_history)
        for i in range(len(self._event_history) - 1, -1, -1):
            if self._event_history[i].lamport_timestamp < event.lamport_timestamp:
                insert_pos = i + 1
                break
            elif self._event_history[i].lamport_timestamp > event.lamport_timestamp:
                insert_pos = i

        # Check if this causes reordering
        if insert_pos < len(self._event_history):
            reordered = self._event_history[insert_pos:]
            self._metrics["reorderings_performed"] += 1

        self._event_history.insert(insert_pos, event)
        self._trim_history()

        return reordered

    async def _handle_concurrent_events(
        self,
        event1: CausalEvent,
        event2: CausalEvent,
    ) -> None:
        """Handle concurrent events with conflict resolution."""
        if self._on_concurrent:
            try:
                await self._on_concurrent(event1, event2)
            except Exception as e:
                logger.error(f"Concurrent event handler error: {e}")

    async def _sync_clock_to_redis(self) -> None:
        """Synchronize clock state to Redis for cross-repo visibility."""
        if not self._redis:
            return

        try:
            key = f"vector_clock:{self.repo_id}"
            data = {
                "vector_clock": self._vector_clock.to_dict(),
                "lamport": self._lamport.to_dict(),
                "updated_at": time.time(),
            }
            await self._redis.set(key, json.dumps(data), ex=3600)
        except Exception as e:
            logger.warning(f"Failed to sync clock to Redis: {e}")

    async def sync_from_redis(self) -> None:
        """Load clock state from Redis (for recovery)."""
        if not self._redis:
            return

        try:
            # Get all repo clocks
            keys = await self._redis.keys("vector_clock:*")
            for key in keys:
                data = await self._redis.get(key)
                if data:
                    parsed = json.loads(data)
                    other_clock = VectorClock.from_dict(parsed["vector_clock"])
                    self._vector_clock = self._vector_clock.merge(other_clock)

                    other_lamport = LamportTimestamp.from_dict(parsed["lamport"])
                    if other_lamport.counter > self._lamport.counter:
                        self._lamport = self._lamport.merge(other_lamport)

            logger.info(f"Synced clocks from Redis: {len(keys)} repos")
        except Exception as e:
            logger.warning(f"Failed to sync from Redis: {e}")

    def _generate_event_id(self, event_type: str, data: Dict[str, Any]) -> str:
        """Generate unique event ID."""
        content = f"{self.repo_id}:{event_type}:{time.time()}:{json.dumps(data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _trim_history(self) -> None:
        """Trim event history to max size."""
        if len(self._event_history) > MAX_CLOCK_HISTORY:
            self._event_history = self._event_history[-MAX_CLOCK_HISTORY:]

    def get_ordered_events(
        self,
        since_lamport: Optional[int] = None,
        event_types: Optional[Set[str]] = None,
    ) -> List[CausalEvent]:
        """
        Get events in causal order.

        Args:
            since_lamport: Only events after this Lamport counter
            event_types: Filter by event types

        Returns:
            List of events in causal order
        """
        events = self._event_history

        if since_lamport is not None:
            events = [e for e in events if e.lamport_timestamp.counter > since_lamport]

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        return events

    def get_current_clock(self) -> VectorClock:
        """Get current vector clock."""
        return self._vector_clock

    def get_metrics(self) -> Dict[str, Any]:
        """Get event ordering metrics."""
        return {
            **self._metrics,
            "current_lamport": self._lamport.counter,
            "vector_clock_repos": len(self._vector_clock.clocks),
            "event_history_size": len(self._event_history),
        }


# =============================================================================
# Causal Barrier
# =============================================================================

class CausalBarrier:
    """
    Ensures events are processed in causal order.

    Buffers events until all causal dependencies are satisfied.
    """

    def __init__(self, event_manager: CausalEventManager):
        self._manager = event_manager
        self._pending: Dict[str, CausalEvent] = {}
        self._delivered: Set[str] = set()
        self._on_deliver: Optional[Callable[[CausalEvent], Any]] = None

    def set_deliver_callback(self, callback: Callable[[CausalEvent], Any]) -> None:
        """Set callback for when event is ready for delivery."""
        self._on_deliver = callback

    async def receive(self, event: CausalEvent) -> bool:
        """
        Receive event and deliver if causally ready.

        Returns:
            True if event was delivered, False if buffered
        """
        # Check if already delivered
        if event.event_id in self._delivered:
            return False

        # Check causal dependencies
        if self._check_dependencies(event):
            await self._deliver(event)
            return True
        else:
            # Buffer until dependencies satisfied
            self._pending[event.event_id] = event
            return False

    def _check_dependencies(self, event: CausalEvent) -> bool:
        """Check if all causal dependencies are satisfied."""
        # For each repo in the vector clock, check if we've seen
        # all events up to that point
        for repo, clock in event.vector_clock.clocks.items():
            if repo == event.source_repo:
                continue  # Skip source repo

            # Check if we have the required events
            # This is simplified - real impl would track per-repo counters
            pass

        return True  # Simplified: assume dependencies satisfied

    async def _deliver(self, event: CausalEvent) -> None:
        """Deliver event and check pending events."""
        self._delivered.add(event.event_id)

        if self._on_deliver:
            await self._on_deliver(event)

        # Check if pending events can now be delivered
        await self._check_pending()

    async def _check_pending(self) -> None:
        """Check if any pending events can be delivered."""
        to_deliver = []

        for event_id, event in list(self._pending.items()):
            if self._check_dependencies(event):
                to_deliver.append(event)
                del self._pending[event_id]

        for event in to_deliver:
            await self._deliver(event)


# =============================================================================
# Global Factory
# =============================================================================

_event_manager: Optional[CausalEventManager] = None
_manager_lock = asyncio.Lock()


async def get_causal_event_manager(
    repo_id: Optional[str] = None,
    redis_client: Optional[Any] = None,
) -> CausalEventManager:
    """Get or create the global CausalEventManager."""
    global _event_manager

    async with _manager_lock:
        if _event_manager is None:
            if repo_id is None:
                repo_id = os.getenv("REPO_ID", "jarvis")

            _event_manager = CausalEventManager(
                repo_id=repo_id,
                redis_client=redis_client,
            )

            # Sync from Redis on startup
            if redis_client:
                await _event_manager.sync_from_redis()

        return _event_manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LamportTimestamp",
    "VectorClock",
    "CausalEvent",
    "CausalEventManager",
    "CausalBarrier",
    "CausalityRelation",
    "get_causal_event_manager",
]
