"""
v77.1: Evolution Event Store - Event Sourcing
==============================================

Event sourcing for complete evolution audit trail.

Every action is an immutable event:
- EvolutionStarted
- FrameworkSelected
- CodeChanged
- ValidationPassed
- EvolutionCompleted

Benefits:
- Complete replay of evolution
- Debugging failed evolutions
- Learning from past evolutions
- Time-travel debugging
- Audit compliance

Author: Ironcliw v77.1
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of evolution events."""
    # Lifecycle events
    EVOLUTION_STARTED = "evolution_started"
    EVOLUTION_COMPLETED = "evolution_completed"
    EVOLUTION_FAILED = "evolution_failed"
    EVOLUTION_ABORTED = "evolution_aborted"

    # Framework events
    FRAMEWORK_SELECTED = "framework_selected"
    FRAMEWORK_EXECUTION_STARTED = "framework_execution_started"
    FRAMEWORK_EXECUTION_COMPLETED = "framework_execution_completed"
    FRAMEWORK_EXECUTION_FAILED = "framework_execution_failed"

    # Code change events
    CODE_ANALYSIS_STARTED = "code_analysis_started"
    CODE_ANALYSIS_COMPLETED = "code_analysis_completed"
    CODE_CHANGE_PROPOSED = "code_change_proposed"
    CODE_CHANGE_APPLIED = "code_change_applied"
    CODE_CHANGE_REJECTED = "code_change_rejected"

    # Validation events
    VALIDATION_STARTED = "validation_started"
    VALIDATION_PASSED = "validation_passed"
    VALIDATION_FAILED = "validation_failed"

    # Transaction events
    TRANSACTION_STARTED = "transaction_started"
    TRANSACTION_PREPARED = "transaction_prepared"
    TRANSACTION_COMMITTED = "transaction_committed"
    TRANSACTION_ABORTED = "transaction_aborted"
    TRANSACTION_ROLLED_BACK = "transaction_rolled_back"

    # Saga events
    SAGA_STARTED = "saga_started"
    SAGA_STEP_COMPLETED = "saga_step_completed"
    SAGA_STEP_FAILED = "saga_step_failed"
    SAGA_COMPENSATED = "saga_compensated"
    SAGA_COMPLETED = "saga_completed"

    # Cross-repo events
    CROSS_REPO_SYNC_STARTED = "cross_repo_sync_started"
    CROSS_REPO_SYNC_COMPLETED = "cross_repo_sync_completed"
    DEPENDENCY_CONFLICT_DETECTED = "dependency_conflict_detected"

    # Recovery events
    CHECKPOINT_CREATED = "checkpoint_created"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"

    # Custom events
    CUSTOM = "custom"


@dataclass
class EvolutionEvent:
    """
    An immutable event in the evolution lifecycle.

    Events are append-only and never modified after creation.
    """
    event_id: str
    event_type: EventType
    timestamp: float
    evolution_id: str
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None  # ID of event that caused this one
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "datetime": self.datetime.isoformat(),
            "evolution_id": self.evolution_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "data": self.data,
            "metadata": self.metadata,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionEvent":
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            evolution_id=data["evolution_id"],
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            version=data.get("version", 1),
        )

    def checksum(self) -> str:
        """Calculate event checksum for integrity verification."""
        content = f"{self.event_id}:{self.event_type.value}:{self.timestamp}:{self.evolution_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class EventStream:
    """
    A stream of events for a specific evolution.

    Enables replaying events to reconstruct state.
    """
    evolution_id: str
    events: List[EvolutionEvent] = field(default_factory=list)
    version: int = 0

    def append(self, event: EvolutionEvent) -> None:
        """Append event to stream."""
        self.events.append(event)
        self.version += 1

    def replay(self) -> Iterator[EvolutionEvent]:
        """Replay events in order."""
        for event in sorted(self.events, key=lambda e: e.timestamp):
            yield event

    def filter_by_type(self, event_type: EventType) -> List[EvolutionEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_latest(self) -> Optional[EvolutionEvent]:
        """Get most recent event."""
        if not self.events:
            return None
        return max(self.events, key=lambda e: e.timestamp)

    def get_duration_ms(self) -> float:
        """Calculate total duration from first to last event."""
        if len(self.events) < 2:
            return 0.0
        events = sorted(self.events, key=lambda e: e.timestamp)
        return (events[-1].timestamp - events[0].timestamp) * 1000


class EvolutionEventStore:
    """
    Persistent event store using SQLite.

    Features:
    - Append-only event storage
    - Event stream retrieval
    - Query by type, time range, evolution
    - Snapshot support for faster replay
    - Event integrity verification
    - Async-safe operations

    Usage:
        store = EvolutionEventStore()
        await store.start()

        # Append event
        event = EvolutionEvent(
            event_type=EventType.EVOLUTION_STARTED,
            evolution_id="ev123",
            data={"description": "Add feature X"},
        )
        await store.append(event)

        # Replay stream
        stream = await store.get_stream("ev123")
        for event in stream.replay():
            print(event)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        snapshot_interval: int = 100,  # Create snapshot every N events
    ):
        self.db_path = db_path or Path.home() / ".jarvis" / "evolution_events.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval

        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._event_handlers: List[Callable[[EvolutionEvent], Coroutine]] = []

    async def start(self) -> None:
        """Initialize the event store."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        await self._create_schema()
        logger.info(f"[EventStore] Started with DB: {self.db_path}")

    async def stop(self) -> None:
        """Close the event store."""
        if self._conn:
            self._conn.close()
            self._conn = None
        logger.info("[EventStore] Stopped")

    async def _create_schema(self) -> None:
        """Create database schema."""
        if not self._conn:
            return

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                evolution_id TEXT NOT NULL,
                correlation_id TEXT,
                causation_id TEXT,
                data TEXT NOT NULL,
                metadata TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                checksum TEXT NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_events_evolution ON events(evolution_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id);

            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evolution_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                state TEXT NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(evolution_id, version)
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_evolution ON snapshots(evolution_id);
        """)
        self._conn.commit()

    async def append(self, event: EvolutionEvent) -> str:
        """
        Append an event to the store.

        Args:
            event: The event to append

        Returns:
            The event ID
        """
        async with self._lock:
            if not self._conn:
                raise RuntimeError("Event store not started")

            checksum = event.checksum()

            self._conn.execute("""
                INSERT INTO events (
                    event_id, event_type, timestamp, evolution_id,
                    correlation_id, causation_id, data, metadata,
                    version, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.value,
                event.timestamp,
                event.evolution_id,
                event.correlation_id,
                event.causation_id,
                json.dumps(event.data),
                json.dumps(event.metadata),
                event.version,
                checksum,
            ))
            self._conn.commit()

        # Notify handlers
        for handler in self._event_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"[EventStore] Handler error: {e}")

        logger.debug(f"[EventStore] Appended: {event.event_type.value} for {event.evolution_id}")
        return event.event_id

    async def append_many(self, events: List[EvolutionEvent]) -> List[str]:
        """Append multiple events atomically."""
        async with self._lock:
            if not self._conn:
                raise RuntimeError("Event store not started")

            event_ids = []
            for event in events:
                checksum = event.checksum()
                self._conn.execute("""
                    INSERT INTO events (
                        event_id, event_type, timestamp, evolution_id,
                        correlation_id, causation_id, data, metadata,
                        version, checksum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event.timestamp,
                    event.evolution_id,
                    event.correlation_id,
                    event.causation_id,
                    json.dumps(event.data),
                    json.dumps(event.metadata),
                    event.version,
                    checksum,
                ))
                event_ids.append(event.event_id)

            self._conn.commit()

        return event_ids

    async def get_stream(self, evolution_id: str) -> EventStream:
        """
        Get the complete event stream for an evolution.

        Args:
            evolution_id: The evolution ID

        Returns:
            EventStream with all events
        """
        async with self._lock:
            if not self._conn:
                raise RuntimeError("Event store not started")

            cursor = self._conn.execute("""
                SELECT event_id, event_type, timestamp, evolution_id,
                       correlation_id, causation_id, data, metadata, version
                FROM events
                WHERE evolution_id = ?
                ORDER BY timestamp ASC
            """, (evolution_id,))

            stream = EventStream(evolution_id=evolution_id)

            for row in cursor.fetchall():
                event = EvolutionEvent(
                    event_id=row[0],
                    event_type=EventType(row[1]),
                    timestamp=row[2],
                    evolution_id=row[3],
                    correlation_id=row[4],
                    causation_id=row[5],
                    data=json.loads(row[6]),
                    metadata=json.loads(row[7]),
                    version=row[8],
                )
                stream.append(event)

            return stream

    async def get_events_by_type(
        self,
        event_type: EventType,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[EvolutionEvent]:
        """Get events by type."""
        async with self._lock:
            if not self._conn:
                return []

            if since:
                cursor = self._conn.execute("""
                    SELECT event_id, event_type, timestamp, evolution_id,
                           correlation_id, causation_id, data, metadata, version
                    FROM events
                    WHERE event_type = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (event_type.value, since, limit))
            else:
                cursor = self._conn.execute("""
                    SELECT event_id, event_type, timestamp, evolution_id,
                           correlation_id, causation_id, data, metadata, version
                    FROM events
                    WHERE event_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (event_type.value, limit))

            events = []
            for row in cursor.fetchall():
                events.append(EvolutionEvent(
                    event_id=row[0],
                    event_type=EventType(row[1]),
                    timestamp=row[2],
                    evolution_id=row[3],
                    correlation_id=row[4],
                    causation_id=row[5],
                    data=json.loads(row[6]),
                    metadata=json.loads(row[7]),
                    version=row[8],
                ))

            return events

    async def get_recent_evolutions(
        self,
        limit: int = 10,
    ) -> List[str]:
        """Get IDs of recent evolutions."""
        async with self._lock:
            if not self._conn:
                return []

            cursor = self._conn.execute("""
                SELECT DISTINCT evolution_id
                FROM events
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            return [row[0] for row in cursor.fetchall()]

    async def query(
        self,
        evolution_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: int = 1000,
    ) -> List[EvolutionEvent]:
        """
        Query events with filters.

        Args:
            evolution_id: Filter by evolution
            event_types: Filter by event types
            since: Filter by start timestamp
            until: Filter by end timestamp
            limit: Maximum events to return
        """
        async with self._lock:
            if not self._conn:
                return []

            conditions = []
            params = []

            if evolution_id:
                conditions.append("evolution_id = ?")
                params.append(evolution_id)

            if event_types:
                placeholders = ",".join("?" * len(event_types))
                conditions.append(f"event_type IN ({placeholders})")
                params.extend(t.value for t in event_types)

            if since:
                conditions.append("timestamp >= ?")
                params.append(since)

            if until:
                conditions.append("timestamp <= ?")
                params.append(until)

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            params.append(limit)

            cursor = self._conn.execute(f"""
                SELECT event_id, event_type, timestamp, evolution_id,
                       correlation_id, causation_id, data, metadata, version
                FROM events
                WHERE {where_clause}
                ORDER BY timestamp ASC
                LIMIT ?
            """, params)

            events = []
            for row in cursor.fetchall():
                events.append(EvolutionEvent(
                    event_id=row[0],
                    event_type=EventType(row[1]),
                    timestamp=row[2],
                    evolution_id=row[3],
                    correlation_id=row[4],
                    causation_id=row[5],
                    data=json.loads(row[6]),
                    metadata=json.loads(row[7]),
                    version=row[8],
                ))

            return events

    async def count_by_evolution(self, evolution_id: str) -> int:
        """Count events for an evolution."""
        async with self._lock:
            if not self._conn:
                return 0

            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE evolution_id = ?",
                (evolution_id,)
            )
            return cursor.fetchone()[0]

    async def verify_integrity(self, evolution_id: str) -> Tuple[bool, List[str]]:
        """
        Verify event integrity using checksums.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        async with self._lock:
            if not self._conn:
                return False, ["Store not started"]

            cursor = self._conn.execute("""
                SELECT event_id, event_type, timestamp, evolution_id, checksum
                FROM events
                WHERE evolution_id = ?
            """, (evolution_id,))

            errors = []
            for row in cursor.fetchall():
                event_id, event_type, timestamp, evo_id, stored_checksum = row
                expected = hashlib.sha256(
                    f"{event_id}:{event_type}:{timestamp}:{evo_id}".encode()
                ).hexdigest()[:16]

                if expected != stored_checksum:
                    errors.append(f"Checksum mismatch for event {event_id}")

            return len(errors) == 0, errors

    def on_event(self, handler: Callable[[EvolutionEvent], Coroutine]) -> None:
        """Register an event handler for real-time notifications."""
        self._event_handlers.append(handler)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get event store statistics."""
        async with self._lock:
            if not self._conn:
                return {}

            total = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            evolutions = self._conn.execute(
                "SELECT COUNT(DISTINCT evolution_id) FROM events"
            ).fetchone()[0]

            type_counts = {}
            for row in self._conn.execute("""
                SELECT event_type, COUNT(*) FROM events GROUP BY event_type
            """).fetchall():
                type_counts[row[0]] = row[1]

            return {
                "total_events": total,
                "total_evolutions": evolutions,
                "events_by_type": type_counts,
                "db_path": str(self.db_path),
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            }


# Convenience functions for common events
class EventFactory:
    """Factory for creating common evolution events."""

    @staticmethod
    def evolution_started(
        evolution_id: str,
        description: str,
        target_files: List[str],
        **kwargs,
    ) -> EvolutionEvent:
        return EvolutionEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.EVOLUTION_STARTED,
            timestamp=time.time(),
            evolution_id=evolution_id,
            data={
                "description": description,
                "target_files": target_files,
                **kwargs,
            },
        )

    @staticmethod
    def evolution_completed(
        evolution_id: str,
        success: bool,
        files_modified: List[str],
        duration_ms: float,
        **kwargs,
    ) -> EvolutionEvent:
        return EvolutionEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.EVOLUTION_COMPLETED,
            timestamp=time.time(),
            evolution_id=evolution_id,
            data={
                "success": success,
                "files_modified": files_modified,
                "duration_ms": duration_ms,
                **kwargs,
            },
        )

    @staticmethod
    def framework_selected(
        evolution_id: str,
        framework: str,
        reason: str,
        score: float,
        **kwargs,
    ) -> EvolutionEvent:
        return EvolutionEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.FRAMEWORK_SELECTED,
            timestamp=time.time(),
            evolution_id=evolution_id,
            data={
                "framework": framework,
                "reason": reason,
                "score": score,
                **kwargs,
            },
        )

    @staticmethod
    def code_change_applied(
        evolution_id: str,
        file_path: str,
        change_type: str,
        lines_added: int,
        lines_removed: int,
        **kwargs,
    ) -> EvolutionEvent:
        return EvolutionEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.CODE_CHANGE_APPLIED,
            timestamp=time.time(),
            evolution_id=evolution_id,
            data={
                "file_path": file_path,
                "change_type": change_type,
                "lines_added": lines_added,
                "lines_removed": lines_removed,
                **kwargs,
            },
        )

    @staticmethod
    def validation_result(
        evolution_id: str,
        passed: bool,
        validation_type: str,
        details: Dict[str, Any],
        **kwargs,
    ) -> EvolutionEvent:
        event_type = EventType.VALIDATION_PASSED if passed else EventType.VALIDATION_FAILED
        return EvolutionEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            evolution_id=evolution_id,
            data={
                "passed": passed,
                "validation_type": validation_type,
                "details": details,
                **kwargs,
            },
        )


# Global instance
_event_store: Optional[EvolutionEventStore] = None


async def get_evolution_event_store() -> EvolutionEventStore:
    """Get or create global event store."""
    global _event_store
    if _event_store is None:
        _event_store = EvolutionEventStore()
        await _event_store.start()
    return _event_store
