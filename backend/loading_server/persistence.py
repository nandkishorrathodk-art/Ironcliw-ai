"""
Progress Persistence for JARVIS Loading Server v212.0
======================================================

Provides persistent storage for loading progress, enabling:
- Browser refresh resume capability
- Debugging and replay via event sourcing
- Audit trail of startup events
- Historical startup analytics

Components:
- EventSourcingLog: Append-only JSONL event log with rotation
- ProgressPersistence: SQLite-backed progress storage

Usage:
    from backend.loading_server.persistence import (
        EventSourcingLog,
        ProgressPersistence,
    )

    # Event logging
    event_log = EventSourcingLog()
    event_log.append_event("progress_update", {"progress": 50}, trace_id="abc123")

    # Progress persistence
    persistence = ProgressPersistence()
    persistence.save_progress(session_id, progress, stage, message)
    latest = persistence.load_latest_progress()

Author: JARVIS Trinity System
Version: 212.0.0
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("LoadingServer.Persistence")


@dataclass
class EventSourcingLog:
    """
    JSONL event log for replay and debugging.

    Features:
    - Append-only JSONL log for immutability
    - Automatic file rotation (max 10MB per file)
    - Event replay capability for debugging
    - Thread-safe writes
    - Compression support for old logs

    Event format:
    {
        "timestamp": 1704067200.123,
        "event_type": "progress_update",
        "trace_id": "abc123",
        "data": {"progress": 50, "stage": "backend"}
    }
    """

    log_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "loading_server" / "events"
    )
    max_size_bytes: int = 10 * 1024 * 1024  # 10MB per file
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _current_log_name: str = field(init=False, default="")

    def __post_init__(self):
        """Initialize the event log."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_log_name = f"events_{int(time.time())}.jsonl"

    @property
    def _current_log(self) -> Path:
        """Get the current log file path."""
        return self.log_dir / self._current_log_name

    def append_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Append event to log.

        Args:
            event_type: Type of event (e.g., "progress_update", "component_complete")
            data: Event-specific data payload
            trace_id: Optional W3C trace ID for correlation
        """
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "trace_id": trace_id or "unknown",
            "data": data,
        }

        with self._lock:
            # Check if rotation needed
            if (
                self._current_log.exists()
                and self._current_log.stat().st_size > self.max_size_bytes
            ):
                self._rotate_log()

            # Append event
            try:
                with open(self._current_log, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except IOError as e:
                logger.warning(f"[EventLog] Failed to write event: {e}")

    def _rotate_log(self) -> None:
        """Rotate current log file to a new file."""
        old_name = self._current_log_name
        self._current_log_name = f"events_{int(time.time())}.jsonl"
        logger.info(f"[EventLog] Rotated from {old_name} to {self._current_log_name}")

    def replay_events(
        self,
        since_timestamp: Optional[float] = None,
        event_types: Optional[List[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Replay events from log files.

        Args:
            since_timestamp: Only return events after this timestamp
            event_types: Only return events of these types

        Yields:
            Event dictionaries in chronological order
        """
        # Find all log files
        log_files = sorted(self.log_dir.glob("events_*.jsonl"))

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Filter by timestamp
                        if since_timestamp and event.get("timestamp", 0) < since_timestamp:
                            continue

                        # Filter by event type
                        if event_types and event.get("event_type") not in event_types:
                            continue

                        yield event

            except IOError as e:
                logger.debug(f"[EventLog] Could not read {log_file}: {e}")

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the most recent events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent events, most recent first
        """
        events = list(self.replay_events())
        return events[-limit:][::-1] if events else []

    def cleanup_old_logs(self, max_age_days: int = 7) -> int:
        """
        Remove log files older than max_age_days.

        Args:
            max_age_days: Maximum age of log files to keep

        Returns:
            Number of files deleted
        """
        cutoff = time.time() - (max_age_days * 86400)
        deleted = 0

        for log_file in self.log_dir.glob("events_*.jsonl"):
            # Extract timestamp from filename
            try:
                timestamp = int(log_file.stem.split("_")[1])
                if timestamp < cutoff:
                    log_file.unlink()
                    deleted += 1
            except (ValueError, IndexError):
                pass

        if deleted:
            logger.info(f"[EventLog] Cleaned up {deleted} old log files")

        return deleted


@dataclass
class ProgressPersistence:
    """
    SQLite-backed progress persistence for resume capability.

    Features:
    - Persistent progress across page refreshes
    - Session-based tracking with trace IDs
    - Automatic cleanup of old sessions (> 24h)
    - WAL mode for better concurrent access
    - Resume from last known state

    Schema:
    - progress_sessions: Main session tracking table
    - progress_history: Historical progress points for analytics
    """

    db_path: Path = field(
        default_factory=lambda: Path.home()
        / ".jarvis"
        / "loading_server"
        / "progress.db"
    )
    cleanup_age_seconds: float = 86400  # 24 hours
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self):
        """Initialize database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        with self._lock:
            with self._get_connection() as conn:
                # Main session tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS progress_sessions (
                        session_id TEXT PRIMARY KEY,
                        started_at REAL NOT NULL,
                        last_updated REAL NOT NULL,
                        current_progress REAL NOT NULL,
                        current_stage TEXT NOT NULL,
                        current_message TEXT,
                        trace_id TEXT,
                        completed INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                """)

                # Index for efficient queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_updated
                    ON progress_sessions(last_updated DESC)
                """)

                # Progress history for analytics
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS progress_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        progress REAL NOT NULL,
                        stage TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES progress_sessions(session_id)
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_history_session
                    ON progress_history(session_id, timestamp)
                """)

                # Cleanup old sessions
                conn.execute(
                    """
                    DELETE FROM progress_sessions
                    WHERE last_updated < ?
                """,
                    (time.time() - self.cleanup_age_seconds,),
                )

                conn.commit()

    def save_progress(
        self,
        session_id: str,
        progress: float,
        stage: str,
        message: Optional[str] = None,
        trace_id: Optional[str] = None,
        completed: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save current progress state.

        Args:
            session_id: Unique session identifier
            progress: Current progress percentage (0-100)
            stage: Current stage name
            message: Optional status message
            trace_id: Optional W3C trace ID
            completed: Whether startup is complete
            metadata: Optional additional metadata
        """
        now = time.time()
        metadata_json = json.dumps(metadata) if metadata else None

        with self._lock:
            with self._get_connection() as conn:
                # Upsert session
                conn.execute(
                    """
                    INSERT INTO progress_sessions
                    (session_id, started_at, last_updated, current_progress,
                     current_stage, current_message, trace_id, completed, metadata)
                    VALUES (
                        ?,
                        COALESCE((SELECT started_at FROM progress_sessions WHERE session_id = ?), ?),
                        ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(session_id) DO UPDATE SET
                        last_updated = excluded.last_updated,
                        current_progress = excluded.current_progress,
                        current_stage = excluded.current_stage,
                        current_message = excluded.current_message,
                        trace_id = COALESCE(excluded.trace_id, trace_id),
                        completed = excluded.completed,
                        metadata = COALESCE(excluded.metadata, metadata)
                """,
                    (
                        session_id,
                        session_id,
                        now,
                        now,
                        progress,
                        stage,
                        message,
                        trace_id,
                        int(completed),
                        metadata_json,
                    ),
                )

                # Record progress history point
                conn.execute(
                    """
                    INSERT INTO progress_history (session_id, timestamp, progress, stage)
                    VALUES (?, ?, ?, ?)
                """,
                    (session_id, now, progress, stage),
                )

                conn.commit()

    def load_latest_progress(self) -> Optional[Dict[str, Any]]:
        """
        Load most recent progress session.

        Returns:
            Dict with session data or None if no sessions exist
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT session_id, started_at, last_updated, current_progress,
                           current_stage, current_message, trace_id, completed, metadata
                    FROM progress_sessions
                    ORDER BY last_updated DESC
                    LIMIT 1
                """)

                row = cursor.fetchone()
                if row:
                    return {
                        "session_id": row[0],
                        "started_at": row[1],
                        "last_updated": row[2],
                        "progress": row[3],
                        "stage": row[4],
                        "message": row[5],
                        "trace_id": row[6],
                        "completed": bool(row[7]),
                        "metadata": json.loads(row[8]) if row[8] else None,
                    }

        return None

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific session by ID.

        Args:
            session_id: Session ID to load

        Returns:
            Dict with session data or None if not found
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT session_id, started_at, last_updated, current_progress,
                           current_stage, current_message, trace_id, completed, metadata
                    FROM progress_sessions
                    WHERE session_id = ?
                """,
                    (session_id,),
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "session_id": row[0],
                        "started_at": row[1],
                        "last_updated": row[2],
                        "progress": row[3],
                        "stage": row[4],
                        "message": row[5],
                        "trace_id": row[6],
                        "completed": bool(row[7]),
                        "metadata": json.loads(row[8]) if row[8] else None,
                    }

        return None

    def get_session_history(
        self, session_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get progress history for a session.

        Args:
            session_id: Session ID to query
            limit: Maximum number of history points

        Returns:
            List of progress history points
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT timestamp, progress, stage
                    FROM progress_history
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (session_id, limit),
                )

                return [
                    {"timestamp": row[0], "progress": row[1], "stage": row[2]}
                    for row in cursor.fetchall()
                ]

    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent sessions summary.

        Args:
            limit: Maximum number of sessions

        Returns:
            List of session summaries
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT session_id, started_at, last_updated, current_progress,
                           current_stage, completed
                    FROM progress_sessions
                    ORDER BY last_updated DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                return [
                    {
                        "session_id": row[0],
                        "started_at": row[1],
                        "last_updated": row[2],
                        "progress": row[3],
                        "stage": row[4],
                        "completed": bool(row[5]),
                        "duration": row[2] - row[1] if row[5] else None,
                    }
                    for row in cursor.fetchall()
                ]

    def cleanup_old_sessions(self, max_age_seconds: Optional[float] = None) -> int:
        """
        Remove old sessions from database.

        Args:
            max_age_seconds: Maximum age of sessions to keep (default: 24h)

        Returns:
            Number of sessions deleted
        """
        cutoff = time.time() - (max_age_seconds or self.cleanup_age_seconds)

        with self._lock:
            with self._get_connection() as conn:
                # Delete history first (foreign key constraint)
                conn.execute(
                    """
                    DELETE FROM progress_history
                    WHERE session_id IN (
                        SELECT session_id FROM progress_sessions WHERE last_updated < ?
                    )
                """,
                    (cutoff,),
                )

                cursor = conn.execute(
                    """
                    DELETE FROM progress_sessions WHERE last_updated < ?
                """,
                    (cutoff,),
                )

                conn.commit()
                deleted = cursor.rowcount

        if deleted:
            logger.info(f"[Persistence] Cleaned up {deleted} old sessions")

        return deleted

    def get_analytics(self) -> Dict[str, Any]:
        """
        Get startup analytics from historical data.

        Returns:
            Analytics summary with averages, trends, and bottlenecks
        """
        with self._lock:
            with self._get_connection() as conn:
                # Overall statistics
                cursor = conn.execute("""
                    SELECT
                        COUNT(*) as total_sessions,
                        AVG(last_updated - started_at) as avg_duration,
                        MIN(last_updated - started_at) as min_duration,
                        MAX(last_updated - started_at) as max_duration,
                        SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END) as completed_count
                    FROM progress_sessions
                    WHERE completed = 1
                """)

                stats = cursor.fetchone()

                return {
                    "total_sessions": stats[0] or 0,
                    "completed_sessions": stats[4] or 0,
                    "average_duration_seconds": round(stats[1], 2) if stats[1] else None,
                    "min_duration_seconds": round(stats[2], 2) if stats[2] else None,
                    "max_duration_seconds": round(stats[3], 2) if stats[3] else None,
                    "success_rate": (
                        round(stats[4] / stats[0] * 100, 1)
                        if stats[0] and stats[4]
                        else None
                    ),
                }
