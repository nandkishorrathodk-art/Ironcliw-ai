"""
v77.2: Memory Management & Retention Policies - Gaps #64, #67
==============================================================

Prevents unbounded memory growth and manages data retention.

Problems:
    - Gap #64: Unbounded memory growth in dicts (_active_sagas, etc.)
    - Gap #67: Event store database grows without limit
    - Gap #80: Event handlers never cleaned up

Solutions:
    - LRU cache with size limits
    - Time-based expiration
    - Automatic cleanup of completed items
    - Retention policies for persistent storage
    - Weak references for handlers

Features:
    - Bounded collections with eviction
    - Time-to-live (TTL) support
    - Automatic background cleanup
    - Storage retention policies
    - Memory usage monitoring
    - Handler lifecycle management

Author: JARVIS v77.2
"""

from __future__ import annotations

import asyncio
import gc
import logging
import sys
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


# ============================================================================
# Bounded Collections
# ============================================================================


class EvictionPolicy(Enum):
    """Policy for evicting items when cache is full."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time-based expiration


@dataclass
class CacheEntry(Generic[V]):
    """Entry in a bounded cache."""

    value: V
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def access(self) -> V:
        """Record access and return value."""
        self.accessed_at = time.time()
        self.access_count += 1
        return self.value


class BoundedDict(Generic[K, V]):
    """
    Dictionary with maximum size and eviction policy.

    Prevents unbounded memory growth by evicting items when full.

    Usage:
        cache = BoundedDict(max_size=1000, policy=EvictionPolicy.LRU)
        cache["key"] = value

        # With TTL
        cache.set("key", value, ttl=300)  # 5 minute TTL
    """

    def __init__(
        self,
        max_size: int = 1000,
        policy: EvictionPolicy = EvictionPolicy.LRU,
        default_ttl: Optional[float] = None,
        on_evict: Optional[Callable[[K, V], None]] = None,
    ):
        self.max_size = max_size
        self.policy = policy
        self.default_ttl = default_ttl
        self.on_evict = on_evict

        self._data: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._eviction_count = 0

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: K) -> bool:
        if key not in self._data:
            return False
        entry = self._data[key]
        if entry.is_expired:
            self._remove(key)
            return False
        return True

    def __getitem__(self, key: K) -> V:
        if key not in self._data:
            raise KeyError(key)

        entry = self._data[key]
        if entry.is_expired:
            self._remove(key)
            raise KeyError(key)

        # Update access time and move to end (LRU)
        if self.policy == EvictionPolicy.LRU:
            self._data.move_to_end(key)

        return entry.access()

    def __setitem__(self, key: K, value: V) -> None:
        self.set(key, value)

    def __delitem__(self, key: K) -> None:
        self._remove(key)

    def __iter__(self) -> Iterator[K]:
        # Clean expired first
        self._cleanup_expired()
        return iter(self._data)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get value or default."""
        try:
            return self[key]
        except KeyError:
            return default

    def set(
        self,
        key: K,
        value: V,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set a value with optional TTL.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (None = use default)
        """
        # Evict if needed
        if key not in self._data and len(self._data) >= self.max_size:
            self._evict_one()

        entry = CacheEntry(
            value=value,
            ttl_seconds=ttl if ttl is not None else self.default_ttl,
        )

        self._data[key] = entry

        # Move to end for LRU
        if self.policy == EvictionPolicy.LRU:
            self._data.move_to_end(key)

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Remove and return value."""
        try:
            value = self[key]
            del self._data[key]
            return value
        except KeyError:
            return default

    def clear(self) -> None:
        """Clear all entries."""
        self._data.clear()

    def _remove(self, key: K) -> None:
        """Remove an entry."""
        if key in self._data:
            entry = self._data[key]
            if self.on_evict:
                try:
                    self.on_evict(key, entry.value)
                except Exception:
                    pass
            del self._data[key]

    def _evict_one(self) -> None:
        """Evict one entry based on policy."""
        if not self._data:
            return

        # Clean expired first
        self._cleanup_expired()

        if len(self._data) < self.max_size:
            return

        # Find victim based on policy
        if self.policy == EvictionPolicy.LRU:
            # First item is least recently used
            key = next(iter(self._data))
        elif self.policy == EvictionPolicy.FIFO:
            key = next(iter(self._data))
        elif self.policy == EvictionPolicy.LFU:
            # Find least accessed
            key = min(self._data.keys(), key=lambda k: self._data[k].access_count)
        else:
            key = next(iter(self._data))

        self._remove(key)
        self._eviction_count += 1

    def _cleanup_expired(self) -> int:
        """Remove all expired entries."""
        expired = [k for k, v in self._data.items() if v.is_expired]
        for key in expired:
            self._remove(key)
        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        return {
            "size": len(self._data),
            "max_size": self.max_size,
            "policy": self.policy.value,
            "eviction_count": self._eviction_count,
            "default_ttl": self.default_ttl,
        }


class BoundedList(Generic[V]):
    """
    List with maximum size and FIFO eviction.

    Useful for bounded history/log keeping.
    """

    def __init__(
        self,
        max_size: int = 1000,
        on_evict: Optional[Callable[[V], None]] = None,
    ):
        self.max_size = max_size
        self.on_evict = on_evict
        self._data: List[V] = []

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[V]:
        return iter(self._data)

    def __getitem__(self, index: int) -> V:
        return self._data[index]

    def append(self, value: V) -> None:
        """Append value, evicting oldest if full."""
        while len(self._data) >= self.max_size:
            evicted = self._data.pop(0)
            if self.on_evict:
                try:
                    self.on_evict(evicted)
                except Exception:
                    pass

        self._data.append(value)

    def extend(self, values: List[V]) -> None:
        """Extend with multiple values."""
        for v in values:
            self.append(v)

    def clear(self) -> None:
        """Clear all entries."""
        self._data.clear()

    def get_recent(self, n: int = 10) -> List[V]:
        """Get n most recent entries."""
        return self._data[-n:]


# ============================================================================
# Handler Lifecycle Management (Gap #80)
# ============================================================================


class WeakHandlerSet:
    """
    Set of handlers using weak references.

    Automatically cleans up when handlers are garbage collected.
    Prevents memory leaks from orphaned handlers.

    Usage:
        handlers = WeakHandlerSet()

        def my_handler(event):
            ...

        handlers.add(my_handler)

        # When my_handler goes out of scope, it's automatically removed
    """

    def __init__(self):
        self._refs: Set[weakref.ref] = set()
        self._finalizer_map: Dict[int, weakref.finalize] = {}

    def add(self, handler: Callable) -> None:
        """Add a handler with weak reference."""

        def remove_ref(ref):
            self._refs.discard(ref)

        ref = weakref.ref(handler, remove_ref)
        self._refs.add(ref)

    def remove(self, handler: Callable) -> bool:
        """Remove a specific handler."""
        for ref in list(self._refs):
            if ref() is handler:
                self._refs.discard(ref)
                return True
        return False

    def clear(self) -> None:
        """Remove all handlers."""
        self._refs.clear()

    def __len__(self) -> int:
        # Clean dead refs first
        self._cleanup()
        return len(self._refs)

    def __iter__(self) -> Iterator[Callable]:
        """Iterate over live handlers."""
        self._cleanup()
        for ref in self._refs:
            handler = ref()
            if handler is not None:
                yield handler

    def _cleanup(self) -> None:
        """Remove dead references."""
        dead = [ref for ref in self._refs if ref() is None]
        for ref in dead:
            self._refs.discard(ref)

    async def call_all(self, *args, **kwargs) -> List[Any]:
        """Call all handlers and return results."""
        results = []
        for handler in self:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"[WeakHandlerSet] Handler error: {e}")
        return results


# ============================================================================
# Retention Policies
# ============================================================================


@dataclass
class RetentionPolicy:
    """
    Policy for data retention.

    Defines how long to keep data and when to archive/delete.
    """

    max_age_days: int = 30
    max_count: int = 10000
    archive_before_delete: bool = True
    archive_path: Optional[Path] = None


class RetentionManager:
    """
    Manages data retention for persistent storage.

    Features:
        - Time-based retention (delete after N days)
        - Count-based retention (keep only N most recent)
        - Archival before deletion
        - Scheduled cleanup

    Usage:
        manager = RetentionManager(
            policy=RetentionPolicy(max_age_days=30, max_count=10000)
        )

        # Run cleanup
        deleted = await manager.cleanup_events(event_store)
    """

    def __init__(self, policy: RetentionPolicy):
        self.policy = policy

    async def cleanup_events(
        self,
        db_path: Path,
        table_name: str = "events",
        timestamp_column: str = "timestamp",
    ) -> int:
        """
        Clean up old events from SQLite database.

        Args:
            db_path: Path to SQLite database
            table_name: Name of events table
            timestamp_column: Name of timestamp column

        Returns:
            Number of deleted rows
        """
        import sqlite3

        if not db_path.exists():
            return 0

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            # Archive if enabled
            if self.policy.archive_before_delete and self.policy.archive_path:
                await self._archive_old_events(
                    conn, table_name, timestamp_column
                )

            # Delete by age
            cutoff = time.time() - (self.policy.max_age_days * 86400)
            cursor.execute(
                f"DELETE FROM {table_name} WHERE {timestamp_column} < ?",
                (cutoff,),
            )
            deleted_by_age = cursor.rowcount

            # Delete by count (keep only max_count most recent)
            cursor.execute(
                f"""
                DELETE FROM {table_name}
                WHERE rowid NOT IN (
                    SELECT rowid FROM {table_name}
                    ORDER BY {timestamp_column} DESC
                    LIMIT ?
                )
                """,
                (self.policy.max_count,),
            )
            deleted_by_count = cursor.rowcount

            conn.commit()

            total_deleted = deleted_by_age + deleted_by_count
            if total_deleted:
                logger.info(
                    f"[RetentionManager] Deleted {total_deleted} old events "
                    f"(age: {deleted_by_age}, count: {deleted_by_count})"
                )

            return total_deleted

        finally:
            conn.close()

    async def _archive_old_events(
        self,
        conn,
        table_name: str,
        timestamp_column: str,
    ) -> None:
        """Archive events before deletion."""
        if not self.policy.archive_path:
            return

        self.policy.archive_path.mkdir(parents=True, exist_ok=True)

        cutoff = time.time() - (self.policy.max_age_days * 86400)
        cursor = conn.cursor()

        # Get old events
        cursor.execute(
            f"SELECT * FROM {table_name} WHERE {timestamp_column} < ?",
            (cutoff,),
        )
        rows = cursor.fetchall()

        if not rows:
            return

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Write to archive file
        archive_file = self.policy.archive_path / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        import json

        archive_data = [dict(zip(columns, row)) for row in rows]
        archive_file.write_text(json.dumps(archive_data, indent=2, default=str))

        logger.info(f"[RetentionManager] Archived {len(rows)} events to {archive_file}")

    async def cleanup_files(
        self,
        directory: Path,
        pattern: str = "*",
        keep_recent: int = 100,
    ) -> int:
        """
        Clean up old files from a directory.

        Args:
            directory: Directory to clean
            pattern: Glob pattern for files
            keep_recent: Number of recent files to keep

        Returns:
            Number of deleted files
        """
        if not directory.exists():
            return 0

        files = sorted(
            directory.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        deleted = 0
        for file in files[keep_recent:]:
            try:
                file.unlink()
                deleted += 1
            except OSError:
                pass

        if deleted:
            logger.info(f"[RetentionManager] Deleted {deleted} old files from {directory}")

        return deleted


# ============================================================================
# Memory Monitor
# ============================================================================


class MemoryMonitor:
    """
    Monitors memory usage and triggers cleanup when needed.

    Features:
        - Track memory usage over time
        - Warn when approaching limits
        - Trigger garbage collection
        - Force cleanup callbacks
    """

    def __init__(
        self,
        warning_threshold_mb: float = 500.0,
        critical_threshold_mb: float = 1000.0,
        check_interval: float = 60.0,
    ):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.check_interval = check_interval

        self._cleanup_callbacks: List[Callable[[], Coroutine]] = []
        self._history: BoundedList[Tuple[float, float]] = BoundedList(max_size=1000)
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def current_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        import resource

        # Get RSS (Resident Set Size)
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_maxrss / 1024  # Convert to MB (macOS returns bytes)

    def register_cleanup(self, callback: Callable[[], Coroutine]) -> None:
        """Register a cleanup callback for high memory situations."""
        self._cleanup_callbacks.append(callback)

    async def start(self) -> None:
        """Start memory monitoring."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("[MemoryMonitor] Started monitoring")

    async def stop(self) -> None:
        """Stop memory monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Periodic memory check loop."""
        while self._running:
            await self._check_memory()
            await asyncio.sleep(self.check_interval)

    async def _check_memory(self) -> None:
        """Check memory and take action if needed."""
        usage = self.current_usage_mb
        self._history.append((time.time(), usage))

        if usage >= self.critical_threshold_mb:
            logger.warning(
                f"[MemoryMonitor] CRITICAL: {usage:.1f}MB (threshold: {self.critical_threshold_mb}MB)"
            )
            await self._trigger_cleanup()
            gc.collect()

        elif usage >= self.warning_threshold_mb:
            logger.warning(
                f"[MemoryMonitor] WARNING: {usage:.1f}MB (threshold: {self.warning_threshold_mb}MB)"
            )
            gc.collect()

    async def _trigger_cleanup(self) -> None:
        """Trigger all cleanup callbacks."""
        for callback in self._cleanup_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"[MemoryMonitor] Cleanup callback error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        history = list(self._history)
        recent_usage = [u for _, u in history[-10:]]

        return {
            "current_mb": self.current_usage_mb,
            "warning_threshold_mb": self.warning_threshold_mb,
            "critical_threshold_mb": self.critical_threshold_mb,
            "avg_recent_mb": sum(recent_usage) / len(recent_usage) if recent_usage else 0,
            "max_recent_mb": max(recent_usage) if recent_usage else 0,
            "sample_count": len(history),
        }


# ============================================================================
# Background Cleanup Service
# ============================================================================


class CleanupService:
    """
    Background service for periodic cleanup tasks.

    Coordinates memory management, retention, and cleanup.

    Usage:
        service = CleanupService()
        service.add_task("events", cleanup_events, interval=3600)
        service.add_task("cache", cleanup_cache, interval=300)
        await service.start()
    """

    def __init__(self):
        self._tasks: Dict[str, Tuple[Callable[[], Coroutine], float]] = {}
        self._running = False
        self._task_handles: Dict[str, asyncio.Task] = {}

    def add_task(
        self,
        name: str,
        task: Callable[[], Coroutine],
        interval_seconds: float,
    ) -> None:
        """
        Add a cleanup task.

        Args:
            name: Task name
            task: Async cleanup function
            interval_seconds: How often to run
        """
        self._tasks[name] = (task, interval_seconds)

    async def start(self) -> None:
        """Start all cleanup tasks."""
        if self._running:
            return

        self._running = True

        for name, (task, interval) in self._tasks.items():
            self._task_handles[name] = asyncio.create_task(
                self._run_periodic(name, task, interval)
            )

        logger.info(f"[CleanupService] Started {len(self._tasks)} cleanup tasks")

    async def stop(self) -> None:
        """Stop all cleanup tasks."""
        self._running = False

        for task in self._task_handles.values():
            task.cancel()

        await asyncio.gather(*self._task_handles.values(), return_exceptions=True)
        self._task_handles.clear()

    async def _run_periodic(
        self,
        name: str,
        task: Callable[[], Coroutine],
        interval: float,
    ) -> None:
        """Run a task periodically."""
        while self._running:
            try:
                await task()
            except Exception as e:
                logger.error(f"[CleanupService:{name}] Task error: {e}")

            await asyncio.sleep(interval)

    async def run_all_now(self) -> Dict[str, bool]:
        """Run all cleanup tasks immediately."""
        results = {}
        for name, (task, _) in self._tasks.items():
            try:
                await task()
                results[name] = True
            except Exception as e:
                logger.error(f"[CleanupService:{name}] Task error: {e}")
                results[name] = False
        return results


# ============================================================================
# Global instances
# ============================================================================


_memory_monitor: Optional[MemoryMonitor] = None
_cleanup_service: Optional[CleanupService] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get or create global memory monitor."""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor


def get_cleanup_service() -> CleanupService:
    """Get or create global cleanup service."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = CleanupService()
    return _cleanup_service
