"""
Robust File Watch Guard with Event Deduplication
================================================

Production-grade file watching for cross-repo file-based RPC.

Features:
    - Event deduplication with LRU cache
    - Graceful recovery from watchdog errors
    - Configurable event batching and debouncing
    - Directory creation handling (watches new subdirs)
    - Checksum-based change detection (avoid false positives)
    - Comprehensive metrics and health status

Author: JARVIS Cross-Repo Resilience
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FileEventType(Enum):
    """Type of file event."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileEvent:
    """Represents a file system event."""

    event_type: FileEventType
    path: Path
    timestamp: float = field(default_factory=time.time)
    checksum: Optional[str] = None  # For content change detection
    old_path: Optional[Path] = None  # For MOVED events
    size: Optional[int] = None
    is_directory: bool = False

    @property
    def event_id(self) -> str:
        """Generate unique ID for deduplication."""
        return f"{self.event_type.value}:{self.path}:{self.checksum or self.timestamp}"


@dataclass
class FileWatchConfig:
    """Configuration for file watch guard."""

    # Basic settings
    recursive: bool = True
    patterns: List[str] = field(default_factory=lambda: ["*"])  # Glob patterns
    ignore_patterns: List[str] = field(default_factory=lambda: ["*.tmp", "*.swp", "*.bak", "*~"])

    # Debouncing
    debounce_seconds: float = 0.1  # Wait before firing event
    batch_timeout_seconds: float = 0.5  # Max wait for batch

    # Deduplication
    dedup_cache_size: int = 1000  # LRU cache size
    dedup_ttl_seconds: float = 5.0  # Events within TTL are deduplicated

    # Content verification
    verify_checksum: bool = True  # Use checksum to detect real changes
    min_stable_seconds: float = 0.05  # File must be stable for this long

    # Recovery
    restart_on_error: bool = True
    error_backoff_seconds: float = 1.0
    max_consecutive_errors: int = 5

    # Health
    health_check_interval: float = 30.0


@dataclass
class WatchMetrics:
    """Metrics for file watching."""

    events_received: int = 0
    events_processed: int = 0
    events_deduplicated: int = 0
    events_filtered: int = 0
    errors: int = 0
    restarts: int = 0
    last_event_time: Optional[float] = None
    avg_processing_time_ms: float = 0.0


class FileWatchGuard:
    """
    Robust file watcher with event deduplication and recovery.

    Wraps watchdog with additional safety measures for production use.

    Usage:
        config = FileWatchConfig(patterns=["*.json"])
        guard = FileWatchGuard(
            watch_dir=Path("~/.jarvis/events"),
            config=config,
            on_event=handle_event,
        )

        await guard.start()
        # ... events flow to handler ...
        await guard.stop()
    """

    def __init__(
        self,
        watch_dir: Path,
        on_event: Callable[[FileEvent], Any],
        config: Optional[FileWatchConfig] = None,
        on_error: Optional[Callable[[Exception], Any]] = None,
    ):
        self.watch_dir = Path(watch_dir).expanduser().resolve()
        self._on_event = on_event
        self._on_error = on_error
        self.config = config or FileWatchConfig()

        self._observer = None
        self._running = False
        self._event_queue: asyncio.Queue[FileEvent] = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

        # Deduplication
        self._seen_events: OrderedDict[str, float] = OrderedDict()  # LRU cache
        self._pending_events: Dict[str, FileEvent] = {}  # Debounce buffer

        # File content cache for checksum
        self._checksums: Dict[str, str] = {}

        # Error tracking
        self._consecutive_errors = 0
        self._last_error: Optional[Exception] = None

        self.metrics = WatchMetrics()

    async def start(self) -> bool:
        """
        Start file watching.

        Returns:
            True if started successfully
        """
        if self._running:
            return True

        # Ensure directory exists
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        try:
            await self._start_watchdog()
            self._running = True

            # Start event processor
            self._processor_task = asyncio.create_task(self._process_events())

            # Start health check
            self._health_task = asyncio.create_task(self._health_check_loop())

            logger.info(f"[FileWatchGuard] Started watching {self.watch_dir}")
            return True

        except Exception as e:
            logger.error(f"[FileWatchGuard] Failed to start: {e}")
            self._last_error = e
            self.metrics.errors += 1
            return False

    async def stop(self) -> None:
        """Stop file watching."""
        self._running = False

        # Stop tasks
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop watchdog
        await self._stop_watchdog()

        logger.info("[FileWatchGuard] Stopped")

    async def _start_watchdog(self) -> None:
        """Start the watchdog observer."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileSystemEvent
        except ImportError:
            raise RuntimeError("watchdog package required: pip install watchdog")

        # Create handler that bridges to async
        guard = self

        class AsyncEventHandler(FileSystemEventHandler):
            def on_any_event(self, event: FileSystemEvent):
                if event.is_directory and event.event_type != "created":
                    return

                try:
                    # Convert to our event type
                    event_type = {
                        "created": FileEventType.CREATED,
                        "modified": FileEventType.MODIFIED,
                        "deleted": FileEventType.DELETED,
                        "moved": FileEventType.MOVED,
                    }.get(event.event_type)

                    if not event_type:
                        return

                    file_event = FileEvent(
                        event_type=event_type,
                        path=Path(event.src_path),
                        is_directory=event.is_directory,
                        old_path=Path(event.dest_path) if hasattr(event, "dest_path") else None,
                    )

                    # Queue for async processing
                    guard._queue_event(file_event)

                except Exception as e:
                    logger.error(f"[FileWatchGuard] Event handler error: {e}")

        self._observer = Observer()
        handler = AsyncEventHandler()

        self._observer.schedule(
            handler,
            str(self.watch_dir),
            recursive=self.config.recursive,
        )
        self._observer.start()

    async def _stop_watchdog(self) -> None:
        """Stop the watchdog observer."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

    def _queue_event(self, event: FileEvent) -> None:
        """Queue an event for processing (called from watchdog thread)."""
        try:
            # Use asyncio-safe method to put in queue
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(
                    lambda: self._event_queue.put_nowait(event)
                )
            else:
                self._event_queue.put_nowait(event)
        except Exception as e:
            logger.error(f"[FileWatchGuard] Queue error: {e}")

    async def _process_events(self) -> None:
        """Process events from queue with debouncing and deduplication."""
        batch_deadline = 0.0
        batch: List[FileEvent] = []

        while self._running:
            try:
                # Get event with timeout
                timeout = self.config.batch_timeout_seconds
                if batch:
                    timeout = max(0, batch_deadline - time.time())

                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=timeout,
                    )
                    self.metrics.events_received += 1

                    # Apply filters
                    if not self._should_process(event):
                        self.metrics.events_filtered += 1
                        continue

                    # Check deduplication
                    if self._is_duplicate(event):
                        self.metrics.events_deduplicated += 1
                        continue

                    # Add to batch
                    batch.append(event)
                    if not batch_deadline:
                        batch_deadline = time.time() + self.config.debounce_seconds

                except asyncio.TimeoutError:
                    pass

                # Process batch if ready
                if batch and (
                    time.time() >= batch_deadline
                    or len(batch) >= 10  # Max batch size
                ):
                    await self._process_batch(batch)
                    batch = []
                    batch_deadline = 0.0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[FileWatchGuard] Processing error: {e}")
                self.metrics.errors += 1
                self._consecutive_errors += 1
                self._last_error = e

                if self._on_error:
                    try:
                        result = self._on_error(e)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        pass

                if self._consecutive_errors >= self.config.max_consecutive_errors:
                    await self._handle_error_overflow()

                await asyncio.sleep(self.config.error_backoff_seconds)

    def _should_process(self, event: FileEvent) -> bool:
        """Check if event should be processed based on patterns."""
        path = event.path
        name = path.name

        # Check ignore patterns
        import fnmatch

        for pattern in self.config.ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return False

        # Check include patterns
        if self.config.patterns:
            matched = False
            for pattern in self.config.patterns:
                if fnmatch.fnmatch(name, pattern):
                    matched = True
                    break
            if not matched:
                return False

        return True

    def _is_duplicate(self, event: FileEvent) -> bool:
        """Check if event is a duplicate."""
        event_id = event.event_id
        now = time.time()

        # Check if we've seen this event recently
        if event_id in self._seen_events:
            seen_time = self._seen_events[event_id]
            if now - seen_time < self.config.dedup_ttl_seconds:
                return True

        # Update LRU cache
        self._seen_events[event_id] = now

        # Maintain cache size
        while len(self._seen_events) > self.config.dedup_cache_size:
            self._seen_events.popitem(last=False)  # Remove oldest

        return False

    async def _process_batch(self, events: List[FileEvent]) -> None:
        """Process a batch of events."""
        # Consolidate events for same file
        by_path: Dict[str, FileEvent] = {}
        for event in events:
            path_key = str(event.path)

            # Later events override earlier ones
            if event.event_type == FileEventType.DELETED:
                # Delete supersedes all
                by_path[path_key] = event
            elif event.event_type == FileEventType.CREATED:
                # Create only if not already have newer event
                if path_key not in by_path:
                    by_path[path_key] = event
            else:
                # Modified replaces create
                existing = by_path.get(path_key)
                if not existing or existing.event_type != FileEventType.DELETED:
                    by_path[path_key] = event

        # Process each unique event
        for event in by_path.values():
            await self._process_single_event(event)

    async def _process_single_event(self, event: FileEvent) -> None:
        """Process a single event."""
        start_time = time.time()

        try:
            # For modifications, verify file is stable and content changed
            if event.event_type == FileEventType.MODIFIED:
                if self.config.verify_checksum:
                    if not await self._verify_content_changed(event):
                        return

            # Wait for file to be stable
            if event.event_type in (FileEventType.CREATED, FileEventType.MODIFIED):
                if not event.is_directory:
                    await self._wait_for_stable(event.path)

            # Add file info
            if event.path.exists() and not event.is_directory:
                event.size = event.path.stat().st_size

            # Call handler
            result = self._on_event(event)
            if asyncio.iscoroutine(result):
                await result

            self.metrics.events_processed += 1
            self.metrics.last_event_time = time.time()
            self._consecutive_errors = 0

            # Update processing time metric
            processing_ms = (time.time() - start_time) * 1000
            total = (
                self.metrics.avg_processing_time_ms * (self.metrics.events_processed - 1)
                + processing_ms
            )
            self.metrics.avg_processing_time_ms = total / self.metrics.events_processed

        except Exception as e:
            logger.error(f"[FileWatchGuard] Event handler error for {event.path}: {e}")
            self.metrics.errors += 1
            raise

    async def _verify_content_changed(self, event: FileEvent) -> bool:
        """Verify file content actually changed (avoid false positives)."""
        path = event.path
        path_key = str(path)

        if not path.exists():
            return True

        try:
            content = await asyncio.to_thread(path.read_bytes)
            new_checksum = hashlib.md5(content).hexdigest()

            old_checksum = self._checksums.get(path_key)
            self._checksums[path_key] = new_checksum

            if old_checksum and old_checksum == new_checksum:
                # Content didn't change
                return False

            event.checksum = new_checksum
            return True

        except Exception:
            return True  # Assume changed on error

    async def _wait_for_stable(self, path: Path) -> None:
        """Wait for file to stop being written."""
        if not path.exists():
            return

        if self.config.min_stable_seconds <= 0:
            return

        last_size = -1
        stable_start = 0.0

        while True:
            try:
                current_size = path.stat().st_size
                now = time.time()

                if current_size != last_size:
                    last_size = current_size
                    stable_start = now
                elif now - stable_start >= self.config.min_stable_seconds:
                    # File is stable
                    return

            except FileNotFoundError:
                # File was deleted
                return

            await asyncio.sleep(0.01)

            # Timeout after 5 seconds
            if not stable_start or time.time() - stable_start > 5.0:
                return

    async def _handle_error_overflow(self) -> None:
        """Handle too many consecutive errors."""
        logger.warning(
            f"[FileWatchGuard] {self._consecutive_errors} consecutive errors, "
            f"restarting watcher"
        )

        if self.config.restart_on_error:
            self.metrics.restarts += 1
            await self._stop_watchdog()
            await asyncio.sleep(self.config.error_backoff_seconds)

            try:
                await self._start_watchdog()
                self._consecutive_errors = 0
            except Exception as e:
                logger.error(f"[FileWatchGuard] Restart failed: {e}")

    async def _health_check_loop(self) -> None:
        """Background health check."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Check watchdog is alive
                if self._observer and not self._observer.is_alive():
                    logger.warning("[FileWatchGuard] Observer died, restarting")
                    self.metrics.restarts += 1
                    await self._stop_watchdog()
                    await self._start_watchdog()

                # Check directory exists
                if not self.watch_dir.exists():
                    logger.warning(
                        f"[FileWatchGuard] Watch directory disappeared: {self.watch_dir}"
                    )
                    self.watch_dir.mkdir(parents=True, exist_ok=True)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[FileWatchGuard] Health check error: {e}")

    @property
    def is_healthy(self) -> bool:
        """Check if watcher is healthy."""
        if not self._running:
            return False

        if self._observer and not self._observer.is_alive():
            return False

        if self._consecutive_errors >= self.config.max_consecutive_errors:
            return False

        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get watcher metrics."""
        return {
            "watch_dir": str(self.watch_dir),
            "is_running": self._running,
            "is_healthy": self.is_healthy,
            "events_received": self.metrics.events_received,
            "events_processed": self.metrics.events_processed,
            "events_deduplicated": self.metrics.events_deduplicated,
            "events_filtered": self.metrics.events_filtered,
            "errors": self.metrics.errors,
            "restarts": self.metrics.restarts,
            "consecutive_errors": self._consecutive_errors,
            "last_event_time": self.metrics.last_event_time,
            "avg_processing_time_ms": round(self.metrics.avg_processing_time_ms, 2),
            "dedup_cache_size": len(self._seen_events),
            "queue_size": self._event_queue.qsize(),
            "last_error": str(self._last_error) if self._last_error else None,
        }

    async def trigger_scan(self) -> int:
        """
        Manually scan directory and emit events for existing files.

        Useful for catching up after restart.

        Returns:
            Number of events emitted
        """
        count = 0

        for pattern in self.config.patterns:
            if self.config.recursive:
                files = self.watch_dir.rglob(pattern)
            else:
                files = self.watch_dir.glob(pattern)

            for path in files:
                if path.is_file():
                    event = FileEvent(
                        event_type=FileEventType.CREATED,
                        path=path,
                        is_directory=False,
                    )
                    if self._should_process(event):
                        self._queue_event(event)
                        count += 1

        logger.info(f"[FileWatchGuard] Triggered scan, found {count} files")
        return count
