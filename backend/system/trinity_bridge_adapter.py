"""
Trinity Bridge Adapter v1.0
===========================

Closes the Trinity Loop by watching Reactor Core event directories
and forwarding MODEL_READY and other events to JARVIS.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Reactor Core                                 │
    │  [Training Pipeline] → [MODEL_READY event] → [File Write]       │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                 TrinityBridgeAdapter (This Module)              │
    │  [File Watcher] → [Event Parser] → [TrinityEventBus]            │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       JARVIS / JARVIS Prime                     │
    │  [TrinityHandlers] → [UnifiedModelServing] → [Hot Swap]         │
    └─────────────────────────────────────────────────────────────────┘

This adapter:
1. Watches ~/.jarvis/reactor/events/ for Reactor Core events
2. Watches ~/.jarvis/trinity/events/ for cross-repo events
3. Parses event files (JSON format)
4. Forwards to TrinityEventBus for handler dispatch
5. Cleans up processed files

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

logger = logging.getLogger("TrinityBridgeAdapter")

# =============================================================================
# Configuration (Environment-Driven)
# =============================================================================

REACTOR_EVENTS_DIR = Path(os.getenv(
    "REACTOR_EVENTS_DIR",
    str(Path.home() / ".jarvis" / "reactor" / "events")
))

TRINITY_EVENTS_DIR = Path(os.getenv(
    "TRINITY_EVENTS_DIR",
    str(Path.home() / ".jarvis" / "trinity" / "events")
))

CROSS_REPO_EVENTS_DIR = Path(os.getenv(
    "CROSS_REPO_EVENTS_DIR",
    str(Path.home() / ".jarvis" / "cross_repo" / "events")
))

# Polling interval for async file check (backup if watchdog misses events)
POLL_INTERVAL = float(os.getenv("TRINITY_BRIDGE_POLL_INTERVAL", "5.0"))

# Event types we care about
CRITICAL_EVENT_TYPES = {
    "model_ready",
    "model_validated",
    "model_failed",
    "model_rollback",
    "training_complete",
    "hot_swap_model",
    "update_model_routing",
}


class BridgeEventType(Enum):
    """Types of events the bridge can handle."""
    MODEL_READY = "model_ready"
    MODEL_VALIDATED = "model_validated"
    MODEL_FAILED = "model_failed"
    MODEL_ROLLBACK = "model_rollback"
    TRAINING_COMPLETE = "training_complete"
    HOT_SWAP_MODEL = "hot_swap_model"
    UPDATE_ROUTING = "update_model_routing"
    EXPERIENCE_BATCH = "experience_batch"
    HEARTBEAT = "heartbeat"
    UNKNOWN = "unknown"


@dataclass
class BridgeMetrics:
    """Metrics for the bridge adapter."""
    events_received: int = 0
    events_forwarded: int = 0
    events_failed: int = 0
    files_processed: int = 0
    last_event_time: Optional[float] = None
    events_by_type: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# Async File Watcher using watchdog + asyncio
# =============================================================================

class AsyncEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that queues events for async processing.

    Uses an asyncio Queue to bridge between watchdog's threading model
    and our async processing.
    """

    def __init__(self, event_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._queue = event_queue
        self._loop = loop
        self._seen_files: Set[str] = set()

    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Only process JSON files
        if file_path.suffix != ".json":
            return

        # Skip temp files
        if file_path.name.startswith(".") or file_path.name.startswith("_"):
            return

        # Deduplicate (watchdog can fire multiple events for same file)
        file_key = str(file_path)
        if file_key in self._seen_files:
            return
        self._seen_files.add(file_key)

        # Limit seen files cache
        if len(self._seen_files) > 1000:
            self._seen_files = set(list(self._seen_files)[-500:])

        # Queue for async processing
        try:
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                file_path,
            )
        except Exception as e:
            logger.warning(f"Failed to queue event: {e}")


# =============================================================================
# Trinity Bridge Adapter
# =============================================================================

class TrinityBridgeAdapter:
    """
    Watches Reactor Core and Trinity event directories and forwards
    events to TrinityEventBus for processing.

    This is the CRITICAL component that closes the Trinity Loop:
    - Reactor Core trains a model and writes MODEL_READY event
    - This adapter picks up the event and forwards to TrinityEventBus
    - TrinityHandlers receive the event and hot-swap the model

    Features:
    - Async file watching via watchdog + asyncio bridge
    - Fallback polling for reliability
    - Deduplication to prevent double-processing
    - Graceful error handling with retry
    - Event metrics and observability
    """

    def __init__(
        self,
        event_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ):
        self.logger = logging.getLogger("TrinityBridgeAdapter")

        # Event callback (defaults to TrinityEventBus dispatch)
        self._event_callback = event_callback

        # Watch directories
        self._watch_dirs: List[Path] = [
            REACTOR_EVENTS_DIR,
            TRINITY_EVENTS_DIR,
            CROSS_REPO_EVENTS_DIR,
        ]

        # Async queue for events from watchdog
        self._event_queue: asyncio.Queue[Path] = asyncio.Queue(maxsize=1000)

        # Watchdog observers (one per directory)
        self._observers: List[Observer] = []

        # Deduplication
        self._processed_events: Set[str] = set()
        self._processed_queue: List[str] = []  # For LRU eviction
        self._max_processed = 10000

        # Metrics
        self._metrics = BridgeMetrics()

        # State
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None

        # Ensure directories exist
        for dir_path in self._watch_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.debug(f"Could not create {dir_path}: {e}")

    async def start(self) -> bool:
        """Start the bridge adapter."""
        if self._running:
            return True

        self._running = True
        self.logger.info("TrinityBridgeAdapter starting...")

        # Get current event loop
        loop = asyncio.get_event_loop()

        # Start watchdog observers for each directory
        for dir_path in self._watch_dirs:
            if dir_path.exists():
                try:
                    handler = AsyncEventHandler(self._event_queue, loop)
                    observer = Observer()
                    observer.schedule(handler, str(dir_path), recursive=False)
                    observer.start()
                    self._observers.append(observer)
                    self.logger.info(f"Watching directory: {dir_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to watch {dir_path}: {e}")

        # Start async processing task
        self._process_task = asyncio.create_task(
            self._process_loop(),
            name="trinity_bridge_process_loop",
        )

        # Start fallback polling task
        self._poll_task = asyncio.create_task(
            self._poll_loop(),
            name="trinity_bridge_poll_loop",
        )

        # Process any existing files on startup
        await self._process_existing_files()

        self.logger.info(
            f"TrinityBridgeAdapter ready "
            f"(watching {len(self._observers)} directories)"
        )
        return True

    async def stop(self) -> None:
        """Stop the bridge adapter."""
        self._running = False

        # Stop watchdog observers
        for observer in self._observers:
            try:
                observer.stop()
                observer.join(timeout=2.0)
            except Exception as e:
                self.logger.debug(f"Observer stop error: {e}")
        self._observers.clear()

        # Cancel async tasks
        for task in [self._process_task, self._poll_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self.logger.info(
            f"TrinityBridgeAdapter stopped "
            f"(processed {self._metrics.events_forwarded} events)"
        )

    async def _process_existing_files(self) -> None:
        """Process any existing event files on startup."""
        for dir_path in self._watch_dirs:
            if not dir_path.exists():
                continue

            try:
                for file_path in dir_path.glob("*.json"):
                    if file_path.name.startswith("."):
                        continue
                    await self._process_file(file_path)
            except Exception as e:
                self.logger.warning(f"Error processing existing files in {dir_path}: {e}")

    async def _process_loop(self) -> None:
        """Main async loop to process queued file events."""
        while self._running:
            try:
                # Wait for file event from watchdog
                file_path = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0,
                )

                # Process the file
                await self._process_file(file_path)

            except asyncio.TimeoutError:
                # Normal timeout, continue loop
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Process loop error: {e}")
                await asyncio.sleep(0.5)

    async def _poll_loop(self) -> None:
        """Fallback polling loop to catch missed events."""
        while self._running:
            try:
                await asyncio.sleep(POLL_INTERVAL)

                if not self._running:
                    break

                # Scan all watch directories
                for dir_path in self._watch_dirs:
                    if dir_path.exists():
                        await self._scan_directory(dir_path)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Poll loop error: {e}")
                await asyncio.sleep(1.0)

    async def _scan_directory(self, dir_path: Path) -> None:
        """Scan a directory for unprocessed event files."""
        try:
            for file_path in dir_path.glob("*.json"):
                # Skip hidden/temp files
                if file_path.name.startswith("."):
                    continue

                # Skip already processed
                event_id = self._get_event_id(file_path)
                if event_id in self._processed_events:
                    continue

                await self._process_file(file_path)

        except Exception as e:
            self.logger.warning(f"Directory scan error for {dir_path}: {e}")

    def _get_event_id(self, file_path: Path) -> str:
        """Generate unique ID for deduplication."""
        try:
            stat = file_path.stat()
            return f"{file_path.name}_{stat.st_mtime}_{stat.st_size}"
        except Exception:
            return str(file_path)

    async def _process_file(self, file_path: Path) -> None:
        """Process a single event file."""
        # Deduplication check
        event_id = self._get_event_id(file_path)
        if event_id in self._processed_events:
            return

        try:
            # Read file content
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, file_path.read_text)
            event = json.loads(content)

            # Mark as processed
            self._mark_processed(event_id)
            self._metrics.files_processed += 1

            # Get event type
            event_type = event.get("event_type", event.get("type", "unknown"))

            # Update metrics
            self._metrics.events_received += 1
            self._metrics.last_event_time = time.time()
            self._metrics.events_by_type[event_type] = \
                self._metrics.events_by_type.get(event_type, 0) + 1

            # Check if this is a critical event we need to forward
            if event_type.lower() in CRITICAL_EVENT_TYPES:
                self.logger.info(
                    f"[TrinityBridge] Critical event received: {event_type} "
                    f"(source: {file_path.name})"
                )

                # Forward to TrinityEventBus
                await self._forward_event(event)

            # Delete processed file
            try:
                await loop.run_in_executor(None, file_path.unlink)
                self.logger.debug(f"Deleted processed file: {file_path.name}")
            except FileNotFoundError:
                pass  # Already deleted
            except Exception as e:
                self.logger.warning(f"Could not delete {file_path.name}: {e}")

        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON in {file_path.name}")
            # Move to .failed
            try:
                file_path.rename(file_path.with_suffix(".json.failed"))
            except Exception:
                pass
        except FileNotFoundError:
            # File already processed by another instance
            pass
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            self._metrics.events_failed += 1

    def _mark_processed(self, event_id: str) -> None:
        """Mark an event as processed with LRU eviction."""
        if event_id in self._processed_events:
            return

        self._processed_events.add(event_id)
        self._processed_queue.append(event_id)

        # LRU eviction
        while len(self._processed_queue) > self._max_processed:
            old_id = self._processed_queue.pop(0)
            self._processed_events.discard(old_id)

    async def _forward_event(self, event: Dict[str, Any]) -> None:
        """Forward event to TrinityEventBus or callback."""
        try:
            if self._event_callback:
                # Use provided callback
                result = self._event_callback(event)
                if asyncio.iscoroutine(result):
                    await result
            else:
                # Try to get TrinityEventBus and dispatch
                await self._dispatch_to_event_bus(event)

            self._metrics.events_forwarded += 1
            self.logger.info(
                f"[TrinityBridge] Forwarded event: {event.get('event_type', 'unknown')}"
            )

        except Exception as e:
            self.logger.error(f"Event forwarding failed: {e}")
            self._metrics.events_failed += 1

    async def _dispatch_to_event_bus(self, event: Dict[str, Any]) -> None:
        """Dispatch event to TrinityEventBus."""
        try:
            # Import TrinityEventBus
            try:
                from backend.core.trinity_event_bus import get_event_bus
            except ImportError:
                from core.trinity_event_bus import get_event_bus

            # Get event bus instance
            event_bus = await get_event_bus()

            if event_bus:
                # Use publish_raw for dict events
                await event_bus.publish_raw(
                    topic=f"reactor.{event.get('event_type', 'event')}",
                    data=event,
                )
            else:
                self.logger.warning("TrinityEventBus not available")

        except ImportError:
            self.logger.warning("TrinityEventBus module not found")
        except Exception as e:
            self.logger.error(f"EventBus dispatch failed: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get bridge metrics."""
        return {
            "events_received": self._metrics.events_received,
            "events_forwarded": self._metrics.events_forwarded,
            "events_failed": self._metrics.events_failed,
            "files_processed": self._metrics.files_processed,
            "last_event_time": self._metrics.last_event_time,
            "events_by_type": self._metrics.events_by_type,
            "running": self._running,
            "watch_dirs": [str(d) for d in self._watch_dirs],
            "observers_active": len(self._observers),
        }


# =============================================================================
# Global Instance Management
# =============================================================================

_bridge_adapter: Optional[TrinityBridgeAdapter] = None
_bridge_lock: Optional[asyncio.Lock] = None


def _get_bridge_lock() -> asyncio.Lock:
    """Get or create the bridge lock."""
    global _bridge_lock
    if _bridge_lock is None:
        _bridge_lock = asyncio.Lock()
    return _bridge_lock


async def get_trinity_bridge() -> TrinityBridgeAdapter:
    """Get the global TrinityBridgeAdapter instance."""
    global _bridge_adapter

    lock = _get_bridge_lock()
    async with lock:
        if _bridge_adapter is None:
            _bridge_adapter = TrinityBridgeAdapter()
            await _bridge_adapter.start()

        return _bridge_adapter


async def shutdown_trinity_bridge() -> None:
    """Shutdown the global TrinityBridgeAdapter."""
    global _bridge_adapter

    if _bridge_adapter:
        await _bridge_adapter.stop()
        _bridge_adapter = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TrinityBridgeAdapter",
    "BridgeEventType",
    "BridgeMetrics",
    "get_trinity_bridge",
    "shutdown_trinity_bridge",
]
