"""
Bounded Async Queue with configurable overflow policies.

Phase 5A of Enterprise Hardening Plan.

Provides backpressure control for asyncio.Queue instances that were previously
unbounded, preventing producers from overwhelming consumers.

Usage:
    from backend.core.bounded_queue import BoundedAsyncQueue, OverflowPolicy

    queue = BoundedAsyncQueue(
        maxsize=1000,
        policy=OverflowPolicy.DROP_OLDEST,
        name="metrics_queue",
    )

    # Or use the factory:
    from backend.core.bounded_queue import create_bounded_queue
    queue = create_bounded_queue(1000, OverflowPolicy.DROP_OLDEST, "metrics")
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import time
import weakref
from typing import Any, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------

_DEFAULT_QUEUE_SIZE = int(os.environ.get("Ironcliw_DEFAULT_QUEUE_SIZE", "1000"))
_LOG_THRESHOLD_PCT = float(os.environ.get("Ironcliw_QUEUE_WARN_PCT", "0.8"))

# ---------------------------------------------------------------------------
# Overflow policy enum
# ---------------------------------------------------------------------------


class OverflowPolicy(enum.Enum):
    """Policy to apply when a bounded queue is full.

    BLOCK         - Default asyncio behaviour; ``put()`` awaits until space is
                    available.
    DROP_OLDEST   - Discard the oldest item in the queue, then enqueue the new
                    item.  Suitable for metrics / logging where stale data is
                    acceptable to lose.
    DROP_NEWEST   - Reject the new item silently (or with a log).  Suitable for
                    commands where a caller should retry on their own.
    WARN_AND_BLOCK - Log a warning at the configured threshold, then block.
                     Useful during development to surface capacity issues.
    """

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    WARN_AND_BLOCK = "warn_and_block"


# ---------------------------------------------------------------------------
# Global registry of all live BoundedAsyncQueue instances (weak refs)
# ---------------------------------------------------------------------------

_queue_registry: Dict[str, weakref.ref] = {}
_registry_counter: int = 0


def get_all_queue_stats() -> Dict[str, Dict[str, Any]]:
    """Return statistics for every live BoundedAsyncQueue.

    Returns:
        Mapping from queue name to its ``get_stats()`` output.  Dead
        references are automatically pruned.
    """
    dead_keys: list[str] = []
    result: Dict[str, Dict[str, Any]] = {}
    for key, ref in _queue_registry.items():
        obj = ref()
        if obj is None:
            dead_keys.append(key)
        else:
            result[key] = obj.get_stats()
    for key in dead_keys:
        _queue_registry.pop(key, None)
    return result


def _register_queue(queue: "BoundedAsyncQueue") -> str:  # type: ignore[type-arg]
    """Register a queue instance in the global registry."""
    global _registry_counter
    _registry_counter += 1
    key = queue.name or f"anon_queue_{_registry_counter}"
    # Deduplicate by appending counter if name already taken
    if key in _queue_registry:
        key = f"{key}_{_registry_counter}"
    _queue_registry[key] = weakref.ref(queue)
    return key


# ---------------------------------------------------------------------------
# BoundedAsyncQueue
# ---------------------------------------------------------------------------


class BoundedAsyncQueue(asyncio.Queue, Generic[T]):
    """Async queue with a hard upper bound and configurable overflow policy.

    Parameters
    ----------
    maxsize:
        Maximum number of items the queue can hold.  Defaults to the
        ``Ironcliw_DEFAULT_QUEUE_SIZE`` env-var (1000).
    policy:
        What to do when the queue is full.  See :class:`OverflowPolicy`.
    name:
        Human-readable label used in log messages and the global registry.
    """

    def __init__(
        self,
        maxsize: int = 0,
        policy: OverflowPolicy = OverflowPolicy.BLOCK,
        name: str = "",
    ) -> None:
        effective_maxsize = maxsize if maxsize > 0 else _DEFAULT_QUEUE_SIZE
        super().__init__(maxsize=effective_maxsize)

        self.policy = policy
        self.name = name

        # Statistics (accessed from a single event-loop so no lock needed)
        self._dropped_count: int = 0
        self._total_puts: int = 0
        self._total_gets: int = 0
        self._peak_size: int = 0
        self._created_at: float = time.monotonic()
        self._last_put_at: float = 0.0
        self._last_get_at: float = 0.0
        self._warn_logged: bool = False

        # Auto-register for monitoring
        self._registry_key = _register_queue(self)

        logger.debug(
            "BoundedAsyncQueue created: name=%s maxsize=%d policy=%s",
            self.name,
            self.maxsize,
            self.policy.value,
        )

    # -- Statistics -----------------------------------------------------------

    @property
    def dropped_count(self) -> int:
        """Number of items dropped due to overflow policy."""
        return self._dropped_count

    @property
    def peak_size(self) -> int:
        """Highest number of items observed in the queue at one time."""
        return self._peak_size

    @property
    def total_puts(self) -> int:
        """Total items successfully enqueued."""
        return self._total_puts

    @property
    def total_gets(self) -> int:
        """Total items successfully dequeued."""
        return self._total_gets

    def get_stats(self) -> Dict[str, Any]:
        """Return a snapshot of queue health metrics."""
        current_size = self.qsize()
        uptime = time.monotonic() - self._created_at
        return {
            "name": self.name,
            "maxsize": self.maxsize,
            "policy": self.policy.value,
            "current_size": current_size,
            "peak_size": self._peak_size,
            "total_puts": self._total_puts,
            "total_gets": self._total_gets,
            "dropped_count": self._dropped_count,
            "utilization_pct": round(
                (current_size / self.maxsize * 100) if self.maxsize else 0.0, 1
            ),
            "uptime_seconds": round(uptime, 1),
            "last_put_age_seconds": (
                round(time.monotonic() - self._last_put_at, 1)
                if self._last_put_at
                else None
            ),
            "last_get_age_seconds": (
                round(time.monotonic() - self._last_get_at, 1)
                if self._last_get_at
                else None
            ),
        }

    # -- Internal helpers -----------------------------------------------------

    def _track_put(self) -> None:
        """Update stats after a successful put."""
        self._total_puts += 1
        self._last_put_at = time.monotonic()
        current = self.qsize()
        if current > self._peak_size:
            self._peak_size = current

    def _maybe_warn(self) -> None:
        """Emit a warning when queue usage crosses the threshold."""
        if self.maxsize <= 0:
            return
        utilization = self.qsize() / self.maxsize
        if utilization >= _LOG_THRESHOLD_PCT and not self._warn_logged:
            self._warn_logged = True
            logger.warning(
                "BoundedAsyncQueue '%s' at %.0f%% capacity (%d/%d) policy=%s",
                self.name,
                utilization * 100,
                self.qsize(),
                self.maxsize,
                self.policy.value,
            )
        elif utilization < _LOG_THRESHOLD_PCT * 0.8:
            # Reset so we can warn again if it fills back up
            self._warn_logged = False

    # -- Overrides: put / put_nowait -----------------------------------------

    async def put(self, item: T) -> None:  # type: ignore[override]
        """Put an item into the queue, respecting the overflow policy.

        Note: ``super().put()`` internally calls ``self.put_nowait()`` (our
        override), which already handles ``_track_put`` and ``_maybe_warn``.
        We therefore do NOT duplicate those calls here for paths that
        delegate to ``super().put()`` or ``self.put_nowait()``.
        """

        if self.policy == OverflowPolicy.BLOCK:
            # super().put() -> self.put_nowait() -> _track_put/_maybe_warn
            await super().put(item)
            return

        if self.policy == OverflowPolicy.WARN_AND_BLOCK:
            if self.full():
                logger.warning(
                    "BoundedAsyncQueue '%s' is full (%d items) -- blocking producer",
                    self.name,
                    self.qsize(),
                )
            # super().put() -> self.put_nowait() -> _track_put/_maybe_warn
            await super().put(item)
            return

        if self.policy == OverflowPolicy.DROP_OLDEST:
            if self.full():
                try:
                    _discarded = self.get_nowait()
                    self._dropped_count += 1
                    if self._dropped_count % 100 == 1:
                        logger.info(
                            "BoundedAsyncQueue '%s' dropped oldest item "
                            "(total dropped: %d)",
                            self.name,
                            self._dropped_count,
                        )
                except asyncio.QueueEmpty:
                    pass
            # put_nowait -> _track_put/_maybe_warn
            try:
                self.put_nowait(item)
            except asyncio.QueueFull:
                # Race condition: another coroutine filled the slot.
                # Fall back to a blocking put (extremely rare).
                # super().put() -> self.put_nowait() -> _track_put
                await super().put(item)
            return

        if self.policy == OverflowPolicy.DROP_NEWEST:
            if self.full():
                self._dropped_count += 1
                if self._dropped_count % 100 == 1:
                    logger.info(
                        "BoundedAsyncQueue '%s' rejected new item "
                        "(total dropped: %d)",
                        self.name,
                        self._dropped_count,
                    )
                return  # silently drop the new item
            # super().put() -> self.put_nowait() -> _track_put/_maybe_warn
            await super().put(item)
            return

    def put_nowait(self, item: T) -> None:  # type: ignore[override]
        """Put an item without blocking, respecting the overflow policy."""

        if self.policy == OverflowPolicy.BLOCK:
            super().put_nowait(item)
            self._track_put()
            self._maybe_warn()
            return

        if self.policy == OverflowPolicy.WARN_AND_BLOCK:
            if self.full():
                logger.warning(
                    "BoundedAsyncQueue '%s' full on put_nowait (%d items)",
                    self.name,
                    self.qsize(),
                )
            super().put_nowait(item)
            self._track_put()
            self._maybe_warn()
            return

        if self.policy == OverflowPolicy.DROP_OLDEST:
            if self.full():
                try:
                    self.get_nowait()
                    self._dropped_count += 1
                except asyncio.QueueEmpty:
                    pass
            super().put_nowait(item)
            self._track_put()
            self._maybe_warn()
            return

        if self.policy == OverflowPolicy.DROP_NEWEST:
            if self.full():
                self._dropped_count += 1
                return
            super().put_nowait(item)
            self._track_put()
            self._maybe_warn()
            return

    # -- Override: get / get_nowait ------------------------------------------

    async def get(self) -> T:  # type: ignore[override]
        """Get an item from the queue.

        Note: ``super().get()`` internally calls ``self.get_nowait()`` (our
        override), which handles stats tracking.  We do NOT duplicate here.
        """
        return await super().get()

    def get_nowait(self) -> T:  # type: ignore[override]
        """Get an item without waiting (tracks stats)."""
        item = super().get_nowait()
        self._total_gets += 1
        self._last_get_at = time.monotonic()
        return item

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BoundedAsyncQueue(name={self.name!r}, maxsize={self.maxsize}, "
            f"policy={self.policy.value}, qsize={self.qsize()})"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_bounded_queue(
    maxsize: int = 0,
    policy: OverflowPolicy = OverflowPolicy.BLOCK,
    name: str = "",
) -> BoundedAsyncQueue:
    """Create a :class:`BoundedAsyncQueue` with the given parameters.

    This is the preferred entry-point for callers who want a one-liner that
    mirrors ``asyncio.Queue()`` but bounded.

    Parameters
    ----------
    maxsize:
        Upper bound.  ``0`` means use ``Ironcliw_DEFAULT_QUEUE_SIZE`` (1000).
    policy:
        Overflow policy to apply.
    name:
        Optional label for monitoring / logging.
    """
    return BoundedAsyncQueue(maxsize=maxsize, policy=policy, name=name)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = [
    "OverflowPolicy",
    "BoundedAsyncQueue",
    "create_bounded_queue",
    "get_all_queue_stats",
]
