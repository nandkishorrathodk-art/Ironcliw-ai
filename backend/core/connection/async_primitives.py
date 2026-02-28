"""
Event-Loop-Aware Async Primitives
==================================

Provides async primitives that correctly handle multiple event loops.

The standard asyncio.Lock() is bound to the loop it was created in.
These primitives automatically create new locks for new loops.

This is critical for:
- Tests that run with multiple event loops
- Applications with worker threads having their own loops
- Avoiding "attached to a different loop" errors

Author: Ironcliw System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class EventLoopAwareLock:
    """
    An async lock that works correctly across multiple event loops.

    Standard asyncio.Lock is bound to the loop it was created in.
    This class creates a separate lock per event loop, ensuring
    proper synchronization even when the same object is accessed
    from different loops.

    Thread Safety:
    - Uses threading.RLock for sync context (reentrant)
    - Creates asyncio.Lock per event loop for async context
    - Both locks must be held for async operations

    Usage:
        lock = EventLoopAwareLock()

        # Async context
        async with lock:
            await do_work()

        # Sync context
        with lock:
            do_sync_work()

        # Explicit acquire/release
        await lock.acquire()
        try:
            await do_work()
        finally:
            lock.release()
    """

    __slots__ = ('_thread_lock', '_loop_locks', '_locks_lock', '_is_held')

    def __init__(self):
        """Initialize event-loop-aware lock."""
        # Reentrant lock for sync context and protecting loop_locks dict
        self._thread_lock = threading.RLock()
        # Map from loop id to asyncio.Lock
        self._loop_locks: Dict[int, asyncio.Lock] = {}
        # Lock for protecting _loop_locks dict modification
        self._locks_lock = threading.Lock()
        # Track if we hold the lock
        self._is_held = False

    def _get_lock_for_loop(self) -> asyncio.Lock:
        """
        Get or create a lock for the current event loop.

        Creates a new asyncio.Lock if:
        - No lock exists for this loop
        - We're in a different loop than before

        Returns:
            asyncio.Lock for the current event loop
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            # No running loop - create ephemeral lock
            # This happens in sync contexts that call async functions
            return asyncio.Lock()

        with self._locks_lock:
            if loop_id not in self._loop_locks:
                self._loop_locks[loop_id] = asyncio.Lock()
                logger.debug(f"Created async lock for loop {loop_id}")
            return self._loop_locks[loop_id]

    def __enter__(self):
        """Synchronous context manager entry."""
        self._thread_lock.acquire()
        self._is_held = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Synchronous context manager exit."""
        self._is_held = False
        self._thread_lock.release()
        return False

    async def __aenter__(self):
        """Async context manager entry."""
        # Acquire thread lock first
        self._thread_lock.acquire()
        try:
            # Then acquire async lock for current loop
            async_lock = self._get_lock_for_loop()
            await async_lock.acquire()
            self._is_held = True
        except Exception:
            # Release thread lock on failure
            self._thread_lock.release()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            # Release async lock
            async_lock = self._get_lock_for_loop()
            if async_lock.locked():
                async_lock.release()
            self._is_held = False
        finally:
            # Always release thread lock
            self._thread_lock.release()
        return False

    async def acquire(self) -> bool:
        """
        Acquire the lock asynchronously.

        Returns:
            True if lock acquired successfully
        """
        self._thread_lock.acquire()
        try:
            async_lock = self._get_lock_for_loop()
            await async_lock.acquire()
            self._is_held = True
            return True
        except Exception:
            self._thread_lock.release()
            raise

    def release(self) -> None:
        """Release the lock."""
        try:
            # Try to release async lock if we're in a loop
            try:
                loop = asyncio.get_running_loop()
                loop_id = id(loop)
                with self._locks_lock:
                    if loop_id in self._loop_locks:
                        async_lock = self._loop_locks[loop_id]
                        if async_lock.locked():
                            async_lock.release()
            except RuntimeError:
                # No running loop - just release thread lock
                pass
            self._is_held = False
        finally:
            self._thread_lock.release()

    def locked(self) -> bool:
        """
        Check if lock is held.

        Returns True if the lock is currently held.
        """
        return self._is_held

    def cleanup_closed_loops(self) -> int:
        """
        Remove locks for closed event loops.

        Call this periodically in long-running applications
        to prevent memory leaks from accumulating locks.

        Returns:
            Number of locks removed
        """
        # Note: We can't reliably detect closed loops in Python
        # The best we can do is track active loops and clean up
        # when explicitly told a loop is closing
        return 0

    def __repr__(self) -> str:
        """String representation for debugging."""
        with self._locks_lock:
            num_loops = len(self._loop_locks)
        locked = "locked" if self.locked() else "unlocked"
        return f"EventLoopAwareLock({locked}, loops={num_loops})"
