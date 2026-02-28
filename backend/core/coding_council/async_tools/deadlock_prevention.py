"""
v77.0: Deadlock Prevention - Gaps #23, #26
==========================================

Comprehensive deadlock prevention:
- Ordered lock acquisition (consistent ordering)
- Lock timeout with configurable backoff
- Dependency graph cycle detection
- Async-safe lock primitives
- Decorator-based timeout management

Author: Ironcliw v77.0
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import weakref
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LockState(Enum):
    """State of a lock."""
    UNLOCKED = "unlocked"
    LOCKED = "locked"
    WAITING = "waiting"
    TIMEOUT = "timeout"


@dataclass
class LockInfo:
    """Information about a lock."""
    lock_id: str
    owner_task: Optional[str] = None
    acquired_at: float = 0.0
    wait_count: int = 0
    state: LockState = LockState.UNLOCKED
    priority: int = 0  # Lower = higher priority in ordering


class LockGraph:
    """
    Directed graph for detecting lock cycles (deadlocks).

    Maintains a graph where:
    - Nodes are tasks
    - Edges represent "task A is waiting for a lock held by task B"

    Cycle detection prevents deadlocks before they occur.
    """

    def __init__(self):
        self._graph: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def add_wait(self, waiter: str, holder: str) -> bool:
        """
        Add a wait edge: waiter is waiting for holder.

        Returns False if this would create a cycle (deadlock).
        """
        async with self._lock:
            # Check if adding this edge would create a cycle
            if self._would_create_cycle(waiter, holder):
                logger.warning(f"[LockGraph] Deadlock detected: {waiter} -> {holder}")
                return False

            self._graph[waiter].add(holder)
            return True

    async def remove_wait(self, waiter: str, holder: str) -> None:
        """Remove a wait edge when lock is acquired."""
        async with self._lock:
            if holder in self._graph[waiter]:
                self._graph[waiter].discard(holder)
                if not self._graph[waiter]:
                    del self._graph[waiter]

    def _would_create_cycle(self, start: str, end: str) -> bool:
        """Check if adding edge start -> end would create a cycle."""
        # DFS from end to see if we can reach start
        visited = set()
        stack = [end]

        while stack:
            node = stack.pop()
            if node == start:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._graph.get(node, set()))

        return False

    async def get_wait_chain(self, task: str) -> List[str]:
        """Get the chain of tasks this task is waiting on."""
        async with self._lock:
            chain = []
            visited = set()
            current = task

            while current and current not in visited:
                visited.add(current)
                chain.append(current)

                # Find who current is waiting for
                waiters = self._graph.get(current, set())
                current = next(iter(waiters), None) if waiters else None

            return chain


# Global lock graph for cross-lock deadlock detection
_global_lock_graph = LockGraph()


class OrderedLock:
    """
    Lock with consistent ordering to prevent deadlocks.

    When multiple OrderedLocks need to be acquired, they should
    always be acquired in order of their priority (lower first).
    """

    _all_locks: Dict[str, weakref.ref] = {}
    _lock_counter = 0

    def __init__(self, name: Optional[str] = None, priority: Optional[int] = None):
        OrderedLock._lock_counter += 1
        self.lock_id = name or f"lock_{OrderedLock._lock_counter}"
        self.priority = priority if priority is not None else OrderedLock._lock_counter
        self._lock = asyncio.Lock()
        self._info = LockInfo(lock_id=self.lock_id, priority=self.priority)
        OrderedLock._all_locks[self.lock_id] = weakref.ref(self)

    @property
    def locked(self) -> bool:
        return self._lock.locked()

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock with optional timeout.

        Returns True if acquired, False if timeout.
        """
        task_name = self._get_current_task_name()
        self._info.wait_count += 1
        self._info.state = LockState.WAITING

        try:
            if timeout is None:
                await self._lock.acquire()
                acquired = True
            else:
                try:
                    await asyncio.wait_for(self._lock.acquire(), timeout)
                    acquired = True
                except asyncio.TimeoutError:
                    acquired = False
                    self._info.state = LockState.TIMEOUT

            if acquired:
                self._info.state = LockState.LOCKED
                self._info.owner_task = task_name
                self._info.acquired_at = time.time()
                logger.debug(f"[OrderedLock] {self.lock_id} acquired by {task_name}")

            return acquired

        finally:
            self._info.wait_count -= 1

    def release(self) -> None:
        """Release the lock."""
        task_name = self._get_current_task_name()

        if self._info.owner_task and self._info.owner_task != task_name:
            logger.warning(
                f"[OrderedLock] {self.lock_id} released by {task_name} "
                f"but owned by {self._info.owner_task}"
            )

        self._info.state = LockState.UNLOCKED
        self._info.owner_task = None
        self._lock.release()
        logger.debug(f"[OrderedLock] {self.lock_id} released by {task_name}")

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def _get_current_task_name(self) -> str:
        try:
            task = asyncio.current_task()
            return task.get_name() if task else "unknown"
        except RuntimeError:
            return "no_event_loop"

    @classmethod
    def get_sorted_locks(cls, locks: List["OrderedLock"]) -> List["OrderedLock"]:
        """Return locks sorted by priority for safe acquisition order."""
        return sorted(locks, key=lambda l: l.priority)


class TimeoutLock:
    """
    Lock with automatic timeout and backoff retry.

    Features:
    - Configurable timeout
    - Exponential backoff on contention
    - Deadlock detection via global graph
    - Automatic cleanup on failure
    """

    def __init__(
        self,
        name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 0.1,
        backoff_max: float = 5.0,
    ):
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._lock = asyncio.Lock()
        self._owner: Optional[str] = None
        self._acquired_at: float = 0.0

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire lock with timeout and retry.

        Usage:
            async with lock.acquire():
                # protected code
        """
        task_name = self._get_current_task_name()

        for attempt in range(self.max_retries + 1):
            # Check for potential deadlock
            if self._owner:
                can_wait = await _global_lock_graph.add_wait(task_name, self._owner)
                if not can_wait:
                    raise DeadlockError(
                        f"Acquiring {self.name} would cause deadlock. "
                        f"Task {task_name} waiting for {self._owner}"
                    )

            try:
                acquired = await asyncio.wait_for(
                    self._lock.acquire(),
                    timeout=self.timeout
                )

                if acquired:
                    self._owner = task_name
                    self._acquired_at = time.time()

                    # Remove wait edge since we acquired
                    if self._owner:
                        await _global_lock_graph.remove_wait(task_name, self._owner)

                    try:
                        yield
                    finally:
                        self._owner = None
                        self._lock.release()
                    return

            except asyncio.TimeoutError:
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = min(
                        self.backoff_base * (2 ** attempt),
                        self.backoff_max
                    )
                    logger.warning(
                        f"[TimeoutLock] {self.name} timeout, retry {attempt + 1} "
                        f"in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise LockTimeoutError(
                        f"Failed to acquire {self.name} after {self.max_retries} retries"
                    )

    def _get_current_task_name(self) -> str:
        try:
            task = asyncio.current_task()
            return task.get_name() if task else "unknown"
        except RuntimeError:
            return "no_event_loop"


class DeadlockError(Exception):
    """Raised when a deadlock would occur."""
    pass


class LockTimeoutError(Exception):
    """Raised when lock acquisition times out."""
    pass


class DeadlockPrevention:
    """
    Central deadlock prevention manager.

    Features:
    - Global lock ordering enforcement
    - Automatic deadlock detection
    - Lock acquisition monitoring
    - Timeout management
    - Resource cleanup
    """

    def __init__(self):
        self._locks: Dict[str, OrderedLock] = {}
        self._timeout_locks: Dict[str, TimeoutLock] = {}
        self._global_timeout: float = 60.0
        self._lock = asyncio.Lock()

    def create_ordered_lock(self, name: str, priority: Optional[int] = None) -> OrderedLock:
        """Create a new ordered lock."""
        if name in self._locks:
            return self._locks[name]

        lock = OrderedLock(name, priority)
        self._locks[name] = lock
        return lock

    def create_timeout_lock(
        self,
        name: str,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> TimeoutLock:
        """Create a new timeout lock."""
        if name in self._timeout_locks:
            return self._timeout_locks[name]

        lock = TimeoutLock(
            name,
            timeout=timeout or self._global_timeout,
            max_retries=max_retries,
        )
        self._timeout_locks[name] = lock
        return lock

    @asynccontextmanager
    async def acquire_multiple(self, lock_names: List[str]):
        """
        Acquire multiple locks in safe order.

        Prevents deadlocks by always acquiring in priority order.
        """
        locks = [self._locks[name] for name in lock_names if name in self._locks]
        sorted_locks = OrderedLock.get_sorted_locks(locks)

        acquired = []
        try:
            for lock in sorted_locks:
                if not await lock.acquire(timeout=self._global_timeout):
                    raise LockTimeoutError(f"Failed to acquire {lock.lock_id}")
                acquired.append(lock)

            yield

        finally:
            # Release in reverse order
            for lock in reversed(acquired):
                lock.release()

    async def check_for_deadlock(self, task: str) -> Optional[List[str]]:
        """Check if a task is in a deadlock cycle."""
        chain = await _global_lock_graph.get_wait_chain(task)
        if len(chain) > 1 and chain[0] in chain[1:]:
            return chain
        return None

    def set_global_timeout(self, timeout: float) -> None:
        """Set default timeout for all locks."""
        self._global_timeout = timeout

    def get_lock_stats(self) -> Dict[str, Any]:
        """Get statistics about all locks."""
        return {
            "ordered_locks": {
                name: {
                    "locked": lock.locked,
                    "owner": lock._info.owner_task,
                    "wait_count": lock._info.wait_count,
                    "priority": lock.priority,
                }
                for name, lock in self._locks.items()
            },
            "timeout_locks": {
                name: {
                    "owner": lock._owner,
                    "timeout": lock.timeout,
                }
                for name, lock in self._timeout_locks.items()
            },
        }


def async_timeout(seconds: float, error_message: Optional[str] = None):
    """
    Decorator to add timeout to async functions.

    Gap #26: Async timeout management

    Usage:
        @async_timeout(30.0)
        async def my_function():
            ...
    """
    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                msg = error_message or f"{func.__name__} timed out after {seconds}s"
                logger.error(f"[async_timeout] {msg}")
                raise TimeoutError(msg)
        return wrapper
    return decorator


def prevent_deadlock(lock_names: List[str]):
    """
    Decorator to safely acquire multiple locks.

    Usage:
        @prevent_deadlock(["lock_a", "lock_b"])
        async def my_function(self):
            # locks are held during execution
            ...
    """
    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Assume self has a deadlock_prevention attribute
            prevention = getattr(self, "deadlock_prevention", None)
            if prevention is None:
                prevention = DeadlockPrevention()

            async with prevention.acquire_multiple(lock_names):
                return await func(self, *args, **kwargs)
        return wrapper
    return decorator


# Global singleton for convenience
_deadlock_prevention = DeadlockPrevention()


def get_deadlock_prevention() -> DeadlockPrevention:
    """Get global deadlock prevention instance."""
    return _deadlock_prevention
