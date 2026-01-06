"""
v77.0: Task Registry - Gaps #24, #27
=====================================

Central task tracking and management:
- Task registration and lifecycle
- Cancellation with cleanup callbacks
- Task groups for coordinated shutdown
- Resource cleanup on failure
- Progress tracking

Author: JARVIS v77.0
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import traceback
import uuid
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TaskState(Enum):
    """State of a registered task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class RegisteredTask:
    """A task registered with the registry."""
    task_id: str
    name: str
    state: TaskState = TaskState.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    error_traceback: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    cleanup_callbacks: List[Callable] = field(default_factory=list)
    _task_ref: Optional[weakref.ref] = field(default=None, repr=False)
    group_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "metadata": self.metadata,
            "error": self.error,
            "group_id": self.group_id,
            "duration": self._get_duration(),
        }

    def _get_duration(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at


@dataclass
class TaskGroup:
    """A group of related tasks for coordinated management."""
    group_id: str
    name: str
    task_ids: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "task_count": len(self.task_ids),
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class TaskRegistry:
    """
    Central registry for tracking async tasks.

    Features:
    - Task registration and lifecycle tracking
    - Cleanup callbacks on task completion/failure
    - Task grouping for coordinated operations
    - Progress tracking
    - Timeout enforcement
    - Graceful cancellation with cleanup
    """

    def __init__(self, max_history: int = 1000):
        self._tasks: Dict[str, RegisteredTask] = {}
        self._groups: Dict[str, TaskGroup] = {}
        self._lock = asyncio.Lock()
        self._max_history = max_history
        self._global_cleanup: List[Callable[[], Coroutine]] = []
        self._task_created_callbacks: List[Callable[[RegisteredTask], Coroutine]] = []
        self._task_completed_callbacks: List[Callable[[RegisteredTask], Coroutine]] = []

    async def register(
        self,
        name: str,
        task_id: Optional[str] = None,
        group_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        cleanup_callback: Optional[Callable] = None,
    ) -> RegisteredTask:
        """
        Register a new task.

        Returns a RegisteredTask that should be updated as the task progresses.
        """
        async with self._lock:
            task_id = task_id or str(uuid.uuid4())

            task = RegisteredTask(
                task_id=task_id,
                name=name,
                metadata=metadata or {},
                group_id=group_id,
            )

            if cleanup_callback:
                task.cleanup_callbacks.append(cleanup_callback)

            self._tasks[task_id] = task

            # Add to group if specified
            if group_id:
                if group_id not in self._groups:
                    self._groups[group_id] = TaskGroup(group_id=group_id, name=f"group_{group_id}")
                self._groups[group_id].task_ids.add(task_id)

            # Notify callbacks
            for callback in self._task_created_callbacks:
                try:
                    await callback(task)
                except Exception as e:
                    logger.error(f"[TaskRegistry] Created callback error: {e}")

            logger.debug(f"[TaskRegistry] Registered task: {name} ({task_id})")
            return task

    async def start(self, task_id: str, asyncio_task: Optional[asyncio.Task] = None) -> bool:
        """Mark a task as started."""
        async with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            task.state = TaskState.RUNNING
            task.started_at = time.time()

            if asyncio_task:
                task._task_ref = weakref.ref(asyncio_task)

            return True

    async def complete(
        self,
        task_id: str,
        result: Any = None,
        error: Optional[Exception] = None,
    ) -> bool:
        """
        Mark a task as completed (successfully or with error).

        Gap #27: Resource cleanup on failure
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            task.completed_at = time.time()
            task.progress = 1.0

            if error:
                task.state = TaskState.FAILED
                task.error = str(error)
                task.error_traceback = traceback.format_exc()
            else:
                task.state = TaskState.COMPLETED
                task.result = result

            # Run cleanup callbacks
            await self._run_cleanup(task)

            # Notify completion callbacks
            for callback in self._task_completed_callbacks:
                try:
                    await callback(task)
                except Exception as e:
                    logger.error(f"[TaskRegistry] Completed callback error: {e}")

            # Cleanup old tasks if over limit
            await self._cleanup_history()

            return True

    async def cancel(self, task_id: str, reason: str = "Cancelled") -> bool:
        """
        Cancel a task and run cleanup.

        Returns True if task was cancelled.
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]

            # Cancel the actual asyncio task if we have a reference
            if task._task_ref:
                actual_task = task._task_ref()
                if actual_task and not actual_task.done():
                    actual_task.cancel()

            task.state = TaskState.CANCELLED
            task.completed_at = time.time()
            task.error = reason

            # Run cleanup callbacks
            await self._run_cleanup(task)

            logger.info(f"[TaskRegistry] Cancelled task: {task.name} ({task_id}): {reason}")
            return True

    async def cancel_group(self, group_id: str, reason: str = "Group cancelled") -> int:
        """Cancel all tasks in a group."""
        if group_id not in self._groups:
            return 0

        group = self._groups[group_id]
        cancelled = 0

        for task_id in list(group.task_ids):
            if await self.cancel(task_id, reason):
                cancelled += 1

        return cancelled

    async def update_progress(self, task_id: str, progress: float) -> bool:
        """Update task progress (0.0 to 1.0)."""
        if task_id not in self._tasks:
            return False

        self._tasks[task_id].progress = max(0.0, min(1.0, progress))
        return True

    async def get_task(self, task_id: str) -> Optional[RegisteredTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    async def get_group_tasks(self, group_id: str) -> List[RegisteredTask]:
        """Get all tasks in a group."""
        if group_id not in self._groups:
            return []

        return [
            self._tasks[tid]
            for tid in self._groups[group_id].task_ids
            if tid in self._tasks
        ]

    async def get_running_tasks(self) -> List[RegisteredTask]:
        """Get all currently running tasks."""
        return [t for t in self._tasks.values() if t.state == TaskState.RUNNING]

    async def get_failed_tasks(self, since: Optional[float] = None) -> List[RegisteredTask]:
        """Get failed tasks, optionally filtered by time."""
        tasks = [t for t in self._tasks.values() if t.state == TaskState.FAILED]
        if since:
            tasks = [t for t in tasks if t.completed_at and t.completed_at >= since]
        return tasks

    def create_group(
        self,
        name: str,
        group_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskGroup:
        """Create a new task group."""
        group_id = group_id or str(uuid.uuid4())

        group = TaskGroup(
            group_id=group_id,
            name=name,
            metadata=metadata or {},
        )
        self._groups[group_id] = group
        return group

    def on_task_created(self, callback: Callable[[RegisteredTask], Coroutine]) -> None:
        """Register callback for task creation."""
        self._task_created_callbacks.append(callback)

    def on_task_completed(self, callback: Callable[[RegisteredTask], Coroutine]) -> None:
        """Register callback for task completion."""
        self._task_completed_callbacks.append(callback)

    def add_global_cleanup(self, cleanup: Callable[[], Coroutine]) -> None:
        """Add global cleanup that runs on shutdown."""
        self._global_cleanup.append(cleanup)

    async def shutdown(self, timeout: float = 30.0) -> Dict[str, int]:
        """
        Graceful shutdown - cancel all running tasks and run cleanup.

        Returns summary of cancelled/completed tasks.
        """
        logger.info("[TaskRegistry] Starting shutdown...")

        running = await self.get_running_tasks()
        cancelled = 0

        # Cancel all running tasks
        for task in running:
            if await self.cancel(task.task_id, "Shutdown"):
                cancelled += 1

        # Run global cleanup
        for cleanup in self._global_cleanup:
            try:
                await asyncio.wait_for(cleanup(), timeout=5.0)
            except Exception as e:
                logger.error(f"[TaskRegistry] Global cleanup error: {e}")

        logger.info(f"[TaskRegistry] Shutdown complete: {cancelled} tasks cancelled")

        return {
            "total_tasks": len(self._tasks),
            "cancelled": cancelled,
            "completed": sum(1 for t in self._tasks.values() if t.state == TaskState.COMPLETED),
            "failed": sum(1 for t in self._tasks.values() if t.state == TaskState.FAILED),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        by_state = {}
        for task in self._tasks.values():
            by_state[task.state.value] = by_state.get(task.state.value, 0) + 1

        return {
            "total_tasks": len(self._tasks),
            "total_groups": len(self._groups),
            "by_state": by_state,
            "running": [t.to_dict() for t in self._tasks.values() if t.state == TaskState.RUNNING],
        }

    async def _run_cleanup(self, task: RegisteredTask) -> None:
        """Run cleanup callbacks for a task."""
        for callback in task.cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"[TaskRegistry] Cleanup callback error for {task.name}: {e}")

    async def _cleanup_history(self) -> None:
        """Remove old completed tasks to prevent memory growth."""
        if len(self._tasks) <= self._max_history:
            return

        # Get completed tasks sorted by completion time
        completed = sorted(
            [t for t in self._tasks.values() if t.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED)],
            key=lambda t: t.completed_at or 0,
        )

        # Remove oldest until under limit
        to_remove = len(self._tasks) - self._max_history
        for task in completed[:to_remove]:
            del self._tasks[task.task_id]

            # Remove from group
            if task.group_id and task.group_id in self._groups:
                self._groups[task.group_id].task_ids.discard(task.task_id)


@asynccontextmanager
async def tracked_task(
    registry: TaskRegistry,
    name: str,
    group_id: Optional[str] = None,
    cleanup: Optional[Callable] = None,
):
    """
    Context manager for tracked task execution.

    Usage:
        async with tracked_task(registry, "my_task") as task:
            # do work
            await registry.update_progress(task.task_id, 0.5)
    """
    task = await registry.register(
        name=name,
        group_id=group_id,
        cleanup_callback=cleanup,
    )

    await registry.start(task.task_id, asyncio.current_task())

    try:
        yield task
        await registry.complete(task.task_id)
    except asyncio.CancelledError:
        await registry.cancel(task.task_id, "Cancelled by event loop")
        raise
    except Exception as e:
        await registry.complete(task.task_id, error=e)
        raise


def register_task(
    registry_attr: str = "_task_registry",
    name: Optional[str] = None,
    group_id: Optional[str] = None,
):
    """
    Decorator to automatically register and track a task.

    Usage:
        class MyService:
            def __init__(self):
                self._task_registry = TaskRegistry()

            @register_task()
            async def my_task(self):
                ...
    """
    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            registry: TaskRegistry = getattr(self, registry_attr, None)
            if registry is None:
                # No registry, just run the function
                return await func(self, *args, **kwargs)

            task_name = name or func.__name__

            async with tracked_task(registry, task_name, group_id):
                return await func(self, *args, **kwargs)

        return wrapper
    return decorator


# Global singleton for convenience
_task_registry = TaskRegistry()


def get_task_registry() -> TaskRegistry:
    """Get global task registry instance."""
    return _task_registry
