"""
v77.1: Saga Pattern Coordinator
================================

Saga pattern for long-running multi-repo operations with compensating actions.

Problem:
    Two-Phase Commit (2PC) works for short operations but:
    - Holds locks for entire duration
    - Doesn't handle long-running operations well
    - Network partitions cause indefinite blocking

Solution:
    Saga Pattern with compensating transactions:
    - Each step has an "undo" action (compensation)
    - If any step fails, run compensations in reverse order
    - No global locks needed

Example:
    1. Modify JARVIS file (Compensate: git checkout)
    2. Modify J-Prime file (Compensate: git checkout)
    3. Run tests (Compensate: N/A)
    4. If step 2 fails → Compensate step 1 automatically

Features:
    - Automatic compensation on failure
    - Parallel step execution where possible
    - Checkpoint recovery
    - Timeout handling per step
    - Event logging for audit

Author: JARVIS v77.1
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SagaState(Enum):
    """State of a saga execution."""
    PENDING = auto()
    RUNNING = auto()
    COMPENSATING = auto()
    COMPLETED = auto()
    COMPENSATED = auto()
    FAILED = auto()
    PARTIALLY_COMPENSATED = auto()


class StepState(Enum):
    """State of an individual saga step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    SKIPPED = "skipped"


@dataclass
class CompensatingAction:
    """
    A compensating action that undoes a saga step.

    Examples:
    - git checkout <original_commit>
    - Delete created file
    - Restore backup
    - Revert API call
    """
    name: str
    action: Callable[[], Coroutine[Any, Any, bool]]
    timeout: float = 30.0
    retries: int = 3
    critical: bool = True  # If True, saga fails if compensation fails


@dataclass
class SagaStep:
    """
    A single step in a saga.

    Each step has:
    - An action to perform
    - An optional compensation to undo it
    - Configurable timeout and retries
    """
    name: str
    action: Callable[[], Coroutine[Any, Any, Any]]
    compensation: Optional[CompensatingAction] = None
    timeout: float = 60.0
    retries: int = 1
    depends_on: List[str] = field(default_factory=list)  # Steps this depends on
    parallel_group: Optional[str] = None  # Steps in same group run in parallel
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of executing a saga step."""
    step_name: str
    state: StepState
    result: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    retries_used: int = 0

    @property
    def duration_ms(self) -> float:
        return (self.completed_at - self.started_at) * 1000


@dataclass
class SagaResult:
    """Result of executing a saga."""
    saga_id: str
    state: SagaState
    step_results: Dict[str, StepResult]
    started_at: float
    completed_at: float = 0.0
    error: Optional[str] = None
    compensations_run: int = 0
    compensations_failed: int = 0

    @property
    def success(self) -> bool:
        return self.state == SagaState.COMPLETED

    @property
    def duration_ms(self) -> float:
        return (self.completed_at - self.started_at) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "saga_id": self.saga_id,
            "state": self.state.name,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "compensations_run": self.compensations_run,
            "compensations_failed": self.compensations_failed,
            "steps": {
                name: {
                    "state": r.state.value,
                    "duration_ms": r.duration_ms,
                    "error": r.error,
                }
                for name, r in self.step_results.items()
            },
        }


class SagaDefinition:
    """
    Definition of a saga with steps and dependencies.

    Supports:
    - Sequential steps
    - Parallel steps (via parallel_group)
    - Dependencies between steps
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._steps: List[SagaStep] = []
        self._step_index: Dict[str, int] = {}

    def add_step(self, step: SagaStep) -> "SagaDefinition":
        """Add a step to the saga."""
        self._step_index[step.name] = len(self._steps)
        self._steps.append(step)
        return self

    def add_sequential(
        self,
        name: str,
        action: Callable[[], Coroutine[Any, Any, Any]],
        compensation: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None,
        timeout: float = 60.0,
    ) -> "SagaDefinition":
        """Add a sequential step (depends on previous step)."""
        depends_on = [self._steps[-1].name] if self._steps else []

        comp_action = None
        if compensation:
            comp_action = CompensatingAction(
                name=f"compensate_{name}",
                action=compensation,
            )

        return self.add_step(SagaStep(
            name=name,
            action=action,
            compensation=comp_action,
            timeout=timeout,
            depends_on=depends_on,
        ))

    def add_parallel(
        self,
        group: str,
        name: str,
        action: Callable[[], Coroutine[Any, Any, Any]],
        compensation: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None,
        timeout: float = 60.0,
    ) -> "SagaDefinition":
        """Add a parallel step (runs with other steps in same group)."""
        comp_action = None
        if compensation:
            comp_action = CompensatingAction(
                name=f"compensate_{name}",
                action=compensation,
            )

        return self.add_step(SagaStep(
            name=name,
            action=action,
            compensation=comp_action,
            timeout=timeout,
            parallel_group=group,
        ))

    @property
    def steps(self) -> List[SagaStep]:
        return self._steps.copy()

    def get_step(self, name: str) -> Optional[SagaStep]:
        idx = self._step_index.get(name)
        return self._steps[idx] if idx is not None else None

    def get_execution_order(self) -> List[List[SagaStep]]:
        """
        Get steps organized by execution order.

        Returns list of step groups. Steps in same group can run in parallel.
        """
        # Build dependency graph
        remaining = set(step.name for step in self._steps)
        completed: Set[str] = set()
        order: List[List[SagaStep]] = []

        while remaining:
            # Find steps whose dependencies are all completed
            ready = []
            for step_name in remaining:
                step = self.get_step(step_name)
                if step and all(dep in completed for dep in step.depends_on):
                    ready.append(step)

            if not ready:
                # Circular dependency or error
                logger.error(f"[Saga] Cannot resolve execution order. Remaining: {remaining}")
                break

            # Group parallel steps
            groups: Dict[Optional[str], List[SagaStep]] = {}
            for step in ready:
                group = step.parallel_group
                if group not in groups:
                    groups[group] = []
                groups[group].append(step)

            # Add non-parallel steps individually
            for group, steps in groups.items():
                if group is None:
                    # Sequential steps
                    for step in steps:
                        order.append([step])
                        completed.add(step.name)
                        remaining.remove(step.name)
                else:
                    # Parallel group
                    order.append(steps)
                    for step in steps:
                        completed.add(step.name)
                        remaining.remove(step.name)

        return order


class SagaCoordinator:
    """
    Saga Pattern Coordinator.

    Executes long-running operations with automatic compensation on failure.

    Features:
    - Parallel step execution
    - Automatic compensation on failure
    - Checkpoint recovery
    - Timeout handling
    - Event logging

    Usage:
        saga = SagaDefinition("multi_repo_evolution")
        saga.add_sequential("modify_jarvis", modify_jarvis_action, compensate_jarvis)
        saga.add_sequential("modify_jprime", modify_jprime_action, compensate_jprime)
        saga.add_sequential("run_tests", run_tests_action)

        coordinator = SagaCoordinator()
        result = await coordinator.execute(saga)
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_parallel: int = 5,
    ):
        self.checkpoint_dir = checkpoint_dir or Path.home() / ".jarvis" / "sagas"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_parallel = max_parallel

        self._active_sagas: Dict[str, SagaResult] = {}
        self._event_handlers: List[Callable[[str, str, Dict], Coroutine]] = []

    async def execute(
        self,
        saga: SagaDefinition,
        saga_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SagaResult:
        """
        Execute a saga.

        Args:
            saga: The saga definition to execute
            saga_id: Optional ID (generated if not provided)
            context: Optional context data passed to steps

        Returns:
            SagaResult with execution outcome
        """
        saga_id = saga_id or str(uuid.uuid4())
        started_at = time.time()

        result = SagaResult(
            saga_id=saga_id,
            state=SagaState.RUNNING,
            step_results={},
            started_at=started_at,
        )
        self._active_sagas[saga_id] = result

        await self._emit_event(saga_id, "saga_started", {"saga_name": saga.name})

        # Get execution order
        execution_order = saga.get_execution_order()
        completed_steps: List[SagaStep] = []

        try:
            # Execute steps in order
            for step_group in execution_order:
                # Save checkpoint before each group
                await self._save_checkpoint(saga_id, result, completed_steps)

                # Execute group (parallel if multiple steps)
                group_results = await self._execute_step_group(saga_id, step_group)

                # Check for failures
                failed_steps = [
                    (step, r) for step, r in zip(step_group, group_results)
                    if r.state == StepState.FAILED
                ]

                # Update result
                for step, step_result in zip(step_group, group_results):
                    result.step_results[step.name] = step_result

                if failed_steps:
                    # Trigger compensation
                    failed_step, failed_result = failed_steps[0]
                    result.error = f"Step '{failed_step.name}' failed: {failed_result.error}"
                    result.state = SagaState.COMPENSATING

                    await self._emit_event(saga_id, "compensation_started", {
                        "failed_step": failed_step.name,
                        "error": failed_result.error,
                    })

                    # Run compensations in reverse order
                    await self._compensate(saga_id, result, completed_steps)
                    break

                # Mark steps as completed
                completed_steps.extend(step_group)

            # All steps completed successfully
            if result.state == SagaState.RUNNING:
                result.state = SagaState.COMPLETED

        except Exception as e:
            logger.error(f"[Saga:{saga_id}] Execution error: {e}")
            result.state = SagaState.FAILED
            result.error = str(e)

            # Try to compensate
            if completed_steps:
                await self._compensate(saga_id, result, completed_steps)

        result.completed_at = time.time()
        await self._emit_event(saga_id, "saga_completed", result.to_dict())

        # Clean up checkpoint
        await self._delete_checkpoint(saga_id)

        del self._active_sagas[saga_id]
        return result

    async def recover(self, saga_id: str, saga: SagaDefinition) -> Optional[SagaResult]:
        """
        Recover a saga from checkpoint after crash.

        Args:
            saga_id: The saga ID to recover
            saga: The saga definition

        Returns:
            SagaResult if recovery was attempted, None if no checkpoint
        """
        checkpoint = await self._load_checkpoint(saga_id)
        if not checkpoint:
            return None

        logger.info(f"[Saga:{saga_id}] Recovering from checkpoint")

        completed_step_names = checkpoint.get("completed_steps", [])
        result = SagaResult(
            saga_id=saga_id,
            state=SagaState.COMPENSATING,
            step_results={},
            started_at=checkpoint.get("started_at", time.time()),
        )

        # Get completed steps
        completed_steps = [
            saga.get_step(name) for name in completed_step_names
            if saga.get_step(name)
        ]

        # Run compensations
        await self._compensate(saga_id, result, completed_steps)

        result.completed_at = time.time()
        result.state = SagaState.COMPENSATED

        await self._delete_checkpoint(saga_id)
        return result

    async def _execute_step_group(
        self,
        saga_id: str,
        steps: List[SagaStep],
    ) -> List[StepResult]:
        """Execute a group of steps (potentially in parallel)."""
        if len(steps) == 1:
            # Single step, execute directly
            return [await self._execute_step(saga_id, steps[0])]

        # Multiple steps, execute in parallel with semaphore
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_with_semaphore(step: SagaStep) -> StepResult:
            async with semaphore:
                return await self._execute_step(saga_id, step)

        tasks = [execute_with_semaphore(step) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for step, result in zip(steps, results):
            if isinstance(result, Exception):
                final_results.append(StepResult(
                    step_name=step.name,
                    state=StepState.FAILED,
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results

    async def _execute_step(self, saga_id: str, step: SagaStep) -> StepResult:
        """Execute a single saga step with retries."""
        result = StepResult(
            step_name=step.name,
            state=StepState.RUNNING,
            started_at=time.time(),
        )

        await self._emit_event(saga_id, "step_started", {"step": step.name})

        for attempt in range(step.retries):
            try:
                result.retries_used = attempt

                step_result = await asyncio.wait_for(
                    step.action(),
                    timeout=step.timeout,
                )

                result.result = step_result
                result.state = StepState.COMPLETED
                result.completed_at = time.time()

                await self._emit_event(saga_id, "step_completed", {
                    "step": step.name,
                    "duration_ms": result.duration_ms,
                })

                return result

            except asyncio.TimeoutError:
                result.error = f"Timeout after {step.timeout}s"
                logger.warning(f"[Saga:{saga_id}] Step {step.name} timeout (attempt {attempt + 1})")

            except Exception as e:
                result.error = str(e)
                logger.warning(f"[Saga:{saga_id}] Step {step.name} failed (attempt {attempt + 1}): {e}")

            # Wait before retry
            if attempt < step.retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))

        # All retries exhausted
        result.state = StepState.FAILED
        result.completed_at = time.time()

        await self._emit_event(saga_id, "step_failed", {
            "step": step.name,
            "error": result.error,
        })

        return result

    async def _compensate(
        self,
        saga_id: str,
        result: SagaResult,
        completed_steps: List[SagaStep],
    ) -> None:
        """Run compensations in reverse order."""
        logger.info(f"[Saga:{saga_id}] Running compensations for {len(completed_steps)} steps")

        for step in reversed(completed_steps):
            if not step.compensation:
                continue

            result.compensations_run += 1
            comp = step.compensation

            await self._emit_event(saga_id, "compensation_step", {
                "step": step.name,
                "compensation": comp.name,
            })

            for attempt in range(comp.retries):
                try:
                    success = await asyncio.wait_for(
                        comp.action(),
                        timeout=comp.timeout,
                    )

                    if success:
                        logger.info(f"[Saga:{saga_id}] Compensated: {step.name}")
                        break
                    else:
                        raise RuntimeError("Compensation returned False")

                except Exception as e:
                    logger.error(f"[Saga:{saga_id}] Compensation failed for {step.name}: {e}")

                    if attempt == comp.retries - 1:
                        result.compensations_failed += 1
                        if comp.critical:
                            result.state = SagaState.PARTIALLY_COMPENSATED
                            logger.error(f"[Saga:{saga_id}] Critical compensation failed!")

        if result.compensations_failed == 0:
            result.state = SagaState.COMPENSATED

    async def _save_checkpoint(
        self,
        saga_id: str,
        result: SagaResult,
        completed_steps: List[SagaStep],
    ) -> None:
        """Save saga checkpoint for recovery."""
        checkpoint = {
            "saga_id": saga_id,
            "started_at": result.started_at,
            "completed_steps": [s.name for s in completed_steps],
            "step_results": {
                name: {"state": r.state.value, "error": r.error}
                for name, r in result.step_results.items()
            },
            "checkpoint_time": time.time(),
        }

        checkpoint_file = self.checkpoint_dir / f"{saga_id}.json"
        checkpoint_file.write_text(json.dumps(checkpoint, indent=2))

    async def _load_checkpoint(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Load saga checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{saga_id}.json"
        if checkpoint_file.exists():
            try:
                return json.loads(checkpoint_file.read_text())
            except Exception:
                pass
        return None

    async def _delete_checkpoint(self, saga_id: str) -> None:
        """Delete saga checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{saga_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    def on_event(self, handler: Callable[[str, str, Dict], Coroutine]) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)

    async def _emit_event(self, saga_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Emit saga event."""
        for handler in self._event_handlers:
            try:
                await handler(saga_id, event_type, data)
            except Exception as e:
                logger.error(f"[Saga] Event handler error: {e}")


# Convenience function for creating evolution sagas
def create_evolution_saga(
    repos: Dict[str, Path],
    changes: Dict[str, List[str]],
    description: str = "",
) -> SagaDefinition:
    """
    Create a saga for multi-repo evolution.

    Args:
        repos: Dict mapping repo_name -> repo_path
        changes: Dict mapping repo_name -> list of files changed
        description: Human-readable description

    Returns:
        SagaDefinition ready for execution
    """
    saga = SagaDefinition(
        name="multi_repo_evolution",
        description=description,
    )

    for repo_name, files in changes.items():
        repo_path = repos.get(repo_name)
        if not repo_path:
            continue

        # Create step for this repo
        async def modify_repo(path=repo_path, file_list=files):
            # Files should already be modified, just stage them
            proc = await asyncio.create_subprocess_exec(
                "git", "add", *file_list,
                cwd=str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0

        async def compensate_repo(path=repo_path):
            # Reset all changes
            proc = await asyncio.create_subprocess_exec(
                "git", "checkout", "HEAD", "--", ".",
                cwd=str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0

        saga.add_sequential(
            name=f"stage_{repo_name}",
            action=modify_repo,
            compensation=compensate_repo,
            timeout=30.0,
        )

    return saga


# Global instance
_saga_coordinator: Optional[SagaCoordinator] = None


def get_saga_coordinator() -> SagaCoordinator:
    """Get or create global saga coordinator."""
    global _saga_coordinator
    if _saga_coordinator is None:
        _saga_coordinator = SagaCoordinator()
    return _saga_coordinator
