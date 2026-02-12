"""
Unified Agent Runtime — The missing outer loop for JARVIS autonomous agent behavior.

Each active goal gets its own coroutine (_goal_runner) that runs
SENSE → THINK → ACT → VERIFY → REFLECT as fast as the goal allows.
A slow housekeeping loop handles promotion, timeouts, and cleanup.

This module provides:
- Goal submission, cancellation, and priority scheduling
- Per-goal async coroutines (event-driven, NOT tick-driven)
- ScreenLease for mutual exclusion on UI interactions
- GoalDataBus for inter-goal data sharing
- SQLite-backed checkpoint persistence for crash recovery
- LLM-based reasoning with ThinkMode routing (DECOMPOSE/NEXT_STEP/REPLAN)
- Per-step dynamic escalation with goal-level floor
- Working memory compaction to prevent context overflow
- Replan loop detection to prevent infinite cycles
- Real-time progress streaming via WebSocket
- Cross-repo integration (J-Prime reasoning, Reactor-Core learning)
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.autonomy.agent_runtime_models import (
    EscalationLevel,
    Goal,
    GoalDataBus,
    GoalPriority,
    GoalStatus,
    GoalStep,
    ScreenLease,
    TERMINAL_STATES,
    ThinkMode,
    VerificationStrategy,
    _env_bool,
    _env_float,
    _env_int,
)

logger = logging.getLogger("jarvis.agent_runtime")


# ─────────────────────────────────────────────────────────────
# GoalCheckpointStore — SQLite-based goal persistence
# ─────────────────────────────────────────────────────────────

class GoalCheckpointStore:
    """SQLite-based goal persistence for crash recovery.

    Uses WAL mode for concurrent reads and crash safety.
    Follows learning_database.py schema migration pattern.
    """

    CURRENT_SCHEMA_VERSION = 1
    DB_PATH = Path(os.getenv(
        "AGENT_RUNTIME_DB_PATH",
        str(Path.home() / ".jarvis" / "agent_runtime.db")
    ))

    def __init__(self):
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Create tables if needed. Handles schema migrations."""
        if self._initialized:
            return
        try:
            import aiosqlite
        except ImportError:
            logger.warning("[CheckpointStore] aiosqlite not installed — persistence disabled")
            return

        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(str(self.DB_PATH)) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    schema_version INTEGER DEFAULT 1,
                    description TEXT,
                    status TEXT,
                    priority INTEGER,
                    source TEXT,
                    state_json TEXT,
                    created_at REAL,
                    updated_at REAL,
                    kernel_pid INTEGER
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_goals_status
                ON goals(status)
            """)
            await db.commit()

        self._initialized = True
        logger.info(f"[CheckpointStore] Initialized at {self.DB_PATH}")

    async def save(self, goal: Goal):
        """Atomic save of a goal state."""
        if not self._initialized:
            return

        import aiosqlite

        state_json = goal.to_json()
        async with self._lock:
            async with aiosqlite.connect(str(self.DB_PATH)) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("""
                    INSERT OR REPLACE INTO goals
                    (goal_id, schema_version, description, status, priority, source,
                     state_json, created_at, updated_at, kernel_pid)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    goal.goal_id, self.CURRENT_SCHEMA_VERSION,
                    goal.description, goal.status.value, goal.priority.value,
                    goal.source, state_json, goal.created_at, time.time(),
                    os.getpid(),
                ))
                await db.commit()

    async def get_incomplete(self) -> List[Goal]:
        """Load goals that aren't in terminal states (for crash recovery)."""
        if not self._initialized:
            return []

        import aiosqlite

        async with aiosqlite.connect(str(self.DB_PATH)) as db:
            cursor = await db.execute("""
                SELECT state_json, schema_version FROM goals
                WHERE status NOT IN ('completed', 'failed', 'abandoned', 'cancelled')
            """)
            rows = await cursor.fetchall()

        goals = []
        for row in rows:
            state_json, version = row
            if version != self.CURRENT_SCHEMA_VERSION:
                state_json = self._migrate(state_json, version)
            try:
                goal = Goal.from_json(state_json)
                goals.append(goal)
            except Exception as e:
                logger.warning(f"[CheckpointStore] Failed to restore goal: {e}")
        return goals

    async def mark_terminal(self, goal_id: str, status: str):
        """Quick update of goal status to a terminal state."""
        if not self._initialized:
            return

        import aiosqlite

        async with self._lock:
            async with aiosqlite.connect(str(self.DB_PATH)) as db:
                await db.execute(
                    "UPDATE goals SET status = ?, updated_at = ? WHERE goal_id = ?",
                    (status, time.time(), goal_id)
                )
                await db.commit()

    async def cleanup_old(self, max_age_seconds: float = 86400 * 7):
        """Remove goals older than max_age_seconds in terminal states."""
        if not self._initialized:
            return

        import aiosqlite

        cutoff = time.time() - max_age_seconds
        async with self._lock:
            async with aiosqlite.connect(str(self.DB_PATH)) as db:
                await db.execute("""
                    DELETE FROM goals
                    WHERE status IN ('completed', 'failed', 'abandoned', 'cancelled')
                    AND updated_at < ?
                """, (cutoff,))
                await db.commit()

    def _migrate(self, state_json: str, from_version: int) -> str:
        """Migrate checkpoint from older schema version."""
        # Schema v1 is current — no migrations needed yet
        # Future: add migration logic here
        return state_json


# ─────────────────────────────────────────────────────────────
# UnifiedAgentRuntime — The outer loop
# ─────────────────────────────────────────────────────────────

class UnifiedAgentRuntime:
    """
    The missing outer loop for JARVIS autonomous agent behavior.

    Each active goal gets its own coroutine (_goal_runner) that runs
    SENSE→THINK→ACT→VERIFY→REFLECT as fast as the goal allows.
    A slow housekeeping loop handles promotion, timeouts, and cleanup.
    """

    def __init__(self, autonomous_agent):
        """
        Args:
            autonomous_agent: The AutonomousAgent instance (provides
                reasoning_engine, tool_orchestrator, memory, etc.)
        """
        self._agent = autonomous_agent
        self._goal_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._active_goals: Dict[str, Goal] = {}
        self._goal_runners: Dict[str, asyncio.Task] = {}
        self._goal_locks: Dict[str, asyncio.Lock] = {}
        self._screen_lease = ScreenLease()
        self._data_bus = GoalDataBus()
        self._checkpoint_store = GoalCheckpointStore()
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._progress_callbacks: List[Callable] = []

        # Config (all env-var driven, no hardcoding)
        self._max_concurrent = _env_int("AGENT_RUNTIME_MAX_CONCURRENT", 3)
        self._max_iterations = _env_int("AGENT_RUNTIME_MAX_ITERATIONS", 20)
        self._max_duration = _env_float("AGENT_RUNTIME_MAX_DURATION", 600.0)
        self._housekeeping_interval = _env_float("AGENT_RUNTIME_HOUSEKEEPING_INTERVAL", 5.0)
        self._think_timeout = _env_float("AGENT_RUNTIME_THINK_TIMEOUT", 30.0)
        self._act_timeout = _env_float("AGENT_RUNTIME_ACT_TIMEOUT", 60.0)
        self._max_queue_size = _env_int("AGENT_RUNTIME_MAX_QUEUE", 50)
        self._step_max_retries = _env_int("AGENT_RUNTIME_STEP_MAX_RETRIES", 3)
        self._goal_gen_interval = _env_float("AGENT_RUNTIME_GOAL_GEN_INTERVAL", 60.0)
        self._goal_gen_threshold = _env_float("AGENT_RUNTIME_GOAL_GEN_THRESHOLD", 0.7)
        self._enabled = _env_bool("AGENT_RUNTIME_ENABLED", True)
        self._cleanup_age = _env_float("AGENT_RUNTIME_CLEANUP_AGE", 86400 * 7)

        # Escalation keywords for dynamic per-step assessment
        self._dangerous_actions = {
            "delete", "remove", "destroy", "kill", "terminate", "format",
            "drop", "truncate", "purge", "wipe", "uninstall", "shutdown",
        }
        self._high_risk_actions = {
            "send", "email", "message", "post", "publish", "deploy",
            "transfer", "payment", "purchase", "sudo", "admin",
        }

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    async def start(self):
        """Start the runtime. Called by JarvisSystemKernel during Phase 4."""
        if not self._enabled:
            logger.info("[AgentRuntime] Disabled via AGENT_RUNTIME_ENABLED=false")
            return

        # Wait for required dependencies
        await self._wait_for_dependencies(timeout=30.0)

        await self._checkpoint_store.initialize()
        await self._resume_incomplete_goals()
        self._running = True
        logger.info("[AgentRuntime] Started (max_concurrent=%d, max_iterations=%d)",
                     self._max_concurrent, self._max_iterations)

    async def stop(self):
        """Graceful shutdown. Checkpoint all active goals."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        # Cancel all goal runners
        for goal_id, task in list(self._goal_runners.items()):
            task.cancel()

        # Wait for graceful completion
        if self._goal_runners:
            await asyncio.gather(
                *self._goal_runners.values(),
                return_exceptions=True
            )

        # Checkpoint everything still active
        for goal in self._active_goals.values():
            if goal.status not in TERMINAL_STATES:
                goal.status = GoalStatus.PAUSED
                await self._checkpoint_store.save(goal)

        # Periodic cleanup of old checkpoints
        await self._checkpoint_store.cleanup_old(self._cleanup_age)

        logger.info("[AgentRuntime] Stopped, %d goals checkpointed",
                     len(self._active_goals))

    async def _wait_for_dependencies(self, timeout: float):
        """Wait for required components with retry."""
        start = time.time()
        while time.time() - start < timeout:
            if self._has_dependencies():
                logger.info("[AgentRuntime] Dependencies ready")
                return
            logger.info("[AgentRuntime] Waiting for dependencies...")
            await asyncio.sleep(2.0)
        logger.warning("[AgentRuntime] Dependencies not ready after %.0fs, "
                       "starting in degraded mode", timeout)

    def _has_dependencies(self) -> bool:
        """Check if the autonomous agent's core components are available."""
        return (
            self._agent is not None
            and getattr(self._agent, 'reasoning_engine', None) is not None
            and getattr(self._agent, 'tool_orchestrator', None) is not None
        )

    async def _resume_incomplete_goals(self):
        """Restore goals from checkpoint store after crash recovery."""
        goals = await self._checkpoint_store.get_incomplete()
        if not goals:
            return

        logger.info("[AgentRuntime] Resuming %d incomplete goals from checkpoint", len(goals))
        for goal in goals:
            # Re-queue with their original priority
            goal.status = GoalStatus.PENDING
            try:
                await self._goal_queue.put((goal.priority.value, goal.goal_id, goal))
            except Exception as e:
                logger.warning("[AgentRuntime] Failed to re-queue goal %s: %s",
                              goal.goal_id, e)

    # ─────────────────────────────────────────────────────────
    # Goal Submission & Cancellation
    # ─────────────────────────────────────────────────────────

    async def submit_goal(
        self,
        description: str,
        priority: GoalPriority = GoalPriority.NORMAL,
        source: str = "user",
        context: Optional[Dict] = None,
        parent_goal_id: Optional[str] = None,
        needs_vision: bool = False,
    ) -> str:
        """Submit a new goal for pursuit. Returns goal_id."""
        # Backpressure check
        if self._goal_queue.qsize() >= self._max_queue_size:
            raise RuntimeError(
                f"Goal queue full ({self._max_queue_size}). "
                "Complete or cancel existing goals first."
            )

        goal = Goal(
            description=description,
            priority=priority,
            source=source,
            escalation_floor=self._assess_initial_escalation(description, source),
            parent_goal_id=parent_goal_id,
            needs_vision=needs_vision,
            metadata=context or {},
        )

        await self._checkpoint_store.save(goal)
        await self._goal_queue.put((priority.value, goal.goal_id, goal))
        await self._emit_progress(goal, "submitted", f"Goal queued: {description[:80]}")

        # If runtime is active, try to promote immediately
        if self._running:
            await self._promote_pending_goals()

        return goal.goal_id

    async def cancel_goal(self, goal_id: str, reason: str = "user_cancelled"):
        """Cancel a specific goal."""
        if goal_id in self._active_goals:
            goal = self._active_goals[goal_id]
            goal.status = GoalStatus.CANCELLED
            goal.cancelled_reason = reason
            goal.completed_at = time.time()
            # Cancel the runner coroutine
            if goal_id in self._goal_runners:
                self._goal_runners[goal_id].cancel()
            await self._on_goal_complete(goal)
            await self._emit_progress(goal, "cancelled", f"Goal cancelled: {reason}")

    async def cancel_active_goals(self, reason: str = "user_cancelled"):
        """Cancel all active goals. Called on 'JARVIS stop'."""
        for goal_id in list(self._active_goals.keys()):
            await self.cancel_goal(goal_id, reason)

    async def get_goal_status(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a goal."""
        if goal_id in self._active_goals:
            goal = self._active_goals[goal_id]
            return {
                "goal_id": goal.goal_id,
                "description": goal.description,
                "status": goal.status.value,
                "steps_completed": goal.completed_step_count(),
                "steps_failed": goal.failed_step_count(),
                "total_steps": len(goal.steps),
                "iteration": goal.working_memory.iteration_count,
                "elapsed_seconds": goal.elapsed_seconds(),
                "confidence": goal.working_memory.last_confidence,
            }
        return None

    async def get_all_goals_status(self) -> List[Dict[str, Any]]:
        """Get status of all active goals."""
        results = []
        for goal_id in self._active_goals:
            status = await self.get_goal_status(goal_id)
            if status:
                results.append(status)
        return results

    def register_progress_callback(self, callback: Callable):
        """Register callback for real-time progress events."""
        self._progress_callbacks.append(callback)

    # ─────────────────────────────────────────────────────────
    # Housekeeping Loop (slow, periodic)
    # ─────────────────────────────────────────────────────────

    async def housekeeping_loop(self):
        """Slow loop for promotion, timeouts, self-directed goals, cleanup.

        This does NOT advance goals — each goal has its own coroutine
        that runs as fast as it can.
        """
        last_goal_gen = time.time()

        while self._running:
            try:
                await self._promote_pending_goals()
                await self._check_timeouts()
                await self._cleanup_completed_runners()

                # Self-directed goal generation (less frequent)
                now = time.time()
                if now - last_goal_gen >= self._goal_gen_interval:
                    last_goal_gen = now
                    await self._maybe_generate_proactive_goal()

            except Exception as e:
                logger.error("[AgentRuntime] Housekeeping error: %s", e, exc_info=True)

            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._housekeeping_interval,
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Normal — keep going

        logger.info("[AgentRuntime] Housekeeping loop exited")

    async def _promote_pending_goals(self):
        """Move goals from queue to active, spawning runner coroutines."""
        while len(self._active_goals) < self._max_concurrent:
            try:
                _, goal_id, goal = self._goal_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            self._active_goals[goal_id] = goal
            self._goal_locks[goal_id] = asyncio.Lock()
            goal.status = GoalStatus.ACTIVE
            goal.started_at = time.time()

            # Spawn dedicated coroutine per goal
            runner = asyncio.create_task(
                self._goal_runner(goal),
                name=f"goal-runner-{goal_id}",
            )
            runner.add_done_callback(
                lambda t, gid=goal_id: self._runner_done(gid, t)
            )
            self._goal_runners[goal_id] = runner

            await self._emit_progress(goal, "started",
                                       f"Starting: {goal.description[:80]}")

    async def _check_timeouts(self):
        """Check for goals that exceeded their max duration."""
        for goal_id, goal in list(self._active_goals.items()):
            if goal.status == GoalStatus.ACTIVE and goal.is_expired():
                logger.warning("[AgentRuntime] Goal %s expired (%.0fs)",
                              goal_id, goal.elapsed_seconds())
                goal.status = GoalStatus.FAILED
                goal.working_memory.blockers.append(
                    f"Exceeded max duration ({goal.max_duration_seconds}s)"
                )
                if goal_id in self._goal_runners:
                    self._goal_runners[goal_id].cancel()

    async def _cleanup_completed_runners(self):
        """Remove finished runner tasks."""
        done_ids = [
            gid for gid, task in self._goal_runners.items()
            if task.done()
        ]
        for gid in done_ids:
            self._goal_runners.pop(gid, None)
            self._goal_locks.pop(gid, None)
            self._active_goals.pop(gid, None)

    def _runner_done(self, goal_id: str, task: asyncio.Task):
        """Callback when a goal runner finishes."""
        if task.cancelled():
            logger.debug("[AgentRuntime] Goal %s runner cancelled", goal_id)
        elif task.exception():
            logger.error("[AgentRuntime] Goal %s runner exception: %s",
                        goal_id, task.exception())

    # ─────────────────────────────────────────────────────────
    # Goal Runner (event-driven per-goal coroutine)
    # ─────────────────────────────────────────────────────────

    async def _goal_runner(self, goal: Goal):
        """Dedicated coroutine per goal.

        Runs steps as fast as the goal allows — no tick waiting.
        Non-vision goals run truly in parallel.
        """
        try:
            while (goal.status == GoalStatus.ACTIVE
                   and not self._shutdown_event.is_set()):

                async with self._goal_locks[goal.goal_id]:
                    await self._advance_goal(goal)

                await self._checkpoint_store.save(goal)

                if goal.status in TERMINAL_STATES:
                    break

                # Compact working memory if needed
                wm = goal.working_memory
                if len(wm.observations) > wm.MAX_OBSERVATIONS:
                    await wm.compact(self._summarize)
                wm.trim_context_if_needed()

        except asyncio.CancelledError:
            logger.info("[AgentRuntime] Goal %s runner cancelled", goal.goal_id)
        except Exception as e:
            logger.error("[AgentRuntime] Goal %s runner error: %s",
                        goal.goal_id, e, exc_info=True)
            goal.status = GoalStatus.FAILED
            goal.working_memory.blockers.append(f"Runner error: {e}")
        finally:
            await self._on_goal_complete(goal)

    # ─────────────────────────────────────────────────────────
    # ReAct Loop — SENSE → THINK → ACT → VERIFY → REFLECT
    # ─────────────────────────────────────────────────────────

    async def _advance_goal(self, goal: Goal):
        """Execute one iteration of the sense→think→act→verify→reflect loop."""
        wm = goal.working_memory
        wm.iteration_count += 1

        # Guard: max iterations
        if wm.iteration_count > goal.max_iterations:
            goal.status = GoalStatus.FAILED
            wm.blockers.append(f"Exceeded max iterations ({goal.max_iterations})")
            return

        # Guard: expired
        if goal.is_expired():
            goal.status = GoalStatus.FAILED
            wm.blockers.append(
                f"Exceeded max duration ({goal.max_duration_seconds}s)"
            )
            return

        # === SENSE (conditional) ===
        if self._step_needs_sensing(goal):
            if goal.needs_vision:
                async with self._screen_lease.acquire(goal.goal_id):
                    observation = await self._sense(goal)
            else:
                observation = await self._sense(goal)
        else:
            observation = f"Iteration {wm.iteration_count}: No sensing needed."
        wm.observations.append(observation)

        # === THINK (mode-aware) ===
        think_mode = self._determine_think_mode(goal)

        # Fast path: if we already have planned steps, skip LLM
        if (think_mode == ThinkMode.NEXT_STEP
                and wm.planned_steps
                and not wm.needs_replan):
            next_planned = wm.planned_steps.pop(0)
            thought = f"Executing pre-planned step: {next_planned.get('description', '')}"
            plan = next_planned
        else:
            context = self._build_think_context(goal, observation)
            try:
                thought, plan = await asyncio.wait_for(
                    self._think(goal, observation, context, think_mode),
                    timeout=self._think_timeout,
                )
            except asyncio.TimeoutError:
                wm.blockers.append(
                    f"THINK timed out at iteration {wm.iteration_count}"
                )
                return  # Retry next iteration

            wm.needs_replan = False
            wm.decisions.append(thought)

            # If DECOMPOSE returned multiple steps, store remainder
            if think_mode == ThinkMode.DECOMPOSE and isinstance(plan, dict):
                remaining = plan.get("remaining_steps", [])
                if remaining:
                    wm.planned_steps = remaining
                    plan = plan.get("first_step", plan)

        # === ACT (with per-step escalation) ===
        step = GoalStep(
            description=plan.get("description", ""),
            action=plan.get("action", {}),
            action_type=plan.get("action_type", ""),
            thought=thought,
            needs_vision=plan.get("needs_vision", False),
            verification_strategy=VerificationStrategy(
                plan.get("verification", "semantic")
            ),
        )
        goal.steps.append(step)
        step.started_at = time.time()
        step.status = "executing"

        # Dynamic per-step escalation
        step_escalation = self._assess_step_escalation(step)
        effective_escalation = max(
            goal.escalation_floor.value, step_escalation.value
        )
        step.escalation_level = EscalationLevel(effective_escalation)

        if step.escalation_level >= EscalationLevel.REFUSE:
            step.status = "skipped"
            step.observation = "Refused: safety constraint violation"
            await self._emit_progress(goal, "step_refused", step.description[:60])
            return

        if step.escalation_level >= EscalationLevel.ASK_BEFORE:
            goal.status = GoalStatus.PAUSED
            await self._escalate_to_human(goal, step)
            return  # Pauses until user responds

        # Execute the action
        if step.needs_vision:
            async with self._screen_lease.acquire(goal.goal_id):
                result = await self._act(goal, step)
        else:
            result = await self._act(goal, step)

        step.result = result
        step.completed_at = time.time()
        wm.context_accumulated[step.step_id] = result

        # Publish to data bus for other goals
        await self._data_bus.publish(
            f"{goal.goal_id}:{step.step_id}", result
        )

        # Emit progress
        await self._emit_progress(
            goal, "step_completed",
            f"Step {len(goal.steps)}: {step.description[:60]}",
        )

        # === VERIFY (strategy-aware) ===
        verification = await self._verify(goal, step, result)
        step.observation = verification.get("observation", "")
        step.confidence = verification.get("confidence", 0.0)
        wm.last_confidence = step.confidence

        if verification.get("success", False):
            step.status = "completed"
        else:
            step.status = "failed"

        # === REFLECT ===
        decision = await self._reflect(goal, step, verification)

        if decision == "complete":
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()
        elif decision == "replan":
            wm.plan_history.append(plan)
            wm.planned_steps.clear()  # Discard remaining pre-planned steps
            wm.needs_replan = True
            # Check for infinite replan loops
            if self._is_plan_repeating(plan, wm.plan_history):
                goal.status = GoalStatus.FAILED
                wm.blockers.append("Detected repeating plan pattern — abandoning")
        elif decision == "escalate":
            goal.status = GoalStatus.PAUSED
            await self._escalate_to_human(goal, step)
        elif decision == "abandon":
            goal.status = GoalStatus.ABANDONED
        # else: "continue" — loop immediately to next iteration

    # ─────────────────────────────────────────────────────────
    # SENSE Phase
    # ─────────────────────────────────────────────────────────

    def _step_needs_sensing(self, goal: Goal) -> bool:
        """Determine if current iteration needs environmental sensing."""
        wm = goal.working_memory
        # First iteration always senses
        if wm.iteration_count <= 1:
            return True
        # Vision goals need sensing when previous step used vision
        if goal.needs_vision:
            return True
        # Sense every 3rd iteration for non-vision goals
        return wm.iteration_count % 3 == 0

    async def _sense(self, goal: Goal) -> str:
        """Gather environmental observations relevant to the goal."""
        observations = []

        # Use the agent's integration manager for context
        try:
            if hasattr(self._agent, 'integration_manager') and self._agent.integration_manager:
                ctx = await self._agent.integration_manager.get_context()
                if ctx:
                    observations.append(f"System context: {json.dumps(ctx, default=str)[:500]}")
        except Exception as e:
            observations.append(f"Context gathering error: {e}")

        # If vision-capable, capture screen state
        if goal.needs_vision:
            try:
                screen_state = await self._capture_screen_state()
                if screen_state:
                    observations.append(f"Screen state: {screen_state}")
            except Exception as e:
                observations.append(f"Vision error: {e}")

        if not observations:
            observations.append(f"Iteration {goal.working_memory.iteration_count}: Ready to proceed.")

        return " | ".join(observations)

    async def _capture_screen_state(self) -> Optional[str]:
        """Capture and describe the current screen state."""
        try:
            from backend.autonomy.vision_decision_pipeline import get_vision_pipeline
            pipeline = await get_vision_pipeline()
            if pipeline:
                result = await pipeline.capture_and_analyze()
                return result.get("description", "Screen captured")
        except ImportError:
            pass
        except Exception as e:
            logger.debug("[AgentRuntime] Screen capture failed: %s", e)
        return None

    # ─────────────────────────────────────────────────────────
    # THINK Phase
    # ─────────────────────────────────────────────────────────

    def _determine_think_mode(self, goal: Goal) -> ThinkMode:
        """Select appropriate reasoning strategy."""
        if goal.working_memory.iteration_count == 1:
            return ThinkMode.DECOMPOSE
        elif goal.working_memory.needs_replan:
            return ThinkMode.REPLAN
        else:
            return ThinkMode.NEXT_STEP

    def _build_think_context(self, goal: Goal, observation: str) -> Dict:
        """Build context dict for the THINK phase."""
        wm = goal.working_memory
        return {
            "goal": goal.description,
            "iteration": wm.iteration_count,
            "observation": observation,
            "prior_decisions": wm.decisions[-5:],  # Last 5 decisions
            "prior_observations": wm.observations[-5:],
            "completed_steps": [
                {"description": s.description, "status": s.status,
                 "confidence": s.confidence, "observation": s.observation}
                for s in goal.steps[-5:]
            ],
            "blockers": wm.blockers,
            "plan_history_count": len(wm.plan_history),
            "elapsed_seconds": goal.elapsed_seconds(),
            "remaining_iterations": goal.max_iterations - wm.iteration_count,
        }

    async def _think(
        self, goal: Goal, observation: str,
        context: Dict, mode: ThinkMode,
    ) -> Tuple[str, Dict]:
        """Route to appropriate thinking strategy."""
        if mode == ThinkMode.DECOMPOSE:
            return await self._think_decompose(goal, observation, context)
        elif mode == ThinkMode.REPLAN:
            return await self._think_replan(goal, observation, context)
        else:
            return await self._think_next_step(goal, observation, context)

    async def _think_decompose(
        self, goal: Goal, observation: str, context: Dict,
    ) -> Tuple[str, Dict]:
        """First iteration: break goal into concrete steps."""
        prompt = (
            f"You are JARVIS, an autonomous agent. Decompose this goal into "
            f"concrete executable steps.\n\n"
            f"Goal: {goal.description}\n"
            f"Current observation: {observation}\n"
            f"Context: {json.dumps(context, default=str)[:2000]}\n\n"
            f"Return a JSON object with:\n"
            f"- 'thought': your reasoning about how to approach this\n"
            f"- 'first_step': {{'description': str, 'action': dict, "
            f"'action_type': str, 'needs_vision': bool, "
            f"'verification': 'visual'|'api_result'|'semantic'|'none'}}\n"
            f"- 'remaining_steps': list of step objects in the same format\n"
            f"- 'estimated_total_steps': int"
        )
        result = await self._call_reasoning(prompt)
        thought = result.get("thought", "Decomposing goal into steps")
        return thought, result

    async def _think_replan(
        self, goal: Goal, observation: str, context: Dict,
    ) -> Tuple[str, Dict]:
        """After failure: devise alternative approach."""
        wm = goal.working_memory
        failed_approaches = [
            {"actions": [s.get("action_type", "") for s in p.get("steps", [p])]}
            for p in wm.plan_history[-3:]
        ]

        prompt = (
            f"You are JARVIS. The previous approach failed. Devise an "
            f"ALTERNATIVE strategy that avoids the same mistakes.\n\n"
            f"Goal: {goal.description}\n"
            f"Observation: {observation}\n"
            f"Failed approaches: {json.dumps(failed_approaches)}\n"
            f"Blockers: {wm.blockers[-3:]}\n"
            f"Context: {json.dumps(context, default=str)[:1500]}\n\n"
            f"Return a JSON object with:\n"
            f"- 'thought': why previous approach failed and what to try instead\n"
            f"- 'description': what this step does\n"
            f"- 'action': dict describing the action\n"
            f"- 'action_type': str\n"
            f"- 'needs_vision': bool\n"
            f"- 'verification': 'visual'|'api_result'|'semantic'|'none'"
        )
        result = await self._call_reasoning(prompt)
        thought = result.get("thought", "Replanning after failure")
        return thought, result

    async def _think_next_step(
        self, goal: Goal, observation: str, context: Dict,
    ) -> Tuple[str, Dict]:
        """Standard continuation: what's next given current results?"""
        prompt = (
            f"You are JARVIS. Determine the next step for this goal.\n\n"
            f"Goal: {goal.description}\n"
            f"Observation: {observation}\n"
            f"Context: {json.dumps(context, default=str)[:2000]}\n\n"
            f"If the goal is complete, set action_type to 'complete'.\n\n"
            f"Return a JSON object with:\n"
            f"- 'thought': your reasoning about what to do next\n"
            f"- 'description': what this step does\n"
            f"- 'action': dict describing the action\n"
            f"- 'action_type': str\n"
            f"- 'needs_vision': bool\n"
            f"- 'verification': 'visual'|'api_result'|'semantic'|'none'"
        )
        result = await self._call_reasoning(prompt)
        thought = result.get("thought", "Determining next step")
        return thought, result

    async def _call_reasoning(self, prompt: str) -> Dict:
        """Call the reasoning engine (J-Prime or fallback)."""
        # Try J-Prime's AGI endpoint first
        try:
            result = await self._think_via_prime(prompt)
            if result and isinstance(result, dict):
                return result
        except Exception as e:
            logger.debug("[AgentRuntime] J-Prime reasoning failed: %s", e)

        # Fall back to the agent's reasoning engine
        try:
            if self._agent.reasoning_engine:
                state = await self._agent.reasoning_engine.run(
                    query=prompt,
                    context={"source": "agent_runtime", "mode": "structured"},
                )
                # Extract structured response from the reasoning result
                if hasattr(state, 'final_response') and state.final_response:
                    return self._parse_json_response(state.final_response)
                if hasattr(state, 'analysis_result') and state.analysis_result:
                    return state.analysis_result
        except Exception as e:
            logger.debug("[AgentRuntime] Reasoning engine failed: %s", e)

        # Minimal fallback
        return {
            "thought": "Proceeding with best guess",
            "description": "Continue working on goal",
            "action": {},
            "action_type": "continue",
            "needs_vision": False,
            "verification": "semantic",
        }

    async def _think_via_prime(self, prompt: str) -> Optional[Dict]:
        """Route thinking to J-Prime's /agi/reason endpoint."""
        try:
            from backend.core.prime_router import get_prime_router
            router = await get_prime_router()
            if router is None:
                return None
            result = await router.call_agi_endpoint("/agi/reason", {
                "query": prompt,
                "strategy": "chain_of_thought",
                "context": {"source": "agent_runtime"},
            })
            return result
        except (ImportError, AttributeError):
            return None

    def _parse_json_response(self, text: str) -> Dict:
        """Extract JSON from an LLM response that may contain markdown."""
        # Try direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting from markdown code block
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

        # Try finding JSON object in text
        for i, ch in enumerate(text):
            if ch == '{':
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(text[i:j+1])
                            except json.JSONDecodeError:
                                break
                break

        return {"thought": text[:200], "description": text[:100],
                "action": {}, "action_type": "unknown",
                "needs_vision": False, "verification": "semantic"}

    # ─────────────────────────────────────────────────────────
    # ACT Phase
    # ─────────────────────────────────────────────────────────

    async def _act(self, goal: Goal, step: GoalStep) -> Dict:
        """Execute a step's action using the tool orchestrator."""
        action = step.action
        if not action:
            return {"success": True, "message": "No action required"}

        try:
            result = await asyncio.wait_for(
                self._execute_action(action),
                timeout=self._act_timeout,
            )
            return result if isinstance(result, dict) else {"success": True, "result": result}
        except asyncio.TimeoutError:
            return {"success": False, "error": f"Action timed out ({self._act_timeout}s)"}
        except Exception as e:
            logger.error("[AgentRuntime] Action execution error: %s", e)
            return {"success": False, "error": str(e)}

    async def _execute_action(self, action: Dict) -> Dict:
        """Dispatch action to appropriate executor."""
        action_type = action.get("type", action.get("action_type", ""))

        # Tool execution via orchestrator
        tool_name = action.get("tool", action.get("tool_name", ""))
        if tool_name and self._agent.tool_orchestrator:
            result = await self._agent.tool_orchestrator.execute(
                tool_name=tool_name,
                **action.get("params", action.get("parameters", {})),
            )
            return {"success": True, "tool": tool_name, "result": result}

        # Reasoning-only action (no tool needed)
        if action_type in ("reason", "analyze", "plan"):
            return {"success": True, "message": f"Reasoning complete: {action_type}"}

        # Computer use action
        if action_type in ("click", "type", "scroll", "screenshot"):
            return await self._execute_computer_use(action)

        # Shell command
        if action_type == "shell" and action.get("command"):
            return await self._execute_shell(action["command"])

        # Default: pass through
        return {"success": True, "action_type": action_type, "action": action}

    async def _execute_computer_use(self, action: Dict) -> Dict:
        """Execute a computer use action (click, type, etc.)."""
        try:
            from backend.autonomy.computer_use_tool import get_computer_use_tool
            tool = await get_computer_use_tool()
            if tool:
                result = await tool.execute(action)
                return {"success": True, "result": result}
        except (ImportError, Exception) as e:
            return {"success": False, "error": f"Computer use unavailable: {e}"}
        return {"success": False, "error": "Computer use tool not found"}

    async def _execute_shell(self, command: str) -> Dict:
        """Execute a shell command safely."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            return {
                "success": proc.returncode == 0,
                "stdout": stdout.decode()[:2000] if stdout else "",
                "stderr": stderr.decode()[:2000] if stderr else "",
                "returncode": proc.returncode,
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": "Shell command timed out (30s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─────────────────────────────────────────────────────────
    # VERIFY Phase
    # ─────────────────────────────────────────────────────────

    async def _verify(self, goal: Goal, step: GoalStep, result: Dict) -> Dict:
        """Strategy-aware verification of step results."""
        strategy = step.verification_strategy

        if strategy == VerificationStrategy.NONE:
            return {
                "observation": "Deterministic step, no verification needed",
                "confidence": 1.0,
                "success": True,
            }

        elif strategy == VerificationStrategy.API_RESULT:
            return self._verify_api_result(step, result)

        elif strategy == VerificationStrategy.VISUAL:
            try:
                async with self._screen_lease.acquire(goal.goal_id, timeout=10.0):
                    return await self._verify_visual(step, result)
            except asyncio.TimeoutError:
                return {
                    "observation": "Could not acquire screen for visual verification",
                    "confidence": 0.5,
                    "success": result.get("success", False),
                }

        else:  # SEMANTIC — LLM-based
            return await self._verify_semantic(goal, step, result)

    def _verify_api_result(self, step: GoalStep, result: Dict) -> Dict:
        """Verify by checking API response structure."""
        success = result.get("success", False)
        error = result.get("error", "")
        return {
            "observation": f"API result: success={success}"
                          + (f", error={error}" if error else ""),
            "confidence": 0.95 if success else 0.1,
            "success": success,
        }

    async def _verify_visual(self, step: GoalStep, result: Dict) -> Dict:
        """Verify by capturing and analyzing screen state."""
        screen = await self._capture_screen_state()
        if not screen:
            return {
                "observation": "Visual verification failed: no screen capture",
                "confidence": 0.5,
                "success": result.get("success", False),
            }

        # Use LLM to assess if screen matches expected state
        prompt = (
            f"Did this step succeed based on the screen state?\n"
            f"Step: {step.description}\n"
            f"Screen: {screen}\n"
            f"Result: {json.dumps(result, default=str)[:500]}\n"
            f"Answer with JSON: {{'success': bool, 'confidence': float, 'observation': str}}"
        )
        assessment = await self._call_reasoning(prompt)
        return {
            "observation": assessment.get("observation", "Visual check complete"),
            "confidence": float(assessment.get("confidence", 0.5)),
            "success": bool(assessment.get("success", False)),
        }

    async def _verify_semantic(self, goal: Goal, step: GoalStep, result: Dict) -> Dict:
        """LLM-based semantic verification."""
        prompt = (
            f"Did this step contribute toward achieving the goal?\n"
            f"Goal: {goal.description}\n"
            f"Step: {step.description}\n"
            f"Result: {json.dumps(result, default=str)[:1000]}\n"
            f"Is the overall goal now complete?\n"
            f"Answer with JSON: {{'success': bool, 'confidence': float, "
            f"'observation': str, 'goal_complete': bool}}"
        )
        assessment = await self._call_reasoning(prompt)
        return {
            "observation": assessment.get("observation", "Semantic check complete"),
            "confidence": float(assessment.get("confidence", 0.5)),
            "success": bool(assessment.get("success", False)),
            "goal_complete": bool(assessment.get("goal_complete", False)),
        }

    # ─────────────────────────────────────────────────────────
    # REFLECT Phase
    # ─────────────────────────────────────────────────────────

    async def _reflect(self, goal: Goal, step: GoalStep,
                       verification: Dict) -> str:
        """Decide what to do next: continue, complete, replan, escalate, abandon."""
        # Quick checks before invoking LLM
        if verification.get("goal_complete", False) and step.confidence >= 0.8:
            return "complete"

        if step.status == "failed":
            # Check retry budget
            failed_count = goal.failed_step_count()
            if failed_count >= self._step_max_retries:
                return "abandon"
            return "replan"

        if step.confidence < 0.3:
            return "replan"

        if step.confidence >= 0.8 and verification.get("success", False):
            # Check if all planned steps are done
            if not goal.working_memory.planned_steps:
                # Ask LLM if goal is complete
                prompt = (
                    f"Is this goal complete?\n"
                    f"Goal: {goal.description}\n"
                    f"Steps taken: {len(goal.steps)}\n"
                    f"Last step: {step.description} (confidence: {step.confidence})\n"
                    f"Last observation: {step.observation}\n"
                    f"Answer: 'complete' or 'continue'"
                )
                result = await self._call_reasoning(prompt)
                decision = result.get("thought", "").lower()
                if "complete" in decision:
                    return "complete"

        return "continue"

    # ─────────────────────────────────────────────────────────
    # Escalation
    # ─────────────────────────────────────────────────────────

    def _assess_initial_escalation(
        self, description: str, source: str,
    ) -> EscalationLevel:
        """Determine initial escalation floor based on goal source and content."""
        if source == "proactive":
            return EscalationLevel.NOTIFY_AFTER

        desc_lower = description.lower()
        # Check for dangerous keywords
        if any(kw in desc_lower for kw in self._dangerous_actions):
            return EscalationLevel.ASK_BEFORE

        # Check for high-risk keywords
        if any(kw in desc_lower for kw in self._high_risk_actions):
            return EscalationLevel.NOTIFY_AFTER

        return EscalationLevel.AUTO_EXECUTE

    def _assess_step_escalation(self, step: GoalStep) -> EscalationLevel:
        """Dynamic per-step escalation assessment."""
        desc_lower = step.description.lower()
        action_str = json.dumps(step.action, default=str).lower()
        combined = desc_lower + " " + action_str

        if any(kw in combined for kw in self._dangerous_actions):
            return EscalationLevel.BLOCK_UNTIL_APPROVED

        if any(kw in combined for kw in self._high_risk_actions):
            return EscalationLevel.ASK_BEFORE

        if step.needs_vision:
            return EscalationLevel.NOTIFY_AFTER

        return EscalationLevel.AUTO_EXECUTE

    async def _escalate_to_human(self, goal: Goal, step: GoalStep):
        """Notify user that a goal needs approval."""
        event = {
            "type": "agent_escalation",
            "goal_id": goal.goal_id,
            "goal_description": goal.description,
            "step_description": step.description,
            "escalation_level": step.escalation_level.name if step.escalation_level else "UNKNOWN",
            "action": step.action,
            "timestamp": time.time(),
        }
        await self._emit_progress(goal, "escalation", step.description[:80])
        # Also broadcast via WebSocket
        await self._broadcast_ws(event)

    async def approve_escalation(self, goal_id: str, approved: bool = True):
        """User approves or rejects an escalated step."""
        if goal_id not in self._active_goals:
            return

        goal = self._active_goals[goal_id]
        if goal.status != GoalStatus.PAUSED:
            return

        if approved:
            goal.status = GoalStatus.ACTIVE
            # Re-spawn runner if it was cancelled
            if goal_id not in self._goal_runners or self._goal_runners[goal_id].done():
                runner = asyncio.create_task(
                    self._goal_runner(goal),
                    name=f"goal-runner-{goal_id}",
                )
                runner.add_done_callback(
                    lambda t, gid=goal_id: self._runner_done(gid, t)
                )
                self._goal_runners[goal_id] = runner
        else:
            goal.status = GoalStatus.CANCELLED
            goal.cancelled_reason = "User rejected escalation"
            await self._on_goal_complete(goal)

    # ─────────────────────────────────────────────────────────
    # Loop Detection
    # ─────────────────────────────────────────────────────────

    def _is_plan_repeating(self, new_plan: Dict, plan_history: List[Dict]) -> bool:
        """Detect if we're replanning in circles using action types."""
        new_actions = self._extract_action_types(new_plan)
        for old_plan in plan_history[-3:]:
            old_actions = self._extract_action_types(old_plan)
            if new_actions and new_actions == old_actions:
                return True
        return False

    def _extract_action_types(self, plan: Dict) -> List[str]:
        """Extract action types from a plan for comparison."""
        steps = plan.get("steps", plan.get("remaining_steps", []))
        if isinstance(steps, list):
            return [s.get("action_type", "") for s in steps if isinstance(s, dict)]
        return [plan.get("action_type", "")]

    # ─────────────────────────────────────────────────────────
    # Working Memory Summarization
    # ─────────────────────────────────────────────────────────

    async def _summarize(self, observations: List[str]) -> str:
        """Summarize a list of observations into a compact form."""
        combined = "\n".join(observations[:20])
        prompt = (
            f"Summarize these agent observations into 2-3 concise sentences "
            f"preserving key facts and decisions:\n\n{combined[:3000]}"
        )
        result = await self._call_reasoning(prompt)
        return result.get("thought", combined[:500])

    # ─────────────────────────────────────────────────────────
    # Progress & WebSocket Broadcasting
    # ─────────────────────────────────────────────────────────

    async def _emit_progress(self, goal: Goal, phase: str, detail: str):
        """Emit real-time progress to frontend via WebSocket."""
        event = {
            "type": "agent_progress",
            "goal_id": goal.goal_id,
            "phase": phase,
            "detail": detail,
            "step_count": len(goal.steps),
            "iteration": goal.working_memory.iteration_count,
            "status": goal.status.value,
            "confidence": goal.working_memory.last_confidence,
            "timestamp": time.time(),
        }

        # Call registered callbacks
        for callback in self._progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception:
                pass

        # Broadcast via WebSocket
        await self._broadcast_ws(event)

    async def _broadcast_ws(self, event: Dict):
        """Broadcast an event via the existing WebSocket infrastructure."""
        try:
            from backend.routers.hybrid import _broadcast_ws_event
            await _broadcast_ws_event(event)
        except (ImportError, Exception):
            pass  # Non-critical — UI just won't update

    # ─────────────────────────────────────────────────────────
    # Goal Completion & Learning
    # ─────────────────────────────────────────────────────────

    async def _on_goal_complete(self, goal: Goal):
        """Handle goal reaching a terminal state."""
        await self._checkpoint_store.save(goal)
        await self._data_bus.clear_goal(goal.goal_id)

        # Record trajectory for learning
        await self._record_goal_trajectory(goal)

        # Emit final status
        await self._emit_progress(
            goal, "terminal",
            f"Goal {goal.status.value}: {goal.description[:60]}",
        )

    async def _record_goal_trajectory(self, goal: Goal):
        """Record completed goal as training trajectory for Reactor-Core."""
        try:
            from backend.autonomy.unified_data_flywheel import get_data_flywheel
            flywheel = get_data_flywheel()
            if flywheel and hasattr(flywheel, 'record_goal_pursuit'):
                await flywheel.record_goal_pursuit(
                    goal_id=goal.goal_id,
                    description=goal.description,
                    steps=[s.to_dict() for s in goal.steps],
                    outcome=goal.status.value,
                    iterations=goal.working_memory.iteration_count,
                    total_time=(
                        (goal.completed_at or time.time()) - goal.created_at
                    ),
                )
        except Exception as e:
            logger.debug("[AgentRuntime] Failed to record trajectory: %s", e)

    # ─────────────────────────────────────────────────────────
    # Self-Directed Goal Generation
    # ─────────────────────────────────────────────────────────

    async def _maybe_generate_proactive_goal(self):
        """Check if the intervention engine suggests a proactive goal."""
        try:
            from backend.autonomy.intervention_decision_engine import (
                get_intervention_engine,
            )
            engine = get_intervention_engine()
            if engine and hasattr(engine, 'generate_goal'):
                goal_spec = await engine.generate_goal()
                if goal_spec:
                    await self.submit_goal(
                        description=goal_spec["description"],
                        priority=GoalPriority[
                            goal_spec.get("priority", "background").upper()
                        ],
                        source="proactive",
                        context=goal_spec.get("context"),
                    )
                    logger.info(
                        "[AgentRuntime] Proactive goal generated: %s",
                        goal_spec["description"][:60],
                    )
        except Exception as e:
            logger.debug("[AgentRuntime] Proactive goal generation failed: %s", e)


# ─────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────

_runtime_instance: Optional[UnifiedAgentRuntime] = None


def get_agent_runtime() -> Optional[UnifiedAgentRuntime]:
    """Get the global agent runtime instance (if started)."""
    return _runtime_instance


async def create_agent_runtime(autonomous_agent) -> UnifiedAgentRuntime:
    """Create and start the global agent runtime."""
    global _runtime_instance
    runtime = UnifiedAgentRuntime(autonomous_agent)
    await runtime.start()
    _runtime_instance = runtime
    return runtime
