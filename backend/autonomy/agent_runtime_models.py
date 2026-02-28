"""
Agent Runtime Data Models — Goal, GoalStep, WorkingMemory, Checkpoint,
EscalationLevel, ScreenLease, GoalDataBus.

Part of the Unified Agent Runtime that provides persistent outer-loop
goal pursuit for Ironcliw autonomous agent behavior.
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
)
from uuid import uuid4

import logging

logger = logging.getLogger("jarvis.agent_runtime.models")


# ─────────────────────────────────────────────────────────────
# Env-var helpers (consistent with Ironcliw patterns)
# ─────────────────────────────────────────────────────────────

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes")


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class EscalationLevel(IntEnum):
    """Risk-based escalation for goal steps."""
    AUTO_EXECUTE = 1          # Low risk, high confidence
    NOTIFY_AFTER = 2          # "I did X, FYI"
    ASK_BEFORE = 3            # "I want to do X, approve?"
    BLOCK_UNTIL_APPROVED = 4  # High-risk, needs explicit approval
    REFUSE = 5                # Safety constraint violation


class GoalStatus(str, Enum):
    """Lifecycle states for a goal."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"           # Waiting for human input
    BLOCKED = "blocked"         # Dependency not met
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    CANCELLED = "cancelled"     # User-initiated cancellation


class GoalPriority(IntEnum):
    """Priority levels for goal scheduling."""
    BACKGROUND = 1    # Self-generated
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ThinkMode(str, Enum):
    """Reasoning strategy selection for THINK phase."""
    DECOMPOSE = "decompose"     # First iteration: break goal into steps
    NEXT_STEP = "next_step"     # Subsequent: what's next given results?
    REPLAN = "replan"           # After failure: devise alternative approach


class VerificationStrategy(str, Enum):
    """How to verify a step succeeded."""
    VISUAL = "visual"           # Screen capture + vision analysis
    API_RESULT = "api_result"   # Check API response for success
    SEMANTIC = "semantic"       # LLM-based: "did this achieve the goal?"
    NONE = "none"               # Deterministic step, no verification needed


# Terminal states — goal will not be advanced further
TERMINAL_STATES: Set[GoalStatus] = {
    GoalStatus.COMPLETED,
    GoalStatus.FAILED,
    GoalStatus.ABANDONED,
    GoalStatus.CANCELLED,
}


# ─────────────────────────────────────────────────────────────
# Core Data Models
# ─────────────────────────────────────────────────────────────

@dataclass
class GoalStep:
    """A single step within a goal's execution plan."""
    step_id: str = field(default_factory=lambda: str(uuid4())[:12])
    description: str = ""
    status: str = "pending"         # pending, executing, completed, failed, skipped
    action: Dict[str, Any] = field(default_factory=dict)
    action_type: str = ""           # For replan loop detection
    result: Optional[Dict] = None
    observation: str = ""           # ReAct trace
    thought: str = ""               # ReAct trace
    confidence: float = 0.0
    attempt: int = 1
    needs_vision: bool = False      # Does this step need screen access?
    verification_strategy: VerificationStrategy = VerificationStrategy.SEMANTIC
    escalation_level: Optional[EscalationLevel] = None  # Per-step escalation
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verification_strategy"] = self.verification_strategy.value
        if self.escalation_level is not None:
            d["escalation_level"] = self.escalation_level.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalStep":
        vs = data.pop("verification_strategy", "semantic")
        el = data.pop("escalation_level", None)
        step = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        step.verification_strategy = VerificationStrategy(vs) if isinstance(vs, str) else VerificationStrategy.SEMANTIC
        if el is not None:
            step.escalation_level = EscalationLevel(el)
        return step


@dataclass
class WorkingMemory:
    """Per-goal scratchpad with memory budget and compaction."""
    goal_id: str = ""
    observations: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    context_accumulated: Dict[str, Any] = field(default_factory=dict)
    iteration_count: int = 0
    last_confidence: float = 0.0
    plan_history: List[Dict] = field(default_factory=list)
    planned_steps: List[Dict] = field(default_factory=list)  # Pre-planned steps
    needs_replan: bool = False

    # Memory budget (env-var configurable)
    MAX_OBSERVATIONS: ClassVar[int] = _env_int("AGENT_RUNTIME_MAX_OBSERVATIONS", 10)
    MAX_CONTEXT_BYTES: ClassVar[int] = _env_int("AGENT_RUNTIME_MAX_CONTEXT_BYTES", 50000)

    async def compact(self, summarize_fn: Callable[[List[str]], Awaitable[str]]):
        """Summarize old observations when memory exceeds budget."""
        if len(self.observations) <= self.MAX_OBSERVATIONS:
            return

        old = self.observations[:-self.MAX_OBSERVATIONS]
        try:
            summary = await summarize_fn(old)
            self.observations = [
                f"[Summary of {len(old)} prior observations]: {summary}"
            ] + self.observations[-self.MAX_OBSERVATIONS:]
        except Exception as e:
            logger.warning(f"[WorkingMemory] Compaction failed: {e}")
            # Fallback: just truncate without summarizing
            self.observations = self.observations[-self.MAX_OBSERVATIONS:]

    def context_size_bytes(self) -> int:
        """Approximate byte size of accumulated context."""
        try:
            return len(json.dumps(self.context_accumulated, default=str).encode())
        except Exception:
            return 0

    def trim_context_if_needed(self):
        """Remove oldest context entries if over budget."""
        while self.context_size_bytes() > self.MAX_CONTEXT_BYTES and self.context_accumulated:
            oldest_key = next(iter(self.context_accumulated))
            del self.context_accumulated[oldest_key]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemory":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Goal:
    """A high-level objective pursued by the agent runtime."""
    goal_id: str = field(default_factory=lambda: str(uuid4())[:12])
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.NORMAL
    source: str = "user"                # "user", "proactive", "scheduled", "chained"
    escalation_floor: EscalationLevel = EscalationLevel.AUTO_EXECUTE
    steps: List[GoalStep] = field(default_factory=list)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    parent_goal_id: Optional[str] = None
    max_iterations: int = 0             # 0 = use env var default
    max_duration_seconds: float = 0.0   # 0 = use env var default
    needs_vision: bool = False          # Does goal need screen access?
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    cancelled_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.working_memory.goal_id != self.goal_id:
            self.working_memory.goal_id = self.goal_id
        if self.max_iterations == 0:
            self.max_iterations = _env_int("AGENT_RUNTIME_MAX_ITERATIONS", 20)
        if self.max_duration_seconds == 0.0:
            self.max_duration_seconds = _env_float("AGENT_RUNTIME_MAX_DURATION", 600.0)

    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.time() - self.started_at

    def is_expired(self) -> bool:
        return self.elapsed_seconds() > self.max_duration_seconds

    def completed_step_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "completed")

    def failed_step_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "source": self.source,
            "escalation_floor": self.escalation_floor.value,
            "steps": [s.to_dict() for s in self.steps],
            "working_memory": self.working_memory.to_dict(),
            "parent_goal_id": self.parent_goal_id,
            "max_iterations": self.max_iterations,
            "max_duration_seconds": self.max_duration_seconds,
            "needs_vision": self.needs_vision,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "cancelled_reason": self.cancelled_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        steps_data = data.pop("steps", [])
        wm_data = data.pop("working_memory", {})
        status_val = data.pop("status", "pending")
        priority_val = data.pop("priority", 2)
        escalation_val = data.pop("escalation_floor", 1)

        goal = cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )
        goal.status = GoalStatus(status_val) if isinstance(status_val, str) else GoalStatus.PENDING
        goal.priority = GoalPriority(priority_val) if isinstance(priority_val, int) else GoalPriority.NORMAL
        goal.escalation_floor = EscalationLevel(escalation_val) if isinstance(escalation_val, int) else EscalationLevel.AUTO_EXECUTE
        goal.steps = [GoalStep.from_dict(s) for s in steps_data]
        goal.working_memory = WorkingMemory.from_dict(wm_data)
        return goal

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "Goal":
        return cls.from_dict(json.loads(json_str))


@dataclass
class RuntimeCheckpoint:
    """Checkpoint metadata for crash recovery."""
    schema_version: int = 1
    checkpoint_id: str = field(default_factory=lambda: str(uuid4())[:12])
    goal_id: str = ""
    goal_state_json: str = ""
    created_at: float = field(default_factory=time.time)
    kernel_pid: int = field(default_factory=os.getpid)


# ─────────────────────────────────────────────────────────────
# ScreenLease — mutual exclusion for screen access
# ─────────────────────────────────────────────────────────────

class ScreenLease:
    """Only one goal can own the screen at a time.

    Non-vision goals run freely in parallel. Vision goals must
    acquire the lease before taking screenshots or performing
    UI actions.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_holder: Optional[str] = None
        self._acquire_timeout = _env_float("AGENT_RUNTIME_SCREEN_LEASE_TIMEOUT", 30.0)

    @property
    def current_holder(self) -> Optional[str]:
        return self._current_holder

    @asynccontextmanager
    async def acquire(self, goal_id: str, timeout: Optional[float] = None):
        """Acquire screen lease. Only one holder at a time."""
        effective_timeout = timeout or self._acquire_timeout
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=effective_timeout)
            self._current_holder = goal_id
            logger.debug(f"[ScreenLease] Acquired by goal {goal_id}")
            yield
        except asyncio.TimeoutError:
            logger.warning(
                f"[ScreenLease] Goal {goal_id} timed out waiting "
                f"({effective_timeout}s). Current holder: {self._current_holder}"
            )
            raise
        finally:
            self._current_holder = None
            if self._lock.locked():
                self._lock.release()
                logger.debug(f"[ScreenLease] Released by goal {goal_id}")


# ─────────────────────────────────────────────────────────────
# GoalDataBus — publish/subscribe for inter-goal data sharing
# ─────────────────────────────────────────────────────────────

class GoalDataBus:
    """Publish/subscribe for intermediate results between concurrent goals.

    Allows Goal A to publish data that Goal B can wait for,
    enabling coordination without tight coupling.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._waiters: Dict[str, List[asyncio.Event]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._max_entries = _env_int("AGENT_RUNTIME_DATABUS_MAX_ENTRIES", 200)

    async def publish(self, key: str, data: Any):
        """Publish data under a key, waking any waiters."""
        async with self._lock:
            self._data[key] = data
            # Enforce max entries to prevent unbounded growth
            if len(self._data) > self._max_entries:
                oldest_key = next(iter(self._data))
                del self._data[oldest_key]
            waiters = self._waiters.pop(key, [])
        # Wake waiters outside lock
        for event in waiters:
            event.set()

    async def wait_for(self, key: str, timeout: float = 60.0) -> Optional[Any]:
        """Wait for data to be published under a key."""
        async with self._lock:
            if key in self._data:
                return self._data[key]
            event = asyncio.Event()
            self._waiters[key].append(event)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._data.get(key)
        except asyncio.TimeoutError:
            logger.debug(f"[GoalDataBus] Timeout waiting for key: {key}")
            return None
        finally:
            async with self._lock:
                waiters = self._waiters.get(key, [])
                if event in waiters:
                    waiters.remove(event)

    async def get(self, key: str) -> Optional[Any]:
        """Non-blocking get. Returns None if key not published."""
        return self._data.get(key)

    async def clear(self):
        """Clear all data and waiters."""
        async with self._lock:
            self._data.clear()
            self._waiters.clear()

    async def clear_goal(self, goal_id: str):
        """Clear all entries belonging to a specific goal."""
        async with self._lock:
            keys_to_remove = [k for k in self._data if k.startswith(f"{goal_id}:")]
            for k in keys_to_remove:
                del self._data[k]
                self._waiters.pop(k, None)
