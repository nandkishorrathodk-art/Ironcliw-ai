"""
JARVIS Agentic Task Runner - Core Module v2.0
==============================================

The unified agentic execution engine for JARVIS. This module provides:

- AgenticTaskRunner: Main orchestrator for Computer Use execution
- RunnerMode: Execution modes (direct, autonomous, supervised)
- AgenticTaskResult: Result dataclass for task execution

Integration:
    This module is designed to be instantiated and managed by the
    JARVISSupervisor (run_supervisor.py). The TieredCommandRouter
    routes Tier 2 commands to this runner for agentic execution.

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                     JARVISSupervisor                           │
    │  ┌──────────────┐   ┌──────────────────┐   ┌───────────────┐  │
    │  │   Tiered     │ → │  Agentic         │ → │   Computer    │  │
    │  │   Router     │   │  TaskRunner      │   │   Use Tool    │  │
    │  │   (Tier 2)   │   │                  │   │               │  │
    │  └──────────────┘   └────────┬─────────┘   └───────────────┘  │
    │                              │                                 │
    │                    ┌─────────▼─────────┐                       │
    │                    │    Watchdog       │                       │
    │                    │   (Safety)        │                       │
    │                    └───────────────────┘                       │
    └────────────────────────────────────────────────────────────────┘

Author: JARVIS AI System
Version: 2.0.0 (Unified)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AgenticRunnerConfig:
    """Configuration for the Agentic Task Runner."""

    # Execution settings
    default_mode: str = field(
        default_factory=lambda: os.getenv("JARVIS_AGENTIC_DEFAULT_MODE", "supervised")
    )
    max_actions_per_task: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_MAX_ACTIONS", "50"))
    )
    task_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_TASK_TIMEOUT", "300"))
    )

    # Component toggles
    uae_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_UAE_ENABLED", "true").lower() == "true"
    )
    neural_mesh_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_NEURAL_MESH_ENABLED", "true").lower() == "true"
    )
    learning_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_LEARNING_ENABLED", "true").lower() == "true"
    )

    # Narration
    narrate_by_default: bool = field(
        default_factory=lambda: os.getenv("JARVIS_NARRATE_TASKS", "true").lower() == "true"
    )

    # Watchdog integration
    watchdog_integration: bool = field(
        default_factory=lambda: os.getenv("JARVIS_WATCHDOG_ENABLED", "true").lower() == "true"
    )
    heartbeat_interval: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_HEARTBEAT_INTERVAL", "2.0"))
    )


# =============================================================================
# Enums
# =============================================================================

class RunnerMode(str, Enum):
    """Execution modes for the agentic task runner."""
    DIRECT = "direct"           # Computer Use only, no reasoning
    SUPERVISED = "supervised"   # With human checkpoints
    AUTONOMOUS = "autonomous"   # Full reasoning + execution


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class AgenticTaskResult:
    """Result from an agentic task execution."""
    success: bool
    goal: str
    mode: str
    execution_time_ms: float
    actions_count: int
    reasoning_steps: int
    final_message: str
    error: Optional[str] = None
    learning_insights: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    uae_used: bool = False
    neural_mesh_used: bool = False
    multi_space_used: bool = False
    watchdog_status: Optional[str] = None


# =============================================================================
# Component Availability Checks
# =============================================================================

def _check_component_availability() -> Dict[str, bool]:
    """Check which components are available."""
    availability = {}

    # Computer Use Tool
    try:
        from autonomy.computer_use_tool import ComputerUseTool
        availability["computer_use_tool"] = True
    except ImportError:
        availability["computer_use_tool"] = False

    # Direct Computer Use Connector
    try:
        from autonomy.claude_computer_use_connector import ClaudeComputerUseConnector
        availability["direct_connector"] = True
    except ImportError:
        availability["direct_connector"] = False

    # Autonomous Agent
    try:
        from autonomy.autonomous_agent import AutonomousAgent
        availability["autonomous_agent"] = True
    except ImportError:
        availability["autonomous_agent"] = False

    # UAE
    try:
        from unified_awareness.uae_core import UnifiedAwarenessEngine
        availability["uae"] = True
    except ImportError:
        availability["uae"] = False

    # Neural Mesh
    try:
        from neural_mesh.neural_mesh_coordinator import NeuralMeshCoordinator
        availability["neural_mesh"] = True
    except ImportError:
        availability["neural_mesh"] = False

    # Watchdog
    try:
        from core.agentic_watchdog import AgenticWatchdog
        availability["watchdog"] = True
    except ImportError:
        availability["watchdog"] = False

    return availability


# =============================================================================
# Agentic Task Runner
# =============================================================================

class AgenticTaskRunner:
    """
    Main orchestrator for agentic task execution.

    This class manages:
    - Computer Use tool for screen interactions
    - Autonomous Agent for reasoning (optional)
    - UAE for context awareness (optional)
    - Neural Mesh for multi-agent coordination (optional)
    - Watchdog for safety monitoring

    Designed to be instantiated by JARVISSupervisor and used by TieredCommandRouter.
    """

    def __init__(
        self,
        config: Optional[AgenticRunnerConfig] = None,
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        watchdog: Optional[Any] = None,  # Type hint as Any to avoid circular import
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the Agentic Task Runner.

        Args:
            config: Runner configuration
            tts_callback: Text-to-speech callback for narration
            watchdog: Pre-initialized watchdog instance (from supervisor)
            logger: Logger instance
        """
        self.config = config or AgenticRunnerConfig()
        self.tts_callback = tts_callback
        self._external_watchdog = watchdog  # Watchdog provided by supervisor
        self.logger = logger or logging.getLogger(__name__)

        # Components (lazy initialized)
        self._uae = None
        self._neural_mesh = None
        self._autonomous_agent = None
        self._computer_use_tool = None
        self._computer_use_connector = None
        self._watchdog = watchdog  # Use external watchdog if provided

        # Component availability
        self._availability = _check_component_availability()

        # State
        self._initialized = False
        self._tasks_executed = 0
        self._tasks_succeeded = 0
        self._current_task_id: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        self.logger.info("[AgenticRunner] Created")
        self._log_availability()

    def _log_availability(self):
        """Log component availability."""
        self.logger.info("[AgenticRunner] Component availability:")
        for name, available in self._availability.items():
            status = "✓" if available else "✗"
            self.logger.debug(f"  {status} {name}")

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self) -> bool:
        """
        Initialize all available components.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        self.logger.info("[AgenticRunner] Initializing...")

        try:
            # Initialize Computer Use Tool (required)
            if self._availability.get("computer_use_tool"):
                try:
                    from autonomy.computer_use_tool import get_computer_use_tool
                    self._computer_use_tool = get_computer_use_tool(
                        tts_callback=self.tts_callback,
                    )
                    self.logger.info("[AgenticRunner] ✓ Computer Use Tool")
                except Exception as e:
                    self.logger.warning(f"[AgenticRunner] ✗ Computer Use Tool: {e}")

            # Initialize Direct Connector (fallback)
            if self._availability.get("direct_connector") and not self._computer_use_tool:
                try:
                    from autonomy.claude_computer_use_connector import get_computer_use_connector
                    self._computer_use_connector = get_computer_use_connector(
                        tts_callback=self.tts_callback
                    )
                    self.logger.info("[AgenticRunner] ✓ Direct Connector (fallback)")
                except Exception as e:
                    self.logger.warning(f"[AgenticRunner] ✗ Direct Connector: {e}")

            # Initialize UAE (optional)
            if self._availability.get("uae") and self.config.uae_enabled:
                try:
                    from unified_awareness.uae_core import get_uae_engine
                    self._uae = get_uae_engine()
                    if not self._uae.is_active:
                        await self._uae.start()
                    self.logger.info("[AgenticRunner] ✓ UAE")
                except Exception as e:
                    self.logger.debug(f"[AgenticRunner] ✗ UAE: {e}")

            # Initialize Neural Mesh (optional)
            if self._availability.get("neural_mesh") and self.config.neural_mesh_enabled:
                try:
                    from neural_mesh.neural_mesh_coordinator import start_neural_mesh
                    self._neural_mesh = await start_neural_mesh()
                    self.logger.info("[AgenticRunner] ✓ Neural Mesh")
                except Exception as e:
                    self.logger.debug(f"[AgenticRunner] ✗ Neural Mesh: {e}")

            # Initialize Autonomous Agent (optional)
            if self._availability.get("autonomous_agent"):
                try:
                    from autonomy.autonomous_agent import (
                        AutonomousAgent, AgentConfig, AgentMode, AgentPersonality
                    )
                    agent_config = AgentConfig(
                        mode=AgentMode.SUPERVISED,
                        personality=AgentPersonality.HELPFUL,
                    )
                    self._autonomous_agent = AutonomousAgent(config=agent_config)
                    await self._autonomous_agent.initialize()
                    self.logger.info("[AgenticRunner] ✓ Autonomous Agent")
                except Exception as e:
                    self.logger.debug(f"[AgenticRunner] ✗ Autonomous Agent: {e}")

            # Initialize Watchdog if not provided externally
            if not self._watchdog and self._availability.get("watchdog") and self.config.watchdog_integration:
                try:
                    from core.agentic_watchdog import start_watchdog
                    self._watchdog = await start_watchdog(tts_callback=self.tts_callback)
                    self._watchdog.on_kill(self._on_watchdog_kill)
                    self.logger.info("[AgenticRunner] ✓ Watchdog (internal)")
                except Exception as e:
                    self.logger.debug(f"[AgenticRunner] ✗ Watchdog: {e}")

            # Verify we have at least one execution capability
            if not self._computer_use_tool and not self._computer_use_connector:
                self.logger.error("[AgenticRunner] No execution capability available!")
                return False

            self._initialized = True
            self.logger.info("[AgenticRunner] Initialization complete")
            return True

        except Exception as e:
            self.logger.error(f"[AgenticRunner] Initialization failed: {e}")
            return False

    # =========================================================================
    # Watchdog Integration
    # =========================================================================

    def set_watchdog(self, watchdog: Any) -> None:
        """Set the watchdog instance (called by supervisor)."""
        self._watchdog = watchdog
        if watchdog:
            watchdog.on_kill(self._on_watchdog_kill)
            self.logger.info("[AgenticRunner] Watchdog attached from supervisor")

    async def _on_watchdog_kill(self):
        """Called by watchdog when kill switch is triggered."""
        self.logger.warning("[AgenticRunner] Watchdog kill - stopping task")

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        self._current_task_id = None

    async def _heartbeat_loop(self, goal: str, mode: str):
        """Emit heartbeats to watchdog during task execution."""
        if not self._watchdog:
            return

        try:
            from core.agentic_watchdog import Heartbeat, AgenticMode

            actions_count = 0
            while True:
                await asyncio.sleep(self.config.heartbeat_interval)

                if not self._current_task_id:
                    break

                heartbeat = Heartbeat(
                    task_id=self._current_task_id,
                    goal=goal,
                    current_action=f"Executing ({mode})",
                    actions_count=actions_count,
                    timestamp=time.time(),
                    mode=AgenticMode.AUTONOMOUS if mode == "autonomous" else AgenticMode.SUPERVISED,
                )

                self._watchdog.receive_heartbeat(heartbeat)
                actions_count += 1

        except asyncio.CancelledError:
            pass
        except ImportError:
            pass

    async def _start_watchdog_task(self, goal: str, mode: str):
        """Start watchdog monitoring for this task."""
        if not self._watchdog:
            return

        try:
            from core.agentic_watchdog import AgenticMode

            self._current_task_id = f"task_{int(time.time())}_{id(self)}"

            watchdog_mode = AgenticMode.AUTONOMOUS if mode == "autonomous" else AgenticMode.SUPERVISED
            await self._watchdog.task_started(
                task_id=self._current_task_id,
                goal=goal,
                mode=watchdog_mode
            )

            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(goal, mode)
            )

            self.logger.debug(f"[AgenticRunner] Watchdog armed: {self._current_task_id}")
        except ImportError:
            pass

    async def _stop_watchdog_task(self, success: bool):
        """Stop watchdog monitoring for this task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._watchdog and self._current_task_id:
            await self._watchdog.task_completed(
                task_id=self._current_task_id,
                success=success
            )

        self._current_task_id = None

    # =========================================================================
    # Task Execution
    # =========================================================================

    async def run(
        self,
        goal: str,
        mode: Optional[RunnerMode] = None,
        context: Optional[Dict[str, Any]] = None,
        narrate: Optional[bool] = None,
    ) -> AgenticTaskResult:
        """
        Execute an agentic task.

        Args:
            goal: Natural language goal to achieve
            mode: Execution mode (defaults to config)
            context: Additional context
            narrate: Whether to enable voice narration

        Returns:
            AgenticTaskResult with execution details
        """
        if not self._initialized:
            await self.initialize()

        # Resolve defaults
        mode = mode or RunnerMode(self.config.default_mode)
        narrate = narrate if narrate is not None else self.config.narrate_by_default

        # Check watchdog permission
        if self._watchdog and not self._watchdog.is_agentic_allowed():
            return AgenticTaskResult(
                success=False,
                goal=goal,
                mode=mode.value,
                execution_time_ms=0,
                actions_count=0,
                reasoning_steps=0,
                final_message="Agentic execution blocked by watchdog safety system",
                error="Watchdog kill switch active or in cooldown",
                watchdog_status="blocked",
            )

        self._tasks_executed += 1
        start_time = time.time()

        self.logger.info(f"[AgenticRunner] Goal: {goal[:50]}...")
        self.logger.info(f"[AgenticRunner] Mode: {mode.value}")

        # Announce start
        if narrate and self.tts_callback:
            await self.tts_callback(f"Starting task: {goal[:50]}")

        # Start watchdog monitoring
        await self._start_watchdog_task(goal, mode.value)

        try:
            # Execute based on mode
            if mode == RunnerMode.DIRECT:
                result = await self._execute_direct(goal, context, narrate)
            elif mode == RunnerMode.AUTONOMOUS:
                result = await self._execute_autonomous(goal, context, narrate)
            else:  # SUPERVISED
                result = await self._execute_supervised(goal, context, narrate)

            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            result.mode = mode.value

            if result.success:
                self._tasks_succeeded += 1

            # Stop watchdog monitoring
            await self._stop_watchdog_task(result.success)

            # Announce completion
            if narrate and self.tts_callback:
                status = "completed successfully" if result.success else "encountered an issue"
                await self.tts_callback(f"Task {status}")

            self.logger.info(f"[AgenticRunner] Complete: success={result.success}, time={execution_time:.0f}ms")
            return result

        except asyncio.TimeoutError:
            await self._stop_watchdog_task(False)
            return AgenticTaskResult(
                success=False,
                goal=goal,
                mode=mode.value,
                execution_time_ms=(time.time() - start_time) * 1000,
                actions_count=0,
                reasoning_steps=0,
                final_message="Task timed out",
                error=f"Timeout after {self.config.task_timeout_seconds}s",
            )

        except Exception as e:
            self.logger.error(f"[AgenticRunner] Failed: {e}", exc_info=True)
            await self._stop_watchdog_task(False)

            return AgenticTaskResult(
                success=False,
                goal=goal,
                mode=mode.value,
                execution_time_ms=(time.time() - start_time) * 1000,
                actions_count=0,
                reasoning_steps=0,
                final_message=f"Task failed: {str(e)}",
                error=str(e),
            )

    # =========================================================================
    # Execution Modes
    # =========================================================================

    async def _execute_direct(
        self,
        goal: str,
        context: Optional[Dict[str, Any]],
        narrate: bool,
    ) -> AgenticTaskResult:
        """Execute goal directly via Computer Use (skip reasoning)."""
        self.logger.debug("[AgenticRunner] DIRECT mode")

        context = context or {}

        # Add UAE context if available
        uae_used = False
        if self._uae:
            try:
                context["uae_active"] = True
                uae_used = True
            except Exception as e:
                self.logger.debug(f"UAE context error: {e}")

        # Use Computer Use Tool
        if self._computer_use_tool:
            result = await self._computer_use_tool.run(
                goal=goal,
                context=context,
                narrate=narrate,
            )
            return AgenticTaskResult(
                success=result.success,
                goal=goal,
                mode="direct",
                execution_time_ms=result.total_duration_ms,
                actions_count=result.actions_count,
                reasoning_steps=0,
                final_message=result.final_message,
                learning_insights=result.learning_insights if hasattr(result, 'learning_insights') else [],
                uae_used=uae_used,
                multi_space_used=hasattr(result, 'multi_space_context') and result.multi_space_context is not None,
            )

        # Fallback to direct connector
        if self._computer_use_connector:
            result = await self._computer_use_connector.execute_task(
                goal=goal,
                context=context,
                narrate=narrate,
            )
            # Handle different result structures
            success = getattr(result, 'success', False) or (hasattr(result, 'status') and str(result.status) == "SUCCESS")
            return AgenticTaskResult(
                success=success,
                goal=goal,
                mode="direct",
                execution_time_ms=getattr(result, 'total_duration_ms', 0),
                actions_count=len(getattr(result, 'actions_executed', [])),
                reasoning_steps=0,
                final_message=getattr(result, 'final_message', "Task completed"),
                learning_insights=getattr(result, 'learning_insights', []),
                uae_used=uae_used,
            )

        raise RuntimeError("No computer use capability available")

    async def _execute_autonomous(
        self,
        goal: str,
        context: Optional[Dict[str, Any]],
        narrate: bool,
    ) -> AgenticTaskResult:
        """Execute goal with full autonomous reasoning + execution."""
        self.logger.debug("[AgenticRunner] AUTONOMOUS mode")

        context = context or {}
        reasoning_steps = 0

        # Phase 1: Autonomous planning (if agent available)
        if self._autonomous_agent:
            try:
                self.logger.debug("[AgenticRunner] Phase 1: Planning...")

                if hasattr(self._autonomous_agent, 'analyze_goal'):
                    plan_result = await self._autonomous_agent.analyze_goal(goal, context)
                    if plan_result:
                        reasoning_steps = plan_result.get("reasoning_steps", 0)
                        context["autonomous_plan"] = plan_result.get("plan", [])
                        context["goal_analysis"] = plan_result.get("analysis", "")
            except Exception as e:
                self.logger.debug(f"[AgenticRunner] Planning failed: {e}")

        # Phase 2: Execute via Computer Use
        self.logger.debug("[AgenticRunner] Phase 2: Execution...")
        context["execution_mode"] = "autonomous"
        context["full_reasoning"] = True

        direct_result = await self._execute_direct(goal, context, narrate)

        # Update result with autonomous metadata
        direct_result.mode = "autonomous"
        direct_result.reasoning_steps = reasoning_steps
        direct_result.neural_mesh_used = self._neural_mesh is not None

        # Phase 3: Record learning (if enabled)
        if self.config.learning_enabled and self._neural_mesh and direct_result.success:
            try:
                await self._record_learning(goal, direct_result)
            except Exception as e:
                self.logger.debug(f"[AgenticRunner] Learning failed: {e}")

        return direct_result

    async def _execute_supervised(
        self,
        goal: str,
        context: Optional[Dict[str, Any]],
        narrate: bool,
    ) -> AgenticTaskResult:
        """Execute goal with supervision (may request confirmation)."""
        self.logger.debug("[AgenticRunner] SUPERVISED mode")

        # For now, supervised behaves like direct with logging
        # In full implementation, this would pause for user confirmation
        return await self._execute_direct(goal, context, narrate)

    async def _record_learning(self, goal: str, result: AgenticTaskResult):
        """Record successful execution for future learning."""
        if not self._neural_mesh:
            return

        try:
            knowledge = {
                "goal": goal,
                "mode": result.mode,
                "actions_count": result.actions_count,
                "execution_time_ms": result.execution_time_ms,
                "success": result.success,
                "timestamp": datetime.now().isoformat(),
            }

            if hasattr(self._neural_mesh, 'knowledge_graph'):
                kg = self._neural_mesh.knowledge_graph
                if hasattr(kg, 'add_fact'):
                    await kg.add_fact(
                        subject=goal,
                        predicate="executed_successfully",
                        object_=f"{result.actions_count} actions",
                        metadata=knowledge
                    )
        except Exception as e:
            self.logger.debug(f"Learning recording error: {e}")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def shutdown(self):
        """Gracefully shutdown all components."""
        self.logger.info("[AgenticRunner] Shutting down...")

        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Don't stop external watchdog (supervisor manages it)
        if self._watchdog and not self._external_watchdog:
            try:
                from core.agentic_watchdog import stop_watchdog
                await stop_watchdog()
            except Exception:
                pass

        # Stop UAE
        if self._uae:
            try:
                if self._uae.is_active:
                    await self._uae.stop()
            except Exception:
                pass

        # Stop Neural Mesh
        if self._neural_mesh:
            try:
                from neural_mesh.neural_mesh_coordinator import stop_neural_mesh
                await stop_neural_mesh()
            except Exception:
                pass

        self.logger.info("[AgenticRunner] Shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get runner statistics."""
        watchdog_status = None
        if self._watchdog:
            try:
                status = self._watchdog.get_status()
                watchdog_status = {
                    "mode": status.mode.value,
                    "kill_switch_armed": status.kill_switch_armed,
                    "heartbeat_healthy": status.heartbeat_healthy,
                    "uptime_seconds": status.uptime_seconds,
                }
            except Exception:
                watchdog_status = {"status": "available"}

        return {
            "initialized": self._initialized,
            "tasks_executed": self._tasks_executed,
            "tasks_succeeded": self._tasks_succeeded,
            "success_rate": (
                self._tasks_succeeded / self._tasks_executed
                if self._tasks_executed > 0 else 0.0
            ),
            "current_task": self._current_task_id,
            "watchdog": watchdog_status,
            "components": {
                "uae": self._uae is not None,
                "neural_mesh": self._neural_mesh is not None,
                "autonomous_agent": self._autonomous_agent is not None,
                "computer_use_tool": self._computer_use_tool is not None,
                "direct_connector": self._computer_use_connector is not None,
            },
            "availability": self._availability,
        }

    @property
    def is_ready(self) -> bool:
        """Check if runner is ready to execute tasks."""
        return self._initialized and (self._computer_use_tool is not None or self._computer_use_connector is not None)


# =============================================================================
# Singleton Access (for backward compatibility)
# =============================================================================

_runner_instance: Optional[AgenticTaskRunner] = None


def get_agentic_runner() -> Optional[AgenticTaskRunner]:
    """Get the global runner instance (if set)."""
    return _runner_instance


def set_agentic_runner(runner: AgenticTaskRunner):
    """Set the global runner instance."""
    global _runner_instance
    _runner_instance = runner


async def create_agentic_runner(
    config: Optional[AgenticRunnerConfig] = None,
    tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    watchdog: Optional[Any] = None,
) -> AgenticTaskRunner:
    """Create and initialize an agentic runner."""
    runner = AgenticTaskRunner(
        config=config,
        tts_callback=tts_callback,
        watchdog=watchdog,
    )
    await runner.initialize()
    return runner
