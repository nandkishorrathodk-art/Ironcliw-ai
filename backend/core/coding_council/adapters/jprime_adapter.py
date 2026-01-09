"""
v84.0: JARVIS Prime Framework Adapter for Unified Coding Council
================================================================

Framework adapter that integrates JARVIS Prime local LLM engine with the
Unified Coding Council orchestrator. Routes coding and reasoning tasks to
local models (CodeLlama, Qwen, Llama) via JARVIS Prime's OpenAI-compatible API.

FEATURES:
    - Intelligent task routing (coding → CodeLlama, reasoning → Qwen)
    - Aider-style code editing via local models
    - Multi-agent planning (MetaGPT-style) via local models
    - Circuit breaker and fallback handling
    - Zero API cost for local inference
    - Seamless integration with DecisionRouter

USAGE:
    from backend.core.coding_council.adapters.jprime_adapter import (
        JPrimeCodingAdapter,
        JPrimeReasoningAdapter,
        JPrimeLocalAdapter,
    )

    # In orchestrator
    adapter = await self._get_adapter(FrameworkType.JPRIME_CODING)
    result = await adapter.execute(task, analysis, plan)

Author: JARVIS v84.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .jprime_engine import (
    JPrimeUnifiedEngine,
    JPrimeConfig,
    ModelTaskType,
    TaskClassifier,
    InferenceResult,
    CodeEditResult,
)

if TYPE_CHECKING:
    from ..types import (
        AnalysisResult,
        CodingCouncilConfig,
        EvolutionTask,
        FrameworkResult,
        FrameworkType,
        PlanResult,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions (Environment-Driven)
# =============================================================================

def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_env_bool(key: str, default: bool = False) -> bool:
    return _get_env(key, str(default)).lower() in ("true", "1", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(_get_env(key, str(default)))
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(_get_env(key, str(default)))
    except ValueError:
        return default


# =============================================================================
# Base J-Prime Adapter
# =============================================================================

class JPrimeBaseAdapter:
    """
    v84.0: Base adapter for JARVIS Prime integration.

    Provides common functionality for all J-Prime framework adapters.
    """

    def __init__(self, config: "CodingCouncilConfig"):
        self.config = config
        self.repo_root = config.repo_root
        self._engine: Optional[JPrimeUnifiedEngine] = None
        self._initialized = False
        self._available: Optional[bool] = None

        # Statistics
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "fallback_executions": 0,
            "total_tokens": 0,
            "total_execution_time_ms": 0.0,
        }

    async def _get_engine(self) -> JPrimeUnifiedEngine:
        """Get or initialize the J-Prime engine."""
        if self._engine is None:
            # Use config settings
            jprime_config = JPrimeConfig(
                base_url=self.config.jprime_url,
                coding_model=self.config.jprime_coding_model,
                reasoning_model=self.config.jprime_reasoning_model,
                general_model=self.config.jprime_general_model,
                request_timeout=self.config.jprime_timeout,
                fallback_to_claude=self.config.jprime_fallback_to_claude,
            )
            self._engine = JPrimeUnifiedEngine(jprime_config)
            await self._engine.initialize()
            self._initialized = True
            logger.info(f"[{self.__class__.__name__}] Engine initialized")

        return self._engine

    async def is_available(self) -> bool:
        """Check if J-Prime is available."""
        if self._available is not None:
            return self._available

        # Check if jprime is enabled in config
        if not self.config.jprime_enabled:
            self._available = False
            return False

        try:
            engine = await self._get_engine()
            self._available = await engine.is_available()
        except Exception as e:
            logger.debug(f"[{self.__class__.__name__}] Availability check failed: {e}")
            self._available = False

        return self._available

    async def close(self):
        """Close the adapter."""
        if self._engine:
            await self._engine.close()
            self._engine = None
            self._initialized = False

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        stats = self._stats.copy()
        stats["initialized"] = self._initialized
        stats["available"] = self._available
        if self._engine:
            stats["engine_stats"] = self._engine.get_stats()
        return stats

    def _build_framework_result(
        self,
        framework_type: "FrameworkType",
        success: bool,
        files_modified: List[str],
        changes_made: List[str],
        output: str = "",
        error: Optional[str] = None,
        execution_time_ms: float = 0.0,
    ) -> "FrameworkResult":
        """Build a FrameworkResult from execution results."""
        from ..types import FrameworkResult

        return FrameworkResult(
            framework=framework_type,
            success=success,
            files_modified=files_modified,
            changes_made=changes_made,
            output=output,
            error=error,
            execution_time_ms=execution_time_ms,
        )


# =============================================================================
# J-Prime Coding Adapter
# =============================================================================

class JPrimeCodingAdapter(JPrimeBaseAdapter):
    """
    v84.0: Adapter for J-Prime coding tasks.

    Routes coding, debugging, refactoring, and code review tasks to
    CodeLlama/DeepSeek Coder models via JARVIS Prime.

    Best for:
    - Code generation and completion
    - Bug fixes and debugging
    - Refactoring
    - Code review
    - Single/multi-file edits
    """

    SUPPORTED_TASK_TYPES = {
        ModelTaskType.CODING,
        ModelTaskType.DEBUGGING,
        ModelTaskType.REFACTORING,
        ModelTaskType.CODE_REVIEW,
    }

    async def execute(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"] = None,
        plan: Optional["PlanResult"] = None,
    ) -> "FrameworkResult":
        """
        Execute a coding task using J-Prime.

        Args:
            task: The evolution task to execute
            analysis: Optional codebase analysis
            plan: Optional execution plan

        Returns:
            FrameworkResult with execution results
        """
        from ..types import FrameworkType

        self._stats["total_executions"] += 1
        start_time = time.time()

        try:
            engine = await self._get_engine()

            # Check availability
            if not await engine.is_available():
                raise RuntimeError("J-Prime is not available for coding tasks")

            # Execute Aider-style code edit
            result: CodeEditResult = await engine.edit_code_aider(
                description=task.description,
                target_files=task.target_files,
                repo_path=self.repo_root,
                context_files=self._get_context_files(analysis),
            )

            execution_time = (time.time() - start_time) * 1000

            if result.success:
                self._stats["successful_executions"] += 1
                if result.inference_result:
                    self._stats["total_tokens"] += result.inference_result.tokens_used
                    if result.inference_result.fallback_used:
                        self._stats["fallback_executions"] += 1

                return self._build_framework_result(
                    framework_type=FrameworkType.JPRIME_CODING,
                    success=True,
                    files_modified=result.files_modified,
                    changes_made=result.changes_made,
                    output=f"Successfully modified {len(result.files_modified)} file(s)",
                    execution_time_ms=execution_time,
                )
            else:
                self._stats["failed_executions"] += 1
                return self._build_framework_result(
                    framework_type=FrameworkType.JPRIME_CODING,
                    success=False,
                    files_modified=[],
                    changes_made=[],
                    error=result.error or "Unknown error during code editing",
                    execution_time_ms=execution_time,
                )

        except Exception as e:
            self._stats["failed_executions"] += 1
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"[JPrimeCodingAdapter] Execution failed: {e}")

            return self._build_framework_result(
                framework_type=FrameworkType.JPRIME_CODING,
                success=False,
                files_modified=[],
                changes_made=[],
                error=str(e),
                execution_time_ms=execution_time,
            )

        finally:
            self._stats["total_execution_time_ms"] += (time.time() - start_time) * 1000

    def _get_context_files(self, analysis: Optional["AnalysisResult"]) -> Optional[List[str]]:
        """Extract context files from analysis results."""
        if not analysis:
            return None

        context_files = set()

        # Add dependency files
        for file, deps in analysis.dependencies.items():
            context_files.update(deps)

        # Limit context files to avoid overloading
        max_context = _get_env_int("JPRIME_MAX_CONTEXT_FILES", 10)
        return list(context_files)[:max_context] if context_files else None


# =============================================================================
# J-Prime Reasoning Adapter
# =============================================================================

class JPrimeReasoningAdapter(JPrimeBaseAdapter):
    """
    v84.0: Adapter for J-Prime reasoning tasks.

    Routes reasoning, planning, analysis, and math tasks to
    Qwen/Llama models via JARVIS Prime.

    Best for:
    - Architectural planning
    - Code analysis
    - Design decisions
    - Problem decomposition
    - Complex reasoning
    """

    SUPPORTED_TASK_TYPES = {
        ModelTaskType.REASONING,
        ModelTaskType.PLANNING,
        ModelTaskType.ANALYSIS,
        ModelTaskType.MATH,
    }

    async def execute(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"] = None,
        plan: Optional["PlanResult"] = None,
    ) -> "FrameworkResult":
        """
        Execute a reasoning/planning task using J-Prime.

        Args:
            task: The evolution task to execute
            analysis: Optional codebase analysis
            plan: Optional execution plan

        Returns:
            FrameworkResult with execution results
        """
        from ..types import FrameworkType

        self._stats["total_executions"] += 1
        start_time = time.time()

        try:
            engine = await self._get_engine()

            # Check availability
            if not await engine.is_available():
                raise RuntimeError("J-Prime is not available for reasoning tasks")

            # Execute multi-agent planning
            plan_result = await engine.plan_multi_agent(
                task=task.description,
                context={
                    "target_files": task.target_files,
                    "analysis": self._serialize_analysis(analysis) if analysis else None,
                },
            )

            execution_time = (time.time() - start_time) * 1000

            # Extract insights from agent outputs
            changes_made = []
            output_parts = []

            for agent, result in plan_result.items():
                if "output" in result:
                    output_parts.append(f"=== {agent} ===\n{result['output']}")
                    changes_made.append(f"Generated {agent} analysis")
                    if "tokens_used" in result:
                        self._stats["total_tokens"] += result["tokens_used"]

            self._stats["successful_executions"] += 1

            return self._build_framework_result(
                framework_type=FrameworkType.JPRIME_REASONING,
                success=True,
                files_modified=[],  # Reasoning doesn't directly modify files
                changes_made=changes_made,
                output="\n\n".join(output_parts),
                execution_time_ms=execution_time,
            )

        except Exception as e:
            self._stats["failed_executions"] += 1
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"[JPrimeReasoningAdapter] Execution failed: {e}")

            return self._build_framework_result(
                framework_type=FrameworkType.JPRIME_REASONING,
                success=False,
                files_modified=[],
                changes_made=[],
                error=str(e),
                execution_time_ms=execution_time,
            )

        finally:
            self._stats["total_execution_time_ms"] += (time.time() - start_time) * 1000

    def _serialize_analysis(self, analysis: "AnalysisResult") -> Dict[str, Any]:
        """Serialize analysis for context."""
        return {
            "target_files": analysis.target_files,
            "insights": analysis.insights,
            "suggestions": analysis.suggestions,
            "complexity_score": analysis.complexity_score,
            "risk_score": analysis.risk_score,
        }


# =============================================================================
# J-Prime Local Adapter (General Purpose)
# =============================================================================

class JPrimeLocalAdapter(JPrimeBaseAdapter):
    """
    v84.0: General-purpose adapter for J-Prime local LLM.

    Handles general tasks that don't fit into coding or reasoning
    categories. Uses the general-purpose model (Llama).

    Best for:
    - General queries
    - Documentation generation
    - Creative tasks
    - Simple Q&A
    """

    async def execute(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"] = None,
        plan: Optional["PlanResult"] = None,
    ) -> "FrameworkResult":
        """
        Execute a general task using J-Prime.

        Args:
            task: The evolution task to execute
            analysis: Optional codebase analysis
            plan: Optional execution plan

        Returns:
            FrameworkResult with execution results
        """
        from ..types import FrameworkType

        self._stats["total_executions"] += 1
        start_time = time.time()

        try:
            engine = await self._get_engine()

            # Check availability
            if not await engine.is_available():
                raise RuntimeError("J-Prime is not available")

            # Build context-aware prompt
            system_prompt = self._build_system_prompt(task, analysis)

            # Execute chat completion
            result: InferenceResult = await engine.chat(
                prompt=task.description,
                system_prompt=system_prompt,
            )

            execution_time = (time.time() - start_time) * 1000

            if result.success:
                self._stats["successful_executions"] += 1
                self._stats["total_tokens"] += result.tokens_used

                if result.fallback_used:
                    self._stats["fallback_executions"] += 1

                return self._build_framework_result(
                    framework_type=FrameworkType.JPRIME_LOCAL,
                    success=True,
                    files_modified=[],
                    changes_made=["Generated response"],
                    output=result.content,
                    execution_time_ms=execution_time,
                )
            else:
                self._stats["failed_executions"] += 1

                return self._build_framework_result(
                    framework_type=FrameworkType.JPRIME_LOCAL,
                    success=False,
                    files_modified=[],
                    changes_made=[],
                    error=result.error or "Unknown error",
                    execution_time_ms=execution_time,
                )

        except Exception as e:
            self._stats["failed_executions"] += 1
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"[JPrimeLocalAdapter] Execution failed: {e}")

            return self._build_framework_result(
                framework_type=FrameworkType.JPRIME_LOCAL,
                success=False,
                files_modified=[],
                changes_made=[],
                error=str(e),
                execution_time_ms=execution_time,
            )

        finally:
            self._stats["total_execution_time_ms"] += (time.time() - start_time) * 1000

    def _build_system_prompt(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"],
    ) -> str:
        """Build a context-aware system prompt."""
        prompt_parts = [
            "You are JARVIS Prime, an expert AI assistant for software development.",
            "Provide clear, accurate, and helpful responses.",
        ]

        if task.target_files:
            prompt_parts.append(f"Target files: {', '.join(task.target_files)}")

        if analysis:
            if analysis.insights:
                prompt_parts.append(f"Codebase insights: {', '.join(analysis.insights[:3])}")

        return "\n".join(prompt_parts)


# =============================================================================
# Availability Checker (for DecisionRouter)
# =============================================================================

class JPrimeAvailabilityChecker:
    """
    v84.0: Async availability checker for J-Prime.

    Used by DecisionRouter to determine if J-Prime can handle tasks.
    """

    _instance: Optional["JPrimeAvailabilityChecker"] = None
    _last_check_time: float = 0.0
    _check_interval: float = 30.0  # Re-check every 30 seconds
    _is_available: bool = False
    _health_data: Dict[str, Any] = {}

    @classmethod
    async def is_available(cls) -> bool:
        """Check if J-Prime is currently available."""
        now = time.time()

        # Return cached result if checked recently
        if now - cls._last_check_time < cls._check_interval:
            return cls._is_available

        cls._last_check_time = now

        try:
            # Check heartbeat file first (fastest)
            heartbeat_file = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_prime.json"

            if heartbeat_file.exists():
                import json
                with open(heartbeat_file) as f:
                    data = json.load(f)
                    heartbeat_age = now - data.get("timestamp", 0)

                    if heartbeat_age < 30:  # Fresh heartbeat
                        cls._is_available = True
                        cls._health_data = data
                        return True

            # Fallback to HTTP health check
            import aiohttp
            jprime_url = os.environ.get("JARVIS_PRIME_URL", "http://localhost:8000")

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{jprime_url}/health", timeout=5.0) as response:
                    cls._is_available = response.status == 200
                    return cls._is_available

        except Exception as e:
            logger.debug(f"[JPrimeAvailabilityChecker] Check failed: {e}")
            cls._is_available = False
            return False

    @classmethod
    def get_health_data(cls) -> Dict[str, Any]:
        """Get cached health data."""
        return cls._health_data

    @classmethod
    def reset(cls):
        """Reset cached availability."""
        cls._last_check_time = 0.0
        cls._is_available = False


# =============================================================================
# Task Classifier Export (for DecisionRouter)
# =============================================================================

def classify_task_for_jprime(
    description: str,
    target_files: Optional[List[str]] = None,
) -> tuple[ModelTaskType, float]:
    """
    Classify a task for J-Prime model selection.

    Returns:
        Tuple of (task_type, confidence_score)
    """
    return TaskClassifier.classify(description, target_files)


def is_task_suitable_for_jprime(
    description: str,
    target_files: Optional[List[str]] = None,
    confidence_threshold: float = 0.3,
) -> bool:
    """
    Determine if a task is suitable for J-Prime local processing.

    Tasks with coding, debugging, refactoring, or reasoning characteristics
    are suitable for J-Prime.
    """
    task_type, confidence = classify_task_for_jprime(description, target_files)

    # These task types are well-suited for J-Prime
    suitable_types = {
        ModelTaskType.CODING,
        ModelTaskType.DEBUGGING,
        ModelTaskType.REFACTORING,
        ModelTaskType.CODE_REVIEW,
        ModelTaskType.REASONING,
        ModelTaskType.PLANNING,
        ModelTaskType.ANALYSIS,
    }

    return task_type in suitable_types and confidence >= confidence_threshold


# =============================================================================
# Factory Functions
# =============================================================================

async def get_jprime_coding_adapter(config: "CodingCouncilConfig") -> JPrimeCodingAdapter:
    """Get an initialized J-Prime coding adapter."""
    adapter = JPrimeCodingAdapter(config)
    await adapter._get_engine()  # Pre-initialize
    return adapter


async def get_jprime_reasoning_adapter(config: "CodingCouncilConfig") -> JPrimeReasoningAdapter:
    """Get an initialized J-Prime reasoning adapter."""
    adapter = JPrimeReasoningAdapter(config)
    await adapter._get_engine()  # Pre-initialize
    return adapter


async def get_jprime_local_adapter(config: "CodingCouncilConfig") -> JPrimeLocalAdapter:
    """Get an initialized J-Prime local adapter."""
    adapter = JPrimeLocalAdapter(config)
    await adapter._get_engine()  # Pre-initialize
    return adapter


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Adapters
    "JPrimeCodingAdapter",
    "JPrimeReasoningAdapter",
    "JPrimeLocalAdapter",
    "JPrimeBaseAdapter",
    # Utilities
    "JPrimeAvailabilityChecker",
    "classify_task_for_jprime",
    "is_task_suitable_for_jprime",
    # Factories
    "get_jprime_coding_adapter",
    "get_jprime_reasoning_adapter",
    "get_jprime_local_adapter",
]
