"""
v77.1: Partial Success Handler - Gap #41
=========================================

Handles cases where some frameworks succeed and others fail.

Problem:
    - MetaGPT: ✅ Plan generated
    - RepoMaster: ✅ Analysis complete
    - Aider: ❌ Execution failed

    What do we do? The plan is valuable, analysis is useful, but
    execution failed. We shouldn't throw everything away.

Solution:
    - Track success/failure per framework
    - Keep successful outputs for retry
    - Merge partial results intelligently
    - Provide recovery strategies

Features:
    - Framework outcome tracking
    - Intelligent result merging
    - Retry with context preservation
    - Fallback framework selection
    - Progressive enhancement

Author: JARVIS v77.1
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OutcomeStatus(Enum):
    """Status of a framework outcome."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"  # Produced some output but with issues
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class RecoveryAction(Enum):
    """Actions to take for recovery."""
    RETRY_SAME = "retry_same"  # Retry the same framework
    RETRY_DIFFERENT = "retry_different"  # Try a different framework
    USE_PARTIAL = "use_partial"  # Use partial results
    SKIP = "skip"  # Skip and continue
    ABORT = "abort"  # Abort the entire evolution


@dataclass
class FrameworkOutcome:
    """
    Outcome of a framework execution.

    Captures both success/failure and any partial outputs.
    """
    framework: str
    status: OutcomeStatus
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    # Outputs (may be partial even on failure)
    plan: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    changes: Optional[List[Dict[str, Any]]] = None
    files_modified: List[str] = field(default_factory=list)
    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    # Quality metrics
    confidence: float = 0.0
    completeness: float = 0.0  # 0-1, how complete the output is

    @property
    def duration_ms(self) -> float:
        if not self.completed_at:
            return 0.0
        return (self.completed_at - self.started_at) * 1000

    @property
    def has_useful_output(self) -> bool:
        """Check if there's any useful partial output."""
        return bool(self.plan or self.analysis or self.changes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "has_useful_output": self.has_useful_output,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "error": self.error,
            "files_modified": self.files_modified,
        }


@dataclass
class RecoveryStrategy:
    """
    Strategy for recovering from partial failure.

    Defines what to do when some frameworks fail.
    """
    action: RecoveryAction
    framework: Optional[str] = None  # For RETRY_DIFFERENT
    reason: str = ""
    preserve_outputs: List[str] = field(default_factory=list)  # Frameworks to keep outputs from
    max_retries: int = 2
    timeout_multiplier: float = 1.5  # Increase timeout on retry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "framework": self.framework,
            "reason": self.reason,
            "preserve_outputs": self.preserve_outputs,
            "max_retries": self.max_retries,
        }


@dataclass
class MergeResult:
    """
    Result of merging partial framework outputs.

    Combines successful outputs from multiple frameworks.
    """
    success: bool
    merged_plan: Optional[str] = None
    merged_analysis: Optional[Dict[str, Any]] = None
    merged_changes: List[Dict[str, Any]] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    frameworks_used: List[str] = field(default_factory=list)
    frameworks_failed: List[str] = field(default_factory=list)
    confidence: float = 0.0
    completeness: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "has_plan": self.merged_plan is not None,
            "has_analysis": self.merged_analysis is not None,
            "change_count": len(self.merged_changes),
            "files_modified": self.files_modified,
            "frameworks_used": self.frameworks_used,
            "frameworks_failed": self.frameworks_failed,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "warnings": self.warnings,
        }


class RecoveryPlanner:
    """
    Plans recovery actions based on failure patterns.

    Uses heuristics and learning to choose optimal recovery.
    """

    # Framework priority for retries (higher = try first)
    FRAMEWORK_PRIORITY = {
        "aider": 90,
        "continue": 80,
        "openhands": 70,
        "repomaster": 60,
        "metagpt": 50,
    }

    # Framework capabilities for fallback selection
    FRAMEWORK_CAPABILITIES = {
        "aider": {"code_edit", "git_aware", "context_rich"},
        "continue": {"code_edit", "ide_integration"},
        "openhands": {"code_edit", "browsing", "terminal"},
        "repomaster": {"analysis", "planning", "refactoring"},
        "metagpt": {"planning", "multi_agent", "design"},
    }

    @classmethod
    def plan_recovery(
        cls,
        outcomes: Dict[str, FrameworkOutcome],
        available_frameworks: List[str],
        task_requirements: Optional[set] = None,
    ) -> RecoveryStrategy:
        """
        Plan recovery action based on outcomes.

        Args:
            outcomes: Dict of framework -> outcome
            available_frameworks: Frameworks that can be used
            task_requirements: Required capabilities for the task

        Returns:
            RecoveryStrategy with recommended action
        """
        # Count successes and failures
        successes = [f for f, o in outcomes.items() if o.status == OutcomeStatus.SUCCESS]
        failures = [f for f, o in outcomes.items() if o.status in (OutcomeStatus.FAILED, OutcomeStatus.TIMEOUT)]
        partials = [f for f, o in outcomes.items() if o.status == OutcomeStatus.PARTIAL]

        # If any full success, we can continue
        if successes:
            return RecoveryStrategy(
                action=RecoveryAction.USE_PARTIAL,
                reason=f"Using successful results from: {', '.join(successes)}",
                preserve_outputs=successes,
            )

        # If partial successes exist, evaluate if usable
        if partials:
            useful_partials = [
                f for f in partials
                if outcomes[f].has_useful_output and outcomes[f].completeness >= 0.5
            ]
            if useful_partials:
                return RecoveryStrategy(
                    action=RecoveryAction.USE_PARTIAL,
                    reason=f"Using partial results from: {', '.join(useful_partials)}",
                    preserve_outputs=useful_partials,
                )

        # All failed - need to retry
        for failed in failures:
            outcome = outcomes[failed]

            # Check retry count
            if outcome.retry_count >= 2:
                continue

            # Check if it's worth retrying
            if outcome.error_type == "timeout":
                return RecoveryStrategy(
                    action=RecoveryAction.RETRY_SAME,
                    framework=failed,
                    reason=f"Retrying {failed} with increased timeout",
                    timeout_multiplier=2.0,
                    max_retries=1,
                )

        # Try a different framework
        tried = set(outcomes.keys())
        untried = [f for f in available_frameworks if f not in tried]

        if untried:
            # Select best untried framework
            if task_requirements:
                # Match capabilities
                for framework in sorted(
                    untried,
                    key=lambda f: len(cls.FRAMEWORK_CAPABILITIES.get(f, set()) & task_requirements),
                    reverse=True,
                ):
                    return RecoveryStrategy(
                        action=RecoveryAction.RETRY_DIFFERENT,
                        framework=framework,
                        reason=f"Trying {framework} as fallback",
                        preserve_outputs=list(tried),
                    )
            else:
                # Use priority
                best = max(untried, key=lambda f: cls.FRAMEWORK_PRIORITY.get(f, 0))
                return RecoveryStrategy(
                    action=RecoveryAction.RETRY_DIFFERENT,
                    framework=best,
                    reason=f"Trying {best} as fallback (highest priority)",
                    preserve_outputs=list(tried),
                )

        # No options left
        return RecoveryStrategy(
            action=RecoveryAction.ABORT,
            reason=f"All frameworks failed: {', '.join(failures)}",
        )


class OutputMerger:
    """
    Merges outputs from multiple frameworks.

    Handles combining partial results intelligently.
    """

    @classmethod
    async def merge(
        cls,
        outcomes: Dict[str, FrameworkOutcome],
        priority_order: Optional[List[str]] = None,
    ) -> MergeResult:
        """
        Merge outputs from multiple framework outcomes.

        Args:
            outcomes: Dict of framework -> outcome
            priority_order: Framework priority for conflicts

        Returns:
            MergeResult with combined outputs
        """
        result = MergeResult(success=False)

        # Default priority order
        if not priority_order:
            priority_order = sorted(
                outcomes.keys(),
                key=lambda f: RecoveryPlanner.FRAMEWORK_PRIORITY.get(f, 0),
                reverse=True,
            )

        # Collect successful/partial outcomes
        usable = {
            f: o for f, o in outcomes.items()
            if o.status in (OutcomeStatus.SUCCESS, OutcomeStatus.PARTIAL) and o.has_useful_output
        }

        if not usable:
            result.frameworks_failed = list(outcomes.keys())
            return result

        # Merge plans (take highest confidence)
        plans = [(f, o.plan, o.confidence) for f, o in usable.items() if o.plan]
        if plans:
            best_plan = max(plans, key=lambda x: x[2])
            result.merged_plan = best_plan[1]
            result.frameworks_used.append(best_plan[0])

        # Merge analyses (combine with priority)
        result.merged_analysis = {}
        for framework in priority_order:
            if framework in usable and usable[framework].analysis:
                # Later (lower priority) frameworks don't overwrite
                for key, value in usable[framework].analysis.items():
                    if key not in result.merged_analysis:
                        result.merged_analysis[key] = value
                if framework not in result.frameworks_used:
                    result.frameworks_used.append(framework)

        # Merge changes (deduplicate by file)
        seen_files: Dict[str, Dict[str, Any]] = {}
        for framework in priority_order:
            if framework in usable and usable[framework].changes:
                for change in usable[framework].changes:
                    file_path = change.get("file", change.get("path", ""))
                    if file_path and file_path not in seen_files:
                        seen_files[file_path] = change
                        if framework not in result.frameworks_used:
                            result.frameworks_used.append(framework)

        result.merged_changes = list(seen_files.values())
        result.files_modified = list(seen_files.keys())

        # Calculate combined metrics
        if usable:
            result.confidence = sum(o.confidence for o in usable.values()) / len(usable)
            result.completeness = max(o.completeness for o in usable.values())

        result.frameworks_failed = [
            f for f in outcomes.keys()
            if f not in usable
        ]

        # Add warnings for partial merges
        if result.frameworks_failed:
            result.warnings.append(
                f"Some frameworks failed: {', '.join(result.frameworks_failed)}"
            )

        if len(result.frameworks_used) > 1:
            result.warnings.append(
                f"Merged outputs from multiple frameworks: {', '.join(result.frameworks_used)}"
            )

        result.success = bool(result.merged_plan or result.merged_changes)
        return result


class PartialSuccessHandler:
    """
    Handles partial success scenarios in evolution.

    Features:
    - Tracks all framework outcomes
    - Plans recovery strategies
    - Merges partial outputs
    - Manages retries

    Usage:
        handler = PartialSuccessHandler()

        # Record outcomes
        handler.record_outcome("aider", OutcomeStatus.SUCCESS, plan="...")
        handler.record_outcome("metagpt", OutcomeStatus.FAILED, error="...")

        # Get merged result
        merged = await handler.get_merged_result()

        # If needed, get recovery strategy
        strategy = handler.get_recovery_strategy()
    """

    def __init__(
        self,
        available_frameworks: Optional[List[str]] = None,
        task_requirements: Optional[set] = None,
    ):
        self.available_frameworks = available_frameworks or [
            "aider", "continue", "openhands", "repomaster", "metagpt"
        ]
        self.task_requirements = task_requirements

        self._outcomes: Dict[str, FrameworkOutcome] = {}
        self._retry_counts: Dict[str, int] = {}

    def record_outcome(
        self,
        framework: str,
        status: OutcomeStatus,
        plan: Optional[str] = None,
        analysis: Optional[Dict[str, Any]] = None,
        changes: Optional[List[Dict[str, Any]]] = None,
        files_modified: Optional[List[str]] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        confidence: float = 1.0,
        completeness: float = 1.0,
    ) -> FrameworkOutcome:
        """
        Record an outcome from a framework.

        Args:
            framework: Framework name
            status: Outcome status
            plan: Generated plan (if any)
            analysis: Analysis results (if any)
            changes: Code changes (if any)
            files_modified: List of modified files
            error: Error message (if failed)
            error_type: Type of error
            confidence: Confidence score (0-1)
            completeness: Completeness score (0-1)

        Returns:
            The recorded FrameworkOutcome
        """
        retry_count = self._retry_counts.get(framework, 0)

        outcome = FrameworkOutcome(
            framework=framework,
            status=status,
            completed_at=time.time(),
            plan=plan,
            analysis=analysis,
            changes=changes,
            files_modified=files_modified or [],
            error=error,
            error_type=error_type,
            retry_count=retry_count,
            confidence=confidence,
            completeness=completeness,
        )

        self._outcomes[framework] = outcome

        logger.info(
            f"[PartialSuccess] {framework}: {status.value} "
            f"(confidence={confidence:.2f}, completeness={completeness:.2f})"
        )

        return outcome

    def record_retry(self, framework: str) -> None:
        """Record a retry attempt."""
        self._retry_counts[framework] = self._retry_counts.get(framework, 0) + 1

    async def get_merged_result(
        self,
        priority_order: Optional[List[str]] = None,
    ) -> MergeResult:
        """
        Get merged result from all outcomes.

        Args:
            priority_order: Framework priority for conflicts

        Returns:
            MergeResult with combined outputs
        """
        return await OutputMerger.merge(self._outcomes, priority_order)

    def get_recovery_strategy(self) -> RecoveryStrategy:
        """
        Get recommended recovery strategy.

        Returns:
            RecoveryStrategy with action and details
        """
        return RecoveryPlanner.plan_recovery(
            outcomes=self._outcomes,
            available_frameworks=self.available_frameworks,
            task_requirements=self.task_requirements,
        )

    def get_successful_frameworks(self) -> List[str]:
        """Get list of frameworks that succeeded."""
        return [
            f for f, o in self._outcomes.items()
            if o.status == OutcomeStatus.SUCCESS
        ]

    def get_failed_frameworks(self) -> List[str]:
        """Get list of frameworks that failed."""
        return [
            f for f, o in self._outcomes.items()
            if o.status in (OutcomeStatus.FAILED, OutcomeStatus.TIMEOUT)
        ]

    def has_any_success(self) -> bool:
        """Check if any framework succeeded."""
        return any(
            o.status == OutcomeStatus.SUCCESS
            for o in self._outcomes.values()
        )

    def has_useful_output(self) -> bool:
        """Check if there's any useful output to merge."""
        return any(
            o.has_useful_output
            for o in self._outcomes.values()
        )

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all outcomes."""
        return {
            "total_frameworks": len(self._outcomes),
            "succeeded": len(self.get_successful_frameworks()),
            "failed": len(self.get_failed_frameworks()),
            "has_useful_output": self.has_useful_output(),
            "outcomes": {f: o.to_dict() for f, o in self._outcomes.items()},
        }

    def reset(self) -> None:
        """Reset all outcomes."""
        self._outcomes.clear()
        self._retry_counts.clear()


# Convenience function for creating handlers
def create_partial_success_handler(
    task_description: str,
    available_frameworks: Optional[List[str]] = None,
) -> PartialSuccessHandler:
    """
    Create a PartialSuccessHandler with appropriate requirements.

    Args:
        task_description: Description of the evolution task
        available_frameworks: Available frameworks

    Returns:
        Configured PartialSuccessHandler
    """
    # Infer requirements from task description
    requirements = set()

    keywords = task_description.lower()

    if any(w in keywords for w in ["edit", "modify", "change", "update", "fix"]):
        requirements.add("code_edit")

    if any(w in keywords for w in ["analyze", "analysis", "review", "inspect"]):
        requirements.add("analysis")

    if any(w in keywords for w in ["plan", "design", "architect", "structure"]):
        requirements.add("planning")

    if any(w in keywords for w in ["refactor", "reorganize", "restructure"]):
        requirements.add("refactoring")

    if any(w in keywords for w in ["git", "commit", "branch", "merge"]):
        requirements.add("git_aware")

    return PartialSuccessHandler(
        available_frameworks=available_frameworks,
        task_requirements=requirements if requirements else None,
    )
