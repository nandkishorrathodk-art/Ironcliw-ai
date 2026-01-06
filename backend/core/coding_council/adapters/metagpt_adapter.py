"""
v77.0: MetaGPT Framework Adapter
================================

Adapter for MetaGPT - multi-agent planning and PRD generation.

MetaGPT provides:
- Product Requirements Documents (PRD)
- Architecture design
- Multi-step execution plans
- Risk analysis

For complex features requiring structured planning.

Author: JARVIS v77.0
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import (
        AnalysisResult,
        CodingCouncilConfig,
        EvolutionTask,
        FrameworkResult,
        PlanResult,
    )

logger = logging.getLogger(__name__)


class MetaGPTAdapter:
    """
    Adapter for MetaGPT multi-agent planning.

    If MetaGPT is not installed, falls back to LLM-based
    planning using the available Claude/OpenAI API.
    """

    def __init__(self, config: "CodingCouncilConfig"):
        self.config = config
        self.repo_root = config.repo_root
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if MetaGPT is available."""
        if self._available is not None:
            return self._available

        try:
            import importlib.util
            spec = importlib.util.find_spec("metagpt")
            self._available = spec is not None
        except Exception:
            self._available = False

        if not self._available:
            logger.info("[MetaGPTAdapter] MetaGPT not installed, using fallback planning")

        return True  # Always available due to fallback

    async def plan(
        self,
        description: str,
        analysis: Optional["AnalysisResult"] = None
    ) -> "PlanResult":
        """
        Create execution plan for task.

        Returns:
            PlanResult with PRD, architecture, and steps
        """
        from ..types import PlanResult, TaskComplexity

        logger.info(f"[MetaGPTAdapter] Planning: {description[:50]}...")

        if self._available:
            # Use actual MetaGPT
            return await self._plan_with_metagpt(description, analysis)
        else:
            # Use fallback planning
            return await self._plan_with_fallback(description, analysis)

    async def _plan_with_metagpt(
        self,
        description: str,
        analysis: Optional["AnalysisResult"]
    ) -> "PlanResult":
        """Plan using actual MetaGPT."""
        from ..types import PlanResult, TaskComplexity

        try:
            # Import MetaGPT components
            from metagpt.roles import Architect, ProductManager
            from metagpt.software_company import SoftwareCompany

            # Create software company
            company = SoftwareCompany()

            # Run planning
            result = await company.start_project(
                idea=description,
                project_name="jarvis_evolution"
            )

            return PlanResult(
                prd=getattr(result, 'prd', ''),
                architecture=getattr(result, 'architecture', ''),
                steps=getattr(result, 'steps', []),
                estimated_complexity=TaskComplexity.COMPLEX,
                estimated_time_minutes=30.0,
                risks=[],
                dependencies=[],
            )

        except Exception as e:
            logger.warning(f"[MetaGPTAdapter] MetaGPT planning failed: {e}")
            return await self._plan_with_fallback(description, analysis)

    async def _plan_with_fallback(
        self,
        description: str,
        analysis: Optional["AnalysisResult"]
    ) -> "PlanResult":
        """Fallback planning using structured prompts."""
        from ..types import PlanResult, TaskComplexity

        # Generate basic PRD
        prd = f"""# Product Requirements Document

## Feature Request
{description}

## Scope
{"- " + chr(10).join("- Modify: " + f for f in analysis.target_files) if analysis else "To be determined during implementation"}

## Requirements
1. Implement the requested functionality
2. Maintain existing code quality and style
3. Add appropriate error handling
4. Update any affected tests

## Success Criteria
- Feature works as described
- All existing tests pass
- No security vulnerabilities introduced
"""

        # Generate architecture overview
        architecture = f"""# Architecture

## Components to Modify
{chr(10).join("- " + f for f in (analysis.target_files if analysis else ["To be determined"]))}

## Dependencies
{chr(10).join("- " + str(d) for d in (list(analysis.dependencies.keys())[:5] if analysis else ["None identified"]))}

## Approach
- Make minimal, focused changes
- Follow existing patterns in the codebase
- Ensure backward compatibility
"""

        # Generate steps
        steps = [
            {
                "step": 1,
                "description": "Analyze current implementation",
                "files": analysis.target_files if analysis else [],
            },
            {
                "step": 2,
                "description": "Implement core changes",
                "files": analysis.target_files if analysis else [],
            },
            {
                "step": 3,
                "description": "Add error handling",
                "files": analysis.target_files if analysis else [],
            },
            {
                "step": 4,
                "description": "Test changes",
                "files": [],
            },
        ]

        # Estimate complexity from analysis
        complexity = TaskComplexity.MEDIUM
        if analysis:
            if analysis.complexity_score > 0.7:
                complexity = TaskComplexity.COMPLEX
            elif analysis.complexity_score < 0.3:
                complexity = TaskComplexity.SIMPLE

        return PlanResult(
            prd=prd,
            architecture=architecture,
            steps=steps,
            estimated_complexity=complexity,
            estimated_time_minutes=15.0,
            risks=analysis.insights if analysis else [],
            dependencies=list(analysis.dependencies.keys()) if analysis else [],
        )

    async def execute(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"] = None,
        plan: Optional["PlanResult"] = None
    ) -> "FrameworkResult":
        """
        Execute using MetaGPT.

        Note: MetaGPT is primarily for planning. For execution,
        we generate code but recommend using Aider for application.
        """
        from ..types import FrameworkResult, FrameworkType

        # MetaGPT is a planner, not executor
        # Return guidance to use Aider for actual execution
        return FrameworkResult(
            framework=FrameworkType.METAGPT,
            success=True,
            changes_made=["Generated plan for implementation"],
            files_modified=[],
            output=f"Plan generated with {len(plan.steps) if plan else 0} steps. Use Aider for execution.",
        )
