"""
v77.0: Unified Coding Council - Self-Evolution Framework
=========================================================

This module provides a bulletproof, production-grade system for JARVIS
self-evolution through coordinated use of multiple AI coding frameworks.

Architecture:
    The Coding Council follows the "Voltron" pattern - combining the strengths
    of multiple specialized frameworks into a unified orchestration layer.

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    Coding Council Orchestrator                       │
    │  ┌─────────────────────────────────────────────────────────────────┐│
    │  │ MetaGPT     │ RepoMaster │ Aider     │ OpenHands │ Continue    ││
    │  │ (Planner)   │ (Analyzer) │ (Builder) │ (Sandbox) │ (IDE)       ││
    │  └─────────────────────────────────────────────────────────────────┘│
    │                              │                                       │
    │  ┌─────────────────────────────────────────────────────────────────┐│
    │  │               Safety & Validation Layer                         ││
    │  │  - AST Validation      - Security Scanning                     ││
    │  │  - Type Checking       - Contract Testing                       ││
    │  │  - Rollback Manager    - Circuit Breaker                        ││
    │  └─────────────────────────────────────────────────────────────────┘│
    └─────────────────────────────────────────────────────────────────────┘

Frameworks:
    1. MetaGPT: Multi-agent planning for complex features (PRD generation)
    2. RepoMaster: Codebase analysis and file discovery
    3. Aider: Fast, direct code editing with git integration
    4. OpenHands: Sandboxed execution for risky changes
    5. Continue.dev: IDE context and real-time assistance

Safety Features:
    - Git-based atomic rollback with savepoints
    - AST parsing before any commit
    - Security vulnerability scanning
    - Type checking validation
    - Circuit breaker for repeated failures
    - Resource monitoring and limits
    - Hot reload lock during evolution

Usage:
    from backend.core.coding_council import get_coding_council

    council = await get_coding_council()
    result = await council.evolve(
        description="Add weather command to voice handler",
        target_files=["backend/voice/intelligent_command_handler.py"]
    )

Author: JARVIS v77.0
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
if TYPE_CHECKING:
    from .orchestrator import UnifiedCodingCouncil

__all__ = [
    "get_coding_council",
    "shutdown_coding_council",
    "initialize_coding_council_full",
    "UnifiedCodingCouncil",
    "EvolutionTask",
    "EvolutionResult",
    "TaskComplexity",
    "FrameworkType",
    "CodingCouncilIntent",
    "CodingCouncilCommand",
]

# Singleton instance
_council_instance: "UnifiedCodingCouncil | None" = None


async def get_coding_council() -> "UnifiedCodingCouncil":
    """
    Get or create the singleton Coding Council instance.

    Returns:
        The initialized Coding Council
    """
    global _council_instance

    if _council_instance is None:
        from .orchestrator import UnifiedCodingCouncil
        _council_instance = UnifiedCodingCouncil()
        await _council_instance.initialize()
        logger.info("[CodingCouncil] v77.0 Unified Coding Council initialized")

    return _council_instance


async def shutdown_coding_council() -> None:
    """Shutdown the Coding Council and cleanup resources."""
    global _council_instance

    if _council_instance is not None:
        # First shutdown Trinity integration
        try:
            from .trinity_integration import shutdown_coding_council_trinity
            await shutdown_coding_council_trinity()
        except Exception as e:
            logger.warning(f"[CodingCouncil] Trinity shutdown warning: {e}")

        await _council_instance.shutdown()
        _council_instance = None
        logger.info("[CodingCouncil] Shutdown complete")


async def initialize_coding_council_full() -> "UnifiedCodingCouncil":
    """
    Full initialization of Coding Council with Trinity integration.

    This is the recommended way to initialize the Coding Council
    when running as part of the full JARVIS system.

    Returns:
        The initialized UnifiedCodingCouncil
    """
    # Get or create the council
    council = await get_coding_council()

    # Initialize Trinity integration
    try:
        from .trinity_integration import initialize_coding_council_trinity
        await initialize_coding_council_trinity(council)
        logger.info("[CodingCouncil] Trinity integration initialized")
    except Exception as e:
        logger.warning(f"[CodingCouncil] Trinity integration failed (continuing): {e}")

    return council


# Re-export key types for convenience
def __getattr__(name: str):
    """Lazy import of types to avoid circular imports."""
    if name in ("EvolutionTask", "EvolutionResult", "TaskComplexity"):
        from .types import EvolutionTask, EvolutionResult, TaskComplexity
        return locals()[name]
    elif name == "FrameworkType":
        from .types import FrameworkType
        return FrameworkType
    elif name == "UnifiedCodingCouncil":
        from .orchestrator import UnifiedCodingCouncil
        return UnifiedCodingCouncil
    elif name == "CodingCouncilIntent":
        from .trinity_integration import CodingCouncilIntent
        return CodingCouncilIntent
    elif name == "CodingCouncilCommand":
        from .trinity_integration import CodingCouncilCommand
        return CodingCouncilCommand
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
