"""
v77.0: Coding Council Framework Adapters
========================================

Adapters for integrating external coding frameworks with the Coding Council.

Each adapter provides a unified interface for:
- analyze(): Codebase analysis
- plan(): Task planning
- execute(): Code modification

Adapters:
    AiderAdapter     - Fast code editing with git integration
    OpenHandsAdapter - Sandboxed execution in Docker
    MetaGPTAdapter   - Multi-agent planning
    RepoMasterAdapter - Codebase analysis
    ContinueAdapter   - IDE integration

Author: JARVIS v77.0
Version: 1.0.0
"""

from __future__ import annotations

__all__ = [
    "AiderAdapter",
    "OpenHandsAdapter",
    "MetaGPTAdapter",
    "RepoMasterAdapter",
    "ContinueAdapter",
]


def __getattr__(name: str):
    """Lazy import adapters to avoid heavy dependencies."""
    if name == "AiderAdapter":
        from .aider_adapter import AiderAdapter
        return AiderAdapter
    elif name == "OpenHandsAdapter":
        from .openhands_adapter import OpenHandsAdapter
        return OpenHandsAdapter
    elif name == "MetaGPTAdapter":
        from .metagpt_adapter import MetaGPTAdapter
        return MetaGPTAdapter
    elif name == "RepoMasterAdapter":
        from .repomaster_adapter import RepoMasterAdapter
        return RepoMasterAdapter
    elif name == "ContinueAdapter":
        from .continue_adapter import ContinueAdapter
        return ContinueAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
