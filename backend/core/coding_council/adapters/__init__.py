"""
v84.0: Coding Council Framework Adapters
========================================

Adapters for integrating coding frameworks with the Coding Council.

Primary: Anthropic Claude API / JARVIS Prime Local LLM
Fallback: External tools (Aider CLI, MetaGPT package, etc.)

Each adapter provides a unified interface for:
- analyze(): Codebase analysis
- plan(): Task planning
- execute(): Code modification

Adapters:
    AnthropicUnifiedEngine - Claude-powered engine
    AiderAdapter     - Aider-style editing (uses Claude, falls back to CLI)
    OpenHandsAdapter - Sandboxed execution in Docker
    MetaGPTAdapter   - Multi-agent planning (uses Claude, falls back to package)
    RepoMasterAdapter - Codebase analysis
    ContinueAdapter   - IDE integration

v84.0 J-Prime Adapters (Local LLM):
    JPrimeCodingAdapter    - CodeLlama/DeepSeek for coding tasks (cost-free)
    JPrimeReasoningAdapter - Qwen/Llama for reasoning tasks (cost-free)
    JPrimeLocalAdapter     - General-purpose local inference (cost-free)
    JPrimeUnifiedEngine    - Unified engine for all J-Prime capabilities

Author: JARVIS v84.0
Version: 3.0.0
"""

from __future__ import annotations

__all__ = [
    # Anthropic Engine
    "AnthropicUnifiedEngine",
    "get_anthropic_engine",
    # Cloud Adapters
    "AiderAdapter",
    "OpenHandsAdapter",
    "MetaGPTAdapter",
    "RepoMasterAdapter",
    "ContinueAdapter",
    # v84.0: J-Prime Adapters (Local LLM)
    "JPrimeCodingAdapter",
    "JPrimeReasoningAdapter",
    "JPrimeLocalAdapter",
    "JPrimeUnifiedEngine",
    "JPrimeAvailabilityChecker",
    "classify_task_for_jprime",
    "is_task_suitable_for_jprime",
]


def __getattr__(name: str):
    """Lazy import adapters to avoid heavy dependencies."""
    # Anthropic Engine
    if name == "AnthropicUnifiedEngine":
        from .anthropic_engine import AnthropicUnifiedEngine
        return AnthropicUnifiedEngine
    elif name == "get_anthropic_engine":
        from .anthropic_engine import get_anthropic_engine
        return get_anthropic_engine

    # Cloud Adapters
    elif name == "AiderAdapter":
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

    # v84.0: J-Prime Adapters (Local LLM)
    elif name == "JPrimeCodingAdapter":
        from .jprime_adapter import JPrimeCodingAdapter
        return JPrimeCodingAdapter
    elif name == "JPrimeReasoningAdapter":
        from .jprime_adapter import JPrimeReasoningAdapter
        return JPrimeReasoningAdapter
    elif name == "JPrimeLocalAdapter":
        from .jprime_adapter import JPrimeLocalAdapter
        return JPrimeLocalAdapter
    elif name == "JPrimeUnifiedEngine":
        from .jprime_engine import JPrimeUnifiedEngine
        return JPrimeUnifiedEngine
    elif name == "JPrimeAvailabilityChecker":
        from .jprime_adapter import JPrimeAvailabilityChecker
        return JPrimeAvailabilityChecker
    elif name == "classify_task_for_jprime":
        from .jprime_adapter import classify_task_for_jprime
        return classify_task_for_jprime
    elif name == "is_task_suitable_for_jprime":
        from .jprime_adapter import is_task_suitable_for_jprime
        return is_task_suitable_for_jprime

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
