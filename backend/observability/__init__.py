"""
JARVIS Observability Package
=============================

Unified observability for the JARVIS AI system.

Components:
- UnifiedObservabilityHub: Central hub for all observability
- Langfuse Integration: LLM tracing and evaluation
- Helicone Integration: Cost tracking and caching
- CostTracker: Budget management and alerts

Usage:
    from observability import (
        get_observability_hub,
        record_llm_call,
        get_cost_report,
    )

    # Record an LLM call
    result = await record_llm_call(
        input_text="Hello",
        output_text="Hi there!",
        tier="cloud_run",
        input_tokens=5,
        output_tokens=3,
        latency_ms=150,
    )

    # Get cost report
    report = await get_cost_report()
"""

from .unified_observability_hub import (
    # Main hub
    UnifiedObservabilityHub,
    get_observability_hub,

    # Configuration
    ObservabilityConfig,

    # Cost tracking
    CostTier,
    CostEstimate,
    CostModel,
    CostTracker,

    # Tracing
    TraceMetadata,
    LangfuseIntegration,
    HeliconeIntegration,

    # Convenience functions
    record_llm_call,
    get_cost_report,
)

__all__ = [
    # Main hub
    "UnifiedObservabilityHub",
    "get_observability_hub",

    # Configuration
    "ObservabilityConfig",

    # Cost tracking
    "CostTier",
    "CostEstimate",
    "CostModel",
    "CostTracker",

    # Tracing
    "TraceMetadata",
    "LangfuseIntegration",
    "HeliconeIntegration",

    # Convenience functions
    "record_llm_call",
    "get_cost_report",
]
