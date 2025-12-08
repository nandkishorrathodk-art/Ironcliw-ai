"""
Voice Authentication Observability Module

Comprehensive observability for voice authentication with:
- Langfuse audit trails for security investigation
- Helicone-style cost tracking and optimization
- Real-time metrics and analytics
"""

from .langfuse_integration import (
    VoiceAuthLangfuseTracer,
    AuthenticationTrace,
    TraceSpan,
    get_langfuse_tracer,
    create_langfuse_tracer,
)

from .helicone_integration import (
    VoiceAuthCostTracker,
    OperationCost,
    CostReport,
    VoiceCacheManager,
    get_cost_tracker,
    create_cost_tracker,
)

__all__ = [
    # Langfuse
    "VoiceAuthLangfuseTracer",
    "AuthenticationTrace",
    "TraceSpan",
    "get_langfuse_tracer",
    "create_langfuse_tracer",
    # Helicone
    "VoiceAuthCostTracker",
    "OperationCost",
    "CostReport",
    "VoiceCacheManager",
    "get_cost_tracker",
    "create_cost_tracker",
]
