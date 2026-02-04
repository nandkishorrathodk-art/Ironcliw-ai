"""
JARVIS Loading Server Package v212.0
====================================

Modular components extracted from the loading server for enterprise-grade
startup orchestration.

This package provides:
- W3C Distributed Tracing for cross-service debugging
- Event Sourcing and SQLite Persistence for debugging and replay
- Enhanced ETA Prediction with ML-based learning
- Lock-Free Progress Updates with CAS atomic operations
- Container Awareness for K8s/Docker timeout scaling
- Adaptive Backpressure (AIMD) for slow client handling
- Cross-Repo Health Aggregation for unified system health
- Intelligent Message Generation for better UX feedback
- Self-Healing Restart Manager for auto-recovery
- Trinity Heartbeat Reader for direct file monitoring
- Progress Reporter client for external callers

Usage:
    from backend.loading_server import (
        W3CTraceContext,
        EventSourcingLog,
        ProgressPersistence,
        PredictiveETACalculator,
        LockFreeProgressUpdate,
        ContainerAwareness,
        AdaptiveBackpressureController,
        CrossRepoHealthAggregator,
        IntelligentMessageGenerator,
        SelfHealingRestartManager,
        TrinityHeartbeatReader,
        ProgressReporter,
    )

Author: JARVIS Trinity System
Version: 212.0.0
"""

from __future__ import annotations

__version__ = "212.0.0"
__all__ = [
    # Tier 1 - Critical
    "W3CTraceContext",
    "EventSourcingLog",
    "ProgressPersistence",
    "PredictiveETACalculator",
    # Tier 2 - High
    "LockFreeProgressUpdate",
    "ContainerAwareness",
    "AdaptiveBackpressureController",
    "CrossRepoHealthAggregator",
    # Tier 3 - Medium
    "IntelligentMessageGenerator",
    "SelfHealingRestartManager",
    "TrinityHeartbeatReader",
    # Client
    "ProgressReporter",
]

# Lazy imports to avoid circular dependencies and reduce startup time
def __getattr__(name: str):
    """Lazy import of submodules."""
    if name == "W3CTraceContext":
        from .tracing import W3CTraceContext
        return W3CTraceContext
    elif name == "EventSourcingLog":
        from .persistence import EventSourcingLog
        return EventSourcingLog
    elif name == "ProgressPersistence":
        from .persistence import ProgressPersistence
        return ProgressPersistence
    elif name == "PredictiveETACalculator":
        from .eta_prediction import PredictiveETACalculator
        return PredictiveETACalculator
    elif name == "LockFreeProgressUpdate":
        from .lock_free import LockFreeProgressUpdate
        return LockFreeProgressUpdate
    elif name == "ContainerAwareness":
        from .container_awareness import ContainerAwareness
        return ContainerAwareness
    elif name == "AdaptiveBackpressureController":
        from .backpressure import AdaptiveBackpressureController
        return AdaptiveBackpressureController
    elif name == "CrossRepoHealthAggregator":
        from .cross_repo_health import CrossRepoHealthAggregator
        return CrossRepoHealthAggregator
    elif name == "IntelligentMessageGenerator":
        from .message_generator import IntelligentMessageGenerator
        return IntelligentMessageGenerator
    elif name == "SelfHealingRestartManager":
        from .self_healing import SelfHealingRestartManager
        return SelfHealingRestartManager
    elif name == "TrinityHeartbeatReader":
        from .trinity_heartbeat import TrinityHeartbeatReader
        return TrinityHeartbeatReader
    elif name == "ProgressReporter":
        from .progress_reporter import ProgressReporter
        return ProgressReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
