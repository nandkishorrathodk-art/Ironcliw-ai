"""
v77.0: Observability Module - Gaps #28-31
==========================================

Observability and monitoring:
- Gap #28: Structured logging with context
- Gap #29: Distributed tracing with correlation IDs
- Gap #30: Metrics collection and aggregation
- Gap #31: Health monitoring and alerting

Author: Ironcliw v77.0
"""

from .structured_logger import (
    StructuredLogger,
    LogContext,
    LogLevel,
    get_logger,
)
from .trace_correlation import (
    TraceCorrelator,
    Span,
    SpanContext,
    trace,
    get_current_span,
)
from .metrics import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    Timer,
)
from .health_monitor import (
    HealthMonitor,
    HealthCheck,
    HealthStatus,
    ComponentHealth,
)

__all__ = [
    # Logging
    "StructuredLogger",
    "LogContext",
    "LogLevel",
    "get_logger",
    # Tracing
    "TraceCorrelator",
    "Span",
    "SpanContext",
    "trace",
    "get_current_span",
    # Metrics
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    # Health
    "HealthMonitor",
    "HealthCheck",
    "HealthStatus",
    "ComponentHealth",
]
