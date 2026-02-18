"""
Distributed Tracing v2.0 -- Trace ID propagation for Trinity IPC.
=================================================================

Full-featured distributed tracing with trace/span IDs propagated across
cross-repo IPC messages, HTTP headers, and structured logs. Compatible
with OpenTelemetry naming conventions but zero external dependencies.

Features:
    - TraceContext dataclass with parent-child linking
    - Span context manager (sync + async) with status tracking
    - Tracer singleton with inject/extract for IPC propagation
    - SpanExporter protocol with LoggingExporter built-in
    - contextvars-based propagation (works across await boundaries)
    - @traced decorator for automatic span wrapping
    - Bounded in-memory span buffer (configurable via env var)
    - Cache registry integration for monitoring
    - Backward-compatible with v1.0 API (start_trace, from_message, stamp_message)

Environment Variables:
    JARVIS_TRACE_BUFFER_SIZE     Max completed spans in memory (default: 1000)
    JARVIS_SERVICE_NAME          Service name tag (default: jarvis-body)
    JARVIS_TRACE_LOG_LEVEL       Log level for span exports (default: DEBUG)
    JARVIS_TRACE_ENABLED         Enable/disable tracing (default: 1)

Usage:
    from backend.core.tracing import get_tracer, start_span, traced

    tracer = get_tracer()

    # Start a new trace (root span)
    with tracer.start_span("voice-command") as span:
        span.set_attribute("user", "derek")
        # ... do work ...

        # Nested child span (auto-linked via contextvars)
        with tracer.start_span("speaker-verify") as child:
            child.add_event("embedding_extracted", {"dims": "192"})
            result = verify_speaker(audio)

    # Async context manager
    async with tracer.start_span("async-operation") as span:
        await do_async_work()

    # Decorator
    @traced("process_command")
    async def handle_command(cmd: str):
        ...

    # IPC propagation
    msg = {"type": "command", "data": {...}}
    inject_trace(msg)   # adds _trace_id, _span_id, etc.

    # On receiving side:
    ctx = extract_trace(msg)
    with tracer.start_span("handle-remote", parent=ctx) as span:
        ...

Version: 2.0.0 (February 2026)
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger("jarvis.tracing")

# ---------------------------------------------------------------------------
# Configuration from environment (no hardcoding)
# ---------------------------------------------------------------------------

def _env_int(key: str, default: int) -> int:
    """Safely parse an integer from environment."""
    raw = os.environ.get(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Safely parse a boolean from environment."""
    raw = os.environ.get(key, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_TRACE_BUFFER_SIZE: int = _env_int("JARVIS_TRACE_BUFFER_SIZE", 1000)
_SERVICE_NAME: str = os.environ.get("JARVIS_SERVICE_NAME", "jarvis-body")
_TRACE_LOG_LEVEL: str = os.environ.get("JARVIS_TRACE_LOG_LEVEL", "DEBUG").upper()
_TRACE_ENABLED: bool = _env_bool("JARVIS_TRACE_ENABLED", True)

# ---------------------------------------------------------------------------
# IPC field name constants (used for inject/extract)
# ---------------------------------------------------------------------------

TRACE_ID_KEY: str = "_trace_id"
SPAN_ID_KEY: str = "_span_id"
PARENT_SPAN_ID_KEY: str = "_parent_span_id"
OPERATION_KEY: str = "_trace_op"
TRACE_ATTRIBUTES_KEY: str = "_trace_attrs"
TRACE_BAGGAGE_KEY: str = "_trace_baggage"

# ---------------------------------------------------------------------------
# Context variable for implicit span propagation
# ---------------------------------------------------------------------------

_current_span: ContextVar[Optional["Span"]] = ContextVar(
    "current_span", default=None
)

# ---------------------------------------------------------------------------
# SpanStatus enum
# ---------------------------------------------------------------------------

class SpanStatus(Enum):
    """Status of a completed span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# SpanEvent -- lightweight event annotation within a span
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpanEvent:
    """An event that occurred during a span's lifetime."""
    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": dict(self.attributes),
        }


# ---------------------------------------------------------------------------
# TraceContext dataclass
# ---------------------------------------------------------------------------

@dataclass
class TraceContext:
    """
    Immutable-ish trace context carrying IDs through the call chain.

    Designed for serialization across IPC (to_dict / from_dict) and
    structured logging (to_log_dict).
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.monotonic)
    wall_time: float = field(default_factory=time.time)
    attributes: Dict[str, str] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)

    # -- Factory helpers ---------------------------------------------------

    @classmethod
    def new_root(
        cls,
        operation_name: str,
        attributes: Optional[Dict[str, str]] = None,
        baggage: Optional[Dict[str, str]] = None,
    ) -> "TraceContext":
        """Create a root trace context (no parent)."""
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=os.urandom(8).hex(),
            parent_span_id=None,
            operation_name=operation_name,
            attributes=dict(attributes) if attributes else {},
            baggage=dict(baggage) if baggage else {},
        )

    def create_child(self, operation_name: str) -> "TraceContext":
        """Create a child context sharing the same trace_id."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=os.urandom(8).hex(),
            parent_span_id=self.span_id,
            operation_name=operation_name,
            attributes={},
            baggage=dict(self.baggage),
        )

    # -- Duration ----------------------------------------------------------

    def elapsed_ms(self) -> float:
        """Milliseconds since span start (monotonic)."""
        return (time.monotonic() - self.start_time) * 1000.0

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for IPC / storage."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "wall_time": self.wall_time,
            "attributes": dict(self.attributes),
            "baggage": dict(self.baggage),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContext":
        """Deserialize from IPC / storage."""
        return cls(
            trace_id=data.get("trace_id", uuid.uuid4().hex),
            span_id=data.get("span_id", os.urandom(8).hex()),
            parent_span_id=data.get("parent_span_id"),
            operation_name=data.get("operation_name", data.get("operation", "")),
            wall_time=data.get("wall_time", time.time()),
            attributes=data.get("attributes", {}),
            baggage=data.get("baggage", {}),
        )

    def to_log_dict(self) -> Dict[str, str]:
        """Return a flat dict suitable for structured log ``extra``."""
        d: Dict[str, str] = {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "operation": self.operation_name,
            "service": _SERVICE_NAME,
        }
        if self.parent_span_id:
            d["parent_span_id"] = self.parent_span_id
        return d

    # -- Backward-compatible helpers (v1.0 API) ----------------------------

    def log_extra(self) -> Dict[str, str]:
        """Alias for to_log_dict() -- v1.0 backward compat."""
        return self.to_log_dict()

    def stamp_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Add tracing fields to an IPC message dict (v1.0 compat)."""
        msg[TRACE_ID_KEY] = self.trace_id
        msg[SPAN_ID_KEY] = self.span_id
        msg[PARENT_SPAN_ID_KEY] = self.parent_span_id or ""
        msg[OPERATION_KEY] = self.operation_name
        if self.attributes:
            msg[TRACE_ATTRIBUTES_KEY] = self.attributes
        if self.baggage:
            msg[TRACE_BAGGAGE_KEY] = self.baggage
        return msg


# ---------------------------------------------------------------------------
# Span -- unit of work within a trace
# ---------------------------------------------------------------------------

class Span:
    """
    Represents a unit of work within a trace.

    Supports both sync (``with``) and async (``async with``) context managers.
    Automatically sets the contextvar so child spans link correctly.
    """

    __slots__ = (
        "context",
        "status",
        "status_description",
        "end_time",
        "events",
        "children",
        "_token",
        "_tracer_ref",
    )

    def __init__(
        self,
        context: TraceContext,
        *,
        tracer_ref: Optional["Tracer"] = None,
    ) -> None:
        self.context: TraceContext = context
        self.status: SpanStatus = SpanStatus.UNSET
        self.status_description: str = ""
        self.end_time: Optional[float] = None  # monotonic
        self.events: List[SpanEvent] = []
        self.children: List["Span"] = []
        self._token: Optional[Token] = None
        self._tracer_ref: Optional["Tracer"] = tracer_ref

    # -- Attribute / event helpers -----------------------------------------

    def set_attribute(self, key: str, value: str) -> None:
        """Set an arbitrary key-value attribute on this span."""
        self.context.attributes[key] = str(value)

    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """Set the span's completion status."""
        self.status = status
        self.status_description = description

    def add_event(
        self, name: str, attributes: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a timestamped event within this span."""
        self.events.append(
            SpanEvent(name=name, attributes=attributes or {})
        )

    def set_error(self, error: BaseException) -> None:
        """Mark span as errored from an exception."""
        self.status = SpanStatus.ERROR
        self.status_description = str(error)
        self.set_attribute("error.type", type(error).__name__)
        self.set_attribute("error.message", str(error))

    # -- Duration ----------------------------------------------------------

    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds, or None if still running."""
        if self.end_time is None:
            return None
        return (self.end_time - self.context.start_time) * 1000.0

    # -- Finalization ------------------------------------------------------

    def _finish(self) -> None:
        """Finalize the span (set end_time, restore contextvar, notify tracer)."""
        if self.end_time is not None:
            return  # already finished
        self.end_time = time.monotonic()
        # Auto-set OK if status was never explicitly changed
        if self.status is SpanStatus.UNSET:
            self.status = SpanStatus.OK
        # Restore previous contextvar
        if self._token is not None:
            try:
                _current_span.reset(self._token)
            except ValueError:
                pass  # token already reset
            self._token = None
        # Notify tracer
        if self._tracer_ref is not None:
            self._tracer_ref._on_span_end(self)

    # -- Sync context manager ----------------------------------------------

    def __enter__(self) -> "Span":
        self._token = _current_span.set(self)
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        if exc_val is not None:
            if isinstance(exc_val, asyncio.TimeoutError):
                self.set_status(SpanStatus.TIMEOUT, str(exc_val))
            else:
                self.set_error(exc_val)
        self._finish()

    # -- Async context manager ---------------------------------------------

    async def __aenter__(self) -> "Span":
        self._token = _current_span.set(self)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        if exc_val is not None:
            if isinstance(exc_val, asyncio.TimeoutError):
                self.set_status(SpanStatus.TIMEOUT, str(exc_val))
            else:
                self.set_error(exc_val)
        self._finish()

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Full span serialization for export."""
        return {
            "context": self.context.to_dict(),
            "status": self.status.value,
            "status_description": self.status_description,
            "duration_ms": self.duration_ms,
            "events": [e.to_dict() for e in self.events],
            "children": [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# SpanExporter protocol + LoggingExporter
# ---------------------------------------------------------------------------

@runtime_checkable
class SpanExporter(Protocol):
    """Protocol for span export backends."""

    def export(self, spans: List[Span]) -> None:
        """Export a batch of completed spans."""
        ...


class LoggingExporter:
    """
    Exports completed spans to Python structured logging.

    Each span is logged at the configured level with trace/span IDs,
    duration, and status as structured ``extra`` fields.
    """

    def __init__(
        self,
        logger_name: str = "jarvis.tracing.export",
        level: Optional[int] = None,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        if level is not None:
            self._level = level
        else:
            self._level = getattr(logging, _TRACE_LOG_LEVEL, logging.DEBUG)

    def export(self, spans: List[Span]) -> None:
        for span in spans:
            extra = span.context.to_log_dict()
            extra["duration_ms"] = (
                f"{span.duration_ms:.2f}" if span.duration_ms is not None else "?"
            )
            extra["status"] = span.status.value
            if span.status_description:
                extra["status_description"] = span.status_description
            if span.events:
                extra["event_count"] = str(len(span.events))
            self._logger.log(
                self._level,
                "span_completed: %s",
                span.context.operation_name,
                extra=extra,
            )


class InMemoryExporter:
    """
    Collects spans in memory for testing / inspection.

    Thread-safe. The buffer is bounded to ``max_spans``.
    """

    def __init__(self, max_spans: int = 0) -> None:
        self._max_spans = max_spans or _TRACE_BUFFER_SIZE
        self._spans: Deque[Span] = deque(maxlen=self._max_spans)
        self._lock = threading.Lock()

    def export(self, spans: List[Span]) -> None:
        with self._lock:
            for span in spans:
                self._spans.append(span)

    @property
    def spans(self) -> List[Span]:
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def find_by_trace(self, trace_id: str) -> List[Span]:
        with self._lock:
            return [
                s for s in self._spans if s.context.trace_id == trace_id
            ]


# ---------------------------------------------------------------------------
# Tracer singleton
# ---------------------------------------------------------------------------

class Tracer:
    """
    Central tracer that manages span lifecycle, context propagation,
    and export.

    Singleton -- obtain via ``get_tracer()`` or ``Tracer.get_instance()``.
    """

    _instance: Optional["Tracer"] = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._service_name: str = _SERVICE_NAME
        self._enabled: bool = _TRACE_ENABLED
        self._exporters: List[SpanExporter] = []
        self._completed_buffer: Deque[Span] = deque(maxlen=_TRACE_BUFFER_SIZE)
        self._lock = threading.Lock()

        # Stats
        self._total_spans_created: int = 0
        self._total_spans_completed: int = 0
        self._total_spans_errored: int = 0
        self._active_span_count: int = 0

        # Default: logging exporter
        self._exporters.append(LoggingExporter())

        # Register with cache registry
        self._register_with_cache_registry()

    @classmethod
    def get_instance(cls) -> "Tracer":
        """Get or create the singleton Tracer."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset the singleton (for testing only)."""
        with cls._init_lock:
            cls._instance = None

    # -- Span lifecycle ----------------------------------------------------

    def start_span(
        self,
        operation_name: str,
        *,
        parent: Optional[TraceContext] = None,
        attributes: Optional[Dict[str, str]] = None,
        baggage: Optional[Dict[str, str]] = None,
    ) -> Span:
        """
        Start a new span.

        If ``parent`` is not provided, uses the current span from
        contextvars. If there is no current span, starts a new root trace.

        Returns a ``Span`` that works as both sync and async context manager.
        """
        if not self._enabled:
            # Return a no-op span with a minimal context
            ctx = TraceContext(
                trace_id="0" * 32,
                span_id="0" * 16,
                operation_name=operation_name,
            )
            return Span(ctx)

        # Resolve parent
        if parent is None:
            current = _current_span.get()
            if current is not None:
                parent = current.context

        if parent is not None:
            ctx = parent.create_child(operation_name)
            # Merge parent baggage, then overlay explicit baggage
            if baggage:
                ctx.baggage.update(baggage)
        else:
            ctx = TraceContext.new_root(
                operation_name,
                baggage=baggage,
            )

        if attributes:
            ctx.attributes.update(attributes)

        # Add service attribute automatically
        ctx.attributes.setdefault("service.name", self._service_name)

        span = Span(ctx, tracer_ref=self)

        with self._lock:
            self._total_spans_created += 1
            self._active_span_count += 1

        return span

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span from contextvars."""
        return _current_span.get()

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID, or None if no span active."""
        span = _current_span.get()
        return span.context.trace_id if span else None

    # -- Span completion callback ------------------------------------------

    def _on_span_end(self, span: Span) -> None:
        """Called when a span finishes (from Span._finish)."""
        with self._lock:
            self._active_span_count = max(0, self._active_span_count - 1)
            self._total_spans_completed += 1
            if span.status in (SpanStatus.ERROR, SpanStatus.TIMEOUT):
                self._total_spans_errored += 1
            self._completed_buffer.append(span)

        # Export
        self._export_spans([span])

    def _export_spans(self, spans: List[Span]) -> None:
        """Send spans to all registered exporters."""
        for exporter in self._exporters:
            try:
                exporter.export(spans)
            except Exception as exc:
                # Never let export failures affect application logic
                logger.debug(
                    "Span export failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

    # -- Exporter management -----------------------------------------------

    def set_exporter(self, exporter: SpanExporter) -> None:
        """
        Replace all exporters with a single one.

        Use ``add_exporter`` to keep existing exporters.
        """
        with self._lock:
            self._exporters = [exporter]

    def add_exporter(self, exporter: SpanExporter) -> None:
        """Add an exporter without removing existing ones."""
        with self._lock:
            self._exporters.append(exporter)

    def clear_exporters(self) -> None:
        """Remove all exporters."""
        with self._lock:
            self._exporters.clear()

    # -- IPC integration ---------------------------------------------------

    def inject(self, carrier: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject current trace context into a carrier dict (IPC message,
        HTTP headers, etc.).

        Returns the carrier for chaining.
        """
        span = _current_span.get()
        if span is None:
            return carrier
        ctx = span.context
        carrier[TRACE_ID_KEY] = ctx.trace_id
        carrier[SPAN_ID_KEY] = ctx.span_id
        carrier[PARENT_SPAN_ID_KEY] = ctx.parent_span_id or ""
        carrier[OPERATION_KEY] = ctx.operation_name
        if ctx.attributes:
            carrier[TRACE_ATTRIBUTES_KEY] = dict(ctx.attributes)
        if ctx.baggage:
            carrier[TRACE_BAGGAGE_KEY] = dict(ctx.baggage)
        return carrier

    def extract(self, carrier: Dict[str, Any]) -> Optional[TraceContext]:
        """
        Extract trace context from a carrier dict.

        Returns None if no trace information present.
        """
        trace_id = carrier.get(TRACE_ID_KEY)
        if not trace_id:
            return None
        span_id = carrier.get(SPAN_ID_KEY, os.urandom(8).hex())
        parent_span_id = carrier.get(PARENT_SPAN_ID_KEY) or None
        if parent_span_id == "":
            parent_span_id = None
        return TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=carrier.get(OPERATION_KEY, "remote"),
            attributes=carrier.get(TRACE_ATTRIBUTES_KEY, {}),
            baggage=carrier.get(TRACE_BAGGAGE_KEY, {}),
        )

    # -- HTTP header propagation -------------------------------------------

    def inject_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into HTTP headers (W3C-like format)."""
        span = _current_span.get()
        if span is None:
            return headers
        ctx = span.context
        headers["X-Trace-ID"] = ctx.trace_id
        headers["X-Span-ID"] = ctx.span_id
        if ctx.parent_span_id:
            headers["X-Parent-Span-ID"] = ctx.parent_span_id
        headers["X-Trace-Operation"] = ctx.operation_name
        for key, value in ctx.baggage.items():
            headers[f"X-Baggage-{key}"] = value
        return headers

    def extract_headers(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extract trace context from HTTP headers."""
        trace_id = headers.get("X-Trace-ID")
        if not trace_id:
            return None
        baggage: Dict[str, str] = {}
        for key, value in headers.items():
            if key.startswith("X-Baggage-"):
                baggage[key[len("X-Baggage-"):]] = value
        return TraceContext(
            trace_id=trace_id,
            span_id=headers.get("X-Span-ID", os.urandom(8).hex()),
            parent_span_id=headers.get("X-Parent-Span-ID"),
            operation_name=headers.get("X-Trace-Operation", "remote"),
            baggage=baggage,
        )

    # -- Query completed spans ---------------------------------------------

    def get_completed_spans(self) -> List[Span]:
        """Return a snapshot of the completed span buffer."""
        with self._lock:
            return list(self._completed_buffer)

    def find_spans_by_trace(self, trace_id: str) -> List[Span]:
        """Find all completed spans belonging to a trace."""
        with self._lock:
            return [
                s for s in self._completed_buffer
                if s.context.trace_id == trace_id
            ]

    def get_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """
        Build a tree representation of a trace from completed spans.

        Returns a dict with root span and nested children.
        """
        spans = self.find_spans_by_trace(trace_id)
        if not spans:
            return {"trace_id": trace_id, "spans": []}

        # Index by span_id
        by_id: Dict[str, Dict[str, Any]] = {}
        for s in spans:
            node = s.to_dict()
            node["_span_id"] = s.context.span_id
            node["_parent_span_id"] = s.context.parent_span_id
            by_id[s.context.span_id] = node

        # Build tree
        roots: List[Dict[str, Any]] = []
        for node in by_id.values():
            parent_id = node.get("_parent_span_id")
            if parent_id and parent_id in by_id:
                parent_node = by_id[parent_id]
                parent_node.setdefault("child_spans", []).append(node)
            else:
                roots.append(node)

        # Clean up internal keys
        def _clean(n: Dict[str, Any]) -> None:
            n.pop("_span_id", None)
            n.pop("_parent_span_id", None)
            for child in n.get("child_spans", []):
                _clean(child)

        for root in roots:
            _clean(root)

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "spans": roots,
        }

    # -- Stats (cache-registry-compatible) ---------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return tracer statistics.

        Compatible with the CacheRegistry ``CacheLike`` protocol.
        """
        with self._lock:
            return {
                "service": self._service_name,
                "enabled": self._enabled,
                "active_spans": self._active_span_count,
                "total_spans_created": self._total_spans_created,
                "total_spans_completed": self._total_spans_completed,
                "total_spans_errored": self._total_spans_errored,
                "buffer_size": len(self._completed_buffer),
                "buffer_capacity": self._completed_buffer.maxlen,
                "exporter_count": len(self._exporters),
            }

    # -- Cache registry integration ----------------------------------------

    def _register_with_cache_registry(self) -> None:
        """Register with the global cache registry if available."""
        try:
            from backend.utils.cache_registry import get_cache_registry

            get_cache_registry().register("distributed_tracer", self)
            logger.debug("[Tracer] Registered with cache registry")
        except Exception:
            pass  # Cache registry not available

    # -- Backward compatibility (v1.0 API) ---------------------------------

    def start_trace(
        self,
        operation: str,
        *,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> TraceContext:
        """
        Start a new trace and return a TraceContext (v1.0 compat).

        The returned context can be used as a sync context manager.
        """
        span = self.start_span(
            operation,
            attributes={k: str(v) for k, v in (attributes or {}).items()},
        )
        # Return the context but also enter the span
        span.__enter__()
        return span.context

    def from_message(self, msg: Dict[str, Any]) -> Optional[TraceContext]:
        """
        Extract trace context from an incoming IPC message (v1.0 compat).
        """
        return self.extract(msg)

    @staticmethod
    def current() -> Optional[TraceContext]:
        """Get the current trace context (v1.0 compat)."""
        span = _current_span.get()
        return span.context if span else None


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_tracer() -> Tracer:
    """Get the singleton Tracer instance."""
    return Tracer.get_instance()


def start_span(
    name: str,
    *,
    parent: Optional[TraceContext] = None,
    attributes: Optional[Dict[str, str]] = None,
) -> Span:
    """Convenience: start a span using the global tracer."""
    return get_tracer().start_span(name, parent=parent, attributes=attributes)


def current_trace_id() -> Optional[str]:
    """Get the current trace ID, or None if no span active."""
    span = _current_span.get()
    return span.context.trace_id if span else None


def current_trace() -> Optional[TraceContext]:
    """Get the current TraceContext, or None."""
    span = _current_span.get()
    return span.context if span else None


def trace_id() -> str:
    """Get the current trace ID, or empty string if none (v1.0 compat)."""
    span = _current_span.get()
    return span.context.trace_id if span else ""


# ---------------------------------------------------------------------------
# IPC integration helpers
# ---------------------------------------------------------------------------

def inject_trace(carrier: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add current trace context to a carrier dict (IPC message, etc.).

    Returns the carrier for chaining.
    """
    return get_tracer().inject(carrier)


def extract_trace(carrier: Dict[str, Any]) -> Optional[TraceContext]:
    """
    Extract trace context from a carrier dict.

    Returns None if no trace information present.
    """
    return get_tracer().extract(carrier)


def inject_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Add current trace context to HTTP headers."""
    return get_tracer().inject_headers(headers)


def extract_headers(headers: Dict[str, str]) -> Optional[TraceContext]:
    """Extract trace context from HTTP headers."""
    return get_tracer().extract_headers(headers)


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------

_F = TypeVar("_F", bound=Callable[..., Any])


def traced(operation_name: str = "") -> Callable[[_F], _F]:
    """
    Decorator that wraps a function (sync or async) in a trace span.

    Usage:
        @traced("process_command")
        async def handle_command(cmd: str):
            ...

        @traced()  # auto-uses function name
        def sync_handler(data):
            ...
    """

    def decorator(func: _F) -> _F:
        op = operation_name or func.__qualname__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                async with tracer.start_span(op) as span:
                    span.set_attribute("function", func.__qualname__)
                    span.set_attribute("module", func.__module__)
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_span(op) as span:
                    span.set_attribute("function", func.__qualname__)
                    span.set_attribute("module", func.__module__)
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# __all__ -- public API
# ---------------------------------------------------------------------------

__all__ = [
    # Core types
    "TraceContext",
    "Span",
    "SpanStatus",
    "SpanEvent",
    # Tracer
    "Tracer",
    "get_tracer",
    # Exporters
    "SpanExporter",
    "LoggingExporter",
    "InMemoryExporter",
    # Module-level convenience
    "start_span",
    "current_trace_id",
    "current_trace",
    "trace_id",
    # IPC helpers
    "inject_trace",
    "extract_trace",
    "inject_headers",
    "extract_headers",
    # IPC field constants
    "TRACE_ID_KEY",
    "SPAN_ID_KEY",
    "PARENT_SPAN_ID_KEY",
    "OPERATION_KEY",
    "TRACE_ATTRIBUTES_KEY",
    "TRACE_BAGGAGE_KEY",
    # Decorator
    "traced",
]
