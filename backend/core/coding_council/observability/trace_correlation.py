"""
v77.0: Trace Correlation - Gap #29
===================================

Distributed tracing with correlation IDs:
- Span-based tracing
- Parent-child span relationships
- Cross-service correlation
- Timing and duration tracking
- Baggage propagation

Author: Ironcliw v77.0
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar

T = TypeVar("T")

# Context variable for current span
_current_span: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "current_span",
    default=None
)


class SpanKind(Enum):
    """Type of span."""
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context identifying a span for propagation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanContext":
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {}),
        )

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {
            "X-Trace-ID": self.trace_id,
            "X-Span-ID": self.span_id,
        }
        if self.parent_span_id:
            headers["X-Parent-Span-ID"] = self.parent_span_id

        for key, value in self.baggage.items():
            headers[f"X-Baggage-{key}"] = value

        return headers

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["SpanContext"]:
        """Extract from HTTP headers."""
        trace_id = headers.get("X-Trace-ID")
        span_id = headers.get("X-Span-ID")

        if not trace_id or not span_id:
            return None

        baggage = {}
        for key, value in headers.items():
            if key.startswith("X-Baggage-"):
                baggage[key[10:]] = value

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=headers.get("X-Parent-Span-ID"),
            baggage=baggage,
        )


@dataclass
class SpanEvent:
    """An event within a span."""
    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single span in a trace."""
    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = SpanStatus.UNSET
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    _token: Optional[contextvars.Token] = field(default=None, repr=False)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        self.attributes[key] = value

    def add_event(self, name: str, **attributes) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(name=name, attributes=attributes))

    def set_status(self, status: SpanStatus, description: Optional[str] = None) -> None:
        """Set the span status."""
        self.status = status
        if description:
            self.set_attribute("status_description", description)

    def set_error(self, error: Exception) -> None:
        """Mark span as error with exception details."""
        self.status = SpanStatus.ERROR
        self.set_attribute("error.type", type(error).__name__)
        self.set_attribute("error.message", str(error))

    def end(self) -> None:
        """End the span."""
        self.end_time = time.time()

        # Reset context
        if self._token:
            _current_span.reset(self._token)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "kind": self.kind.value,
            "status": self.status.value,
            "start_time": datetime.utcfromtimestamp(self.start_time).isoformat() + "Z",
            "end_time": datetime.utcfromtimestamp(self.end_time).isoformat() + "Z" if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [{"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes} for e in self.events],
        }


class TraceCorrelator:
    """
    Distributed tracing correlator.

    Features:
    - Trace and span management
    - Context propagation
    - Baggage support
    - Export to various backends
    """

    def __init__(self, service_name: str = "coding_council"):
        self.service_name = service_name
        self._spans: Dict[str, Span] = {}
        self._completed_spans: List[Span] = []
        self._exporters: List[Callable[[Span], Coroutine]] = []
        self._max_completed = 1000

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """
        Start a new span.

        If parent is None, uses current span as parent.
        If no current span, starts a new trace.
        """
        # Determine parent
        if parent is None:
            current = _current_span.get()
            if current:
                parent = current.context

        # Generate IDs
        if parent:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id
            baggage = parent.baggage.copy()
        else:
            trace_id = self._generate_id()
            parent_span_id = None
            baggage = {}

        span_id = self._generate_id()

        # Create span
        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
        )

        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
        )

        # Add service info
        span.set_attribute("service.name", self.service_name)

        # Set as current
        span._token = _current_span.set(span)

        self._spans[span_id] = span
        return span

    def end_span(self, span: Span) -> None:
        """End a span and export it."""
        span.end()

        # Move to completed
        self._spans.pop(span.context.span_id, None)
        self._completed_spans.append(span)

        # Trim completed spans
        if len(self._completed_spans) > self._max_completed:
            self._completed_spans = self._completed_spans[-self._max_completed:]

        # Export
        asyncio.create_task(self._export_span(span))

    async def _export_span(self, span: Span) -> None:
        """Export span to registered exporters."""
        for exporter in self._exporters:
            try:
                await exporter(span)
            except Exception:
                pass  # Don't fail on export errors

    def add_exporter(self, exporter: Callable[[Span], Coroutine]) -> None:
        """Add a span exporter."""
        self._exporters.append(exporter)

    def get_current_span(self) -> Optional[Span]:
        """Get the current span."""
        return _current_span.get()

    def get_current_context(self) -> Optional[SpanContext]:
        """Get the current span context."""
        span = _current_span.get()
        return span.context if span else None

    def inject_context(self, carrier: Dict[str, str]) -> None:
        """Inject current context into carrier (e.g., HTTP headers)."""
        ctx = self.get_current_context()
        if ctx:
            carrier.update(ctx.to_headers())

    def extract_context(self, carrier: Dict[str, str]) -> Optional[SpanContext]:
        """Extract context from carrier."""
        return SpanContext.from_headers(carrier)

    def set_baggage(self, key: str, value: str) -> None:
        """Set baggage on current span context."""
        span = _current_span.get()
        if span:
            span.context.baggage[key] = value

    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage from current span context."""
        span = _current_span.get()
        if span:
            return span.context.baggage.get(key)
        return None

    @asynccontextmanager
    async def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        **attributes,
    ):
        """
        Context manager for span lifecycle.

        Usage:
            async with tracer.span("operation") as span:
                span.add_event("started")
                await do_work()
        """
        span = self.start_span(name, kind, attributes=attributes)
        try:
            yield span
            span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.end_span(span)

    @contextmanager
    def span_sync(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        **attributes,
    ):
        """Synchronous context manager for spans."""
        span = self.start_span(name, kind, attributes=attributes)
        try:
            yield span
            span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.end_span(span)

    def _generate_id(self) -> str:
        """Generate a trace/span ID."""
        return uuid.uuid4().hex[:16]

    def get_trace_summary(self, trace_id: str) -> Dict[str, Any]:
        """Get summary of a trace."""
        spans = [
            s for s in self._completed_spans
            if s.context.trace_id == trace_id
        ]

        if not spans:
            return {"trace_id": trace_id, "spans": []}

        spans.sort(key=lambda s: s.start_time)

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "duration_ms": sum(s.duration_ms or 0 for s in spans),
            "spans": [s.to_dict() for s in spans],
        }


# Global tracer
_tracer: Optional[TraceCorrelator] = None


def get_tracer(service_name: str = "coding_council") -> TraceCorrelator:
    """Get global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = TraceCorrelator(service_name)
    return _tracer


def get_current_span() -> Optional[Span]:
    """Get current span from context."""
    return _current_span.get()


def trace(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    **default_attributes,
):
    """
    Decorator to trace a function.

    Usage:
        @trace("my_operation")
        async def my_function():
            ...
    """
    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            async with tracer.span(span_name, kind, **default_attributes) as span:
                span.set_attribute("function", func.__name__)
                return await func(*args, **kwargs)

        return wrapper
    return decorator


def trace_sync(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
):
    """Decorator for synchronous functions."""
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.span_sync(span_name, kind):
                return func(*args, **kwargs)

        return wrapper
    return decorator
