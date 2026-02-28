"""
W3C Distributed Tracing for Ironcliw Loading Server v212.0
=========================================================

Implements W3C Trace Context specification for distributed tracing across
Ironcliw Body, Ironcliw Prime, and Reactor-Core.

Features:
- W3C traceparent header generation and parsing
- Span ID generation for tracing request chains
- Trace context propagation for cross-service debugging
- Configurable trace flags (sampled/not-sampled)

Usage:
    from backend.loading_server.tracing import W3CTraceContext

    # Create new trace context
    ctx = W3CTraceContext()

    # Add to outgoing request
    headers = {"traceparent": ctx.to_traceparent()}

    # Parse from incoming request
    ctx = W3CTraceContext.from_traceparent(request.headers.get("traceparent"))

    # Create child span
    child_ctx = ctx.create_child_span()

Specification: https://www.w3.org/TR/trace-context/

Author: Ironcliw Trinity System
Version: 212.0.0
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class W3CTraceContext:
    """
    W3C Distributed Tracing context for cross-repo correlation.

    Implements W3C Trace Context specification for distributed tracing across
    Ironcliw Body, Ironcliw Prime, and Reactor-Core.

    Attributes:
        trace_id: 32 hex chars (128 bits) - identifies the entire trace
        span_id: 16 hex chars (64 bits) - identifies this specific span
        parent_span_id: 16 hex chars - ID of the parent span (if any)
        trace_flags: 8 bits - trace flags (01 = sampled)
        trace_state: Vendor-specific trace state

    Example traceparent header:
        00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
        ^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^ ^^
        |  |                                |                trace-flags
        |  |                                span-id
        |  trace-id
        version
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    trace_flags: int = 1  # 1 = sampled
    trace_state: str = ""

    def __post_init__(self):
        """Validate and normalize trace context."""
        # Ensure trace_id is 32 hex chars
        if len(self.trace_id) != 32:
            self.trace_id = (self.trace_id * 2)[:32]

        # Ensure span_id is 16 hex chars
        if len(self.span_id) != 16:
            self.span_id = (self.span_id * 2)[:16]

        # Normalize trace_flags to valid range
        self.trace_flags = self.trace_flags & 0xFF

    def to_traceparent(self) -> str:
        """
        Generate W3C traceparent header value.

        Format: {version}-{trace-id}-{span-id}-{trace-flags}
        Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

        Returns:
            Formatted traceparent header string
        """
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    def to_tracestate(self) -> str:
        """
        Generate W3C tracestate header value.

        The tracestate header contains vendor-specific data.

        Returns:
            Formatted tracestate header string (may be empty)
        """
        if self.trace_state:
            return f"jarvis={self.trace_state}"
        return ""

    def to_headers(self) -> dict[str, str]:
        """
        Generate all W3C trace context headers.

        Returns:
            Dict with traceparent and optionally tracestate headers
        """
        headers = {"traceparent": self.to_traceparent()}
        if self.trace_state:
            headers["tracestate"] = self.to_tracestate()
        return headers

    @classmethod
    def from_traceparent(cls, traceparent: Optional[str]) -> "W3CTraceContext":
        """
        Parse W3C traceparent header.

        Args:
            traceparent: The traceparent header value

        Returns:
            Parsed W3CTraceContext or new context if parsing fails

        Format: {version}-{trace-id}-{span-id}-{trace-flags}
        Example: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
        """
        if not traceparent:
            return cls()

        try:
            parts = traceparent.strip().split("-")
            if len(parts) >= 4:
                version = parts[0]
                trace_id = parts[1]
                span_id = parts[2]
                trace_flags = int(parts[3], 16)

                # Version 00 is the only currently supported version
                if version == "00" and len(trace_id) == 32 and len(span_id) == 16:
                    return cls(
                        trace_id=trace_id,
                        span_id=span_id,
                        trace_flags=trace_flags,
                    )
        except (ValueError, IndexError):
            pass

        # Return new context if parsing fails
        return cls()

    @classmethod
    def from_headers(
        cls, headers: dict[str, str], case_insensitive: bool = True
    ) -> "W3CTraceContext":
        """
        Parse trace context from request headers.

        Args:
            headers: Request headers dict
            case_insensitive: Whether to do case-insensitive header lookup

        Returns:
            Parsed W3CTraceContext or new context if not found
        """
        if case_insensitive:
            headers_lower = {k.lower(): v for k, v in headers.items()}
            traceparent = headers_lower.get("traceparent")
            tracestate = headers_lower.get("tracestate", "")
        else:
            traceparent = headers.get("traceparent")
            tracestate = headers.get("tracestate", "")

        ctx = cls.from_traceparent(traceparent)

        # Parse tracestate for Ironcliw-specific data
        if tracestate and "jarvis=" in tracestate:
            for part in tracestate.split(","):
                if part.strip().startswith("jarvis="):
                    ctx.trace_state = part.strip()[7:]
                    break

        return ctx

    def create_child_span(self) -> "W3CTraceContext":
        """
        Create a child span context.

        The child span:
        - Inherits the same trace_id
        - Gets a new span_id
        - Records current span_id as parent_span_id
        - Inherits trace_flags and trace_state

        Returns:
            New W3CTraceContext for the child span
        """
        return W3CTraceContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            trace_flags=self.trace_flags,
            trace_state=self.trace_state,
        )

    def is_sampled(self) -> bool:
        """
        Check if this trace should be sampled (recorded).

        Returns:
            True if the sampled flag is set
        """
        return bool(self.trace_flags & 0x01)

    def set_sampled(self, sampled: bool) -> None:
        """
        Set the sampled flag.

        Args:
            sampled: Whether this trace should be sampled
        """
        if sampled:
            self.trace_flags |= 0x01
        else:
            self.trace_flags &= ~0x01

    def __str__(self) -> str:
        """String representation (same as traceparent header)."""
        return self.to_traceparent()

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"W3CTraceContext(trace_id={self.trace_id!r}, "
            f"span_id={self.span_id!r}, "
            f"parent_span_id={self.parent_span_id!r}, "
            f"trace_flags={self.trace_flags:#04x})"
        )
