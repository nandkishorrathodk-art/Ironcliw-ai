"""
v77.0: Metrics Collection - Gap #30
====================================

Metrics collection and aggregation:
- Counter, Gauge, Histogram types
- Labels/tags support
- Aggregation windows
- Export to various backends
- Timer utilities

Author: JARVIS v77.0
"""

from __future__ import annotations

import asyncio
import functools
import statistics
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")

# Labels type
Labels = Dict[str, str]


def _labels_key(labels: Optional[Labels]) -> str:
    """Convert labels dict to a hashable key."""
    if not labels:
        return ""
    return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


@dataclass
class MetricValue:
    """A metric value with labels."""
    value: float
    labels: Labels
    timestamp: float = field(default_factory=time.time)


class Metric(ABC):
    """Base class for metrics."""

    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[str, MetricValue] = {}

    @abstractmethod
    def _get_type(self) -> str:
        pass

    def _validate_labels(self, labels: Optional[Labels]) -> Labels:
        """Validate and fill in missing labels."""
        labels = labels or {}
        for label in self.label_names:
            if label not in labels:
                labels[label] = ""
        return labels

    def get_all_values(self) -> List[MetricValue]:
        """Get all metric values."""
        return list(self._values.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self._get_type(),
            "description": self.description,
            "values": [
                {"value": v.value, "labels": v.labels, "timestamp": v.timestamp}
                for v in self._values.values()
            ],
        }


class Counter(Metric):
    """
    Monotonically increasing counter.

    Use for: request counts, error counts, etc.
    """

    def _get_type(self) -> str:
        return "counter"

    def inc(self, value: float = 1.0, labels: Optional[Labels] = None) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter can only increase")

        labels = self._validate_labels(labels)
        key = _labels_key(labels)

        if key in self._values:
            self._values[key].value += value
            self._values[key].timestamp = time.time()
        else:
            self._values[key] = MetricValue(value=value, labels=labels)

    def get(self, labels: Optional[Labels] = None) -> float:
        """Get current counter value."""
        labels = self._validate_labels(labels)
        key = _labels_key(labels)
        return self._values[key].value if key in self._values else 0.0


class Gauge(Metric):
    """
    Value that can go up and down.

    Use for: current queue size, temperature, etc.
    """

    def _get_type(self) -> str:
        return "gauge"

    def set(self, value: float, labels: Optional[Labels] = None) -> None:
        """Set the gauge value."""
        labels = self._validate_labels(labels)
        key = _labels_key(labels)
        self._values[key] = MetricValue(value=value, labels=labels)

    def inc(self, value: float = 1.0, labels: Optional[Labels] = None) -> None:
        """Increment the gauge."""
        labels = self._validate_labels(labels)
        key = _labels_key(labels)

        if key in self._values:
            self._values[key].value += value
            self._values[key].timestamp = time.time()
        else:
            self._values[key] = MetricValue(value=value, labels=labels)

    def dec(self, value: float = 1.0, labels: Optional[Labels] = None) -> None:
        """Decrement the gauge."""
        self.inc(-value, labels)

    def get(self, labels: Optional[Labels] = None) -> float:
        """Get current gauge value."""
        labels = self._validate_labels(labels)
        key = _labels_key(labels)
        return self._values[key].value if key in self._values else 0.0


@dataclass
class HistogramBucket:
    """A histogram bucket."""
    upper_bound: float
    count: int = 0


class Histogram(Metric):
    """
    Distribution of values across buckets.

    Use for: request latencies, response sizes, etc.
    """

    DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__(name, description, labels)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._observations: Dict[str, List[float]] = defaultdict(list)
        self._bucket_counts: Dict[str, Dict[float, int]] = {}

    def _get_type(self) -> str:
        return "histogram"

    def observe(self, value: float, labels: Optional[Labels] = None) -> None:
        """Record an observation."""
        labels = self._validate_labels(labels)
        key = _labels_key(labels)

        self._observations[key].append(value)

        # Update buckets
        if key not in self._bucket_counts:
            self._bucket_counts[key] = {b: 0 for b in self.buckets}
            self._bucket_counts[key][float("inf")] = 0

        for bucket in self.buckets:
            if value <= bucket:
                self._bucket_counts[key][bucket] += 1
        self._bucket_counts[key][float("inf")] += 1

        # Update values dict with summary stats
        obs = self._observations[key]
        self._values[key] = MetricValue(
            value=sum(obs),
            labels=labels,
        )

    def get_percentile(self, percentile: float, labels: Optional[Labels] = None) -> Optional[float]:
        """Get a percentile value (0-100)."""
        labels = self._validate_labels(labels)
        key = _labels_key(labels)

        if key not in self._observations or not self._observations[key]:
            return None

        obs = sorted(self._observations[key])
        idx = int(len(obs) * percentile / 100)
        return obs[min(idx, len(obs) - 1)]

    def get_stats(self, labels: Optional[Labels] = None) -> Dict[str, float]:
        """Get summary statistics."""
        labels = self._validate_labels(labels)
        key = _labels_key(labels)

        if key not in self._observations or not self._observations[key]:
            return {}

        obs = self._observations[key]
        return {
            "count": len(obs),
            "sum": sum(obs),
            "mean": statistics.mean(obs),
            "median": statistics.median(obs),
            "stddev": statistics.stdev(obs) if len(obs) > 1 else 0,
            "min": min(obs),
            "max": max(obs),
            "p50": self.get_percentile(50, labels),
            "p90": self.get_percentile(90, labels),
            "p95": self.get_percentile(95, labels),
            "p99": self.get_percentile(99, labels),
        }


class Timer:
    """
    Context manager for timing operations.

    Records duration to a Histogram.
    """

    def __init__(self, histogram: Histogram, labels: Optional[Labels] = None):
        self.histogram = histogram
        self.labels = labels
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start:
            duration = time.time() - self._start
            self.histogram.observe(duration, self.labels)

    async def __aenter__(self) -> "Timer":
        self._start = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._start:
            duration = time.time() - self._start
            self.histogram.observe(duration, self.labels)


class MetricsCollector:
    """
    Central metrics collector and registry.

    Features:
    - Metric registration
    - Collection and aggregation
    - Export to various formats
    """

    def __init__(self, prefix: str = "coding_council"):
        self.prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._exporters: List[Callable[[Dict[str, Any]], Coroutine]] = []

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """Create or get a counter."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._metrics:
            self._metrics[full_name] = Counter(full_name, description, labels)
        return self._metrics[full_name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """Create or get a gauge."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._metrics:
            self._metrics[full_name] = Gauge(full_name, description, labels)
        return self._metrics[full_name]

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """Create or get a histogram."""
        full_name = f"{self.prefix}_{name}"
        if full_name not in self._metrics:
            self._metrics[full_name] = Histogram(full_name, description, labels, buckets)
        return self._metrics[full_name]

    def timer(
        self,
        name: str,
        description: str = "",
        labels: Optional[Labels] = None,
    ) -> Timer:
        """Create a timer for a histogram."""
        histogram = self.histogram(name, description)
        return Timer(histogram, labels)

    @asynccontextmanager
    async def time(
        self,
        name: str,
        labels: Optional[Labels] = None,
    ):
        """Context manager for timing."""
        histogram = self.histogram(name)
        async with Timer(histogram, labels):
            yield

    def collect(self) -> Dict[str, Any]:
        """Collect all metrics."""
        return {
            "timestamp": time.time(),
            "prefix": self.prefix,
            "metrics": {
                name: metric.to_dict()
                for name, metric in self._metrics.items()
            },
        }

    def add_exporter(self, exporter: Callable[[Dict[str, Any]], Coroutine]) -> None:
        """Add a metrics exporter."""
        self._exporters.append(exporter)

    async def export(self) -> None:
        """Export metrics to all exporters."""
        data = self.collect()
        for exporter in self._exporters:
            try:
                await exporter(data)
            except Exception:
                pass  # Don't fail on export errors

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()


# Global metrics collector
_collector: Optional[MetricsCollector] = None


def get_metrics_collector(prefix: str = "coding_council") -> MetricsCollector:
    """Get global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector(prefix)
    return _collector


def timed(
    name: Optional[str] = None,
    labels: Optional[Labels] = None,
):
    """
    Decorator to time function execution.

    Usage:
        @timed("api_call")
        async def call_api():
            ...
    """
    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        metric_name = name or f"{func.__name__}_duration"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            async with collector.time(metric_name, labels):
                return await func(*args, **kwargs)

        return wrapper
    return decorator


def counted(
    name: Optional[str] = None,
    labels: Optional[Labels] = None,
    count_errors: bool = True,
):
    """
    Decorator to count function calls.

    Usage:
        @counted("api_calls")
        async def call_api():
            ...
    """
    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        metric_name = name or f"{func.__name__}_total"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            counter = collector.counter(metric_name, labels=["status"])

            try:
                result = await func(*args, **kwargs)
                counter.inc(labels={**(labels or {}), "status": "success"})
                return result
            except Exception:
                if count_errors:
                    counter.inc(labels={**(labels or {}), "status": "error"})
                raise

        return wrapper
    return decorator
