"""
Unified Observability Hub - Langfuse + Helicone Integration
============================================================

Provides comprehensive observability for the Ironcliw Memory-Aware Hybrid Routing system:

Features:
- Langfuse: LLM tracing, prompt management, and evaluation
- Helicone: Cost tracking, caching, and rate limiting
- Unified metrics: Latency, tokens, costs, routing decisions
- Real-time cost monitoring with alerts
- Automatic trace correlation across backends

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  Unified Observability Hub                  │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
    │  │  Langfuse   │  │  Helicone   │  │   Cost Optimizer     │ │
    │  │  - Traces   │  │  - Costs    │  │   - Budget alerts    │ │
    │  │  - Evals    │  │  - Cache    │  │   - Auto-routing     │ │
    │  │  - Prompts  │  │  - Metrics  │  │   - Optimization     │ │
    │  └─────────────┘  └─────────────┘  └──────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │    LOCAL    │ │  CLOUD_RUN  │ │ GEMINI_API  │
        │    FREE     │ │   ~$0.02/hr │ │ ~$0.0003/req│
        └─────────────┘ └─────────────┘ └─────────────┘
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ObservabilityConfig:
    """Configuration for the Unified Observability Hub."""

    # Langfuse
    langfuse_enabled: bool = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "") != ""
    )
    langfuse_public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    )

    # Helicone
    helicone_enabled: bool = field(
        default_factory=lambda: os.getenv("HELICONE_ENABLED", "true").lower() == "true"
    )
    helicone_api_key: str = field(
        default_factory=lambda: os.getenv("HELICONE_API_KEY", "")
    )
    helicone_cache_enabled: bool = field(
        default_factory=lambda: os.getenv("HELICONE_CACHE_ENABLED", "true").lower() == "true"
    )
    helicone_cache_ttl: int = field(
        default_factory=lambda: int(os.getenv("HELICONE_CACHE_TTL", "3600"))
    )

    # Cost tracking
    daily_budget_usd: float = field(
        default_factory=lambda: float(os.getenv("DAILY_BUDGET_USD", "1.0"))
    )
    monthly_budget_usd: float = field(
        default_factory=lambda: float(os.getenv("MONTHLY_BUDGET_USD", "10.0"))
    )
    cost_alert_threshold: float = field(
        default_factory=lambda: float(os.getenv("COST_ALERT_THRESHOLD", "0.8"))
    )

    # Local storage
    metrics_dir: Path = field(
        default_factory=lambda: Path(os.getenv("METRICS_DIR", "./observability"))
    )

    # Telemetry
    telemetry_enabled: bool = field(
        default_factory=lambda: os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"
    )
    telemetry_batch_size: int = 50
    telemetry_flush_interval: float = 30.0


# =============================================================================
# Cost Model
# =============================================================================

class CostTier(Enum):
    """Cost tiers for different backends."""
    LOCAL = "local"           # FREE
    CLOUD_RUN = "cloud_run"   # ~$0.02/hour when running
    GEMINI_API = "gemini_api" # Pay-per-token


@dataclass
class CostEstimate:
    """Cost estimate for a request."""
    tier: CostTier
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    compute_time_ms: float = 0.0
    compute_cost: float = 0.0  # For Cloud Run time-based billing
    cached: bool = False
    cache_savings: float = 0.0


class CostModel:
    """
    Cost model for all Ironcliw backends.

    Pricing (as of 2024):
    - LOCAL: FREE (your hardware)
    - CLOUD_RUN: ~$0.00002/second CPU time + memory
    - GEMINI_API: $0.075/1M input, $0.30/1M output (flash)
    """

    # Gemini 1.5 Flash pricing per million tokens
    GEMINI_INPUT_COST_PER_M = 0.075
    GEMINI_OUTPUT_COST_PER_M = 0.30

    # Cloud Run pricing (approximate)
    CLOUD_RUN_CPU_COST_PER_SECOND = 0.00002  # 2 vCPU
    CLOUD_RUN_MEMORY_COST_PER_SECOND = 0.000003  # 4GB

    @classmethod
    def estimate_cost(
        cls,
        tier: CostTier,
        input_tokens: int,
        output_tokens: int,
        compute_time_ms: float = 0,
        cached: bool = False,
    ) -> CostEstimate:
        """Estimate cost for a request."""

        if tier == CostTier.LOCAL:
            return CostEstimate(
                tier=tier,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost=0.0,  # FREE
            )

        elif tier == CostTier.CLOUD_RUN:
            compute_seconds = compute_time_ms / 1000
            compute_cost = (
                cls.CLOUD_RUN_CPU_COST_PER_SECOND * compute_seconds +
                cls.CLOUD_RUN_MEMORY_COST_PER_SECOND * compute_seconds
            )
            return CostEstimate(
                tier=tier,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                compute_time_ms=compute_time_ms,
                compute_cost=compute_cost,
                total_cost=compute_cost,
            )

        elif tier == CostTier.GEMINI_API:
            if cached:
                # Cached responses are free
                return CostEstimate(
                    tier=tier,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached=True,
                    cache_savings=(
                        (input_tokens / 1_000_000) * cls.GEMINI_INPUT_COST_PER_M +
                        (output_tokens / 1_000_000) * cls.GEMINI_OUTPUT_COST_PER_M
                    ),
                )

            input_cost = (input_tokens / 1_000_000) * cls.GEMINI_INPUT_COST_PER_M
            output_cost = (output_tokens / 1_000_000) * cls.GEMINI_OUTPUT_COST_PER_M

            return CostEstimate(
                tier=tier,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=input_cost + output_cost,
            )

        return CostEstimate(tier=tier)


# =============================================================================
# Trace Model
# =============================================================================

@dataclass
class TraceMetadata:
    """Metadata for a trace."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0

    # Request
    input_text: str = ""
    output_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    # Routing
    routing_mode: str = ""
    routing_reason: str = ""
    memory_available_gb: float = 0.0

    # Cost
    cost_estimate: Optional[CostEstimate] = None

    # Outcome
    success: bool = True
    error: Optional[str] = None

    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, output: str = "", success: bool = True, error: Optional[str] = None):
        """Complete the trace."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.output_text = output
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "routing_mode": self.routing_mode,
            "routing_reason": self.routing_reason,
            "memory_available_gb": self.memory_available_gb,
            "cost": self.cost_estimate.total_cost if self.cost_estimate else 0,
            "cached": self.cost_estimate.cached if self.cost_estimate else False,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


# =============================================================================
# Cost Tracker
# =============================================================================

class CostTracker:
    """
    Tracks costs across all backends with budget alerts.

    Features:
    - Real-time cost accumulation
    - Daily/monthly budgets with alerts
    - Cost breakdown by tier
    - Historical cost data
    """

    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self._lock = asyncio.Lock()

        # Current period costs
        self._daily_costs: Dict[str, float] = {
            "local": 0.0,
            "cloud_run": 0.0,
            "gemini_api": 0.0,
        }
        self._monthly_costs: Dict[str, float] = {
            "local": 0.0,
            "cloud_run": 0.0,
            "gemini_api": 0.0,
        }

        # Tracking
        self._last_reset_daily = datetime.now().date()
        self._last_reset_monthly = datetime.now().replace(day=1).date()
        self._request_count = 0
        self._cached_count = 0
        self._total_savings = 0.0

        # Alerts
        self._alert_callbacks: List[Callable[[str, float, float], None]] = []

    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """Add a callback for budget alerts."""
        self._alert_callbacks.append(callback)

    async def record_cost(self, estimate: CostEstimate) -> None:
        """Record a cost estimate."""
        async with self._lock:
            self._check_reset()

            tier_name = estimate.tier.value
            cost = estimate.total_cost

            self._daily_costs[tier_name] += cost
            self._monthly_costs[tier_name] += cost
            self._request_count += 1

            if estimate.cached:
                self._cached_count += 1
                self._total_savings += estimate.cache_savings

            # Check budget alerts
            await self._check_alerts()

    def _check_reset(self):
        """Check if we need to reset daily/monthly counters."""
        today = datetime.now().date()
        first_of_month = datetime.now().replace(day=1).date()

        if today > self._last_reset_daily:
            self._daily_costs = {k: 0.0 for k in self._daily_costs}
            self._last_reset_daily = today

        if first_of_month > self._last_reset_monthly:
            self._monthly_costs = {k: 0.0 for k in self._monthly_costs}
            self._last_reset_monthly = first_of_month

    async def _check_alerts(self):
        """Check and trigger budget alerts."""
        daily_total = sum(self._daily_costs.values())
        monthly_total = sum(self._monthly_costs.values())

        daily_pct = daily_total / self.config.daily_budget_usd if self.config.daily_budget_usd > 0 else 0
        monthly_pct = monthly_total / self.config.monthly_budget_usd if self.config.monthly_budget_usd > 0 else 0

        if daily_pct >= self.config.cost_alert_threshold:
            for callback in self._alert_callbacks:
                try:
                    callback("daily", daily_total, self.config.daily_budget_usd)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        if monthly_pct >= self.config.cost_alert_threshold:
            for callback in self._alert_callbacks:
                try:
                    callback("monthly", monthly_total, self.config.monthly_budget_usd)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        daily_total = sum(self._daily_costs.values())
        monthly_total = sum(self._monthly_costs.values())

        return {
            "daily": {
                "total": round(daily_total, 6),
                "budget": self.config.daily_budget_usd,
                "remaining": round(self.config.daily_budget_usd - daily_total, 6),
                "percent_used": round((daily_total / self.config.daily_budget_usd) * 100, 1) if self.config.daily_budget_usd > 0 else 0,
                "by_tier": {k: round(v, 6) for k, v in self._daily_costs.items()},
            },
            "monthly": {
                "total": round(monthly_total, 6),
                "budget": self.config.monthly_budget_usd,
                "remaining": round(self.config.monthly_budget_usd - monthly_total, 6),
                "percent_used": round((monthly_total / self.config.monthly_budget_usd) * 100, 1) if self.config.monthly_budget_usd > 0 else 0,
                "by_tier": {k: round(v, 6) for k, v in self._monthly_costs.items()},
            },
            "efficiency": {
                "total_requests": self._request_count,
                "cached_requests": self._cached_count,
                "cache_hit_rate": round(self._cached_count / max(self._request_count, 1) * 100, 1),
                "total_savings": round(self._total_savings, 6),
            },
        }


# =============================================================================
# Langfuse Integration
# =============================================================================

class LangfuseIntegration:
    """
    Langfuse integration for LLM observability.

    Provides:
    - Tracing for all LLM calls
    - Prompt management
    - Evaluation tracking
    - Cost attribution
    """

    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self._client = None
        self._enabled = config.langfuse_enabled and config.langfuse_public_key

        if self._enabled:
            self._init_client()

    def _init_client(self):
        """Initialize Langfuse client."""
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=self.config.langfuse_public_key,
                secret_key=self.config.langfuse_secret_key,
                host=self.config.langfuse_host,
            )
            logger.info("Langfuse client initialized")
        except ImportError:
            logger.warning("Langfuse not installed. Install with: pip install langfuse")
            self._enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self._enabled = False

    def create_trace(
        self,
        name: str,
        input_text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceMetadata:
        """Create a new trace."""
        trace = TraceMetadata(
            session_id=session_id,
            user_id=user_id,
            input_text=input_text,
            metadata=metadata or {},
        )

        if self._enabled and self._client:
            try:
                self._client.trace(
                    id=trace.trace_id,
                    name=name,
                    input=input_text,
                    session_id=session_id,
                    user_id=user_id,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Langfuse trace error: {e}")

        return trace

    def complete_trace(
        self,
        trace: TraceMetadata,
        output: str,
        cost_estimate: Optional[CostEstimate] = None,
    ):
        """Complete a trace."""
        trace.complete(output=output)
        trace.cost_estimate = cost_estimate

        if self._enabled and self._client:
            try:
                # Update trace with completion
                self._client.trace(
                    id=trace.trace_id,
                    output=output,
                    metadata={
                        **trace.metadata,
                        "duration_ms": trace.duration_ms,
                        "routing_mode": trace.routing_mode,
                        "cost": cost_estimate.total_cost if cost_estimate else 0,
                    },
                )

                # Log generation span
                if cost_estimate:
                    self._client.generation(
                        trace_id=trace.trace_id,
                        name=f"{trace.routing_mode}_generation",
                        input=trace.input_text,
                        output=output,
                        model=trace.routing_mode,
                        usage={
                            "input": trace.input_tokens,
                            "output": trace.output_tokens,
                        },
                        metadata={
                            "cached": cost_estimate.cached,
                            "cost": cost_estimate.total_cost,
                        },
                    )
            except Exception as e:
                logger.error(f"Langfuse completion error: {e}")

    def flush(self):
        """Flush pending events."""
        if self._enabled and self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.error(f"Langfuse flush error: {e}")


# =============================================================================
# Helicone Integration
# =============================================================================

class HeliconeIntegration:
    """
    Helicone integration for cost tracking and caching.

    Provides:
    - Automatic cost tracking for API calls
    - Response caching
    - Rate limiting
    - Usage analytics
    """

    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self._enabled = config.helicone_enabled and config.helicone_api_key
        self._cache: Dict[str, Tuple[str, datetime]] = {}
        self._cache_lock = asyncio.Lock()

    def get_headers(self, properties: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get Helicone headers for API calls."""
        if not self._enabled:
            return {}

        headers = {
            "Helicone-Auth": f"Bearer {self.config.helicone_api_key}",
        }

        if self.config.helicone_cache_enabled:
            headers["Helicone-Cache-Enabled"] = "true"
            headers["Cache-Control"] = f"max-age={self.config.helicone_cache_ttl}"

        if properties:
            for key, value in properties.items():
                headers[f"Helicone-Property-{key}"] = value

        return headers

    async def check_cache(self, prompt_hash: str) -> Optional[str]:
        """Check local cache for a prompt."""
        async with self._cache_lock:
            if prompt_hash in self._cache:
                response, timestamp = self._cache[prompt_hash]
                if datetime.now() - timestamp < timedelta(seconds=self.config.helicone_cache_ttl):
                    return response
                else:
                    del self._cache[prompt_hash]
        return None

    async def set_cache(self, prompt_hash: str, response: str):
        """Set local cache for a prompt."""
        async with self._cache_lock:
            self._cache[prompt_hash] = (response, datetime.now())

            # Limit cache size
            if len(self._cache) > 1000:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._cache.keys(),
                    key=lambda k: self._cache[k][1]
                )
                for key in sorted_keys[:100]:
                    del self._cache[key]


# =============================================================================
# Unified Observability Hub
# =============================================================================

class UnifiedObservabilityHub:
    """
    Unified Observability Hub for Ironcliw.

    Integrates:
    - Langfuse for tracing and evaluation
    - Helicone for cost tracking and caching
    - Local cost tracker for budgeting
    - Metrics persistence for historical analysis

    Usage:
        hub = UnifiedObservabilityHub()
        await hub.start()

        # Start a trace
        trace = hub.start_trace("chat", "Hello!")

        # Record completion
        cost = hub.record_completion(
            trace=trace,
            output="Hi there!",
            tier=CostTier.CLOUD_RUN,
            input_tokens=5,
            output_tokens=3,
            latency_ms=150,
        )

        # Get cost summary
        summary = hub.get_cost_summary()
    """

    _instance: Optional["UnifiedObservabilityHub"] = None

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        self.config = config or ObservabilityConfig()

        # Components
        self._langfuse = LangfuseIntegration(self.config)
        self._helicone = HeliconeIntegration(self.config)
        self._cost_tracker = CostTracker(self.config)

        # State
        self._running = False
        self._metrics_buffer: List[TraceMetadata] = []
        self._flush_task: Optional[asyncio.Task] = None

        # Ensure metrics directory exists
        self.config.metrics_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"UnifiedObservabilityHub initialized")
        logger.info(f"  Langfuse: {'enabled' if self._langfuse._enabled else 'disabled'}")
        logger.info(f"  Helicone: {'enabled' if self._helicone._enabled else 'disabled'}")

    @classmethod
    def get_instance(cls) -> "UnifiedObservabilityHub":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self):
        """Start the observability hub."""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

        # Register cost alert callback
        self._cost_tracker.add_alert_callback(self._on_cost_alert)

        logger.info("UnifiedObservabilityHub started")

    async def stop(self):
        """Stop the observability hub."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_metrics()
        self._langfuse.flush()

        logger.info("UnifiedObservabilityHub stopped")

    def start_trace(
        self,
        name: str,
        input_text: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceMetadata:
        """Start a new trace."""
        return self._langfuse.create_trace(
            name=name,
            input_text=input_text,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
        )

    async def record_completion(
        self,
        trace: TraceMetadata,
        output: str,
        tier: CostTier,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        routing_mode: str = "",
        routing_reason: str = "",
        memory_available_gb: float = 0.0,
        cached: bool = False,
    ) -> CostEstimate:
        """Record a completion and calculate costs."""
        # Estimate cost
        cost_estimate = CostModel.estimate_cost(
            tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            compute_time_ms=latency_ms,
            cached=cached,
        )

        # Update trace
        trace.input_tokens = input_tokens
        trace.output_tokens = output_tokens
        trace.routing_mode = routing_mode
        trace.routing_reason = routing_reason
        trace.memory_available_gb = memory_available_gb
        trace.cost_estimate = cost_estimate

        # Complete trace in Langfuse
        self._langfuse.complete_trace(trace, output, cost_estimate)

        # Record cost
        await self._cost_tracker.record_cost(cost_estimate)

        # Buffer for persistence
        trace.complete(output=output)
        self._metrics_buffer.append(trace)

        if len(self._metrics_buffer) >= self.config.telemetry_batch_size:
            await self._flush_metrics()

        return cost_estimate

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return self._cost_tracker.get_summary()

    def get_helicone_headers(self, properties: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get Helicone headers for API calls."""
        return self._helicone.get_headers(properties)

    async def check_cache(self, prompt_hash: str) -> Optional[str]:
        """Check cache for a prompt."""
        return await self._helicone.check_cache(prompt_hash)

    async def set_cache(self, prompt_hash: str, response: str):
        """Set cache for a prompt."""
        await self._helicone.set_cache(prompt_hash, response)

    def _on_cost_alert(self, period: str, current: float, budget: float):
        """Handle cost alerts."""
        logger.warning(
            f"⚠️ COST ALERT: {period.upper()} spending ${current:.4f} "
            f"({current/budget*100:.1f}% of ${budget:.2f} budget)"
        )

    async def _periodic_flush(self):
        """Periodically flush metrics."""
        while self._running:
            try:
                await asyncio.sleep(self.config.telemetry_flush_interval)
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic flush error: {e}")

    async def _flush_metrics(self):
        """Flush metrics to disk."""
        if not self._metrics_buffer:
            return

        buffer = self._metrics_buffer
        self._metrics_buffer = []

        # Write to daily file
        today = datetime.now().strftime("%Y-%m-%d")
        metrics_file = self.config.metrics_dir / f"metrics_{today}.jsonl"

        try:
            with open(metrics_file, "a") as f:
                for trace in buffer:
                    f.write(json.dumps(trace.to_dict()) + "\n")

            logger.debug(f"Flushed {len(buffer)} metrics to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            # Put back in buffer
            self._metrics_buffer = buffer + self._metrics_buffer


# =============================================================================
# Convenience Functions
# =============================================================================

_hub: Optional[UnifiedObservabilityHub] = None


def get_observability_hub() -> UnifiedObservabilityHub:
    """Get the global observability hub."""
    global _hub
    if _hub is None:
        _hub = UnifiedObservabilityHub()
    return _hub


async def record_llm_call(
    input_text: str,
    output_text: str,
    tier: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    routing_mode: str = "",
    routing_reason: str = "",
    memory_available_gb: float = 0.0,
    cached: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record an LLM call with full observability.

    Returns cost estimate and trace info.
    """
    hub = get_observability_hub()

    # Map tier string to CostTier
    tier_map = {
        "local": CostTier.LOCAL,
        "cloud_run": CostTier.CLOUD_RUN,
        "gemini": CostTier.GEMINI_API,
        "gemini_api": CostTier.GEMINI_API,
    }
    cost_tier = tier_map.get(tier.lower(), CostTier.LOCAL)

    # Create and complete trace
    trace = hub.start_trace(
        name=f"llm_{tier}",
        input_text=input_text,
        session_id=session_id,
    )

    cost = await hub.record_completion(
        trace=trace,
        output=output_text,
        tier=cost_tier,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        routing_mode=routing_mode,
        routing_reason=routing_reason,
        memory_available_gb=memory_available_gb,
        cached=cached,
    )

    return {
        "trace_id": trace.trace_id,
        "cost": cost.total_cost,
        "cached": cost.cached,
        "cache_savings": cost.cache_savings,
        "tier": tier,
    }


async def get_cost_report() -> Dict[str, Any]:
    """Get a cost report."""
    hub = get_observability_hub()
    return hub.get_cost_summary()
