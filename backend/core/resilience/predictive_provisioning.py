"""
Predictive VM Provisioning with Time-Series Forecasting
========================================================

Provides intelligent VM provisioning that predicts demand before it occurs.

Features:
    - Exponential smoothing for trend detection
    - Moving average for noise reduction
    - Seasonal pattern detection (hourly, daily, weekly)
    - Anomaly detection for unusual spikes
    - Multi-step ahead forecasting
    - Confidence intervals for predictions
    - Auto-scaling recommendations
    - Historical pattern learning

Theory:
    Traditional auto-scaling is reactive - it provisions VMs after
    demand increases. This causes latency spikes during scale-up.

    Predictive provisioning uses time-series forecasting to anticipate
    demand and pre-provision VMs, ensuring capacity is ready when needed.

Usage:
    predictor = await get_predictive_provisioner()

    # Record current metrics
    await predictor.record_metric("cpu_usage", 75.0)
    await predictor.record_metric("memory_usage", 82.0)
    await predictor.record_metric("request_rate", 1000)

    # Get provisioning recommendation
    recommendation = await predictor.get_recommendation()
    if recommendation.should_scale_up:
        await provision_vms(recommendation.recommended_count)

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("PredictiveProvisioning")


# =============================================================================
# Configuration
# =============================================================================

# Forecasting parameters
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "5"))  # minutes ahead
FORECAST_CONFIDENCE_LEVEL = float(os.getenv("FORECAST_CONFIDENCE_LEVEL", "0.95"))
SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", "0.3"))
TREND_BETA = float(os.getenv("TREND_BETA", "0.1"))
SEASONAL_GAMMA = float(os.getenv("SEASONAL_GAMMA", "0.1"))

# Data retention
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "1440"))  # minutes (24 hours)
SEASONAL_CYCLE = int(os.getenv("SEASONAL_CYCLE", "60"))  # minutes (hourly cycles)

# Provisioning thresholds
SCALE_UP_THRESHOLD = float(os.getenv("SCALE_UP_THRESHOLD", "80.0"))
SCALE_DOWN_THRESHOLD = float(os.getenv("SCALE_DOWN_THRESHOLD", "30.0"))
SCALE_UP_MARGIN = float(os.getenv("SCALE_UP_MARGIN", "10.0"))  # Pre-scale before hitting threshold

# Anomaly detection
ANOMALY_ZSCORE_THRESHOLD = float(os.getenv("ANOMALY_ZSCORE_THRESHOLD", "3.0"))


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ScalingAction(Enum):
    """Recommended scaling action."""
    NONE = "none"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    EMERGENCY_SCALE = "emergency_scale"


class MetricType(Enum):
    """Types of metrics to track."""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_DEPTH = "queue_depth"
    ERROR_RATE = "error_rate"


@dataclass
class MetricPoint:
    """A single metric data point."""
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class Forecast:
    """A forecast result."""
    metric_name: str
    current_value: float
    predicted_value: float
    predicted_at: float  # timestamp of prediction
    horizon_minutes: int
    confidence_lower: float
    confidence_upper: float
    trend: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    anomaly_detected: bool = False


@dataclass
class ScalingRecommendation:
    """A VM scaling recommendation."""
    action: ScalingAction
    recommended_count: int  # VMs to add/remove
    confidence: float
    reasons: List[str]
    forecasts: List[Forecast]
    urgency: str  # "low", "medium", "high", "critical"
    estimated_time_to_threshold: Optional[float] = None  # minutes


@dataclass
class SeasonalPattern:
    """Detected seasonal pattern."""
    period_minutes: int
    amplitude: float
    phase_offset: float
    confidence: float


# =============================================================================
# Exponential Smoothing Forecaster
# =============================================================================

class HoltWintersForecaster:
    """
    Holt-Winters triple exponential smoothing forecaster.

    Handles level, trend, and seasonality components.
    """

    def __init__(
        self,
        alpha: float = SMOOTHING_ALPHA,
        beta: float = TREND_BETA,
        gamma: float = SEASONAL_GAMMA,
        seasonal_period: int = SEASONAL_CYCLE,
    ):
        self._alpha = alpha  # Level smoothing
        self._beta = beta    # Trend smoothing
        self._gamma = gamma  # Seasonal smoothing
        self._period = seasonal_period

        # State
        self._level: float = 0.0
        self._trend: float = 0.0
        self._seasonals: List[float] = [0.0] * seasonal_period
        self._initialized = False
        self._history: Deque[float] = deque(maxlen=HISTORY_WINDOW)

    def update(self, value: float) -> None:
        """Update the model with a new observation."""
        self._history.append(value)

        if not self._initialized:
            if len(self._history) >= self._period * 2:
                self._initialize()
            return

        # Get current seasonal index
        idx = len(self._history) % self._period

        # Update level
        old_level = self._level
        self._level = self._alpha * (value - self._seasonals[idx]) + \
                      (1 - self._alpha) * (old_level + self._trend)

        # Update trend
        self._trend = self._beta * (self._level - old_level) + \
                      (1 - self._beta) * self._trend

        # Update seasonal
        self._seasonals[idx] = self._gamma * (value - self._level) + \
                               (1 - self._gamma) * self._seasonals[idx]

    def _initialize(self) -> None:
        """Initialize the model from history."""
        data = list(self._history)

        # Initial level: average of first season
        self._level = statistics.mean(data[:self._period])

        # Initial trend: average difference between seasons
        if len(data) >= self._period * 2:
            season1 = statistics.mean(data[:self._period])
            season2 = statistics.mean(data[self._period:self._period * 2])
            self._trend = (season2 - season1) / self._period
        else:
            self._trend = 0.0

        # Initial seasonals: deviations from level
        for i in range(self._period):
            if i < len(data):
                self._seasonals[i] = data[i] - self._level
            else:
                self._seasonals[i] = 0.0

        self._initialized = True

    def forecast(self, steps: int = 1) -> List[float]:
        """Forecast future values."""
        if not self._initialized:
            # Not enough data - return current average
            if self._history:
                avg = statistics.mean(self._history)
                return [avg] * steps
            return [0.0] * steps

        forecasts = []
        for h in range(1, steps + 1):
            idx = (len(self._history) + h) % self._period
            pred = self._level + h * self._trend + self._seasonals[idx]
            forecasts.append(pred)

        return forecasts

    def get_trend_direction(self) -> Tuple[str, float]:
        """Get current trend direction and strength."""
        if not self._initialized:
            return "stable", 0.0

        # Normalize trend relative to level
        if self._level == 0:
            return "stable", 0.0

        normalized_trend = self._trend / max(abs(self._level), 1.0)

        if normalized_trend > 0.01:
            return "increasing", min(abs(normalized_trend) * 10, 1.0)
        elif normalized_trend < -0.01:
            return "decreasing", min(abs(normalized_trend) * 10, 1.0)
        else:
            return "stable", 0.0


# =============================================================================
# Anomaly Detector
# =============================================================================

class AnomalyDetector:
    """
    Z-score based anomaly detection.

    Detects values that deviate significantly from the norm.
    """

    def __init__(
        self,
        threshold: float = ANOMALY_ZSCORE_THRESHOLD,
        window_size: int = 60,
    ):
        self._threshold = threshold
        self._window: Deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> bool:
        """
        Update with new value and check for anomaly.

        Returns True if value is anomalous.
        """
        is_anomaly = self.is_anomaly(value)
        self._window.append(value)
        return is_anomaly

    def is_anomaly(self, value: float) -> bool:
        """Check if value is anomalous (without adding to history)."""
        if len(self._window) < 10:
            return False

        mean = statistics.mean(self._window)
        stdev = statistics.stdev(self._window) if len(self._window) > 1 else 1.0

        if stdev == 0:
            return False

        zscore = abs(value - mean) / stdev
        return zscore > self._threshold

    def get_zscore(self, value: float) -> float:
        """Get z-score for a value."""
        if len(self._window) < 2:
            return 0.0

        mean = statistics.mean(self._window)
        stdev = statistics.stdev(self._window)

        if stdev == 0:
            return 0.0

        return (value - mean) / stdev


# =============================================================================
# Pattern Detector
# =============================================================================

class PatternDetector:
    """
    Detects recurring patterns in time-series data.

    Uses autocorrelation to find seasonal patterns.
    """

    def __init__(self, max_period: int = 1440):  # Up to 24 hours
        self._max_period = max_period
        self._data: Deque[float] = deque(maxlen=max_period * 2)

    def update(self, value: float) -> None:
        """Add a data point."""
        self._data.append(value)

    def detect_patterns(self) -> List[SeasonalPattern]:
        """Detect seasonal patterns in the data."""
        if len(self._data) < 60:  # Need at least 1 hour of data
            return []

        data = list(self._data)
        patterns = []

        # Check common periods: 15min, 30min, 1hr, 2hr, 4hr, 6hr, 12hr, 24hr
        periods_to_check = [15, 30, 60, 120, 240, 360, 720, 1440]

        for period in periods_to_check:
            if len(data) < period * 2:
                continue

            # Calculate autocorrelation at this lag
            corr = self._autocorrelation(data, period)

            if corr > 0.5:  # Strong correlation indicates pattern
                amplitude = self._estimate_amplitude(data, period)
                phase = self._estimate_phase(data, period)

                patterns.append(SeasonalPattern(
                    period_minutes=period,
                    amplitude=amplitude,
                    phase_offset=phase,
                    confidence=corr,
                ))

        return patterns

    def _autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        n = len(data)
        if lag >= n:
            return 0.0

        mean = statistics.mean(data)
        var = statistics.variance(data) if len(data) > 1 else 1.0

        if var == 0:
            return 0.0

        cov = sum(
            (data[i] - mean) * (data[i + lag] - mean)
            for i in range(n - lag)
        ) / (n - lag)

        return cov / var

    def _estimate_amplitude(self, data: List[float], period: int) -> float:
        """Estimate amplitude of seasonal pattern."""
        n = len(data)
        num_periods = n // period

        if num_periods < 1:
            return 0.0

        # Average amplitude across complete periods
        amplitudes = []
        for i in range(num_periods):
            start = i * period
            end = start + period
            period_data = data[start:end]
            amplitude = max(period_data) - min(period_data)
            amplitudes.append(amplitude)

        return statistics.mean(amplitudes)

    def _estimate_phase(self, data: List[float], period: int) -> float:
        """Estimate phase offset of seasonal pattern."""
        # Find position of maximum within first period
        first_period = data[:period]
        max_idx = first_period.index(max(first_period))
        return max_idx / period  # Normalized phase (0-1)


# =============================================================================
# Predictive Provisioner
# =============================================================================

class PredictiveProvisioner:
    """
    Main predictive provisioning manager.

    Combines forecasting, anomaly detection, and pattern recognition
    to make intelligent scaling recommendations.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
    ):
        self._redis = redis_client

        # Per-metric forecasters
        self._forecasters: Dict[str, HoltWintersForecaster] = {}
        self._anomaly_detectors: Dict[str, AnomalyDetector] = {}
        self._pattern_detectors: Dict[str, PatternDetector] = {}

        # Current VM state
        self._current_vm_count: int = 1
        self._vm_capacity: float = 100.0  # Capacity per VM

        # Historical recommendations
        self._recommendation_history: Deque[ScalingRecommendation] = deque(maxlen=100)

        # Metrics
        self._metrics_recorded = 0
        self._forecasts_made = 0
        self._recommendations_made = 0

        # Background task
        self._running = False
        self._analysis_task: Optional[asyncio.Task] = None

        logger.info("PredictiveProvisioner initialized")

    async def start(self) -> None:
        """Start background analysis task."""
        self._running = True
        self._analysis_task = asyncio.create_task(
            self._periodic_analysis(),
            name="predictive_analysis",
        )
        logger.info("PredictiveProvisioner started")

    async def stop(self) -> None:
        """Stop background tasks."""
        self._running = False
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("PredictiveProvisioner stopped")

    def _get_or_create_forecaster(self, metric_name: str) -> HoltWintersForecaster:
        """Get or create a forecaster for a metric."""
        if metric_name not in self._forecasters:
            self._forecasters[metric_name] = HoltWintersForecaster()
            self._anomaly_detectors[metric_name] = AnomalyDetector()
            self._pattern_detectors[metric_name] = PatternDetector()
        return self._forecasters[metric_name]

    async def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Current value
            timestamp: Optional timestamp (defaults to now)
        """
        forecaster = self._get_or_create_forecaster(metric_name)
        forecaster.update(value)

        anomaly_detector = self._anomaly_detectors[metric_name]
        anomaly_detector.update(value)

        pattern_detector = self._pattern_detectors[metric_name]
        pattern_detector.update(value)

        self._metrics_recorded += 1

        # Store in Redis if available
        if self._redis:
            try:
                key = f"metrics:{metric_name}:latest"
                await self._redis.set(key, json.dumps({
                    "value": value,
                    "timestamp": timestamp or time.time(),
                }))
            except Exception as e:
                logger.debug(f"Failed to store metric in Redis: {e}")

    async def forecast_metric(
        self,
        metric_name: str,
        horizon_minutes: int = FORECAST_HORIZON,
    ) -> Optional[Forecast]:
        """
        Forecast future values for a metric.

        Args:
            metric_name: Name of the metric
            horizon_minutes: How far ahead to forecast

        Returns:
            Forecast object or None if insufficient data
        """
        if metric_name not in self._forecasters:
            return None

        forecaster = self._forecasters[metric_name]
        anomaly_detector = self._anomaly_detectors[metric_name]

        # Get forecasts
        predictions = forecaster.forecast(horizon_minutes)
        if not predictions:
            return None

        predicted_value = predictions[-1]  # Value at horizon

        # Get current value and trend
        current_value = forecaster._history[-1] if forecaster._history else 0.0
        trend_direction, trend_strength = forecaster.get_trend_direction()

        # Calculate confidence interval
        if len(forecaster._history) > 1:
            stdev = statistics.stdev(forecaster._history)
            z_score = 1.96  # 95% confidence
            margin = z_score * stdev * math.sqrt(horizon_minutes)
        else:
            margin = 0.0

        # Check for anomaly
        is_anomaly = anomaly_detector.is_anomaly(current_value)

        self._forecasts_made += 1

        return Forecast(
            metric_name=metric_name,
            current_value=current_value,
            predicted_value=predicted_value,
            predicted_at=time.time() + horizon_minutes * 60,
            horizon_minutes=horizon_minutes,
            confidence_lower=predicted_value - margin,
            confidence_upper=predicted_value + margin,
            trend=trend_direction,
            trend_strength=trend_strength,
            anomaly_detected=is_anomaly,
        )

    async def get_recommendation(
        self,
        metrics: Optional[List[str]] = None,
    ) -> ScalingRecommendation:
        """
        Get a scaling recommendation based on forecasts.

        Args:
            metrics: List of metrics to consider (default: all)

        Returns:
            ScalingRecommendation
        """
        metrics_to_check = metrics or list(self._forecasters.keys())
        forecasts = []
        reasons = []

        # Forecast each metric
        for metric_name in metrics_to_check:
            forecast = await self.forecast_metric(metric_name)
            if forecast:
                forecasts.append(forecast)

        if not forecasts:
            return ScalingRecommendation(
                action=ScalingAction.NONE,
                recommended_count=0,
                confidence=0.0,
                reasons=["Insufficient data for forecasting"],
                forecasts=[],
                urgency="low",
            )

        # Analyze forecasts
        action = ScalingAction.NONE
        urgency = "low"
        recommended_count = 0
        time_to_threshold: Optional[float] = None

        for forecast in forecasts:
            # Check if approaching threshold
            if forecast.predicted_value >= SCALE_UP_THRESHOLD - SCALE_UP_MARGIN:
                if forecast.trend == "increasing":
                    # Calculate time to threshold
                    if forecast.trend_strength > 0:
                        remaining = SCALE_UP_THRESHOLD - forecast.current_value
                        rate = (forecast.predicted_value - forecast.current_value) / forecast.horizon_minutes
                        if rate > 0:
                            time_to_threshold = remaining / rate

                    action = ScalingAction.SCALE_UP
                    recommended_count = max(recommended_count, 1)

                    if forecast.predicted_value >= SCALE_UP_THRESHOLD:
                        urgency = "high"
                        recommended_count = max(recommended_count, 2)
                        reasons.append(
                            f"{forecast.metric_name} predicted to reach {forecast.predicted_value:.1f}% "
                            f"(threshold: {SCALE_UP_THRESHOLD}%)"
                        )
                    else:
                        urgency = "medium"
                        reasons.append(
                            f"{forecast.metric_name} trending up toward threshold"
                        )

            # Check for anomaly - may need emergency scale
            if forecast.anomaly_detected:
                if forecast.current_value > SCALE_UP_THRESHOLD:
                    action = ScalingAction.EMERGENCY_SCALE
                    urgency = "critical"
                    recommended_count = max(recommended_count, 3)
                    reasons.append(f"Anomalous spike detected in {forecast.metric_name}")

            # Check for scale down opportunity
            if forecast.predicted_value < SCALE_DOWN_THRESHOLD:
                if forecast.trend == "decreasing" and action == ScalingAction.NONE:
                    # Only suggest scale down if all metrics are low
                    all_low = all(
                        f.predicted_value < SCALE_DOWN_THRESHOLD
                        for f in forecasts
                    )
                    if all_low and self._current_vm_count > 1:
                        action = ScalingAction.SCALE_DOWN
                        recommended_count = 1
                        urgency = "low"
                        reasons.append("All metrics below scale-down threshold")

        # Calculate confidence based on forecast confidence intervals
        if forecasts:
            # Narrower confidence intervals = higher confidence
            avg_width = statistics.mean(
                f.confidence_upper - f.confidence_lower
                for f in forecasts
            ) / 100  # Normalize

            confidence = max(0.0, 1.0 - avg_width)
        else:
            confidence = 0.0

        self._recommendations_made += 1

        recommendation = ScalingRecommendation(
            action=action,
            recommended_count=recommended_count,
            confidence=confidence,
            reasons=reasons if reasons else ["No scaling needed"],
            forecasts=forecasts,
            urgency=urgency,
            estimated_time_to_threshold=time_to_threshold,
        )

        self._recommendation_history.append(recommendation)
        return recommendation

    async def _periodic_analysis(self) -> None:
        """Background task for periodic pattern analysis."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute

                # Detect patterns in each metric
                for metric_name, detector in self._pattern_detectors.items():
                    patterns = detector.detect_patterns()
                    if patterns:
                        logger.debug(
                            f"Detected patterns in {metric_name}: "
                            f"{[p.period_minutes for p in patterns]}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic analysis error: {e}")
                await asyncio.sleep(60)

    def set_vm_state(self, count: int, capacity_per_vm: float = 100.0) -> None:
        """Update current VM state."""
        self._current_vm_count = count
        self._vm_capacity = capacity_per_vm

    def get_patterns(self, metric_name: str) -> List[SeasonalPattern]:
        """Get detected patterns for a metric."""
        if metric_name in self._pattern_detectors:
            return self._pattern_detectors[metric_name].detect_patterns()
        return []

    def get_metrics(self) -> Dict[str, Any]:
        """Get provisioner metrics."""
        return {
            "metrics_recorded": self._metrics_recorded,
            "forecasts_made": self._forecasts_made,
            "recommendations_made": self._recommendations_made,
            "tracked_metrics": list(self._forecasters.keys()),
            "current_vm_count": self._current_vm_count,
            "recent_recommendations": [
                {
                    "action": r.action.value,
                    "count": r.recommended_count,
                    "urgency": r.urgency,
                }
                for r in list(self._recommendation_history)[-5:]
            ],
        }


# =============================================================================
# Global Factory
# =============================================================================

_provisioner_instance: Optional[PredictiveProvisioner] = None
_provisioner_lock = asyncio.Lock()


async def get_predictive_provisioner(
    redis_client: Optional[Any] = None,
) -> PredictiveProvisioner:
    """Get or create the global PredictiveProvisioner instance."""
    global _provisioner_instance

    async with _provisioner_lock:
        if _provisioner_instance is None:
            _provisioner_instance = PredictiveProvisioner(redis_client=redis_client)
            await _provisioner_instance.start()

        return _provisioner_instance


async def shutdown_predictive_provisioner() -> None:
    """Shutdown the global provisioner."""
    global _provisioner_instance

    if _provisioner_instance:
        await _provisioner_instance.stop()
        _provisioner_instance = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PredictiveProvisioner",
    "HoltWintersForecaster",
    "AnomalyDetector",
    "PatternDetector",
    "Forecast",
    "ScalingRecommendation",
    "ScalingAction",
    "SeasonalPattern",
    "get_predictive_provisioner",
    "shutdown_predictive_provisioner",
]
