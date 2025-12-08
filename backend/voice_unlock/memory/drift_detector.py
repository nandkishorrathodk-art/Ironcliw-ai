"""
Voice Drift Detector

Intelligent detection and analysis of voice evolution over time.
Identifies natural drift, illness, equipment changes, and anomalies.

Features:
- Statistical drift analysis with multiple methods
- Automatic baseline adaptation
- Drift classification and cause inference
- Trend detection and prediction
- Confidence adjustment recommendations
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

from .schemas import (
    VoiceMemoryConfig,
    VoiceEvolutionRecord,
    DriftType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class DriftConfig:
    """Environment-driven configuration for drift detection."""

    @staticmethod
    def get_drift_threshold() -> float:
        """Threshold for significant drift."""
        return float(os.getenv("VOICE_DRIFT_THRESHOLD", "0.05"))

    @staticmethod
    def get_severe_drift_threshold() -> float:
        """Threshold for severe drift requiring action."""
        return float(os.getenv("VOICE_DRIFT_SEVERE_THRESHOLD", "0.15"))

    @staticmethod
    def get_adaptation_rate() -> float:
        """Rate of baseline adaptation (0-1)."""
        return float(os.getenv("VOICE_DRIFT_ADAPTATION_RATE", "0.10"))

    @staticmethod
    def get_min_samples_for_analysis() -> int:
        """Minimum samples needed for drift analysis."""
        return int(os.getenv("VOICE_DRIFT_MIN_SAMPLES", "5"))

    @staticmethod
    def get_analysis_window_days() -> int:
        """Default analysis window in days."""
        return int(os.getenv("VOICE_DRIFT_ANALYSIS_WINDOW_DAYS", "30"))

    @staticmethod
    def get_short_term_window_hours() -> int:
        """Short-term drift window in hours."""
        return int(os.getenv("VOICE_DRIFT_SHORT_TERM_HOURS", "24"))

    @staticmethod
    def get_seasonal_period_days() -> int:
        """Period for seasonal drift detection."""
        return int(os.getenv("VOICE_DRIFT_SEASONAL_PERIOD_DAYS", "90"))

    @staticmethod
    def get_enable_auto_adaptation() -> bool:
        """Whether to automatically adapt baseline."""
        return os.getenv("VOICE_DRIFT_AUTO_ADAPTATION", "true").lower() == "true"

    @staticmethod
    def get_illness_drift_threshold() -> float:
        """Drift threshold suggesting illness."""
        return float(os.getenv("VOICE_DRIFT_ILLNESS_THRESHOLD", "0.08"))

    @staticmethod
    def get_equipment_drift_threshold() -> float:
        """Drift threshold suggesting equipment change."""
        return float(os.getenv("VOICE_DRIFT_EQUIPMENT_THRESHOLD", "0.12"))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DriftAnalysisResult:
    """Result of drift analysis."""

    # Primary metrics
    drift_detected: bool = False
    drift_magnitude: float = 0.0
    drift_type: DriftType = DriftType.NONE
    drift_direction: str = "stable"  # "improving", "degrading", "stable", "oscillating"

    # Temporal analysis
    short_term_drift: float = 0.0  # Last 24 hours
    medium_term_drift: float = 0.0  # Last 7 days
    long_term_drift: float = 0.0  # Last 30 days

    # Statistical measures
    mean_similarity: float = 0.0
    similarity_std: float = 0.0
    similarity_trend: float = 0.0  # Positive = improving
    similarity_min: float = 0.0
    similarity_max: float = 0.0

    # Cause inference
    probable_cause: str = "unknown"
    cause_confidence: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)

    # Recommendations
    should_adapt_baseline: bool = False
    recommended_adaptation_weight: float = 0.0
    confidence_adjustment: float = 0.0
    should_alert_user: bool = False
    alert_message: str = ""

    # Metadata
    samples_analyzed: int = 0
    analysis_window_days: int = 0
    analysis_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "drift_detected": self.drift_detected,
            "drift_magnitude": self.drift_magnitude,
            "drift_type": self.drift_type.value,
            "drift_direction": self.drift_direction,
            "short_term_drift": self.short_term_drift,
            "medium_term_drift": self.medium_term_drift,
            "long_term_drift": self.long_term_drift,
            "mean_similarity": self.mean_similarity,
            "similarity_std": self.similarity_std,
            "similarity_trend": self.similarity_trend,
            "probable_cause": self.probable_cause,
            "cause_confidence": self.cause_confidence,
            "contributing_factors": self.contributing_factors,
            "should_adapt_baseline": self.should_adapt_baseline,
            "recommended_adaptation_weight": self.recommended_adaptation_weight,
            "confidence_adjustment": self.confidence_adjustment,
            "samples_analyzed": self.samples_analyzed,
            "analysis_window_days": self.analysis_window_days,
        }


@dataclass
class BaselineState:
    """Current state of a user's voice baseline."""

    user_id: str
    embedding: List[float]
    embedding_model: str = "ecapa-tdnn"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    update_count: int = 0
    sample_count: int = 0

    # Quality metrics
    mean_similarity: float = 1.0
    similarity_variance: float = 0.0

    # Adaptation history
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        new_embedding: List[float],
        weight: float = 0.1,
    ) -> None:
        """Update baseline with new embedding."""
        if len(new_embedding) != len(self.embedding):
            raise ValueError("Embedding dimension mismatch")

        # Weighted average
        self.embedding = [
            old * (1 - weight) + new * weight
            for old, new in zip(self.embedding, new_embedding)
        ]

        # Normalize
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = [x / norm for x in self.embedding]

        self.last_updated = datetime.now(timezone.utc)
        self.update_count += 1

        # Record adaptation
        self.adaptation_history.append({
            "timestamp": self.last_updated.isoformat(),
            "weight": weight,
            "update_number": self.update_count,
        })

        # Keep only last 100 adaptations
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]


# =============================================================================
# VOICE DRIFT DETECTOR
# =============================================================================

class VoiceDriftDetector:
    """
    Intelligent voice drift detection and analysis.

    Monitors voice evolution over time and provides:
    - Drift magnitude and direction analysis
    - Cause inference (illness, equipment, natural aging)
    - Baseline adaptation recommendations
    - Confidence adjustments
    - User alerts for significant changes

    Usage:
        detector = await get_drift_detector()
        result = await detector.analyze_drift(user_id, samples)
    """

    def __init__(self):
        """Initialize the drift detector."""
        self._baselines: Dict[str, BaselineState] = {}
        self._analysis_cache: Dict[str, Tuple[DriftAnalysisResult, float]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "analyses": 0,
            "drifts_detected": 0,
            "adaptations": 0,
            "alerts_generated": 0,
        }

        logger.info("VoiceDriftDetector initialized")

    async def analyze_drift(
        self,
        user_id: str,
        samples: List[VoiceEvolutionRecord],
        baseline_embedding: Optional[List[float]] = None,
        window_days: Optional[int] = None,
    ) -> DriftAnalysisResult:
        """
        Analyze voice drift from a collection of samples.

        Args:
            user_id: User identifier
            samples: Voice evolution samples
            baseline_embedding: Optional baseline for comparison
            window_days: Analysis window override

        Returns:
            DriftAnalysisResult with comprehensive analysis
        """
        # Check cache
        cache_key = f"{user_id}:{len(samples)}"
        if cache_key in self._analysis_cache:
            cached_result, cached_time = self._analysis_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_result

        window = window_days or DriftConfig.get_analysis_window_days()
        result = DriftAnalysisResult(
            analysis_window_days=window,
            samples_analyzed=len(samples),
        )

        # Check minimum samples
        min_samples = DriftConfig.get_min_samples_for_analysis()
        if len(samples) < min_samples:
            result.probable_cause = "insufficient_data"
            return result

        try:
            # Sort samples by time
            sorted_samples = sorted(samples, key=lambda x: x.created_at)

            # Get or create baseline
            if baseline_embedding is None:
                baseline_embedding = await self._get_baseline(user_id, sorted_samples)

            # Calculate similarities
            similarities = [s.baseline_similarity for s in sorted_samples]

            # Basic statistics
            result.mean_similarity = float(np.mean(similarities))
            result.similarity_std = float(np.std(similarities))
            result.similarity_min = float(np.min(similarities))
            result.similarity_max = float(np.max(similarities))

            # Calculate drift magnitude
            result.drift_magnitude = 1.0 - result.mean_similarity

            # Temporal drift analysis
            result.short_term_drift = await self._calculate_temporal_drift(
                sorted_samples,
                hours=DriftConfig.get_short_term_window_hours(),
            )
            result.medium_term_drift = await self._calculate_temporal_drift(
                sorted_samples,
                hours=7 * 24,
            )
            result.long_term_drift = await self._calculate_temporal_drift(
                sorted_samples,
                hours=30 * 24,
            )

            # Trend analysis
            result.similarity_trend = await self._calculate_trend(sorted_samples)

            # Determine drift direction
            result.drift_direction = self._classify_direction(
                result.similarity_trend,
                result.similarity_std,
            )

            # Classify drift type
            result.drift_type = await self._classify_drift_type(
                sorted_samples,
                result,
            )

            # Check if drift is significant
            threshold = DriftConfig.get_drift_threshold()
            result.drift_detected = result.drift_magnitude >= threshold

            if result.drift_detected:
                self._stats["drifts_detected"] += 1

            # Infer cause
            result.probable_cause, result.cause_confidence = await self._infer_cause(
                sorted_samples,
                result,
            )
            result.contributing_factors = await self._identify_factors(
                sorted_samples,
                result,
            )

            # Generate recommendations
            await self._generate_recommendations(result)

            # Cache result
            self._analysis_cache[cache_key] = (result, time.time())
            self._stats["analyses"] += 1

            logger.debug(
                f"Drift analysis for {user_id}: "
                f"magnitude={result.drift_magnitude:.3f}, "
                f"type={result.drift_type.value}"
            )

            return result

        except Exception as e:
            logger.exception(f"Error analyzing drift: {e}")
            result.probable_cause = f"analysis_error: {str(e)}"
            return result

    async def _get_baseline(
        self,
        user_id: str,
        samples: List[VoiceEvolutionRecord],
    ) -> List[float]:
        """Get or compute baseline embedding."""
        async with self._lock:
            if user_id in self._baselines:
                return self._baselines[user_id].embedding

            # Compute from high-quality samples
            high_quality = [s for s in samples if s.is_high_quality]

            if high_quality:
                # Average of high-quality embeddings
                embeddings = np.array([s.embedding for s in high_quality])
                baseline = np.mean(embeddings, axis=0)
                # Normalize
                baseline = baseline / np.linalg.norm(baseline)

                self._baselines[user_id] = BaselineState(
                    user_id=user_id,
                    embedding=baseline.tolist(),
                    sample_count=len(high_quality),
                )

                return baseline.tolist()

            # Fall back to first sample
            if samples:
                return samples[0].embedding

            # Default zero embedding
            dim = VoiceMemoryConfig.get_embedding_dimension()
            return [0.0] * dim

    async def _calculate_temporal_drift(
        self,
        samples: List[VoiceEvolutionRecord],
        hours: int,
    ) -> float:
        """Calculate drift within a time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = [s for s in samples if s.created_at >= cutoff]

        if not recent:
            return 0.0

        similarities = [s.baseline_similarity for s in recent]
        return 1.0 - float(np.mean(similarities))

    async def _calculate_trend(
        self,
        samples: List[VoiceEvolutionRecord],
    ) -> float:
        """Calculate similarity trend over time."""
        if len(samples) < 2:
            return 0.0

        # Simple linear regression
        times = [s.created_at.timestamp() for s in samples]
        similarities = [s.baseline_similarity for s in samples]

        # Normalize times to [0, 1]
        t_min, t_max = min(times), max(times)
        if t_max == t_min:
            return 0.0

        normalized_times = [(t - t_min) / (t_max - t_min) for t in times]

        # Calculate slope
        n = len(samples)
        sum_x = sum(normalized_times)
        sum_y = sum(similarities)
        sum_xy = sum(t * s for t, s in zip(normalized_times, similarities))
        sum_xx = sum(t * t for t in normalized_times)

        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        return float(slope)

    def _classify_direction(
        self,
        trend: float,
        std: float,
    ) -> str:
        """Classify drift direction from trend and variance."""
        if abs(trend) < 0.01:
            return "stable"
        elif trend > 0.02:
            return "improving"
        elif trend < -0.02:
            return "degrading"
        elif std > 0.1:
            return "oscillating"
        else:
            return "stable"

    async def _classify_drift_type(
        self,
        samples: List[VoiceEvolutionRecord],
        result: DriftAnalysisResult,
    ) -> DriftType:
        """Classify the type of drift."""
        magnitude = result.drift_magnitude
        short_term = result.short_term_drift
        long_term = result.long_term_drift

        # Check for sudden drift (short-term >> long-term)
        if short_term > 2 * long_term and short_term > 0.10:
            return DriftType.SUDDEN

        # Check for equipment change (very sudden, specific pattern)
        if short_term > DriftConfig.get_equipment_drift_threshold():
            # Check device fingerprints
            devices = set(s.device_fingerprint for s in samples[-5:])
            if len(devices) > 1:
                return DriftType.EQUIPMENT

        # Check for illness (moderate sudden drift)
        if short_term > DriftConfig.get_illness_drift_threshold():
            # Check for audio quality patterns
            recent_snr = [s.snr_db for s in samples[-5:]]
            if any(snr < 10 for snr in recent_snr):
                return DriftType.ILLNESS

        # Check for recovering (trend improving after drift)
        if result.drift_direction == "improving" and magnitude > 0.05:
            return DriftType.RECOVERING

        # Check for gradual drift
        if magnitude < 0.05 and result.analysis_window_days > 30:
            return DriftType.GRADUAL

        # Check for environmental drift
        environments = set(s.environment_type for s in samples)
        if len(environments) > 2:
            return DriftType.ENVIRONMENTAL

        # Check for stress-related
        if 0.05 <= magnitude <= 0.10:
            return DriftType.STRESS

        # No significant drift
        if magnitude < DriftConfig.get_drift_threshold():
            return DriftType.NONE

        return DriftType.GRADUAL

    async def _infer_cause(
        self,
        samples: List[VoiceEvolutionRecord],
        result: DriftAnalysisResult,
    ) -> Tuple[str, float]:
        """Infer probable cause of drift."""
        if result.drift_type == DriftType.NONE:
            return "no_drift", 0.95

        # Evidence scores for different causes
        evidence: Dict[str, float] = {
            "natural_aging": 0.0,
            "illness": 0.0,
            "equipment_change": 0.0,
            "environment": 0.0,
            "stress": 0.0,
            "unknown": 0.2,  # Prior for unknown
        }

        # Check for equipment change
        if len(samples) >= 2:
            devices = [s.device_fingerprint for s in samples]
            if len(set(devices[-5:])) > len(set(devices[:-5])):
                evidence["equipment_change"] += 0.7

        # Check for environmental variation
        environments = [s.environment_type for s in samples]
        if len(set(environments[-5:])) > 2:
            evidence["environment"] += 0.5

        # Check for illness indicators
        recent_quality = [s.audio_quality_score for s in samples[-5:]]
        if any(q < 0.6 for q in recent_quality):
            evidence["illness"] += 0.4

        # Check time pattern for natural aging
        if result.analysis_window_days > 60 and result.drift_type == DriftType.GRADUAL:
            evidence["natural_aging"] += 0.6

        # Check for stress indicators
        if result.similarity_std > 0.15:
            evidence["stress"] += 0.3

        # Normalize
        total = sum(evidence.values())
        if total > 0:
            evidence = {k: v / total for k, v in evidence.items()}

        # Get most likely cause
        cause = max(evidence.keys(), key=lambda k: evidence[k])
        confidence = evidence[cause]

        return cause, confidence

    async def _identify_factors(
        self,
        samples: List[VoiceEvolutionRecord],
        result: DriftAnalysisResult,
    ) -> List[str]:
        """Identify contributing factors to drift."""
        factors = []

        # Time-based factors
        if result.short_term_drift > result.long_term_drift * 1.5:
            factors.append("recent_change")

        # Quality factors
        low_quality = [s for s in samples if s.audio_quality_score < 0.7]
        if len(low_quality) > len(samples) * 0.3:
            factors.append("audio_quality_issues")

        # Environmental factors
        environments = set(s.environment_type for s in samples)
        if len(environments) > 3:
            factors.append("varied_environments")

        # Device factors
        devices = set(s.device_fingerprint for s in samples if s.device_fingerprint)
        if len(devices) > 2:
            factors.append("multiple_devices")

        # SNR factors
        snr_values = [s.snr_db for s in samples]
        if any(snr < 10 for snr in snr_values):
            factors.append("low_snr_samples")

        return factors

    async def _generate_recommendations(
        self,
        result: DriftAnalysisResult,
    ) -> None:
        """Generate recommendations based on analysis."""
        magnitude = result.drift_magnitude
        drift_type = result.drift_type

        # Baseline adaptation recommendation
        if drift_type == DriftType.GRADUAL and magnitude < 0.10:
            result.should_adapt_baseline = DriftConfig.get_enable_auto_adaptation()
            result.recommended_adaptation_weight = min(0.15, magnitude * 2)
        elif drift_type == DriftType.RECOVERING and result.drift_direction == "improving":
            result.should_adapt_baseline = True
            result.recommended_adaptation_weight = 0.1

        # Confidence adjustment
        if magnitude < 0.03:
            result.confidence_adjustment = 0.0
        elif magnitude < 0.05:
            result.confidence_adjustment = -0.02
        elif magnitude < 0.10:
            result.confidence_adjustment = -0.05
        else:
            result.confidence_adjustment = -0.10

        # Alert generation
        if magnitude > DriftConfig.get_severe_drift_threshold():
            result.should_alert_user = True
            result.alert_message = (
                f"Significant voice change detected ({magnitude:.1%}). "
                f"Probable cause: {result.probable_cause}. "
                "Authentication may require additional verification."
            )
            self._stats["alerts_generated"] += 1
        elif drift_type in (DriftType.ILLNESS, DriftType.STRESS):
            result.should_alert_user = True
            result.alert_message = (
                f"Your voice sounds different today ({result.probable_cause}). "
                "I'll adjust authentication accordingly."
            )
            self._stats["alerts_generated"] += 1

    async def adapt_baseline(
        self,
        user_id: str,
        new_embedding: List[float],
        weight: Optional[float] = None,
    ) -> bool:
        """
        Adapt the baseline for a user.

        Args:
            user_id: User identifier
            new_embedding: New voice embedding
            weight: Adaptation weight (default from config)

        Returns:
            True if adaptation was performed
        """
        async with self._lock:
            if user_id not in self._baselines:
                self._baselines[user_id] = BaselineState(
                    user_id=user_id,
                    embedding=new_embedding,
                )
                logger.info(f"Created new baseline for {user_id}")
                return True

            adaptation_weight = weight or DriftConfig.get_adaptation_rate()
            self._baselines[user_id].update(new_embedding, adaptation_weight)
            self._stats["adaptations"] += 1

            logger.info(
                f"Adapted baseline for {user_id} with weight {adaptation_weight:.2f}"
            )
            return True

    async def get_baseline(self, user_id: str) -> Optional[BaselineState]:
        """Get baseline state for a user."""
        async with self._lock:
            return self._baselines.get(user_id)

    async def should_adapt(
        self,
        user_id: str,
        current_similarity: float,
        audio_quality: float,
    ) -> Tuple[bool, float]:
        """
        Determine if baseline should be adapted with current sample.

        Args:
            user_id: User identifier
            current_similarity: Similarity to current baseline
            audio_quality: Audio quality score

        Returns:
            Tuple of (should_adapt, recommended_weight)
        """
        if not DriftConfig.get_enable_auto_adaptation():
            return False, 0.0

        # Don't adapt with low-quality samples
        if audio_quality < 0.7:
            return False, 0.0

        # Don't adapt if similarity is too low (might be attacker)
        if current_similarity < 0.7:
            return False, 0.0

        # Adapt with high-quality, high-similarity samples
        if current_similarity >= 0.9 and audio_quality >= 0.85:
            weight = 0.05  # Small weight for good samples
            return True, weight

        # Adapt moderately for good samples
        if current_similarity >= 0.85 and audio_quality >= 0.8:
            weight = 0.03
            return True, weight

        return False, 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get drift detector statistics."""
        return {
            **self._stats,
            "baselines_tracked": len(self._baselines),
            "cache_size": len(self._analysis_cache),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_detector_instance: Optional[VoiceDriftDetector] = None
_detector_lock = asyncio.Lock()


async def get_drift_detector() -> VoiceDriftDetector:
    """Get or create the drift detector instance."""
    global _detector_instance

    async with _detector_lock:
        if _detector_instance is None:
            _detector_instance = VoiceDriftDetector()
        return _detector_instance


def create_drift_detector() -> VoiceDriftDetector:
    """Create a new drift detector instance."""
    return VoiceDriftDetector()


__all__ = [
    "VoiceDriftDetector",
    "DriftAnalysisResult",
    "DriftType",
    "DriftConfig",
    "BaselineState",
    "get_drift_detector",
    "create_drift_detector",
]
