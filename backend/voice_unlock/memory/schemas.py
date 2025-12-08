"""
Voice Pattern Memory Schemas

Enterprise-grade Pydantic models for voice pattern storage, retrieval,
and analysis. Designed for ChromaDB integration with rich metadata support.

Features:
- Comprehensive voice evolution tracking
- Behavioral pattern modeling
- Attack signature schemas
- Environmental profile storage
- Temporal query support
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    computed_field,
    ConfigDict,
)
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

class VoiceMemoryConfig:
    """Environment-driven configuration for voice pattern memory."""

    @staticmethod
    def get_memory_dir() -> str:
        """Directory for persistent memory storage."""
        default = os.path.expanduser("~/.jarvis/voice_pattern_memory")
        return os.getenv("VOICE_PATTERN_MEMORY_DIR", default)

    @staticmethod
    def get_collection_prefix() -> str:
        """Prefix for all ChromaDB collections."""
        return os.getenv("VOICE_MEMORY_COLLECTION_PREFIX", "jarvis_voice_")

    @staticmethod
    def get_embedding_dimension() -> int:
        """Voice embedding dimension (ECAPA-TDNN = 192)."""
        return int(os.getenv("VOICE_EMBEDDING_DIMENSION", "192"))

    @staticmethod
    def get_drift_threshold() -> float:
        """Threshold for significant voice drift."""
        return float(os.getenv("VOICE_DRIFT_THRESHOLD", "0.05"))

    @staticmethod
    def get_drift_adaptation_rate() -> float:
        """Rate at which baseline adapts to drift."""
        return float(os.getenv("VOICE_DRIFT_ADAPTATION_RATE", "0.10"))

    @staticmethod
    def get_max_history_days() -> int:
        """Maximum days of pattern history to retain."""
        return int(os.getenv("VOICE_MEMORY_MAX_HISTORY_DAYS", "365"))

    @staticmethod
    def get_min_patterns_for_analysis() -> int:
        """Minimum patterns needed for statistical analysis."""
        return int(os.getenv("VOICE_MEMORY_MIN_PATTERNS", "10"))

    @staticmethod
    def get_attack_similarity_threshold() -> float:
        """Similarity threshold for attack pattern matching."""
        return float(os.getenv("VOICE_ATTACK_SIMILARITY_THRESHOLD", "0.90"))

    @staticmethod
    def get_behavioral_confidence_boost() -> float:
        """Confidence boost from strong behavioral match."""
        return float(os.getenv("VOICE_BEHAVIORAL_CONFIDENCE_BOOST", "0.05"))

    @staticmethod
    def get_environmental_adaptation_factor() -> float:
        """Factor for environmental voiceprint adaptation."""
        return float(os.getenv("VOICE_ENV_ADAPTATION_FACTOR", "0.15"))

    @staticmethod
    def get_cache_ttl_seconds() -> int:
        """Cache TTL for pattern queries."""
        return int(os.getenv("VOICE_MEMORY_CACHE_TTL", "300"))

    @staticmethod
    def get_batch_size() -> int:
        """Batch size for bulk operations."""
        return int(os.getenv("VOICE_MEMORY_BATCH_SIZE", "100"))


# =============================================================================
# ENUMS
# =============================================================================

class DriftType(str, Enum):
    """Types of voice drift that can be detected."""

    NONE = "none"                      # No significant drift
    GRADUAL = "gradual"                # Slow, natural aging
    SEASONAL = "seasonal"              # Seasonal variations
    ILLNESS = "illness"                # Temporary illness-related
    STRESS = "stress"                  # Stress-induced changes
    ENVIRONMENTAL = "environmental"    # Environment adaptation
    EQUIPMENT = "equipment"            # New microphone/device
    SUDDEN = "sudden"                  # Sudden unexplained change
    RECOVERING = "recovering"          # Returning to baseline

    @classmethod
    def from_magnitude(
        cls,
        magnitude: float,
        days_span: int,
    ) -> "DriftType":
        """Classify drift type based on magnitude and timespan."""
        if magnitude < 0.02:
            return cls.NONE
        elif magnitude < 0.05 and days_span > 30:
            return cls.GRADUAL
        elif magnitude < 0.05 and days_span <= 30:
            return cls.SEASONAL
        elif magnitude >= 0.10:
            return cls.SUDDEN
        else:
            return cls.GRADUAL


class AttackType(str, Enum):
    """Types of spoofing attacks that can be detected."""

    REPLAY = "replay"                  # Recorded audio playback
    SYNTHESIS = "synthesis"            # AI-generated voice
    CONVERSION = "conversion"          # Voice conversion
    DEEPFAKE = "deepfake"             # Deep learning synthesis
    IMPERSONATION = "impersonation"   # Human impersonation
    CONCATENATION = "concatenation"   # Spliced audio segments
    UNKNOWN = "unknown"               # Unclassified attack

    @property
    def severity(self) -> int:
        """Get severity level (1-5)."""
        severity_map = {
            self.REPLAY: 3,
            self.SYNTHESIS: 5,
            self.CONVERSION: 4,
            self.DEEPFAKE: 5,
            self.IMPERSONATION: 2,
            self.CONCATENATION: 3,
            self.UNKNOWN: 4,
        }
        return severity_map.get(self, 3)


class EnvironmentType(str, Enum):
    """Types of acoustic environments."""

    QUIET_HOME = "quiet_home"
    QUIET_OFFICE = "quiet_office"
    NOISY_OFFICE = "noisy_office"
    OUTDOOR = "outdoor"
    CAFE = "cafe"
    CAR = "car"
    PUBLIC_TRANSPORT = "public_transport"
    UNKNOWN = "unknown"

    @property
    def typical_snr_range(self) -> Tuple[float, float]:
        """Get typical SNR range for this environment."""
        ranges = {
            self.QUIET_HOME: (25.0, 45.0),
            self.QUIET_OFFICE: (20.0, 35.0),
            self.NOISY_OFFICE: (10.0, 20.0),
            self.OUTDOOR: (5.0, 20.0),
            self.CAFE: (5.0, 15.0),
            self.CAR: (8.0, 18.0),
            self.PUBLIC_TRANSPORT: (3.0, 12.0),
            self.UNKNOWN: (5.0, 30.0),
        }
        return ranges.get(self, (5.0, 30.0))


# =============================================================================
# BASE MODELS
# =============================================================================

class VoiceMemoryRecord(BaseModel):
    """Base class for all voice memory records."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        extra="allow",
    )

    # Identification
    record_id: str = Field(default="", description="Unique record identifier")
    user_id: str = Field(..., description="User identifier")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record creation time"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Last update time"
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Record expiration time"
    )

    # Metadata
    source: str = Field(
        default="jarvis",
        description="Source system"
    )
    version: str = Field(
        default="1.0",
        description="Schema version"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Record tags"
    )

    @model_validator(mode="before")
    @classmethod
    def generate_record_id(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Generate record ID if not provided."""
        if not values.get("record_id"):
            user_id = values.get("user_id", "unknown")
            timestamp = datetime.now(timezone.utc).isoformat()
            content = f"{user_id}:{timestamp}:{id(values)}"
            values["record_id"] = hashlib.sha256(content.encode()).hexdigest()[:16]
        return values

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if record has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @computed_field
    @property
    def age_days(self) -> float:
        """Get record age in days."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 86400

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB-compatible metadata dict."""
        metadata = {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "version": self.version,
        }
        if self.updated_at:
            metadata["updated_at"] = self.updated_at.isoformat()
        if self.expires_at:
            metadata["expires_at"] = self.expires_at.isoformat()
        if self.tags:
            metadata["tags"] = ",".join(self.tags)
        return metadata


# =============================================================================
# VOICE EVOLUTION RECORD
# =============================================================================

class VoiceEvolutionRecord(VoiceMemoryRecord):
    """
    Record of voice evolution over time.

    Tracks how a user's voice changes, enabling:
    - Natural drift detection and adaptation
    - Illness/stress detection
    - Equipment change detection
    - Baseline updates
    """

    # Voice embedding
    embedding: List[float] = Field(
        ...,
        description="Voice embedding vector"
    )
    embedding_model: str = Field(
        default="ecapa-tdnn",
        description="Model used for embedding"
    )

    # Comparison to baseline
    baseline_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity to user baseline"
    )
    drift_magnitude: float = Field(
        default=0.0,
        ge=0.0,
        description="Magnitude of drift from baseline"
    )
    drift_type: DriftType = Field(
        default=DriftType.NONE,
        description="Classified drift type"
    )

    # Audio quality
    audio_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Audio quality score"
    )
    snr_db: float = Field(
        default=0.0,
        description="Signal-to-noise ratio in dB"
    )

    # Context
    environment_type: EnvironmentType = Field(
        default=EnvironmentType.UNKNOWN,
        description="Recording environment"
    )
    device_fingerprint: str = Field(
        default="",
        description="Recording device fingerprint"
    )

    # Statistics
    sample_duration_ms: int = Field(
        default=0,
        ge=0,
        description="Audio sample duration"
    )
    confidence_at_recording: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Verification confidence when recorded"
    )

    # Adaptation
    was_used_for_adaptation: bool = Field(
        default=False,
        description="Whether this sample was used to update baseline"
    )
    adaptation_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weight given in baseline adaptation"
    )

    @computed_field
    @property
    def is_high_quality(self) -> bool:
        """Check if this is a high-quality sample for adaptation."""
        return (
            self.audio_quality_score >= 0.8 and
            self.snr_db >= 15.0 and
            self.confidence_at_recording >= 0.85 and
            self.sample_duration_ms >= 2000
        )

    @computed_field
    @property
    def drift_severity(self) -> str:
        """Get drift severity classification."""
        if self.drift_magnitude < 0.02:
            return "none"
        elif self.drift_magnitude < 0.05:
            return "minor"
        elif self.drift_magnitude < 0.10:
            return "moderate"
        else:
            return "significant"

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata."""
        base = super().to_chromadb_metadata()
        base.update({
            "embedding_model": self.embedding_model,
            "baseline_similarity": self.baseline_similarity,
            "drift_magnitude": self.drift_magnitude,
            "drift_type": self.drift_type.value,
            "audio_quality_score": self.audio_quality_score,
            "snr_db": self.snr_db,
            "environment_type": self.environment_type.value,
            "device_fingerprint": self.device_fingerprint,
            "sample_duration_ms": self.sample_duration_ms,
            "confidence_at_recording": self.confidence_at_recording,
            "is_high_quality": self.is_high_quality,
        })
        return base


# =============================================================================
# BEHAVIORAL PATTERN RECORD
# =============================================================================

class BehavioralPatternRecord(VoiceMemoryRecord):
    """
    Record of behavioral unlock patterns.

    Captures when, where, and how a user authenticates,
    enabling behavioral biometric verification.
    """

    # Temporal patterns
    hour_of_day: int = Field(
        ...,
        ge=0,
        lt=24,
        description="Hour of authentication (0-23)"
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        lt=7,
        description="Day of week (0=Monday)"
    )
    is_weekend: bool = Field(
        default=False,
        description="Whether this is a weekend"
    )
    time_since_last_auth_seconds: Optional[int] = Field(
        default=None,
        ge=0,
        description="Seconds since last authentication"
    )

    # Location patterns
    wifi_network_hash: str = Field(
        default="",
        description="Hash of WiFi network SSID"
    )
    location_cluster_id: str = Field(
        default="",
        description="Cluster ID for location"
    )
    is_known_location: bool = Field(
        default=False,
        description="Whether location is known/trusted"
    )

    # Device patterns
    device_id: str = Field(
        default="",
        description="Device identifier"
    )
    microphone_type: str = Field(
        default="unknown",
        description="Type of microphone used"
    )

    # Speech patterns
    phrase_used: str = Field(
        default="",
        description="Unlock phrase used"
    )
    speech_rate_wpm: float = Field(
        default=0.0,
        ge=0.0,
        description="Speech rate in words per minute"
    )

    # Authentication outcome
    authentication_result: str = Field(
        default="unknown",
        description="Result: success, failure, challenge"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final confidence score"
    )

    # Pattern frequency
    pattern_frequency: int = Field(
        default=1,
        ge=1,
        description="Times this pattern has occurred"
    )
    pattern_hash: str = Field(
        default="",
        description="Hash of this behavioral pattern"
    )

    @model_validator(mode="before")
    @classmethod
    def compute_pattern_hash(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Compute hash for this behavioral pattern."""
        if not values.get("pattern_hash"):
            components = [
                str(values.get("hour_of_day", 0)),
                str(values.get("day_of_week", 0)),
                values.get("wifi_network_hash", ""),
                values.get("device_id", ""),
                values.get("microphone_type", ""),
            ]
            content = ":".join(components)
            values["pattern_hash"] = hashlib.sha256(content.encode()).hexdigest()[:16]
        return values

    @computed_field
    @property
    def time_category(self) -> str:
        """Categorize time of day."""
        if 5 <= self.hour_of_day < 9:
            return "early_morning"
        elif 9 <= self.hour_of_day < 12:
            return "morning"
        elif 12 <= self.hour_of_day < 14:
            return "midday"
        elif 14 <= self.hour_of_day < 18:
            return "afternoon"
        elif 18 <= self.hour_of_day < 22:
            return "evening"
        else:
            return "night"

    @computed_field
    @property
    def is_typical_time(self) -> bool:
        """Check if this is a typical authentication time."""
        # Most unlocks happen during waking hours
        return 6 <= self.hour_of_day <= 23

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata."""
        base = super().to_chromadb_metadata()
        base.update({
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "wifi_network_hash": self.wifi_network_hash,
            "location_cluster_id": self.location_cluster_id,
            "is_known_location": self.is_known_location,
            "device_id": self.device_id,
            "microphone_type": self.microphone_type,
            "authentication_result": self.authentication_result,
            "confidence_score": self.confidence_score,
            "pattern_frequency": self.pattern_frequency,
            "pattern_hash": self.pattern_hash,
            "time_category": self.time_category,
        })
        return base


# =============================================================================
# ATTACK PATTERN RECORD
# =============================================================================

class AttackPatternRecord(VoiceMemoryRecord):
    """
    Record of detected attack/spoofing attempts.

    Stores failed spoofing attempts for pattern matching
    and improved detection.
    """

    # Attack classification
    attack_type: AttackType = Field(
        ...,
        description="Classified attack type"
    )
    attack_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in attack classification"
    )

    # Audio fingerprint for matching
    audio_fingerprint: List[float] = Field(
        default_factory=list,
        description="Audio fingerprint for similarity search"
    )
    audio_hash: str = Field(
        default="",
        description="Hash of audio content"
    )

    # Detection scores (from anti-spoofing)
    replay_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Replay detection score"
    )
    synthesis_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Synthesis detection score"
    )
    liveness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Liveness detection score"
    )
    physics_anomaly_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Physics-based anomaly score"
    )

    # Attack context
    target_user_id: str = Field(
        default="",
        description="User being impersonated"
    )
    voice_similarity_to_target: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Similarity to target user's voice"
    )

    # Detection details
    detection_method: str = Field(
        default="multi_layer",
        description="Method that detected the attack"
    )
    detection_layer: int = Field(
        default=0,
        ge=0,
        description="Anti-spoofing layer that triggered"
    )
    detection_latency_ms: int = Field(
        default=0,
        ge=0,
        description="Time to detect attack"
    )

    # Frequency tracking
    similar_attacks_count: int = Field(
        default=0,
        ge=0,
        description="Count of similar attacks seen"
    )
    last_similar_attack: Optional[datetime] = Field(
        default=None,
        description="Last similar attack timestamp"
    )

    @computed_field
    @property
    def severity_score(self) -> float:
        """Calculate attack severity score."""
        base_severity = self.attack_type.severity / 5.0
        confidence_factor = self.attack_confidence
        similarity_factor = self.voice_similarity_to_target

        return min(1.0, base_severity * 0.4 + confidence_factor * 0.4 + similarity_factor * 0.2)

    @computed_field
    @property
    def is_sophisticated(self) -> bool:
        """Check if this is a sophisticated attack."""
        return (
            self.attack_type in (AttackType.SYNTHESIS, AttackType.DEEPFAKE, AttackType.CONVERSION) or
            self.voice_similarity_to_target >= 0.7
        )

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata."""
        base = super().to_chromadb_metadata()
        base.update({
            "attack_type": self.attack_type.value,
            "attack_confidence": self.attack_confidence,
            "audio_hash": self.audio_hash,
            "replay_score": self.replay_score,
            "synthesis_score": self.synthesis_score,
            "liveness_score": self.liveness_score,
            "physics_anomaly_score": self.physics_anomaly_score,
            "target_user_id": self.target_user_id,
            "voice_similarity_to_target": self.voice_similarity_to_target,
            "detection_method": self.detection_method,
            "detection_layer": self.detection_layer,
            "severity_score": self.severity_score,
            "is_sophisticated": self.is_sophisticated,
        })
        return base


# =============================================================================
# ENVIRONMENTAL PROFILE RECORD
# =============================================================================

class EnvironmentalProfileRecord(VoiceMemoryRecord):
    """
    Environment-specific voice profile.

    Stores adapted voiceprints for different acoustic environments,
    improving authentication in noisy/varying conditions.
    """

    # Environment identification
    environment_hash: str = Field(
        ...,
        description="Hash identifying this environment"
    )
    environment_type: EnvironmentType = Field(
        default=EnvironmentType.UNKNOWN,
        description="Classified environment type"
    )
    environment_name: str = Field(
        default="",
        description="Human-readable environment name"
    )

    # Acoustic characteristics
    typical_snr_db: float = Field(
        default=0.0,
        description="Typical SNR in this environment"
    )
    snr_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="SNR variance in this environment"
    )
    typical_noise_floor_db: float = Field(
        default=-60.0,
        description="Typical noise floor level"
    )

    # Adapted voiceprint
    adapted_embedding: List[float] = Field(
        default_factory=list,
        description="Environment-adapted voice embedding"
    )
    baseline_embedding: List[float] = Field(
        default_factory=list,
        description="Original baseline embedding"
    )
    adaptation_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Adaptation factor applied"
    )

    # Confidence adjustments
    confidence_adjustment: float = Field(
        default=0.0,
        description="Confidence adjustment for this environment"
    )
    threshold_adjustment: float = Field(
        default=0.0,
        description="Threshold adjustment for this environment"
    )

    # Statistics
    sample_count: int = Field(
        default=0,
        ge=0,
        description="Number of samples from this environment"
    )
    success_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Authentication success rate"
    )
    average_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average confidence in this environment"
    )

    @computed_field
    @property
    def is_challenging(self) -> bool:
        """Check if this is a challenging environment."""
        return (
            self.typical_snr_db < 15.0 or
            self.success_rate < 0.8 or
            self.environment_type in (
                EnvironmentType.CAFE,
                EnvironmentType.PUBLIC_TRANSPORT,
                EnvironmentType.OUTDOOR,
            )
        )

    @computed_field
    @property
    def recommended_retry_count(self) -> int:
        """Get recommended retry count for this environment."""
        if self.typical_snr_db >= 20.0:
            return 1
        elif self.typical_snr_db >= 10.0:
            return 2
        else:
            return 3

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata."""
        base = super().to_chromadb_metadata()
        base.update({
            "environment_hash": self.environment_hash,
            "environment_type": self.environment_type.value,
            "environment_name": self.environment_name,
            "typical_snr_db": self.typical_snr_db,
            "snr_variance": self.snr_variance,
            "confidence_adjustment": self.confidence_adjustment,
            "sample_count": self.sample_count,
            "success_rate": self.success_rate,
            "average_confidence": self.average_confidence,
            "is_challenging": self.is_challenging,
        })
        return base


# =============================================================================
# SPEECH BIOMETRICS RECORD
# =============================================================================

class SpeechBiometricsRecord(VoiceMemoryRecord):
    """
    Speech-level biometric patterns.

    Captures rhythm, cadence, and micro-patterns that are
    unique to a speaker beyond voice embedding.
    """

    # Speech rate
    words_per_minute: float = Field(
        default=0.0,
        ge=0.0,
        description="Speaking rate in WPM"
    )
    syllables_per_second: float = Field(
        default=0.0,
        ge=0.0,
        description="Syllable rate"
    )

    # Pause patterns
    pause_frequency: float = Field(
        default=0.0,
        ge=0.0,
        description="Pauses per minute"
    )
    average_pause_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Average pause duration"
    )
    pause_duration_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Variance in pause duration"
    )

    # Pitch patterns
    mean_pitch_hz: float = Field(
        default=0.0,
        ge=0.0,
        description="Mean fundamental frequency"
    )
    pitch_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Pitch variance"
    )
    pitch_range_hz: float = Field(
        default=0.0,
        ge=0.0,
        description="Pitch range"
    )

    # Energy patterns
    mean_energy_db: float = Field(
        default=0.0,
        description="Mean energy level"
    )
    energy_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Energy variance"
    )

    # Micro-patterns
    jitter_percent: float = Field(
        default=0.0,
        ge=0.0,
        description="Pitch jitter percentage"
    )
    shimmer_percent: float = Field(
        default=0.0,
        ge=0.0,
        description="Amplitude shimmer percentage"
    )
    harmonics_to_noise_ratio: float = Field(
        default=0.0,
        description="HNR in dB"
    )

    # Breathing patterns
    breath_rate_per_minute: float = Field(
        default=0.0,
        ge=0.0,
        description="Detected breath rate"
    )
    breath_pattern_regularity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Regularity of breathing"
    )

    # Context
    phrase_text: str = Field(
        default="",
        description="Text that was spoken"
    )
    emotional_tone: str = Field(
        default="neutral",
        description="Detected emotional tone"
    )

    @computed_field
    @property
    def speech_pattern_vector(self) -> List[float]:
        """Get normalized speech pattern feature vector."""
        # Normalize features for comparison
        return [
            self.words_per_minute / 200.0,  # Normalize to ~150-180 WPM typical
            self.pause_frequency / 20.0,    # Normalize pauses
            self.mean_pitch_hz / 300.0,     # Normalize pitch
            self.pitch_variance / 50.0,     # Normalize variance
            self.jitter_percent / 5.0,      # Normalize jitter
            self.shimmer_percent / 10.0,    # Normalize shimmer
        ]

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata."""
        base = super().to_chromadb_metadata()
        base.update({
            "words_per_minute": self.words_per_minute,
            "pause_frequency": self.pause_frequency,
            "mean_pitch_hz": self.mean_pitch_hz,
            "pitch_variance": self.pitch_variance,
            "jitter_percent": self.jitter_percent,
            "shimmer_percent": self.shimmer_percent,
            "emotional_tone": self.emotional_tone,
        })
        return base


# =============================================================================
# AUTHENTICATION EVENT RECORD
# =============================================================================

class AuthenticationEventRecord(VoiceMemoryRecord):
    """
    Complete record of an authentication event.

    Captures full context for audit trails, debugging,
    and system improvement.
    """

    # Event identification
    session_id: str = Field(
        ...,
        description="Authentication session ID"
    )
    attempt_number: int = Field(
        default=1,
        ge=1,
        description="Attempt number in session"
    )

    # Result
    decision: str = Field(
        ...,
        description="Final decision: authenticate, reject, challenge, etc."
    )
    final_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final confidence score"
    )

    # Factor scores
    ml_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="ML verification confidence"
    )
    physics_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Physics analysis confidence"
    )
    behavioral_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Behavioral analysis confidence"
    )
    context_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Context analysis confidence"
    )

    # Timing
    total_duration_ms: int = Field(
        default=0,
        ge=0,
        description="Total authentication duration"
    )
    phase_durations: Dict[str, float] = Field(
        default_factory=dict,
        description="Duration per phase"
    )

    # Reasoning
    reasoning_used: bool = Field(
        default=False,
        description="Whether deep reasoning was used"
    )
    hypotheses_generated: int = Field(
        default=0,
        ge=0,
        description="Number of hypotheses generated"
    )
    primary_hypothesis: str = Field(
        default="",
        description="Primary hypothesis if reasoning was used"
    )
    reasoning_chain: List[str] = Field(
        default_factory=list,
        description="Chain of reasoning thoughts"
    )

    # Context
    environment_type: EnvironmentType = Field(
        default=EnvironmentType.UNKNOWN,
        description="Environment type"
    )
    audio_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Audio quality"
    )

    # Response
    response_text: str = Field(
        default="",
        description="Response given to user"
    )
    response_category: str = Field(
        default="standard",
        description="Category of response"
    )

    # Anomalies
    anomalies_detected: List[str] = Field(
        default_factory=list,
        description="List of anomalies detected"
    )
    spoofing_suspected: bool = Field(
        default=False,
        description="Whether spoofing was suspected"
    )

    @computed_field
    @property
    def is_fast_path(self) -> bool:
        """Check if this was a fast-path authentication."""
        return (
            self.total_duration_ms < 500 and
            not self.reasoning_used and
            self.decision == "authenticate"
        )

    @computed_field
    @property
    def was_challenging(self) -> bool:
        """Check if this was a challenging authentication."""
        return (
            self.reasoning_used or
            self.hypotheses_generated > 0 or
            self.attempt_number > 1 or
            self.final_confidence < 0.85
        )

    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB metadata."""
        base = super().to_chromadb_metadata()
        base.update({
            "session_id": self.session_id,
            "attempt_number": self.attempt_number,
            "decision": self.decision,
            "final_confidence": self.final_confidence,
            "ml_confidence": self.ml_confidence,
            "physics_confidence": self.physics_confidence,
            "behavioral_confidence": self.behavioral_confidence,
            "context_confidence": self.context_confidence,
            "total_duration_ms": self.total_duration_ms,
            "reasoning_used": self.reasoning_used,
            "hypotheses_generated": self.hypotheses_generated,
            "environment_type": self.environment_type.value,
            "is_fast_path": self.is_fast_path,
            "was_challenging": self.was_challenging,
            "spoofing_suspected": self.spoofing_suspected,
        })
        return base


# =============================================================================
# QUERY RESULTS
# =============================================================================

@dataclass
class MemoryQueryResult:
    """Result from a memory query."""

    records: List[VoiceMemoryRecord] = field(default_factory=list)
    total_count: int = 0
    query_time_ms: float = 0.0
    collection_name: str = ""

    # Aggregations
    avg_confidence: Optional[float] = None
    success_rate: Optional[float] = None
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None

    def is_empty(self) -> bool:
        """Check if result is empty."""
        return len(self.records) == 0

    def get_latest(self) -> Optional[VoiceMemoryRecord]:
        """Get the most recent record."""
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.created_at)

    def get_by_user(self, user_id: str) -> List[VoiceMemoryRecord]:
        """Filter records by user ID."""
        return [r for r in self.records if r.user_id == user_id]


__all__ = [
    # Config
    "VoiceMemoryConfig",
    # Enums
    "DriftType",
    "AttackType",
    "EnvironmentType",
    # Base
    "VoiceMemoryRecord",
    # Records
    "VoiceEvolutionRecord",
    "BehavioralPatternRecord",
    "AttackPatternRecord",
    "EnvironmentalProfileRecord",
    "SpeechBiometricsRecord",
    "AuthenticationEventRecord",
    # Results
    "MemoryQueryResult",
]
