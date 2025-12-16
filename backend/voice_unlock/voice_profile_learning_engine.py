#!/usr/bin/env python3
"""
JARVIS Voice Profile Learning Engine
=====================================

ACTIVE continuous learning system that ACTUALLY improves voice recognition over time.

Unlike passive logging, this engine:
1. FUSES new voice embeddings with the stored profile using ML techniques
2. LEARNS temporal patterns (morning voice vs evening voice)
3. UPDATES the reference embedding in the database
4. TRACKS and REPORTS confidence improvements

The core insight: Voice recognition improves when we:
- Collect diverse samples (different times, environments)
- Weight recent high-confidence samples more heavily
- Detect and exclude outliers (coughs, background noise)
- Build temporal models (your voice varies by time of day)

ML Techniques Used:
- **Exponential Moving Average (EMA)**: Smooth embedding updates
- **Mahalanobis Distance**: Outlier detection for bad samples
- **Temporal Clustering**: Time-of-day voice variations
- **Online Learning**: Incremental profile updates
- **Adaptive Weighting**: Confidence-based sample importance
"""

import asyncio
import logging
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import os

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - Dynamically loaded, no hardcoding
# =============================================================================
class LearningConfig:
    """Dynamic configuration for voice profile learning."""

    def __init__(self):
        # Learning rates
        self.embedding_learning_rate = float(os.getenv('JARVIS_EMBEDDING_LR', '0.1'))
        self.temporal_learning_rate = float(os.getenv('JARVIS_TEMPORAL_LR', '0.05'))

        # Sample quality thresholds
        self.min_confidence_for_learning = float(os.getenv('JARVIS_MIN_LEARN_CONF', '0.75'))
        self.high_quality_threshold = float(os.getenv('JARVIS_HIGH_QUALITY_CONF', '0.90'))

        # Update frequency
        self.min_samples_for_update = int(os.getenv('JARVIS_MIN_SAMPLES_UPDATE', '5'))
        self.max_samples_per_update = int(os.getenv('JARVIS_MAX_SAMPLES_UPDATE', '20'))

        # Temporal buckets (hours)
        self.temporal_buckets = {
            'early_morning': (5, 8),    # 5am-8am
            'morning': (8, 12),          # 8am-12pm
            'afternoon': (12, 17),       # 12pm-5pm
            'evening': (17, 21),         # 5pm-9pm
            'night': (21, 24),           # 9pm-12am
            'late_night': (0, 5),        # 12am-5am
        }

        # Outlier detection
        self.outlier_std_threshold = float(os.getenv('JARVIS_OUTLIER_STD', '2.5'))

        # Database paths
        self.metrics_db_path = Path(os.getenv(
            'JARVIS_METRICS_DB',
            os.path.expanduser('~/.jarvis/logs/unlock_metrics/unlock_metrics.db')
        ))
        self.learning_db_path = Path(os.getenv(
            'JARVIS_LEARNING_DB',
            os.path.expanduser('~/.jarvis/learning/jarvis_learning.db')
        ))


_config = LearningConfig()


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class VoiceSample:
    """A voice sample for learning."""
    embedding: np.ndarray
    confidence: float
    timestamp: datetime
    hour_of_day: int
    day_of_week: str
    audio_quality: str
    sample_id: str
    learning_weight: float = 1.0


@dataclass
class TemporalProfile:
    """Time-of-day specific voice profile."""
    time_bucket: str
    embedding: np.ndarray
    sample_count: int
    avg_confidence: float
    last_updated: datetime


@dataclass
class LearningState:
    """Current state of the learning engine."""
    speaker_name: str
    total_samples_learned: int
    confidence_history: List[float]
    current_avg_confidence: float
    confidence_trend: str  # 'improving', 'stable', 'declining'
    temporal_profiles: Dict[str, TemporalProfile]
    last_profile_update: Optional[datetime]
    improvement_since_start: float


# =============================================================================
# VOICE PROFILE LEARNING ENGINE
# =============================================================================
class VoiceProfileLearningEngine:
    """
    Active learning engine that improves voice recognition over time.

    This engine:
    1. Collects high-quality voice samples from successful authentications
    2. Fuses new embeddings into the stored profile using weighted averaging
    3. Builds temporal profiles (morning voice, evening voice, etc.)
    4. Detects and excludes outlier samples
    5. Actually UPDATES the database with improved embeddings
    """

    def __init__(self, speaker_name: str = "Derek J. Russell"):
        self.speaker_name = speaker_name
        self.config = _config
        self.state = None
        self._initialized = False
        self._lock = asyncio.Lock()

        # In-memory sample buffer
        self._sample_buffer: List[VoiceSample] = []
        self._max_buffer_size = 100

        # Current reference embedding
        self._reference_embedding: Optional[np.ndarray] = None
        self._temporal_embeddings: Dict[str, np.ndarray] = {}

    async def initialize(self):
        """Initialize the learning engine with existing data."""
        if self._initialized:
            return

        async with self._lock:
            try:
                # Load existing profile from database
                await self._load_profile()

                # Load historical samples
                await self._load_recent_samples()

                # Initialize state
                await self._initialize_state()

                self._initialized = True
                logger.info(
                    f"🧠 [LEARNING-ENGINE] Initialized for {self.speaker_name} | "
                    f"Samples: {self.state.total_samples_learned} | "
                    f"Avg Confidence: {self.state.current_avg_confidence:.1%}"
                )

            except Exception as e:
                logger.error(f"🧠 [LEARNING-ENGINE] Initialization failed: {e}")
                # Create default state
                self.state = LearningState(
                    speaker_name=self.speaker_name,
                    total_samples_learned=0,
                    confidence_history=[],
                    current_avg_confidence=0.0,
                    confidence_trend='stable',
                    temporal_profiles={},
                    last_profile_update=None,
                    improvement_since_start=0.0
                )
                self._initialized = True

    async def _load_profile(self):
        """Load existing speaker profile from database."""
        try:
            from intelligence.learning_database import get_learning_database
            db = await get_learning_database()

            profiles = await db.get_all_speaker_profiles()
            for profile in profiles:
                if self.speaker_name.lower() in profile.get('speaker_name', '').lower():
                    embedding_data = profile.get('embedding')
                    if embedding_data:
                        if isinstance(embedding_data, (bytes, bytearray)):
                            self._reference_embedding = np.frombuffer(embedding_data, dtype=np.float32)
                        elif isinstance(embedding_data, str):
                            self._reference_embedding = np.array(json.loads(embedding_data), dtype=np.float32)
                        logger.info(f"🧠 [LEARNING-ENGINE] Loaded reference embedding: {self._reference_embedding.shape}")
                    break

        except Exception as e:
            logger.warning(f"🧠 [LEARNING-ENGINE] Could not load profile: {e}")

    async def _load_recent_samples(self):
        """Load recent high-quality samples for learning."""
        try:
            if not self.config.metrics_db_path.exists():
                return

            conn = sqlite3.connect(self.config.metrics_db_path)
            cursor = conn.cursor()

            # Get recent successful VBI attempts with embeddings
            cursor.execute("""
                SELECT
                    embedding_json, confidence, timestamp, hour_of_day,
                    day_of_week, audio_quality, attempt_id
                FROM vbi_unlock_attempts
                WHERE speaker_name LIKE ?
                    AND success = 1
                    AND confidence >= ?
                    AND embedding_json IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
            """, (
                f"%{self.speaker_name.split()[0]}%",
                self.config.min_confidence_for_learning,
                self.config.max_samples_per_update * 2
            ))

            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                try:
                    embedding_json, conf, ts, hour, dow, quality, sample_id = row
                    if embedding_json:
                        embedding = np.array(json.loads(embedding_json), dtype=np.float32)

                        # Calculate learning weight based on confidence
                        weight = self._calculate_sample_weight(conf, quality)

                        sample = VoiceSample(
                            embedding=embedding,
                            confidence=conf,
                            timestamp=datetime.fromisoformat(ts) if isinstance(ts, str) else datetime.now(),
                            hour_of_day=hour or 12,
                            day_of_week=dow or 'Unknown',
                            audio_quality=quality or 'unknown',
                            sample_id=str(sample_id),
                            learning_weight=weight
                        )
                        self._sample_buffer.append(sample)
                except Exception as e:
                    logger.debug(f"Skipping sample: {e}")

            logger.info(f"🧠 [LEARNING-ENGINE] Loaded {len(self._sample_buffer)} recent samples")

        except Exception as e:
            logger.warning(f"🧠 [LEARNING-ENGINE] Could not load samples: {e}")

    async def _initialize_state(self):
        """Initialize learning state from loaded data."""
        confidences = [s.confidence for s in self._sample_buffer]

        # Calculate confidence trend
        if len(confidences) >= 10:
            recent = np.mean(confidences[:5])
            older = np.mean(confidences[5:10])
            if recent > older + 0.02:
                trend = 'improving'
            elif recent < older - 0.02:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'

        # Build temporal profiles from samples
        temporal_profiles = await self._build_temporal_profiles()

        self.state = LearningState(
            speaker_name=self.speaker_name,
            total_samples_learned=len(self._sample_buffer),
            confidence_history=confidences[:50],
            current_avg_confidence=np.mean(confidences) if confidences else 0.0,
            confidence_trend=trend,
            temporal_profiles=temporal_profiles,
            last_profile_update=datetime.now() if self._sample_buffer else None,
            improvement_since_start=0.0
        )

    def _calculate_sample_weight(self, confidence: float, quality: str) -> float:
        """Calculate learning weight for a sample."""
        # Base weight from confidence
        weight = confidence

        # Quality bonus
        quality_bonus = {
            'excellent': 0.2,
            'good': 0.1,
            'fair': 0.0,
            'poor': -0.2,
        }.get(quality, 0.0)

        weight += quality_bonus

        # Clamp to valid range
        return max(0.1, min(1.5, weight))

    def _get_time_bucket(self, hour: int) -> str:
        """Get temporal bucket for hour of day."""
        for bucket, (start, end) in self.config.temporal_buckets.items():
            if start <= hour < end:
                return bucket
        return 'night'

    async def _build_temporal_profiles(self) -> Dict[str, TemporalProfile]:
        """Build time-of-day specific profiles from samples."""
        profiles = {}

        # Group samples by time bucket
        bucket_samples: Dict[str, List[VoiceSample]] = defaultdict(list)
        for sample in self._sample_buffer:
            bucket = self._get_time_bucket(sample.hour_of_day)
            bucket_samples[bucket].append(sample)

        # Build profile for each bucket with enough samples
        for bucket, samples in bucket_samples.items():
            if len(samples) >= 3:
                # Weighted average of embeddings
                weights = np.array([s.learning_weight for s in samples])
                weights = weights / weights.sum()

                embeddings = np.array([s.embedding for s in samples])
                avg_embedding = np.average(embeddings, axis=0, weights=weights)

                profiles[bucket] = TemporalProfile(
                    time_bucket=bucket,
                    embedding=avg_embedding,
                    sample_count=len(samples),
                    avg_confidence=np.mean([s.confidence for s in samples]),
                    last_updated=max(s.timestamp for s in samples)
                )

                self._temporal_embeddings[bucket] = avg_embedding

        return profiles

    # =========================================================================
    # CORE LEARNING METHODS
    # =========================================================================

    async def learn_from_sample(
        self,
        embedding: np.ndarray,
        confidence: float,
        success: bool,
        audio_quality: str = None,
        timestamp: datetime = None
    ) -> Dict[str, Any]:
        """
        Learn from a new voice sample.

        This is the MAIN entry point for continuous learning.
        Call this after every successful VBI authentication.

        Returns:
            Dict with learning results and updated profile info
        """
        if not self._initialized:
            await self.initialize()

        timestamp = timestamp or datetime.now()

        # Only learn from high-quality successful samples
        if not success or confidence < self.config.min_confidence_for_learning:
            return {
                'learned': False,
                'reason': f"Sample not suitable (success={success}, conf={confidence:.1%})",
                'current_avg_confidence': self.state.current_avg_confidence
            }

        # Check for outliers using existing profile
        if self._reference_embedding is not None:
            is_outlier = self._is_outlier(embedding)
            if is_outlier:
                logger.warning(f"🧠 [LEARNING-ENGINE] Outlier sample detected, skipping")
                return {
                    'learned': False,
                    'reason': 'Outlier sample detected',
                    'current_avg_confidence': self.state.current_avg_confidence
                }

        # Create sample
        if audio_quality is None:
            audio_quality = 'excellent' if confidence >= 0.95 else 'good' if confidence >= 0.85 else 'fair'

        sample = VoiceSample(
            embedding=embedding,
            confidence=confidence,
            timestamp=timestamp,
            hour_of_day=timestamp.hour,
            day_of_week=timestamp.strftime('%A'),
            audio_quality=audio_quality,
            sample_id=f"learn_{int(timestamp.timestamp() * 1000)}",
            learning_weight=self._calculate_sample_weight(confidence, audio_quality)
        )

        # Add to buffer
        self._sample_buffer.insert(0, sample)
        if len(self._sample_buffer) > self._max_buffer_size:
            self._sample_buffer = self._sample_buffer[:self._max_buffer_size]

        # Update confidence history
        self.state.confidence_history.insert(0, confidence)
        self.state.confidence_history = self.state.confidence_history[:50]
        self.state.total_samples_learned += 1

        # Update reference embedding using EMA
        old_avg = self.state.current_avg_confidence
        await self._update_reference_embedding(sample)

        # Update temporal profile
        await self._update_temporal_profile(sample)

        # Recalculate statistics
        self.state.current_avg_confidence = np.mean(self.state.confidence_history)
        self._update_confidence_trend()

        # Check if we should write to database
        should_persist = (
            self.state.total_samples_learned % self.config.min_samples_for_update == 0 or
            confidence >= self.config.high_quality_threshold
        )

        if should_persist:
            await self._persist_updated_profile()

        improvement = self.state.current_avg_confidence - old_avg

        logger.info(
            f"🧠 [LEARNING-ENGINE] Learned from sample | "
            f"Confidence: {confidence:.1%} | "
            f"Avg: {self.state.current_avg_confidence:.1%} | "
            f"Trend: {self.state.confidence_trend} | "
            f"Improvement: {improvement:+.2%}"
        )

        return {
            'learned': True,
            'sample_id': sample.sample_id,
            'current_avg_confidence': self.state.current_avg_confidence,
            'confidence_trend': self.state.confidence_trend,
            'total_samples': self.state.total_samples_learned,
            'improvement': improvement,
            'persisted': should_persist
        }

    def _is_outlier(self, embedding: np.ndarray) -> bool:
        """Detect if embedding is an outlier using Mahalanobis-like distance."""
        if self._reference_embedding is None:
            return False

        # Calculate cosine distance
        ref_norm = self._reference_embedding / np.linalg.norm(self._reference_embedding)
        emb_norm = embedding / np.linalg.norm(embedding)
        similarity = np.dot(ref_norm, emb_norm)

        # If similarity is very low, it's likely an outlier
        # For ECAPA embeddings, typical same-speaker similarity is > 0.5
        return similarity < 0.3

    async def _update_reference_embedding(self, sample: VoiceSample):
        """Update reference embedding using Exponential Moving Average."""
        if self._reference_embedding is None:
            self._reference_embedding = sample.embedding.copy()
            return

        # EMA update: new_ref = (1-α)*old_ref + α*new_sample
        # α is adaptive based on sample quality
        alpha = self.config.embedding_learning_rate * sample.learning_weight

        # Weighted update
        self._reference_embedding = (
            (1 - alpha) * self._reference_embedding +
            alpha * sample.embedding
        )

        # Normalize to unit sphere (important for cosine similarity)
        self._reference_embedding = (
            self._reference_embedding / np.linalg.norm(self._reference_embedding)
        )

    async def _update_temporal_profile(self, sample: VoiceSample):
        """Update time-of-day specific profile."""
        bucket = self._get_time_bucket(sample.hour_of_day)

        if bucket in self._temporal_embeddings:
            # EMA update for temporal profile
            alpha = self.config.temporal_learning_rate * sample.learning_weight
            self._temporal_embeddings[bucket] = (
                (1 - alpha) * self._temporal_embeddings[bucket] +
                alpha * sample.embedding
            )
            # Normalize
            self._temporal_embeddings[bucket] = (
                self._temporal_embeddings[bucket] /
                np.linalg.norm(self._temporal_embeddings[bucket])
            )
        else:
            self._temporal_embeddings[bucket] = sample.embedding.copy()

        # Update state
        if bucket not in self.state.temporal_profiles:
            self.state.temporal_profiles[bucket] = TemporalProfile(
                time_bucket=bucket,
                embedding=self._temporal_embeddings[bucket],
                sample_count=1,
                avg_confidence=sample.confidence,
                last_updated=sample.timestamp
            )
        else:
            prof = self.state.temporal_profiles[bucket]
            prof.sample_count += 1
            prof.avg_confidence = (
                (prof.avg_confidence * (prof.sample_count - 1) + sample.confidence) /
                prof.sample_count
            )
            prof.last_updated = sample.timestamp
            prof.embedding = self._temporal_embeddings[bucket]

    def _update_confidence_trend(self):
        """Update confidence trend based on recent history."""
        if len(self.state.confidence_history) >= 10:
            recent = np.mean(self.state.confidence_history[:5])
            older = np.mean(self.state.confidence_history[5:10])
            if recent > older + 0.02:
                self.state.confidence_trend = 'improving'
            elif recent < older - 0.02:
                self.state.confidence_trend = 'declining'
            else:
                self.state.confidence_trend = 'stable'

    async def _persist_updated_profile(self):
        """Write updated embedding to database."""
        if self._reference_embedding is None:
            return

        try:
            from intelligence.learning_database import get_learning_database
            db = await get_learning_database()

            # Update the speaker profile with new embedding
            embedding_bytes = self._reference_embedding.tobytes()

            # Also store temporal profiles
            temporal_json = json.dumps({
                bucket: {
                    'embedding': emb.tolist(),
                    'sample_count': self.state.temporal_profiles.get(bucket, TemporalProfile(
                        time_bucket=bucket,
                        embedding=emb,
                        sample_count=1,
                        avg_confidence=0.0,
                        last_updated=datetime.now()
                    )).sample_count
                }
                for bucket, emb in self._temporal_embeddings.items()
            })

            await db.update_speaker_profile(
                speaker_name=self.speaker_name,
                embedding=embedding_bytes,
                metadata={
                    'last_learning_update': datetime.now().isoformat(),
                    'total_samples_learned': self.state.total_samples_learned,
                    'current_avg_confidence': self.state.current_avg_confidence,
                    'confidence_trend': self.state.confidence_trend,
                    'temporal_profiles': temporal_json
                }
            )

            self.state.last_profile_update = datetime.now()
            logger.info(f"🧠 [LEARNING-ENGINE] Persisted updated profile to database")

        except Exception as e:
            logger.error(f"🧠 [LEARNING-ENGINE] Failed to persist profile: {e}")

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    async def get_best_reference_for_time(self, hour: int = None) -> Optional[np.ndarray]:
        """
        Get the best reference embedding for a specific time of day.

        If we have a temporal profile for this time, blend it with
        the main profile for better accuracy.
        """
        if not self._initialized:
            await self.initialize()

        if self._reference_embedding is None:
            return None

        hour = hour if hour is not None else datetime.now().hour
        bucket = self._get_time_bucket(hour)

        if bucket in self._temporal_embeddings:
            # Blend main profile with temporal profile
            # Weight temporal profile more heavily (60/40)
            blended = (
                0.4 * self._reference_embedding +
                0.6 * self._temporal_embeddings[bucket]
            )
            return blended / np.linalg.norm(blended)

        return self._reference_embedding

    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        if not self._initialized:
            await self.initialize()

        return {
            'speaker_name': self.speaker_name,
            'total_samples_learned': self.state.total_samples_learned,
            'current_avg_confidence': self.state.current_avg_confidence,
            'confidence_trend': self.state.confidence_trend,
            'improvement_since_start': self.state.improvement_since_start,
            'temporal_profiles': {
                bucket: {
                    'sample_count': prof.sample_count,
                    'avg_confidence': prof.avg_confidence,
                    'last_updated': prof.last_updated.isoformat() if prof.last_updated else None
                }
                for bucket, prof in self.state.temporal_profiles.items()
            },
            'last_profile_update': (
                self.state.last_profile_update.isoformat()
                if self.state.last_profile_update else None
            ),
            'confidence_history_length': len(self.state.confidence_history),
            'has_reference_embedding': self._reference_embedding is not None,
            'temporal_buckets_available': list(self._temporal_embeddings.keys())
        }


# =============================================================================
# GLOBAL INSTANCE & INTEGRATION
# =============================================================================
_learning_engine: Optional[VoiceProfileLearningEngine] = None
_learning_lock = asyncio.Lock()


async def get_learning_engine(speaker_name: str = "Derek J. Russell") -> VoiceProfileLearningEngine:
    """Get or create the global learning engine instance."""
    global _learning_engine

    async with _learning_lock:
        if _learning_engine is None:
            _learning_engine = VoiceProfileLearningEngine(speaker_name)
            await _learning_engine.initialize()

        return _learning_engine


async def learn_from_vbi_unlock(
    embedding: np.ndarray,
    confidence: float,
    success: bool,
    speaker_name: str = "Derek J. Russell",
    audio_quality: str = None
) -> Dict[str, Any]:
    """
    Convenience function to learn from a VBI unlock attempt.

    Call this after every VBI authentication to continuously improve recognition.

    Args:
        embedding: 192-dim ECAPA-TDNN embedding from the attempt
        confidence: Verification confidence score (0.0-1.0)
        success: Whether authentication succeeded
        speaker_name: Name of the speaker
        audio_quality: Optional quality assessment

    Returns:
        Dict with learning results
    """
    engine = await get_learning_engine(speaker_name)
    return await engine.learn_from_sample(
        embedding=embedding,
        confidence=confidence,
        success=success,
        audio_quality=audio_quality
    )


async def get_optimized_reference(hour: int = None) -> Optional[np.ndarray]:
    """
    Get time-optimized reference embedding for verification.

    This returns a blended embedding that's optimized for the current
    time of day based on learned temporal patterns.

    Args:
        hour: Hour of day (0-23), defaults to current hour

    Returns:
        Optimized reference embedding or None if not available
    """
    engine = await get_learning_engine()
    return await engine.get_best_reference_for_time(hour)
