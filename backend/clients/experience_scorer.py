"""
Quality-weighted experience scoring for intelligent training triggers.

Instead of triggering training after a flat count of experiences,
this module assigns weights based on experience quality and type.
Corrections and negative feedback are worth more than normal interactions.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │            Experience Flow                           │
    │                                                      │
    │  experience ─► ExperienceScorer.score()              │
    │                  │ CORRECTION ──► 10.0               │
    │                  │ NEG FEEDBACK ─► 5.0               │
    │                  │ ERROR ────────► 3.0               │
    │                  │ NOVEL TASK ───► 3.0               │
    │                  │ LOW CONF ─────► 2.0               │
    │                  │ NORMAL ───────► 1.0               │
    │                  ▼                                   │
    │  WeightedExperienceTracker.add()                     │
    │                  │ dedup check (bloom/hash)          │
    │                  │ DUPLICATE ────► 0.1               │
    │                  ▼                                   │
    │  cumulative_score += score                           │
    │  should_trigger() ─► cumulative >= threshold?        │
    └──────────────────────────────────────────────────────┘

Author: JARVIS AI System
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Configurable weights for different experience types.

    All weights can be overridden via environment variables for
    runtime tuning without code changes.
    """

    correction: float = field(
        default_factory=lambda: float(os.getenv("REACTOR_WEIGHT_CORRECTION", "10.0"))
    )
    feedback_negative: float = field(
        default_factory=lambda: float(os.getenv("REACTOR_WEIGHT_FEEDBACK_NEG", "5.0"))
    )
    error: float = field(
        default_factory=lambda: float(os.getenv("REACTOR_WEIGHT_ERROR", "3.0"))
    )
    novel_task: float = field(
        default_factory=lambda: float(os.getenv("REACTOR_WEIGHT_NOVEL_TASK", "3.0"))
    )
    low_confidence: float = field(
        default_factory=lambda: float(os.getenv("REACTOR_WEIGHT_LOW_CONF", "2.0"))
    )
    normal: float = field(
        default_factory=lambda: float(os.getenv("REACTOR_WEIGHT_NORMAL", "1.0"))
    )
    near_duplicate: float = field(
        default_factory=lambda: float(os.getenv("REACTOR_WEIGHT_DUPLICATE", "0.1"))
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("REACTOR_CONFIDENCE_THRESHOLD", "0.7"))
    )


class ExperienceScorer:
    """Score individual experiences by their training value.

    Uses a priority-ordered classification:
    1. CORRECTION events (highest value -- the model got something wrong)
    2. Negative feedback (user explicitly said output was bad)
    3. ERROR events (system failures are learning opportunities)
    4. Novel task types (first time seeing a task category)
    5. Low-confidence interactions (borderline = informative)
    6. Normal interactions (baseline)

    The scorer maintains a set of known task types so it can detect
    novelty. This state persists across the scorer's lifetime.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()
        self._known_task_types: Set[str] = set()

    def score(self, experience: Dict) -> float:
        """
        Score an experience by its training value.

        Args:
            experience: Dict with keys like event_type, confidence,
                        task_type, feedback_sentiment, user_input.

        Returns:
            Weighted score (higher = more valuable for training).
        """
        event_type = experience.get("event_type", "INTERACTION")

        # Highest priority: corrections
        if event_type == "CORRECTION":
            return self.weights.correction

        # Negative feedback
        if event_type == "FEEDBACK" and experience.get("feedback_sentiment") == "negative":
            return self.weights.feedback_negative

        # Errors
        if event_type == "ERROR":
            return self.weights.error

        # Novel task type (first time seeing this task_type)
        task_type = experience.get("task_type", "")
        if task_type and task_type not in self._known_task_types:
            self._known_task_types.add(task_type)
            return self.weights.novel_task

        # Low confidence (borderline = informative)
        confidence = experience.get("confidence", 1.0)
        if confidence < self.weights.confidence_threshold:
            return self.weights.low_confidence

        # Normal interaction
        return self.weights.normal


class WeightedExperienceTracker:
    """Track cumulative weighted score and check training threshold.

    Instead of a flat experience count, this tracker accumulates weighted
    scores so that high-value experiences (corrections, errors, negative
    feedback) push toward the training threshold faster than routine
    interactions.

    Near-duplicates (same user_input + task_type) are detected via a
    hash-based dedup set and assigned a reduced weight (0.1x default).
    The dedup set persists across reset() calls so that duplicates remain
    detected even after a training cycle completes.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        scorer: Optional[ExperienceScorer] = None,
        max_bloom_size: int = 10000,
    ):
        self.threshold = threshold if threshold is not None else float(
            os.getenv("REACTOR_CORE_WEIGHTED_THRESHOLD", "100.0")
        )
        self.scorer = scorer or ExperienceScorer()
        self._cumulative_score: float = 0.0
        self._experience_count: int = 0
        self._bloom: Set[str] = set()  # Simple set-based dedup (bloom filter approximation)
        self._max_bloom_size = max_bloom_size

    def add(self, experience: Dict) -> float:
        """
        Add an experience and return its weighted score.

        Near-duplicates (same user_input + task_type) get reduced weight.

        Args:
            experience: Experience dictionary.

        Returns:
            The weighted score assigned to this experience.
        """
        # Dedup check
        dedup_key = self._make_dedup_key(experience)
        is_duplicate = dedup_key in self._bloom

        if not is_duplicate:
            self._bloom.add(dedup_key)
            if len(self._bloom) > self._max_bloom_size:
                # Evict oldest ~10% (approximation -- set doesn't preserve order)
                to_remove = list(self._bloom)[: self._max_bloom_size // 10]
                for key in to_remove:
                    self._bloom.discard(key)

        # Score
        if is_duplicate:
            score = self.scorer.weights.near_duplicate
        else:
            score = self.scorer.score(experience)

        self._cumulative_score += score
        self._experience_count += 1
        return score

    @property
    def cumulative_score(self) -> float:
        """Current accumulated weighted score."""
        return self._cumulative_score

    @property
    def experience_count(self) -> int:
        """Total number of experiences added (including duplicates)."""
        return self._experience_count

    def should_trigger(self) -> bool:
        """Check if cumulative score has reached the training threshold."""
        return self._cumulative_score >= self.threshold

    def reset(self) -> None:
        """Reset score and count after training trigger.

        The bloom filter is intentionally preserved so that duplicates
        remain detected across training cycles.
        """
        self._cumulative_score = 0.0
        self._experience_count = 0
        # Keep bloom filter -- dedup should persist across training cycles

    def _make_dedup_key(self, experience: Dict) -> str:
        """Create a hash key for deduplication.

        Uses user_input + task_type as the identity signal. Two experiences
        with the same user input and task type are considered near-duplicates
        regardless of other fields.
        """
        user_input = experience.get("user_input", "")
        task_type = experience.get("task_type", "")
        raw = f"{user_input}:{task_type}"
        return hashlib.md5(raw.encode()).hexdigest()
