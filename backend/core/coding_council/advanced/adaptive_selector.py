"""
v77.1: Adaptive Framework Selector - ML-Based
==============================================

Selects the best framework for each task using machine learning.

Problem:
    Different frameworks excel at different tasks:
    - Aider: Great for targeted edits
    - MetaGPT: Great for complex planning
    - RepoMaster: Great for refactoring

    Static selection wastes potential.

Solution:
    - Learn from historical success rates
    - Extract features from tasks
    - Predict best framework per task
    - Continuously improve with feedback

Features:
    - Feature extraction from task descriptions
    - Historical success rate tracking
    - Multi-armed bandit exploration
    - Thompson sampling for selection
    - Online learning with feedback

Author: Ironcliw v77.1
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# ARM64 SIMD Acceleration (40-50x faster similarity calculations)
# =============================================================================

try:
    from ..acceleration import (
        UnifiedAccelerator,
        get_accelerator,
        get_acceleration_registry,
    )
    _ACCELERATOR: Optional[UnifiedAccelerator] = None

    def _get_accelerator() -> Optional[UnifiedAccelerator]:
        """Get or create accelerator instance (lazy initialization)."""
        global _ACCELERATOR
        if _ACCELERATOR is None:
            try:
                _ACCELERATOR = get_accelerator()
                # Register this component
                registry = get_acceleration_registry()
                registry.register(
                    component_name="adaptive_selector",
                    repo="jarvis",
                    operations={"dot_product", "weighted_combination", "batch_similarity"}
                )
                logger.info("[AdaptiveSelector] ARM64 acceleration enabled (40-50x speedup)")
            except Exception as e:
                logger.debug(f"[AdaptiveSelector] Acceleration init failed: {e}")
        return _ACCELERATOR

    ACCELERATION_AVAILABLE = True
except ImportError:
    ACCELERATION_AVAILABLE = False
    _ACCELERATOR = None

    def _get_accelerator():
        return None

    logger.debug("[AdaptiveSelector] Acceleration module not available")


@dataclass
class FrameworkScore:
    """Score for a framework on a specific task."""
    framework: str
    score: float  # 0-1, probability of success
    confidence: float  # 0-1, confidence in the score
    reasons: List[str] = field(default_factory=list)
    features_matched: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "features_matched": list(self.features_matched),
        }


@dataclass
class SelectionContext:
    """Context for framework selection."""
    description: str
    target_files: List[str] = field(default_factory=list)
    file_types: Set[str] = field(default_factory=set)
    estimated_complexity: float = 0.5  # 0-1
    requires_planning: bool = False
    requires_tests: bool = False
    is_refactoring: bool = False
    is_new_feature: bool = False
    is_bug_fix: bool = False
    repo_size: int = 0  # Lines of code
    previous_frameworks: List[str] = field(default_factory=list)

    def to_features(self) -> Dict[str, float]:
        """Convert context to feature vector."""
        features = {
            "complexity": self.estimated_complexity,
            "requires_planning": 1.0 if self.requires_planning else 0.0,
            "requires_tests": 1.0 if self.requires_tests else 0.0,
            "is_refactoring": 1.0 if self.is_refactoring else 0.0,
            "is_new_feature": 1.0 if self.is_new_feature else 0.0,
            "is_bug_fix": 1.0 if self.is_bug_fix else 0.0,
            "file_count": min(len(self.target_files) / 10.0, 1.0),
            "repo_size_normalized": min(self.repo_size / 100000.0, 1.0),
        }

        # File type features
        for ft in ["py", "js", "ts", "go", "rust", "java"]:
            features[f"has_{ft}"] = 1.0 if ft in self.file_types else 0.0

        return features


@dataclass
class FrameworkStats:
    """Historical statistics for a framework."""
    framework: str
    total_attempts: int = 0
    successes: int = 0
    partial_successes: int = 0
    failures: int = 0
    total_duration_ms: float = 0.0
    # Bayesian prior (Beta distribution parameters)
    alpha: float = 1.0  # Prior successes
    beta: float = 1.0  # Prior failures

    @property
    def success_rate(self) -> float:
        """Empirical success rate."""
        if self.total_attempts == 0:
            return 0.5  # Prior
        return self.successes / self.total_attempts

    @property
    def bayesian_success_rate(self) -> float:
        """Bayesian estimated success rate (Beta posterior mean)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def average_duration_ms(self) -> float:
        """Average execution duration."""
        if self.total_attempts == 0:
            return 0.0
        return self.total_duration_ms / self.total_attempts

    def sample_thompson(self) -> float:
        """Sample from Thompson sampling (Beta posterior)."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool, duration_ms: float = 0.0) -> None:
        """Update statistics with new outcome."""
        self.total_attempts += 1
        self.total_duration_ms += duration_ms

        if success:
            self.successes += 1
            self.alpha += 1
        else:
            self.failures += 1
            self.beta += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "total_attempts": self.total_attempts,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 4),
            "bayesian_rate": round(self.bayesian_success_rate, 4),
            "avg_duration_ms": round(self.average_duration_ms, 2),
        }


class FeatureExtractor:
    """
    Extracts features from task descriptions.

    Uses keyword matching and heuristics to identify task characteristics.
    """

    # Keywords for task classification
    PLANNING_KEYWORDS = {
        "design", "architect", "plan", "structure", "organize",
        "create new", "implement new", "add new", "build"
    }

    REFACTORING_KEYWORDS = {
        "refactor", "reorganize", "restructure", "cleanup", "clean up",
        "improve", "optimize", "simplify", "extract", "rename"
    }

    BUG_FIX_KEYWORDS = {
        "fix", "bug", "error", "issue", "problem", "broken",
        "crash", "fail", "wrong", "incorrect", "doesn't work"
    }

    TEST_KEYWORDS = {
        "test", "testing", "unittest", "pytest", "spec",
        "coverage", "assertion", "mock", "verify"
    }

    NEW_FEATURE_KEYWORDS = {
        "add", "create", "implement", "new feature", "introduce",
        "support for", "enable", "allow"
    }

    COMPLEXITY_INDICATORS = {
        "simple": 0.2,
        "basic": 0.2,
        "quick": 0.2,
        "small": 0.2,
        "minor": 0.3,
        "complex": 0.8,
        "complicated": 0.8,
        "large": 0.7,
        "major": 0.7,
        "extensive": 0.8,
        "comprehensive": 0.8,
        "entire": 0.9,
        "all": 0.7,
    }

    @classmethod
    def extract(cls, description: str, target_files: List[str] = None) -> SelectionContext:
        """
        Extract features from task description.

        Args:
            description: Task description
            target_files: Optional list of target files

        Returns:
            SelectionContext with extracted features
        """
        desc_lower = description.lower()
        target_files = target_files or []

        context = SelectionContext(
            description=description,
            target_files=target_files,
        )

        # Extract file types
        for file in target_files:
            if "." in file:
                ext = file.rsplit(".", 1)[-1]
                context.file_types.add(ext)

        # Classify task type
        context.requires_planning = any(kw in desc_lower for kw in cls.PLANNING_KEYWORDS)
        context.is_refactoring = any(kw in desc_lower for kw in cls.REFACTORING_KEYWORDS)
        context.is_bug_fix = any(kw in desc_lower for kw in cls.BUG_FIX_KEYWORDS)
        context.requires_tests = any(kw in desc_lower for kw in cls.TEST_KEYWORDS)
        context.is_new_feature = any(kw in desc_lower for kw in cls.NEW_FEATURE_KEYWORDS)

        # Estimate complexity
        complexity_scores = []
        for indicator, score in cls.COMPLEXITY_INDICATORS.items():
            if indicator in desc_lower:
                complexity_scores.append(score)

        if complexity_scores:
            context.estimated_complexity = sum(complexity_scores) / len(complexity_scores)
        else:
            # Default based on description length and file count
            context.estimated_complexity = min(
                0.3 + (len(description) / 500) * 0.3 + (len(target_files) / 10) * 0.2,
                1.0
            )

        return context


class LearningModel:
    """
    Machine learning model for framework selection.

    Uses a combination of:
    - Historical success rates
    - Feature-based scoring
    - Thompson sampling for exploration
    - ARM64 SIMD acceleration for vector operations (40-50x faster)

    Learns online from feedback.
    """

    # Feature weights for each framework (learned over time)
    INITIAL_WEIGHTS: Dict[str, Dict[str, float]] = {
        "aider": {
            "is_bug_fix": 0.3,
            "file_count": -0.2,  # Prefers fewer files
            "has_py": 0.2,
            "complexity": -0.1,
        },
        "metagpt": {
            "requires_planning": 0.4,
            "is_new_feature": 0.3,
            "complexity": 0.2,
        },
        "repomaster": {
            "is_refactoring": 0.4,
            "file_count": 0.2,  # Good with multiple files
            "complexity": 0.1,
        },
        "openhands": {
            "is_new_feature": 0.2,
            "complexity": 0.2,
            "requires_tests": 0.2,
        },
        "continue": {
            "is_bug_fix": 0.2,
            "complexity": -0.2,  # Better for simpler tasks
            "file_count": -0.1,
        },
    }

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".jarvis" / "framework_learning"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._stats: Dict[str, FrameworkStats] = {}
        self._weights: Dict[str, Dict[str, float]] = {}
        self._feature_history: List[Tuple[Dict[str, float], str, bool]] = []

        # ARM64 SIMD acceleration (40-50x faster for vector ops)
        self._accelerator = _get_accelerator()
        self._use_acceleration = self._accelerator is not None
        if self._use_acceleration:
            logger.debug("[LearningModel] Using ARM64 SIMD acceleration")

        # Initialize
        self._load_model()

    def _load_model(self) -> None:
        """Load model from disk."""
        model_file = self.data_dir / "model.json"
        if model_file.exists():
            try:
                data = json.loads(model_file.read_text())

                # Load stats
                for name, stats_data in data.get("stats", {}).items():
                    self._stats[name] = FrameworkStats(
                        framework=name,
                        total_attempts=stats_data.get("total_attempts", 0),
                        successes=stats_data.get("successes", 0),
                        partial_successes=stats_data.get("partial_successes", 0),
                        failures=stats_data.get("failures", 0),
                        total_duration_ms=stats_data.get("total_duration_ms", 0),
                        alpha=stats_data.get("alpha", 1.0),
                        beta=stats_data.get("beta", 1.0),
                    )

                # Load weights
                self._weights = data.get("weights", {})

            except Exception as e:
                logger.warning(f"[LearningModel] Failed to load model: {e}")

        # Initialize defaults
        for framework in self.INITIAL_WEIGHTS:
            if framework not in self._stats:
                self._stats[framework] = FrameworkStats(framework=framework)
            if framework not in self._weights:
                self._weights[framework] = self.INITIAL_WEIGHTS[framework].copy()

    def _save_model(self) -> None:
        """Save model to disk."""
        model_file = self.data_dir / "model.json"
        data = {
            "stats": {
                name: {
                    "total_attempts": s.total_attempts,
                    "successes": s.successes,
                    "partial_successes": s.partial_successes,
                    "failures": s.failures,
                    "total_duration_ms": s.total_duration_ms,
                    "alpha": s.alpha,
                    "beta": s.beta,
                }
                for name, s in self._stats.items()
            },
            "weights": self._weights,
            "updated_at": time.time(),
        }
        model_file.write_text(json.dumps(data, indent=2))

    def score_framework(
        self,
        framework: str,
        context: SelectionContext,
        exploration_factor: float = 0.1,
    ) -> FrameworkScore:
        """
        Score a framework for a given context.

        Combines:
        - Historical success rate (Thompson sampling)
        - Feature-based prediction (ARM64 accelerated when available)
        - Exploration bonus

        Args:
            framework: Framework to score
            context: Selection context
            exploration_factor: How much to explore (0-1)

        Returns:
            FrameworkScore with score and details
        """
        features = context.to_features()
        stats = self._stats.get(framework, FrameworkStats(framework=framework))
        weights = self._weights.get(framework, {})

        reasons = []
        matched_features = set()

        # 1. Historical success rate (Thompson sampling)
        thompson_sample = stats.sample_thompson()
        reasons.append(f"Historical success: {stats.bayesian_success_rate:.1%}")

        # 2. Feature-based score (ARM64 SIMD accelerated when available)
        feature_score = 0.5  # Base score

        if self._use_acceleration and self._accelerator and len(weights) > 3:
            # Use vectorized computation for larger feature sets (40-50x faster)
            try:
                import numpy as np
                # Convert to aligned vectors for SIMD
                feature_keys = sorted(set(features.keys()) | set(weights.keys()))
                feature_vec = np.array(
                    [features.get(k, 0.0) for k in feature_keys],
                    dtype=np.float32
                )
                weight_vec = np.array(
                    [weights.get(k, 0.0) for k in feature_keys],
                    dtype=np.float32
                )

                # ARM64 NEON dot product (40-50x faster)
                contribution = self._accelerator.dot_product(feature_vec, weight_vec)
                feature_score += float(contribution)

                # Track matched features for explainability
                for i, key in enumerate(feature_keys):
                    if abs(feature_vec[i] * weight_vec[i]) > 0.05:
                        matched_features.add(key)
                        reasons.append(
                            f"{key}: {'+' if weight_vec[i] > 0 else ''}"
                            f"{feature_vec[i] * weight_vec[i]:.2f}"
                        )
            except Exception as e:
                logger.debug(f"[LearningModel] Acceleration fallback: {e}")
                # Fall through to standard computation
                self._use_acceleration = False

        if not self._use_acceleration or len(weights) <= 3:
            # Standard computation for small feature sets or fallback
            for feature, weight in weights.items():
                if feature in features:
                    contribution = features[feature] * weight
                    feature_score += contribution
                    if abs(contribution) > 0.05:
                        matched_features.add(feature)
                        reasons.append(f"{feature}: {'+' if contribution > 0 else ''}{contribution:.2f}")

        feature_score = max(0.0, min(1.0, feature_score))

        # 3. Exploration bonus (UCB-style)
        if stats.total_attempts > 0:
            exploration_bonus = exploration_factor * math.sqrt(
                math.log(sum(s.total_attempts for s in self._stats.values()) + 1)
                / stats.total_attempts
            )
        else:
            exploration_bonus = exploration_factor  # High bonus for untried

        # 4. Combine scores
        # Weight: 40% Thompson, 40% features, 20% exploration
        combined_score = (
            0.4 * thompson_sample +
            0.4 * feature_score +
            0.2 * exploration_bonus
        )

        # Calculate confidence based on sample size
        confidence = min(stats.total_attempts / 20.0, 1.0)

        return FrameworkScore(
            framework=framework,
            score=combined_score,
            confidence=confidence,
            reasons=reasons,
            features_matched=matched_features,
        )

    def update(
        self,
        framework: str,
        context: SelectionContext,
        success: bool,
        duration_ms: float = 0.0,
    ) -> None:
        """
        Update model with feedback.

        Args:
            framework: Framework that was used
            context: Context of the task
            success: Whether it succeeded
            duration_ms: Execution duration
        """
        # Update stats
        if framework not in self._stats:
            self._stats[framework] = FrameworkStats(framework=framework)

        self._stats[framework].update(success, duration_ms)

        # Store for weight learning
        features = context.to_features()
        self._feature_history.append((features, framework, success))

        # Online weight update (simple gradient descent)
        if len(self._feature_history) >= 10:
            self._update_weights()

        # Save model
        self._save_model()

        logger.info(
            f"[LearningModel] Updated {framework}: "
            f"{'success' if success else 'failure'}, "
            f"rate now {self._stats[framework].bayesian_success_rate:.1%}"
        )

    def _update_weights(self, learning_rate: float = 0.01) -> None:
        """
        Update feature weights using recent history.

        Simple online gradient descent.
        """
        if not self._feature_history:
            return

        # Use last 100 examples
        recent = self._feature_history[-100:]

        for features, framework, success in recent:
            if framework not in self._weights:
                self._weights[framework] = {}

            target = 1.0 if success else 0.0
            weights = self._weights[framework]

            # Simple prediction
            prediction = 0.5
            for feature, value in features.items():
                prediction += weights.get(feature, 0.0) * value

            prediction = max(0.0, min(1.0, prediction))
            error = target - prediction

            # Update weights
            for feature, value in features.items():
                if feature not in weights:
                    weights[feature] = 0.0
                weights[feature] += learning_rate * error * value

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "frameworks": {
                name: stats.to_dict()
                for name, stats in self._stats.items()
            },
            "total_observations": len(self._feature_history),
            "weights": self._weights,
        }


class AdaptiveFrameworkSelector:
    """
    Selects the best framework for each task using ML.

    Features:
    - Feature extraction from task descriptions
    - Historical success rate tracking
    - Thompson sampling for exploration
    - Online learning from feedback

    Usage:
        selector = AdaptiveFrameworkSelector()

        # Select framework
        scores = await selector.select(
            description="Fix the login bug",
            target_files=["auth.py"],
        )
        best = scores[0]  # Highest scoring framework

        # Later, provide feedback
        await selector.record_outcome(
            framework="aider",
            context=context,
            success=True,
        )
    """

    def __init__(
        self,
        available_frameworks: Optional[List[str]] = None,
        exploration_factor: float = 0.1,
        data_dir: Optional[Path] = None,
    ):
        self.available_frameworks = available_frameworks or [
            "aider", "metagpt", "repomaster", "openhands", "continue"
        ]
        self.exploration_factor = exploration_factor

        self._model = LearningModel(data_dir)
        self._last_context: Optional[SelectionContext] = None

    async def select(
        self,
        description: str,
        target_files: Optional[List[str]] = None,
        exclude_frameworks: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[FrameworkScore]:
        """
        Select best frameworks for the task.

        Args:
            description: Task description
            target_files: Target files for the task
            exclude_frameworks: Frameworks to exclude
            top_k: Number of frameworks to return

        Returns:
            List of FrameworkScore sorted by score (descending)
        """
        # Extract features
        context = FeatureExtractor.extract(description, target_files or [])
        self._last_context = context

        # Score all available frameworks
        exclude = set(exclude_frameworks or [])
        scores = []

        for framework in self.available_frameworks:
            if framework in exclude:
                continue

            score = self._model.score_framework(
                framework,
                context,
                self.exploration_factor,
            )
            scores.append(score)

        # Sort by score
        scores.sort(key=lambda s: s.score, reverse=True)

        logger.info(
            f"[AdaptiveSelector] Top frameworks for '{description[:50]}...': "
            f"{', '.join(f'{s.framework}({s.score:.2f})' for s in scores[:top_k])}"
        )

        return scores[:top_k]

    async def record_outcome(
        self,
        framework: str,
        success: bool,
        duration_ms: float = 0.0,
        context: Optional[SelectionContext] = None,
    ) -> None:
        """
        Record outcome for learning.

        Args:
            framework: Framework that was used
            success: Whether it succeeded
            duration_ms: Execution duration
            context: Optional context (uses last context if not provided)
        """
        context = context or self._last_context
        if not context:
            context = SelectionContext(description="")

        self._model.update(framework, context, success, duration_ms)

    async def get_recommendation(
        self,
        description: str,
        target_files: Optional[List[str]] = None,
    ) -> Tuple[str, FrameworkScore]:
        """
        Get single best framework recommendation.

        Args:
            description: Task description
            target_files: Target files

        Returns:
            Tuple of (framework_name, score)
        """
        scores = await self.select(description, target_files, top_k=1)
        if not scores:
            # Fallback to random
            framework = random.choice(self.available_frameworks)
            return framework, FrameworkScore(
                framework=framework,
                score=0.5,
                confidence=0.0,
                reasons=["Fallback selection"],
            )

        return scores[0].framework, scores[0]

    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        return {
            "available_frameworks": self.available_frameworks,
            "exploration_factor": self.exploration_factor,
            "model_stats": self._model.get_stats(),
        }

    def set_exploration_factor(self, factor: float) -> None:
        """
        Set exploration factor.

        Higher = more exploration of less-tried frameworks.
        Lower = more exploitation of known-good frameworks.
        """
        self.exploration_factor = max(0.0, min(1.0, factor))


# Global instance
_selector: Optional[AdaptiveFrameworkSelector] = None


def get_adaptive_selector() -> AdaptiveFrameworkSelector:
    """Get or create global adaptive selector."""
    global _selector
    if _selector is None:
        _selector = AdaptiveFrameworkSelector()
    return _selector
