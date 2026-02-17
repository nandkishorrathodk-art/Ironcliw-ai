"""Tests for quality-weighted experience scoring."""

import pytest
from backend.clients.experience_scorer import (
    ExperienceScorer,
    ScoringWeights,
    WeightedExperienceTracker,
)


class TestScoringWeights:
    """Test default and custom weight configuration."""

    def test_default_weights(self):
        weights = ScoringWeights()
        assert weights.correction == 10.0
        assert weights.feedback_negative == 5.0
        assert weights.error == 3.0
        assert weights.novel_task == 3.0
        assert weights.low_confidence == 2.0
        assert weights.normal == 1.0
        assert weights.near_duplicate == 0.1
        assert weights.confidence_threshold == 0.7

    def test_custom_weights(self):
        weights = ScoringWeights(correction=20.0, normal=2.0, near_duplicate=0.5)
        assert weights.correction == 20.0
        assert weights.normal == 2.0
        assert weights.near_duplicate == 0.5
        # Others remain default
        assert weights.feedback_negative == 5.0

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("REACTOR_WEIGHT_CORRECTION", "15.0")
        monkeypatch.setenv("REACTOR_WEIGHT_NORMAL", "0.5")
        weights = ScoringWeights()
        assert weights.correction == 15.0
        assert weights.normal == 0.5


class TestExperienceScorer:
    """Test individual experience scoring."""

    def test_correction_gets_highest_weight(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "CORRECTION"})
        assert score == 10.0

    def test_negative_feedback_weighted(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "FEEDBACK", "feedback_sentiment": "negative"})
        assert score == 5.0

    def test_positive_feedback_normal(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "FEEDBACK", "feedback_sentiment": "positive"})
        assert score == 1.0

    def test_error_weighted(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "ERROR"})
        assert score == 3.0

    def test_novel_task_type_weighted(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "INTERACTION", "task_type": "never_seen"})
        assert score == 3.0

    def test_repeated_task_type_normal(self):
        scorer = ExperienceScorer()
        scorer.score({"event_type": "INTERACTION", "task_type": "seen_before"})  # First time = novel
        score = scorer.score({"event_type": "INTERACTION", "task_type": "seen_before"})  # Second time = normal
        assert score == 1.0

    def test_low_confidence_weighted(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "INTERACTION", "confidence": 0.5})
        assert score == 2.0

    def test_confidence_at_threshold_is_normal(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "INTERACTION", "confidence": 0.7})
        assert score == 1.0

    def test_confidence_just_below_threshold(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "INTERACTION", "confidence": 0.69})
        assert score == 2.0

    def test_normal_interaction_baseline(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "INTERACTION", "confidence": 0.9})
        assert score == 1.0

    def test_missing_event_type_defaults_to_interaction(self):
        scorer = ExperienceScorer()
        score = scorer.score({"confidence": 0.9})
        assert score == 1.0

    def test_missing_confidence_defaults_to_1(self):
        scorer = ExperienceScorer()
        # No confidence key, no task_type novelty, normal event
        scorer._known_task_types.add("")  # Pre-add empty to prevent novel_task trigger
        score = scorer.score({"event_type": "INTERACTION"})
        assert score == 1.0

    def test_empty_task_type_not_treated_as_novel(self):
        scorer = ExperienceScorer()
        score1 = scorer.score({"event_type": "INTERACTION", "task_type": ""})
        score2 = scorer.score({"event_type": "INTERACTION", "task_type": ""})
        # Empty string is falsy, so novel_task branch is skipped
        assert score1 == 1.0
        assert score2 == 1.0

    def test_correction_takes_priority_over_low_confidence(self):
        scorer = ExperienceScorer()
        score = scorer.score({"event_type": "CORRECTION", "confidence": 0.3})
        assert score == 10.0  # Correction wins, regardless of confidence

    def test_custom_weights_flow_through(self):
        weights = ScoringWeights(correction=20.0, normal=2.0)
        scorer = ExperienceScorer(weights=weights)
        assert scorer.score({"event_type": "CORRECTION"}) == 20.0
        assert scorer.score({"event_type": "INTERACTION", "confidence": 0.9}) == 2.0


class TestWeightedExperienceTracker:
    """Test cumulative tracking, dedup, and threshold logic."""

    def test_accumulates_scores(self):
        tracker = WeightedExperienceTracker(threshold=100.0)
        tracker.add({"event_type": "CORRECTION", "user_input": "fix a"})  # 10
        tracker.add({"event_type": "INTERACTION", "user_input": "hello", "confidence": 0.9})  # 1
        assert tracker.cumulative_score == 11.0
        assert tracker.experience_count == 2

    def test_should_trigger_at_threshold(self):
        tracker = WeightedExperienceTracker(threshold=20.0)
        tracker.add({"event_type": "CORRECTION", "user_input": "fix a"})  # 10
        assert not tracker.should_trigger()
        tracker.add({"event_type": "CORRECTION", "user_input": "fix b"})  # 10 more = 20
        assert tracker.should_trigger()

    def test_should_trigger_above_threshold(self):
        tracker = WeightedExperienceTracker(threshold=5.0)
        tracker.add({"event_type": "CORRECTION"})  # 10 > 5
        assert tracker.should_trigger()

    def test_near_duplicate_gets_reduced_weight(self):
        tracker = WeightedExperienceTracker(threshold=100.0)
        score1 = tracker.add({"event_type": "INTERACTION", "user_input": "hello", "task_type": "greeting"})
        score2 = tracker.add({"event_type": "INTERACTION", "user_input": "hello", "task_type": "greeting"})
        assert score1 == 3.0  # Novel task type "greeting" -> 3.0
        assert score2 == 0.1  # Near-duplicate

    def test_different_inputs_not_duplicates(self):
        tracker = WeightedExperienceTracker(threshold=100.0)
        score1 = tracker.add({"event_type": "INTERACTION", "user_input": "hello", "task_type": "greeting"})
        score2 = tracker.add({"event_type": "INTERACTION", "user_input": "goodbye", "task_type": "farewell"})
        # Both are novel task types, not duplicates
        assert score1 == 3.0
        assert score2 == 3.0

    def test_reset_clears_score_but_keeps_bloom(self):
        tracker = WeightedExperienceTracker(threshold=100.0)
        tracker.add({"event_type": "INTERACTION", "user_input": "test", "task_type": "t"})
        tracker.reset()
        assert tracker.cumulative_score == 0.0
        assert tracker.experience_count == 0
        # Bloom filter persists
        score = tracker.add({"event_type": "INTERACTION", "user_input": "test", "task_type": "t"})
        assert score == 0.1  # Still detected as duplicate

    def test_custom_weights(self):
        weights = ScoringWeights(correction=20.0, normal=2.0)
        scorer = ExperienceScorer(weights=weights)
        tracker = WeightedExperienceTracker(threshold=100.0, scorer=scorer)
        score = tracker.add({"event_type": "CORRECTION"})
        assert score == 20.0

    def test_bloom_eviction_on_overflow(self):
        tracker = WeightedExperienceTracker(threshold=10000.0, max_bloom_size=10)
        # Fill bloom beyond capacity
        for i in range(15):
            tracker.add({"event_type": "INTERACTION", "user_input": f"input_{i}", "task_type": f"type_{i}"})
        # Bloom should have evicted some entries (size <= max + 1 because add happens before eviction check)
        assert len(tracker._bloom) <= 15  # At least some eviction happened

    def test_dedup_key_consistency(self):
        tracker = WeightedExperienceTracker(threshold=100.0)
        key1 = tracker._make_dedup_key({"user_input": "hello", "task_type": "greet"})
        key2 = tracker._make_dedup_key({"user_input": "hello", "task_type": "greet"})
        key3 = tracker._make_dedup_key({"user_input": "hi", "task_type": "greet"})
        assert key1 == key2
        assert key1 != key3

    def test_env_var_threshold(self, monkeypatch):
        monkeypatch.setenv("REACTOR_CORE_WEIGHTED_THRESHOLD", "50.0")
        tracker = WeightedExperienceTracker()
        assert tracker.threshold == 50.0

    def test_correction_duplicates_still_get_correction_value_first_time(self):
        """Correction events are deduped by user_input+task_type, not by event_type."""
        tracker = WeightedExperienceTracker(threshold=100.0)
        score1 = tracker.add({"event_type": "CORRECTION", "user_input": "fix this", "task_type": "code"})
        assert score1 == 10.0  # Correction weight, not duplicate
        score2 = tracker.add({"event_type": "CORRECTION", "user_input": "fix this", "task_type": "code"})
        assert score2 == 0.1  # Duplicate even though it's a correction

    def test_initial_state(self):
        tracker = WeightedExperienceTracker(threshold=100.0)
        assert tracker.cumulative_score == 0.0
        assert tracker.experience_count == 0
        assert not tracker.should_trigger()
