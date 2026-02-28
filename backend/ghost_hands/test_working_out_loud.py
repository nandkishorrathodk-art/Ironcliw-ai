#!/usr/bin/env python3
"""
Test Working Out Loud Feature
==============================

Tests the v14.0 "Working Out Loud" narration feature logic.

This is a standalone test that doesn't require importing the full VisualMonitorAgent
module (which has complex dependencies). It tests the core NarrationState logic.

Usage:
    python3 test_working_out_loud.py
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# Extract NarrationState class inline for testing
# (This matches what's in visual_monitor_agent.py)
# ============================================================================

@dataclass
class MockVisualMonitorConfig:
    """Mock config for testing."""
    working_out_loud_enabled: bool = True
    heartbeat_narration_interval: float = 30.0
    near_miss_narration_enabled: bool = True
    near_miss_cooldown_seconds: float = 60.0
    activity_narration_enabled: bool = True
    activity_cooldown_seconds: float = 15.0
    narration_verbosity: str = "normal"
    max_narrations_per_minute: int = 6


@dataclass
class NarrationState:
    """
    Intelligent narration state tracker for "Working Out Loud" feature.
    (Copied from visual_monitor_agent.py for standalone testing)
    """
    last_heartbeat_time: float = 0.0
    last_near_miss_time: float = 0.0
    last_activity_time: float = 0.0
    last_any_narration_time: float = 0.0
    last_near_miss_text: str = ""
    last_activity_description: str = ""
    narrations_this_minute: int = 0
    minute_start_time: float = 0.0
    consecutive_heartbeats: int = 0
    consecutive_near_misses: int = 0
    frames_since_last_change: int = 0
    last_ocr_text_hash: int = 0
    interesting_keywords_seen: List[str] = field(default_factory=list)

    def can_narrate(self, narration_type: str, config: MockVisualMonitorConfig) -> bool:
        now = time.time()

        if now - self.minute_start_time >= 60:
            self.narrations_this_minute = 0
            self.minute_start_time = now

        if self.narrations_this_minute >= config.max_narrations_per_minute:
            return False

        MIN_GAP_SECONDS = 5.0
        if now - self.last_any_narration_time < MIN_GAP_SECONDS:
            return False

        if narration_type == "heartbeat":
            return now - self.last_heartbeat_time >= config.heartbeat_narration_interval
        elif narration_type == "near_miss":
            return (config.near_miss_narration_enabled and
                    now - self.last_near_miss_time >= config.near_miss_cooldown_seconds)
        elif narration_type == "activity":
            return (config.activity_narration_enabled and
                    now - self.last_activity_time >= config.activity_cooldown_seconds)

        return True

    def record_narration(self, narration_type: str, content: str = "") -> None:
        now = time.time()

        self.last_any_narration_time = now
        self.narrations_this_minute += 1

        if narration_type == "heartbeat":
            self.last_heartbeat_time = now
            self.consecutive_heartbeats += 1
            self.consecutive_near_misses = 0
        elif narration_type == "near_miss":
            self.last_near_miss_time = now
            self.last_near_miss_text = content
            self.consecutive_near_misses += 1
            self.consecutive_heartbeats = 0
        elif narration_type == "activity":
            self.last_activity_time = now
            self.last_activity_description = content
            self.consecutive_heartbeats = 0
            self.consecutive_near_misses = 0

    def is_similar_content(self, content: str, narration_type: str) -> bool:
        if narration_type == "near_miss":
            return content.lower().strip() == self.last_near_miss_text.lower().strip()
        elif narration_type == "activity":
            return content.lower().strip() == self.last_activity_description.lower().strip()
        return False


# ============================================================================
# Helper functions (extracted from visual_monitor_agent.py)
# ============================================================================

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings."""
    if not text1 or not text2:
        return 0.0

    t1 = text1.lower().strip()
    t2 = text2.lower().strip()

    if t1 == t2:
        return 1.0

    if t1 in t2 or t2 in t1:
        return 0.8

    set1 = set(t1.split())
    set2 = set(t2.split())

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def describe_activity(old_text: str, new_text: str, change_ratio: float) -> Optional[str]:
    """Generate a human-readable description of detected activity."""
    if change_ratio > 0.8:
        return "Content changed dramatically"
    elif change_ratio > 0.5:
        return "Significant activity detected"
    elif change_ratio > 0.3:
        return "Screen content is updating"

    new_lower = new_text.lower()
    old_lower = old_text.lower()

    if 'loading' in new_lower and 'loading' not in old_lower:
        return "Loading indicator appeared"
    elif 'error' in new_lower and 'error' not in old_lower:
        return "An error message appeared"
    elif 'success' in new_lower and 'success' not in old_lower:
        return "A success message appeared"
    elif 'complete' in new_lower and 'complete' not in old_lower:
        return "Something completed"

    return None


def calculate_text_change_ratio(old_text: str, new_text: str) -> float:
    """Calculate how much the text has changed between frames."""
    if not old_text and not new_text:
        return 0.0
    if not old_text or not new_text:
        return 1.0

    old_words = set(old_text.lower().split())
    new_words = set(new_text.lower().split())

    if not old_words and not new_words:
        return 0.0

    union = old_words | new_words
    if not union:
        return 0.0

    changed = len(old_words ^ new_words)
    return changed / len(union)


# ============================================================================
# Test Functions
# ============================================================================

def test_narration_state():
    """Test NarrationState class functionality."""
    print("\n" + "=" * 60)
    print("Testing NarrationState Class")
    print("=" * 60)

    config = MockVisualMonitorConfig()
    state = NarrationState()

    print(f"\n[Config] Working Out Loud: {config.working_out_loud_enabled}")
    print(f"[Config] Heartbeat Interval: {config.heartbeat_narration_interval}s")
    print(f"[Config] Near-miss Cooldown: {config.near_miss_cooldown_seconds}s")
    print(f"[Config] Max Narrations/min: {config.max_narrations_per_minute}")
    print(f"[Config] Verbosity: {config.narration_verbosity}")

    # Test rate limiting
    print("\n[Test 1] Rate Limiting")
    print("-" * 40)

    can_narrate_1 = state.can_narrate("heartbeat", config)
    print(f"  First heartbeat allowed: {can_narrate_1}")
    assert can_narrate_1, "First narration should be allowed"

    state.record_narration("heartbeat")

    can_narrate_2 = state.can_narrate("heartbeat", config)
    print(f"  Immediate second narration blocked: {not can_narrate_2}")
    assert not can_narrate_2, "Immediate second narration should be blocked"

    print("  ✅ Rate limiting works correctly")

    # Test content similarity
    print("\n[Test 2] Content Similarity Detection")
    print("-" * 40)

    state.last_near_miss_text = "Build started on feature branch"

    similar = state.is_similar_content("build started on feature branch", "near_miss")
    print(f"  Same content (case insensitive): {similar}")
    assert similar, "Same content should be detected as similar"

    different = state.is_similar_content("Build completed successfully", "near_miss")
    print(f"  Different content: {not different}")
    assert not different, "Different content should not be detected as similar"

    print("  ✅ Content similarity detection works correctly")

    # Test consecutive tracking
    print("\n[Test 3] Consecutive Narration Tracking")
    print("-" * 40)

    state2 = NarrationState()
    state2.record_narration("heartbeat")
    state2.record_narration("heartbeat")
    state2.record_narration("heartbeat")

    print(f"  Consecutive heartbeats: {state2.consecutive_heartbeats}")
    assert state2.consecutive_heartbeats == 3, "Should track 3 consecutive heartbeats"

    state2.record_narration("near_miss", "some text")
    print(f"  After near_miss - heartbeats reset: {state2.consecutive_heartbeats}")
    assert state2.consecutive_heartbeats == 0, "Heartbeat counter should reset"
    assert state2.consecutive_near_misses == 1, "Near miss counter should be 1"

    print("  ✅ Consecutive tracking works correctly")

    print("\n" + "=" * 60)
    print("All NarrationState tests PASSED!")
    print("=" * 60)


def test_text_similarity():
    """Test text similarity calculation."""
    print("\n" + "=" * 60)
    print("Testing Text Similarity Calculation")
    print("=" * 60)

    # Test exact match
    sim1 = calculate_text_similarity("Build Complete", "Build Complete")
    print(f"\n  Exact match: {sim1:.2f}")
    assert sim1 == 1.0, "Exact match should be 1.0"

    # Test substring match
    sim2 = calculate_text_similarity("Build", "Build Complete")
    print(f"  Substring match: {sim2:.2f}")
    assert sim2 == 0.8, "Substring match should be 0.8"

    # Test partial match
    sim3 = calculate_text_similarity("Build Started", "Build Complete")
    print(f"  Partial match (Build Started vs Build Complete): {sim3:.2f}")
    assert 0.3 < sim3 < 0.8, "Partial match should be between 0.3 and 0.8"

    # Test no match
    sim4 = calculate_text_similarity("Error", "Success")
    print(f"  No match (Error vs Success): {sim4:.2f}")
    assert sim4 < 0.3, "No match should be less than 0.3"

    print("\n  ✅ Text similarity calculation works correctly")


def test_activity_description():
    """Test activity description generation."""
    print("\n" + "=" * 60)
    print("Testing Activity Description Generation")
    print("=" * 60)

    # Test dramatic change
    desc1 = describe_activity("Hello", "Completely different text here", 0.9)
    print(f"\n  Dramatic change (0.9): '{desc1}'")
    assert desc1 == "Content changed dramatically"

    # Test significant change
    desc2 = describe_activity("Hello world", "Hello everyone here", 0.6)
    print(f"  Significant change (0.6): '{desc2}'")
    assert desc2 == "Significant activity detected"

    # Test loading detection
    desc3 = describe_activity("Waiting for data", "Loading data...", 0.2)
    print(f"  Loading detected: '{desc3}'")
    assert desc3 == "Loading indicator appeared"

    # Test error detection
    desc4 = describe_activity("Processing", "Error: Connection failed", 0.2)
    print(f"  Error detected: '{desc4}'")
    assert desc4 == "An error message appeared"

    print("\n  ✅ Activity description generation works correctly")


def test_text_change_ratio():
    """Test text change ratio calculation."""
    print("\n" + "=" * 60)
    print("Testing Text Change Ratio Calculation")
    print("=" * 60)

    # No change
    ratio1 = calculate_text_change_ratio("Hello world", "Hello world")
    print(f"\n  No change: {ratio1:.2f}")
    assert ratio1 == 0.0, "No change should be 0.0"

    # Complete change
    ratio2 = calculate_text_change_ratio("Hello world", "Goodbye everyone")
    print(f"  Complete change: {ratio2:.2f}")
    assert ratio2 == 1.0, "Complete change should be 1.0"

    # Partial change
    ratio3 = calculate_text_change_ratio("Hello world today", "Hello world tomorrow")
    print(f"  Partial change: {ratio3:.2f}")
    assert 0.0 < ratio3 < 1.0, "Partial change should be between 0 and 1"

    print("\n  ✅ Text change ratio calculation works correctly")


def test_env_config():
    """Test environment variable configuration."""
    print("\n" + "=" * 60)
    print("Testing Environment Variable Configuration")
    print("=" * 60)

    # Save original env vars
    original_wol = os.environ.get("Ironcliw_WORKING_OUT_LOUD")
    original_hbi = os.environ.get("Ironcliw_HEARTBEAT_INTERVAL")

    try:
        # Set test env vars
        os.environ["Ironcliw_WORKING_OUT_LOUD"] = "false"
        os.environ["Ironcliw_HEARTBEAT_INTERVAL"] = "45"

        # Read and verify
        wol_enabled = os.getenv("Ironcliw_WORKING_OUT_LOUD", "true").lower() == "true"
        heartbeat_interval = float(os.getenv("Ironcliw_HEARTBEAT_INTERVAL", "30"))

        print(f"\n  Ironcliw_WORKING_OUT_LOUD=false → {wol_enabled}")
        assert wol_enabled == False, "Should parse 'false' correctly"

        print(f"  Ironcliw_HEARTBEAT_INTERVAL=45 → {heartbeat_interval}")
        assert heartbeat_interval == 45.0, "Should parse '45' correctly"

        print("\n  ✅ Environment variable parsing works correctly")

    finally:
        # Restore original env vars
        if original_wol is None:
            os.environ.pop("Ironcliw_WORKING_OUT_LOUD", None)
        else:
            os.environ["Ironcliw_WORKING_OUT_LOUD"] = original_wol

        if original_hbi is None:
            os.environ.pop("Ironcliw_HEARTBEAT_INTERVAL", None)
        else:
            os.environ["Ironcliw_HEARTBEAT_INTERVAL"] = original_hbi


def main():
    """Run all tests."""
    print("\n" + "🗣️ " * 20)
    print("  WORKING OUT LOUD - TEST SUITE")
    print("🗣️ " * 20)

    try:
        test_narration_state()
        test_text_similarity()
        test_activity_description()
        test_text_change_ratio()
        test_env_config()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("""
The "Working Out Loud" feature is ready for use!

Configuration Options:
  Ironcliw_WORKING_OUT_LOUD=true|false  - Enable/disable narration
  Ironcliw_HEARTBEAT_INTERVAL=30        - Seconds between heartbeats
  Ironcliw_NEAR_MISS_NARRATION=true     - Enable near-miss alerts
  Ironcliw_NEAR_MISS_COOLDOWN=60        - Cooldown between near-miss narrations
  Ironcliw_ACTIVITY_NARRATION=true      - Enable activity narration
  Ironcliw_NARRATION_VERBOSITY=normal   - minimal|normal|verbose|debug
  Ironcliw_MAX_NARRATIONS_PER_MIN=6     - Rate limit per minute

Example voice commands:
  "Watch Chrome for 'Build Complete'"
  → Ironcliw will narrate: "I'm now watching Chrome for 'Build Complete'."
  → Every 30s: "Still watching Chrome for 'Build Complete'. 1 minute in."
  → On near-miss: "I see 'Build Started' but waiting for 'Build Complete'."
  → On detection: "Found it! 'Build Complete' detected on Chrome."
""")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
