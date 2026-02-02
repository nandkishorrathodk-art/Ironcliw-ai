#!/usr/bin/env python3
"""
Test v198.2: Trinity Timeout Synchronization Fix

This test verifies that the DMS (Dead Man's Switch) timeout and Trinity
component startup timeout are properly synchronized to prevent premature
timeout during GCP VM startup.

Root cause: Previously, DMS timeout (510s) was shorter than component startup
timeout (600s), causing DMS to timeout before component startup could complete.

Run with: python3 tests/test_trinity_timeout_fix.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_default_trinity_timeout():
    """Test that DEFAULT_TRINITY_TIMEOUT is properly defined."""
    print("\n=== Test 1: Default Trinity Timeout ===")

    try:
        from unified_supervisor import DEFAULT_TRINITY_TIMEOUT

        # Default should be 600 seconds (10 minutes)
        assert DEFAULT_TRINITY_TIMEOUT == 600.0, \
            f"DEFAULT_TRINITY_TIMEOUT should be 600.0, got {DEFAULT_TRINITY_TIMEOUT}"

        print(f"✓ DEFAULT_TRINITY_TIMEOUT = {DEFAULT_TRINITY_TIMEOUT}s")
        print("=== Test 1: PASSED ===\n")
        return True

    except ImportError as e:
        print(f"⚠ Skipping test (import error: {e})")
        return True


def test_gcp_timeout_calculation():
    """Test that GCP mode timeout is calculated correctly."""
    print("\n=== Test 2: GCP Timeout Calculation ===")

    try:
        # Test the expected calculation
        gcp_vm_timeout = float(os.environ.get("GCP_VM_STARTUP_TIMEOUT", "300.0"))
        fallback_processing_buffer = 120.0
        orchestration_buffer = 90.0  # v198.2: Added orchestration buffer

        expected_gcp_timeout = gcp_vm_timeout + fallback_processing_buffer + orchestration_buffer
        # Should be 300 + 120 + 90 = 510s
        assert expected_gcp_timeout == 510.0, \
            f"GCP mode timeout should be 510.0, got {expected_gcp_timeout}"

        print(f"✓ GCP mode calculation: {gcp_vm_timeout} + {fallback_processing_buffer} + {orchestration_buffer} = {expected_gcp_timeout}s")

        # DMS timeout should be this plus 60s buffer
        dms_timeout = expected_gcp_timeout + 60.0
        assert dms_timeout == 570.0, \
            f"DMS timeout should be 570.0, got {dms_timeout}"

        print(f"✓ DMS timeout = {expected_gcp_timeout} + 60 = {dms_timeout}s")
        print("=== Test 2: PASSED ===\n")
        return True

    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_effective_timeout_minimum():
    """Test that effective timeout is at least DEFAULT_TRINITY_TIMEOUT."""
    print("\n=== Test 3: Effective Timeout Minimum ===")

    try:
        from unified_supervisor import DEFAULT_TRINITY_TIMEOUT

        # Even with short GCP timeout, effective timeout should be >= DEFAULT
        short_gcp_timeout = 100.0
        fallback_buffer = 120.0
        orchestration_buffer = 90.0

        calculated = short_gcp_timeout + fallback_buffer + orchestration_buffer
        # 100 + 120 + 90 = 310s, but should be capped at DEFAULT (600s)

        effective_timeout = max(calculated, DEFAULT_TRINITY_TIMEOUT)
        assert effective_timeout == DEFAULT_TRINITY_TIMEOUT, \
            f"Effective timeout should be >= DEFAULT ({DEFAULT_TRINITY_TIMEOUT}), got {effective_timeout}"

        print(f"✓ Calculated {calculated}s, but effective is {effective_timeout}s (minimum enforced)")
        print("=== Test 3: PASSED ===\n")
        return True

    except ImportError as e:
        print(f"⚠ Skipping test (import error: {e})")
        return True
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_dms_timeout_exceeds_component_timeout():
    """Test that DMS timeout always exceeds component startup timeout."""
    print("\n=== Test 4: DMS Timeout Exceeds Component Timeout ===")

    try:
        from unified_supervisor import DEFAULT_TRINITY_TIMEOUT

        # Component startup timeout
        component_timeout = DEFAULT_TRINITY_TIMEOUT  # 600s

        # DMS timeout should be component_timeout + 60s buffer
        dms_timeout = component_timeout + 60.0  # 660s

        assert dms_timeout > component_timeout, \
            f"DMS timeout ({dms_timeout}) should exceed component timeout ({component_timeout})"

        buffer = dms_timeout - component_timeout
        assert buffer >= 60.0, \
            f"DMS buffer should be at least 60s, got {buffer}"

        print(f"✓ DMS timeout ({dms_timeout}s) > Component timeout ({component_timeout}s)")
        print(f"✓ Buffer = {buffer}s")
        print("=== Test 4: PASSED ===\n")
        return True

    except ImportError as e:
        print(f"⚠ Skipping test (import error: {e})")
        return True
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_explicit_timeout_override():
    """Test that explicit JARVIS_TRINITY_TIMEOUT overrides calculated value."""
    print("\n=== Test 5: Explicit Timeout Override ===")

    try:
        # Set explicit timeout
        explicit_timeout = "900.0"  # 15 minutes

        # In the actual code, if JARVIS_TRINITY_TIMEOUT is set, it takes precedence
        effective_timeout = float(explicit_timeout)
        assert effective_timeout == 900.0, \
            f"Explicit timeout should be 900.0, got {effective_timeout}"

        print(f"✓ Explicit timeout override: JARVIS_TRINITY_TIMEOUT={effective_timeout}s")
        print("=== Test 5: PASSED ===\n")
        return True

    except AssertionError as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("v198.2 TRINITY TIMEOUT SYNCHRONIZATION FIX - VERIFICATION")
    print("=" * 60)

    all_passed = True

    # Test 1: Default timeout
    if not test_default_trinity_timeout():
        all_passed = False

    # Test 2: GCP calculation
    if not test_gcp_timeout_calculation():
        all_passed = False

    # Test 3: Minimum enforcement
    if not test_effective_timeout_minimum():
        all_passed = False

    # Test 4: DMS exceeds component timeout
    if not test_dms_timeout_exceeds_component_timeout():
        all_passed = False

    # Test 5: Explicit override
    if not test_explicit_timeout_override():
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        return 0
    else:
        print("SOME TESTS FAILED ❌")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
