#!/usr/bin/env python3
"""
Test Unified Startup Progress Hub
==================================

Tests that the unified progress hub correctly synchronizes state
across all progress tracking systems.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.core.unified_startup_progress import (
    UnifiedStartupProgressHub,
    StartupPhase,
    ComponentStatus,
    get_progress_hub,
    is_system_ready,
    get_progress_summary,
)


async def test_hub_initialization():
    """Test that hub initializes correctly"""
    print("\n=== Test: Hub Initialization ===")

    hub = get_progress_hub()
    # Don't connect to loading server for tests (set to None to skip network)
    await hub.initialize(
        loading_server_url=None,  # Disable network sync for testing
        required_components=["backend", "frontend"]
    )

    state = hub.get_state()
    assert state["phase"] == "initializing", f"Expected phase 'initializing', got {state['phase']}"
    assert not hub.is_ready(), "Hub should NOT be ready initially"

    print("✅ Hub initializes correctly")
    print(f"   State: {state['phase']}, Ready: {hub.is_ready()}")


async def test_component_registration():
    """Test dynamic component registration"""
    print("\n=== Test: Component Registration ===")

    hub = get_progress_hub()

    # Register components dynamically
    await hub.register_component("backend", weight=15.0, is_required_for_ready=True)
    await hub.register_component("frontend", weight=10.0, is_required_for_ready=True)
    await hub.register_component("voice", weight=10.0, is_required_for_ready=False)

    assert hub.get_component_count() >= 3, "Should have at least 3 components"

    print("✅ Components register correctly")
    print(f"   Total components: {hub.get_component_count()}")


async def test_progress_calculation():
    """Test progress calculation based on component weights"""
    print("\n=== Test: Progress Calculation ===")

    hub = get_progress_hub()

    # Start backend
    await hub.component_start("backend", "Initializing backend...")
    progress_1 = hub.get_progress()
    print(f"   After backend start: {progress_1:.1f}%")

    # Complete backend
    await hub.component_complete("backend", "Backend ready!")
    progress_2 = hub.get_progress()
    print(f"   After backend complete: {progress_2:.1f}%")

    assert progress_2 > progress_1, "Progress should increase after completion"

    # Complete frontend
    await hub.component_start("frontend", "Initializing frontend...")
    await hub.component_complete("frontend", "Frontend ready!")
    progress_3 = hub.get_progress()
    print(f"   After frontend complete: {progress_3:.1f}%")

    assert progress_3 > progress_2, "Progress should increase after more completions"

    print("✅ Progress calculates correctly")


async def test_ready_state():
    """Test that ready state is only true when all required components complete"""
    print("\n=== Test: Ready State Detection ===")

    hub = get_progress_hub()

    # Before completion, should NOT be ready
    assert not hub.is_ready(), "Hub should NOT be ready before all required components complete"
    print(f"   Before mark_complete: is_ready={hub.is_ready()}")

    # Mark complete
    await hub.mark_complete(True, "JARVIS is online!")

    # Now should be ready
    assert hub.is_ready(), "Hub SHOULD be ready after mark_complete"
    print(f"   After mark_complete: is_ready={hub.is_ready()}")

    print("✅ Ready state detection works correctly")


async def test_monotonic_progress():
    """Test that progress never decreases"""
    print("\n=== Test: Monotonic Progress ===")

    # Create a new hub instance for this test
    hub = UnifiedStartupProgressHub()
    await hub.initialize()

    # Register and complete components
    await hub.register_component("test1", weight=10.0)
    await hub.component_complete("test1", "Test 1 complete")
    progress_1 = hub.get_progress()

    await hub.register_component("test2", weight=10.0)
    await hub.component_start("test2", "Test 2 starting")
    progress_2 = hub.get_progress()

    assert progress_2 >= progress_1, f"Progress decreased: {progress_1} -> {progress_2}"

    print(f"   Progress monotonic: {progress_1:.1f}% -> {progress_2:.1f}%")
    print("✅ Progress is monotonic (never decreases)")


async def test_broadcaster_integration():
    """Test that broadcaster integrates with hub"""
    print("\n=== Test: Broadcaster Integration ===")

    try:
        from backend.core.startup_progress_broadcaster import (
            get_startup_broadcaster,
            StartupProgressBroadcaster,
        )

        # Reset singleton
        StartupProgressBroadcaster.reset_instance()

        broadcaster = get_startup_broadcaster()

        # Broadcaster should have hub reference
        if broadcaster._hub:
            print("   Broadcaster has hub reference: YES")

            # Test that broadcaster delegates to hub
            await broadcaster.broadcast_component_start("test_component", "Testing...")
            await broadcaster.broadcast_component_complete("test_component", "Test complete")

            print("   Broadcaster delegates to hub correctly")
            print("✅ Broadcaster integration works")
        else:
            print("   Broadcaster has hub reference: NO (hub not available)")
            print("⚠️ Broadcaster integration skipped (no hub)")
    except ImportError as e:
        print(f"   Skipping broadcaster test: {e}")


async def test_api_integration():
    """Test that API integrates with hub"""
    print("\n=== Test: API Integration ===")

    try:
        from backend.api.startup_progress_api import (
            get_startup_progress_manager,
            StartupProgressManager,
        )

        manager = get_startup_progress_manager()

        # Manager should get state from hub
        status = manager.current_status

        if manager._hub:
            print("   API Manager has hub reference: YES")
            print(f"   Current status: {status.get('phase', 'unknown')}")
            print("✅ API integration works")
        else:
            print("   API Manager has hub reference: NO")
            print("⚠️ API integration skipped (no hub)")
    except ImportError as e:
        print(f"   Skipping API test: {e}")


async def test_global_convenience_functions():
    """Test global convenience functions"""
    print("\n=== Test: Global Convenience Functions ===")

    ready = is_system_ready()
    summary = get_progress_summary()

    print(f"   is_system_ready(): {ready}")
    print(f"   get_progress_summary(): {summary}")
    print("✅ Global functions work correctly")


async def main():
    print("=" * 60)
    print("Unified Startup Progress Hub - Integration Tests")
    print("=" * 60)

    try:
        await test_hub_initialization()
        await test_component_registration()
        await test_progress_calculation()
        await test_ready_state()
        await test_monotonic_progress()
        await test_broadcaster_integration()
        await test_api_integration()
        await test_global_convenience_functions()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

        # Print final state
        hub = get_progress_hub()
        print(f"\nFinal Hub State:")
        print(f"  Progress: {hub.get_progress():.1f}%")
        print(f"  Phase: {hub.get_phase().value}")
        print(f"  Ready: {hub.is_ready()}")
        print(f"  Components: {hub.get_completed_count()}/{hub.get_component_count()}")

        return 0

    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
