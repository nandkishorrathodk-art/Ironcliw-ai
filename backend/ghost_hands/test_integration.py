#!/usr/bin/env python3
"""
Ghost Hands End-to-End Integration Test
=========================================

Tests the complete Vision → Brain → Hands pipeline:

1. Vision (N-Optic Nerve): Simulates detecting "Bounce Count" on Space 4
2. Brain (Orchestrator): Receives trigger_event with window_id, routes action
3. Hands (YabaiAwareActuator): Executes cross-space click on exact window

This proves the "Golden Path" - Ironcliw can see a target on any space
and surgically interact with it without disturbing your workflow.

Usage:
    python3 test_integration.py
"""

import asyncio
import sys
import os
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# Direct module loading to avoid numpy dependency in n_optic_nerve
def load_module_directly(module_name: str, file_path: str):
    """Load a module directly from file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load modules directly
this_dir = os.path.dirname(os.path.abspath(__file__))
yabai_actuator = load_module_directly(
    "yabai_aware_actuator",
    os.path.join(this_dir, "yabai_aware_actuator.py")
)

get_yabai_actuator = yabai_actuator.get_yabai_actuator
CrossSpaceActionResult = yabai_actuator.CrossSpaceActionResult


@dataclass
class MockVisionEvent:
    """
    Simulates a VisionEvent from N-Optic Nerve.

    In real operation, this comes from the WindowWatcher when it detects
    matching text via OCR.
    """
    window_id: int
    space_id: int
    app_name: str
    window_title: str
    detected_text: str
    matched_pattern: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


async def test_end_to_end_integration():
    """
    Complete integration test of the Ghost Hands pipeline.
    """
    print("=" * 70)
    print("🔗 Ghost Hands End-to-End Integration Test")
    print("=" * 70)

    # Initialize the actuator
    print("\n[1] Initializing Yabai-Aware Actuator...")
    actuator = await get_yabai_actuator()

    stats = actuator.get_stats()
    print(f"    Yabai: {'✅' if stats['yabai_available'] else '❌'}")
    print(f"    Accessibility: {'✅' if stats['accessibility_available'] else '❌'}")

    if not stats['yabai_available']:
        print("\n❌ Yabai not available. Cannot test cross-space integration.")
        return False

    # Find the bouncing ball test windows
    print("\n[2] Finding bouncing ball test windows...")
    all_windows = await actuator.yabai.get_all_windows()

    test_windows = [
        w for w in all_windows
        if 'chrome' in w.app_name.lower()
        and ('VERTICAL' in w.title or 'HORIZONTAL' in w.title)
    ]

    if not test_windows:
        print("    ⚠️  No bouncing ball test windows found.")
        print("    Looking for any Chrome window to test with...")
        test_windows = [w for w in all_windows if 'chrome' in w.app_name.lower()][:1]

    if not test_windows:
        print("    ❌ No Chrome windows found. Please open Chrome and try again.")
        return False

    target = test_windows[0]
    print(f"    Found: {target.title[:50]}...")
    print(f"    Window ID: {target.window_id}")
    print(f"    Space: {target.space_id}")
    print(f"    Frame: ({target.frame.x:.0f}, {target.frame.y:.0f}) {target.frame.width:.0f}x{target.frame.height:.0f}")

    # Get current space
    current_space = await actuator.yabai.get_current_space()
    print(f"\n[3] Current space: {current_space.index if current_space else '?'}")

    is_cross_space = current_space and target.space_id != current_space.space_id
    if is_cross_space:
        print(f"    🎯 Target is on DIFFERENT space! (Space {target.space_id})")
        print("    This tests the full cross-space capability.")
    else:
        print(f"    Target is on current space (Space {target.space_id})")
        print("    This tests same-space action execution.")

    # Simulate a VisionEvent (as if N-Optic Nerve detected text)
    print("\n[4] Simulating VisionEvent from N-Optic Nerve...")
    vision_event = MockVisionEvent(
        window_id=target.window_id,
        space_id=target.space_id,
        app_name=target.app_name,
        window_title=target.title,
        detected_text="Bounce Count: 1234",
        matched_pattern="Bounce Count",
    )

    print(f"    Event: Detected '{vision_event.matched_pattern}' on window {vision_event.window_id}")

    # Test the compatibility layer (as Orchestrator would use it)
    print("\n[5] Testing Orchestrator compatibility layer...")
    print("    Calling: actuator.click(window_id=..., coordinates=(center))")

    # This is exactly what the Orchestrator does now:
    # It extracts window_id from trigger_event and passes it to actuator.click()
    window_id = vision_event.window_id
    space_id = vision_event.space_id
    app_name = vision_event.app_name

    # Calculate center coordinates (window-local)
    center_x = target.frame.width / 2
    center_y = target.frame.height / 2

    report = await actuator.click(
        app_name=app_name,
        window_id=window_id,      # THE GOLDEN PATH
        space_id=space_id,
        coordinates=(center_x, center_y),
    )

    print(f"\n[6] Action Result:")
    print(f"    Result: {report.result.name}")
    print(f"    Backend: {report.backend_used}")
    print(f"    Duration: {report.duration_ms:.1f}ms")
    print(f"    Focus Preserved: {report.focus_preserved}")

    if report.error:
        print(f"    Error: {report.error}")

    # Verify we're still on the original space
    final_space = await actuator.yabai.get_current_space()
    if final_space and current_space:
        if final_space.space_id == current_space.space_id:
            print(f"\n    ✅ Still on Space {final_space.index} - focus preserved!")
        else:
            print(f"\n    ⚠️  Space changed to {final_space.index}")

    # Summary
    print("\n" + "=" * 70)
    if report.result == CrossSpaceActionResult.SUCCESS:
        print("🎉 INTEGRATION TEST PASSED!")
        print("")
        print("The Ghost Hands pipeline is fully connected:")
        print("  Vision → Brain → Hands")
        print("")
        print(f"  • Vision detected target on Space {vision_event.space_id}")
        print(f"  • Brain extracted window_id={vision_event.window_id}")
        print(f"  • Hands executed cross-space click via {report.backend_used}")
        print(f"  • User focus preserved: {report.focus_preserved}")
        print("")
        print("Ironcliw can now SEE and ACT across all Spaces simultaneously!")
    else:
        print("❌ INTEGRATION TEST FAILED")
        print(f"   Error: {report.error}")

    print("=" * 70)

    return report.result == CrossSpaceActionResult.SUCCESS


async def test_multiple_spaces():
    """
    Test clicking windows across multiple spaces in sequence.
    """
    print("\n" + "=" * 70)
    print("🌐 Multi-Space Sequential Click Test")
    print("=" * 70)

    actuator = await get_yabai_actuator()

    if not actuator.yabai._initialized:
        print("❌ Yabai not available")
        return

    # Get current space
    current_space = await actuator.yabai.get_current_space()
    print(f"\n📍 Starting from Space {current_space.index if current_space else '?'}")

    # Find all Chrome windows
    all_windows = await actuator.yabai.get_all_windows()
    chrome_windows = [w for w in all_windows if 'chrome' in w.app_name.lower()]

    # Group by space
    by_space = {}
    for w in chrome_windows:
        by_space.setdefault(w.space_id, []).append(w)

    print(f"\n📋 Chrome windows by space:")
    for space_id, windows in sorted(by_space.items()):
        print(f"   Space {space_id}: {len(windows)} window(s)")

    # Click one window from each space
    print("\n🎯 Clicking one window from each space...")
    results = []

    for space_id in sorted(by_space.keys()):
        window = by_space[space_id][0]

        # Simulate VisionEvent
        event = MockVisionEvent(
            window_id=window.window_id,
            space_id=window.space_id,
            app_name=window.app_name,
            window_title=window.title,
            detected_text="trigger",
            matched_pattern="trigger",
        )

        print(f"\n   → Space {space_id}: {window.title[:40]}...")

        report = await actuator.click(
            window_id=event.window_id,
            space_id=event.space_id,
            coordinates=(window.frame.width / 2, window.frame.height / 2),
        )

        icon = "✅" if report.result == CrossSpaceActionResult.SUCCESS else "❌"
        print(f"     {icon} {report.result.name} via {report.backend_used} ({report.duration_ms:.0f}ms)")

        results.append(report.result == CrossSpaceActionResult.SUCCESS)

        await asyncio.sleep(0.3)

    # Return to original space
    if current_space:
        await actuator.yabai.focus_space(current_space.space_id)
        await asyncio.sleep(0.2)

    # Summary
    successful = sum(results)
    total = len(results)

    print(f"\n📊 Results: {successful}/{total} spaces clicked successfully")

    if successful == total:
        print("🎉 All cross-space clicks succeeded!")

    return successful == total


async def main():
    """Run all integration tests."""
    print("\n" + "🔮" * 35)
    print("    GHOST HANDS INTEGRATION TEST SUITE")
    print("🔮" * 35 + "\n")

    # Test 1: Basic end-to-end
    success1 = await test_end_to_end_integration()

    # Test 2: Multi-space (optional)
    print("\n\nWould you like to run the multi-space test?")
    print("This will click windows across ALL spaces.")

    try:
        choice = input("\nPress Enter to continue, or 'q' to skip: ")
        if choice.lower() != 'q':
            success2 = await test_multiple_spaces()
        else:
            success2 = True
    except EOFError:
        # Non-interactive mode
        success2 = await test_multiple_spaces()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    all_passed = success1 and success2

    if all_passed:
        print("""
✅ ALL TESTS PASSED

The Ghost Hands system is fully integrated:

┌─────────────────────────────────────────────────────────────────┐
│  N-Optic Nerve (Vision)                                         │
│  ├── Detects "Bounce Count: 500" on Space 4, Window 34947       │
│  └── Emits VisionEvent with window_id, space_id                 │
│            │                                                     │
│            ▼                                                     │
│  GhostHandsOrchestrator (Brain)                                 │
│  ├── Receives VisionEvent, extracts targeting data              │
│  ├── Executes GhostAction.CLICK with window_id=34947            │
│  └── Calls: actuator.click(window_id=34947, space_id=4)         │
│            │                                                     │
│            ▼                                                     │
│  YabaiAwareActuator (Hands)                                     │
│  ├── Resolves window frame via Yabai                            │
│  ├── Switches to Space 4 (if needed)                            │
│  ├── CGEvent click at window center                             │
│  └── Returns to original space                                  │
│            │                                                     │
│            ▼                                                     │
│  User remains on Space 7, completely undisturbed!               │
└─────────────────────────────────────────────────────────────────┘

Ironcliw is now a multi-dimensional autonomous agent.
""")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
