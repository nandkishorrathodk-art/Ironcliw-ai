#!/usr/bin/env python3
"""
Cross-Space Click Test
=======================

Demonstrates Ghost Hands ability to click buttons on windows
in different Spaces WITHOUT switching focus.

Usage:
    python3 test_cross_space_click.py
    python3 test_cross_space_click.py --demo  # Automated demo

What this does:
1. Finds all Chrome windows across all Spaces
2. Shows you which windows are on which Spaces
3. Lets you select a window to click
4. Clicks a point in that window without switching Space

Requirements:
- Yabai installed and running
- Accessibility permissions granted
- Chrome running with windows on different Spaces
"""

import asyncio
import sys
import os
import importlib.util

# Direct module load to avoid __init__.py which has numpy-dependent imports
def load_module_directly(module_name: str, file_path: str):
    """Load a module directly from file, bypassing package __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Get the path to yabai_aware_actuator.py
this_dir = os.path.dirname(os.path.abspath(__file__))
actuator_path = os.path.join(this_dir, "yabai_aware_actuator.py")

# Load it directly
yabai_actuator = load_module_directly("yabai_aware_actuator", actuator_path)

# Import what we need
get_yabai_actuator = yabai_actuator.get_yabai_actuator
YabaiWindowInfo = yabai_actuator.YabaiWindowInfo
CrossSpaceActionResult = yabai_actuator.CrossSpaceActionResult


async def list_windows(actuator, app_name: str = "Chrome") -> list[YabaiWindowInfo]:
    """List all windows for an app."""
    windows = await actuator.find_windows(app_name)

    if not windows:
        print(f"\n❌ No {app_name} windows found.")
        return []

    print(f"\n📋 Found {len(windows)} {app_name} window(s):")
    print("-" * 60)

    for i, w in enumerate(windows):
        focus_icon = "🔷" if w.is_focused else "⬜"
        visible_icon = "👁" if w.is_visible else "👻"

        print(f"  [{i}] {focus_icon} {visible_icon} Space {w.space_id}: {w.title[:50]}")
        print(f"      Window ID: {w.window_id} | PID: {w.pid}")
        print(f"      Frame: ({w.frame.x:.0f}, {w.frame.y:.0f}) {w.frame.width:.0f}x{w.frame.height:.0f}")
        print()

    return windows


async def click_center(actuator, window: YabaiWindowInfo) -> bool:
    """Click the center of a window."""
    print(f"\n🖱️  Clicking center of window {window.window_id} on Space {window.space_id}...")

    center = window.frame.center()
    # Convert to window-local coordinates
    local_x = window.frame.width / 2
    local_y = window.frame.height / 2

    report = await actuator.click_in_window(
        window_id=window.window_id,
        coordinates=(local_x, local_y),
    )

    if report.result == CrossSpaceActionResult.SUCCESS:
        print(f"✅ Click successful via {report.backend_used}")
        print(f"   Duration: {report.duration_ms:.1f}ms")
        print(f"   Focus preserved: {report.focus_preserved}")
        return True
    else:
        print(f"❌ Click failed: {report.error}")
        return False


async def click_reset_button(actuator, window: YabaiWindowInfo) -> bool:
    """Try to click a Reset button using Accessibility API."""
    print(f"\n🔄 Attempting to click 'Reset' button in window {window.window_id}...")

    report = await actuator.click_in_window(
        window_id=window.window_id,
        element_title="Reset",
        element_role="button",
    )

    if report.result == CrossSpaceActionResult.SUCCESS:
        print(f"✅ Reset button clicked via {report.backend_used}")
        return True
    else:
        print(f"⚠️  Could not find Reset button, trying generic click...")
        # Fallback to center click
        return await click_center(actuator, window)


async def interactive_test():
    """Interactive test mode."""
    print("=" * 60)
    print("🔮 Ghost Hands Cross-Space Click Test")
    print("=" * 60)

    # Initialize actuator
    print("\n⏳ Initializing Yabai-Aware Actuator...")
    actuator = await get_yabai_actuator()

    stats = actuator.get_stats()
    print(f"   Yabai: {'✅' if stats['yabai_available'] else '❌'}")
    print(f"   Accessibility: {'✅' if stats['accessibility_available'] else '❌'}")

    if not stats['yabai_available']:
        print("\n❌ Yabai is not available. Please ensure:")
        print("   1. Yabai is installed: brew install koekeishiya/formulae/yabai")
        print("   2. Yabai is running: brew services start yabai")
        return

    # Get current space
    current_space = await actuator.yabai.get_current_space()
    print(f"\n📍 You are currently on Space {current_space.index if current_space else '?'}")

    # List windows
    windows = await list_windows(actuator, "Chrome")

    if not windows:
        print("\nPlease open some Chrome windows and try again.")
        return

    # Find windows on OTHER spaces
    other_space_windows = [
        w for w in windows
        if current_space and w.space_id != current_space.space_id
    ]

    if other_space_windows:
        print(f"\n👻 {len(other_space_windows)} window(s) on OTHER spaces (ghost-clickable!):")
        for w in other_space_windows:
            print(f"   Space {w.space_id}: {w.title[:40]}...")

    # Interactive selection
    print("\n" + "-" * 60)
    print("Enter window index to click (or 'q' to quit):")

    try:
        choice = input("> ")
        if choice.lower() == 'q':
            return

        idx = int(choice)
        if 0 <= idx < len(windows):
            target = windows[idx]

            print(f"\n🎯 Target: {target.title[:50]}")
            print(f"   On Space {target.space_id} (you're on Space {current_space.index if current_space else '?'})")

            # Confirm
            print("\nPress Enter to click, or 'r' to find Reset button, or 'q' to quit:")
            action = input("> ")

            if action.lower() == 'q':
                return
            elif action.lower() == 'r':
                await click_reset_button(actuator, target)
            else:
                await click_center(actuator, target)

            # Show final stats
            print("\n📊 Final Statistics:")
            for k, v in actuator.get_stats().items():
                print(f"   {k}: {v}")
        else:
            print(f"Invalid index. Choose 0-{len(windows)-1}")

    except ValueError:
        print("Please enter a number.")
    except KeyboardInterrupt:
        print("\nAborted.")


async def demo_mode():
    """Automated demo: click all Chrome windows on other spaces."""
    print("=" * 60)
    print("🤖 Ghost Hands Automated Demo")
    print("=" * 60)
    print("\nThis will click the center of all Chrome windows on OTHER spaces.")
    print("Watch your screen - you should NOT see space switching!\n")

    actuator = await get_yabai_actuator()

    current_space = await actuator.yabai.get_current_space()
    print(f"📍 Current space: {current_space.index if current_space else '?'}")

    windows = await actuator.find_windows("Chrome")
    other_space_windows = [
        w for w in windows
        if current_space and w.space_id != current_space.space_id
    ]

    if not other_space_windows:
        print("\n⚠️  No Chrome windows on other spaces to test.")
        print("   Open Chrome windows on different spaces and try again.")
        return

    print(f"\n🎯 Will click {len(other_space_windows)} windows on other spaces...")
    await asyncio.sleep(2)

    for w in other_space_windows:
        print(f"\n→ Clicking window on Space {w.space_id}: {w.title[:40]}...")

        report = await actuator.click_in_window(
            window_id=w.window_id,
            coordinates=(w.frame.width / 2, w.frame.height / 2),
        )

        result_icon = "✅" if report.result == CrossSpaceActionResult.SUCCESS else "❌"
        print(f"   {result_icon} {report.result.name} via {report.backend_used}")

        await asyncio.sleep(0.5)

    # Verify we're still on the same space
    final_space = await actuator.yabai.get_current_space()
    if final_space and current_space and final_space.space_id == current_space.space_id:
        print(f"\n✅ SUCCESS! Still on Space {final_space.index} - focus was preserved!")
    else:
        print(f"\n⚠️  Space changed to {final_space.index if final_space else '?'}")

    print("\n📊 Statistics:")
    for k, v in actuator.get_stats().items():
        print(f"   {k}: {v}")


async def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        await demo_mode()
    else:
        await interactive_test()


if __name__ == "__main__":
    asyncio.run(main())
