#!/usr/bin/env python3
"""
Simplified God Mode Test - Diagnostic Version
==============================================

A simpler version of the God Mode test for debugging initialization issues.
This version has minimal dependencies and better error handling.
"""

import asyncio
import os
import sys
import subprocess

# Add backend to path
sys.path.insert(0, os.path.join(os.getcwd(), "backend"))


def speak(message: str):
    """Simple synchronous speak function."""
    print(f"🗣️  Ironcliw: {message}")
    try:
        subprocess.run(
            ["say", "-v", "Daniel", "-r", "200", message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10
        )
    except Exception as e:
        print(f"   (TTS error: {e})")


async def test_god_mode_simple():
    """Simple God Mode test with minimal dependencies."""

    print("\n" + "="*70)
    print("🧪 Ironcliw GOD MODE - SIMPLIFIED DIAGNOSTIC TEST")
    print("="*70)
    print()

    speak("Initializing God Mode diagnostic test.")

    # Step 1: Test basic imports
    print("📦 Step 1: Testing imports...")
    speak("Testing module imports.")

    try:
        from backend.neural_mesh.agents.visual_monitor_agent import VisualMonitorAgent
        print("   ✅ VisualMonitorAgent imported")
    except Exception as e:
        print(f"   ❌ VisualMonitorAgent import failed: {e}")
        speak("Visual Monitor Agent import failed. Aborting test.")
        return

    try:
        from backend.vision.multi_space_window_detector import MultiSpaceWindowDetector
        print("   ✅ MultiSpaceWindowDetector imported")
        multi_space_available = True
    except Exception as e:
        print(f"   ⚠️  MultiSpaceWindowDetector unavailable: {e}")
        multi_space_available = False

    # Step 2: Create agent
    print("\n🔧 Step 2: Creating VisualMonitorAgent...")
    speak("Creating visual monitoring agent.")

    try:
        agent = VisualMonitorAgent()
        print("   ✅ Agent created")
    except Exception as e:
        print(f"   ❌ Agent creation failed: {e}")
        speak("Agent creation failed. Aborting test.")
        return

    # Step 3: Initialize agent
    print("\n⚙️  Step 3: Initializing agent...")
    speak("Initializing agent systems.")

    try:
        await agent.on_initialize()
        print("   ✅ Agent initialized")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        speak("Initialization failed. Continuing anyway.")

    try:
        await agent.on_start()
        print("   ✅ Agent started")
    except Exception as e:
        print(f"   ❌ Start failed: {e}")
        speak("Agent start failed. Continuing anyway.")

    # Step 4: Test window detection
    print("\n🔍 Step 4: Testing window detection...")
    speak("Testing multi-space window detection.")

    try:
        # Test finding windows
        windows = await agent._find_window("Terminal", find_all=True)
        if windows and len(windows) > 0:
            print(f"   ✅ Found {len(windows)} Terminal windows")
            speak(f"Found {len(windows)} Terminal windows across desktop spaces.")
            for w in windows:
                print(f"      - Window {w.get('window_id')} on Space {w.get('space_id')}")
        else:
            print("   ⚠️  No Terminal windows found")
            speak("No Terminal windows found. Please open at least one Terminal.")
    except Exception as e:
        print(f"   ❌ Window detection failed: {e}")
        import traceback
        traceback.print_exc()
        speak("Window detection failed.")

    # Step 5: Test basic watch (single window, short timeout)
    print("\n👁️  Step 5: Testing basic watch (10 second timeout)...")
    speak("Testing basic visual monitoring for 10 seconds. Type test text now.")

    try:
        result = await agent.watch(
            app_name="Terminal",
            trigger_text="TEST",
            all_spaces=False,  # Single window first
            max_duration=10.0
        )

        print(f"\n   Result: {result.get('status', 'unknown')}")

        if result.get('trigger_detected'):
            print("   ✅ Trigger detected!")
            speak("Test successful! Trigger text detected.")
        else:
            print("   ⏱️  Timeout (expected if you didn't type TEST)")
            speak("Test completed. No trigger detected, which is normal.")

    except Exception as e:
        print(f"   ❌ Watch failed: {e}")
        import traceback
        traceback.print_exc()
        speak("Watch test failed.")

    # Cleanup
    print("\n🛑 Cleanup...")
    speak("Shutting down test systems.")

    try:
        await agent.on_stop()
        print("   ✅ Agent stopped cleanly")
    except Exception as e:
        print(f"   ⚠️  Stop warning: {e}")

    print()
    print("="*70)
    print("✅ DIAGNOSTIC TEST COMPLETE")
    print("="*70)
    speak("Diagnostic test complete.")


if __name__ == "__main__":
    try:
        asyncio.run(test_god_mode_simple())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
        speak("Test interrupted.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        speak("Fatal error occurred.")
