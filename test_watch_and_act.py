#!/usr/bin/env python3
"""
Watch & Act Smoke Test - Ironcliw Autonomous Loop Verification
=============================================================

This script verifies that Ironcliw can:
1. Watch a Terminal window for specific text
2. Detect when that text appears (via OCR)
3. AUTOMATICALLY take control and execute an action

This is the proof-of-concept for the complete autonomous loop:
SpatialAwareness → Visual Monitoring → Computer Use Action

Author: Ironcliw AI System
Version: 11.0 - Watch & Act Smoke Test
"""

import asyncio
import os
import sys
import logging
import subprocess
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.join(os.getcwd(), "backend"))

# Configure logging with DEBUG level for detailed diagnostics
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG for detailed monitoring
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger('backend.vision.macos_video_capture_advanced').setLevel(logging.DEBUG)
logging.getLogger('backend.vision.visual_event_detector').setLevel(logging.DEBUG)
logging.getLogger('backend.neural_mesh.agents.visual_monitor_agent').setLevel(logging.INFO)


# ============================================================================
# Ironcliw Voice - Daniel's TTS
# ============================================================================

async def jarvis_speak(message: str, rate: int = 180) -> None:
    """
    Ironcliw speaks using Daniel's voice (macOS TTS).

    Args:
        message: Text for Ironcliw to speak
        rate: Speaking rate (words per minute, default 180)
    """
    try:
        # Run macOS 'say' command asynchronously
        process = await asyncio.create_subprocess_exec(
            "say", "-v", "Daniel", "-r", str(rate), message,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        logger.debug(f"🗣️ Ironcliw: {message}")
    except Exception as e:
        logger.warning(f"TTS failed: {e}")
        # Fallback to silent operation if TTS fails
        pass


async def test_autonomous_loop():
    """
    Test the complete Watch & Act autonomous loop.

    Test Flow:
    1. Initialize VisualMonitorAgent
    2. Watch Terminal for "DEPLOYMENT READY"
    3. When detected, execute action: "Type 'echo SUCCESS' and press Enter"
    4. Verify the action was executed successfully
    """

    print("=" * 70)
    print("🚀 Ironcliw Watch & Act Smoke Test")
    print("=" * 70)
    print()
    print("This test will verify the complete autonomous loop:")
    print("  1. Visual Monitoring (watch for text)")
    print("  2. Event Detection (OCR detects trigger)")
    print("  3. Autonomous Action (Computer Use executes command)")
    print()
    print("=" * 70)
    print()

    # Check Screen Recording permission upfront
    print("🔒 Checking macOS Screen Recording permission...")
    try:
        from Quartz import (
            CGWindowListCreateImage,
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            kCGWindowImageDefault,
            CGImageGetWidth,
        )

        # Try a test capture
        test_image = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            1,  # Test window ID
            kCGWindowImageDefault
        )

        if not test_image or CGImageGetWidth(test_image) == 0:
            print()
            print("⚠️  CGWindowListCreateImage not available (expected on macOS Sequoia)")
            print("   → Will use screencapture fallback method instead")
            print("   → This is normal and expected on newer macOS versions")
            print()
            await jarvis_speak("Screen capture API requires fallback method. Using screencapture command instead.")
        else:
            print("   ✅ Screen Recording permission granted (using Quartz API)!")
            print()
    except Exception as e:
        logger.warning(f"Could not check Screen Recording permission: {e}")
        print("   ⚠️  Could not verify Screen Recording permission")
        print("   → Proceeding with fallback capture method")
        print()

    try:
        # Import after path is set
        from backend.neural_mesh.agents.visual_monitor_agent import (
            VisualMonitorAgent,
            ActionConfig,
            ActionType
        )

        # Initialize Computer Use connector WITH TTS before agent is created
        # This ensures Ironcliw can speak during autonomous actions
        from backend.display.computer_use_connector import get_computer_use_connector
        computer_use = get_computer_use_connector(tts_callback=jarvis_speak)
        logger.info("✓ Computer Use connector initialized with Ironcliw voice (Daniel)")

        print("📦 Step 1: Initializing VisualMonitorAgent...")
        await jarvis_speak("Initializing Visual Monitor Agent for autonomous loop test")

        agent = VisualMonitorAgent()

        print("   ⏳ Calling on_initialize()...")
        await agent.on_initialize()

        print("   ⏳ Calling on_start()...")
        await agent.on_start()

        print("   ✅ Agent initialized successfully!")
        await jarvis_speak("All systems initialized. Ready for autonomous monitoring.")
        print()

        # Check if components are available
        if not agent._watcher_manager:
            print("   ❌ ERROR: VideoWatcherManager not available")
            print("   Please ensure the vision system is properly configured.")
            return False

        if not agent._detector:
            print("   ❌ ERROR: VisualEventDetector not available")
            print("   Please ensure OCR dependencies are installed.")
            return False

        if not agent._computer_use_connector:
            print("   ⚠️  WARNING: Computer Use connector not available")
            print("   Action execution will fail. This test requires Computer Use.")
            print("   Continue anyway? This will test monitoring only.")
            print()

        print("=" * 70)
        print("🧪 Step 2: Starting Watch & Act Test")
        print("=" * 70)
        print()
        print("INSTRUCTIONS FOR YOU:")
        print("  1. Open a Terminal window (make sure it's visible)")
        print("  2. Type this command (but DON'T press Enter yet):")
        print()
        print("     sleep 10 && echo \"DEPLOYMENT READY\"")
        print()
        print("  3. After starting the test, immediately switch to Terminal")
        print("     and press Enter to start the countdown")
        print()
        print("  4. WATCH CLOSELY:")
        print("     - Ironcliw will watch your Terminal")
        print("     - When 'DEPLOYMENT READY' appears (after 10 seconds)")
        print("     - Ironcliw will AUTOMATICALLY type 'echo SUCCESS' and press Enter")
        print()
        print("=" * 70)
        print()

        input("Press Enter when you're ready to start the test...")
        print()

        print("🎯 STARTING AUTONOMOUS MONITORING...")
        await jarvis_speak("Starting autonomous monitoring test. Watching Terminal for deployment ready signal.")
        print("   Target App: Terminal")
        print("   Trigger Text: 'DEPLOYMENT READY'")
        print("   Action: Type 'echo SUCCESS' and press Enter")
        print()
        print("⏰ You have ~5 seconds to switch to Terminal and press Enter!")
        await jarvis_speak("You have 5 seconds to switch to Terminal and start the countdown.")
        print("   Starting monitoring in: 3...")
        await asyncio.sleep(1)
        print("   2...")
        await asyncio.sleep(1)
        print("   1...")
        await asyncio.sleep(1)
        print()
        print("🔍 MONITORING ACTIVE - Ironcliw is watching your Terminal...")
        await jarvis_speak("Monitoring active. I am now watching your Terminal window.")
        print()

        # Define the action configuration
        action_config = ActionConfig(
            action_type=ActionType.SIMPLE_GOAL,
            goal="Type 'echo SUCCESS' into the terminal and press Enter",
            switch_to_window=True,  # Ensure we switch to Terminal first
            narrate=True,  # Voice narration during execution
            timeout_seconds=30.0  # 30 seconds to execute the action
        )

        # Start the watch & act operation (BLOCKING MODE)
        # This will wait until the event is detected and action is executed
        result = await agent.watch_and_alert(
            app_name="Terminal",
            trigger_text="DEPLOYMENT READY",
            action_config=action_config,
            wait_for_completion=True  # NEW: Wait for autonomous action to complete!
        )

        print()
        print("=" * 70)
        print("📊 Test Results")
        print("=" * 70)
        print()

        if result.get("success"):
            print("✅ MONITORING PHASE: SUCCESS")
            print(f"   Watcher ID: {result.get('watcher_id')}")
            print(f"   Window ID: {result.get('window_id')}")
            print(f"   Space ID: {result.get('space_id')}")
            print()

            # Check if action was executed
            action_result = result.get('action_result')
            if action_result:
                print("🚀 ACTION EXECUTION PHASE:")
                if action_result.get('success'):
                    print("   ✅ ACTION EXECUTED SUCCESSFULLY!")
                    await jarvis_speak("Event detected! Action executed successfully. The autonomous loop is now complete.")
                    print(f"   Action Type: {action_result.get('action_type')}")
                    print(f"   Goal: {action_result.get('goal_executed')}")
                    print(f"   Duration: {action_result.get('duration_ms', 0):.2f}ms")
                    print()
                    print("🎉 AUTONOMOUS LOOP COMPLETE!")
                    print()
                    print("   Check your Terminal window - you should see:")
                    print("   1. 'DEPLOYMENT READY' (from your command)")
                    print("   2. 'SUCCESS' (typed by Ironcliw automatically!)")
                    print()
                    await jarvis_speak("Check your Terminal window. You should see the success message that I typed autonomously.")
                    return True
                else:
                    print("   ❌ ACTION EXECUTION FAILED")
                    await jarvis_speak("Action execution failed. The monitoring worked, but I could not execute the action.")
                    print(f"   Error: {action_result.get('error', 'Unknown error')}")
                    print()
                    print("   The monitoring worked, but action execution failed.")
                    print("   This usually means Computer Use is not properly configured.")
                    return False
            else:
                print("⏳ EVENT DETECTION IN PROGRESS...")
                print("   The watcher is running in the background.")
                print("   Results will be logged when the event is detected.")
                print()
                print("   If you don't see 'DEPLOYMENT READY' appear, check:")
                print("   1. Did you press Enter on the sleep command?")
                print("   2. Is the Terminal window visible?")
                print("   3. Wait the full 10 seconds + processing time")
                return None  # Pending
        else:
            print("❌ TEST FAILED")
            error_msg = result.get('error', 'Unknown error')
            print(f"   Error: {error_msg}")
            print()

            if "Could not find Terminal" in error_msg:
                await jarvis_speak("Test failed. I could not find the Terminal window. Please ensure Terminal is open and visible.")
                print("   Troubleshooting:")
                print("   1. Make sure Terminal app is open and visible")
                print("   2. Try opening a new Terminal window")
                print("   3. Ensure Terminal is not minimized")
            elif "Timeout" in error_msg or "timeout" in error_msg:
                await jarvis_speak("Test timed out. The deployment ready signal did not appear within the timeout period.")
            else:
                await jarvis_speak(f"Test failed with error: {error_msg}")

            return False

    except ImportError as e:
        print()
        print("❌ IMPORT ERROR")
        print(f"   {e}")
        print()
        print("   Make sure you're running this script from the project root:")
        print("   python3 test_watch_and_act.py")
        return False

    except Exception as e:
        print()
        print("❌ UNEXPECTED ERROR")
        print(f"   {e}")
        logger.exception("Test failed with exception:")
        return False


async def run_test_with_cleanup():
    """Run test with proper cleanup."""
    try:
        result = await test_autonomous_loop()
        return result
    finally:
        print()
        print("=" * 70)
        print("🧹 Cleanup complete")
        print("=" * 70)


def main():
    """Main entry point."""
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║           Ironcliw Watch & Act Smoke Test v11.0                  ║")
    print("║                                                                 ║")
    print("║  This test verifies the complete autonomous loop:              ║")
    print("║  Vision → Detection → Action                                   ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    # Check if we're in the right directory
    if not Path("backend/neural_mesh/agents/visual_monitor_agent.py").exists():
        print("❌ ERROR: Must be run from project root directory")
        print("   Current directory:", os.getcwd())
        print()
        print("   Please cd to the Ironcliw-AI-Agent directory and run:")
        print("   python3 test_watch_and_act.py")
        sys.exit(1)

    # Run the async test
    result = asyncio.run(run_test_with_cleanup())

    print()

    if result is True:
        print("🎉 TEST PASSED! The autonomous loop is working!")
        print()
        print("Next steps:")
        print("  1. Try more complex actions")
        print("  2. Test conditional branching")
        print("  3. Enable voice command parsing")
        sys.exit(0)
    elif result is False:
        print("❌ TEST FAILED - See errors above")
        print()
        print("Common issues:")
        print("  1. Computer Use not configured - check ClaudeComputerUseConnector")
        print("  2. Terminal not found - make sure Terminal is open and visible")
        print("  3. OCR not detecting text - check pytesseract installation")
        sys.exit(1)
    else:
        print("⏳ TEST PENDING - Monitoring is running in background")
        print("   Check logs for updates")
        sys.exit(2)


if __name__ == "__main__":
    main()
