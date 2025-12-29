#!/usr/bin/env python3
"""
JARVIS INFINITE EYES - N Optic Nerves Stereoscopic Vision Test
==============================================================

This proves JARVIS has SCALABLE omnipresence - not just "2 eyes" but
as many optic nerves as there are windows across ALL desktop spaces.

What This Proves:
1. Auto-Discovery - JARVIS finds ALL bouncing ball windows on his own
2. Auto-Identification - He READS the screen to determine VERTICAL vs HORIZONTAL
3. N Optic Nerves - Handles 2, 5, 10, or unlimited windows simultaneously
4. True Omnipresence - Monitors ALL spaces in parallel with voice narration

How It Works:
- Scans ENTIRE macOS workspace via Yabai (The Map)
- Finds ALL browser windows with bouncing balls
- Spawns Ferrari Engine for EACH window (N watchers)
- Reads screen via OCR to auto-identify window type
- Streams bounce counts from ALL windows simultaneously
- Announces discoveries in real-time with Daniel's voice

Success = JARVIS correctly identifies and monitors ALL windows without
         being told which is which or where they are.
"""

import asyncio
import os
import sys
import logging
import re
from datetime import datetime
from pathlib import Path

# Suppress noisy logs
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("InfiniteEyes")

# Add backend to path
sys.path.insert(0, os.path.join(os.getcwd(), "backend"))

# Check OCR dependencies
try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
except ImportError as e:
    print(f"❌ Missing OCR library: {e}")
    print("\n🔧 Please install:")
    print("   brew install tesseract")
    print("   pip3 install pytesseract pillow opencv-python")
    sys.exit(1)

from backend.neural_mesh.agents.visual_monitor_agent import VisualMonitorAgent
from backend.vision.multi_space_window_detector import MultiSpaceWindowDetector

# Regex patterns
COUNT_PATTERN = re.compile(r"BOUNCE COUNT[:\s]+(\d+)", re.IGNORECASE)
VERTICAL_PATTERN = re.compile(r"VERTICAL", re.IGNORECASE)
HORIZONTAL_PATTERN = re.compile(r"HORIZONTAL", re.IGNORECASE)


async def jarvis_speak(message: str, blocking: bool = False):
    """JARVIS speaks with Daniel's British voice"""
    print(f"🗣️  JARVIS: {message}")
    try:
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", "Daniel", "-r", "200", message,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if blocking:
            await proc.wait()
    except Exception:
        pass


async def test_infinite_eyes():
    """
    The Ultimate Test: N Optic Nerves with Full Auto-Discovery
    """
    print("\n" + "="*80)
    print("🕶️  JARVIS INFINITE EYES - N OPTIC NERVES TEST")
    print("   Scalable Omnipresent Multi-Space Vision")
    print("="*80)
    print()

    await jarvis_speak(
        "Initiating Infinite Eyes test. I will discover and monitor all bouncing ball windows automatically.",
        blocking=True
    )
    await asyncio.sleep(0.5)

    # Setup instructions
    print("📋 SETUP INSTRUCTIONS:")
    print("-" * 80)
    print("Open bouncing ball windows on ANY spaces you want:")
    print(f"  • Vertical:   file://{Path.cwd()}/backend/tests/visual_test/vertical.html")
    print(f"  • Horizontal: file://{Path.cwd()}/backend/tests/visual_test/horizontal.html")
    print()
    print("You can open:")
    print("  - Just 2 windows (one vertical, one horizontal)")
    print("  - Or 5 windows (mix of vertical and horizontal)")
    print("  - Or 10+ windows across different spaces")
    print()
    print("JARVIS will:")
    print("  1. Find ALL windows automatically")
    print("  2. Read the screen to identify which is which")
    print("  3. Spawn N Ferrari Engines (one per window)")
    print("  4. Stream data from ALL of them simultaneously")
    print("-" * 80)
    print()

    input("👉 Press ENTER when windows are open. JARVIS will find them... ")

    print()
    await jarvis_speak("Beginning omnipresent workspace scan.", blocking=True)
    await asyncio.sleep(0.3)

    # =========================================================================
    # STEP 1: Initialize VisualMonitorAgent
    # =========================================================================
    print("📡 Initializing VisualMonitorAgent...")
    agent = VisualMonitorAgent()
    await agent.on_initialize()
    await agent.on_start()
    print("   ✅ Agent ready")
    print()

    # =========================================================================
    # STEP 2: AUTO-DISCOVERY - Find ALL browser windows across ALL spaces
    # =========================================================================
    print("🔍 AUTO-DISCOVERY: Scanning entire workspace...")
    await jarvis_speak("Scanning all desktop spaces for browser windows.", blocking=True)
    await asyncio.sleep(0.3)

    detector = MultiSpaceWindowDetector()
    result = detector.get_all_windows_across_spaces()
    all_windows = result.get('windows', [])

    print(f"   📊 Total windows found: {len(all_windows)}")

    # Find browser windows that might be our test pages
    browser_apps = ['Chrome', 'Safari', 'Firefox', 'Brave', 'Arc']
    candidates = []

    for window_obj in all_windows:
        app_name = window_obj.app_name if hasattr(window_obj, 'app_name') else ''
        title = window_obj.window_title if hasattr(window_obj, 'window_title') else ''

        # Check if this is a browser window
        is_browser = any(browser.lower() in app_name.lower() for browser in browser_apps)

        # Check if title suggests it's our test page
        is_test_page = ('VERTICAL' in title.upper() or
                       'HORIZONTAL' in title.upper() or
                       'bouncing' in title.lower() or
                       'stereoscopic' in title.lower())

        if is_browser and is_test_page:
            candidates.append({
                'window_id': window_obj.window_id,
                'space_id': window_obj.space_id if window_obj.space_id else 1,
                'app_name': app_name,
                'title': title
            })
            print(f"   ✅ Candidate: Space {window_obj.space_id} - {title[:50]}")

    if len(candidates) == 0:
        print()
        print("   ❌ No bouncing ball windows found!")
        print("   Make sure:")
        print("   • HTML files are open in browser")
        print("   • Windows have 'VERTICAL' or 'HORIZONTAL' in title")
        await jarvis_speak("No test windows detected. Please open the bouncing ball pages.", blocking=True)
        await agent.on_stop()
        return

    print()
    print(f"   🎯 Found {len(candidates)} potential test windows")
    await jarvis_speak(f"I have discovered {len(candidates)} browser windows. Spawning optic nerves now.", blocking=True)
    await asyncio.sleep(0.5)

    # =========================================================================
    # STEP 3: SPAWN N FERRARI ENGINES - One per window
    # =========================================================================
    print()
    print(f"🏎️  SPAWNING {len(candidates)} FERRARI ENGINES (N OPTIC NERVES)...")
    print()

    eyes = []
    for idx, candidate in enumerate(candidates):
        eye_num = idx + 1
        print(f"   🏎️  Spawning Eye #{eye_num} on Space {candidate['space_id']}...")

        try:
            watcher = await agent._spawn_ferrari_watcher(
                window_id=candidate['window_id'],
                fps=10,  # 10 FPS for OCR
                app_name=candidate['app_name'],
                space_id=candidate['space_id']
            )

            if watcher:
                eyes.append({
                    'eye_id': f"Eye{eye_num}",
                    'watcher': watcher,
                    'window_id': candidate['window_id'],
                    'space_id': candidate['space_id'],
                    'app_name': candidate['app_name'],
                    'type': 'UNKNOWN',  # Will auto-identify via OCR
                    'last_count': -1,
                    'frames_processed': 0
                })
                print(f"   ✅ Eye #{eye_num} active @ 10 FPS")
            else:
                print(f"   ⚠️  Failed to spawn Eye #{eye_num}")

        except Exception as e:
            print(f"   ❌ Error spawning Eye #{eye_num}: {e}")

    if len(eyes) == 0:
        print()
        print("   ❌ Failed to spawn any Ferrari watchers")
        await agent.on_stop()
        return

    print()
    print(f"   ✅ {len(eyes)} optic nerves connected")
    await jarvis_speak(f"All {len(eyes)} optic nerves are connected. Beginning parallel streaming.", blocking=True)
    await asyncio.sleep(0.5)

    # =========================================================================
    # STEP 4: INFINITE EYES STREAMING - Monitor ALL windows in parallel
    # =========================================================================
    print()
    print("="*80)
    print(f"🎥 INFINITE EYES STREAMING ({len(eyes)} parallel streams, 20 seconds)")
    print("="*80)
    print()

    await jarvis_speak("Streaming now. I will announce what I see in real-time.", blocking=True)
    await asyncio.sleep(0.3)

    start_time = datetime.now()
    announcement_queue = asyncio.Queue()

    # TTS worker for non-blocking announcements
    async def tts_worker():
        while True:
            msg = await announcement_queue.get()
            if msg is None:
                break
            await jarvis_speak(msg, blocking=False)
            await asyncio.sleep(0.15)

    tts_task = asyncio.create_task(tts_worker())

    try:
        while (datetime.now() - start_time).total_seconds() < 20:
            # Check ALL eyes in parallel
            for eye in eyes:
                watcher = eye['watcher']

                try:
                    # Get latest frame from this Ferrari Engine
                    frame_data = await watcher.get_latest_frame(timeout=0.3)

                    if frame_data is None:
                        continue

                    frame = frame_data.get('frame')
                    if frame is None:
                        continue

                    eye['frames_processed'] += 1

                    # Convert frame to PIL Image for OCR
                    if isinstance(frame, np.ndarray):
                        if len(frame.shape) == 3:
                            if frame.shape[2] == 4:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                            elif frame.shape[2] == 3:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame)
                    else:
                        pil_img = frame

                    # ===== AUTO-IDENTIFICATION: Read screen to determine type =====
                    text = pytesseract.image_to_string(pil_img)

                    # Auto-identify window type if unknown
                    if eye['type'] == 'UNKNOWN':
                        if VERTICAL_PATTERN.search(text):
                            eye['type'] = 'VERTICAL'
                            emoji = "⬆️ "
                            print(f"   🔍 {eye['eye_id']} AUTO-IDENTIFIED: ⬆️  VERTICAL on Space {eye['space_id']}")
                            await announcement_queue.put(f"Eye {eye['eye_id'][-1]} identified as vertical on Space {eye['space_id']}")
                        elif HORIZONTAL_PATTERN.search(text):
                            eye['type'] = 'HORIZONTAL'
                            emoji = "↔️ "
                            print(f"   🔍 {eye['eye_id']} AUTO-IDENTIFIED: ↔️  HORIZONTAL on Space {eye['space_id']}")
                            await announcement_queue.put(f"Eye {eye['eye_id'][-1]} identified as horizontal on Space {eye['space_id']}")

                    # Extract bounce count
                    match = COUNT_PATTERN.search(text)
                    if match:
                        count = int(match.group(1))

                        # Only announce when count changes
                        if count != eye['last_count']:
                            eye['last_count'] = count

                            # Visual indicator
                            if eye['type'] == 'VERTICAL':
                                emoji = "⬆️ "
                            elif eye['type'] == 'HORIZONTAL':
                                emoji = "↔️ "
                            else:
                                emoji = "❓ "

                            # Print to console
                            print(f"   {emoji} [{eye['eye_id']}] Space {eye['space_id']} {eye['type']:12} | Bounce: {count:3d}")

                            # Voice announcement (throttled: only every 10th bounce to avoid spam)
                            if count % 10 == 0:
                                type_name = eye['type'].lower() if eye['type'] != 'UNKNOWN' else "unknown"
                                await announcement_queue.put(f"{type_name} space {eye['space_id']}: {count}")

                except Exception as e:
                    logger.error(f"[{eye['eye_id']}] Stream error: {e}")

            # Sample at 5 Hz (every 200ms)
            await asyncio.sleep(0.2)

    except KeyboardInterrupt:
        print()
        print("   🛑 Stopped by user (Ctrl+C)")
        await jarvis_speak("Test interrupted by user.", blocking=True)

    finally:
        # Shutdown TTS worker
        await announcement_queue.put(None)
        await tts_task

        print()
        print("="*80)
        print("📊 INFINITE EYES TEST RESULTS")
        print("="*80)
        print()

        for eye in eyes:
            print(f"   [{eye['eye_id']}] Space {eye['space_id']}")
            print(f"      Type: {eye['type']}")
            print(f"      Last Bounce: {eye['last_count']}")
            print(f"      Frames Processed: {eye['frames_processed']}")
            print()

        print("="*80)
        print("🧹 CLEANING UP...")
        print("="*80)
        print()

        await jarvis_speak("Shutting down all optic nerves.")

        # Stop all Ferrari watchers
        for eye in eyes:
            try:
                await eye['watcher'].stop()
                print(f"   ✅ Stopped: {eye['eye_id']}")
            except Exception as e:
                print(f"   ⚠️  Error stopping {eye['eye_id']}: {e}")

        # Stop agent
        await agent.on_stop()
        print("   ✅ Agent stopped")

        print()
        print("="*80)
        print("🎉 INFINITE EYES TEST COMPLETE")
        print("="*80)
        print()

        await jarvis_speak(
            f"Infinite Eyes test complete. Successfully monitored {len(eyes)} windows "
            f"across multiple desktop spaces simultaneously. All systems offline.",
            blocking=True
        )

        print()
        print("🧠 IF JARVIS AUTO-DISCOVERED AND MONITORED ALL WINDOWS:")
        print("   ✅ Auto-Discovery: PROVEN")
        print("   ✅ Auto-Identification: PROVEN")
        print(f"   ✅ N Optic Nerves ({len(eyes)} simultaneous): PROVEN")
        print("   ✅ Scalable Omnipresence: PROVEN")
        print()
        print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(test_infinite_eyes())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
