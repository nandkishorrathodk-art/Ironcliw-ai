#!/usr/bin/env python3
"""
Ferrari Engine Integration Test
Verify that ScreenCaptureKit is automatically selected as Priority 1
"""
import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from backend.vision.macos_video_capture_advanced import create_video_capture

async def test_ferrari():
    print("=" * 70)
    print("🏎️  FERRARI ENGINE INTEGRATION TEST")
    print("=" * 70)
    print("\n→ Initializing Capture Manager...")

    manager = await create_video_capture()

    # This should trigger the Priority 1: SCK path

    frame_count = 0
    methods_seen = set()

    async def on_frame(frame, metadata):
        nonlocal frame_count
        frame_count += 1
        method = metadata.get('method', 'UNKNOWN')
        methods_seen.add(method)
        fps = metadata.get('fps', 0.0)
        latency = metadata.get('capture_latency_ms', 0.0)

        # Log every 10 frames
        if frame_count % 10 == 1:
            print(f"   📸 Frame {frame_count}: Method=[{method}] FPS=[{fps:.1f}] "
                  f"Latency=[{latency:.1f}ms] Shape={frame.shape}")

    print("→ Starting capture (should use Ferrari Engine as Priority 1)...\n")
    success = await manager.start_capture(on_frame)

    if not success:
        print("❌ Failed to start capture!")
        return False

    print("→ Capture started. Collecting frames for 5 seconds...\n")
    await asyncio.sleep(5)

    print("\n→ Stopping capture...")
    await manager.stop_capture()

    # Print results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    print(f"Total Frames Captured: {frame_count}")
    print(f"Capture Methods Used: {methods_seen}")
    print(f"Capture Manager Metrics: {manager.get_metrics()}")

    # Verify Ferrari Engine was used
    print("\n" + "=" * 70)
    print("🔍 VERIFICATION")
    print("=" * 70)

    if 'screencapturekit' in methods_seen:
        print("✅ SUCCESS: Ferrari Engine (ScreenCaptureKit) is active!")
        print("   Priority 1 capture method confirmed.")
        print("   GPU-accelerated, adaptive FPS streaming operational.")
        return True
    elif frame_count > 0:
        print("⚠️  WARNING: Frames captured, but NOT using Ferrari Engine")
        print(f"   Methods used: {methods_seen}")
        print("   This suggests fallback to AVFoundation or screencapture")
        return False
    else:
        print("❌ FAILURE: No frames captured at all!")
        return False

if __name__ == "__main__":
    try:
        passed = asyncio.run(test_ferrari())
        print("\n" + "=" * 70)
        if passed:
            print("🏁 FERRARI ENGINE TEST: PASSED ✅")
            print("   The engine swap was successful!")
            print("   Ironcliw is now running on the Ferrari Engine.")
        else:
            print("🏁 FERRARI ENGINE TEST: FAILED ❌")
            print("   Review logs above for fallback reasons.")
        print("=" * 70 + "\n")
        sys.exit(0 if passed else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
