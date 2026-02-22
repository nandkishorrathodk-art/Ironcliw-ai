"""
Test Suite for JARVIS Windows Vision System
═══════════════════════════════════════════════════════════════════════════════

Comprehensive tests for Phase 7: Vision System Port
Tests screen capture, multi-monitor, and cross-platform functionality.

Run with:
    pytest tests/vision/test_windows_vision.py -v
    python tests/vision/test_windows_vision.py  # Standalone

Author: JARVIS System
Version: 1.0.0 (Windows Port - Phase 7)
"""

import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'backend'))


def test_platform_detection():
    """Test 1: Platform Detection"""
    logger.info("=" * 70)
    logger.info("TEST 1: Platform Detection")
    logger.info("=" * 70)
    
    try:
        from platform import detector
        info = detector.PlatformDetector.get_platform_info()
        
        logger.info(f"✅ Platform: {info.os_family}")
        logger.info(f"   OS Name: {info.os_name}")
        logger.info(f"   OS Version: {info.os_version}")
        logger.info(f"   Architecture: {info.architecture}")
        logger.info(f"   Python: {info.python_version}")
        logger.info(f"   DirectML: {info.has_directml}")
        logger.info(f"   GPU: {info.has_gpu}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Platform detection failed: {e}")
        return False


def test_platform_capture_import():
    """Test 2: Platform Capture Import"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Platform Capture Import")
    logger.info("=" * 70)
    
    try:
        from vision.platform_capture import (
            get_vision_capture,
            capture_screen,
            get_monitors,
            PlatformVisionCapture,
            CaptureFrame,
            MonitorInfo
        )
        
        logger.info("✅ All platform_capture imports successful")
        logger.info(f"   - PlatformVisionCapture: {PlatformVisionCapture}")
        logger.info(f"   - CaptureFrame: {CaptureFrame}")
        logger.info(f"   - MonitorInfo: {MonitorInfo}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_windows_vision_import():
    """Test 3: Windows Vision Capture Import"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Windows Vision Capture Import")
    logger.info("=" * 70)
    
    if sys.platform != 'win32':
        logger.info("⏭️  Skipped (not Windows)")
        return True
    
    try:
        from vision.windows_vision_capture import WindowsVisionCapture
        from vision.windows_multi_monitor import WindowsMultiMonitorDetector
        
        logger.info("✅ Windows vision imports successful")
        logger.info(f"   - WindowsVisionCapture: {WindowsVisionCapture}")
        logger.info(f"   - WindowsMultiMonitorDetector: {WindowsMultiMonitorDetector}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Windows import failed: {e}")
        logger.warning("   Make sure C# DLLs are built and pythonnet is installed")
        logger.warning("   cd backend\\windows_native && .\\build.ps1")
        import traceback
        traceback.print_exc()
        return False


def test_monitor_detection():
    """Test 4: Monitor Detection"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Monitor Detection")
    logger.info("=" * 70)
    
    try:
        from vision.platform_capture import get_monitors
        
        monitors = get_monitors()
        logger.info(f"✅ Detected {len(monitors)} monitor(s)")
        
        for monitor in monitors:
            logger.info(f"   Monitor {monitor.monitor_id}:")
            logger.info(f"      - Resolution: {monitor.width}x{monitor.height}")
            logger.info(f"      - Position: ({monitor.x}, {monitor.y})")
            logger.info(f"      - Primary: {monitor.is_primary}")
            logger.info(f"      - Name: {monitor.name}")
        
        return len(monitors) > 0
    except Exception as e:
        logger.error(f"❌ Monitor detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_screen_capture():
    """Test 5: Screen Capture"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 5: Screen Capture")
    logger.info("=" * 70)
    
    try:
        from vision.platform_capture import capture_screen
        
        logger.info("Capturing screen...")
        start_time = time.time()
        frame = capture_screen(monitor_id=0)
        capture_time = (time.time() - start_time) * 1000
        
        if frame is None:
            logger.error("❌ Capture returned None")
            return False
        
        logger.info(f"✅ Screen captured successfully")
        logger.info(f"   - Capture time: {capture_time:.1f}ms")
        logger.info(f"   - Resolution: {frame.width}x{frame.height}")
        logger.info(f"   - Format: {frame.format}")
        logger.info(f"   - Data size: {frame.image_data.nbytes:,} bytes")
        
        # Test PIL conversion
        pil_image = frame.to_pil()
        logger.info(f"   - PIL conversion: {pil_image.size}")
        
        # Test bytes conversion
        png_bytes = frame.to_bytes('png')
        logger.info(f"   - PNG bytes: {len(png_bytes):,} bytes")
        
        # Performance check
        if capture_time > 100:
            logger.warning(f"⚠️  Capture time ({capture_time:.1f}ms) > 100ms target")
        
        return True
    except Exception as e:
        logger.error(f"❌ Screen capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fps_performance():
    """Test 6: FPS Performance"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 6: FPS Performance (15 FPS target)")
    logger.info("=" * 70)
    
    try:
        from vision.platform_capture import get_vision_capture
        
        capturer = get_vision_capture()
        num_captures = 30
        
        logger.info(f"Performing {num_captures} captures...")
        start_time = time.time()
        
        successful = 0
        for i in range(num_captures):
            frame = capturer.capture_screen(0)
            if frame:
                successful += 1
        
        total_time = time.time() - start_time
        fps = successful / total_time
        avg_time = (total_time / successful) * 1000 if successful > 0 else 0
        
        logger.info(f"✅ Performance test complete")
        logger.info(f"   - Successful: {successful}/{num_captures}")
        logger.info(f"   - Total time: {total_time:.2f}s")
        logger.info(f"   - Average FPS: {fps:.1f}")
        logger.info(f"   - Average capture time: {avg_time:.1f}ms")
        
        if fps >= 15:
            logger.info(f"   ✅ Target FPS achieved (>15 FPS)")
        else:
            logger.warning(f"   ⚠️  Below target FPS (<15 FPS)")
        
        return fps >= 10  # Minimum acceptable FPS
    except Exception as e:
        logger.error(f"❌ FPS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_screen_vision_integration():
    """Test 7: screen_vision.py Integration"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 7: screen_vision.py Integration")
    logger.info("=" * 70)
    
    try:
        from vision.screen_vision import ScreenVisionSystem
        import asyncio
        
        vision_system = ScreenVisionSystem()
        logger.info("✅ ScreenVisionSystem initialized")
        
        # Test async capture
        async def test_async_capture():
            image = await vision_system.capture_screen()
            return image
        
        image = asyncio.run(test_async_capture())
        
        if image:
            logger.info(f"✅ Async capture successful: {image.size}")
            return True
        else:
            logger.error("❌ Async capture returned None")
            return False
    
    except Exception as e:
        logger.error(f"❌ screen_vision integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reliable_screenshot_capture():
    """Test 8: reliable_screenshot_capture.py Integration"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 8: reliable_screenshot_capture.py Integration")
    logger.info("=" * 70)
    
    try:
        from vision.reliable_screenshot_capture import ReliableScreenshotCapture
        
        capturer = ReliableScreenshotCapture()
        logger.info(f"✅ ReliableScreenshotCapture initialized")
        logger.info(f"   - Available methods: {len(capturer.methods)}")
        
        for method_name, _ in capturer.methods:
            logger.info(f"      - {method_name}")
        
        # Test capture
        result = capturer.capture_space(0)
        
        if result.success:
            logger.info(f"✅ Capture successful")
            logger.info(f"   - Method used: {result.method}")
            logger.info(f"   - Image size: {result.image.size}")
            logger.info(f"   - Timestamp: {result.timestamp}")
            return True
        else:
            logger.error(f"❌ Capture failed: {result.error}")
            return False
    
    except Exception as e:
        logger.error(f"❌ reliable_screenshot_capture integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 70)
    logger.info("JARVIS WINDOWS VISION SYSTEM TEST SUITE")
    logger.info("Phase 7: Vision System Port - Verification")
    logger.info("=" * 70 + "\n")
    
    tests = [
        test_platform_detection,
        test_platform_capture_import,
        test_windows_vision_import,
        test_monitor_detection,
        test_screen_capture,
        test_fps_performance,
        test_screen_vision_integration,
        test_reliable_screenshot_capture,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} crashed: {e}")
            results.append((test.__name__, False))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info("=" * 70)
    logger.info(f"Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    logger.info("=" * 70)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Vision system ready for production.")
        return 0
    elif passed >= total * 0.7:
        logger.warning(f"\n⚠️  {total - passed} tests failed. Review errors above.")
        return 1
    else:
        logger.error(f"\n❌ Critical failures. Only {passed}/{total} tests passed.")
        return 2


if __name__ == '__main__':
    sys.exit(main())
