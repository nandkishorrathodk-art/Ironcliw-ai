"""
Unit tests for cross-platform screen capture implementation.

Tests the platform_capture module including:
- Platform auto-detection
- Windows capture (mss-based)
- Linux capture (X11/Wayland)
- macOS capture (wrapper)
- Base abstractions

Created: 2026-02-23
Purpose: Windows/Linux porting - Phase 3 (Screen Capture)
"""

import asyncio
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.platform_capture import (
    create_capture,
    get_available_capture_methods,
    is_capture_supported,
    CaptureConfig,
    CaptureFrame,
    CaptureMethod,
    CaptureQuality,
    CaptureStats,
    CaptureError,
    CaptureNotSupportedError,
)

from vision.platform_capture.base_capture import ScreenCaptureInterface
from core.platform_abstraction import PlatformDetector, SupportedPlatform


class TestCaptureDataClasses(unittest.TestCase):
    """Test data classes and configuration."""
    
    def test_capture_config_defaults(self):
        """Test CaptureConfig default values."""
        config = CaptureConfig()
        self.assertEqual(config.quality, CaptureQuality.HIGH)
        self.assertEqual(config.fps_target, 30)
        self.assertIsNone(config.method)
        self.assertTrue(config.capture_cursor)
        self.assertTrue(config.enable_monitoring)
    
    def test_capture_config_custom(self):
        """Test CaptureConfig with custom values."""
        config = CaptureConfig(
            method=CaptureMethod.MSS,
            quality=CaptureQuality.LOW,
            fps_target=60,
            display_id="2",
        )
        self.assertEqual(config.method, CaptureMethod.MSS)
        self.assertEqual(config.quality, CaptureQuality.LOW)
        self.assertEqual(config.fps_target, 60)
        self.assertEqual(config.display_id, "2")
    
    def test_capture_frame(self):
        """Test CaptureFrame creation and properties."""
        data = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = CaptureFrame(
            data=data,
            timestamp=datetime.now(),
            display_id="1",
            width=1920,
            height=1080,
            format="RGB",
        )
        
        self.assertEqual(frame.shape, (1080, 1920, 3))
        self.assertEqual(frame.width, 1920)
        self.assertEqual(frame.height, 1080)
        self.assertEqual(frame.format, "RGB")
        self.assertGreater(frame.size_bytes, 0)
    
    def test_capture_frame_to_rgb(self):
        """Test frame format conversion to RGB."""
        # Test RGB (no conversion)
        rgb_data = np.ones((100, 100, 3), dtype=np.uint8)
        frame = CaptureFrame(
            data=rgb_data,
            timestamp=datetime.now(),
            display_id="1",
            width=100,
            height=100,
            format="RGB",
        )
        converted = frame.to_rgb()
        self.assertTrue(np.array_equal(converted, rgb_data))
        
        # Test RGBA (strip alpha)
        rgba_data = np.ones((100, 100, 4), dtype=np.uint8)
        frame_rgba = CaptureFrame(
            data=rgba_data,
            timestamp=datetime.now(),
            display_id="1",
            width=100,
            height=100,
            format="RGBA",
        )
        converted_rgba = frame_rgba.to_rgb()
        self.assertEqual(converted_rgba.shape, (100, 100, 3))
    
    def test_capture_stats(self):
        """Test CaptureStats tracking."""
        stats = CaptureStats()
        
        # Initial state
        self.assertEqual(stats.frames_captured, 0)
        self.assertEqual(stats.frames_dropped, 0)
        self.assertEqual(stats.current_fps, 0.0)
        
        # Update FPS
        stats.update_fps(30.0)
        self.assertEqual(stats.current_fps, 30.0)
        self.assertEqual(stats.average_fps, 30.0)
        
        # Update again (should use moving average)
        stats.frames_captured = 1
        stats.update_fps(60.0)
        self.assertGreater(stats.average_fps, 30.0)
        self.assertLess(stats.average_fps, 60.0)


class TestPlatformDetection(unittest.TestCase):
    """Test platform detection and factory function."""
    
    def test_create_capture_auto_detect(self):
        """Test auto-detection creates correct capture type."""
        detector = PlatformDetector()
        platform = detector.get_platform()
        
        # Should not raise exception
        capture = create_capture()
        self.assertIsInstance(capture, ScreenCaptureInterface)
        
        # Check platform-specific type
        if platform == SupportedPlatform.WINDOWS:
            from vision.platform_capture.windows_capture import WindowsScreenCapture
            self.assertIsInstance(capture, WindowsScreenCapture)
        elif platform == SupportedPlatform.LINUX:
            from vision.platform_capture.linux_capture import LinuxScreenCapture
            self.assertIsInstance(capture, LinuxScreenCapture)
        elif platform == SupportedPlatform.MACOS:
            # macOS capture may not be available on non-macOS systems
            try:
                from vision.platform_capture.macos_capture import MacOSScreenCapture
                self.assertIsInstance(capture, MacOSScreenCapture)
            except ImportError:
                pass
    
    def test_is_capture_supported(self):
        """Test capture support detection."""
        supported = is_capture_supported()
        self.assertIsInstance(supported, bool)
        
        # On Windows/Linux, should be True (mss available)
        detector = PlatformDetector()
        if detector.is_windows() or detector.is_linux():
            self.assertTrue(supported)
    
    def test_get_available_methods(self):
        """Test getting available capture methods."""
        methods = get_available_capture_methods()
        self.assertIsInstance(methods, list)
        
        # Should have at least one method
        detector = PlatformDetector()
        if detector.is_windows() or detector.is_linux():
            self.assertGreater(len(methods), 0)


class TestWindowsCapture(unittest.TestCase):
    """Test Windows-specific capture implementation."""
    
    def setUp(self):
        """Skip tests if not on Windows."""
        detector = PlatformDetector()
        if not detector.is_windows():
            self.skipTest("Windows-only tests")
    
    def test_windows_capture_creation(self):
        """Test Windows capture instantiation."""
        from vision.platform_capture.windows_capture import WindowsScreenCapture
        
        config = CaptureConfig()
        capture = WindowsScreenCapture(config)
        
        self.assertIsInstance(capture, WindowsScreenCapture)
        self.assertFalse(capture.is_running)
    
    def test_windows_get_displays(self):
        """Test Windows display enumeration."""
        from vision.platform_capture.windows_capture import WindowsScreenCapture
        
        capture = WindowsScreenCapture()
        displays = capture.get_available_displays()
        
        self.assertIsInstance(displays, list)
        self.assertGreater(len(displays), 0)
        
        # Check display info structure
        for display in displays:
            self.assertIn("id", display)
            self.assertIn("name", display)
            self.assertIn("width", display)
            self.assertIn("height", display)
            self.assertIn("is_primary", display)
    
    def test_windows_capture_methods(self):
        """Test Windows available capture methods."""
        from vision.platform_capture.windows_capture import WindowsScreenCapture
        
        capture = WindowsScreenCapture()
        methods = capture.get_capture_methods()
        
        self.assertIsInstance(methods, list)
        self.assertIn(CaptureMethod.MSS, methods)
    
    @unittest.skipIf(
        not is_capture_supported(),
        "Screen capture not supported"
    )
    def test_windows_single_frame_capture(self):
        """Test Windows single frame capture."""
        from vision.platform_capture.windows_capture import WindowsScreenCapture
        
        async def run_test():
            capture = WindowsScreenCapture()
            frame = await capture.capture_single_frame()
            
            if frame:
                self.assertIsInstance(frame, CaptureFrame)
                self.assertGreater(frame.width, 0)
                self.assertGreater(frame.height, 0)
                self.assertIsInstance(frame.data, np.ndarray)
        
        asyncio.run(run_test())


class TestLinuxCapture(unittest.TestCase):
    """Test Linux-specific capture implementation."""
    
    def setUp(self):
        """Skip tests if not on Linux."""
        detector = PlatformDetector()
        if not detector.is_linux():
            self.skipTest("Linux-only tests")
    
    def test_linux_capture_creation(self):
        """Test Linux capture instantiation."""
        from vision.platform_capture.linux_capture import LinuxScreenCapture
        
        config = CaptureConfig()
        capture = LinuxScreenCapture(config)
        
        self.assertIsInstance(capture, LinuxScreenCapture)
        self.assertFalse(capture.is_running)
    
    def test_linux_display_server_detection(self):
        """Test X11/Wayland detection."""
        from vision.platform_capture.linux_capture import _detect_display_server
        
        server = _detect_display_server()
        self.assertIn(server, ["x11", "wayland", "unknown"])
    
    def test_linux_get_displays(self):
        """Test Linux display enumeration."""
        from vision.platform_capture.linux_capture import LinuxScreenCapture
        
        capture = LinuxScreenCapture()
        displays = capture.get_available_displays()
        
        self.assertIsInstance(displays, list)
        self.assertGreater(len(displays), 0)


class TestMacOSCapture(unittest.TestCase):
    """Test macOS-specific capture implementation."""
    
    def setUp(self):
        """Skip tests if not on macOS."""
        detector = PlatformDetector()
        if not detector.is_macos():
            self.skipTest("macOS-only tests")
    
    def test_macos_capture_creation(self):
        """Test macOS capture instantiation."""
        try:
            from vision.platform_capture.macos_capture import MacOSScreenCapture
            
            config = CaptureConfig()
            capture = MacOSScreenCapture(config)
            
            self.assertIsInstance(capture, MacOSScreenCapture)
            self.assertFalse(capture.is_running)
        except CaptureNotSupportedError:
            self.skipTest("macOS capture dependencies not available")


class TestCaptureInterface(unittest.TestCase):
    """Test base capture interface and common functionality."""
    
    def test_callback_registration(self):
        """Test frame callback registration."""
        capture = create_capture()
        
        callback_called = []
        
        def test_callback(frame):
            callback_called.append(frame)
        
        # Register callback
        capture.register_callback(test_callback)
        self.assertIn(test_callback, capture._callbacks)
        
        # Unregister callback
        capture.unregister_callback(test_callback)
        self.assertNotIn(test_callback, capture._callbacks)
    
    def test_stats_reset(self):
        """Test stats reset functionality."""
        capture = create_capture()
        
        # Modify stats
        capture.stats.frames_captured = 100
        capture.stats.frames_dropped = 5
        
        # Reset
        capture.reset_stats()
        
        # Check reset
        self.assertEqual(capture.stats.frames_captured, 0)
        self.assertEqual(capture.stats.frames_dropped, 0)
    
    def test_get_stats(self):
        """Test getting capture stats."""
        capture = create_capture()
        stats = capture.get_stats()
        
        self.assertIsInstance(stats, CaptureStats)


class TestCaptureLifecycle(unittest.TestCase):
    """Test capture start/stop lifecycle."""
    
    @unittest.skipIf(
        not is_capture_supported(),
        "Screen capture not supported"
    )
    def test_start_stop_cycle(self):
        """Test basic start/stop cycle."""
        async def run_test():
            capture = create_capture(CaptureConfig(fps_target=10))
            
            # Start
            success = await capture.start()
            if success:
                self.assertTrue(capture.is_running)
                
                # Wait briefly
                await asyncio.sleep(0.5)
                
                # Stop
                await capture.stop()
                self.assertFalse(capture.is_running)
        
        asyncio.run(run_test())
    
    @unittest.skipIf(
        not is_capture_supported(),
        "Screen capture not supported"
    )
    def test_frame_capture(self):
        """Test capturing frames."""
        async def run_test():
            capture = create_capture(CaptureConfig(fps_target=10))
            
            success = await capture.start()
            if success:
                # Get a frame
                frame = await capture.get_frame(timeout=5.0)
                
                if frame:
                    self.assertIsInstance(frame, CaptureFrame)
                    self.assertGreater(frame.width, 0)
                    self.assertGreater(frame.height, 0)
                    self.assertIsNotNone(frame.timestamp)
                    self.assertIsInstance(frame.data, np.ndarray)
                
                await capture.stop()
        
        asyncio.run(run_test())


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
