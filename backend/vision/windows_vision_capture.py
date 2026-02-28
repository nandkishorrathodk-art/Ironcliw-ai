"""
Ironcliw Windows Vision Capture - Enhanced Implementation
═══════════════════════════════════════════════════════════════════════════════

Advanced Windows screen capture implementation with multi-monitor support,
FPS-controlled continuous capture, and integration with YOLO and Claude Vision.

Features:
    - Windows.Graphics.Capture API integration via C# DLL
    - Multi-monitor detection and per-monitor capture
    - Continuous capture with precise FPS control
    - Thread-safe operation
    - Memory-efficient buffering
    - Monitor layout detection
    - Region capture support
    - Integration-ready for YOLO object detection

Architecture:
    Python (this file) ← pythonnet → C# ScreenCapture.dll ← Windows APIs

Performance Targets:
    - >15 FPS for single monitor capture
    - >10 FPS for multi-monitor capture
    - <100ms latency per frame
    - <500MB memory footprint

Author: Ironcliw System
Version: 1.0.0 (Windows Port - Phase 7)
"""
from __future__ import annotations

import os
import time
import threading
import logging
from typing import List, Optional, Callable, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import io

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Import pythonnet for C# interop
try:
    import clr
    PYTHONNET_AVAILABLE = True
except ImportError:
    PYTHONNET_AVAILABLE = False
    logger.error(
        "pythonnet (clr) not available. Install with: pip install pythonnet"
    )


@dataclass
class WindowsScreenFrame:
    """Windows-specific screen capture frame"""
    image_data: np.ndarray  # RGB numpy array
    width: int
    height: int
    timestamp: float
    monitor_id: int
    format: str = 'rgb'  # 'rgb', 'bgr', 'rgba'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WindowsMonitorInfo:
    """Windows monitor information"""
    monitor_id: int
    width: int
    height: int
    x: int
    y: int
    is_primary: bool
    name: str = ""
    device_name: str = ""
    refresh_rate: float = 60.0


class WindowsVisionCapture:
    """
    Enhanced Windows screen capture implementation
    
    Provides high-performance screen capture using Windows.Graphics.Capture API
    via C# DLL with pythonnet integration.
    """
    
    def __init__(self):
        """Initialize Windows vision capture system"""
        if not PYTHONNET_AVAILABLE:
            raise RuntimeError(
                "pythonnet is required for Windows vision capture. "
                "Install with: pip install pythonnet"
            )
        
        self._capturer = None
        self._system_control = None
        self._is_capturing = False
        self._capture_thread = None
        self._capture_callback = None
        self._target_fps = 15
        self._current_monitor = 0
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._last_fps_check = time.time()
        self._actual_fps = 0.0
        
        self._load_native_dlls()
        logger.info("WindowsVisionCapture initialized successfully")
    
    def _load_native_dlls(self):
        """Load C# ScreenCapture and SystemControl DLLs"""
        dll_path = os.environ.get(
            'WINDOWS_NATIVE_DLL_PATH',
            str(Path(__file__).parent.parent / 'windows_native' / 'bin' / 'Release')
        )
        
        dll_base = Path(dll_path)
        screen_capture_dll = dll_base / 'ScreenCapture.dll'
        system_control_dll = dll_base / 'SystemControl.dll'
        
        # Check if DLLs exist
        if not screen_capture_dll.exists():
            raise FileNotFoundError(
                f"ScreenCapture.dll not found at: {screen_capture_dll}\n"
                f"Please build the C# project:\n"
                f"  cd backend\\windows_native\n"
                f"  .\\build.ps1\n"
                f"Or set WINDOWS_NATIVE_DLL_PATH environment variable."
            )
        
        try:
            # Load ScreenCapture.dll
            clr.AddReference(str(screen_capture_dll.resolve()))
            from JarvisWindowsNative.ScreenCapture import ScreenCaptureEngine
            self._capturer = ScreenCaptureEngine()
            logger.info(f"✅ Loaded ScreenCapture.dll from {dll_path}")
            
            # Load SystemControl.dll for monitor enumeration
            if system_control_dll.exists():
                clr.AddReference(str(system_control_dll.resolve()))
                from JarvisWindowsNative.SystemControl import SystemControlEngine
                self._system_control = SystemControlEngine()
                logger.info(f"✅ Loaded SystemControl.dll from {dll_path}")
            else:
                logger.warning("SystemControl.dll not found - monitor detection may be limited")
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to load C# DLLs: {e}\n"
                f"Ensure .NET Runtime is installed and DLLs are properly built.\n"
                f"Path: {dll_path}"
            ) from e
    
    def capture_screen(self, monitor_id: int = 0) -> Optional[WindowsScreenFrame]:
        """
        Capture screenshot from specified monitor
        
        Args:
            monitor_id: Monitor index (0 = primary/all, 1+ = specific monitor)
        
        Returns:
            WindowsScreenFrame or None if capture failed
        """
        try:
            # Capture using C# DLL
            if monitor_id == 0:
                image_bytes = self._capturer.CaptureScreen()
            else:
                try:
                    image_bytes = self._capturer.CaptureMonitor(monitor_id)
                except:
                    logger.warning(f"Monitor {monitor_id} not available, falling back to primary")
                    image_bytes = self._capturer.CaptureScreen()
            
            if image_bytes is None or len(image_bytes) == 0:
                logger.error("Capture returned empty data")
                return None
            
            # Convert C# byte array to PIL Image
            image_stream = io.BytesIO(bytes(image_bytes))
            pil_image = Image.open(image_stream)
            
            # Convert to numpy array (RGB)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image_array = np.array(pil_image)
            
            return WindowsScreenFrame(
                image_data=image_array,
                width=pil_image.width,
                height=pil_image.height,
                timestamp=time.time(),
                monitor_id=monitor_id,
                format='rgb',
                metadata={'capture_method': 'windows_graphics_capture'}
            )
        
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[WindowsScreenFrame]:
        """
        Capture specific screen region
        
        Args:
            x, y: Top-left corner coordinates
            width, height: Region dimensions
        
        Returns:
            WindowsScreenFrame or None if capture failed
        """
        try:
            image_bytes = self._capturer.CaptureRegion(x, y, width, height)
            
            if image_bytes is None or len(image_bytes) == 0:
                logger.error("Region capture returned empty data")
                return None
            
            # Convert to PIL Image
            image_stream = io.BytesIO(bytes(image_bytes))
            pil_image = Image.open(image_stream)
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            image_array = np.array(pil_image)
            
            return WindowsScreenFrame(
                image_data=image_array,
                width=width,
                height=height,
                timestamp=time.time(),
                monitor_id=0,
                format='rgb',
                metadata={
                    'capture_method': 'region_capture',
                    'region': (x, y, width, height)
                }
            )
        
        except Exception as e:
            logger.error(f"Region capture failed: {e}")
            return None
    
    def get_monitors(self) -> List[WindowsMonitorInfo]:
        """
        Get list of all connected monitors
        
        Returns:
            List of WindowsMonitorInfo objects
        """
        monitors = []
        
        try:
            # Use Windows Forms API to enumerate screens
            import System.Windows.Forms as WinForms
            screens = WinForms.Screen.AllScreens
            
            for idx, screen in enumerate(screens):
                bounds = screen.Bounds
                monitors.append(WindowsMonitorInfo(
                    monitor_id=idx + 1,
                    width=bounds.Width,
                    height=bounds.Height,
                    x=bounds.X,
                    y=bounds.Y,
                    is_primary=screen.Primary,
                    name=screen.DeviceName,
                    device_name=screen.DeviceName,
                    refresh_rate=60.0  # Default, could query actual refresh rate
                ))
            
            logger.info(f"Detected {len(monitors)} monitor(s)")
            return monitors
        
        except Exception as e:
            logger.error(f"Monitor enumeration failed: {e}")
            # Return default primary monitor
            return [WindowsMonitorInfo(
                monitor_id=0,
                width=1920,
                height=1080,
                x=0,
                y=0,
                is_primary=True,
                name="Primary Display",
                device_name="\\\\.\\DISPLAY1"
            )]
    
    def get_monitor_layout(self) -> Dict[str, Any]:
        """
        Get comprehensive monitor layout information
        
        Returns:
            Dictionary with monitor layout details
        """
        monitors = self.get_monitors()
        
        if not monitors:
            return {'total_monitors': 0, 'monitors': []}
        
        # Calculate total desktop bounds
        min_x = min(m.x for m in monitors)
        min_y = min(m.y for m in monitors)
        max_x = max(m.x + m.width for m in monitors)
        max_y = max(m.y + m.height for m in monitors)
        
        return {
            'total_monitors': len(monitors),
            'monitors': [
                {
                    'id': m.monitor_id,
                    'bounds': (m.x, m.y, m.width, m.height),
                    'is_primary': m.is_primary,
                    'name': m.name,
                    'device_name': m.device_name
                }
                for m in monitors
            ],
            'desktop_bounds': (min_x, min_y, max_x - min_x, max_y - min_y),
            'primary_monitor': next((m.monitor_id for m in monitors if m.is_primary), 0)
        }
    
    def start_continuous_capture(
        self,
        callback: Callable[[WindowsScreenFrame], None],
        fps: int = 15,
        monitor_id: int = 0
    ) -> bool:
        """
        Start continuous screen capture at specified FPS
        
        Args:
            callback: Function to call with each captured frame
            fps: Target frames per second (default: 15)
            monitor_id: Monitor to capture (0 = primary/all)
        
        Returns:
            True if started successfully
        """
        if self._is_capturing:
            logger.warning("Continuous capture already running")
            return False
        
        if callback is None:
            logger.error("Callback function is required")
            return False
        
        self._capture_callback = callback
        self._target_fps = fps
        self._current_monitor = monitor_id
        self._stop_event.clear()
        self._frame_count = 0
        self._last_fps_check = time.time()
        
        # Start capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="WindowsVisionCaptureThread"
        )
        self._is_capturing = True
        self._capture_thread.start()
        
        logger.info(f"Started continuous capture at {fps} FPS on monitor {monitor_id}")
        return True
    
    def stop_continuous_capture(self) -> bool:
        """
        Stop continuous screen capture
        
        Returns:
            True if stopped successfully
        """
        if not self._is_capturing:
            logger.warning("Continuous capture not running")
            return False
        
        logger.info("Stopping continuous capture...")
        self._is_capturing = False
        self._stop_event.set()
        
        # Wait for thread to finish (with timeout)
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                logger.warning("Capture thread did not stop gracefully")
        
        self._capture_callback = None
        self._capture_thread = None
        
        logger.info(f"Stopped continuous capture (final FPS: {self._actual_fps:.1f})")
        return True
    
    def _capture_loop(self):
        """Main capture loop running in background thread"""
        frame_interval = 1.0 / self._target_fps
        next_frame_time = time.time()
        
        logger.info(f"Capture loop started (target: {self._target_fps} FPS)")
        
        while not self._stop_event.is_set():
            try:
                # Wait until next frame time
                current_time = time.time()
                if current_time < next_frame_time:
                    time.sleep(next_frame_time - current_time)
                
                # Capture frame
                frame = self.capture_screen(self._current_monitor)
                
                if frame is not None:
                    # Call user callback
                    try:
                        self._capture_callback(frame)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                    
                    # Update FPS stats
                    self._frame_count += 1
                    if self._frame_count % 30 == 0:  # Update every 30 frames
                        elapsed = time.time() - self._last_fps_check
                        self._actual_fps = 30.0 / elapsed
                        self._last_fps_check = time.time()
                        logger.debug(f"Actual FPS: {self._actual_fps:.1f} (target: {self._target_fps})")
                
                # Schedule next frame
                next_frame_time += frame_interval
                
                # Prevent drift accumulation
                if next_frame_time < time.time() - 1.0:
                    next_frame_time = time.time()
            
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.1)  # Back off on error
        
        logger.info("Capture loop stopped")
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """
        Get capture performance statistics
        
        Returns:
            Dictionary with capture stats
        """
        return {
            'is_capturing': self._is_capturing,
            'target_fps': self._target_fps,
            'actual_fps': self._actual_fps,
            'frame_count': self._frame_count,
            'current_monitor': self._current_monitor,
            'monitors_available': len(self.get_monitors())
        }
    
    @property
    def is_capturing(self) -> bool:
        """Check if continuous capture is running"""
        return self._is_capturing
    
    @property
    def actual_fps(self) -> float:
        """Get actual capture FPS"""
        return self._actual_fps


# Convenience functions
def capture_screen_windows(monitor_id: int = 0) -> Optional[WindowsScreenFrame]:
    """
    Quick capture from Windows screen
    
    Args:
        monitor_id: Monitor index
    
    Returns:
        WindowsScreenFrame or None
    """
    try:
        capturer = WindowsVisionCapture()
        return capturer.capture_screen(monitor_id)
    except Exception as e:
        logger.error(f"Quick capture failed: {e}")
        return None


def get_windows_monitors() -> List[WindowsMonitorInfo]:
    """
    Quick monitor enumeration
    
    Returns:
        List of WindowsMonitorInfo
    """
    try:
        capturer = WindowsVisionCapture()
        return capturer.get_monitors()
    except Exception as e:
        logger.error(f"Monitor enumeration failed: {e}")
        return []
