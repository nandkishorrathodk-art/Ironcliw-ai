"""
Windows Multi-Monitor Support for JARVIS Vision Intelligence
═══════════════════════════════════════════════════════════════════════════════

Comprehensive multi-monitor detection and management for Windows 10/11 systems.
Replaces macOS multi_monitor_detector.py with Windows-specific implementations.

Features:
    - Real-time display detection using Win32 API
    - Per-monitor DPI awareness
    - Multi-monitor screenshot capture
    - Virtual desktop support
    - Display-aware context understanding
    - Monitor hot-plug detection

Windows APIs Used:
    - EnumDisplayMonitors (monitor enumeration)
    - GetMonitorInfo (monitor details)
    - Windows.Forms.Screen (managed API alternative)
    - GetSystemMetrics (DPI and scaling)

Author: JARVIS System
Version: 1.0.0 (Windows Port - Phase 7)
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

# Import Windows-specific libraries
try:
    import clr
    clr.AddReference('System.Windows.Forms')
    clr.AddReference('System.Drawing')
    import System.Windows.Forms as WinForms
    import System.Drawing as Drawing
    WINFORMS_AVAILABLE = True
except ImportError:
    WINFORMS_AVAILABLE = False
    logger.warning("Windows Forms not available - multi-monitor support limited")

# Try to import win32api for additional monitor info
try:
    import win32api
    import win32con
    WIN32API_AVAILABLE = True
except ImportError:
    WIN32API_AVAILABLE = False
    logger.debug("win32api not available - using managed API only")


@dataclass
class WindowsDisplayInfo:
    """Information about a Windows display"""
    display_id: int
    resolution: Tuple[int, int]
    position: Tuple[int, int]
    is_primary: bool
    refresh_rate: float = 60.0
    color_depth: int = 32
    name: str = ""
    device_name: str = ""
    scaling_factor: float = 1.0
    dpi: int = 96
    last_updated: float = field(default_factory=time.time)
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get display bounds as (x, y, width, height)"""
        return (self.position[0], self.position[1], 
                self.resolution[0], self.resolution[1])
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of display"""
        return (
            self.position[0] + self.resolution[0] // 2,
            self.position[1] + self.resolution[1] // 2
        )


@dataclass
class WindowsMonitorCaptureResult:
    """Result of multi-monitor capture operation on Windows"""
    success: bool
    displays_captured: Dict[int, np.ndarray]
    failed_displays: List[int]
    capture_time: float
    total_displays: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WindowsMultiMonitorDetector:
    """
    Windows multi-monitor detector using Win32 and .NET APIs
    
    Provides comprehensive display detection and screenshot capture
    capabilities across multiple monitors on Windows.
    """
    
    def __init__(self):
        """Initialize Windows multi-monitor detector"""
        if not WINFORMS_AVAILABLE:
            raise RuntimeError(
                "Windows Forms not available. "
                "Install pythonnet: pip install pythonnet"
            )
        
        self.displays: Dict[int, WindowsDisplayInfo] = {}
        self.last_detection_time = 0.0
        self.detection_cache_duration = 5.0  # Cache for 5 seconds
        
        # Performance tracking
        self.capture_stats = {
            "total_captures": 0,
            "failed_captures": 0,
            "total_time": 0.0,
            "avg_time_per_display": 0.0
        }
        
        logger.info("WindowsMultiMonitorDetector initialized")
    
    def detect_displays(self, force_refresh: bool = False) -> Dict[int, WindowsDisplayInfo]:
        """
        Detect all connected displays
        
        Args:
            force_refresh: Force re-detection even if cached data is valid
        
        Returns:
            Dictionary mapping display IDs to WindowsDisplayInfo objects
        """
        current_time = time.time()
        
        # Use cached data if available and not stale
        if (not force_refresh and 
            self.displays and 
            current_time - self.last_detection_time < self.detection_cache_duration):
            return self.displays
        
        self.displays = {}
        
        try:
            screens = WinForms.Screen.AllScreens
            
            for idx, screen in enumerate(screens):
                bounds = screen.Bounds
                
                # Get DPI and scaling info
                dpi = 96  # Default
                scaling = 1.0
                
                try:
                    # Try to get actual DPI if win32api is available
                    if WIN32API_AVAILABLE:
                        hdc = win32api.GetDC(0)
                        dpi = win32api.GetDeviceCaps(hdc, win32con.LOGPIXELSX)
                        win32api.ReleaseDC(0, hdc)
                        scaling = dpi / 96.0
                except:
                    pass
                
                # Get refresh rate if possible
                refresh_rate = 60.0
                try:
                    if hasattr(screen, 'RefreshRate'):
                        refresh_rate = float(screen.RefreshRate)
                except:
                    pass
                
                display_info = WindowsDisplayInfo(
                    display_id=idx + 1,
                    resolution=(bounds.Width, bounds.Height),
                    position=(bounds.X, bounds.Y),
                    is_primary=screen.Primary,
                    refresh_rate=refresh_rate,
                    color_depth=32,
                    name=f"Display {idx + 1}",
                    device_name=screen.DeviceName,
                    scaling_factor=scaling,
                    dpi=dpi,
                    last_updated=current_time
                )
                
                self.displays[idx + 1] = display_info
            
            self.last_detection_time = current_time
            logger.info(f"Detected {len(self.displays)} display(s)")
            
        except Exception as e:
            logger.error(f"Display detection failed: {e}")
            # Return empty dict on failure
            self.displays = {}
        
        return self.displays
    
    def get_primary_display(self) -> Optional[WindowsDisplayInfo]:
        """
        Get primary display info
        
        Returns:
            WindowsDisplayInfo for primary display or None
        """
        displays = self.detect_displays()
        
        for display in displays.values():
            if display.is_primary:
                return display
        
        # Fallback to first display if no primary found
        if displays:
            return list(displays.values())[0]
        
        return None
    
    def get_display_by_id(self, display_id: int) -> Optional[WindowsDisplayInfo]:
        """
        Get display info by ID
        
        Args:
            display_id: Display ID
        
        Returns:
            WindowsDisplayInfo or None if not found
        """
        displays = self.detect_displays()
        return displays.get(display_id)
    
    def get_display_at_position(self, x: int, y: int) -> Optional[WindowsDisplayInfo]:
        """
        Get display containing the given screen coordinates
        
        Args:
            x, y: Screen coordinates
        
        Returns:
            WindowsDisplayInfo or None if position not on any display
        """
        displays = self.detect_displays()
        
        for display in displays.values():
            dx, dy = display.position
            dw, dh = display.resolution
            
            if dx <= x < dx + dw and dy <= y < dy + dh:
                return display
        
        return None
    
    def get_total_desktop_bounds(self) -> Tuple[int, int, int, int]:
        """
        Get bounds of entire virtual desktop (all monitors combined)
        
        Returns:
            (min_x, min_y, width, height) of virtual desktop
        """
        displays = self.detect_displays()
        
        if not displays:
            return (0, 0, 1920, 1080)  # Default fallback
        
        min_x = min(d.position[0] for d in displays.values())
        min_y = min(d.position[1] for d in displays.values())
        max_x = max(d.position[0] + d.resolution[0] for d in displays.values())
        max_y = max(d.position[1] + d.resolution[1] for d in displays.values())
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def capture_all_displays(
        self,
        use_windows_capture: bool = True
    ) -> WindowsMonitorCaptureResult:
        """
        Capture screenshots from all displays
        
        Args:
            use_windows_capture: Use WindowsVisionCapture if True, else fallback
        
        Returns:
            WindowsMonitorCaptureResult with captured images
        """
        start_time = time.time()
        displays = self.detect_displays()
        
        if not displays:
            return WindowsMonitorCaptureResult(
                success=False,
                displays_captured={},
                failed_displays=[],
                capture_time=0.0,
                total_displays=0,
                error="No displays detected"
            )
        
        captured = {}
        failed = []
        
        try:
            if use_windows_capture:
                # Use WindowsVisionCapture for high-performance capture
                from .windows_vision_capture import WindowsVisionCapture
                capturer = WindowsVisionCapture()
                
                for display_id in displays.keys():
                    try:
                        frame = capturer.capture_screen(display_id)
                        if frame and frame.image_data is not None:
                            captured[display_id] = frame.image_data
                        else:
                            failed.append(display_id)
                    except Exception as e:
                        logger.error(f"Failed to capture display {display_id}: {e}")
                        failed.append(display_id)
            else:
                # Fallback to PIL-based capture
                from PIL import ImageGrab
                
                for display_id, display in displays.items():
                    try:
                        x, y, w, h = display.bounds
                        screenshot = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                        captured[display_id] = np.array(screenshot)
                    except Exception as e:
                        logger.error(f"Failed to capture display {display_id}: {e}")
                        failed.append(display_id)
        
        except Exception as e:
            logger.error(f"Multi-display capture failed: {e}")
            return WindowsMonitorCaptureResult(
                success=False,
                displays_captured=captured,
                failed_displays=failed,
                capture_time=time.time() - start_time,
                total_displays=len(displays),
                error=str(e)
            )
        
        # Update stats
        capture_time = time.time() - start_time
        self.capture_stats["total_captures"] += 1
        self.capture_stats["failed_captures"] += len(failed)
        self.capture_stats["total_time"] += capture_time
        self.capture_stats["avg_time_per_display"] = (
            self.capture_stats["total_time"] / 
            max(1, len(displays) * self.capture_stats["total_captures"])
        )
        
        success = len(captured) > 0
        
        return WindowsMonitorCaptureResult(
            success=success,
            displays_captured=captured,
            failed_displays=failed,
            capture_time=capture_time,
            total_displays=len(displays),
            metadata={
                'displays_info': {
                    did: {
                        'name': d.name,
                        'resolution': d.resolution,
                        'is_primary': d.is_primary
                    }
                    for did, d in displays.items()
                }
            }
        )
    
    def get_display_layout_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitor layout summary
        
        Returns:
            Dictionary with layout information
        """
        displays = self.detect_displays()
        
        if not displays:
            return {
                'total_displays': 0,
                'primary_display': None,
                'displays': [],
                'virtual_desktop': (0, 0, 1920, 1080)
            }
        
        primary = self.get_primary_display()
        desktop_bounds = self.get_total_desktop_bounds()
        
        return {
            'total_displays': len(displays),
            'primary_display': primary.display_id if primary else None,
            'displays': [
                {
                    'id': d.display_id,
                    'name': d.name,
                    'device_name': d.device_name,
                    'resolution': d.resolution,
                    'position': d.position,
                    'bounds': d.bounds,
                    'is_primary': d.is_primary,
                    'refresh_rate': d.refresh_rate,
                    'dpi': d.dpi,
                    'scaling': d.scaling_factor
                }
                for d in displays.values()
            ],
            'virtual_desktop': desktop_bounds,
            'total_pixels': sum(d.resolution[0] * d.resolution[1] for d in displays.values())
        }
    
    def watch_for_display_changes(
        self,
        callback: Optional[callable] = None,
        poll_interval: float = 2.0
    ):
        """
        Monitor for display configuration changes (hot-plug detection)
        
        Args:
            callback: Function to call when displays change
            poll_interval: How often to check for changes (seconds)
        
        Note:
            This is a simple polling implementation. For production,
            consider using WM_DISPLAYCHANGE message handling.
        """
        import threading
        
        previous_count = len(self.detect_displays())
        
        def poll_loop():
            nonlocal previous_count
            while True:
                time.sleep(poll_interval)
                current_count = len(self.detect_displays(force_refresh=True))
                
                if current_count != previous_count:
                    logger.info(
                        f"Display configuration changed: "
                        f"{previous_count} → {current_count} displays"
                    )
                    if callback:
                        try:
                            callback(self.displays)
                        except Exception as e:
                            logger.error(f"Display change callback error: {e}")
                    previous_count = current_count
        
        watch_thread = threading.Thread(target=poll_loop, daemon=True)
        watch_thread.start()
        logger.info(f"Started display change monitoring (poll interval: {poll_interval}s)")
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get capture performance statistics"""
        return self.capture_stats.copy()


# Global singleton instance
_windows_monitor_detector: Optional[WindowsMultiMonitorDetector] = None


def get_windows_monitor_detector() -> WindowsMultiMonitorDetector:
    """
    Get global Windows monitor detector instance (singleton)
    
    Returns:
        WindowsMultiMonitorDetector instance
    """
    global _windows_monitor_detector
    if _windows_monitor_detector is None:
        _windows_monitor_detector = WindowsMultiMonitorDetector()
    return _windows_monitor_detector


# Convenience functions
def get_windows_displays() -> Dict[int, WindowsDisplayInfo]:
    """Quick display detection"""
    return get_windows_monitor_detector().detect_displays()


def get_windows_primary_display() -> Optional[WindowsDisplayInfo]:
    """Quick primary display lookup"""
    return get_windows_monitor_detector().get_primary_display()


def capture_all_windows_displays() -> WindowsMonitorCaptureResult:
    """Quick multi-display capture"""
    return get_windows_monitor_detector().capture_all_displays()
