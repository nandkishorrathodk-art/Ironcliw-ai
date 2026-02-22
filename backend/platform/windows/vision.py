"""
JARVIS Windows Vision Capture Implementation
═══════════════════════════════════════════════════════════════════════════════

Windows implementation of screen capture using C# ScreenCapture DLL with GDI+.

Features:
    - Full screen capture
    - Region capture
    - Multi-monitor support
    - Continuous capture with FPS control
    - Monitor layout detection

C# DLL Methods Used:
    - ScreenCaptureEngine.CaptureScreen()
    - ScreenCaptureEngine.CaptureRegion(x, y, width, height)
    - ScreenCaptureEngine.CaptureWindow(handle)
    - ScreenCaptureEngine.GetMonitorLayout()
    - ScreenCaptureEngine.CaptureMonitor(monitorId)

Author: JARVIS System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

import os
import time
import threading
from typing import List, Optional, Callable, Dict, Any
from pathlib import Path

try:
    import clr
except ImportError:
    raise ImportError(
        "pythonnet (clr) is not installed. Install with: pip install pythonnet"
    )

from ..base import (
    BaseVisionCapture,
    ScreenCaptureFrame,
)


class WindowsVisionCapture(BaseVisionCapture):
    """Windows implementation of screen capture using C# DLL"""
    
    def __init__(self):
        """Initialize Windows vision capture with C# DLL"""
        self._capturer = None
        self._is_capturing = False
        self._capture_thread = None
        self._capture_callback = None
        self._capture_fps = 15
        self._capture_monitor = 0
        self._load_native_dll()
    
    def _load_native_dll(self):
        """Load C# ScreenCapture DLL"""
        dll_path = os.environ.get(
            'WINDOWS_NATIVE_DLL_PATH',
            str(Path(__file__).parent.parent.parent / 'windows_native' / 'bin' / 'Release')
        )
        
        dll_file = Path(dll_path) / 'ScreenCapture.dll'
        
        if not dll_file.exists():
            raise FileNotFoundError(
                f"ScreenCapture.dll not found at: {dll_file}\n"
                f"Please build the C# project first:\n"
                f"  cd backend/windows_native\n"
                f"  .\\build.ps1"
            )
        
        try:
            clr.AddReference(str(dll_file.resolve()))
            from JarvisWindowsNative.ScreenCapture import ScreenCaptureEngine
            self._capturer = ScreenCaptureEngine()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ScreenCapture.dll: {e}\n"
                f"Make sure .NET Runtime is installed and DLL is built."
            ) from e
    
    def capture_screen(self, monitor_id: int = 0) -> Optional[ScreenCaptureFrame]:
        """Capture screenshot from specified monitor"""
        try:
            if monitor_id == 0:
                image_data = self._capturer.CaptureScreen()
            else:
                try:
                    image_data = self._capturer.CaptureMonitor(monitor_id)
                except:
                    image_data = self._capturer.CaptureScreen()
            
            if image_data is None or len(image_data) == 0:
                return None
            
            import System.Windows.Forms as WinForms
            screens = WinForms.Screen.AllScreens
            
            if monitor_id < len(screens):
                screen = screens[monitor_id]
                width = screen.Bounds.Width
                height = screen.Bounds.Height
            else:
                width = 1920
                height = 1080
            
            return ScreenCaptureFrame(
                image_data=bytes(image_data),
                width=width,
                height=height,
                timestamp=time.time(),
                monitor_id=monitor_id,
                format='png',
            )
        except Exception as e:
            print(f"Warning: Failed to capture screen: {e}")
            return None
    
    def capture_all_screens(self) -> List[ScreenCaptureFrame]:
        """Capture all monitors at once"""
        try:
            import System.Windows.Forms as WinForms
            screens = WinForms.Screen.AllScreens
            
            frames = []
            for idx in range(len(screens)):
                frame = self.capture_screen(monitor_id=idx)
                if frame:
                    frames.append(frame)
            
            return frames
        except Exception as e:
            print(f"Warning: Failed to capture all screens: {e}")
            primary = self.capture_screen(monitor_id=0)
            return [primary] if primary else []
    
    def start_continuous_capture(self, 
                                 fps: int = 15, 
                                 monitor_id: int = 0,
                                 callback: Optional[Callable] = None) -> bool:
        """Start continuous screen capture"""
        try:
            if self._is_capturing:
                self.stop_continuous_capture()
            
            self._capture_fps = max(1, min(fps, 60))
            self._capture_monitor = monitor_id
            self._capture_callback = callback
            self._is_capturing = True
            
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True
            )
            self._capture_thread.start()
            
            return True
        except Exception as e:
            print(f"Warning: Failed to start continuous capture: {e}")
            self._is_capturing = False
            return False
    
    def _capture_loop(self):
        """Background thread for continuous capture"""
        interval = 1.0 / self._capture_fps
        
        while self._is_capturing:
            try:
                start_time = time.time()
                
                frame = self.capture_screen(monitor_id=self._capture_monitor)
                
                if frame and self._capture_callback:
                    try:
                        self._capture_callback(frame)
                    except Exception as e:
                        print(f"Warning: Capture callback error: {e}")
                
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                print(f"Warning: Capture loop error: {e}")
                time.sleep(interval)
    
    def stop_continuous_capture(self) -> bool:
        """Stop continuous capture"""
        try:
            if not self._is_capturing:
                return True
            
            self._is_capturing = False
            
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
            
            self._capture_thread = None
            self._capture_callback = None
            
            return True
        except Exception as e:
            print(f"Warning: Failed to stop continuous capture: {e}")
            return False
    
    def is_capturing(self) -> bool:
        """Check if currently capturing"""
        return self._is_capturing
    
    def get_monitor_layout(self) -> List[Dict[str, Any]]:
        """Get layout information for all monitors"""
        try:
            import System.Windows.Forms as WinForms
            screens = WinForms.Screen.AllScreens
            
            monitors = []
            for idx, screen in enumerate(screens):
                monitors.append({
                    'id': idx,
                    'name': str(screen.DeviceName),
                    'bounds': {
                        'x': screen.Bounds.X,
                        'y': screen.Bounds.Y,
                        'width': screen.Bounds.Width,
                        'height': screen.Bounds.Height,
                    },
                    'working_area': {
                        'x': screen.WorkingArea.X,
                        'y': screen.WorkingArea.Y,
                        'width': screen.WorkingArea.Width,
                        'height': screen.WorkingArea.Height,
                    },
                    'is_primary': screen.Primary,
                    'bits_per_pixel': screen.BitsPerPixel,
                })
            
            return monitors
        except Exception as e:
            print(f"Warning: Failed to get monitor layout: {e}")
            return [{
                'id': 0,
                'name': 'Primary Display',
                'bounds': {'x': 0, 'y': 0, 'width': 1920, 'height': 1080},
                'working_area': {'x': 0, 'y': 0, 'width': 1920, 'height': 1080},
                'is_primary': True,
                'bits_per_pixel': 32,
            }]
