"""
Ironcliw Windows System Control Implementation
═══════════════════════════════════════════════════════════════════════════════

Windows implementation of system control operations using C# SystemControl DLL.

Features:
    - Window management (list, focus, minimize, maximize, close)
    - Volume control (get/set system volume)
    - System notifications
    - Display information
    - Command execution

C# DLL Methods Used:
    - SystemController.GetAllWindows()
    - SystemController.GetFocusedWindow()
    - SystemController.FocusWindow(handle)
    - SystemController.MinimizeWindow(handle)
    - SystemController.MaximizeWindow(handle)
    - SystemController.CloseWindow(handle)
    - SystemController.GetVolume()
    - SystemController.SetVolume(level)
    - SystemController.ShowNotification(title, message)

Author: Ironcliw System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

import os
import sys
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:
    import clr
except ImportError:
    raise ImportError(
        "pythonnet (clr) is not installed. Install with: pip install pythonnet"
    )

from ..base import (
    BaseSystemControl,
    WindowInfo,
)


class WindowsSystemControl(BaseSystemControl):
    """Windows implementation of system control using C# DLL"""
    
    def __init__(self):
        """Initialize Windows system control with C# DLL"""
        self._controller = None
        self._load_native_dll()
    
    def _load_native_dll(self):
        """Load C# SystemControl DLL"""
        dll_path = os.environ.get(
            'WINDOWS_NATIVE_DLL_PATH',
            str(Path(__file__).parent.parent.parent / 'windows_native' / 'bin' / 'Release')
        )
        
        dll_file = Path(dll_path) / 'SystemControl.dll'
        
        if not dll_file.exists():
            raise FileNotFoundError(
                f"SystemControl.dll not found at: {dll_file}\n"
                f"Please build the C# project first:\n"
                f"  cd backend/windows_native\n"
                f"  .\\build.ps1"
            )
        
        try:
            clr.AddReference(str(dll_file.resolve()))
            from JarvisWindowsNative.SystemControl import SystemController
            self._controller = SystemController()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SystemControl.dll: {e}\n"
                f"Make sure .NET Runtime is installed and DLL is built."
            ) from e
    
    def get_window_list(self) -> List[WindowInfo]:
        """Get list of all visible windows"""
        try:
            windows = self._controller.GetAllWindows()
            result = []
            
            for win in windows:
                result.append(WindowInfo(
                    window_id=int(win.Handle),
                    title=win.Title,
                    app_name=win.ProcessName,
                    bounds=(0, 0, 0, 0),
                    is_minimized=False,
                    is_maximized=False,
                    is_focused=False,
                    process_id=int(win.ProcessId),
                ))
            
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to get window list: {e}") from e
    
    def focus_window(self, window_id: int) -> bool:
        """Bring window to front and focus it"""
        try:
            import System
            handle = System.IntPtr(window_id)
            return bool(self._controller.FocusWindow(handle))
        except Exception as e:
            print(f"Warning: Failed to focus window {window_id}: {e}")
            return False
    
    def minimize_window(self, window_id: int) -> bool:
        """Minimize a window"""
        try:
            import System
            handle = System.IntPtr(window_id)
            return bool(self._controller.MinimizeWindow(handle))
        except Exception as e:
            print(f"Warning: Failed to minimize window {window_id}: {e}")
            return False
    
    def maximize_window(self, window_id: int) -> bool:
        """Maximize a window"""
        try:
            import System
            handle = System.IntPtr(window_id)
            return bool(self._controller.MaximizeWindow(handle))
        except Exception as e:
            print(f"Warning: Failed to maximize window {window_id}: {e}")
            return False
    
    def close_window(self, window_id: int) -> bool:
        """Close a window"""
        try:
            import System
            handle = System.IntPtr(window_id)
            return bool(self._controller.CloseWindowByHandle(handle))
        except Exception as e:
            print(f"Warning: Failed to close window {window_id}: {e}")
            return False
    
    def get_active_window(self) -> Optional[WindowInfo]:
        """Get currently focused window"""
        try:
            win = self._controller.GetFocusedWindow()
            if win is None:
                return None
            
            return WindowInfo(
                window_id=int(win.Handle),
                title=win.Title,
                app_name=win.ProcessName,
                bounds=(0, 0, 0, 0),
                is_minimized=False,
                is_maximized=False,
                is_focused=True,
                process_id=int(win.ProcessId),
            )
        except Exception as e:
            print(f"Warning: Failed to get active window: {e}")
            return None
    
    def set_volume(self, level: float) -> bool:
        """Set system volume (0.0-1.0)"""
        try:
            level = max(0.0, min(1.0, level))
            return bool(self._controller.SetVolume(float(level)))
        except Exception as e:
            print(f"Warning: Failed to set volume: {e}")
            return False
    
    def get_volume(self) -> float:
        """Get system volume (0.0-1.0)"""
        try:
            return float(self._controller.GetVolume())
        except Exception as e:
            print(f"Warning: Failed to get volume: {e}")
            return 0.5
    
    def show_notification(self, title: str, message: str, icon: Optional[str] = None) -> bool:
        """Show Windows notification"""
        try:
            import asyncio
            task = self._controller.ShowNotificationAsync(title, message)
            return bool(task.Result)
        except Exception as e:
            print(f"Warning: Failed to show notification: {e}")
            return False
    
    def execute_command(self, command: str, args: List[str] = None) -> Tuple[int, str, str]:
        """Execute shell command (returncode, stdout, stderr)"""
        try:
            cmd_args = [command]
            if args:
                cmd_args.extend(args)
            
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                shell=True,
                timeout=30
            )
            
            return (
                result.returncode,
                result.stdout,
                result.stderr
            )
        except subprocess.TimeoutExpired:
            return (-1, "", "Command timed out")
        except Exception as e:
            return (-1, "", str(e))
    
    def get_display_count(self) -> int:
        """Get number of connected displays"""
        try:
            import System.Windows.Forms as WinForms
            return len(WinForms.Screen.AllScreens)
        except Exception as e:
            print(f"Warning: Failed to get display count: {e}")
            return 1
    
    def get_display_info(self) -> List[Dict[str, Any]]:
        """Get information about all displays"""
        try:
            import System.Windows.Forms as WinForms
            displays = []
            
            for idx, screen in enumerate(WinForms.Screen.AllScreens):
                displays.append({
                    'id': idx,
                    'name': str(screen.DeviceName),
                    'bounds': {
                        'x': screen.Bounds.X,
                        'y': screen.Bounds.Y,
                        'width': screen.Bounds.Width,
                        'height': screen.Bounds.Height,
                    },
                    'is_primary': screen.Primary,
                    'working_area': {
                        'x': screen.WorkingArea.X,
                        'y': screen.WorkingArea.Y,
                        'width': screen.WorkingArea.Width,
                        'height': screen.WorkingArea.Height,
                    }
                })
            
            return displays
        except Exception as e:
            print(f"Warning: Failed to get display info: {e}")
            return [{
                'id': 0,
                'name': 'Primary Display',
                'bounds': {'x': 0, 'y': 0, 'width': 1920, 'height': 1080},
                'is_primary': True,
                'working_area': {'x': 0, 'y': 0, 'width': 1920, 'height': 1080},
            }]
