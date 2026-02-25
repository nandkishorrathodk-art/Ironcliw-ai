"""
Windows Platform Implementation
================================

Windows-specific implementation of JARVIS platform abstraction layer.
Uses pywin32, pyautogui, mss, and Windows APIs.
"""

import asyncio
import io
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from ctypes import Structure, windll, c_uint, sizeof, byref

try:
    import win32gui
    import win32con
    import win32api
    import win32process
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False

try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable failsafe for automation
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    from win10toast import ToastNotifier
    HAS_WIN10TOAST = True
except ImportError:
    HAS_WIN10TOAST = False

import psutil

from .abstraction import PlatformInterface


class LASTINPUTINFO(Structure):
    """Windows LASTINPUTINFO structure for idle time detection."""
    _fields_ = [
        ('cbSize', c_uint),
        ('dwTime', c_uint),
    ]


class WindowsPlatform(PlatformInterface):
    """
    Windows platform implementation.
    
    Requires:
    - pywin32 for window management
    - pyautogui for automation
    - mss for screen capture
    - sounddevice for audio
    - win10toast for notifications
    """
    
    def __init__(self):
        super().__init__()
        
        if not HAS_PYWIN32:
            print("⚠️  Warning: pywin32 not installed. Window management will be limited.")
        if not HAS_PYAUTOGUI:
            print("⚠️  Warning: pyautogui not installed. Automation will be limited.")
        if not HAS_MSS:
            print("⚠️  Warning: mss not installed. Screen capture unavailable.")
        
        # Initialize components
        if HAS_MSS:
            self.sct = mss.mss()
        if HAS_WIN10TOAST:
            self.toaster = ToastNotifier()
    
    # ========================================================================
    # SYSTEM INFORMATION
    # ========================================================================
    
    async def get_idle_time(self) -> float:
        """Get system idle time using GetLastInputInfo API."""
        try:
            lastInputInfo = LASTINPUTINFO()
            lastInputInfo.cbSize = sizeof(lastInputInfo)
            windll.user32.GetLastInputInfo(byref(lastInputInfo))
            millis = win32api.GetTickCount() - lastInputInfo.dwTime
            return millis / 1000.0
        except Exception as e:
            print(f"Error getting idle time: {e}")
            return 0.0
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get Windows system information."""
        return {
            "os_name": "Windows",
            "os_version": self.os_version,
            "hostname": subprocess.run(
                ["hostname"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "free": psutil.disk_usage("/").free,
            },
        }
    
    async def get_battery_status(self) -> Optional[Dict[str, Any]]:
        """Get battery information if available."""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    "percent": battery.percent,
                    "plugged": battery.power_plugged,
                    "time_left": battery.secsleft if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None,
                }
        except Exception:
            pass
        return None
    
    # ========================================================================
    # WINDOW MANAGEMENT
    # ========================================================================
    
    async def get_window_info(self) -> List[Dict[str, Any]]:
        """Get information about all visible windows."""
        if not HAS_PYWIN32:
            return []
        
        windows = []
        
        def enum_callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:  # Only include windows with titles
                    try:
                        rect = win32gui.GetWindowRect(hwnd)
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        try:
                            process = psutil.Process(pid)
                            app_name = process.name()
                        except:
                            app_name = "Unknown"
                        
                        windows.append({
                            "id": hwnd,
                            "title": title,
                            "app_name": app_name,
                            "x": rect[0],
                            "y": rect[1],
                            "width": rect[2] - rect[0],
                            "height": rect[3] - rect[1],
                            "is_focused": hwnd == win32gui.GetForegroundWindow(),
                        })
                    except Exception:
                        pass
        
        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception as e:
            print(f"Error enumerating windows: {e}")
        
        return windows
    
    async def get_active_window(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently focused window."""
        if not HAS_PYWIN32:
            return None
        
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                title = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    process = psutil.Process(pid)
                    app_name = process.name()
                except:
                    app_name = "Unknown"
                
                return {
                    "id": hwnd,
                    "title": title,
                    "app_name": app_name,
                    "x": rect[0],
                    "y": rect[1],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1],
                    "is_focused": True,
                }
        except Exception as e:
            print(f"Error getting active window: {e}")
        
        return None
    
    async def focus_window(self, window_id: Any) -> bool:
        """Focus/activate a specific window by HWND."""
        if not HAS_PYWIN32:
            return False
        
        try:
            win32gui.SetForegroundWindow(window_id)
            return True
        except Exception as e:
            print(f"Error focusing window: {e}")
            return False
    
    # ========================================================================
    # AUTOMATION (MOUSE & KEYBOARD)
    # ========================================================================
    
    async def click_at(self, x: int, y: int, button: str = "left") -> bool:
        """Click at specific screen coordinates."""
        if not HAS_PYAUTOGUI:
            return False
        
        try:
            pyautogui.click(x, y, button=button)
            return True
        except Exception as e:
            print(f"Error clicking at ({x}, {y}): {e}")
            return False
    
    async def double_click_at(self, x: int, y: int) -> bool:
        """Double-click at specific screen coordinates."""
        if not HAS_PYAUTOGUI:
            return False
        
        try:
            pyautogui.doubleClick(x, y)
            return True
        except Exception as e:
            print(f"Error double-clicking at ({x}, {y}): {e}")
            return False
    
    async def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text using keyboard automation."""
        if not HAS_PYAUTOGUI:
            return False
        
        try:
            pyautogui.write(text, interval=interval)
            return True
        except Exception as e:
            print(f"Error typing text: {e}")
            return False
    
    async def press_key(self, key: str) -> bool:
        """Press a keyboard key."""
        if not HAS_PYAUTOGUI:
            return False
        
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            print(f"Error pressing key '{key}': {e}")
            return False
    
    async def hotkey(self, *keys: str) -> bool:
        """Press a combination of keys simultaneously."""
        if not HAS_PYAUTOGUI:
            return False
        
        try:
            pyautogui.hotkey(*keys)
            return True
        except Exception as e:
            print(f"Error pressing hotkey {keys}: {e}")
            return False
    
    async def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse cursor position."""
        if HAS_PYAUTOGUI:
            try:
                return pyautogui.position()
            except Exception:
                pass
        
        # Fallback to Windows API
        try:
            import win32api
            x, y = win32api.GetCursorPos()
            return (x, y)
        except:
            return (0, 0)
    
    # ========================================================================
    # SCREEN CAPTURE
    # ========================================================================
    
    async def capture_screen(
        self,
        monitor: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> bytes:
        """Capture screenshot as PNG bytes."""
        if not HAS_MSS:
            raise NotImplementedError("mss library not installed")
        
        try:
            from PIL import Image
            
            if region:
                # Capture specific region
                bbox = {
                    "left": region[0],
                    "top": region[1],
                    "width": region[2],
                    "height": region[3],
                }
                screenshot = self.sct.grab(bbox)
            elif monitor is not None:
                # Capture specific monitor (1-indexed)
                if monitor == 0:
                    # All monitors
                    monitor = 0
                screenshot = self.sct.grab(self.sct.monitors[monitor])
            else:
                # Capture primary monitor
                screenshot = self.sct.grab(self.sct.monitors[1])
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            
            # Encode as PNG
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return buffer.getvalue()
        
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return b""
    
    async def get_monitors(self) -> List[Dict[str, Any]]:
        """Get information about all connected monitors."""
        if not HAS_MSS:
            return []
        
        monitors = []
        for i, mon in enumerate(self.sct.monitors[1:], start=1):
            monitors.append({
                "id": i,
                "x": mon["left"],
                "y": mon["top"],
                "width": mon["width"],
                "height": mon["height"],
                "is_primary": i == 1,  # First monitor is usually primary
            })
        
        return monitors
    
    # ========================================================================
    # AUDIO
    # ========================================================================
    
    async def get_audio_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available audio input and output devices."""
        if not HAS_SOUNDDEVICE:
            return {"inputs": [], "outputs": []}
        
        try:
            devices = sd.query_devices()
            inputs = []
            outputs = []
            
            for i, dev in enumerate(devices):
                device_info = {
                    "id": i,
                    "name": dev['name'],
                    "channels": dev.get('max_input_channels', 0) or dev.get('max_output_channels', 0),
                    "sample_rate": dev.get('default_samplerate', 44100),
                }
                
                if dev['max_input_channels'] > 0:
                    inputs.append(device_info)
                if dev['max_output_channels'] > 0:
                    outputs.append(device_info)
            
            return {
                "inputs": inputs,
                "outputs": outputs,
            }
        except Exception as e:
            print(f"Error getting audio devices: {e}")
            return {"inputs": [], "outputs": []}
    
    async def play_notification_sound(self) -> bool:
        """Play system notification sound."""
        if HAS_WINSOUND:
            try:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
                return True
            except Exception as e:
                print(f"Error playing notification sound: {e}")
        return False
    
    # ========================================================================
    # NOTIFICATIONS
    # ========================================================================
    
    async def show_notification(
        self,
        title: str,
        message: str,
        duration: int = 5,
        icon: Optional[str] = None,
    ) -> bool:
        """Show Windows 10/11 toast notification."""
        if not HAS_WIN10TOAST:
            print(f"[NOTIFICATION] {title}: {message}")
            return False
        
        try:
            # Run in thread to avoid blocking
            def show_toast():
                self.toaster.show_toast(
                    title,
                    message,
                    duration=duration,
                    icon_path=icon,
                    threaded=True,
                )
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, show_toast)
            return True
        except Exception as e:
            print(f"Error showing notification: {e}")
            return False
    
    # ========================================================================
    # FILE SYSTEM OPERATIONS
    # ========================================================================
    
    async def open_file(self, path: Path) -> bool:
        """Open file with default application."""
        try:
            import os
            os.startfile(str(path))
            return True
        except Exception as e:
            print(f"Error opening file: {e}")
            return False
    
    async def open_url(self, url: str) -> bool:
        """Open URL in default browser."""
        try:
            import webbrowser
            webbrowser.open(url)
            return True
        except Exception as e:
            print(f"Error opening URL: {e}")
            return False
    
    # ========================================================================
    # CAPABILITIES
    # ========================================================================
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get Windows platform capabilities."""
        return {
            "has_gui": True,
            "has_audio": HAS_SOUNDDEVICE,
            "has_notifications": HAS_WIN10TOAST,
            "has_automation": HAS_PYAUTOGUI,
            "has_screen_capture": HAS_MSS,
            "has_battery": psutil.sensors_battery() is not None,
            "has_window_management": HAS_PYWIN32,
        }

    def get_active_window_title(self) -> str:
        """Get the title of the currently focused window (sync)."""
        if not HAS_PYWIN32:
            try:
                import ctypes
                buf = ctypes.create_unicode_buffer(512)
                ctypes.windll.user32.GetWindowTextW(
                    ctypes.windll.user32.GetForegroundWindow(), buf, 512
                )
                return buf.value
            except Exception:
                return ""
        try:
            hwnd = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(hwnd)
        except Exception:
            return ""

    def set_volume(self, level: int) -> bool:
        """Set system master volume 0-100 (sync)."""
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol_ctl = cast(interface, POINTER(IAudioEndpointVolume))
            vol_ctl.SetMasterVolumeLevelScalar(max(0, min(100, level)) / 100.0, None)
            return True
        except Exception:
            pass
        try:
            import ctypes
            vol = int(max(0, min(100, level)) / 100 * 0xFFFF)
            ctypes.windll.winmm.waveOutSetVolume(None, (vol << 16) | vol)
            return True
        except Exception:
            return False

    def get_volume(self) -> int:
        """Get current system master volume 0-100 (sync)."""
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol_ctl = cast(interface, POINTER(IAudioEndpointVolume))
            return int(vol_ctl.GetMasterVolumeLevelScalar() * 100)
        except Exception:
            return 50

    def lock_screen(self) -> bool:
        """Lock the Windows workstation (sync)."""
        try:
            import ctypes
            ctypes.windll.user32.LockWorkStation()
            return True
        except Exception:
            return False

    def is_screen_locked(self) -> bool:
        """Detect if the Windows screen is locked (sync)."""
        try:
            for proc in psutil.process_iter(['name']):
                name = proc.info.get('name') or ''
                if name.lower() == 'logonui.exe':
                    return True
        except Exception:
            pass
        return False

    def prevent_sleep(self) -> bool:
        """Prevent system sleep via SetThreadExecutionState (sync)."""
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            return True
        except Exception:
            return False

    def allow_sleep(self) -> bool:
        """Re-enable system sleep (sync)."""
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            return True
        except Exception:
            return False

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to Windows clipboard."""
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
        except ImportError:
            pass
        except Exception:
            pass
        try:
            process = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-16le'))
            return True
        except Exception:
            return False

    def paste_from_clipboard(self) -> str:
        """Paste text from Windows clipboard."""
        try:
            import pyperclip
            return pyperclip.paste()
        except ImportError:
            pass
        except Exception:
            pass
        try:
            result = subprocess.run(
                ['powershell', 'Get-Clipboard'],
                capture_output=True, text=True
            )
            return result.stdout.strip()
        except Exception:
            return ""
