"""
Linux Platform Implementation
==============================

Linux-specific implementation of JARVIS platform abstraction layer.
Supports both X11 and Wayland display servers.
"""

import asyncio
import io
import os
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    from pynput.keyboard import Controller as KeyboardController
    from pynput.mouse import Controller as MouseController, Button
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

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

import psutil

from .abstraction import PlatformInterface


class LinuxPlatform(PlatformInterface):
    """
    Linux platform implementation.
    
    Supports both X11 and Wayland display servers.
    
    Requires:
    - pynput for automation
    - mss for screen capture
    - sounddevice for audio
    - wmctrl, xdotool (system packages for X11)
    """
    
    def __init__(self):
        super().__init__()
        
        # Detect display server
        self.display_server = self._detect_display_server()
        
        if not HAS_PYNPUT:
            print("⚠️  Warning: pynput not installed. Automation will be limited.")
        if not HAS_MSS:
            print("⚠️  Warning: mss not installed. Screen capture unavailable.")
        
        # Initialize controllers
        if HAS_PYNPUT:
            self.keyboard = KeyboardController()
            self.mouse = MouseController()
        
        if HAS_MSS:
            self.sct = mss.mss()
    
    def _detect_display_server(self) -> str:
        """Detect if running X11 or Wayland."""
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type in ["x11", "wayland"]:
            return session_type
        
        # Fallback detection
        if os.environ.get("WAYLAND_DISPLAY"):
            return "wayland"
        elif os.environ.get("DISPLAY"):
            return "x11"
        
        return "unknown"
    
    def _run_command(self, cmd: List[str], timeout: float = 5.0) -> Optional[str]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None
    
    # ========================================================================
    # SYSTEM INFORMATION
    # ========================================================================
    
    async def get_idle_time(self) -> float:
        """Get system idle time."""
        if self.display_server == "x11":
            # Use xprintidle if available
            output = self._run_command(["xprintidle"])
            if output:
                try:
                    return int(output) / 1000.0
                except ValueError:
                    pass
        
        # Fallback: return 0 (can't detect on Wayland without compositor support)
        return 0.0
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get Linux system information."""
        return {
            "os_name": "Linux",
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
            "display_server": self.display_server,
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
        """Get information about all visible windows (X11 only)."""
        if self.display_server != "x11":
            return []  # Wayland doesn't expose window info for security
        
        windows = []
        output = self._run_command(["wmctrl", "-l", "-G"])
        
        if output:
            for line in output.split("\n"):
                if line:
                    parts = line.split(None, 7)
                    if len(parts) >= 8:
                        try:
                            windows.append({
                                "id": parts[0],
                                "desktop": int(parts[1]),
                                "x": int(parts[2]),
                                "y": int(parts[3]),
                                "width": int(parts[4]),
                                "height": int(parts[5]),
                                "app_name": parts[6],
                                "title": parts[7],
                                "is_focused": False,  # Would need additional detection
                            })
                        except ValueError:
                            continue
        
        return windows
    
    async def get_active_window(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently focused window."""
        if self.display_server != "x11":
            return None
        
        # Get active window ID
        active_id = self._run_command(["xdotool", "getactivewindow"])
        if not active_id:
            return None
        
        # Get window info
        windows = await self.get_window_info()
        for win in windows:
            if str(win["id"]) == active_id:
                win["is_focused"] = True
                return win
        
        return None
    
    async def focus_window(self, window_id: Any) -> bool:
        """Focus/activate a specific window by ID."""
        if self.display_server != "x11":
            return False
        
        try:
            subprocess.run(
                ["wmctrl", "-i", "-a", str(window_id)],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    # ========================================================================
    # AUTOMATION (MOUSE & KEYBOARD)
    # ========================================================================
    
    async def click_at(self, x: int, y: int, button: str = "left") -> bool:
        """Click at specific screen coordinates."""
        if not HAS_PYNPUT:
            return False
        
        try:
            self.mouse.position = (x, y)
            await asyncio.sleep(0.01)  # Small delay for position to register
            
            btn = Button.left if button == "left" else Button.right
            self.mouse.click(btn)
            return True
        except Exception as e:
            print(f"Error clicking at ({x}, {y}): {e}")
            return False
    
    async def double_click_at(self, x: int, y: int) -> bool:
        """Double-click at specific screen coordinates."""
        if not HAS_PYNPUT:
            return False
        
        try:
            self.mouse.position = (x, y)
            await asyncio.sleep(0.01)
            self.mouse.click(Button.left, 2)
            return True
        except Exception as e:
            print(f"Error double-clicking at ({x}, {y}): {e}")
            return False
    
    async def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text using keyboard automation."""
        if not HAS_PYNPUT:
            return False
        
        try:
            for char in text:
                self.keyboard.type(char)
                if interval > 0:
                    await asyncio.sleep(interval)
            return True
        except Exception as e:
            print(f"Error typing text: {e}")
            return False
    
    async def press_key(self, key: str) -> bool:
        """Press a keyboard key."""
        if not HAS_PYNPUT:
            return False
        
        try:
            from pynput.keyboard import Key
            
            # Map common key names
            key_map = {
                "enter": Key.enter,
                "tab": Key.tab,
                "space": Key.space,
                "backspace": Key.backspace,
                "delete": Key.delete,
                "esc": Key.esc,
                "escape": Key.esc,
                "ctrl": Key.ctrl,
                "alt": Key.alt,
                "shift": Key.shift,
                "cmd": Key.cmd,
                "meta": Key.cmd,
            }
            
            key_obj = key_map.get(key.lower(), key)
            self.keyboard.press(key_obj)
            self.keyboard.release(key_obj)
            return True
        except Exception as e:
            print(f"Error pressing key '{key}': {e}")
            return False
    
    async def hotkey(self, *keys: str) -> bool:
        """Press a combination of keys simultaneously."""
        if not HAS_PYNPUT:
            return False
        
        try:
            from pynput.keyboard import Key
            
            key_map = {
                "ctrl": Key.ctrl,
                "alt": Key.alt,
                "shift": Key.shift,
                "cmd": Key.cmd,
                "meta": Key.cmd,
            }
            
            # Press all keys
            key_objects = [key_map.get(k.lower(), k) for k in keys]
            for key in key_objects:
                self.keyboard.press(key)
            
            # Small delay
            await asyncio.sleep(0.01)
            
            # Release all keys
            for key in reversed(key_objects):
                self.keyboard.release(key)
            
            return True
        except Exception as e:
            print(f"Error pressing hotkey {keys}: {e}")
            return False
    
    async def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse cursor position."""
        if HAS_PYNPUT:
            try:
                x, y = self.mouse.position
                return (x, y)
            except Exception:
                pass
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
                bbox = {
                    "left": region[0],
                    "top": region[1],
                    "width": region[2],
                    "height": region[3],
                }
                screenshot = self.sct.grab(bbox)
            elif monitor is not None:
                screenshot = self.sct.grab(self.sct.monitors[monitor])
            else:
                screenshot = self.sct.grab(self.sct.monitors[1])
            
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            
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
                "is_primary": i == 1,
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
        """Play notification sound using system audio."""
        # Try paplay (PulseAudio)
        sound_paths = [
            "/usr/share/sounds/freedesktop/stereo/message.oga",
            "/usr/share/sounds/ubuntu/stereo/message.ogg",
            "/usr/share/sounds/gnome/default/alerts/drip.ogg",
        ]
        
        for sound_path in sound_paths:
            if Path(sound_path).exists():
                try:
                    subprocess.run(
                        ["paplay", sound_path],
                        check=False,
                        capture_output=True,
                        timeout=1,
                    )
                    return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
        
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
        """Show notification using notify-send."""
        try:
            cmd = ["notify-send", title, message]
            if duration:
                cmd.extend(["-t", str(duration * 1000)])
            if icon:
                cmd.extend(["-i", icon])
            
            subprocess.run(cmd, check=False, capture_output=True)
            return True
        except FileNotFoundError:
            # notify-send not available, print to console
            print(f"[NOTIFICATION] {title}: {message}")
            return False
    
    # ========================================================================
    # FILE SYSTEM OPERATIONS
    # ========================================================================
    
    async def open_file(self, path: Path) -> bool:
        """Open file with default application."""
        try:
            subprocess.run(
                ["xdg-open", str(path)],
                check=False,
                capture_output=True,
            )
            return True
        except FileNotFoundError:
            return False
    
    async def open_url(self, url: str) -> bool:
        """Open URL in default browser."""
        try:
            subprocess.run(
                ["xdg-open", url],
                check=False,
                capture_output=True,
            )
            return True
        except FileNotFoundError:
            return False
    
    # ========================================================================
    # CAPABILITIES
    # ========================================================================
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get Linux platform capabilities."""
        return {
            "has_gui": self.display_server != "unknown",
            "has_audio": HAS_SOUNDDEVICE,
            "has_notifications": self._run_command(["which", "notify-send"]) is not None,
            "has_automation": HAS_PYNPUT,
            "has_screen_capture": HAS_MSS,
            "has_battery": psutil.sensors_battery() is not None,
            "has_window_management": self.display_server == "x11",
        }

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to Linux clipboard."""
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
        except ImportError:
            pass
        except Exception:
            pass
        try:
            proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE
            )
            proc.communicate(text.encode())
            return True
        except Exception:
            pass
        try:
            proc = subprocess.Popen(
                ["xsel", "--clipboard", "--input"],
                stdin=subprocess.PIPE
            )
            proc.communicate(text.encode())
            return True
        except Exception:
            return False

    def paste_from_clipboard(self) -> str:
        """Paste text from Linux clipboard."""
        try:
            import pyperclip
            return pyperclip.paste()
        except ImportError:
            pass
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=3
            )
            return result.stdout
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["xsel", "--clipboard", "--output"],
                capture_output=True, text=True, timeout=3
            )
            return result.stdout
        except Exception:
            return ""
