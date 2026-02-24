"""
macOS Platform Implementation
==============================

macOS-specific implementation of JARVIS platform abstraction layer.
Wraps existing macOS functionality to maintain compatibility.
"""

import asyncio
import subprocess
import io
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

import psutil

from .abstraction import PlatformInterface


class MacOSPlatform(PlatformInterface):
    """
    macOS platform implementation.
    
    Wraps existing JARVIS macOS functionality.
    Maintains backward compatibility with existing Swift/AppleScript code.
    """
    
    def __init__(self):
        super().__init__()
        
        if HAS_MSS:
            self.sct = mss.mss()
    
    def _run_applescript(self, script: str) -> Optional[str]:
        """Run AppleScript and return output."""
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return None
    
    def _run_command(self, cmd: List[str]) -> Optional[str]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
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
        """Get system idle time using ioreg."""
        output = self._run_command([
            "ioreg", "-c", "IOHIDSystem", "-d", "4", "-r", "HIDIdleTime"
        ])
        
        if output:
            try:
                # Parse ioreg output
                for line in output.split("\n"):
                    if "HIDIdleTime" in line:
                        # Extract the number
                        import re
                        match = re.search(r'= (\d+)', line)
                        if match:
                            nanoseconds = int(match.group(1))
                            return nanoseconds / 1_000_000_000.0
            except Exception:
                pass
        
        return 0.0
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get macOS system information."""
        return {
            "os_name": "macOS",
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
        """Get battery information."""
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
        """Get information about all visible windows using yabai or AppleScript."""
        # Try yabai first (if installed)
        output = self._run_command(["yabai", "-m", "query", "--windows"])
        if output:
            try:
                import json
                windows = json.loads(output)
                return [
                    {
                        "id": w.get("id"),
                        "title": w.get("title", ""),
                        "app_name": w.get("app", ""),
                        "x": w.get("frame", {}).get("x", 0),
                        "y": w.get("frame", {}).get("y", 0),
                        "width": w.get("frame", {}).get("w", 0),
                        "height": w.get("frame", {}).get("h", 0),
                        "is_focused": w.get("has-focus", False),
                    }
                    for w in windows
                ]
            except:
                pass
        
        # Fallback to AppleScript
        script = '''
        tell application "System Events"
            set windowList to {}
            repeat with proc in (every process whose background only is false)
                try
                    repeat with win in (every window of proc)
                        set end of windowList to {name of proc, name of win}
                    end repeat
                end try
            end repeat
            return windowList
        end tell
        '''
        
        output = self._run_applescript(script)
        if output:
            # Parse AppleScript output (limited info)
            windows = []
            # Note: AppleScript has limitations for window positions
            return windows
        
        return []
    
    async def get_active_window(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently focused window."""
        script = '''
        tell application "System Events"
            set frontApp to name of first application process whose frontmost is true
            try
                set frontWindow to name of front window of application process frontApp
                return frontApp & " - " & frontWindow
            on error
                return frontApp
            end try
        end tell
        '''
        
        output = self._run_applescript(script)
        if output:
            parts = output.split(" - ", 1)
            return {
                "id": None,
                "app_name": parts[0],
                "title": parts[1] if len(parts) > 1 else "",
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "is_focused": True,
            }
        
        return None
    
    async def focus_window(self, window_id: Any) -> bool:
        """Focus/activate a specific window."""
        # This would require more sophisticated AppleScript or yabai
        return False
    
    # ========================================================================
    # AUTOMATION (MOUSE & KEYBOARD)
    # ========================================================================
    
    async def click_at(self, x: int, y: int, button: str = "left") -> bool:
        """Click at specific screen coordinates using cliclick."""
        click_type = "c" if button == "left" else "rc"
        try:
            subprocess.run(
                ["cliclick", f"{click_type}:{x},{y}"],
                check=True,
                capture_output=True,
                timeout=2,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    async def double_click_at(self, x: int, y: int) -> bool:
        """Double-click at specific screen coordinates."""
        try:
            subprocess.run(
                ["cliclick", f"dc:{x},{y}"],
                check=True,
                capture_output=True,
                timeout=2,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    async def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text using cliclick or AppleScript."""
        # Escape special characters
        escaped_text = text.replace('"', '\\"')
        
        try:
            subprocess.run(
                ["cliclick", f"t:{escaped_text}"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            # Fallback to AppleScript keystroke
            script = f'tell application "System Events" to keystroke "{escaped_text}"'
            output = self._run_applescript(script)
            return output is not None
    
    async def press_key(self, key: str) -> bool:
        """Press a keyboard key."""
        # Map common keys
        key_map = {
            "enter": "return",
            "esc": "escape",
            "escape": "escape",
            "ctrl": "control",
        }
        
        mapped_key = key_map.get(key.lower(), key)
        script = f'tell application "System Events" to key code {mapped_key}'
        output = self._run_applescript(script)
        return output is not None
    
    async def hotkey(self, *keys: str) -> bool:
        """Press a combination of keys simultaneously."""
        # Build AppleScript keystroke with modifiers
        modifiers = []
        regular_key = None
        
        for key in keys:
            if key.lower() in ["ctrl", "control", "cmd", "command", "alt", "option", "shift"]:
                modifiers.append(key.lower())
            else:
                regular_key = key
        
        if not regular_key:
            return False
        
        modifier_str = ", ".join([f"{m} down" for m in modifiers])
        if modifier_str:
            script = f'tell application "System Events" to keystroke "{regular_key}" using {{{modifier_str}}}'
        else:
            script = f'tell application "System Events" to keystroke "{regular_key}"'
        
        output = self._run_applescript(script)
        return output is not None
    
    async def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse cursor position."""
        script = '''
        tell application "System Events"
            return (do shell script "echo $(printf '%s %s' $(osascript -e 'tell application \\"System Events\\" to get position of mouse'))")
        end tell
        '''
        
        output = self._run_applescript(script)
        if output:
            try:
                x, y = map(int, output.split())
                return (x, y)
            except:
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
            # Fallback to screencapture command
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                subprocess.run(
                    ["screencapture", "-x", tmp_path],
                    check=True,
                    capture_output=True,
                )
                
                with open(tmp_path, "rb") as f:
                    data = f.read()
                
                Path(tmp_path).unlink()
                return data
            except:
                return b""
        
        # Use mss
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
        """Play system notification sound."""
        try:
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                check=False,
                capture_output=True,
            )
            return True
        except FileNotFoundError:
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
        """Show macOS notification."""
        script = f'''
        display notification "{message}" with title "{title}"
        '''
        
        output = self._run_applescript(script)
        return output is not None
    
    # ========================================================================
    # FILE SYSTEM OPERATIONS
    # ========================================================================
    
    async def open_file(self, path: Path) -> bool:
        """Open file with default application."""
        try:
            subprocess.run(
                ["open", str(path)],
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
                ["open", url],
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
        """Get macOS platform capabilities."""
        return {
            "has_gui": True,
            "has_audio": HAS_SOUNDDEVICE,
            "has_notifications": True,
            "has_automation": True,  # AppleScript/cliclick
            "has_screen_capture": True,
            "has_battery": psutil.sensors_battery() is not None,
            "has_window_management": True,
        }
