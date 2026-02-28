# IRONCLIW-AI / Ironcliw — AI Agent Master Blueprint
## Complete Instructions for Fresh Install → Windows Build → Linux Port
**Give this entire document to any AI agent to continue the work.**
**Repo:** `https://github.com/nandkishorrathodk-art/Ironcliw-ai`
**Original upstream:** `https://github.com/drussell23/Ironcliw`
**Owner:** Nandkishor Rathod (`nandkishorrathodk-art`)
**Current Phase:** 12 Complete — Whisper fixed, WebSocket loop fixed, ECAPA fast-fail 0.5s, startup 86s

---

## PART 1 — WHO YOU ARE AND WHAT THIS PROJECT IS

You are an AI coding agent continuing the Windows port of Ironcliw — an advanced personal AI assistant (like Iron Man's Ironcliw). Your job is to:

1. **Help user do a fresh install** on their Windows laptop
2. **Fix any remaining issues** on Windows
3. **Port everything to Linux** after Windows is stable
4. **Never break existing working features**
5. **Always commit and push to GitHub** after every fix

### Project Identity
- **Name in GitHub:** `Ironcliw-ai` (also called Ironcliw)
- **What it does:** Voice-activated AI assistant with screen vision, autonomous browser control, neural TTS, hybrid cloud routing
- **Tech stack:** Python 3.12 (FastAPI backend) + React 18 (frontend) + SQLite + ChromaDB
- **Voice:** `en-GB-RyanNeural` via `edge-tts` (Microsoft Neural — sounds like Iron Man's Ironcliw)
- **LLM:** Fireworks AI (primary) + Claude (fallback)
- **STT:** OpenAI Whisper (local, CPU mode)

### Golden Rule
> **Ironcliw runs entirely on Windows 10/11 without any macOS tools.**
> No `osascript`, no `screencapture`, no `caffeinate`, no `yabai`, no CoreGraphics.
> Use: `pyautogui`, `mss`, `pywin32`, `ctypes`, `edge-tts`, `plyer`

---

## PART 2 — CURRENT STATE (WHAT IS ALREADY DONE)

### ✅ Phase 1–11 Complete (DO NOT REDO THESE)

| What | Status | Key File |
|------|--------|----------|
| FastAPI backend | ✅ Working | `backend/main.py` |
| React frontend | ✅ Working | `frontend/src/` |
| Neural TTS `en-GB-RyanNeural` | ✅ Working | `backend/agi_os/realtime_voice_communicator.py` |
| Whisper STT (CPU) | ✅ Working | `backend/voice/hybrid_stt_router.py` |
| Screen capture (mss) | ✅ Working | `backend/vision/` |
| SQLite database | ✅ Working | `~/.jarvis/learning/jarvis_learning.db` |
| ChromaDB vector memory | ✅ Working | `backend/intelligence/` |
| Ghost Hands automation | ✅ Working | `backend/ghost_hands/background_actuator.py` |
| Platform Adapter (Windows) | ✅ Working | `backend/platform_adapter/windows_platform.py` |
| Notification bridge | ✅ Working | `backend/agi_os/notification_bridge.py` |
| Hardware control | ✅ Working | `backend/autonomy/hardware_control.py` |
| Screen lock detection | ✅ Working | `backend/context_intelligence/detectors/screen_lock_detector.py` |
| fcntl Windows shim | ✅ Working | `backend/core/resilience/atomic_file_ops.py` |
| ECAPA fast-fail (0.5s not 25s) | ✅ Working | `backend/voice_unlock/ml_engine_registry.py` |
| UNIQUE constraint fix | ✅ Working | `backend/intelligence/learning_database.py` |
| os.uname() fix | ✅ Working | `backend/core/infrastructure_orchestrator.py` |
| Keychain WinError 2 fix | ✅ Working | `start_system.py` |
| UTF-8 stdout/stderr | ✅ Working | `backend/main.py` |
| Whisper STT (llvmlite 0.44.0) | ✅ Working | `requirements.txt` |
| WebSocket no-restart-loop | ✅ Working | `start_system.py` |
| Owner profile cross-platform | ✅ Working | `backend/voice/speaker_verification_service.py` |
| 14/14 integration tests pass | ✅ Verified | `test_integration.py` |

### Current HEAD commit
```
32b10ccc  fix: ECAPA fast-fail on Windows avoid 25s timeout
```

### Startup time
- Session 12: **86 seconds** (was 171s before fixes)

### Working access points (when running)
- Frontend: http://localhost:3001/ (falls back from 3000 if stuck)
- Backend API: http://localhost:8010/docs
- Health: http://localhost:8010/health/ping
- WebSocket: ws://localhost:8010/ws

---

## PART 3 — STEP 1: FRESH INSTALL INSTRUCTIONS

### 3.1 Delete Old Ironcliw

```powershell
# Open PowerShell as Administrator

# Stop running Ironcliw first (Ctrl+C in its terminal OR:)
taskkill /F /IM python.exe 2>$null
taskkill /F /IM node.exe 2>$null

# Go to parent directory
cd C:\Users\nandk

# DELETE the old folder
Remove-Item -Recurse -Force "C:\Users\nandk\Ironcliw"

# Verify gone:
Test-Path "C:\Users\nandk\Ironcliw"
# Must print: False
```

### 3.2 Fresh Clone

```powershell
cd C:\Users\nandk
git clone https://github.com/nandkishorrathodk-art/Ironcliw-ai.git Ironcliw
cd Ironcliw

# Verify latest commit:
git log --oneline -5
# Should show: 0afb6259 at top
```

### 3.3 Install Python Dependencies

```powershell
# From C:\Users\nandk\Ironcliw

# Step 1 — Main requirements
pip install -r requirements.txt

# Step 2 — Windows-specific (MANDATORY)
pip install edge-tts mss pyautogui pywin32 pyttsx3 plyer

# Step 3 — Audio
pip install soundfile sounddevice openai-whisper numba

# Step 4 — Database & memory
pip install chromadb asyncpg aiohttp aiohttp-cors

# Step 5 — AI APIs
pip install anthropic fireworks-ai openai

# Step 6 — Optional but recommended
pip install pycaw comtypes psutil pyperclip

# Verify all critical ones:
python -c "import edge_tts; print('edge-tts OK')"
python -c "import mss; print('mss OK')"
python -c "import pyautogui; print('pyautogui OK')"
python -c "import whisper; print('whisper OK')"
python -c "import chromadb; print('chromadb OK')"
python -c "import anthropic; print('anthropic OK')"
```

### 3.4 Install Frontend

```powershell
cd frontend
npm install
cd ..
```

### 3.5 Configure .env

```powershell
copy .env.windows .env
notepad .env
```

**Fill in these values (everything else is already correct):**
```env
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
FIREWORKS_API_KEY=fw-YOUR_KEY_HERE
```

**Everything else in .env.windows is already correct for Windows. Do NOT change:**
```env
Ironcliw_AUTO_BYPASS_WINDOWS=true
Ironcliw_VOICE_BIOMETRIC_ENABLED=false
Ironcliw_ML_DEVICE=cpu
Ironcliw_DYNAMIC_PORTS=false
Ironcliw_SKIP_GCP=true
Ironcliw_SKIP_DOCKER=true
WHISPER_MODEL_SIZE=base
Ironcliw_PORT=8010
FRONTEND_PORT=3000
```

### 3.6 Run Integration Tests (Must Pass 14/14)

```powershell
python test_integration.py
```

Expected output:
```
PASSED: 14/14
```

If any test FAILS — see Part 7 (Troubleshooting) of this document.

### 3.7 Start Ironcliw

```powershell
python start_system.py
```

Open: http://localhost:3000 — Say "Hey Ironcliw" to test.

---

## PART 4 — WINDOWS REMAINING WORK (PHASE 12+)

These are features that still need to be implemented or improved on Windows.
**Do them in order. Commit after each one.**

---

### Phase 12A — Windows Notifications (PRIORITY: HIGH)

**File:** `backend/agi_os/notification_bridge.py`

**Problem:** `osascript display notification` is macOS only.

**Fix to implement:**
```python
import sys

async def _notify_platform(title: str, body: str, urgency: str) -> bool:
    if sys.platform == "win32":
        return await _notify_windows(title, body, urgency)
    return await _notify_osascript(title, body, urgency)

async def _notify_windows(title: str, body: str, urgency: str) -> bool:
    import asyncio
    loop = asyncio.get_event_loop()

    def _show():
        try:
            from plyer import notification
            notification.notify(
                title=title,
                message=body,
                app_name="Ironcliw",
                timeout=5,
            )
            return True
        except Exception:
            pass
        try:
            from win10toast import ToastNotifier
            ToastNotifier().show_toast(title, body, duration=5, threaded=True)
            return True
        except Exception:
            return False

    return await loop.run_in_executor(None, _show)
```

**Test:**
```powershell
python -c "
import asyncio, sys
sys.path.insert(0, 'backend')
from agi_os.notification_bridge import _notify_windows
result = asyncio.run(_notify_windows('Ironcliw Test', 'Windows notification works!', 'NORMAL'))
print('Notification:', 'OK' if result else 'FAILED - install: pip install plyer win10toast')
"
```

---

### Phase 12B — ECAPA Fast-Fail (PRIORITY: HIGH — saves 50s on startup)

**File:** `backend/voice_unlock/ml_engine_registry.py`

**Problem:** On Windows without speechbrain, ECAPA waits 25 seconds before failing.

**Find the `_load_ecapa` or `ensure_ecapa` function and add at the very top:**
```python
async def _load_ecapa_engine(self) -> Optional[Any]:
    # Windows fast-fail: skip 25s timeout if speechbrain not installed
    if sys.platform == "win32":
        try:
            import speechbrain  # noqa
        except ImportError:
            logger.info("[ECAPA] speechbrain not installed on Windows — skipping immediately")
            return None
    # ... rest of existing code below ...
```

**Test:** Startup should now reach "Ironcliw is ready!" in < 60 seconds (was 90+ seconds).

---

### Phase 13 — Volume Control (PRIORITY: MEDIUM)

**File:** `backend/autonomy/hardware_control.py`

**Problem:** Volume control uses `osascript set volume output volume X`

**Fix:**
```python
import sys

def _set_volume_platform(volume: int) -> bool:
    """Set system volume 0-100 cross-platform."""
    if sys.platform == "win32":
        return _set_volume_windows(volume)
    return _set_volume_macos(volume)

def _set_volume_windows(volume: int) -> bool:
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        vol_ctl = cast(interface, POINTER(IAudioEndpointVolume))
        vol_ctl.SetMasterVolumeLevelScalar(max(0, min(100, volume)) / 100.0, None)
        return True
    except Exception as e:
        logger.warning(f"pycaw volume failed: {e}")
    try:
        # Fallback: WinMM
        import ctypes
        vol = int(volume / 100 * 0xFFFF)
        ctypes.windll.winmm.waveOutSetVolume(None, (vol << 16) | vol)
        return True
    except Exception:
        return False

def _set_volume_macos(volume: int) -> bool:
    import subprocess
    script = f'set volume output volume {volume}'
    result = subprocess.run(['osascript', '-e', script], capture_output=True)
    return result.returncode == 0
```

**Install:** `pip install pycaw comtypes`

---

### Phase 14 — System Sleep Control (PRIORITY: MEDIUM)

**File:** `backend/autonomy/hardware_control.py`

**Replace `caffeinate` calls:**
```python
import sys
import ctypes

_sleep_handle = None

def prevent_sleep() -> bool:
    global _sleep_handle
    if sys.platform == "win32":
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        _sleep_handle = True
        return True
    else:
        import subprocess
        _sleep_handle = subprocess.Popen(['caffeinate', '-d', '-i', '-m'])
        return True

def allow_sleep() -> bool:
    global _sleep_handle
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        _sleep_handle = None
        return True
    else:
        if _sleep_handle:
            _sleep_handle.terminate()
            _sleep_handle = None
        return True
```

---

### Phase 15 — Clipboard (PRIORITY: LOW)

**Replace `pbcopy`/`pbpaste`:**
```python
import sys

def copy_to_clipboard(text: str) -> bool:
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import subprocess
            process = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-16le'))
            return True
        except Exception:
            return False
    return False

def paste_from_clipboard() -> str:
    try:
        import pyperclip
        return pyperclip.paste()
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import subprocess
            result = subprocess.run(['powershell', 'Get-Clipboard'],
                                    capture_output=True, text=True)
            return result.stdout.strip()
        except Exception:
            return ""
    return ""
```

---

### Phase 16 — System Tray Icon (PRIORITY: LOW)

**Create new file:** `backend/system_tray/tray_manager.py`

```python
"""Windows system tray icon for Ironcliw."""
import sys
import threading
import logging

logger = logging.getLogger(__name__)

def start_tray_icon(on_show=None, on_quit=None):
    """Start Ironcliw system tray icon (Windows/Linux via pystray)."""
    if sys.platform not in ("win32", "linux"):
        logger.info("System tray not supported on this platform")
        return None

    try:
        import pystray
        from PIL import Image, ImageDraw

        # Create a simple icon (blue circle = Ironcliw active)
        def create_icon():
            img = Image.new('RGB', (64, 64), color=(0, 100, 200))
            draw = ImageDraw.Draw(img)
            draw.ellipse([8, 8, 56, 56], fill=(0, 150, 255))
            return img

        menu = pystray.Menu(
            pystray.MenuItem('Open Ironcliw', on_show or (lambda: None)),
            pystray.MenuItem('Quit', on_quit or (lambda: None)),
        )

        icon = pystray.Icon("Ironcliw", create_icon(), "Ironcliw AI", menu)

        # Run in background thread
        tray_thread = threading.Thread(target=icon.run, daemon=True)
        tray_thread.start()
        logger.info("System tray icon started")
        return icon
    except ImportError:
        logger.warning("pystray not installed — no system tray. pip install pystray")
        return None
    except Exception as e:
        logger.warning(f"System tray failed: {e}")
        return None
```

**Install:** `pip install pystray Pillow`

---

### Phase 17 — Windows Hello Auth (FUTURE — Phase 12+)

**Goal:** Replace voice biometric bypass with Windows Hello (fingerprint/face).

**File to create:** `backend/auth/windows_hello.py`

```python
"""Windows Hello biometric authentication."""
import sys
import asyncio
import logging

logger = logging.getLogger(__name__)

async def authenticate_windows_hello(reason: str = "Ironcliw authentication") -> bool:
    """
    Authenticate via Windows Hello (fingerprint, face, PIN).
    Returns True if authenticated, False otherwise.
    """
    if sys.platform != "win32":
        return False

    try:
        # Method 1: Windows.Security.Credentials.UI (UWP API via winrt)
        import winrt.windows.security.credentials.ui as ui
        verifier = ui.UserConsentVerifier()
        result = await verifier.request_verification_async(reason)
        return result == ui.UserConsentVerificationResult.VERIFIED
    except ImportError:
        logger.info("winrt not installed — pip install winrt-Windows.Security.Credentials.UI")
    except Exception as e:
        logger.warning(f"Windows Hello failed: {e}")

    # Fallback: Accept if bypass mode is on
    import os
    if os.environ.get("Ironcliw_AUTO_BYPASS_WINDOWS") == "true":
        logger.info("Windows Hello unavailable — auth bypass active")
        return True
    return False
```

---

### Phase 18 — Brightness Control (FUTURE)

```python
import sys

def set_brightness(level: int) -> bool:
    """Set display brightness 0-100."""
    if sys.platform == "win32":
        try:
            # Method 1: WMI
            import wmi
            c = wmi.WMI(namespace='wmi')
            methods = c.WmiMonitorBrightnessMethods()[0]
            methods.WmiSetBrightness(level, 0)
            return True
        except Exception:
            pass
        try:
            # Method 2: PowerShell
            import subprocess
            cmd = f"(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})"
            subprocess.run(["powershell", "-Command", cmd],
                          capture_output=True, timeout=5)
            return True
        except Exception:
            return False
    else:
        import subprocess
        script = f'tell application "System Preferences" to set brightness to {level/100}'
        subprocess.run(['osascript', '-e', script], capture_output=True)
        return True
```

---

## PART 5 — LINUX PORT (DO AFTER WINDOWS IS STABLE)

### 5.1 Linux Overview

Linux port is easier than Windows because:
- Linux has `fcntl` natively (already used)
- No pywin32 needed
- `notify-send` replaces osascript notifications
- `pactl`/`amixer` replaces volume control
- `xdotool` replaces cliclick/pyautogui for some tasks
- Screen capture: `mss` works on Linux too (same as Windows fix)
- TTS: edge-tts works on Linux too (same fix)

### 5.2 Linux-Specific Replacements

| macOS | Windows (done) | Linux |
|-------|---------------|-------|
| osascript notification | plyer/win10toast | `notify-send` |
| caffeinate | SetThreadExecutionState | `xdg-screensaver` / `systemd-inhibit` |
| screencapture | mss ✅ | mss ✅ (same) |
| say -v | edge-tts ✅ | edge-tts ✅ (same) |
| pbcopy | pyperclip ✅ | `xclip` / pyperclip ✅ |
| cliclick | pyautogui ✅ | pyautogui ✅ (same) |
| yabai | pygetwindow/win32gui | `wmctrl` / `xdotool` |
| AppKit menu bar | pystray | pystray ✅ (same) |
| CoreGraphics | ctypes ✅ | Xlib / python-xlib |
| fcntl | msvcrt shim | fcntl ✅ (native) |
| Keychain | Windows Credential | `keyring` / `secret-service` |

### 5.3 Linux Platform Adapter

**Create:** `backend/platform_adapter/linux_platform.py`

```python
"""
Linux Platform Implementation for Ironcliw.
Uses xdotool, wmctrl, notify-send, pactl, and standard Unix tools.
"""

import os
import sys
import subprocess
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

import psutil

from .abstraction import PlatformInterface

logger = logging.getLogger(__name__)

# Optional imports
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False


class LinuxPlatform(PlatformInterface):
    """Linux-native platform implementation."""

    def __init__(self):
        super().__init__()
        self._display = os.environ.get("DISPLAY", ":0")

    # ─── Notifications ────────────────────────────────────────────────────────
    def show_notification(self, title: str, message: str, timeout: int = 5) -> bool:
        try:
            subprocess.run(
                ["notify-send", "-t", str(timeout * 1000), title, message],
                capture_output=True, timeout=5
            )
            return True
        except Exception:
            pass
        try:
            # fallback: plyer
            from plyer import notification
            notification.notify(title=title, message=message,
                                app_name="Ironcliw", timeout=timeout)
            return True
        except Exception:
            return False

    # ─── Screen capture ───────────────────────────────────────────────────────
    def take_screenshot(self) -> Optional[bytes]:
        if HAS_MSS:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                shot = sct.grab(monitor)
                from mss.tools import to_png
                return to_png(shot.rgb, shot.size)
        return None

    # ─── Window management ────────────────────────────────────────────────────
    def get_active_window_title(self) -> str:
        try:
            result = subprocess.run(
                ["xdotool", "getwindowfocus", "getwindowname"],
                capture_output=True, text=True, timeout=3
            )
            return result.stdout.strip()
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["wmctrl", "-l"],
                capture_output=True, text=True, timeout=3
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def focus_window(self, title: str) -> bool:
        try:
            subprocess.run(["wmctrl", "-a", title], timeout=3)
            return True
        except Exception:
            return False

    # ─── Volume control ───────────────────────────────────────────────────────
    def set_volume(self, level: int) -> bool:
        level = max(0, min(100, level))
        try:
            # PulseAudio
            subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                capture_output=True, timeout=3
            )
            return True
        except Exception:
            pass
        try:
            # ALSA fallback
            subprocess.run(
                ["amixer", "set", "Master", f"{level}%"],
                capture_output=True, timeout=3
            )
            return True
        except Exception:
            return False

    def get_volume(self) -> int:
        try:
            result = subprocess.run(
                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                capture_output=True, text=True, timeout=3
            )
            import re
            match = re.search(r'(\d+)%', result.stdout)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 50

    # ─── Sleep control ────────────────────────────────────────────────────────
    def prevent_sleep(self) -> bool:
        try:
            # systemd-inhibit is best but requires elevated perms
            # Use xdg-screensaver instead
            subprocess.Popen(
                ["xdg-screensaver", "reset"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception:
            return False

    def allow_sleep(self) -> bool:
        return True  # Linux resumes sleep automatically

    # ─── Screen lock ─────────────────────────────────────────────────────────
    def lock_screen(self) -> bool:
        for cmd in [
            ["loginctl", "lock-session"],
            ["gnome-screensaver-command", "--lock"],
            ["xdg-screensaver", "lock"],
            ["i3lock"],
        ]:
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                return True
            except Exception:
                continue
        return False

    def is_screen_locked(self) -> bool:
        try:
            result = subprocess.run(
                ["loginctl", "show-session", "--property=LockedHint"],
                capture_output=True, text=True, timeout=3
            )
            return "LockedHint=yes" in result.stdout
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["gnome-screensaver-command", "--query"],
                capture_output=True, text=True, timeout=3
            )
            return "is active" in result.stdout
        except Exception:
            pass
        return False

    # ─── Mouse / keyboard ────────────────────────────────────────────────────
    def click(self, x: int, y: int, button: str = "left") -> bool:
        if HAS_PYAUTOGUI:
            try:
                pyautogui.click(x, y, button=button)
                return True
            except Exception:
                pass
        try:
            btn = {"left": "1", "right": "3", "middle": "2"}.get(button, "1")
            subprocess.run(["xdotool", "mousemove", str(x), str(y),
                           "click", btn], timeout=3)
            return True
        except Exception:
            return False

    def type_text(self, text: str) -> bool:
        if HAS_PYAUTOGUI:
            try:
                pyautogui.write(text, interval=0.02)
                return True
            except Exception:
                pass
        try:
            subprocess.run(["xdotool", "type", "--", text], timeout=10)
            return True
        except Exception:
            return False

    def hotkey(self, *keys) -> bool:
        if HAS_PYAUTOGUI:
            try:
                pyautogui.hotkey(*keys)
                return True
            except Exception:
                pass
        try:
            key_combo = "+".join(keys)
            subprocess.run(["xdotool", "key", key_combo], timeout=3)
            return True
        except Exception:
            return False

    # ─── Clipboard ───────────────────────────────────────────────────────────
    def copy_to_clipboard(self, text: str) -> bool:
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
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
            return False

    def paste_from_clipboard(self) -> str:
        try:
            import pyperclip
            return pyperclip.paste()
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=3
            )
            return result.stdout
        except Exception:
            return ""

    # ─── System info ─────────────────────────────────────────────────────────
    def get_active_window_title(self) -> str:
        try:
            result = subprocess.run(
                ["xdotool", "getwindowfocus", "getwindowname"],
                capture_output=True, text=True, timeout=3
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def get_capabilities(self) -> Dict[str, bool]:
        return {
            "has_gui": bool(os.environ.get("DISPLAY")),
            "has_audio": True,
            "has_notifications": True,
            "has_automation": HAS_PYAUTOGUI,
            "has_screen_capture": HAS_MSS,
            "has_battery": psutil.sensors_battery() is not None,
            "has_window_management": True,
        }
```

### 5.4 Update Platform Adapter `__init__.py`

**File:** `backend/platform_adapter/__init__.py`

```python
import sys

def get_platform():
    """Auto-select platform implementation."""
    if sys.platform == "win32":
        from .windows_platform import WindowsPlatform
        return WindowsPlatform()
    elif sys.platform == "darwin":
        from .macos_platform import MacOSPlatform
        return MacOSPlatform()
    elif sys.platform.startswith("linux"):
        from .linux_platform import LinuxPlatform
        return LinuxPlatform()
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")
```

### 5.5 Linux Dependencies

**Create:** `requirements-linux.txt`

```text
# Core (same as Windows)
fastapi
uvicorn
websockets
anthropic
fireworks-ai
openai
openai-whisper
chromadb
aiohttp
aiohttp-cors
edge-tts
mss
Pillow
pyautogui
psutil
pyttsx3
pyperclip
plyer
pystray

# Linux-specific (NOT needed on Windows)
# xdotool — install via apt: sudo apt install xdotool
# wmctrl — install via apt: sudo apt install wmctrl
# xclip — install via apt: sudo apt install xclip
# notify-send — install via apt: sudo apt install libnotify-bin
# pactl — install via apt: sudo apt install pulseaudio-utils

# Python Linux audio
pyaudio
sounddevice
soundfile
```

### 5.6 Linux System Packages

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
  xdotool \
  wmctrl \
  xclip \
  libnotify-bin \
  pulseaudio-utils \
  scrot \
  python3-dev \
  build-essential \
  portaudio19-dev \
  ffmpeg

# Fedora/RHEL
sudo dnf install -y \
  xdotool wmctrl xclip libnotify \
  pulseaudio-utils scrot python3-devel \
  portaudio-devel ffmpeg

# Arch Linux
sudo pacman -S --noconfirm \
  xdotool wmctrl xclip libnotify \
  pulseaudio-utils scrot portaudio ffmpeg
```

### 5.7 Linux TTS (edge-tts — same as Windows!)

edge-tts works on Linux too. Same fix applies:
```python
# In realtime_voice_communicator.py — add Linux to the condition
if sys.platform in ("win32", "linux"):
    # ... edge-tts code (same as Windows) ...
    # MCI won't work — use subprocess mpg123 instead
    import subprocess
    subprocess.run(["mpg123", "-q", _tmp], check=True)
```

**Install mpg123 on Linux:**
```bash
sudo apt install mpg123
```

### 5.8 Linux Screen Lock Detection

```python
def is_screen_locked_linux() -> bool:
    import subprocess
    # Method 1: loginctl (systemd)
    try:
        result = subprocess.run(
            ["loginctl", "show-session", "--property=LockedHint"],
            capture_output=True, text=True, timeout=3
        )
        if "LockedHint=yes" in result.stdout:
            return True
    except Exception:
        pass
    # Method 2: gnome-screensaver
    try:
        result = subprocess.run(
            ["gnome-screensaver-command", "--query"],
            capture_output=True, text=True, timeout=3
        )
        if "is active" in result.stdout:
            return True
    except Exception:
        pass
    # Method 3: xscreensaver
    try:
        result = subprocess.run(
            ["xscreensaver-command", "-time"],
            capture_output=True, text=True, timeout=3
        )
        if "locked" in result.stdout:
            return True
    except Exception:
        pass
    return False
```

### 5.9 Linux Audio (ALSA/PulseAudio vs CoreAudio)

```python
# In full_duplex_device.py — replace CoreAudio with PortAudio
import sys

def get_audio_device():
    if sys.platform == "win32":
        return _get_windows_audio_device()
    elif sys.platform == "linux":
        return _get_linux_audio_device()
    else:
        return _get_macos_audio_device()

def _get_linux_audio_device():
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        info = pa.get_default_input_device_info()
        pa.terminate()
        return {
            'name': info.get('name', 'Default'),
            'index': info.get('index', 0),
            'sample_rate': int(info.get('defaultSampleRate', 44100)),
        }
    except Exception:
        return {'name': 'Default', 'index': 0, 'sample_rate': 44100}
```

---

## PART 6 — TESTING STRATEGY

### 6.1 Run Integration Tests After Every Change

```powershell
# Windows
python test_integration.py

# Linux
python3 test_integration.py
```

Must always show: `PASSED: 14/14` (more tests will be added as features grow).

### 6.2 Add Tests for New Features

When you implement a new feature, add a test to `test_integration.py`:

```python
def t_notifications():
    import asyncio
    import sys
    sys.path.insert(0, 'backend')
    from agi_os.notification_bridge import _notify_windows
    result = asyncio.run(_notify_windows("Test", "Integration test", "NORMAL"))
    # Don't assert True — notifications can fail in CI
    # Just test it doesn't throw an exception
test("Phase 12A — Windows notifications", t_notifications)
```

### 6.3 Backend Health Check

After every startup:
```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "http://localhost:8010/health/ping" -UseBasicParsing

# Or curl
curl http://localhost:8010/health/ping
# Expected: {"status":"ok"}

curl http://localhost:8010/health/startup
# Expected: {"progress":100,"status":"FULL MODE"}
```

### 6.4 Voice Test

```python
# test_voice.py — run while Ironcliw is running
import asyncio
import sys
sys.path.insert(0, 'backend')

async def test_tts():
    from agi_os.realtime_voice_communicator import RealtimeVoiceCommunicator
    comm = RealtimeVoiceCommunicator()
    # Should speak without error
    await comm.speak("Integration test successful. Ironcliw is working on Windows.")
    print("TTS test complete")

asyncio.run(test_tts())
```

---

## PART 7 — TROUBLESHOOTING GUIDE

### Error: `No module named 'fcntl'`

```powershell
git pull origin main
# Fix is in: backend/core/resilience/atomic_file_ops.py
# The msvcrt shim handles this automatically on Windows
```

### Error: `edge-tts` DNS resolution fails

```powershell
# The fix is already in realtime_voice_communicator.py
# It patches aiohttp.resolver.DefaultResolver → ThreadedResolver
# If it still fails:
pip install aiohttp --upgrade
```

### Error: Backend starts on wrong port

```powershell
# Check .env:
Ironcliw_DYNAMIC_PORTS=false
Ironcliw_PORT=8010

# Kill stale processes:
netstat -ano | findstr :8010
taskkill /PID <pid> /F
```

### Error: `UNIQUE constraint failed`

```powershell
git pull origin main
# Fix is in: backend/intelligence/learning_database.py
# Hash pre-check before every INSERT
```

### Error: `WinError 2` on Keychain

```powershell
git pull origin main
# Fix is in: start_system.py (Windows guard)
```

### Error: `speechbrain not installed` — 25s delay

```powershell
git pull origin main
# ECAPA fast-fail in: backend/voice_unlock/ml_engine_registry.py
```

### Error: Frontend shows "CONNECTING..." forever

```powershell
# Check backend health first:
curl http://localhost:8010/health/ping

# Rebuild frontend:
cd frontend && npm install && npm run build && cd ..
python start_system.py
```

### Error: `No module named 'resource'` warning

This is non-fatal on Windows. psutil provides the fallback. Ignore it.

### Integration test FAIL: Phase 15 hardware_control

```powershell
# Check atomic_file_ops.py has the fcntl shim:
python -c "import sys; sys.path.insert(0,'backend'); import core.resilience.atomic_file_ops; print('OK')"
```

### Integration test FAIL: Phase 16 platform_adapter

```powershell
# Check windows_platform.py has get_active_window_title:
python -c "
import sys; sys.path.insert(0,'backend')
from platform_adapter import get_platform
p = get_platform()
print(hasattr(p, 'get_active_window_title'))  # Must print True
"
```

### Integration test FAIL: Phase 20 screen lock

```powershell
# Check screen_lock_detector.py has module-level is_screen_locked():
python -c "
import sys; sys.path.insert(0,'backend')
from context_intelligence.detectors.screen_lock_detector import is_screen_locked
print(is_screen_locked())  # Must print True or False (not error)
"
```

---

## PART 8 — GIT WORKFLOW (ALWAYS FOLLOW THIS)

### After Every Fix — Commit and Push

```powershell
cd C:\Users\nandk\Ironcliw

# Stage the changed files
git add <file1> <file2>

# Commit (use a message file to avoid quoting issues on Windows CMD)
echo fix: description of what you fixed > tmp.txt
git commit -F tmp.txt
del tmp.txt

# Push
git push origin main

# Verify on GitHub:
# https://github.com/nandkishorrathodk-art/Ironcliw-ai
```

### If GitHub Actions fail

1. Click "Actions" tab on GitHub
2. Click the failing workflow
3. Read the error log
4. Fix the code
5. Commit and push again

### Checking for upstream updates (drussell23/Ironcliw)

```powershell
# See if original repo has new commits
git fetch upstream
git log --oneline upstream/main..HEAD  # Our commits not in upstream
git log --oneline HEAD..upstream/main  # Upstream commits not in ours

# Merge upstream changes (carefully — keep Windows fixes!)
git merge upstream/main --no-commit --no-ff -X patience
# Resolve conflicts: always keep Windows-specific code
# Conflicts will be in: main.py, realtime_voice_communicator.py, async_pipeline.py
```

---

## PART 9 — ARCHITECTURE DECISIONS (NEVER CHANGE THESE)

| Decision | Reason |
|----------|--------|
| Use `sys.platform == "win32"` guards | Never import macOS libs on Windows |
| `edge-tts` + MCI playback for TTS | Zero deps, built into Windows, sounds human |
| `mss` for screen capture | Cross-platform (Win/Linux/Mac) |
| `pyautogui` for mouse/keyboard | Works on Win/Linux/Mac without extra tools |
| `msvcrt` shim for `fcntl` | Windows file locking compatibility |
| `plyer` for notifications | Cross-platform (Win/Linux/Mac) |
| `Ironcliw_DYNAMIC_PORTS=false` | Keep port 8010 stable — no port scanning |
| Fallback chain: new → old → stub | Never crash, always degrade gracefully |
| `try/except ImportError` everywhere | Missing package = warning, not crash |

---

## PART 10 — FEATURE PRIORITY MATRIX

### Do These NOW (Windows — high impact)

1. ✅ Neural TTS (`en-GB-RyanNeural`) — DONE
2. ✅ Screen capture (`mss`) — DONE
3. ✅ Ghost Hands (`pyautogui`) — DONE
4. ✅ Platform adapter (get_active_window_title etc.) — DONE
5. ✅ Notifications (plyer fallback) — DONE
6. ⬜ Volume control (`pycaw`) — Next
7. ⬜ ECAPA fast-fail (save 25s) — Next
8. ⬜ System tray icon (`pystray`) — After volume
9. ⬜ Windows Hello auth — Future

### Do These AFTER Windows is stable (Linux)

1. ⬜ Linux platform adapter (`linux_platform.py`) — See Part 5
2. ⬜ Linux TTS (`mpg123` + edge-tts) — See Part 5
3. ⬜ Linux notifications (`notify-send`) — See Part 5
4. ⬜ Linux screen lock (`loginctl`) — See Part 5
5. ⬜ Linux volume (`pactl`/`amixer`) — See Part 5
6. ⬜ Linux window management (`xdotool`/`wmctrl`) — See Part 5

---

## PART 11 — FILES MAP (WHERE EVERYTHING IS)

```
Ironcliw-ai/
├── start_system.py              ← Main launcher (Windows-patched)
├── unified_supervisor.py        ← Process manager (cross-platform)
├── test_integration.py          ← Run this! 14/14 must pass
├── .env.windows                 ← Copy to .env and add API keys
├── FRESH_INSTALL_WINDOWS.md     ← Fresh install guide
├── WINDOWS_PORT_BLUEPRINT.md    ← macOS→Windows conversion guide
├── AI_AGENT_MASTER_BLUEPRINT.md ← This file (give to any AI agent)
│
├── backend/
│   ├── main.py                  ← UTF-8 fix + _bootstrap_import_paths
│   ├── platform_adapter/
│   │   ├── __init__.py          ← get_platform() auto-selector
│   │   ├── windows_platform.py  ← Windows impl (pywin32, ctypes)
│   │   └── macos_platform.py    ← macOS impl (keep untouched)
│   ├── agi_os/
│   │   ├── realtime_voice_communicator.py  ← edge-tts + MCI TTS
│   │   └── notification_bridge.py          ← Notifications (plyer)
│   ├── voice/
│   │   ├── hybrid_stt_router.py            ← Whisper STT
│   │   └── speaker_verification_service.py ← ECAPA bypass
│   ├── voice_unlock/
│   │   └── ml_engine_registry.py           ← ECAPA fast-fail
│   ├── ghost_hands/
│   │   └── background_actuator.py          ← pyautogui automation
│   ├── autonomy/
│   │   └── hardware_control.py             ← Volume, sleep, camera
│   ├── context_intelligence/
│   │   └── detectors/screen_lock_detector.py ← is_screen_locked()
│   ├── core/
│   │   ├── resilience/
│   │   │   └── atomic_file_ops.py          ← fcntl/msvcrt shim
│   │   └── infrastructure_orchestrator.py  ← os.uname() fix
│   ├── intelligence/
│   │   └── learning_database.py            ← UNIQUE constraint fix
│   └── chatbots/
│       └── claude_vision_chatbot.py        ← mss screenshot
│
├── frontend/
│   └── src/components/
│       └── JarvisVoice.js       ← sanitizeForSpeech, TTS UI
│
└── .github/workflows/
    └── auto-diagram-generator.yml ← continue-on-error: true (fixed)
```

---

## PART 12 — QUICK COMMAND REFERENCE

### Windows (PowerShell)

```powershell
# Fresh clone
git clone https://github.com/nandkishorrathodk-art/Ironcliw-ai.git Ironcliw
cd Ironcliw

# Install deps
pip install -r requirements.txt
pip install edge-tts mss pyautogui pywin32 pyttsx3 plyer soundfile openai-whisper chromadb aiohttp anthropic fireworks-ai
cd frontend && npm install && cd ..

# Configure
copy .env.windows .env
# Edit .env and add ANTHROPIC_API_KEY and FIREWORKS_API_KEY

# Test
python test_integration.py

# Run
python start_system.py

# After fixing something
git add <files>
echo fix: what you fixed > tmp.txt && git commit -F tmp.txt && del tmp.txt && git push origin main

# Check logs (when running)
curl http://localhost:8010/health/ping
curl http://localhost:8010/health/startup
```

### Linux (Bash)

```bash
# System deps
sudo apt install -y xdotool wmctrl xclip libnotify-bin pulseaudio-utils mpg123 ffmpeg portaudio19-dev

# Clone
git clone https://github.com/nandkishorrathodk-art/Ironcliw-ai.git Ironcliw
cd Ironcliw

# Install deps
pip3 install -r requirements.txt
pip3 install edge-tts mss pyautogui pyttsx3 plyer soundfile openai-whisper chromadb aiohttp anthropic fireworks-ai pyperclip

# Frontend
cd frontend && npm install && cd ..

# Configure
cp .env.windows .env
# Edit .env — add API keys

# Test
python3 test_integration.py

# Run
python3 start_system.py
```

---

## PART 13 — WHAT TO TELL THE NEXT AI AGENT

**Copy-paste this as the first message:**

```
You are continuing work on the Ironcliw Windows port for user Nandkishor Rathod.

REPO: https://github.com/nandkishorrathodk-art/Ironcliw-ai
LOCAL PATH: C:\Users\nandk\Ironcliw (Windows) or ~/Ironcliw (Linux)
BRANCH: main
LATEST COMMIT: 0afb6259

CURRENT STATUS:
- Phase 1-11 complete — all 14 integration tests pass on Windows
- Neural TTS (en-GB-RyanNeural edge-tts) working
- Screen capture (mss) working
- Ghost Hands (pyautogui) working
- Platform adapter working
- Backend: FastAPI on port 8010
- Frontend: React on port 3000

IMMEDIATE TASKS:
1. Fresh install from GitHub (see FRESH_INSTALL_WINDOWS.md)
2. Run: python test_integration.py → must show PASSED: 14/14
3. Run: python start_system.py → open http://localhost:3000
4. Implement volume control (pycaw) in hardware_control.py
5. Implement ECAPA fast-fail in ml_engine_registry.py (saves 25s)
6. After Windows is stable → port to Linux (see AI_AGENT_MASTER_BLUEPRINT.md Part 5)

RULES:
- Always run test_integration.py before and after any change
- Always commit and push after every fix
- Never break existing 14/14 test score
- Use sys.platform == "win32" guards for Windows-specific code
- Never use: osascript, screencapture, caffeinate, yabai, cliclick, pyobjc
- Always use: pyautogui, mss, pywin32, ctypes, edge-tts, plyer

FULL INSTRUCTIONS: Read AI_AGENT_MASTER_BLUEPRINT.md in the repo root.
```

---

*AI Agent Master Blueprint v1.0*
*Phase 11 Complete — Windows: ✅ 14/14 tests pass — Linux: ⬜ Pending*
*Repo: nandkishorrathodk-art/Ironcliw-ai | Last Updated: 2026-02-25*
