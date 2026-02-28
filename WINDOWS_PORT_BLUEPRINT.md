# Ironcliw Windows Port Blueprint
## Complete macOS → Windows Conversion Guide
**Repo:** `drussell23/Ironcliw` → `nandkishorrathodk-art/Ironcliw-ai`
**Status:** Phase 11 Session 9 Complete | Phase 12 Remaining

---

## TABLE OF CONTENTS
1. [Current Status Summary](#1-current-status-summary)
2. [What Has Been Done (Phases 1–11)](#2-what-has-been-done-phases-111)
3. [Remaining Work — Full Audit](#3-remaining-work--full-audit)
4. [Phase 12 — Notifications & System Control](#4-phase-12--notifications--system-control)
5. [Phase 13 — Ghost Hands & Automation](#5-phase-13--ghost-hands--automation)
6. [Phase 14 — Vision & Screen Capture](#6-phase-14--vision--screen-capture)
7. [Phase 15 — Hardware Control](#7-phase-15--hardware-control)
8. [Phase 16 — Platform Adapter Completion](#8-phase-16--platform-adapter-completion)
9. [Phase 17 — Yabai / Window Management](#9-phase-17--yabai--window-management)
10. [Phase 18 — System-Level & Unix Primitives](#10-phase-18--system-level--unix-primitives)
11. [Phase 19 — Audio System](#11-phase-19--audio-system)
12. [Phase 20 — Screen Lock / Unlock](#12-phase-20--screen-lock--unlock)
13. [Windows API Reference Cheatsheet](#13-windows-api-reference-cheatsheet)
14. [Testing Each Phase](#14-testing-each-phase)
15. [Dependency Matrix](#15-dependency-matrix)

---

## 1. Current Status Summary

### ✅ DONE — Working on Windows Right Now
| Component | Status | Notes |
|-----------|--------|-------|
| Python backend startup | ✅ Working | Port 8010 |
| FastAPI server | ✅ Working | All routes operational |
| Frontend (React) | ✅ Working | Port 3000 |
| WebSocket router | ✅ Working | npm.cmd fix applied |
| SQLite database | ✅ Working | Cloud SQL gracefully degraded |
| TTS Voice (pyttsx3 → edge-tts) | ✅ Working | en-GB-RyanNeural neural voice |
| STT (Whisper) | ✅ Working | CPU mode, base model |
| Claude API | ✅ Working | Anthropic fallback |
| Fireworks AI | ✅ Working | Primary LLM |
| Screen capture | ✅ Working | mss library |
| UNIQUE constraint spam | ✅ Fixed | learning_database.py |
| os.uname() crash | ✅ Fixed | infrastructure_orchestrator.py |
| NoneType traceback | ✅ Fixed | speaker_verification_service.py |
| Keychain WinError 2 | ✅ Fixed | start_system.py |
| fcntl import crash | ✅ Fixed | intelligent_gcp_optimizer.py |
| Unicode crash | ✅ Fixed | main.py UTF-8 reconfigure |
| Chrome launch | ✅ Fixed | webbrowser module fallback |
| ECAPA 25s timeout | ⚠️ Known | speechbrain not installed |
| Backend status "degraded" | ⚠️ Known | Minor — non-breaking |

### ❌ NOT DONE — macOS Only Features
| Component | macOS API | Windows Replacement |
|-----------|-----------|---------------------|
| System notifications | osascript | win10toast / plyer |
| Screen lock/unlock | osascript | ctypes LockWorkStation() |
| App launch/quit | osascript | subprocess / pywin32 |
| Volume control | osascript | pycaw / ctypes |
| Window management | yabai | pygetwindow / win32gui |
| Mouse/keyboard control | cliclick | pyautogui / pywin32 |
| Screen recording | screencapture | mss (already done) |
| File open dialog | AppKit | tkinter / pywin32 |
| macOS menu bar | AppKit | pystray |
| Apple Watch proximity | CoreBluetooth | Windows BLE API |
| AirPlay display | macOS only | Skip / Windows Display API |
| CoreGraphics clicks | Quartz | pyautogui / ctypes |
| SIGSTOP process | Unix only | ctypes NtSuspendProcess |
| fcntl file lock | Unix only | msvcrt.locking |
| resource module | Unix only | psutil fallback |
| vm_stat memory | macOS only | psutil.virtual_memory() |

---

## 2. What Has Been Done (Phases 1–11)

### Phase 1 — Foundation & Platform Abstraction Layer (PAL)
**Files created:** `backend/platform_adapter/windows_platform.py`, `backend/platform_adapter/__init__.py`
```
backend/platform_adapter/
├── __init__.py          ← Auto-selects platform at import time
├── macos_platform.py    ← Original macOS impl
├── linux_platform.py    ← Linux impl
└── windows_platform.py  ← NEW: Windows impl (pywin32 + ctypes)
```
**Key changes:**
- `get_platform()` returns `WindowsPlatform` on Windows
- All platform calls route through PAL → zero macOS imports on Windows
- `sys.platform == "win32"` guards added everywhere

### Phase 2 — Dependencies (192 packages ported)
**File:** `requirements.txt` + `requirements-windows.txt`
- Removed: `pyobjc-*`, `CoreML`, `Metal`, `AppKit`
- Added: `pywin32`, `pyttsx3`, `edge-tts`, `mss`, `pyautogui`
- Fixed: torch CPU-only, speechbrain skipped, torchaudio skipped
- `.env` set: `Ironcliw_ML_DEVICE=cpu`, `Ironcliw_SKIP_DOCKER=true`

### Phase 3 — Screen Capture
**Files modified:** `backend/vision/`, `backend/chatbots/claude_vision_chatbot.py`
- `screencapture` command → `mss.mss()` library
- PIL/Pillow pipeline preserved
- Cross-platform: same code works macOS/Linux/Windows

### Phase 4 — Auth Bypass (macOS screen unlock → stub)
**Files modified:** `backend/voice_unlock/secure_password_typer.py`
- Removed CoreGraphics dependency
- Windows: uses `pyautogui.typewrite()` or ctypes `SendInput()`
- Keychain removed → Windows Credential Manager integration

### Phase 5 — Unified Supervisor
**File:** `unified_supervisor.py`
- Replaced `signal.SIGTERM` fork patterns with Windows-compatible subprocess management
- `run_supervisor.py` works on Windows without posix signals

### Phase 6 — ECAPA Voice Biometrics Bypass
**Files:** `backend/voice_unlock/ml_engine_registry.py`
- ECAPA loads fail gracefully (speechbrain not installed)
- 25s timeout still waits — Phase 12 will fast-fail this
- Basic mode (45% threshold) works without ECAPA

### Phase 7 — Critical Runtime Bug Fixes
- WebSocket router: `node` → `node.cmd` / `npm.cmd` on Windows
- DynamicConfigService: port scan expanded
- `Ironcliw_DYNAMIC_PORTS=false` env var
- DB migration inline for `confidence` column
- GCP shift spam eliminated

### Phase 8 — Security & Database
- `secure_logging.py` (CWE-117/532 log injection prevention)
- `ATOMIC_WRITE_PERMS` 0o644 → 0o600
- DB UPSERT PK condition fixed
- PostgreSQL GROUP BY alias fixed
- `fcntl` Windows guard in `intelligent_gcp_optimizer.py`
- Installed: `soundfile`, `asyncpg`

### Phase 9 — Push & Runtime Fixes
- GitHub push (API keys redacted from `.env.windows`)
- UNIQUE constraint spam fixed
- `os.uname()` crash fixed
- `_detect_current_model_dimension` NoneType fixed
- Keychain WinError 2 silenced
- TTS voice: pyttsx3 → edge-tts `en-GB-RyanNeural`

### Phase 10-11 — Upstream Sync
- 130 new commits from `drussell23/Ironcliw` merged
- 4 conflicts resolved (all keeping Windows fixes)
- Pushed to fork as merge commit `3ce7237a`

---

## 3. Remaining Work — Full Audit

### Priority Matrix
```
CRITICAL (breaks core features):
  P1 → Notification system (osascript → Windows notifications)
  P2 → Screen lock detection (for voice unlock flow)
  P3 → Ghost Hands automation (osascript → pyautogui)

HIGH (reduces functionality):
  P4 → Hardware control (volume, brightness, sleep)
  P5 → Window management (yabai → pygetwindow/win32gui)
  P6 → Platform adapter completion (macos_controller.py)

MEDIUM (nice to have):
  P7 → ECAPA fast-fail (save 25s on startup × 2)
  P8 → macOS menu bar → pystray system tray
  P9 → Unix primitives (fcntl, resource, SIGSTOP)

LOW (macOS-exclusive, skip or stub):
  P10 → Apple Watch proximity (CoreBluetooth)
  P11 → AirPlay display (macOS Display API)
  P12 → Metal/CoreML acceleration (use CUDA/DirectML instead)
```

---

## 4. Phase 12 — Notifications & System Control

### 4.1 Problem: `notification_bridge.py`
**File:** `backend/agi_os/notification_bridge.py`

Current macOS code:
```python
# LINE 322 — osascript notification
async def _notify_osascript(title, body, urgency):
    script = f'display notification "{body}" with title "{title}"'
    proc = await asyncio.create_subprocess_exec(
        "osascript", "-e", script, ...
    )
```

### 4.2 Windows Fix: win10toast + plyer fallback

**Install:**
```bash
pip install plyer win10toast
```

**Replacement code for `notification_bridge.py`:**
```python
import sys

async def _notify_platform(title: str, body: str, urgency: str) -> bool:
    if sys.platform == "win32":
        return await _notify_windows(title, body, urgency)
    else:
        return await _notify_osascript(title, body, urgency)

async def _notify_windows(title: str, body: str, urgency: str) -> bool:
    """Windows 10/11 toast notification via plyer (cross-platform) or win10toast."""
    loop = asyncio.get_event_loop()

    def _show_toast():
        try:
            # Method 1: plyer (recommended, works on Win/Mac/Linux)
            from plyer import notification
            notification.notify(
                title=title,
                message=body,
                app_name="Ironcliw",
                timeout=5 if urgency == "HIGH" else 3,
            )
            return True
        except Exception:
            pass
        try:
            # Method 2: win10toast (Windows-only fallback)
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(
                title,
                body,
                duration=5,
                threaded=True,
            )
            return True
        except Exception:
            pass
        return False

    return await loop.run_in_executor(None, _show_toast)

# Keep original macOS version
async def _notify_osascript(title: str, body: str, urgency: str) -> bool:
    body_escaped = body.replace('"', '\\"').replace("'", "\\'")
    title_escaped = title.replace('"', '\\"').replace("'", "\\'")
    script = f'display notification "{body_escaped}" with title "{title_escaped}"'
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=5.0)
        return proc.returncode == 0
    except Exception:
        return False
```

### 4.3 ECAPA Fast-Fail (saves 50s on startup)
**File:** `backend/voice_unlock/ml_engine_registry.py`

Find the ECAPA load function and add at the top:
```python
async def _load_ecapa_engine(self) -> Optional[Any]:
    # WINDOWS FAST-FAIL: speechbrain requires torchaudio which is not
    # available on Windows without GPU. Skip immediately.
    if sys.platform == "win32":
        try:
            import speechbrain  # noqa
        except ImportError:
            logger.info("[ecapa_tdnn] speechbrain not installed on Windows — skipping (use cloud ECAPA)")
            return None  # Return immediately — no 25s timeout

    # ... rest of loading code
```

---

## 5. Phase 13 — Ghost Hands & Automation

### 5.1 Problem Files
```
backend/api/action_executors.py        — 8+ osascript calls
backend/autonomy/hardware_control.py   — caffeinate, osascript
backend/autonomy/macos_integration.py  — entire file is macOS only
backend/ghost_hands/background_actuator.py — cliclick
backend/ghost_hands/yabai_aware_actuator.py — yabai + cliclick
backend/autonomy/action_executor.py    — osascript
backend/autonomy/langchain_tools.py    — osascript
```

### 5.2 osascript → Windows Replacement Map

| macOS (osascript) | Windows Replacement | Python Library |
|-------------------|---------------------|----------------|
| `key code 12 using {command down}` | `win32api.keybd_event()` | pywin32 |
| `keystroke "q" using {command down}` | `keyboard.send('ctrl+q')` | keyboard |
| `click at {x, y}` | `win32api.SetCursorPos((x,y)); click` | pywin32 |
| `set volume output volume 50` | pycaw / ctypes | pycaw |
| `tell application "Safari"` | `subprocess.Popen(['chrome.exe'])` | subprocess |
| `get name of first process` | `psutil.process_iter()` | psutil |
| `display dialog "text"` | `tkinter.messagebox.askokcancel()` | tkinter |
| `quit application "App"` | `psutil` kill by name | psutil |

### 5.3 Full Implementation: `action_executors.py`

**Pattern for every osascript block:**
```python
import sys

def _run_osascript(script: str, timeout: int = 10) -> bool:
    """Run AppleScript — macOS only."""
    if sys.platform == "win32":
        logger.warning("[ActionExecutor] osascript not available on Windows — action skipped")
        return False
    import subprocess
    result = subprocess.run(
        ['osascript', '-e', script],
        capture_output=True, text=True, timeout=timeout
    )
    return result.returncode == 0

async def _run_osascript_async(script: str, timeout: float = 10.0) -> bool:
    """Async osascript — macOS only."""
    if sys.platform == "win32":
        return False
    proc = await asyncio.create_subprocess_exec(
        "osascript", "-e", script,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        return proc.returncode == 0
    except asyncio.TimeoutError:
        proc.kill()
        return False
```

### 5.4 cliclick → pyautogui

**Install:**
```bash
pip install pyautogui
```

**Replacement:**
```python
import sys

def click_at(x: int, y: int) -> bool:
    """Click at screen coordinates — cross-platform."""
    try:
        import pyautogui
        pyautogui.click(x, y)
        return True
    except Exception as e:
        logger.warning(f"Click failed: {e}")
        return False

def move_mouse(x: int, y: int) -> bool:
    """Move mouse — cross-platform."""
    try:
        import pyautogui
        pyautogui.moveTo(x, y, duration=0.1)
        return True
    except Exception as e:
        logger.warning(f"Mouse move failed: {e}")
        return False

def type_text(text: str, interval: float = 0.05) -> bool:
    """Type text — cross-platform."""
    try:
        import pyautogui
        pyautogui.write(text, interval=interval)
        return True
    except Exception as e:
        logger.warning(f"Type text failed: {e}")
        return False

def press_key(key: str) -> bool:
    """Press a key — cross-platform."""
    try:
        import pyautogui
        pyautogui.press(key)
        return True
    except Exception as e:
        logger.warning(f"Key press failed: {e}")
        return False

def hotkey(*keys) -> bool:
    """Press key combination — cross-platform."""
    try:
        import pyautogui
        pyautogui.hotkey(*keys)
        return True
    except Exception as e:
        logger.warning(f"Hotkey failed: {e}")
        return False
```

### 5.5 caffeinate → Windows Power Request

```python
import sys

def prevent_sleep() -> Optional[Any]:
    """Prevent system sleep — cross-platform."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        return True
    else:
        # macOS: caffeinate
        import subprocess
        return subprocess.Popen(['caffeinate', '-d', '-i', '-m'])

def allow_sleep(handle=None) -> None:
    """Allow system to sleep again — cross-platform."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    else:
        if handle:
            handle.terminate()
```

---

## 6. Phase 14 — Vision & Screen Capture

### 6.1 Already Fixed
`mss` library handles cross-platform screen capture. All `screencapture` subprocess calls need to be replaced:

**File:** `backend/chatbots/claude_vision_chatbot.py`

**Current macOS code (lines 736–775):**
```python
async def _capture_screencapture_cmd_fast(self):
    cmd = ["screencapture", "-x", "-t", "png", tmp_path]
```

**Windows replacement (already in platform adapter, just wire it up):**
```python
async def _capture_screencapture_cmd_fast(self) -> Optional[Image.Image]:
    """Cross-platform fast screenshot."""
    try:
        import mss
        import mss.tools
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # All monitors combined
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return img
    except Exception as e:
        logger.debug(f"mss screenshot failed: {e}")
        return None
```

### 6.2 Window Capture (Multi-Monitor)
**File:** `backend/context_intelligence/managers/window_capture_manager.py`

```python
import sys

def get_window_screenshot(hwnd_or_title: str) -> Optional[Image.Image]:
    """Capture specific window — platform-specific."""
    if sys.platform == "win32":
        return _capture_window_win32(hwnd_or_title)
    else:
        return _capture_window_macos(hwnd_or_title)

def _capture_window_win32(title: str) -> Optional[Image.Image]:
    """Win32 window capture via pywin32."""
    try:
        import win32gui
        import win32ui
        import win32con
        from ctypes import windll

        hwnd = win32gui.FindWindow(None, title)
        if not hwnd:
            # Try partial match
            def enum_cb(h, results):
                if title.lower() in win32gui.GetWindowText(h).lower():
                    results.append(h)
            hwnds = []
            win32gui.EnumWindows(enum_cb, hwnds)
            hwnd = hwnds[0] if hwnds else None

        if not hwnd:
            return None

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width, height = right - left, bottom - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)
        windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                               bmpstr, 'raw', 'BGRX', 0, 1)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        return img
    except Exception as e:
        logger.warning(f"Win32 window capture failed: {e}")
        return None
```

---

## 7. Phase 15 — Hardware Control

### 7.1 Volume Control
**File:** `backend/autonomy/hardware_control.py`

**macOS (osascript):**
```python
script = f'set volume output volume {volume}'
subprocess.run(['osascript', '-e', script])
```

**Windows replacement:**
```python
def set_volume_windows(volume: int) -> bool:
    """Set system volume 0-100 on Windows."""
    try:
        # Method 1: pycaw (COM-based, most reliable)
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_ctl = cast(interface, POINTER(IAudioEndpointVolume))
        volume_scalar = volume / 100.0
        volume_ctl.SetMasterVolumeLevelScalar(volume_scalar, None)
        return True
    except Exception:
        pass
    try:
        # Method 2: ctypes WinMM
        import ctypes
        # Volume range is 0x0000 to 0xFFFF
        vol = int(volume / 100 * 0xFFFF)
        ctypes.windll.winmm.waveOutSetVolume(None, (vol << 16) | vol)
        return True
    except Exception:
        return False

def get_volume_windows() -> int:
    """Get current system volume 0-100 on Windows."""
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_ctl = cast(interface, POINTER(IAudioEndpointVolume))
        return int(volume_ctl.GetMasterVolumeLevelScalar() * 100)
    except Exception:
        return 50  # Unknown
```

**Install:** `pip install pycaw comtypes`

### 7.2 Brightness Control
**macOS:** `brightness` CLI tool or osascript
**Windows:** WMI or registry

```python
def set_brightness_windows(level: int) -> bool:
    """Set display brightness 0-100 on Windows (requires WMI)."""
    try:
        import wmi
        c = wmi.WMI(namespace='wmi')
        methods = c.WmiMonitorBrightnessMethods()[0]
        methods.WmiSetBrightness(level, 0)
        return True
    except Exception:
        try:
            # Fallback: PowerShell
            import subprocess
            subprocess.run([
                "powershell", "-Command",
                f"(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})"
            ], capture_output=True, timeout=5)
            return True
        except Exception:
            return False
```

### 7.3 System Sleep
```python
import sys

def sleep_system() -> bool:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.PowrProf.SetSuspendState(0, 1, 0)
        return True
    else:
        import subprocess
        subprocess.run(['osascript', '-e', 'tell application "System Events" to sleep'])
        return True
```

---

## 8. Phase 16 — Platform Adapter Completion

### 8.1 Files to Complete

#### `backend/platform_adapter/windows_platform.py` (EXPAND)
Current state: basic stubs exist. Need to complete:

```python
import sys
import os
import ctypes
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

class WindowsPlatform:
    """Complete Windows platform implementation."""

    # ─── Notifications ───────────────────────────────────────────────
    def show_notification(self, title: str, message: str, timeout: int = 5) -> bool:
        try:
            from plyer import notification
            notification.notify(title=title, message=message,
                                app_name="Ironcliw", timeout=timeout)
            return True
        except Exception:
            return False

    # ─── Screen Lock ─────────────────────────────────────────────────
    def lock_screen(self) -> bool:
        ctypes.windll.user32.LockWorkStation()
        return True

    def is_screen_locked(self) -> bool:
        """Check if Windows workstation is locked."""
        # Detect via GetForegroundWindow returning 0 when locked
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if hwnd == 0:
            return True
        # Check for LogonUI.exe (lock screen process)
        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] and 'LogonUI' in proc.info['name']:
                    return True
        except Exception:
            pass
        return False

    # ─── App Control ─────────────────────────────────────────────────
    def launch_app(self, app_name: str) -> bool:
        """Launch app by name or path."""
        try:
            subprocess.Popen([app_name])
            return True
        except Exception:
            try:
                os.startfile(app_name)
                return True
            except Exception:
                return False

    def quit_app(self, app_name: str) -> bool:
        """Quit app by process name."""
        try:
            import psutil
            killed = False
            for proc in psutil.process_iter(['name', 'pid']):
                if app_name.lower() in proc.info['name'].lower():
                    proc.terminate()
                    killed = True
            return killed
        except Exception:
            return False

    # ─── Window Management ───────────────────────────────────────────
    def get_active_window_title(self) -> str:
        """Get title of currently active window."""
        try:
            import win32gui
            hwnd = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(hwnd)
        except Exception:
            return ""

    def list_windows(self) -> List[Dict[str, Any]]:
        """List all visible windows with title, position, size."""
        windows = []
        try:
            import win32gui
            def enum_cb(hwnd, results):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        rect = win32gui.GetWindowRect(hwnd)
                        results.append({
                            'hwnd': hwnd,
                            'title': title,
                            'x': rect[0], 'y': rect[1],
                            'width': rect[2] - rect[0],
                            'height': rect[3] - rect[1],
                        })
            win32gui.EnumWindows(enum_cb, windows)
        except Exception:
            pass
        return windows

    def focus_window(self, title: str) -> bool:
        """Bring window to foreground by title."""
        try:
            import win32gui
            hwnd = win32gui.FindWindow(None, title)
            if hwnd:
                win32gui.SetForegroundWindow(hwnd)
                return True
        except Exception:
            pass
        return False

    # ─── Volume ──────────────────────────────────────────────────────
    def set_volume(self, level: int) -> bool:
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol_ctl = cast(interface, POINTER(IAudioEndpointVolume))
            vol_ctl.SetMasterVolumeLevelScalar(level / 100.0, None)
            return True
        except Exception:
            return False

    def get_volume(self) -> int:
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

    # ─── Mouse & Keyboard ────────────────────────────────────────────
    def click(self, x: int, y: int, button: str = "left") -> bool:
        try:
            import pyautogui
            pyautogui.click(x, y, button=button)
            return True
        except Exception:
            return False

    def type_text(self, text: str) -> bool:
        try:
            import pyautogui
            pyautogui.write(text, interval=0.02)
            return True
        except Exception:
            return False

    def hotkey(self, *keys) -> bool:
        try:
            import pyautogui
            pyautogui.hotkey(*keys)
            return True
        except Exception:
            return False

    # ─── Sleep Prevention ────────────────────────────────────────────
    def prevent_sleep(self) -> bool:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        return True

    def allow_sleep(self) -> bool:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        return True
```

#### `backend/system_control/macos_controller.py` → `windows_controller.py`
Create `backend/system_control/windows_controller.py` with equivalent functionality using pywin32.

---

## 9. Phase 17 — Yabai / Window Management

### 9.1 What is Yabai?
Yabai is a macOS tiling window manager. Ironcliw uses it to:
- Detect which "space" (virtual desktop) is active
- Move windows between spaces
- Get window positions for ghost overlay display

### 9.2 Files affected
```
backend/vision/yabai_space_detector.py      — main yabai integration
backend/ghost_hands/yabai_aware_actuator.py — uses yabai positions
backend/agi_os/agi_os_coordinator.py        — imports yabai detector
backend/api/component_warmup_config.py      — loads yabai as component
```

### 9.3 Windows Replacement: Virtual Desktop + Win32

```python
import sys

class VirtualDesktopManager:
    """Cross-platform virtual desktop/space manager."""

    def get_current_space(self) -> int:
        """Get current virtual desktop index."""
        if sys.platform == "win32":
            return self._get_windows_desktop()
        else:
            return self._get_yabai_space()

    def _get_windows_desktop(self) -> int:
        """Get current Windows virtual desktop (requires pyvda or win11toast)."""
        try:
            import pyvda
            return pyvda.GetCurrentDesktopNumber()
        except ImportError:
            return 0  # Fallback: assume desktop 0

    def get_window_info(self, title: str = None) -> List[Dict]:
        """Get window position/size info — cross-platform."""
        if sys.platform == "win32":
            return self._get_windows_window_info(title)
        else:
            return self._get_yabai_window_info(title)

    def _get_windows_window_info(self, title=None) -> List[Dict]:
        """Get window info via win32gui."""
        import win32gui
        windows = []
        def cb(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            t = win32gui.GetWindowText(hwnd)
            if not t:
                return
            if title and title.lower() not in t.lower():
                return
            rect = win32gui.GetWindowRect(hwnd)
            windows.append({
                'id': hwnd,
                'title': t,
                'frame': {
                    'x': rect[0], 'y': rect[1],
                    'w': rect[2] - rect[0],
                    'h': rect[3] - rect[1],
                }
            })
        win32gui.EnumWindows(cb, None)
        return windows
```

**Install:** `pip install pyvda pywin32`

### 9.4 Guard all yabai imports

In every file that imports yabai:
```python
if sys.platform != "win32":
    from vision.yabai_space_detector import YabaiSpaceDetector
    YABAI_AVAILABLE = True
else:
    YABAI_AVAILABLE = False
    YabaiSpaceDetector = None
```

---

## 10. Phase 18 — System-Level & Unix Primitives

### 10.1 `fcntl` — File Locking

**Files affected:** 
- `backend/core/coding_council/edge_cases/memory_monitor.py`
- `backend/core/coding_council/framework/resource_coordinator.py`
- `backend/core/coding_council/advanced/atomic_locking.py`
- `backend/core/atomic_service_coordination.py`

**Pattern to apply everywhere:**
```python
import sys

if sys.platform == "win32":
    import msvcrt

    def lock_file(f) -> None:
        """Lock file on Windows using msvcrt."""
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)

    def unlock_file(f) -> None:
        """Unlock file on Windows."""
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def lock_file(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock_file(f) -> None:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

### 10.2 `resource` Module — Memory/CPU Limits

**Files affected:** `start_system.py`, `backend/core/advanced_ram_monitor.py`

```python
import sys

def get_memory_info() -> Dict[str, int]:
    """Get memory usage — cross-platform."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            'total': vm.total,
            'available': vm.available,
            'used': vm.used,
            'percent': vm.percent,
        }
    except Exception:
        return {'total': 0, 'available': 0, 'used': 0, 'percent': 0}

def set_memory_limit(max_bytes: int) -> bool:
    """Set memory limit — Unix only, skip on Windows."""
    if sys.platform == "win32":
        return False  # Windows uses Job Objects — complex, skip
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        return True
    except Exception:
        return False
```

### 10.3 `signal.SIGSTOP` — Process Suspension

**File:** `backend/core/gcp_hybrid_prime_router.py`

**macOS/Linux:**
```python
os.kill(pid, signal.SIGSTOP)
```

**Windows (using ctypes NtSuspendProcess):**
```python
import sys

def suspend_process(pid: int) -> bool:
    """Suspend process — cross-platform."""
    if sys.platform == "win32":
        import ctypes
        PROCESS_SUSPEND_RESUME = 0x0800
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if handle:
            result = ctypes.windll.ntdll.NtSuspendProcess(handle)
            ctypes.windll.kernel32.CloseHandle(handle)
            return result == 0
        return False
    else:
        import signal
        os.kill(pid, signal.SIGSTOP)
        return True

def resume_process(pid: int) -> bool:
    """Resume suspended process — cross-platform."""
    if sys.platform == "win32":
        import ctypes
        PROCESS_SUSPEND_RESUME = 0x0800
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
        if handle:
            result = ctypes.windll.ntdll.NtResumeProcess(handle)
            ctypes.windll.kernel32.CloseHandle(handle)
            return result == 0
        return False
    else:
        import signal
        os.kill(pid, signal.SIGCONT)
        return True
```

### 10.4 `vm_stat` — macOS Memory Stats

**File:** `backend/core/gcp_hybrid_prime_router.py`

```python
import sys

def get_memory_pressure() -> float:
    """Get memory pressure 0.0-1.0 — cross-platform."""
    try:
        import psutil
        return psutil.virtual_memory().percent / 100.0
    except Exception:
        if sys.platform == "win32":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        # ... other fields
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.dwMemoryLoad / 100.0
            except Exception:
                return 0.5
        return 0.5
```

---

## 11. Phase 19 — Audio System

### 11.1 PortAudio / PyAudio
PyAudio works on Windows. Issue: startup deadlock in AudioBus (fixed upstream).

### 11.2 `full_duplex_device.py`
**File:** `backend/audio/full_duplex_device.py`

macOS uses CoreAudio. On Windows use:
```python
import sys

def get_default_audio_device() -> Dict[str, Any]:
    """Get default audio input/output device — cross-platform."""
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        info = pa.get_default_output_device_info()
        pa.terminate()
        return {
            'name': info.get('name', 'Default'),
            'index': info.get('index', 0),
            'sample_rate': int(info.get('defaultSampleRate', 44100)),
        }
    except Exception:
        return {'name': 'Default', 'index': 0, 'sample_rate': 44100}
```

### 11.3 `async_tts_handler.py`

**File:** `backend/api/async_tts_handler.py`

This file uses the `say` command for audio file generation:
```python
# macOS only
cmd = ['say', '-v', voice, '-r', str(rate), '-o', output_file, '--data-format=LEF32@22050']
```

**Windows fix:**
```python
import sys

async def generate_tts_file(text: str, voice: str, rate: int, output_path: str) -> bool:
    """Generate TTS audio file — cross-platform."""
    if sys.platform == "win32":
        return await _generate_tts_edge(text, output_path)
    else:
        return await _generate_tts_say(text, voice, rate, output_path)

async def _generate_tts_edge(text: str, output_path: str) -> bool:
    """Generate TTS using edge-tts (Windows)."""
    try:
        import aiohttp.resolver as _ar
        import aiohttp.connector as _ac
        _ar.DefaultResolver = _ar.ThreadedResolver
        _ac.DefaultResolver = _ar.ThreadedResolver
        import edge_tts
        communicate = edge_tts.Communicate(text, voice="en-GB-RyanNeural")
        await communicate.save(output_path)
        return True
    except Exception as e:
        logger.warning(f"edge-tts file generation failed: {e}")
        return False
```

---

## 12. Phase 20 — Screen Lock / Unlock

### 12.1 macOS Screen Lock Detection
Currently: `screencapture` returns error, or osascript check

### 12.2 Windows Screen Lock Detection

**File:** `backend/context_intelligence/detectors/screen_lock_detector.py`

```python
import sys
import ctypes

def is_screen_locked() -> bool:
    """Detect if screen is locked — cross-platform."""
    if sys.platform == "win32":
        return _is_locked_windows()
    else:
        return _is_locked_macos()

def _is_locked_windows() -> bool:
    """
    Windows lock detection using multiple methods:
    1. Desktop session check via OpenDesktop
    2. LogonUI.exe process check
    """
    # Method 1: Try to switch to the default desktop
    # If locked, this call returns 0
    try:
        import ctypes
        DESKTOP_SWITCHDESKTOP = 0x0100
        hDesktop = ctypes.windll.user32.OpenDesktopW("Default", 0, False, DESKTOP_SWITCHDESKTOP)
        if hDesktop:
            result = not ctypes.windll.user32.SwitchDesktop(hDesktop)
            ctypes.windll.user32.CloseDesktop(hDesktop)
            if result:
                return True
    except Exception:
        pass

    # Method 2: Check for LogonUI.exe (lock screen process)
    try:
        import psutil
        for proc in psutil.process_iter(['name']):
            if proc.info['name'] and proc.info['name'].lower() == 'logonui.exe':
                return True
    except Exception:
        pass

    return False

def lock_screen() -> bool:
    """Lock screen — cross-platform."""
    if sys.platform == "win32":
        ctypes.windll.user32.LockWorkStation()
        return True
    else:
        import subprocess
        subprocess.run(['osascript', '-e',
            'tell application "System Events" to key code 12 using {control down, command down}'])
        return True
```

---

## 13. Windows API Reference Cheatsheet

### Quick Reference: macOS → Windows

| Need | macOS | Windows |
|------|-------|---------|
| Notify user | `osascript display notification` | `plyer.notification.notify()` |
| Lock screen | `osascript key code` | `ctypes.windll.user32.LockWorkStation()` |
| Mouse click | `cliclick c:x,y` | `pyautogui.click(x, y)` |
| Type text | `cliclick t:text` | `pyautogui.write(text)` |
| Hotkey | `osascript keystroke` | `pyautogui.hotkey(*keys)` |
| Screenshot | `screencapture -x file.png` | `mss.mss().grab(monitor)` |
| Get volume | `osascript get volume settings` | `pycaw.GetMasterVolumeLevelScalar()` |
| Set volume | `osascript set volume output volume` | `pycaw.SetMasterVolumeLevelScalar()` |
| Open file | `open file.txt` | `os.startfile('file.txt')` |
| Launch app | `open -a "Safari"` | `subprocess.Popen(['chrome.exe'])` |
| Quit app | `osascript quit application` | `psutil` kill by name |
| Sleep system | `pmset sleepnow` | `ctypes.windll.PowrProf.SetSuspendState(0,1,0)` |
| Prevent sleep | `caffeinate` | `SetThreadExecutionState(ES_CONTINUOUS\|ES_SYSTEM_REQUIRED)` |
| File lock | `fcntl.flock()` | `msvcrt.locking()` |
| Process suspend | `signal.SIGSTOP` | `NtSuspendProcess()` |
| Memory stats | `vm_stat` | `psutil.virtual_memory()` |
| Window title | `osascript get name of window` | `win32gui.GetForegroundWindow()` |
| Virtual desktop | `yabai -m query --spaces` | `pyvda.GetCurrentDesktopNumber()` |
| System tray | `AppKit NSStatusItem` | `pystray` |
| File dialog | `AppKit NSOpenPanel` | `tkinter.filedialog` |
| Clipboard copy | `pbcopy` | `pyperclip.copy()` or `win32clipboard` |
| Clipboard paste | `pbpaste` | `pyperclip.paste()` |
| TTS speak | `say -v Daniel` | `edge_tts.Communicate()` ✅ Done |
| Screen lock detect | `screencapture` error | `LogonUI.exe` process check |
| BLE proximity | CoreBluetooth | Windows BLE API / skip |
| Metal GPU | Metal framework | DirectML / skip |
| CoreML inference | CoreML | ONNX Runtime / skip |
| AirPlay | macOS exclusive | Skip (Windows has Miracast) |

### Windows Packages Required
```
pip install pywin32        # Win32 API access
pip install pyautogui      # Mouse/keyboard control  
pip install plyer          # Cross-platform notifications
pip install pycaw          # Windows audio (COM-based)
pip install comtypes       # COM interface for pycaw
pip install psutil         # Process/memory monitoring
pip install pyvda          # Virtual desktop detection
pip install pystray        # System tray
pip install pyperclip      # Clipboard cross-platform
pip install wmi            # Windows Management Instrumentation
pip install edge-tts       # Neural TTS ✅ Done
pip install mss            # Screen capture ✅ Done
```

---

## 14. Testing Each Phase

### Test Command Template
After each phase fix, run:
```powershell
cd C:\Users\nandk\Ironcliw

# Test backend import
python -c "from backend.main import app; print('OK')"

# Test platform adapter
python -c "from backend.platform_adapter import get_platform; p=get_platform(); print(p.__class__.__name__)"

# Run specific module test
python -m pytest backend/tests/ -k "windows" -v

# Full startup test (check logs for errors)
python start_system.py 2>&1 | Select-String -Pattern "ERROR|WARNING|CRITICAL" | Select-Object -First 20
```

### Test: Notifications
```python
# test_notifications.py
import asyncio
from backend.agi_os.notification_bridge import show_notification

async def test():
    ok = await show_notification("Ironcliw Test", "Windows notification working!", "NORMAL")
    print("Notification:", "OK" if ok else "FAILED")

asyncio.run(test())
```

### Test: Screen Control
```python
# test_screen.py
from backend.platform_adapter import get_platform
p = get_platform()

print("Volume:", p.get_volume())
p.set_volume(50)
print("Active window:", p.get_active_window_title())
windows = p.list_windows()
print("Open windows:", len(windows))
```

### Test: Ghost Hands / Automation
```python
# test_ghost_hands.py
from backend.platform_adapter import get_platform
p = get_platform()

print("Testing mouse click at 100,100...")
p.click(100, 100)

print("Testing type text...")
p.type_text("Hello Ironcliw")

print("Testing hotkey Ctrl+Z...")
p.hotkey('ctrl', 'z')
```

---

## 15. Dependency Matrix

### What Each Windows Package Replaces

| Windows Package | Replaces macOS | Used In |
|-----------------|----------------|---------|
| `pywin32` | AppKit, osascript | platform_adapter, window mgmt |
| `pyautogui` | cliclick, osascript mouse | ghost_hands, action_executors |
| `plyer` | osascript notifications | notification_bridge |
| `pycaw` | osascript volume | hardware_control |
| `comtypes` | macOS COM-free | pycaw dependency |
| `psutil` | `resource`, `vm_stat` | ram_monitor, process mgmt |
| `pyvda` | yabai spaces | space_detector |
| `pystray` | AppKit NSStatusItem | menu bar |
| `pyperclip` | pbcopy/pbpaste | clipboard ops |
| `wmi` | IOKit brightness | hardware_control |
| `edge-tts` ✅ | `say -v Daniel` | realtime_voice_communicator |
| `mss` ✅ | screencapture | vision, screenshot |
| `msvcrt` (stdlib) | fcntl | atomic_locking, file_locker |
| `ctypes` (stdlib) | SIGSTOP, CoreGraphics | process mgmt, gcp_router |

### Install All at Once
```powershell
pip install pywin32 pyautogui plyer pycaw comtypes psutil pyvda pystray pyperclip wmi
```

---

## Appendix: File-by-File Quick Fix Guide

### Critical Files (Fix Now)
1. `backend/agi_os/notification_bridge.py` — Phase 12
2. `backend/api/action_executors.py` — Phase 13
3. `backend/autonomy/hardware_control.py` — Phase 15
4. `backend/api/async_tts_handler.py` — Phase 19
5. `backend/context_intelligence/detectors/screen_lock_detector.py` — Phase 20
6. `backend/voice_unlock/ml_engine_registry.py` — ECAPA fast-fail

### Medium Priority
7. `backend/chatbots/claude_vision_chatbot.py` — screencapture → mss
8. `backend/autonomy/macos_integration.py` — entire file needs Windows equivalent
9. `backend/ghost_hands/background_actuator.py` — cliclick → pyautogui
10. `backend/ghost_hands/yabai_aware_actuator.py` — yabai → win32gui

### Low Priority (Stub/Skip)
11. `backend/voice_unlock/apple_watch_proximity.py` — skip (no Watch on Windows)
12. `backend/display/airplay_protocol.py` — skip (macOS exclusive)
13. `backend/macos_helper/` — entire directory — port to `windows_helper/`
14. `backend/system_control/macos_controller.py` → `windows_controller.py`

---

*Blueprint version 1.0 — Phase 11 Complete — Phase 12-20 Remaining*
*Generated: 2026-02-24 | Repo: nandkishorrathodk-art/Ironcliw-ai*
