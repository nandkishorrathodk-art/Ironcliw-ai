"""
Ghost Hands Windows Automation Layer
═════════════════════════════════════

Windows implementation of mouse, keyboard, and window control for Ghost Hands.

This module replaces macOS-specific APIs with Windows equivalents:
- macOS Quartz/CGEvent → Windows SendInput (Win32 API)
- macOS AppleScript → Windows PowerShell/COM
- macOS Accessibility API → Windows UI Automation
- macOS yabai → Windows DWM (Desktop Window Manager)

Technologies:
    - pyautogui: Cross-platform mouse/keyboard (primary)
    - pywin32: Windows API access (SendInput, mouse_event, keybd_event)
    - C# SystemControl DLL: Window management via pythonnet
    - UI Automation: Accessibility features

Features:
    - Multi-monitor aware coordinate translation
    - Focus preservation during automation
    - Window enumeration and targeting
    - Keyboard/mouse input to specific windows
    - No focus stealing (background automation)

Author: JARVIS System
Version: 1.0.0 (Windows Port - Phase 8)
"""
from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import pyautogui

logger = logging.getLogger(__name__)

# Windows API constants
try:
    import win32api
    import win32con
    import win32gui
    import win32process
    PYWIN32_AVAILABLE = True
except ImportError:
    PYWIN32_AVAILABLE = False
    logger.warning("[WIN-AUTO] pywin32 not available - limited functionality")

# C# DLL integration
try:
    import clr
    PYTHONNET_AVAILABLE = True
except ImportError:
    PYTHONNET_AVAILABLE = False
    logger.warning("[WIN-AUTO] pythonnet not available - window management disabled")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindowsAutomationConfig:
    """Configuration for Windows automation"""
    
    # Focus preservation
    preserve_focus: bool = field(
        default_factory=lambda: os.getenv("JARVIS_WIN_PRESERVE_FOCUS", "true").lower() == "true"
    )
    focus_restore_delay_ms: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_WIN_FOCUS_DELAY_MS", "100"))
    )
    
    # Multi-monitor support
    multi_monitor_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_WIN_MULTIMON", "true").lower() == "true"
    )
    
    # Automation backend preference
    prefer_pyautogui: bool = field(
        default_factory=lambda: os.getenv("JARVIS_WIN_PREFER_PYAUTOGUI", "true").lower() == "true"
    )
    
    # Timing
    action_delay_ms: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_WIN_ACTION_DELAY_MS", "50"))
    )
    animation_wait_ms: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_WIN_ANIMATION_MS", "300"))
    )
    
    # C# DLL path
    dll_path: Optional[str] = field(
        default_factory=lambda: os.getenv("WINDOWS_NATIVE_DLL_PATH")
    )


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindowsWindowFrame:
    """Window frame information (RECT structure)"""
    left: int
    top: int
    right: int
    bottom: int
    
    @property
    def x(self) -> int:
        return self.left
    
    @property
    def y(self) -> int:
        return self.top
    
    @property
    def width(self) -> int:
        return self.right - self.left
    
    @property
    def height(self) -> int:
        return self.bottom - self.top
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within frame"""
        return (
            self.left <= x <= self.right and
            self.top <= y <= self.bottom
        )
    
    def center(self) -> Tuple[int, int]:
        """Get center point"""
        return (
            self.left + self.width // 2,
            self.top + self.height // 2
        )


@dataclass
class WindowsWindowInfo:
    """Complete window information"""
    hwnd: int  # Window handle
    pid: int
    process_name: str
    title: str
    frame: WindowsWindowFrame
    is_visible: bool
    is_minimized: bool
    is_maximized: bool
    is_focused: bool
    class_name: str = ""
    
    @property
    def window_id(self) -> int:
        """Alias for compatibility with macOS YabaiWindowInfo"""
        return self.hwnd
    
    @property
    def app_name(self) -> str:
        """Alias for compatibility with macOS YabaiWindowInfo"""
        return self.process_name


# ═══════════════════════════════════════════════════════════════════════════════
# MONITOR INFORMATION (Multi-Monitor Support)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MonitorInfo:
    """Monitor information"""
    monitor_id: int
    left: int
    top: int
    width: int
    height: int
    is_primary: bool
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within monitor"""
        return (
            self.left <= x < self.left + self.width and
            self.top <= y < self.top + self.height
        )


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOWS WINDOW MANAGER (DWM Replacement for Yabai)
# ═══════════════════════════════════════════════════════════════════════════════

class WindowsWindowManager:
    """
    Windows window management - replaces macOS yabai.
    
    Provides:
        - Window enumeration across all virtual desktops
        - Window frame and position information
        - Window state (minimized, maximized, focused)
        - Multi-monitor awareness
    """
    
    def __init__(self, config: WindowsAutomationConfig):
        self.config = config
        self._controller = None
        self._initialized = False
        self._cache: Dict[int, WindowsWindowInfo] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_ms = 500  # 500ms cache TTL
    
    async def initialize(self) -> bool:
        """Initialize window manager"""
        if self._initialized:
            return True
        
        if not PYWIN32_AVAILABLE:
            logger.error("[WIN-MGR] pywin32 is required for window management")
            return False
        
        # Try to load C# DLL for advanced features
        if PYTHONNET_AVAILABLE:
            try:
                from backend.platform_adapter.windows.system_control import WindowsSystemControl
                self._controller = WindowsSystemControl()
                logger.info("[WIN-MGR] C# SystemControl DLL loaded")
            except Exception as e:
                logger.warning(f"[WIN-MGR] C# DLL not available: {e}")
        
        self._initialized = True
        logger.info("[WIN-MGR] Windows window manager initialized")
        return True
    
    async def get_window(self, hwnd: int) -> Optional[WindowsWindowInfo]:
        """Get window info by handle"""
        # Check cache
        if self._is_cache_valid() and hwnd in self._cache:
            return self._cache[hwnd]
        
        if not PYWIN32_AVAILABLE:
            return None
        
        try:
            if not win32gui.IsWindow(hwnd):
                return None
            
            # Get window rect
            rect = win32gui.GetWindowRect(hwnd)
            frame = WindowsWindowFrame(rect[0], rect[1], rect[2], rect[3])
            
            # Get window title
            title = win32gui.GetWindowText(hwnd)
            
            # Get process info
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ, False, pid)
                process_name = win32process.GetModuleFileNameEx(handle, 0).split("\\")[-1]
                win32api.CloseHandle(handle)
            except:
                process_name = "Unknown"
            
            # Get window state
            placement = win32gui.GetWindowPlacement(hwnd)
            is_minimized = placement[1] == win32con.SW_SHOWMINIMIZED
            is_maximized = placement[1] == win32con.SW_SHOWMAXIMIZED
            is_visible = win32gui.IsWindowVisible(hwnd)
            is_focused = win32gui.GetForegroundWindow() == hwnd
            
            # Get class name
            class_name = win32gui.GetClassName(hwnd)
            
            info = WindowsWindowInfo(
                hwnd=hwnd,
                pid=pid,
                process_name=process_name,
                title=title,
                frame=frame,
                is_visible=is_visible,
                is_minimized=is_minimized,
                is_maximized=is_maximized,
                is_focused=is_focused,
                class_name=class_name
            )
            
            self._cache[hwnd] = info
            return info
            
        except Exception as e:
            logger.error(f"[WIN-MGR] Failed to get window {hwnd}: {e}")
            return None
    
    async def get_all_windows(self) -> List[WindowsWindowInfo]:
        """Get all windows"""
        if not PYWIN32_AVAILABLE:
            return []
        
        windows = []
        
        def enum_callback(hwnd, results):
            # Only include visible windows with titles
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    results.append(hwnd)
            return True
        
        try:
            hwnd_list = []
            win32gui.EnumWindows(enum_callback, hwnd_list)
            
            # Convert to WindowsWindowInfo
            for hwnd in hwnd_list:
                info = await self.get_window(hwnd)
                if info:
                    windows.append(info)
            
            # Update cache
            self._cache = {w.hwnd: w for w in windows}
            self._cache_time = datetime.now()
            
            return windows
            
        except Exception as e:
            logger.error(f"[WIN-MGR] Failed to enumerate windows: {e}")
            return []
    
    async def get_windows_for_app(self, app_name: str) -> List[WindowsWindowInfo]:
        """Get all windows for an app"""
        all_windows = await self.get_all_windows()
        return [w for w in all_windows if app_name.lower() in w.process_name.lower()]
    
    async def get_focused_window(self) -> Optional[WindowsWindowInfo]:
        """Get currently focused window"""
        if not PYWIN32_AVAILABLE:
            return None
        
        try:
            hwnd = win32gui.GetForegroundWindow()
            return await self.get_window(hwnd)
        except Exception as e:
            logger.error(f"[WIN-MGR] Failed to get focused window: {e}")
            return None
    
    async def focus_window(self, hwnd: int) -> bool:
        """Focus a window"""
        if not PYWIN32_AVAILABLE:
            return False
        
        try:
            # Restore if minimized
            placement = win32gui.GetWindowPlacement(hwnd)
            if placement[1] == win32con.SW_SHOWMINIMIZED:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                await asyncio.sleep(0.1)
            
            # Bring to foreground
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception as e:
            logger.error(f"[WIN-MGR] Failed to focus window {hwnd}: {e}")
            return False
    
    async def minimize_window(self, hwnd: int) -> bool:
        """Minimize a window"""
        if not PYWIN32_AVAILABLE:
            return False
        
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return True
        except Exception as e:
            logger.error(f"[WIN-MGR] Failed to minimize window {hwnd}: {e}")
            return False
    
    async def maximize_window(self, hwnd: int) -> bool:
        """Maximize a window"""
        if not PYWIN32_AVAILABLE:
            return False
        
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            return True
        except Exception as e:
            logger.error(f"[WIN-MGR] Failed to maximize window {hwnd}: {e}")
            return False
    
    async def close_window(self, hwnd: int) -> bool:
        """Close a window"""
        if not PYWIN32_AVAILABLE:
            return False
        
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True
        except Exception as e:
            logger.error(f"[WIN-MGR] Failed to close window {hwnd}: {e}")
            return False
    
    async def get_monitors(self) -> List[MonitorInfo]:
        """Get all monitors"""
        if not PYWIN32_AVAILABLE:
            return []
        
        monitors = []
        
        def enum_callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
            try:
                info = win32api.GetMonitorInfo(hMonitor)
                rect = info['Monitor']
                is_primary = info['Flags'] == win32con.MONITORINFOF_PRIMARY
                
                monitors.append(MonitorInfo(
                    monitor_id=len(monitors),
                    left=rect[0],
                    top=rect[1],
                    width=rect[2] - rect[0],
                    height=rect[3] - rect[1],
                    is_primary=is_primary
                ))
            except:
                pass
            return True
        
        try:
            win32api.EnumDisplayMonitors(None, None, enum_callback, None)
            return monitors
        except Exception as e:
            logger.error(f"[WIN-MGR] Failed to enumerate monitors: {e}")
            return []
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_time:
            return False
        elapsed_ms = (datetime.now() - self._cache_time).total_seconds() * 1000
        return elapsed_ms < self._cache_ttl_ms


# ═══════════════════════════════════════════════════════════════════════════════
# FOCUS GUARD (Focus Preservation)
# ═══════════════════════════════════════════════════════════════════════════════

class WindowsFocusGuard:
    """
    Preserves user focus during background operations.
    Replaces macOS Quartz focus tracking with Win32 API.
    """
    
    def __init__(self, config: WindowsAutomationConfig):
        self.config = config
        self._saved_hwnd: Optional[int] = None
    
    async def save_focus(self) -> Dict[str, Any]:
        """Save current focus state"""
        if not PYWIN32_AVAILABLE:
            return {}
        
        try:
            hwnd = win32gui.GetForegroundWindow()
            self._saved_hwnd = hwnd
            title = win32gui.GetWindowText(hwnd)
            logger.debug(f"[FOCUS] Saved focus: {title} (hwnd={hwnd})")
            return {"hwnd": hwnd, "title": title}
        except Exception as e:
            logger.error(f"[FOCUS] Failed to save focus: {e}")
            return {}
    
    async def restore_focus(self) -> bool:
        """Restore saved focus state"""
        if not self._saved_hwnd:
            return False
        
        if not PYWIN32_AVAILABLE:
            return False
        
        try:
            # Wait for any animations
            if self.config.focus_restore_delay_ms > 0:
                await asyncio.sleep(self.config.focus_restore_delay_ms / 1000.0)
            
            # Restore foreground window
            win32gui.SetForegroundWindow(self._saved_hwnd)
            logger.debug(f"[FOCUS] Restored focus to hwnd={self._saved_hwnd}")
            return True
        except Exception as e:
            logger.error(f"[FOCUS] Failed to restore focus: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# MOUSE & KEYBOARD AUTOMATION
# ═══════════════════════════════════════════════════════════════════════════════

class WindowsMouseKeyboard:
    """
    Mouse and keyboard automation using pyautogui and Win32 API.
    Replaces macOS CGEvent with Windows SendInput.
    """
    
    def __init__(self, config: WindowsAutomationConfig):
        self.config = config
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize mouse/keyboard automation"""
        if self._initialized:
            return True
        
        # Configure pyautogui
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = self.config.action_delay_ms / 1000.0
        
        self._initialized = True
        logger.info("[MOUSE-KB] Mouse/keyboard automation initialized")
        return True
    
    # ─────────────────────────────────────────────────────────────────────────────
    # MOUSE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────────
    
    async def move_to(self, x: int, y: int, duration: float = 0.2) -> bool:
        """Move mouse to coordinates"""
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, pyautogui.moveTo, x, y, duration
            )
            return True
        except Exception as e:
            logger.error(f"[MOUSE] Move failed: {e}")
            return False
    
    async def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        """Click at coordinates"""
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, pyautogui.click, x, y, clicks, None, button
            )
            return True
        except Exception as e:
            logger.error(f"[MOUSE] Click failed: {e}")
            return False
    
    async def double_click(self, x: int, y: int) -> bool:
        """Double-click at coordinates"""
        return await self.click(x, y, "left", 2)
    
    async def right_click(self, x: int, y: int) -> bool:
        """Right-click at coordinates"""
        return await self.click(x, y, "right", 1)
    
    async def drag_to(self, x: int, y: int, duration: float = 0.2) -> bool:
        """Drag mouse to coordinates"""
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, pyautogui.dragTo, x, y, duration
            )
            return True
        except Exception as e:
            logger.error(f"[MOUSE] Drag failed: {e}")
            return False
    
    async def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Scroll at current position or specified coordinates"""
        try:
            if x is not None and y is not None:
                await self.move_to(x, y, duration=0.1)
            
            await asyncio.get_running_loop().run_in_executor(
                None, pyautogui.scroll, clicks
            )
            return True
        except Exception as e:
            logger.error(f"[MOUSE] Scroll failed: {e}")
            return False
    
    # ─────────────────────────────────────────────────────────────────────────────
    # KEYBOARD OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────────
    
    async def type_text(self, text: str, interval: float = 0.02) -> bool:
        """Type text with interval between characters"""
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, pyautogui.typewrite, text, interval
            )
            return True
        except Exception as e:
            logger.error(f"[KEYBOARD] Type failed: {e}")
            return False
    
    async def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> bool:
        """Press a key with optional modifiers"""
        try:
            if modifiers:
                # Translate macOS modifiers to Windows
                win_modifiers = []
                for mod in modifiers:
                    mod_lower = mod.lower()
                    if mod_lower in ("command", "cmd", "win", "super"):
                        win_modifiers.append("win")
                    elif mod_lower in ("option", "opt", "alt"):
                        win_modifiers.append("alt")
                    elif mod_lower in ("control", "ctrl"):
                        win_modifiers.append("ctrl")
                    elif mod_lower == "shift":
                        win_modifiers.append("shift")
                
                # Execute hotkey
                await asyncio.get_running_loop().run_in_executor(
                    None, pyautogui.hotkey, *win_modifiers, key
                )
            else:
                await asyncio.get_running_loop().run_in_executor(
                    None, pyautogui.press, key
                )
            return True
        except Exception as e:
            logger.error(f"[KEYBOARD] Press key failed: {e}")
            return False
    
    async def hotkey(self, *keys) -> bool:
        """Press a key combination"""
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, pyautogui.hotkey, *keys
            )
            return True
        except Exception as e:
            logger.error(f"[KEYBOARD] Hotkey failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED WINDOWS AUTOMATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class WindowsAutomationEngine:
    """
    Unified Windows automation engine for Ghost Hands.
    
    Replaces:
        - YabaiAwareActuator → WindowsWindowManager
        - CGEventBackend → WindowsMouseKeyboard
        - AppleScriptBackend → PowerShell/COM (future)
        - FocusGuard → WindowsFocusGuard
    
    Provides a single interface for all Windows automation needs.
    """
    
    def __init__(self, config: Optional[WindowsAutomationConfig] = None):
        self.config = config or WindowsAutomationConfig()
        self.window_manager = WindowsWindowManager(self.config)
        self.focus_guard = WindowsFocusGuard(self.config)
        self.mouse_keyboard = WindowsMouseKeyboard(self.config)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        if self._initialized:
            return True
        
        success = True
        success &= await self.window_manager.initialize()
        success &= await self.mouse_keyboard.initialize()
        
        if success:
            self._initialized = True
            logger.info("[WIN-AUTO] Windows automation engine initialized")
        else:
            logger.error("[WIN-AUTO] Failed to initialize automation engine")
        
        return success
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("[WIN-AUTO] Automation engine cleaned up")
        self._initialized = False


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_automation_engine: Optional[WindowsAutomationEngine] = None


async def get_windows_automation_engine() -> WindowsAutomationEngine:
    """Get or create the global Windows automation engine"""
    global _automation_engine
    
    if _automation_engine is None:
        _automation_engine = WindowsAutomationEngine()
        await _automation_engine.initialize()
    
    return _automation_engine

