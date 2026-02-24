"""
Ghost Hands Platform Abstraction Layer
═══════════════════════════════════════

Cross-platform automation abstraction for Ghost Hands.
Automatically selects the appropriate backend (macOS or Windows) based on the platform.

This module provides a unified interface that works on both macOS and Windows:
    - macOS: Uses yabai + Quartz + AppleScript
    - Windows: Uses DWM + SendInput + PowerShell

Usage:
    from backend.ghost_hands.platform_automation import get_automation_engine
    
    engine = await get_automation_engine()
    
    # Window management (works on both platforms)
    windows = await engine.get_all_windows()
    await engine.focus_window(window_id)
    
    # Mouse/keyboard (works on both platforms)
    await engine.click(100, 200)
    await engine.type_text("Hello")

Author: JARVIS System
Version: 1.0.0 (Windows Port - Phase 8)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Detect platform
try:
    from backend.platform_adapter import get_platform, is_windows, is_macos
    PLATFORM_DETECTION_AVAILABLE = True
except ImportError:
    import platform
    PLATFORM_DETECTION_AVAILABLE = False
    
    def get_platform() -> str:
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        elif system == "windows":
            return "windows"
        elif system == "linux":
            return "linux"
        return "unknown"
    
    def is_windows() -> bool:
        return get_platform() == "windows"
    
    def is_macos() -> bool:
        return get_platform() == "macos"


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT AUTOMATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BaseAutomationEngine(ABC):
    """Abstract base class for platform-specific automation engines"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the automation engine"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    # ─────────────────────────────────────────────────────────────────────────────
    # WINDOW MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    async def get_all_windows(self) -> List[Any]:
        """Get all windows across all spaces/virtual desktops"""
        pass
    
    @abstractmethod
    async def get_window(self, window_id: int) -> Optional[Any]:
        """Get window info by ID"""
        pass
    
    @abstractmethod
    async def get_windows_for_app(self, app_name: str) -> List[Any]:
        """Get all windows for an app"""
        pass
    
    @abstractmethod
    async def get_focused_window(self) -> Optional[Any]:
        """Get currently focused window"""
        pass
    
    @abstractmethod
    async def focus_window(self, window_id: int) -> bool:
        """Focus a window"""
        pass
    
    @abstractmethod
    async def minimize_window(self, window_id: int) -> bool:
        """Minimize a window"""
        pass
    
    @abstractmethod
    async def maximize_window(self, window_id: int) -> bool:
        """Maximize a window"""
        pass
    
    @abstractmethod
    async def close_window(self, window_id: int) -> bool:
        """Close a window"""
        pass
    
    # ─────────────────────────────────────────────────────────────────────────────
    # FOCUS PRESERVATION
    # ─────────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    async def save_focus(self) -> Dict[str, Any]:
        """Save current focus state"""
        pass
    
    @abstractmethod
    async def restore_focus(self) -> bool:
        """Restore saved focus state"""
        pass
    
    # ─────────────────────────────────────────────────────────────────────────────
    # MOUSE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    async def move_to(self, x: int, y: int, duration: float = 0.2) -> bool:
        """Move mouse to coordinates"""
        pass
    
    @abstractmethod
    async def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        """Click at coordinates"""
        pass
    
    @abstractmethod
    async def double_click(self, x: int, y: int) -> bool:
        """Double-click at coordinates"""
        pass
    
    @abstractmethod
    async def right_click(self, x: int, y: int) -> bool:
        """Right-click at coordinates"""
        pass
    
    @abstractmethod
    async def drag_to(self, x: int, y: int, duration: float = 0.2) -> bool:
        """Drag mouse to coordinates"""
        pass
    
    @abstractmethod
    async def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Scroll at current position or specified coordinates"""
        pass
    
    # ─────────────────────────────────────────────────────────────────────────────
    # KEYBOARD OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    async def type_text(self, text: str, interval: float = 0.02) -> bool:
        """Type text with interval between characters"""
        pass
    
    @abstractmethod
    async def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> bool:
        """Press a key with optional modifiers"""
        pass
    
    @abstractmethod
    async def hotkey(self, *keys) -> bool:
        """Press a key combination"""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOWS IMPLEMENTATION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class WindowsAutomationAdapter(BaseAutomationEngine):
    """Adapter for Windows automation engine"""
    
    def __init__(self):
        from .windows_automation import WindowsAutomationEngine
        self._engine = WindowsAutomationEngine()
    
    async def initialize(self) -> bool:
        return await self._engine.initialize()
    
    async def cleanup(self) -> None:
        await self._engine.cleanup()
    
    # Window management (delegate to window_manager)
    async def get_all_windows(self) -> List[Any]:
        return await self._engine.window_manager.get_all_windows()
    
    async def get_window(self, window_id: int) -> Optional[Any]:
        return await self._engine.window_manager.get_window(window_id)
    
    async def get_windows_for_app(self, app_name: str) -> List[Any]:
        return await self._engine.window_manager.get_windows_for_app(app_name)
    
    async def get_focused_window(self) -> Optional[Any]:
        return await self._engine.window_manager.get_focused_window()
    
    async def focus_window(self, window_id: int) -> bool:
        return await self._engine.window_manager.focus_window(window_id)
    
    async def minimize_window(self, window_id: int) -> bool:
        return await self._engine.window_manager.minimize_window(window_id)
    
    async def maximize_window(self, window_id: int) -> bool:
        return await self._engine.window_manager.maximize_window(window_id)
    
    async def close_window(self, window_id: int) -> bool:
        return await self._engine.window_manager.close_window(window_id)
    
    # Focus preservation (delegate to focus_guard)
    async def save_focus(self) -> Dict[str, Any]:
        return await self._engine.focus_guard.save_focus()
    
    async def restore_focus(self) -> bool:
        return await self._engine.focus_guard.restore_focus()
    
    # Mouse operations (delegate to mouse_keyboard)
    async def move_to(self, x: int, y: int, duration: float = 0.2) -> bool:
        return await self._engine.mouse_keyboard.move_to(x, y, duration)
    
    async def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        return await self._engine.mouse_keyboard.click(x, y, button, clicks)
    
    async def double_click(self, x: int, y: int) -> bool:
        return await self._engine.mouse_keyboard.double_click(x, y)
    
    async def right_click(self, x: int, y: int) -> bool:
        return await self._engine.mouse_keyboard.right_click(x, y)
    
    async def drag_to(self, x: int, y: int, duration: float = 0.2) -> bool:
        return await self._engine.mouse_keyboard.drag_to(x, y, duration)
    
    async def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        return await self._engine.mouse_keyboard.scroll(clicks, x, y)
    
    # Keyboard operations (delegate to mouse_keyboard)
    async def type_text(self, text: str, interval: float = 0.02) -> bool:
        return await self._engine.mouse_keyboard.type_text(text, interval)
    
    async def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> bool:
        return await self._engine.mouse_keyboard.press_key(key, modifiers)
    
    async def hotkey(self, *keys) -> bool:
        return await self._engine.mouse_keyboard.hotkey(*keys)


# ═══════════════════════════════════════════════════════════════════════════════
# MACOS IMPLEMENTATION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class MacOSAutomationAdapter(BaseAutomationEngine):
    """Adapter for macOS yabai automation engine"""
    
    def __init__(self):
        # Lazy import to avoid errors on Windows
        from .yabai_aware_actuator import YabaiAwareActuator, YabaiActuatorConfig
        from .background_actuator import BackgroundActuator, ActuatorConfig
        
        self._yabai_actuator = YabaiAwareActuator(YabaiActuatorConfig())
        self._background_actuator = BackgroundActuator(ActuatorConfig())
    
    async def initialize(self) -> bool:
        success = True
        success &= await self._yabai_actuator.initialize()
        success &= await self._background_actuator.initialize()
        return success
    
    async def cleanup(self) -> None:
        await self._yabai_actuator.cleanup()
        await self._background_actuator.cleanup()
    
    # Window management (delegate to yabai)
    async def get_all_windows(self) -> List[Any]:
        return await self._yabai_actuator.window_resolver.get_all_windows()
    
    async def get_window(self, window_id: int) -> Optional[Any]:
        return await self._yabai_actuator.window_resolver.get_window(window_id)
    
    async def get_windows_for_app(self, app_name: str) -> List[Any]:
        return await self._yabai_actuator.window_resolver.get_windows_for_app(app_name)
    
    async def get_focused_window(self) -> Optional[Any]:
        return await self._yabai_actuator.window_resolver.get_focused_window()
    
    async def focus_window(self, window_id: int) -> bool:
        # Use yabai to focus
        result = await self._yabai_actuator.window_resolver._run_yabai_command([
            "-m", "window", "--focus", str(window_id)
        ])
        return result is not None
    
    async def minimize_window(self, window_id: int) -> bool:
        result = await self._yabai_actuator.window_resolver._run_yabai_command([
            "-m", "window", str(window_id), "--minimize"
        ])
        return result is not None
    
    async def maximize_window(self, window_id: int) -> bool:
        result = await self._yabai_actuator.window_resolver._run_yabai_command([
            "-m", "window", str(window_id), "--toggle", "zoom-fullscreen"
        ])
        return result is not None
    
    async def close_window(self, window_id: int) -> bool:
        result = await self._yabai_actuator.window_resolver._run_yabai_command([
            "-m", "window", str(window_id), "--close"
        ])
        return result is not None
    
    # Focus preservation (delegate to background actuator)
    async def save_focus(self) -> Dict[str, Any]:
        return await self._background_actuator.focus_guard.save_focus()
    
    async def restore_focus(self) -> bool:
        return await self._background_actuator.focus_guard.restore_focus()
    
    # Mouse/keyboard operations (use background actuator's CGEventBackend)
    # NOTE: These are simplified - the actual implementation would use
    # the appropriate backend from BackgroundActuator
    
    async def move_to(self, x: int, y: int, duration: float = 0.2) -> bool:
        import pyautogui
        pyautogui.moveTo(x, y, duration=duration)
        return True
    
    async def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        import pyautogui
        pyautogui.click(x, y, clicks=clicks, button=button)
        return True
    
    async def double_click(self, x: int, y: int) -> bool:
        import pyautogui
        pyautogui.doubleClick(x, y)
        return True
    
    async def right_click(self, x: int, y: int) -> bool:
        import pyautogui
        pyautogui.rightClick(x, y)
        return True
    
    async def drag_to(self, x: int, y: int, duration: float = 0.2) -> bool:
        import pyautogui
        pyautogui.dragTo(x, y, duration=duration)
        return True
    
    async def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        import pyautogui
        if x is not None and y is not None:
            pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.scroll(clicks)
        return True
    
    async def type_text(self, text: str, interval: float = 0.02) -> bool:
        import pyautogui
        pyautogui.typewrite(text, interval=interval)
        return True
    
    async def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> bool:
        import pyautogui
        if modifiers:
            pyautogui.hotkey(*modifiers, key)
        else:
            pyautogui.press(key)
        return True
    
    async def hotkey(self, *keys) -> bool:
        import pyautogui
        pyautogui.hotkey(*keys)
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_automation_engine: Optional[BaseAutomationEngine] = None


async def get_automation_engine(force_platform: Optional[str] = None) -> BaseAutomationEngine:
    """
    Get the platform-appropriate automation engine.
    
    Args:
        force_platform: Override platform detection ("windows", "macos", or None)
    
    Returns:
        Platform-specific automation engine
    """
    global _automation_engine
    
    if _automation_engine is not None:
        return _automation_engine
    
    # Determine platform
    platform = force_platform or get_platform()
    
    # Create appropriate engine
    if platform == "windows":
        logger.info("[AUTOMATION] Using Windows automation engine")
        _automation_engine = WindowsAutomationAdapter()
    elif platform == "macos":
        logger.info("[AUTOMATION] Using macOS automation engine")
        _automation_engine = MacOSAutomationAdapter()
    else:
        raise RuntimeError(f"Unsupported platform for Ghost Hands: {platform}")
    
    # Initialize
    success = await _automation_engine.initialize()
    if not success:
        raise RuntimeError(f"Failed to initialize {platform} automation engine")
    
    return _automation_engine


def reset_automation_engine() -> None:
    """Reset the global automation engine (for testing)"""
    global _automation_engine
    _automation_engine = None

