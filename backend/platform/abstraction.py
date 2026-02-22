"""
Platform Abstraction Layer - Base Interfaces
=============================================

Defines abstract base classes for all platform-specific operations.
Each platform (macOS, Windows, Linux) provides concrete implementations.
"""

import platform
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class PlatformInterface(ABC):
    """
    Base interface for platform-specific operations.
    
    All methods are async to support non-blocking I/O operations.
    Implementations should handle errors gracefully and provide
    meaningful error messages.
    """
    
    def __init__(self):
        self.os_name = platform.system()
        self.os_version = platform.version()
        self.architecture = platform.machine()
    
    # ========================================================================
    # SYSTEM INFORMATION
    # ========================================================================
    
    @abstractmethod
    async def get_idle_time(self) -> float:
        """
        Get system idle time in seconds.
        
        Returns:
            float: Seconds since last user input (mouse, keyboard)
        
        Raises:
            NotImplementedError: If platform doesn't support idle detection
        """
        pass
    
    @abstractmethod
    async def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            dict: System information including:
                - os_name: Operating system name
                - os_version: OS version string
                - hostname: Computer hostname
                - cpu_count: Number of CPU cores
                - memory_total: Total RAM in bytes
                - memory_available: Available RAM in bytes
                - disk_usage: Disk usage statistics
        """
        pass
    
    @abstractmethod
    async def get_battery_status(self) -> Optional[Dict[str, Any]]:
        """
        Get battery information (if available).
        
        Returns:
            dict or None: Battery info with keys:
                - percent: Battery percentage (0-100)
                - plugged: Whether AC adapter is connected
                - time_left: Estimated seconds remaining (or None)
            Returns None if no battery present.
        """
        pass
    
    # ========================================================================
    # WINDOW MANAGEMENT
    # ========================================================================
    
    @abstractmethod
    async def get_window_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all visible windows.
        
        Returns:
            list: List of window dictionaries with keys:
                - id: Window identifier
                - title: Window title
                - app_name: Application name
                - x, y, width, height: Window geometry
                - is_focused: Whether window has focus
        """
        pass
    
    @abstractmethod
    async def get_active_window(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently focused window.
        
        Returns:
            dict or None: Active window info (same structure as get_window_info)
        """
        pass
    
    @abstractmethod
    async def focus_window(self, window_id: Any) -> bool:
        """
        Focus/activate a specific window by ID.
        
        Args:
            window_id: Platform-specific window identifier
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    # ========================================================================
    # AUTOMATION (MOUSE & KEYBOARD)
    # ========================================================================
    
    @abstractmethod
    async def click_at(self, x: int, y: int, button: str = "left") -> bool:
        """
        Click at specific screen coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button ("left", "right", "middle")
        
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def double_click_at(self, x: int, y: int) -> bool:
        """
        Double-click at specific screen coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def type_text(self, text: str, interval: float = 0.0) -> bool:
        """
        Type text using keyboard automation.
        
        Args:
            text: Text to type
            interval: Delay between keystrokes in seconds
        
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def press_key(self, key: str) -> bool:
        """
        Press a keyboard key.
        
        Args:
            key: Key name (e.g., "enter", "tab", "ctrl", "alt")
        
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def hotkey(self, *keys: str) -> bool:
        """
        Press a combination of keys simultaneously.
        
        Args:
            *keys: Variable number of key names (e.g., "ctrl", "c")
        
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def get_mouse_position(self) -> Tuple[int, int]:
        """
        Get current mouse cursor position.
        
        Returns:
            tuple: (x, y) coordinates
        """
        pass
    
    # ========================================================================
    # SCREEN CAPTURE
    # ========================================================================
    
    @abstractmethod
    async def capture_screen(
        self,
        monitor: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> bytes:
        """
        Capture screenshot as PNG bytes.
        
        Args:
            monitor: Monitor number (None = primary, 0 = all monitors)
            region: Optional (x, y, width, height) to capture specific area
        
        Returns:
            bytes: PNG-encoded image data
        """
        pass
    
    @abstractmethod
    async def get_monitors(self) -> List[Dict[str, Any]]:
        """
        Get information about all connected monitors.
        
        Returns:
            list: Monitor info dictionaries with keys:
                - id: Monitor identifier
                - x, y: Position
                - width, height: Resolution
                - is_primary: Whether this is the primary display
        """
        pass
    
    # ========================================================================
    # AUDIO
    # ========================================================================
    
    @abstractmethod
    async def get_audio_devices(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get available audio input and output devices.
        
        Returns:
            dict: Dictionary with keys:
                - inputs: List of input device dicts
                - outputs: List of output device dicts
            Each device dict contains:
                - id: Device identifier
                - name: Device name
                - channels: Number of channels
                - sample_rate: Default sample rate
        """
        pass
    
    @abstractmethod
    async def play_notification_sound(self) -> bool:
        """
        Play system notification sound.
        
        Returns:
            bool: True if successful
        """
        pass
    
    # ========================================================================
    # NOTIFICATIONS
    # ========================================================================
    
    @abstractmethod
    async def show_notification(
        self,
        title: str,
        message: str,
        duration: int = 5,
        icon: Optional[str] = None,
    ) -> bool:
        """
        Show system notification.
        
        Args:
            title: Notification title
            message: Notification message
            duration: Duration in seconds
            icon: Optional path to icon image
        
        Returns:
            bool: True if notification was shown
        """
        pass
    
    # ========================================================================
    # FILE SYSTEM OPERATIONS
    # ========================================================================
    
    @abstractmethod
    async def open_file(self, path: Path) -> bool:
        """
        Open file with default application.
        
        Args:
            path: Path to file
        
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def open_url(self, url: str) -> bool:
        """
        Open URL in default browser.
        
        Args:
            url: URL to open
        
        Returns:
            bool: True if successful
        """
        pass
    
    # ========================================================================
    # PLATFORM-SPECIFIC CAPABILITIES
    # ========================================================================
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get dictionary of platform capabilities.
        
        Returns:
            dict: Capability flags:
                - has_gui: GUI/windowing system available
                - has_audio: Audio system available
                - has_notifications: Notification support
                - has_automation: UI automation support
                - has_screen_capture: Screen capture support
                - has_battery: Battery status available
        """
        return {
            "has_gui": True,
            "has_audio": True,
            "has_notifications": True,
            "has_automation": True,
            "has_screen_capture": True,
            "has_battery": False,
        }


def get_platform() -> PlatformInterface:
    """
    Factory function to get platform-specific implementation.
    
    Returns:
        PlatformInterface: Concrete platform implementation
    
    Raises:
        NotImplementedError: If current platform is not supported
    """
    system = platform.system()
    
    if system == "Darwin":
        from .macos_platform import MacOSPlatform
        return MacOSPlatform()
    elif system == "Windows":
        from .windows_platform import WindowsPlatform
        return WindowsPlatform()
    elif system == "Linux":
        from .linux_platform import LinuxPlatform
        return LinuxPlatform()
    else:
        raise NotImplementedError(
            f"Platform '{system}' is not supported. "
            f"Supported platforms: Darwin (macOS), Windows, Linux"
        )
