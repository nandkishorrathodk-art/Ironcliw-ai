"""
Platform Abstraction Layer - Display Module

This module provides cross-platform display and screen abstraction,
enabling JARVIS to detect monitors, enumerate virtual desktops,
and prepare for screen capture across different operating systems.

Created: 2026-02-22
Purpose: Windows/Linux porting - Phase 1 (PAL)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from backend.core.platform_abstraction import PlatformDetector, SupportedPlatform


logger = logging.getLogger(__name__)


@dataclass
class DisplayInfo:
    """Information about a display/monitor."""
    id: str
    name: str
    width: int
    height: int
    x: int
    y: int
    is_primary: bool
    scale_factor: float = 1.0
    refresh_rate: Optional[int] = None


@dataclass
class VirtualDesktopInfo:
    """Information about a virtual desktop/space."""
    id: str
    name: str
    is_current: bool
    displays: List[str]  # List of display IDs


class DisplayInterface(ABC):
    """
    Abstract base class for platform-specific display operations.
    
    All platform implementations must inherit from this class and
    implement the required abstract methods.
    """
    
    @abstractmethod
    def get_displays(self) -> List[DisplayInfo]:
        """
        Get information about all connected displays.
        
        Returns:
            list: List of DisplayInfo objects
        """
        pass
    
    @abstractmethod
    def get_primary_display(self) -> Optional[DisplayInfo]:
        """
        Get information about the primary display.
        
        Returns:
            DisplayInfo | None: Primary display info or None if not found
        """
        pass
    
    @abstractmethod
    def get_virtual_desktops(self) -> List[VirtualDesktopInfo]:
        """
        Get information about virtual desktops/spaces.
        
        Returns:
            list: List of VirtualDesktopInfo objects
        """
        pass
    
    @abstractmethod
    def get_current_virtual_desktop(self) -> Optional[VirtualDesktopInfo]:
        """
        Get information about the current virtual desktop.
        
        Returns:
            VirtualDesktopInfo | None: Current desktop info or None if not available
        """
        pass
    
    @abstractmethod
    def get_total_screen_size(self) -> Tuple[int, int]:
        """
        Get the total screen size across all displays.
        
        Returns:
            tuple: (width, height) in pixels
        """
        pass
    
    @abstractmethod
    def supports_multi_monitor(self) -> bool:
        """
        Check if multi-monitor operations are supported.
        
        Returns:
            bool: True if multi-monitor is supported
        """
        pass
    
    @abstractmethod
    def supports_virtual_desktops(self) -> bool:
        """
        Check if virtual desktop enumeration is supported.
        
        Returns:
            bool: True if virtual desktops are supported
        """
        pass


class MacOSDisplay(DisplayInterface):
    """macOS-specific display operations."""
    
    def __init__(self):
        """Initialize macOS display interface."""
        self._displays_cache: Optional[List[DisplayInfo]] = None
        self._cache_valid = False
    
    def get_displays(self) -> List[DisplayInfo]:
        """
        Get all displays using macOS APIs.
        
        Note: This is a placeholder that will integrate with existing
        Swift-based screen capture code in backend/vision/swift_capture/.
        For Phase 1, we provide the interface structure.
        """
        if self._cache_valid and self._displays_cache:
            return self._displays_cache
        
        # TODO: Integrate with existing Swift display detection
        # For now, use a basic implementation with pyobjc or subprocess
        logger.warning("macOS display detection using fallback method")
        
        try:
            import subprocess
            # Use system_profiler to get display info
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                # Parse display data (simplified)
                displays = []
                # TODO: Parse actual display data from system_profiler
                # For now, return minimal info
                displays.append(DisplayInfo(
                    id="0",
                    name="Main Display",
                    width=1920,
                    height=1080,
                    x=0,
                    y=0,
                    is_primary=True,
                    scale_factor=1.0,
                ))
                self._displays_cache = displays
                self._cache_valid = True
                return displays
        except Exception as e:
            logger.error(f"Failed to get macOS displays: {e}")
        
        # Fallback: return single default display
        return [DisplayInfo(
            id="0",
            name="Display",
            width=1920,
            height=1080,
            x=0,
            y=0,
            is_primary=True,
        )]
    
    def get_primary_display(self) -> Optional[DisplayInfo]:
        """Get primary display on macOS."""
        displays = self.get_displays()
        for display in displays:
            if display.is_primary:
                return display
        return displays[0] if displays else None
    
    def get_virtual_desktops(self) -> List[VirtualDesktopInfo]:
        """
        Get macOS Spaces (virtual desktops).
        
        Note: Requires accessibility permissions and integration with
        existing backend/display/space_manager.py.
        """
        # TODO: Integrate with existing space_manager.py
        logger.warning("macOS virtual desktop enumeration not yet implemented")
        return []
    
    def get_current_virtual_desktop(self) -> Optional[VirtualDesktopInfo]:
        """Get current macOS Space."""
        # TODO: Integrate with space_manager.py
        return None
    
    def get_total_screen_size(self) -> Tuple[int, int]:
        """Get total screen size across all displays."""
        displays = self.get_displays()
        if not displays:
            return (1920, 1080)
        
        max_x = max(d.x + d.width for d in displays)
        max_y = max(d.y + d.height for d in displays)
        return (max_x, max_y)
    
    def supports_multi_monitor(self) -> bool:
        """macOS supports multi-monitor."""
        return True
    
    def supports_virtual_desktops(self) -> bool:
        """macOS supports Spaces (virtual desktops)."""
        return True


class WindowsDisplay(DisplayInterface):
    """Windows-specific display operations."""
    
    def __init__(self):
        """Initialize Windows display interface."""
        self._displays_cache: Optional[List[DisplayInfo]] = None
        self._cache_valid = False
    
    def get_displays(self) -> List[DisplayInfo]:
        """Get all displays using Windows APIs."""
        if self._cache_valid and self._displays_cache:
            return self._displays_cache
        
        displays = []
        
        try:
            # Try using win32api (pywin32)
            import win32api
            import win32con
            
            monitor_enum = []
            
            def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
                monitor_enum.append({
                    'handle': hMonitor,
                    'rect': lprcMonitor,
                })
                return True
            
            win32api.EnumDisplayMonitors(None, None, callback, 0)
            
            for idx, monitor in enumerate(monitor_enum):
                rect = monitor['rect']
                info = win32api.GetMonitorInfo(monitor['handle'])
                
                is_primary = (info['Flags'] & win32con.MONITORINFOF_PRIMARY) != 0
                
                displays.append(DisplayInfo(
                    id=str(idx),
                    name=info.get('Device', f'Monitor {idx + 1}'),
                    width=rect[2] - rect[0],
                    height=rect[3] - rect[1],
                    x=rect[0],
                    y=rect[1],
                    is_primary=is_primary,
                ))
            
            self._displays_cache = displays
            self._cache_valid = True
            return displays
            
        except ImportError:
            logger.warning("pywin32 not available, using fallback display detection")
        except Exception as e:
            logger.error(f"Failed to get Windows displays: {e}")
        
        # Fallback: return single default display
        return [DisplayInfo(
            id="0",
            name="Display 1",
            width=1920,
            height=1080,
            x=0,
            y=0,
            is_primary=True,
        )]
    
    def get_primary_display(self) -> Optional[DisplayInfo]:
        """Get primary display on Windows."""
        displays = self.get_displays()
        for display in displays:
            if display.is_primary:
                return display
        return displays[0] if displays else None
    
    def get_virtual_desktops(self) -> List[VirtualDesktopInfo]:
        """
        Get Windows virtual desktops.
        
        Note: Windows 10+ supports virtual desktops, but API access
        requires COM interfaces or third-party libraries.
        """
        logger.warning("Windows virtual desktop enumeration not yet implemented")
        return []
    
    def get_current_virtual_desktop(self) -> Optional[VirtualDesktopInfo]:
        """Get current Windows virtual desktop."""
        return None
    
    def get_total_screen_size(self) -> Tuple[int, int]:
        """Get total screen size across all displays."""
        displays = self.get_displays()
        if not displays:
            return (1920, 1080)
        
        max_x = max(d.x + d.width for d in displays)
        max_y = max(d.y + d.height for d in displays)
        return (max_x, max_y)
    
    def supports_multi_monitor(self) -> bool:
        """Windows supports multi-monitor."""
        return True
    
    def supports_virtual_desktops(self) -> bool:
        """Windows 10+ supports virtual desktops, but enumeration is complex."""
        return False  # Set to True when implemented


class LinuxDisplay(DisplayInterface):
    """Linux-specific display operations."""
    
    def __init__(self):
        """Initialize Linux display interface."""
        self._displays_cache: Optional[List[DisplayInfo]] = None
        self._cache_valid = False
    
    def get_displays(self) -> List[DisplayInfo]:
        """Get all displays using X11 or Wayland."""
        if self._cache_valid and self._displays_cache:
            return self._displays_cache
        
        displays = []
        
        try:
            # Try using xrandr for X11
            import subprocess
            result = subprocess.run(
                ["xrandr", "--query"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0:
                # Parse xrandr output
                lines = result.stdout.split('\n')
                idx = 0
                for line in lines:
                    if ' connected' in line:
                        parts = line.split()
                        name = parts[0]
                        is_primary = 'primary' in parts
                        
                        # Try to parse resolution
                        width, height = 1920, 1080
                        x, y = 0, 0
                        for part in parts:
                            if 'x' in part and '+' in part:
                                # Format: 1920x1080+0+0
                                res_part = part.split('+')
                                if len(res_part) >= 3:
                                    w_h = res_part[0].split('x')
                                    if len(w_h) == 2:
                                        width = int(w_h[0])
                                        height = int(w_h[1])
                                        x = int(res_part[1])
                                        y = int(res_part[2])
                                break
                        
                        displays.append(DisplayInfo(
                            id=str(idx),
                            name=name,
                            width=width,
                            height=height,
                            x=x,
                            y=y,
                            is_primary=is_primary,
                        ))
                        idx += 1
                
                if displays:
                    self._displays_cache = displays
                    self._cache_valid = True
                    return displays
        except FileNotFoundError:
            logger.warning("xrandr not found, trying alternative methods")
        except Exception as e:
            logger.error(f"Failed to get Linux displays via xrandr: {e}")
        
        # Fallback: return single default display
        return [DisplayInfo(
            id="0",
            name="Display",
            width=1920,
            height=1080,
            x=0,
            y=0,
            is_primary=True,
        )]
    
    def get_primary_display(self) -> Optional[DisplayInfo]:
        """Get primary display on Linux."""
        displays = self.get_displays()
        for display in displays:
            if display.is_primary:
                return display
        return displays[0] if displays else None
    
    def get_virtual_desktops(self) -> List[VirtualDesktopInfo]:
        """
        Get Linux virtual desktops (workspaces).
        
        Note: Implementation varies by desktop environment
        (GNOME, KDE, i3, etc.). Requires desktop-specific APIs.
        """
        logger.warning("Linux virtual desktop enumeration not yet implemented")
        return []
    
    def get_current_virtual_desktop(self) -> Optional[VirtualDesktopInfo]:
        """Get current Linux workspace."""
        return None
    
    def get_total_screen_size(self) -> Tuple[int, int]:
        """Get total screen size across all displays."""
        displays = self.get_displays()
        if not displays:
            return (1920, 1080)
        
        max_x = max(d.x + d.width for d in displays)
        max_y = max(d.y + d.height for d in displays)
        return (max_x, max_y)
    
    def supports_multi_monitor(self) -> bool:
        """Linux supports multi-monitor."""
        return True
    
    def supports_virtual_desktops(self) -> bool:
        """Linux supports virtual desktops, but enumeration is desktop-specific."""
        return False  # Set to True when implemented


class DisplayFactory:
    """
    Factory class to create platform-specific display instances.
    
    This factory automatically detects the current platform and returns
    the appropriate DisplayInterface implementation.
    """
    
    _instance: Optional[DisplayInterface] = None
    
    @classmethod
    def get_instance(cls) -> DisplayInterface:
        """
        Get the platform-specific display instance (singleton).
        
        Returns:
            DisplayInterface: Platform-specific implementation
        """
        if cls._instance is None:
            detector = PlatformDetector()
            platform = detector.get_platform()
            
            if platform == SupportedPlatform.MACOS:
                cls._instance = MacOSDisplay()
                logger.info("Display interface: macOS implementation")
            elif platform == SupportedPlatform.WINDOWS:
                cls._instance = WindowsDisplay()
                logger.info("Display interface: Windows implementation")
            elif platform == SupportedPlatform.LINUX:
                cls._instance = LinuxDisplay()
                logger.info("Display interface: Linux implementation")
            else:
                logger.warning("Unknown platform, using macOS display interface")
                cls._instance = MacOSDisplay()
        
        return cls._instance


# Convenience function to get display interface
def get_display_interface() -> DisplayInterface:
    """Get the platform-specific display interface."""
    return DisplayFactory.get_instance()
