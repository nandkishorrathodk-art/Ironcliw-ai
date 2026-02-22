"""
Platform Abstraction Layer - Platform Detection Module

This module provides cross-platform detection capabilities for JARVIS,
enabling support for macOS, Windows, and Linux.

Created: 2026-02-22
Purpose: Windows/Linux porting - Phase 1 (PAL)
"""

import platform
import sys
from enum import Enum
from typing import Dict, Any, Optional
import os


class SupportedPlatform(Enum):
    """Enum representing platforms supported by JARVIS."""
    MACOS = "macos"
    WINDOWS = "windows"
    LINUX = "linux"
    UNKNOWN = "unknown"


class PlatformDetector:
    """
    Singleton class for detecting and querying the current platform.
    
    This class provides a centralized way to detect the operating system
    and query platform-specific information throughout JARVIS.
    
    Usage:
        detector = PlatformDetector()
        if detector.is_windows():
            # Windows-specific code
        elif detector.is_macos():
            # macOS-specific code
        elif detector.is_linux():
            # Linux-specific code
    """
    
    _instance: Optional['PlatformDetector'] = None
    _platform: Optional[SupportedPlatform] = None
    _platform_info: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize platform detection on first creation."""
        if self._platform is None:
            self._detect_platform()
    
    def _detect_platform(self) -> None:
        """
        Detect the current platform and gather system information.
        
        This method examines sys.platform and platform.system() to determine
        the operating system and collects additional metadata.
        """
        system = platform.system().lower()
        
        # Detect platform
        if system == "darwin":
            self._platform = SupportedPlatform.MACOS
        elif system == "windows":
            self._platform = SupportedPlatform.WINDOWS
        elif system == "linux":
            self._platform = SupportedPlatform.LINUX
        else:
            self._platform = SupportedPlatform.UNKNOWN
        
        # Gather platform info
        self._platform_info = {
            "system": system,
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
            "architecture": platform.architecture()[0],
        }
        
        # Add platform-specific details
        if self._platform == SupportedPlatform.MACOS:
            self._platform_info["macos_version"] = platform.mac_ver()[0]
        elif self._platform == SupportedPlatform.WINDOWS:
            self._platform_info["windows_version"] = platform.win32_ver()[0]
            self._platform_info["windows_edition"] = platform.win32_edition()
        elif self._platform == SupportedPlatform.LINUX:
            try:
                import distro
                self._platform_info["linux_distro"] = distro.name()
                self._platform_info["linux_version"] = distro.version()
            except ImportError:
                # distro not available, use platform.linux_distribution (deprecated)
                try:
                    dist_info = platform.linux_distribution()
                    self._platform_info["linux_distro"] = dist_info[0]
                    self._platform_info["linux_version"] = dist_info[1]
                except AttributeError:
                    # platform.linux_distribution removed in Python 3.8+
                    self._platform_info["linux_distro"] = "unknown"
                    self._platform_info["linux_version"] = "unknown"
    
    def get_platform(self) -> SupportedPlatform:
        """
        Get the detected platform.
        
        Returns:
            SupportedPlatform: The current operating system platform
        """
        return self._platform
    
    def get_platform_name(self) -> str:
        """
        Get the platform name as a string.
        
        Returns:
            str: Platform name (e.g., "macos", "windows", "linux")
        """
        return self._platform.value
    
    def get_platform_info(self) -> Dict[str, Any]:
        """
        Get detailed platform information.
        
        Returns:
            dict: Dictionary containing platform metadata
        """
        return self._platform_info.copy()
    
    def is_macos(self) -> bool:
        """
        Check if running on macOS.
        
        Returns:
            bool: True if running on macOS, False otherwise
        """
        return self._platform == SupportedPlatform.MACOS
    
    def is_windows(self) -> bool:
        """
        Check if running on Windows.
        
        Returns:
            bool: True if running on Windows, False otherwise
        """
        return self._platform == SupportedPlatform.WINDOWS
    
    def is_linux(self) -> bool:
        """
        Check if running on Linux.
        
        Returns:
            bool: True if running on Linux, False otherwise
        """
        return self._platform == SupportedPlatform.LINUX
    
    def is_unix_like(self) -> bool:
        """
        Check if running on a Unix-like system (macOS or Linux).
        
        Returns:
            bool: True if running on macOS or Linux, False otherwise
        """
        return self._platform in (SupportedPlatform.MACOS, SupportedPlatform.LINUX)
    
    def is_supported(self) -> bool:
        """
        Check if the current platform is supported by JARVIS.
        
        Returns:
            bool: True if platform is supported, False otherwise
        """
        return self._platform != SupportedPlatform.UNKNOWN
    
    def get_config_dir(self) -> str:
        """
        Get the platform-specific configuration directory for JARVIS.
        
        Returns:
            str: Path to the configuration directory
        """
        if self.is_macos() or self.is_linux():
            # Unix-like: ~/.jarvis
            return os.path.expanduser("~/.jarvis")
        elif self.is_windows():
            # Windows: %APPDATA%\JARVIS
            appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
            return os.path.join(appdata, "JARVIS")
        else:
            # Fallback to current directory
            return os.path.join(os.getcwd(), ".jarvis")
    
    def get_log_dir(self) -> str:
        """
        Get the platform-specific log directory for JARVIS.
        
        Returns:
            str: Path to the log directory
        """
        if self.is_macos() or self.is_linux():
            # Unix-like: ~/.jarvis/logs
            return os.path.join(self.get_config_dir(), "logs")
        elif self.is_windows():
            # Windows: %APPDATA%\JARVIS\logs
            return os.path.join(self.get_config_dir(), "logs")
        else:
            return os.path.join(self.get_config_dir(), "logs")
    
    def get_data_dir(self) -> str:
        """
        Get the platform-specific data directory for JARVIS.
        
        Returns:
            str: Path to the data directory
        """
        if self.is_macos() or self.is_linux():
            # Unix-like: ~/.jarvis/data
            return os.path.join(self.get_config_dir(), "data")
        elif self.is_windows():
            # Windows: %APPDATA%\JARVIS\data
            return os.path.join(self.get_config_dir(), "data")
        else:
            return os.path.join(self.get_config_dir(), "data")
    
    def get_cache_dir(self) -> str:
        """
        Get the platform-specific cache directory for JARVIS.
        
        Returns:
            str: Path to the cache directory
        """
        if self.is_macos():
            # macOS: ~/Library/Caches/JARVIS
            return os.path.expanduser("~/Library/Caches/JARVIS")
        elif self.is_linux():
            # Linux: ~/.cache/jarvis
            xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            return os.path.join(xdg_cache, "jarvis")
        elif self.is_windows():
            # Windows: %LOCALAPPDATA%\JARVIS\Cache
            localappdata = os.environ.get("LOCALAPPDATA", os.path.join(os.environ.get("APPDATA", ""), "..", "Local"))
            return os.path.join(localappdata, "JARVIS", "Cache")
        else:
            return os.path.join(self.get_config_dir(), "cache")
    
    def __repr__(self) -> str:
        """String representation of the platform detector."""
        return f"PlatformDetector(platform={self.get_platform_name()})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        info = self.get_platform_info()
        return (
            f"JARVIS Platform: {self.get_platform_name()}\n"
            f"System: {info.get('system', 'unknown')}\n"
            f"Release: {info.get('release', 'unknown')}\n"
            f"Architecture: {info.get('architecture', 'unknown')}"
        )


# Convenience functions for quick platform checks
def get_platform() -> SupportedPlatform:
    """Get the current platform (convenience function)."""
    return PlatformDetector().get_platform()


def is_macos() -> bool:
    """Check if running on macOS (convenience function)."""
    return PlatformDetector().is_macos()


def is_windows() -> bool:
    """Check if running on Windows (convenience function)."""
    return PlatformDetector().is_windows()


def is_linux() -> bool:
    """Check if running on Linux (convenience function)."""
    return PlatformDetector().is_linux()


def is_unix_like() -> bool:
    """Check if running on Unix-like system (convenience function)."""
    return PlatformDetector().is_unix_like()


def is_supported() -> bool:
    """Check if platform is supported (convenience function)."""
    return PlatformDetector().is_supported()


def get_platform_info() -> Dict[str, Any]:
    """Get platform information (convenience function)."""
    return PlatformDetector().get_platform_info()
