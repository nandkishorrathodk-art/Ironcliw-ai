"""
Platform-Agnostic Screen Capture Module

Auto-detects the current platform and imports the appropriate
screen capture implementation.

Created: 2026-02-23
Purpose: Windows/Linux porting - Phase 3 (Screen Capture)

Usage:
    from backend.vision.platform_capture import create_capture, CaptureConfig
    
    # Auto-detect platform and create appropriate capture
    config = CaptureConfig(fps_target=30)
    capture = create_capture(config)
    
    await capture.start()
    frame = await capture.get_frame()
    await capture.stop()
"""

import logging
from typing import Optional

from .base_capture import (
    ScreenCaptureInterface,
    CaptureConfig,
    CaptureFrame,
    CaptureMethod,
    CaptureQuality,
    CaptureStats,
    CaptureError,
    CaptureNotSupportedError,
    CapturePermissionError,
    CaptureTimeoutError,
)

# Import platform detection
try:
    from backend.core.platform_abstraction import PlatformDetector, SupportedPlatform
except ImportError:
    # Fallback if not in backend context
    import sys
    from pathlib import Path
    core_path = Path(__file__).parent.parent.parent / "core"
    sys.path.insert(0, str(core_path))
    from platform_abstraction import PlatformDetector, SupportedPlatform


logger = logging.getLogger(__name__)


def create_capture(config: Optional[CaptureConfig] = None) -> ScreenCaptureInterface:
    """
    Factory function to create platform-specific screen capture.
    
    Auto-detects the current platform and instantiates the appropriate
    capture implementation.
    
    Args:
        config: Capture configuration (uses defaults if None)
    
    Returns:
        Platform-specific ScreenCaptureInterface implementation
    
    Raises:
        CaptureNotSupportedError: If no capture method available on this platform
    
    Example:
        >>> config = CaptureConfig(fps_target=30, quality=CaptureQuality.HIGH)
        >>> capture = create_capture(config)
        >>> await capture.start()
    """
    detector = PlatformDetector()
    platform = detector.get_platform()
    
    logger.info(f"Creating screen capture for platform: {platform.value}")
    
    if platform == SupportedPlatform.WINDOWS:
        from .windows_capture import WindowsScreenCapture
        return WindowsScreenCapture(config)
    
    elif platform == SupportedPlatform.LINUX:
        from .linux_capture import LinuxScreenCapture
        return LinuxScreenCapture(config)
    
    elif platform == SupportedPlatform.MACOS:
        from .macos_capture import MacOSScreenCapture
        return MacOSScreenCapture(config)
    
    else:
        raise CaptureNotSupportedError(
            f"Screen capture not supported on platform: {platform.value}"
        )


def get_available_capture_methods() -> list:
    """
    Get list of capture methods available on the current platform.
    
    Returns:
        List of CaptureMethod enum values
    
    Example:
        >>> methods = get_available_capture_methods()
        >>> print(f"Available: {[m.value for m in methods]}")
    """
    try:
        capture = create_capture()
        return capture.get_capture_methods()
    except Exception as e:
        logger.error(f"Failed to detect capture methods: {e}")
        return []


def is_capture_supported() -> bool:
    """
    Check if screen capture is supported on the current platform.
    
    Returns:
        bool: True if at least one capture method is available
    
    Example:
        >>> if is_capture_supported():
        ...     capture = create_capture()
        ... else:
        ...     print("Screen capture not supported")
    """
    try:
        methods = get_available_capture_methods()
        return len(methods) > 0
    except Exception:
        return False


# Export public API
__all__ = [
    # Factory function
    "create_capture",
    "get_available_capture_methods",
    "is_capture_supported",
    
    # Base classes and interfaces
    "ScreenCaptureInterface",
    
    # Configuration and data classes
    "CaptureConfig",
    "CaptureFrame",
    "CaptureStats",
    
    # Enums
    "CaptureMethod",
    "CaptureQuality",
    
    # Exceptions
    "CaptureError",
    "CaptureNotSupportedError",
    "CapturePermissionError",
    "CaptureTimeoutError",
]


# Platform-specific implementations (optional direct imports)
__all__ += [
    "WindowsScreenCapture",
    "LinuxScreenCapture",
    "MacOSScreenCapture",
]


# Lazy imports for platform-specific implementations
def __getattr__(name: str):
    """Lazy import platform-specific classes."""
    if name == "WindowsScreenCapture":
        from .windows_capture import WindowsScreenCapture
        return WindowsScreenCapture
    elif name == "LinuxScreenCapture":
        from .linux_capture import LinuxScreenCapture
        return LinuxScreenCapture
    elif name == "MacOSScreenCapture":
        from .macos_capture import MacOSScreenCapture
        return MacOSScreenCapture
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Log module initialization
detector = PlatformDetector()
logger.info(
    f"✅ Platform capture module initialized for {detector.get_platform().value}"
)
