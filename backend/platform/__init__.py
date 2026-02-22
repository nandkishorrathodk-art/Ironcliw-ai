"""
JARVIS Platform Abstraction Layer (PAL)
========================================

Cross-platform abstraction layer providing consistent APIs across macOS, Windows, and Linux.

Usage:
    from backend.platform import get_platform
    
    platform = get_platform()
    idle_time = await platform.get_idle_time()
    await platform.click_at(100, 200)
"""

from .abstraction import PlatformInterface, get_platform

__all__ = ["PlatformInterface", "get_platform"]
