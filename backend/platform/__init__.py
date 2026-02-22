"""
JARVIS Platform Abstraction Layer
═══════════════════════════════════════════════════════════════════════════════

Cross-platform abstraction layer for JARVIS AI Assistant.

This module provides a unified API for platform-specific functionality:
- System control (windows, volume, notifications)
- Audio I/O (microphone, speakers, recording)
- Vision (screen capture, monitoring)
- Authentication (voice, biometric, password)
- Permissions (microphone, camera, screen recording)
- Process management (start, stop, schedule)
- File watching (FSEvents, ReadDirectoryChangesW, inotify)
- Notifications (native system notifications)

Supported Platforms:
    - macOS (Darwin) - Apple Silicon & Intel
    - Windows 10/11 - x86_64, ARM64
    - Linux - Ubuntu, Debian, Fedora, Arch

Usage:
    from backend.platform import get_platform, PlatformFactory
    
    # Detect current platform
    platform = get_platform()  # 'macos', 'windows', or 'linux'
    
    # Create platform-specific implementations
    system_control = PlatformFactory.create_system_control()
    audio_engine = PlatformFactory.create_audio_engine()
    vision_capture = PlatformFactory.create_vision_capture()
    
    # Use uniform API regardless of platform
    windows = system_control.get_window_list()
    system_control.set_volume(0.5)
    
    frame = vision_capture.capture_screen()
    audio_engine.start_recording()

Architecture:
    backend/platform/
    ├── __init__.py           # This file - exports
    ├── base.py               # Abstract base classes
    ├── detector.py           # Platform detection
    ├── macos/                # macOS implementations
    │   ├── system_control.py
    │   ├── audio.py
    │   ├── vision.py
    │   └── ...
    ├── windows/              # Windows implementations
    │   ├── system_control.py
    │   ├── audio.py
    │   ├── vision.py
    │   └── ...
    └── linux/                # Linux implementations
        ├── system_control.py
        ├── audio.py
        ├── vision.py
        └── ...

Author: JARVIS System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

from .detector import (
    PlatformDetector,
    PlatformInfo,
    get_platform,
    get_platform_info,
    is_macos,
    is_windows,
    is_linux,
)

from .base import (
    # Base classes
    BaseSystemControl,
    BaseAudioEngine,
    BaseVisionCapture,
    BaseAuthentication,
    BasePermissions,
    BaseProcessManager,
    BaseFileWatcher,
    BaseNotifications,
    
    # Factory
    PlatformFactory,
    
    # Data structures
    WindowInfo,
    AudioDeviceInfo,
    ScreenCaptureFrame,
    AuthenticationResult,
    PermissionType,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Platform detection
    'get_platform',
    'get_platform_info',
    'is_macos',
    'is_windows',
    'is_linux',
    'PlatformDetector',
    'PlatformInfo',
    
    # Factory
    'PlatformFactory',
    
    # Base classes (for type hints and subclassing)
    'BaseSystemControl',
    'BaseAudioEngine',
    'BaseVisionCapture',
    'BaseAuthentication',
    'BasePermissions',
    'BaseProcessManager',
    'BaseFileWatcher',
    'BaseNotifications',
    
    # Data structures
    'WindowInfo',
    'AudioDeviceInfo',
    'ScreenCaptureFrame',
    'AuthenticationResult',
    'PermissionType',
]


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Auto-detect platform on module import
_current_platform = get_platform()

# Print platform info on first import (can be disabled via env var)
import os
if os.getenv('JARVIS_VERBOSE_PLATFORM', '').lower() in ('1', 'true', 'yes'):
    info = get_platform_info()
    print(f"[JARVIS Platform] Detected: {info.os_family} ({info.os_release})")
    print(f"[JARVIS Platform] Hardware: GPU={info.has_gpu}, NPU={info.has_npu}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_system_control() -> BaseSystemControl:
    """Create platform-specific system control instance"""
    return PlatformFactory.create_system_control()


def create_audio_engine() -> BaseAudioEngine:
    """Create platform-specific audio engine instance"""
    return PlatformFactory.create_audio_engine()


def create_vision_capture() -> BaseVisionCapture:
    """Create platform-specific vision capture instance"""
    return PlatformFactory.create_vision_capture()


def create_authentication() -> BaseAuthentication:
    """Create platform-specific authentication instance"""
    return PlatformFactory.create_authentication()


def create_permissions() -> BasePermissions:
    """Create platform-specific permissions manager instance"""
    return PlatformFactory.create_permissions()


def create_process_manager() -> BaseProcessManager:
    """Create platform-specific process manager instance"""
    return PlatformFactory.create_process_manager()


def create_file_watcher() -> BaseFileWatcher:
    """Create platform-specific file watcher instance"""
    return PlatformFactory.create_file_watcher()


def create_notifications() -> BaseNotifications:
    """Create platform-specific notifications instance"""
    return PlatformFactory.create_notifications()


# Add convenience functions to __all__
__all__.extend([
    'create_system_control',
    'create_audio_engine',
    'create_vision_capture',
    'create_authentication',
    'create_permissions',
    'create_process_manager',
    'create_file_watcher',
    'create_notifications',
])
