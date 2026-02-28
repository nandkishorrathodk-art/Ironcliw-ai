"""
Ironcliw Windows Platform Implementation
═══════════════════════════════════════════════════════════════════════════════

Windows 10/11 platform-specific implementations using C# native DLLs.

This module provides Windows implementations of all Ironcliw platform abstractions:
    - WindowsSystemControl (window management, volume, notifications)
    - WindowsAudioEngine (WASAPI audio I/O)
    - WindowsVisionCapture (Windows.Graphics.Capture)
    - WindowsAuthentication (bypass mode for MVP)
    - WindowsPermissions (UAC handling)
    - WindowsProcessManager (Task Scheduler integration)
    - WindowsFileWatcher (ReadDirectoryChangesW)

Architecture:
    Python (backend/platform/windows/)
        ↓ pythonnet (clr)
    C# DLLs (backend/windows_native/)
        ↓ P/Invoke
    Win32 API / WinRT

Author: Ironcliw System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

from .system_control import WindowsSystemControl
from .audio import WindowsAudioEngine
from .vision import WindowsVisionCapture
from .auth import WindowsAuthentication
from .permissions import WindowsPermissions
from .process_manager import WindowsProcessManager
from .file_watcher import WindowsFileWatcher

__all__ = [
    'WindowsSystemControl',
    'WindowsAudioEngine',
    'WindowsVisionCapture',
    'WindowsAuthentication',
    'WindowsPermissions',
    'WindowsProcessManager',
    'WindowsFileWatcher',
]
