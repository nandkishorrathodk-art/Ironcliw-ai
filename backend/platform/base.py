"""
JARVIS Platform Abstraction Layer - Base Classes
═══════════════════════════════════════════════════════════════════════════════

This module defines the abstract base classes for all platform-specific implementations.
Each platform (macOS, Windows, Linux) must implement these interfaces.

Architecture:
    BaseSystemControl    - Window management, system operations
    BaseAudioEngine      - Audio I/O and processing
    BaseVisionCapture    - Screen capture and analysis
    BaseAuthentication   - Biometric and password authentication
    BasePermissions      - Permission management (TCC, UAC, etc.)
    BaseProcessManager   - Process lifecycle management
    BaseFileWatcher      - File system monitoring
    BaseNotifications    - System notifications

Author: JARVIS System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindowInfo:
    """Information about a window"""
    window_id: int
    title: str
    app_name: str
    bounds: Tuple[int, int, int, int]  # (x, y, width, height)
    is_minimized: bool
    is_maximized: bool
    is_focused: bool
    process_id: int
    
@dataclass
class AudioDeviceInfo:
    """Information about an audio device"""
    device_id: str
    name: str
    is_input: bool
    is_default: bool
    sample_rate: int
    channels: int
    
@dataclass
class ScreenCaptureFrame:
    """A captured screen frame"""
    image_data: bytes  # Raw image data (PNG or JPEG)
    width: int
    height: int
    timestamp: float
    monitor_id: int
    format: str  # 'png', 'jpeg', 'rgb', 'bgr'
    
@dataclass
class AuthenticationResult:
    """Result of an authentication attempt"""
    success: bool
    method: str  # 'voice', 'password', 'biometric', 'bypass'
    confidence: float  # 0.0-1.0
    message: str
    user_id: Optional[str] = None
    
class PermissionType(Enum):
    """Types of permissions"""
    MICROPHONE = "microphone"
    CAMERA = "camera"
    SCREEN_RECORDING = "screen_recording"
    ACCESSIBILITY = "accessibility"
    AUTOMATION = "automation"
    NOTIFICATIONS = "notifications"


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class BaseSystemControl(ABC):
    """Abstract base class for system control operations"""
    
    @abstractmethod
    def get_window_list(self) -> List[WindowInfo]:
        """Get list of all windows"""
        pass
    
    @abstractmethod
    def focus_window(self, window_id: int) -> bool:
        """Bring window to front and focus it"""
        pass
    
    @abstractmethod
    def minimize_window(self, window_id: int) -> bool:
        """Minimize a window"""
        pass
    
    @abstractmethod
    def maximize_window(self, window_id: int) -> bool:
        """Maximize a window"""
        pass
    
    @abstractmethod
    def close_window(self, window_id: int) -> bool:
        """Close a window"""
        pass
    
    @abstractmethod
    def get_active_window(self) -> Optional[WindowInfo]:
        """Get currently focused window"""
        pass
    
    @abstractmethod
    def set_volume(self, level: float) -> bool:
        """Set system volume (0.0-1.0)"""
        pass
    
    @abstractmethod
    def get_volume(self) -> float:
        """Get system volume (0.0-1.0)"""
        pass
    
    @abstractmethod
    def show_notification(self, title: str, message: str, icon: Optional[str] = None) -> bool:
        """Show system notification"""
        pass
    
    @abstractmethod
    def execute_command(self, command: str, args: List[str] = None) -> Tuple[int, str, str]:
        """Execute shell command (returncode, stdout, stderr)"""
        pass
    
    @abstractmethod
    def get_display_count(self) -> int:
        """Get number of connected displays"""
        pass
    
    @abstractmethod
    def get_display_info(self) -> List[Dict[str, Any]]:
        """Get information about all displays"""
        pass


class BaseAudioEngine(ABC):
    """Abstract base class for audio processing"""
    
    @abstractmethod
    def list_devices(self, input_only: bool = False) -> List[AudioDeviceInfo]:
        """List available audio devices"""
        pass
    
    @abstractmethod
    def get_default_input_device(self) -> Optional[AudioDeviceInfo]:
        """Get default microphone device"""
        pass
    
    @abstractmethod
    def get_default_output_device(self) -> Optional[AudioDeviceInfo]:
        """Get default speaker device"""
        pass
    
    @abstractmethod
    def start_recording(self, device_id: Optional[str] = None, 
                       sample_rate: int = 16000, 
                       channels: int = 1,
                       callback: Optional[Callable] = None) -> bool:
        """Start audio recording"""
        pass
    
    @abstractmethod
    def stop_recording(self) -> Optional[bytes]:
        """Stop recording and return audio data"""
        pass
    
    @abstractmethod
    def play_audio(self, audio_data: bytes, sample_rate: int = 16000) -> bool:
        """Play audio data"""
        pass
    
    @abstractmethod
    def is_recording(self) -> bool:
        """Check if currently recording"""
        pass


class BaseVisionCapture(ABC):
    """Abstract base class for screen capture"""
    
    @abstractmethod
    def capture_screen(self, monitor_id: int = 0) -> Optional[ScreenCaptureFrame]:
        """Capture screenshot from specified monitor"""
        pass
    
    @abstractmethod
    def capture_all_screens(self) -> List[ScreenCaptureFrame]:
        """Capture all monitors at once"""
        pass
    
    @abstractmethod
    def start_continuous_capture(self, 
                                 fps: int = 15, 
                                 monitor_id: int = 0,
                                 callback: Optional[Callable] = None) -> bool:
        """Start continuous screen capture"""
        pass
    
    @abstractmethod
    def stop_continuous_capture(self) -> bool:
        """Stop continuous capture"""
        pass
    
    @abstractmethod
    def is_capturing(self) -> bool:
        """Check if currently capturing"""
        pass
    
    @abstractmethod
    def get_monitor_layout(self) -> List[Dict[str, Any]]:
        """Get layout information for all monitors"""
        pass


class BaseAuthentication(ABC):
    """Abstract base class for authentication"""
    
    @abstractmethod
    def authenticate_voice(self, audio_data: bytes, speaker_id: str) -> AuthenticationResult:
        """Authenticate user via voice biometrics"""
        pass
    
    @abstractmethod
    def authenticate_password(self, password: str) -> AuthenticationResult:
        """Authenticate user via password"""
        pass
    
    @abstractmethod
    def authenticate_biometric(self) -> AuthenticationResult:
        """Authenticate user via platform biometrics (Windows Hello, Touch ID)"""
        pass
    
    @abstractmethod
    def enroll_voice(self, audio_samples: List[bytes], speaker_id: str) -> bool:
        """Enroll new voice profile"""
        pass
    
    @abstractmethod
    def is_enrolled(self, speaker_id: str) -> bool:
        """Check if speaker is enrolled"""
        pass
    
    @abstractmethod
    def bypass_authentication(self) -> AuthenticationResult:
        """Bypass authentication (dev mode only)"""
        pass


class BasePermissions(ABC):
    """Abstract base class for permission management"""
    
    @abstractmethod
    def check_permission(self, permission: PermissionType) -> bool:
        """Check if permission is granted"""
        pass
    
    @abstractmethod
    def request_permission(self, permission: PermissionType) -> bool:
        """Request permission from user"""
        pass
    
    @abstractmethod
    def has_all_required_permissions(self) -> bool:
        """Check if all required permissions are granted"""
        pass
    
    @abstractmethod
    def get_missing_permissions(self) -> List[PermissionType]:
        """Get list of missing permissions"""
        pass
    
    @abstractmethod
    def open_permission_settings(self, permission: Optional[PermissionType] = None) -> bool:
        """Open system permission settings"""
        pass


class BaseProcessManager(ABC):
    """Abstract base class for process management"""
    
    @abstractmethod
    def start_process(self, 
                     command: str, 
                     args: List[str] = None,
                     env: Dict[str, str] = None,
                     background: bool = False) -> int:
        """Start a new process, returns PID"""
        pass
    
    @abstractmethod
    def stop_process(self, pid: int, graceful: bool = True) -> bool:
        """Stop a running process"""
        pass
    
    @abstractmethod
    def is_process_running(self, pid: int) -> bool:
        """Check if process is running"""
        pass
    
    @abstractmethod
    def get_process_info(self, pid: int) -> Optional[Dict[str, Any]]:
        """Get information about a process"""
        pass
    
    @abstractmethod
    def list_processes(self, filter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all processes, optionally filtered by name"""
        pass
    
    @abstractmethod
    def schedule_startup(self, command: str, name: str) -> bool:
        """Schedule process to start on system boot (launchd/Task Scheduler)"""
        pass
    
    @abstractmethod
    def unschedule_startup(self, name: str) -> bool:
        """Remove from startup schedule"""
        pass


class BaseFileWatcher(ABC):
    """Abstract base class for file system monitoring"""
    
    @abstractmethod
    def watch_directory(self, 
                       path: Path, 
                       callback: Callable[[str, str], None],
                       recursive: bool = True,
                       patterns: Optional[List[str]] = None) -> str:
        """
        Start watching a directory for changes
        Returns watch_id for later removal
        callback(event_type, file_path) where event_type is 'created', 'modified', 'deleted'
        """
        pass
    
    @abstractmethod
    def unwatch_directory(self, watch_id: str) -> bool:
        """Stop watching a directory"""
        pass
    
    @abstractmethod
    def stop_all_watches(self) -> bool:
        """Stop all active watches"""
        pass


class BaseNotifications(ABC):
    """Abstract base class for notification handling"""
    
    @abstractmethod
    def show_notification(self, 
                         title: str, 
                         message: str,
                         icon: Optional[str] = None,
                         sound: bool = True,
                         actions: Optional[List[str]] = None) -> str:
        """Show notification, returns notification_id"""
        pass
    
    @abstractmethod
    def clear_notification(self, notification_id: str) -> bool:
        """Clear a specific notification"""
        pass
    
    @abstractmethod
    def clear_all_notifications(self) -> bool:
        """Clear all JARVIS notifications"""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

class PlatformFactory:
    """Factory for creating platform-specific implementations"""
    
    _instances: Dict[str, Any] = {}
    
    @staticmethod
    def create_system_control() -> BaseSystemControl:
        """Create platform-specific system control"""
        if 'system_control' not in PlatformFactory._instances:
            from . import get_platform
            platform = get_platform()
            
            if platform == 'macos':
                from .macos.system_control import MacOSSystemControl
                PlatformFactory._instances['system_control'] = MacOSSystemControl()
            elif platform == 'windows':
                from .windows.system_control import WindowsSystemControl
                PlatformFactory._instances['system_control'] = WindowsSystemControl()
            elif platform == 'linux':
                from .linux.system_control import LinuxSystemControl
                PlatformFactory._instances['system_control'] = LinuxSystemControl()
            else:
                raise RuntimeError(f"Unsupported platform: {platform}")
        
        return PlatformFactory._instances['system_control']
    
    @staticmethod
    def create_audio_engine() -> BaseAudioEngine:
        """Create platform-specific audio engine"""
        if 'audio_engine' not in PlatformFactory._instances:
            from . import get_platform
            platform = get_platform()
            
            if platform == 'macos':
                from .macos.audio import MacOSAudioEngine
                PlatformFactory._instances['audio_engine'] = MacOSAudioEngine()
            elif platform == 'windows':
                from .windows.audio import WindowsAudioEngine
                PlatformFactory._instances['audio_engine'] = WindowsAudioEngine()
            elif platform == 'linux':
                from .linux.audio import LinuxAudioEngine
                PlatformFactory._instances['audio_engine'] = LinuxAudioEngine()
            else:
                raise RuntimeError(f"Unsupported platform: {platform}")
        
        return PlatformFactory._instances['audio_engine']
    
    @staticmethod
    def create_vision_capture() -> BaseVisionCapture:
        """Create platform-specific vision capture"""
        if 'vision_capture' not in PlatformFactory._instances:
            from . import get_platform
            platform = get_platform()
            
            if platform == 'macos':
                from .macos.vision import MacOSVisionCapture
                PlatformFactory._instances['vision_capture'] = MacOSVisionCapture()
            elif platform == 'windows':
                from .windows.vision import WindowsVisionCapture
                PlatformFactory._instances['vision_capture'] = WindowsVisionCapture()
            elif platform == 'linux':
                from .linux.vision import LinuxVisionCapture
                PlatformFactory._instances['vision_capture'] = LinuxVisionCapture()
            else:
                raise RuntimeError(f"Unsupported platform: {platform}")
        
        return PlatformFactory._instances['vision_capture']
    
    @staticmethod
    def create_authentication() -> BaseAuthentication:
        """Create platform-specific authentication"""
        if 'authentication' not in PlatformFactory._instances:
            from . import get_platform
            platform = get_platform()
            
            if platform == 'macos':
                from .macos.auth import MacOSAuthentication
                PlatformFactory._instances['authentication'] = MacOSAuthentication()
            elif platform == 'windows':
                from .windows.auth import WindowsAuthentication
                PlatformFactory._instances['authentication'] = WindowsAuthentication()
            elif platform == 'linux':
                from .linux.auth import LinuxAuthentication
                PlatformFactory._instances['authentication'] = LinuxAuthentication()
            else:
                raise RuntimeError(f"Unsupported platform: {platform}")
        
        return PlatformFactory._instances['authentication']
    
    @staticmethod
    def create_permissions() -> BasePermissions:
        """Create platform-specific permissions manager"""
        if 'permissions' not in PlatformFactory._instances:
            from . import get_platform
            platform = get_platform()
            
            if platform == 'macos':
                from .macos.permissions import MacOSPermissions
                PlatformFactory._instances['permissions'] = MacOSPermissions()
            elif platform == 'windows':
                from .windows.permissions import WindowsPermissions
                PlatformFactory._instances['permissions'] = WindowsPermissions()
            elif platform == 'linux':
                from .linux.permissions import LinuxPermissions
                PlatformFactory._instances['permissions'] = LinuxPermissions()
            else:
                raise RuntimeError(f"Unsupported platform: {platform}")
        
        return PlatformFactory._instances['permissions']
    
    @staticmethod
    def create_process_manager() -> BaseProcessManager:
        """Create platform-specific process manager"""
        if 'process_manager' not in PlatformFactory._instances:
            from . import get_platform
            platform = get_platform()
            
            if platform == 'macos':
                from .macos.process_manager import MacOSProcessManager
                PlatformFactory._instances['process_manager'] = MacOSProcessManager()
            elif platform == 'windows':
                from .windows.process_manager import WindowsProcessManager
                PlatformFactory._instances['process_manager'] = WindowsProcessManager()
            elif platform == 'linux':
                from .linux.process_manager import LinuxProcessManager
                PlatformFactory._instances['process_manager'] = LinuxProcessManager()
            else:
                raise RuntimeError(f"Unsupported platform: {platform}")
        
        return PlatformFactory._instances['process_manager']
    
    @staticmethod
    def create_file_watcher() -> BaseFileWatcher:
        """Create platform-specific file watcher"""
        if 'file_watcher' not in PlatformFactory._instances:
            from . import get_platform
            platform = get_platform()
            
            if platform == 'macos':
                from .macos.file_watcher import MacOSFileWatcher
                PlatformFactory._instances['file_watcher'] = MacOSFileWatcher()
            elif platform == 'windows':
                from .windows.file_watcher import WindowsFileWatcher
                PlatformFactory._instances['file_watcher'] = WindowsFileWatcher()
            elif platform == 'linux':
                from .linux.file_watcher import LinuxFileWatcher
                PlatformFactory._instances['file_watcher'] = LinuxFileWatcher()
            else:
                raise RuntimeError(f"Unsupported platform: {platform}")
        
        return PlatformFactory._instances['file_watcher']
    
    @staticmethod
    def create_notifications() -> BaseNotifications:
        """Create platform-specific notifications"""
        if 'notifications' not in PlatformFactory._instances:
            from . import get_platform
            platform = get_platform()
            
            if platform == 'macos':
                from .macos.notifications import MacOSNotifications
                PlatformFactory._instances['notifications'] = MacOSNotifications()
            elif platform == 'windows':
                from .windows.notifications import WindowsNotifications
                PlatformFactory._instances['notifications'] = WindowsNotifications()
            elif platform == 'linux':
                from .linux.notifications import LinuxNotifications
                PlatformFactory._instances['notifications'] = LinuxNotifications()
            else:
                raise RuntimeError(f"Unsupported platform: {platform}")
        
        return PlatformFactory._instances['notifications']
    
    @staticmethod
    def reset() -> None:
        """Reset all cached instances (for testing)"""
        PlatformFactory._instances.clear()
