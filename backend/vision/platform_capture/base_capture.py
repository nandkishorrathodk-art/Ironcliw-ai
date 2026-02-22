"""
Platform Abstraction Layer - Base Screen Capture Module

Abstract base class for cross-platform screen capture implementations.
Defines the interface that all platform-specific capture implementations must follow.

Created: 2026-02-23
Purpose: Windows/Linux porting - Phase 3 (Screen Capture)
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import logging


logger = logging.getLogger(__name__)


class CaptureMethod(Enum):
    """Capture methods available on each platform."""
    # macOS
    AVFOUNDATION = "avfoundation"
    SCREENCAPTUREKIT = "screencapturekit"
    MACOS_SCREENCAPTURE = "macos_screencapture"
    
    # Windows
    MSS = "mss"
    WINDOWS_GDI = "windows_gdi"
    WINDOWS_DXGI = "windows_dxgi"
    
    # Linux
    X11 = "x11"
    WAYLAND = "wayland"
    LINUX_SCROT = "linux_scrot"
    
    # Cross-platform fallback
    PIL_SCREENSHOT = "pil_screenshot"


class CaptureQuality(Enum):
    """Quality presets for screen capture."""
    LOW = "low"          # 640x480 or scaled down
    MEDIUM = "medium"    # 1280x720
    HIGH = "high"        # 1920x1080
    ULTRA = "ultra"      # Native resolution


@dataclass
class CaptureConfig:
    """Configuration for screen capture."""
    method: Optional[CaptureMethod] = None  # None = auto-detect
    quality: CaptureQuality = CaptureQuality.HIGH
    fps_target: int = 30
    display_id: Optional[str] = None  # None = primary display
    capture_cursor: bool = True
    enable_monitoring: bool = True
    timeout: float = 5.0
    
    # Performance tuning
    buffer_size: int = 5
    use_hardware_acceleration: bool = True
    
    # Edge case handling
    retry_on_failure: bool = True
    max_retries: int = 3
    black_screen_detection: bool = True


@dataclass
class CaptureStats:
    """Real-time capture statistics."""
    frames_captured: int = 0
    frames_dropped: int = 0
    current_fps: float = 0.0
    average_fps: float = 0.0
    last_frame_time: Optional[datetime] = None
    total_bytes_captured: int = 0
    capture_errors: int = 0
    
    def update_fps(self, new_fps: float):
        """Update FPS statistics."""
        if self.frames_captured == 0:
            self.average_fps = new_fps
        else:
            # Exponential moving average
            alpha = 0.2
            self.average_fps = alpha * new_fps + (1 - alpha) * self.average_fps
        self.current_fps = new_fps


@dataclass
class CaptureFrame:
    """Container for captured screen frame."""
    data: np.ndarray  # RGB or RGBA numpy array
    timestamp: datetime
    display_id: str
    width: int
    height: int
    format: str = "RGB"  # RGB, RGBA, BGR, BGRA
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get frame shape (height, width, channels)."""
        return self.data.shape
    
    @property
    def size_bytes(self) -> int:
        """Get frame size in bytes."""
        return self.data.nbytes
    
    def to_rgb(self) -> np.ndarray:
        """Convert frame to RGB format if needed."""
        if self.format == "RGB":
            return self.data
        elif self.format == "RGBA":
            return self.data[:, :, :3]
        elif self.format == "BGR":
            return self.data[:, :, ::-1]
        elif self.format == "BGRA":
            return self.data[:, :, [2, 1, 0]]
        else:
            logger.warning(f"Unknown format {self.format}, returning as-is")
            return self.data


class ScreenCaptureInterface(ABC):
    """
    Abstract base class for platform-specific screen capture.
    
    All platform implementations must inherit from this class and
    implement the required abstract methods.
    
    Usage:
        capture = PlatformCapture.create()  # Auto-detect platform
        await capture.start()
        frame = await capture.get_frame()
        await capture.stop()
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """
        Initialize capture interface.
        
        Args:
            config: Capture configuration (uses defaults if None)
        """
        self.config = config or CaptureConfig()
        self.stats = CaptureStats()
        self._running = False
        self._callbacks: List[Callable[[CaptureFrame], None]] = []
    
    @abstractmethod
    async def start(self) -> bool:
        """
        Start screen capture.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """
        Stop screen capture and cleanup resources.
        """
        pass
    
    @abstractmethod
    async def get_frame(self, timeout: Optional[float] = None) -> Optional[CaptureFrame]:
        """
        Get the next captured frame.
        
        Args:
            timeout: Maximum time to wait for frame (None = use config timeout)
        
        Returns:
            CaptureFrame if available, None on timeout or error
        """
        pass
    
    @abstractmethod
    def get_available_displays(self) -> List[Dict[str, Any]]:
        """
        Get list of available displays for capture.
        
        Returns:
            List of display info dicts with keys: id, name, width, height, is_primary
        """
        pass
    
    @abstractmethod
    def get_capture_methods(self) -> List[CaptureMethod]:
        """
        Get list of capture methods available on this platform.
        
        Returns:
            List of CaptureMethod enum values
        """
        pass
    
    @property
    def is_running(self) -> bool:
        """Check if capture is currently running."""
        return self._running
    
    def register_callback(self, callback: Callable[[CaptureFrame], None]) -> None:
        """
        Register a callback to be called for each captured frame.
        
        Args:
            callback: Function to call with each CaptureFrame
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            logger.debug(f"Registered capture callback: {callback.__name__}")
    
    def unregister_callback(self, callback: Callable[[CaptureFrame], None]) -> None:
        """
        Unregister a frame callback.
        
        Args:
            callback: Function to unregister
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            logger.debug(f"Unregistered capture callback: {callback.__name__}")
    
    def _notify_callbacks(self, frame: CaptureFrame) -> None:
        """
        Notify all registered callbacks with new frame.
        
        Args:
            frame: Captured frame to send to callbacks
        """
        for callback in self._callbacks:
            try:
                callback(frame)
            except Exception as e:
                logger.error(f"Error in capture callback {callback.__name__}: {e}")
    
    async def capture_single_frame(self, display_id: Optional[str] = None) -> Optional[CaptureFrame]:
        """
        Capture a single frame without starting continuous capture.
        
        Args:
            display_id: Display to capture (None = primary)
        
        Returns:
            CaptureFrame if successful, None otherwise
        """
        # Default implementation: start, capture one, stop
        # Platforms can override for more efficient single-shot capture
        original_display = self.config.display_id
        if display_id:
            self.config.display_id = display_id
        
        try:
            await self.start()
            frame = await self.get_frame(timeout=self.config.timeout)
            return frame
        finally:
            await self.stop()
            self.config.display_id = original_display
    
    def get_stats(self) -> CaptureStats:
        """
        Get current capture statistics.
        
        Returns:
            CaptureStats object with current stats
        """
        return self.stats
    
    def reset_stats(self) -> None:
        """Reset capture statistics."""
        self.stats = CaptureStats()
        logger.debug("Capture statistics reset")


class CaptureError(Exception):
    """Base exception for capture-related errors."""
    pass


class CaptureNotSupportedError(CaptureError):
    """Raised when capture method is not supported on this platform."""
    pass


class CapturePermissionError(CaptureError):
    """Raised when screen capture permissions are not granted."""
    pass


class CaptureTimeoutError(CaptureError):
    """Raised when capture operation times out."""
    pass
