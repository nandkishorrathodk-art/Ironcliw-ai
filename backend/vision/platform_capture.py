"""
Ironcliw Platform-Aware Vision Capture System
═══════════════════════════════════════════════════════════════════════════════

Unified vision capture router that detects platform and routes to Windows/macOS
implementations transparently.

Features:
    - Automatic platform detection
    - Unified API across Windows/macOS/Linux
    - Multi-monitor support
    - FPS-controlled continuous capture
    - Integration with YOLO and Claude Vision
    - Graceful fallback handling

Architecture:
    platform_capture.py (this file)
         ↓
    Platform detection
         ↓
    ┌────────────┬────────────┬────────────┐
    │   Windows  │   macOS    │   Linux    │
    └────────────┴────────────┴────────────┘

Author: Ironcliw System
Version: 1.0.0 (Windows Port - Phase 7)
"""
from __future__ import annotations

import logging
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CaptureFrame:
    """Unified capture frame structure across platforms"""
    image_data: np.ndarray  # RGB numpy array
    width: int
    height: int
    timestamp: float
    monitor_id: int
    format: str = 'rgb'  # 'rgb', 'bgr', 'rgba'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_pil(self) -> Image.Image:
        """Convert to PIL Image"""
        if self.format == 'bgr':
            # Convert BGR to RGB
            import cv2
            rgb_data = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_data)
        elif self.format in ('rgb', 'rgba'):
            return Image.fromarray(self.image_data)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def to_bytes(self, format: str = 'png') -> bytes:
        """Convert to bytes (PNG or JPEG)"""
        import io
        img = self.to_pil()
        buffer = io.BytesIO()
        img.save(buffer, format=format.upper())
        return buffer.getvalue()


@dataclass
class MonitorInfo:
    """Unified monitor information structure"""
    monitor_id: int
    width: int
    height: int
    x: int
    y: int
    is_primary: bool
    name: str = ""
    refresh_rate: float = 60.0


class PlatformVisionCapture:
    """
    Platform-aware vision capture system
    
    Automatically detects platform and delegates to appropriate implementation.
    Provides unified API for screen capture, multi-monitor, and continuous capture.
    """
    
    def __init__(self):
        """Initialize platform-specific vision capture"""
        self._platform = self._detect_platform()
        self._capturer = None
        self._is_capturing = False
        self._capture_callback = None
        self._initialize_capturer()
        
        logger.info(f"Initialized PlatformVisionCapture for {self._platform}")
    
    def _detect_platform(self) -> str:
        """Detect current platform"""
        try:
            from backend.platform_adapter import get_platform
            return get_platform()
        except ImportError:
            import platform
            system = platform.system().lower()
            if system == 'darwin':
                return 'macos'
            elif system == 'windows':
                return 'windows'
            elif system == 'linux':
                return 'linux'
            else:
                raise RuntimeError(f"Unsupported platform: {system}")
    
    def _initialize_capturer(self):
        """Initialize platform-specific capturer"""
        if self._platform == 'windows':
            self._initialize_windows_capturer()
        elif self._platform == 'macos':
            self._initialize_macos_capturer()
        elif self._platform == 'linux':
            self._initialize_linux_capturer()
        else:
            raise RuntimeError(f"No capturer available for platform: {self._platform}")
    
    def _initialize_windows_capturer(self):
        """Initialize Windows vision capturer"""
        try:
            from .windows_vision_capture import WindowsVisionCapture
            self._capturer = WindowsVisionCapture()
            logger.info("✅ Initialized Windows vision capturer")
        except ImportError as e:
            logger.error(f"Failed to import WindowsVisionCapture: {e}")
            logger.info("Attempting fallback to platform.windows.vision...")
            try:
                from backend.platform_adapter.windows.vision import WindowsVisionCapture as FallbackCapture
                self._capturer = FallbackCapture()
                logger.info("✅ Initialized Windows vision capturer (fallback)")
            except ImportError as e2:
                raise RuntimeError(
                    f"Failed to initialize Windows capturer: {e2}\n"
                    f"Please ensure C# DLLs are built and pythonnet is installed."
                ) from e2
    
    def _initialize_macos_capturer(self):
        """Initialize macOS vision capturer"""
        try:
            from .macos_video_capture_advanced import MacOSAdvancedVideoCapture
            self._capturer = MacOSAdvancedVideoCapture()
            logger.info("✅ Initialized macOS vision capturer")
        except ImportError as e:
            logger.warning(f"Advanced macOS capturer not available: {e}")
            logger.info("Falling back to native macOS capture...")
            try:
                from .macos_native_capture import MacOSNativeCapture
                self._capturer = MacOSNativeCapture()
                logger.info("✅ Initialized macOS native capturer (fallback)")
            except ImportError as e2:
                raise RuntimeError(
                    f"Failed to initialize macOS capturer: {e2}\n"
                    f"Please ensure PyObjC is installed."
                ) from e2
    
    def _initialize_linux_capturer(self):
        """Initialize Linux vision capturer (placeholder)"""
        logger.warning("Linux vision capture not yet implemented - using fallback")
        try:
            from .linux_vision_capture import LinuxVisionCapture
            self._capturer = LinuxVisionCapture()
            logger.info("✅ Initialized Linux vision capturer")
        except ImportError:
            logger.warning("Linux capturer not available - using screenshot fallback")
            from .screenshot_fallback import ScreenshotFallbackCapture
            self._capturer = ScreenshotFallbackCapture()
    
    def capture_screen(self, monitor_id: int = 0) -> Optional[CaptureFrame]:
        """
        Capture screenshot from specified monitor
        
        Args:
            monitor_id: Monitor index (0 = primary/all, 1+ = specific monitor)
        
        Returns:
            CaptureFrame with image data or None if failed
        """
        if self._capturer is None:
            logger.error("Capturer not initialized")
            return None
        
        try:
            raw_frame = self._capturer.capture_screen(monitor_id)
            if raw_frame is None:
                return None
            
            # Convert platform-specific frame to unified CaptureFrame
            return self._convert_to_unified_frame(raw_frame, monitor_id)
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return None
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[CaptureFrame]:
        """
        Capture specific screen region
        
        Args:
            x, y: Top-left corner coordinates
            width, height: Region dimensions
        
        Returns:
            CaptureFrame with region image or None if failed
        """
        if self._capturer is None or not hasattr(self._capturer, 'capture_region'):
            logger.warning("Region capture not supported - falling back to full screen")
            frame = self.capture_screen()
            if frame is None:
                return None
            
            # Crop the region
            try:
                cropped = frame.image_data[y:y+height, x:x+width]
                return CaptureFrame(
                    image_data=cropped,
                    width=width,
                    height=height,
                    timestamp=time.time(),
                    monitor_id=0,
                    format=frame.format,
                    metadata={'cropped_from': 'full_screen', 'original_bounds': (x, y, width, height)}
                )
            except Exception as e:
                logger.error(f"Failed to crop region: {e}")
                return None
        
        try:
            raw_frame = self._capturer.capture_region(x, y, width, height)
            if raw_frame is None:
                return None
            
            return self._convert_to_unified_frame(raw_frame, 0)
        except Exception as e:
            logger.error(f"Failed to capture region: {e}")
            return None
    
    def get_monitors(self) -> List[MonitorInfo]:
        """
        Get list of all connected monitors
        
        Returns:
            List of MonitorInfo objects
        """
        if self._capturer is None or not hasattr(self._capturer, 'get_monitors'):
            logger.warning("Monitor detection not supported on this platform")
            return [MonitorInfo(
                monitor_id=0,
                width=1920,
                height=1080,
                x=0,
                y=0,
                is_primary=True,
                name="Primary Display"
            )]
        
        try:
            monitors = self._capturer.get_monitors()
            return [self._convert_to_monitor_info(m) for m in monitors]
        except Exception as e:
            logger.error(f"Failed to get monitors: {e}")
            return []
    
    def start_continuous_capture(
        self,
        callback: Callable[[CaptureFrame], None],
        fps: int = 15,
        monitor_id: int = 0
    ) -> bool:
        """
        Start continuous screen capture at specified FPS
        
        Args:
            callback: Function to call with each captured frame
            fps: Target frames per second
            monitor_id: Monitor to capture (0 = all/primary)
        
        Returns:
            True if started successfully
        """
        if self._is_capturing:
            logger.warning("Continuous capture already running")
            return False
        
        if self._capturer is None or not hasattr(self._capturer, 'start_continuous_capture'):
            logger.warning("Continuous capture not supported - falling back to polling")
            return self._start_polling_capture(callback, fps, monitor_id)
        
        def wrapper_callback(raw_frame):
            """Wrap callback to convert to unified frame"""
            unified_frame = self._convert_to_unified_frame(raw_frame, monitor_id)
            if unified_frame:
                callback(unified_frame)
        
        try:
            self._capture_callback = callback
            success = self._capturer.start_continuous_capture(wrapper_callback, fps, monitor_id)
            if success:
                self._is_capturing = True
                logger.info(f"Started continuous capture at {fps} FPS on monitor {monitor_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to start continuous capture: {e}")
            return False
    
    def stop_continuous_capture(self) -> bool:
        """
        Stop continuous screen capture
        
        Returns:
            True if stopped successfully
        """
        if not self._is_capturing:
            logger.warning("Continuous capture not running")
            return False
        
        if self._capturer is None or not hasattr(self._capturer, 'stop_continuous_capture'):
            logger.warning("Cannot stop continuous capture - not supported")
            return False
        
        try:
            success = self._capturer.stop_continuous_capture()
            if success:
                self._is_capturing = False
                self._capture_callback = None
                logger.info("Stopped continuous capture")
            return success
        except Exception as e:
            logger.error(f"Failed to stop continuous capture: {e}")
            return False
    
    def _start_polling_capture(
        self,
        callback: Callable[[CaptureFrame], None],
        fps: int,
        monitor_id: int
    ) -> bool:
        """Fallback polling-based capture for platforms without native continuous capture"""
        import threading
        
        def poll_loop():
            interval = 1.0 / fps
            while self._is_capturing:
                try:
                    frame = self.capture_screen(monitor_id)
                    if frame:
                        callback(frame)
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in polling capture: {e}")
                    time.sleep(interval)
        
        self._is_capturing = True
        self._capture_callback = callback
        self._poll_thread = threading.Thread(target=poll_loop, daemon=True)
        self._poll_thread.start()
        
        logger.info(f"Started polling capture at {fps} FPS (fallback)")
        return True
    
    def _convert_to_unified_frame(self, raw_frame: Any, monitor_id: int) -> Optional[CaptureFrame]:
        """Convert platform-specific frame to unified CaptureFrame"""
        try:
            # Handle different platform-specific frame types
            if hasattr(raw_frame, 'image_data'):
                # Already in compatible format
                return CaptureFrame(
                    image_data=raw_frame.image_data if isinstance(raw_frame.image_data, np.ndarray) else np.array(raw_frame.image_data),
                    width=raw_frame.width,
                    height=raw_frame.height,
                    timestamp=getattr(raw_frame, 'timestamp', time.time()),
                    monitor_id=monitor_id,
                    format=getattr(raw_frame, 'format', 'rgb'),
                    metadata=getattr(raw_frame, 'metadata', {})
                )
            elif isinstance(raw_frame, np.ndarray):
                # Raw numpy array
                h, w = raw_frame.shape[:2]
                return CaptureFrame(
                    image_data=raw_frame,
                    width=w,
                    height=h,
                    timestamp=time.time(),
                    monitor_id=monitor_id,
                    format='rgb',
                    metadata={}
                )
            elif isinstance(raw_frame, Image.Image):
                # PIL Image
                return CaptureFrame(
                    image_data=np.array(raw_frame),
                    width=raw_frame.width,
                    height=raw_frame.height,
                    timestamp=time.time(),
                    monitor_id=monitor_id,
                    format='rgb',
                    metadata={}
                )
            else:
                logger.error(f"Unknown frame type: {type(raw_frame)}")
                return None
        except Exception as e:
            logger.error(f"Failed to convert frame: {e}")
            return None
    
    def _convert_to_monitor_info(self, monitor: Any) -> MonitorInfo:
        """Convert platform-specific monitor info to unified MonitorInfo"""
        return MonitorInfo(
            monitor_id=getattr(monitor, 'monitor_id', getattr(monitor, 'display_id', 0)),
            width=getattr(monitor, 'width', getattr(monitor, 'resolution', (0, 0))[0]),
            height=getattr(monitor, 'height', getattr(monitor, 'resolution', (0, 0))[1]),
            x=getattr(monitor, 'x', getattr(monitor, 'position', (0, 0))[0]),
            y=getattr(monitor, 'y', getattr(monitor, 'position', (0, 0))[1]),
            is_primary=getattr(monitor, 'is_primary', False),
            name=getattr(monitor, 'name', f"Monitor {getattr(monitor, 'monitor_id', 0)}"),
            refresh_rate=getattr(monitor, 'refresh_rate', 60.0)
        )
    
    @property
    def platform(self) -> str:
        """Get current platform"""
        return self._platform
    
    @property
    def is_capturing(self) -> bool:
        """Check if continuous capture is running"""
        return self._is_capturing


# Global singleton instance
_vision_capture_instance: Optional[PlatformVisionCapture] = None


def get_vision_capture() -> PlatformVisionCapture:
    """
    Get global vision capture instance (singleton)
    
    Returns:
        PlatformVisionCapture instance
    """
    global _vision_capture_instance
    if _vision_capture_instance is None:
        _vision_capture_instance = PlatformVisionCapture()
    return _vision_capture_instance


def capture_screen(monitor_id: int = 0) -> Optional[CaptureFrame]:
    """
    Convenience function to capture screen
    
    Args:
        monitor_id: Monitor index
    
    Returns:
        CaptureFrame or None
    """
    return get_vision_capture().capture_screen(monitor_id)


def get_monitors() -> List[MonitorInfo]:
    """
    Convenience function to get monitors
    
    Returns:
        List of MonitorInfo
    """
    return get_vision_capture().get_monitors()

