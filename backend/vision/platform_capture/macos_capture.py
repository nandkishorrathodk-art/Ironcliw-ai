"""
macOS Screen Capture Wrapper

Wraps the existing advanced macOS video capture implementation
to conform to the cross-platform interface.

Created: 2026-02-23
Purpose: Windows/Linux porting - Phase 3 (Screen Capture)
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from .base_capture import (
    ScreenCaptureInterface,
    CaptureConfig,
    CaptureFrame,
    CaptureMethod,
    CaptureQuality,
    CaptureError,
    CaptureNotSupportedError,
)

logger = logging.getLogger(__name__)

# Try importing existing macOS capture implementation
MACOS_CAPTURE_AVAILABLE = False
try:
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import existing implementation
    vision_path = Path(__file__).parent.parent
    if str(vision_path) not in sys.path:
        sys.path.insert(0, str(vision_path))
    
    from macos_video_capture_advanced import (
        VideoWatcher,
        CaptureMode,
        is_capture_available,
    )
    MACOS_CAPTURE_AVAILABLE = True
    logger.info("✅ macOS advanced video capture available")
except ImportError as e:
    logger.warning(f"⚠️ macOS video capture not available: {e}")


class MacOSScreenCapture(ScreenCaptureInterface):
    """
    macOS screen capture wrapper.
    
    Wraps the existing VideoWatcher implementation to provide
    a consistent cross-platform interface while maintaining
    all the advanced features of the macOS implementation.
    
    Features (from existing implementation):
    - Native AVFoundation integration
    - ScreenCaptureKit support (macOS 12.3+)
    - Edge case resilience (permissions, fullscreen, display health)
    - Performance monitoring
    - Multiple capture modes
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """
        Initialize macOS screen capture wrapper.
        
        Args:
            config: Capture configuration
        """
        super().__init__(config)
        
        if not MACOS_CAPTURE_AVAILABLE:
            raise CaptureNotSupportedError(
                "macOS video capture not available. "
                "Ensure PyObjC frameworks are installed."
            )
        
        self._watcher: Optional[VideoWatcher] = None
        self._frame_queue: Optional[asyncio.Queue] = None
        
        # Map quality to capture mode
        self._capture_mode = self._map_quality_to_mode(config.quality)
        
        logger.info(f"macOS capture initialized with mode: {self._capture_mode}")
    
    @staticmethod
    def _map_quality_to_mode(quality: CaptureQuality) -> CaptureMode:
        """Map CaptureQuality to macOS CaptureMode."""
        mapping = {
            CaptureQuality.LOW: CaptureMode.PERFORMANCE,
            CaptureQuality.MEDIUM: CaptureMode.BALANCED,
            CaptureQuality.HIGH: CaptureMode.QUALITY,
            CaptureQuality.ULTRA: CaptureMode.MAXIMUM,
        }
        return mapping.get(quality, CaptureMode.BALANCED)
    
    def get_capture_methods(self) -> List[CaptureMethod]:
        """Get list of available capture methods on macOS."""
        methods = []
        
        if is_capture_available():
            methods.append(CaptureMethod.AVFOUNDATION)
            methods.append(CaptureMethod.SCREENCAPTUREKIT)
            methods.append(CaptureMethod.MACOS_SCREENCAPTURE)
        
        return methods
    
    def get_available_displays(self) -> List[Dict[str, Any]]:
        """
        Get list of available displays on macOS.
        
        Returns:
            List of display info dicts
        """
        displays = []
        
        # The existing implementation auto-detects displays
        # For now, return a default primary display
        # TODO: Integrate with existing display detection
        displays.append({
            "id": "main",
            "name": "Main Display",
            "width": 1920,
            "height": 1080,
            "x": 0,
            "y": 0,
            "is_primary": True,
        })
        
        return displays
    
    async def start(self) -> bool:
        """
        Start screen capture on macOS.
        
        Returns:
            bool: True if started successfully
        """
        if self._running:
            logger.warning("Capture already running")
            return True
        
        try:
            # Create frame queue
            self._frame_queue = asyncio.Queue(maxsize=self.config.buffer_size)
            
            # Initialize VideoWatcher
            self._watcher = VideoWatcher(
                mode=self._capture_mode,
                display_id=int(self.config.display_id) if self.config.display_id else None,
            )
            
            # Register frame callback
            self._watcher.register_frame_callback(self._on_frame_callback)
            
            # Start capture
            success = await self._watcher.start()
            
            if success:
                self._running = True
                logger.info("macOS screen capture started")
                return True
            else:
                logger.error("Failed to start macOS screen capture")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start macOS capture: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> None:
        """Stop screen capture and cleanup resources."""
        if not self._running:
            return
        
        logger.info("Stopping macOS screen capture...")
        self._running = False
        
        # Stop watcher
        if self._watcher:
            await self._watcher.stop()
            self._watcher = None
        
        # Clear queue
        if self._frame_queue:
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._frame_queue = None
        
        logger.info("macOS screen capture stopped")
    
    def _on_frame_callback(self, frame_data: np.ndarray, metadata: Dict[str, Any]) -> None:
        """
        Callback from VideoWatcher when a new frame is captured.
        
        Args:
            frame_data: Numpy array of frame data
            metadata: Frame metadata from VideoWatcher
        """
        try:
            # Create CaptureFrame
            frame = CaptureFrame(
                data=frame_data,
                timestamp=datetime.now(),
                display_id=str(metadata.get("display_id", "main")),
                width=frame_data.shape[1],
                height=frame_data.shape[0],
                format=metadata.get("format", "RGB"),
                metadata=metadata,
            )
            
            # Update stats
            self.stats.frames_captured += 1
            self.stats.last_frame_time = frame.timestamp
            self.stats.total_bytes_captured += frame.size_bytes
            
            # Update FPS from metadata if available
            if "fps" in metadata:
                self.stats.update_fps(metadata["fps"])
            
            # Put in queue (non-blocking)
            if self._frame_queue and not self._frame_queue.full():
                try:
                    self._frame_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    self.stats.frames_dropped += 1
            elif self._frame_queue and self._frame_queue.full():
                # Drop oldest frame
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(frame)
                    self.stats.frames_dropped += 1
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass
            
            # Notify callbacks
            self._notify_callbacks(frame)
            
        except Exception as e:
            logger.error(f"Error in frame callback: {e}")
            self.stats.capture_errors += 1
    
    async def get_frame(self, timeout: Optional[float] = None) -> Optional[CaptureFrame]:
        """
        Get the next captured frame from queue.
        
        Args:
            timeout: Maximum time to wait for frame
        
        Returns:
            CaptureFrame or None on timeout
        """
        if not self._running or not self._frame_queue:
            logger.warning("Capture not running")
            return None
        
        timeout_val = timeout if timeout is not None else self.config.timeout
        
        try:
            frame = await asyncio.wait_for(
                self._frame_queue.get(),
                timeout=timeout_val
            )
            return frame
        except asyncio.TimeoutError:
            logger.debug(f"Frame timeout after {timeout_val}s")
            return None
    
    async def capture_single_frame(self, display_id: Optional[str] = None) -> Optional[CaptureFrame]:
        """
        Capture a single frame without starting continuous capture.
        
        Args:
            display_id: Display to capture (None = primary)
        
        Returns:
            CaptureFrame or None
        """
        # Use existing implementation's single-shot capture if available
        # For now, use the default start/get/stop implementation
        return await super().capture_single_frame(display_id)
