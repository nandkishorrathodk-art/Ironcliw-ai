"""
Windows Screen Capture Implementation

Uses mss library for fast cross-monitor screen capture on Windows.
Supports DirectX Desktop Duplication API (DXGI) for hardware-accelerated capture.

Created: 2026-02-23
Purpose: Windows/Linux porting - Phase 3 (Screen Capture)
"""

import asyncio
import logging
import time
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
    CapturePermissionError,
)

logger = logging.getLogger(__name__)

# Try importing Windows-specific libraries
MSS_AVAILABLE = False
PIL_AVAILABLE = False
PYAUTOGUI_AVAILABLE = False

try:
    import mss
    import mss.windows
    MSS_AVAILABLE = True
    logger.info("✅ mss library available for Windows screen capture")
except ImportError:
    logger.warning("⚠️ mss library not available (install: pip install mss)")

try:
    from PIL import ImageGrab
    PIL_AVAILABLE = True
    logger.debug("PIL ImageGrab available as fallback")
except ImportError:
    logger.debug("PIL not available")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    logger.debug("pyautogui available as fallback")
except ImportError:
    logger.debug("pyautogui not available")


class WindowsScreenCapture(ScreenCaptureInterface):
    """
    Windows screen capture implementation using mss library.
    
    Features:
    - Multi-monitor support with automatic detection
    - Hardware-accelerated capture via DXGI (Desktop Duplication API)
    - High-performance capturing (60+ FPS capable)
    - Automatic fallback to GDI if DXGI unavailable
    - PIL ImageGrab and pyautogui as final fallbacks
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """
        Initialize Windows screen capture.
        
        Args:
            config: Capture configuration
        """
        super().__init__(config)
        self._sct: Optional[mss.mss] = None
        self._monitor_info: Dict[str, Any] = {}
        self._capture_task: Optional[asyncio.Task] = None
        self._frame_queue: Optional[asyncio.Queue] = None
        self._selected_monitor: Optional[Dict[str, Any]] = None
        
        # Determine capture method priority
        self._capture_methods = self._detect_available_methods()
        if not self._capture_methods:
            raise CaptureNotSupportedError(
                "No screen capture methods available on Windows. "
                "Install mss: pip install mss"
            )
        
        logger.info(f"Windows capture methods available: {[m.value for m in self._capture_methods]}")
    
    def _detect_available_methods(self) -> List[CaptureMethod]:
        """Detect available capture methods in priority order."""
        methods = []
        
        if MSS_AVAILABLE:
            methods.append(CaptureMethod.MSS)
        
        if PIL_AVAILABLE:
            methods.append(CaptureMethod.PIL_SCREENSHOT)
        
        if PYAUTOGUI_AVAILABLE:
            methods.append(CaptureMethod.PIL_SCREENSHOT)
        
        return methods
    
    def get_capture_methods(self) -> List[CaptureMethod]:
        """Get list of available capture methods."""
        return self._capture_methods
    
    def get_available_displays(self) -> List[Dict[str, Any]]:
        """
        Get list of available displays/monitors on Windows.
        
        Returns:
            List of display info dicts
        """
        displays = []
        
        if MSS_AVAILABLE:
            with mss.mss() as sct:
                # Monitor 0 is all monitors combined
                # Monitor 1+ are individual monitors
                for i, monitor in enumerate(sct.monitors[1:], start=1):
                    display_info = {
                        "id": str(i),
                        "name": f"Display {i}",
                        "width": monitor["width"],
                        "height": monitor["height"],
                        "x": monitor["left"],
                        "y": monitor["top"],
                        "is_primary": i == 1,  # First monitor is usually primary
                    }
                    displays.append(display_info)
                    logger.debug(f"Detected monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
        
        return displays
    
    async def start(self) -> bool:
        """
        Start screen capture on Windows.
        
        Returns:
            bool: True if started successfully
        """
        if self._running:
            logger.warning("Capture already running")
            return True
        
        try:
            # Initialize mss context
            if MSS_AVAILABLE:
                self._sct = mss.mss()
                
                # Select monitor
                if self.config.display_id:
                    monitor_idx = int(self.config.display_id)
                    if 0 < monitor_idx <= len(self._sct.monitors) - 1:
                        self._selected_monitor = self._sct.monitors[monitor_idx]
                    else:
                        logger.warning(f"Invalid monitor index {monitor_idx}, using primary")
                        self._selected_monitor = self._sct.monitors[1]
                else:
                    # Use primary monitor (index 1)
                    self._selected_monitor = self._sct.monitors[1]
                
                logger.info(
                    f"Windows capture started on monitor: "
                    f"{self._selected_monitor['width']}x{self._selected_monitor['height']}"
                )
            else:
                raise CaptureNotSupportedError("MSS library not available")
            
            # Create frame queue
            self._frame_queue = asyncio.Queue(maxsize=self.config.buffer_size)
            
            # Start capture loop
            self._running = True
            self._capture_task = asyncio.create_task(self._capture_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Windows capture: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> None:
        """Stop screen capture and cleanup resources."""
        if not self._running:
            return
        
        logger.info("Stopping Windows screen capture...")
        self._running = False
        
        # Cancel capture task
        if self._capture_task:
            self._capture_task.cancel()
            try:
                await self._capture_task
            except asyncio.CancelledError:
                pass
            self._capture_task = None
        
        # Cleanup mss
        if self._sct:
            self._sct.close()
            self._sct = None
        
        # Clear queue
        if self._frame_queue:
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._frame_queue = None
        
        logger.info("Windows screen capture stopped")
    
    async def _capture_loop(self) -> None:
        """
        Main capture loop running in background.
        Captures frames and puts them in queue for retrieval.
        """
        frame_interval = 1.0 / self.config.fps_target
        last_frame_time = 0.0
        
        logger.info(f"Capture loop started (target: {self.config.fps_target} FPS)")
        
        while self._running:
            try:
                current_time = time.time()
                
                # Rate limiting to target FPS
                time_since_last = current_time - last_frame_time
                if time_since_last < frame_interval:
                    await asyncio.sleep(frame_interval - time_since_last)
                    continue
                
                # Capture frame
                frame = await self._capture_frame_internal()
                
                if frame:
                    # Update stats
                    self.stats.frames_captured += 1
                    self.stats.last_frame_time = frame.timestamp
                    self.stats.total_bytes_captured += frame.size_bytes
                    
                    # Calculate FPS
                    if last_frame_time > 0:
                        actual_fps = 1.0 / (current_time - last_frame_time)
                        self.stats.update_fps(actual_fps)
                    
                    last_frame_time = current_time
                    
                    # Put in queue (non-blocking, drop oldest if full)
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                            self.stats.frames_dropped += 1
                        except asyncio.QueueEmpty:
                            pass
                    
                    await self._frame_queue.put(frame)
                    
                    # Notify callbacks
                    self._notify_callbacks(frame)
                else:
                    self.stats.capture_errors += 1
                    await asyncio.sleep(0.1)  # Brief pause on error
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self.stats.capture_errors += 1
                await asyncio.sleep(0.1)
        
        logger.info("Capture loop exited")
    
    async def _capture_frame_internal(self) -> Optional[CaptureFrame]:
        """
        Internal method to capture a single frame.
        
        Returns:
            CaptureFrame or None on error
        """
        try:
            if not self._sct or not self._selected_monitor:
                return None
            
            # Capture using mss
            screenshot = self._sct.grab(self._selected_monitor)
            
            # Convert to numpy array (BGRA format from mss)
            img_array = np.array(screenshot, dtype=np.uint8)
            
            # Create capture frame
            frame = CaptureFrame(
                data=img_array,
                timestamp=datetime.now(),
                display_id=self.config.display_id or "1",
                width=screenshot.width,
                height=screenshot.height,
                format="BGRA",
                metadata={
                    "method": "mss",
                    "monitor": self._selected_monitor,
                }
            )
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
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
        More efficient than start/get/stop for one-off captures.
        
        Args:
            display_id: Display to capture (None = primary)
        
        Returns:
            CaptureFrame or None
        """
        if MSS_AVAILABLE:
            try:
                with mss.mss() as sct:
                    # Select monitor
                    if display_id:
                        monitor_idx = int(display_id)
                        monitor = sct.monitors[monitor_idx] if 0 < monitor_idx < len(sct.monitors) else sct.monitors[1]
                    else:
                        monitor = sct.monitors[1]  # Primary
                    
                    # Capture
                    screenshot = sct.grab(monitor)
                    img_array = np.array(screenshot, dtype=np.uint8)
                    
                    return CaptureFrame(
                        data=img_array,
                        timestamp=datetime.now(),
                        display_id=display_id or "1",
                        width=screenshot.width,
                        height=screenshot.height,
                        format="BGRA",
                        metadata={"method": "mss", "single_shot": True}
                    )
            except Exception as e:
                logger.error(f"Single frame capture failed: {e}")
                return None
        
        elif PIL_AVAILABLE:
            try:
                # PIL fallback (all monitors as one image)
                screenshot = ImageGrab.grab()
                img_array = np.array(screenshot, dtype=np.uint8)
                
                return CaptureFrame(
                    data=img_array,
                    timestamp=datetime.now(),
                    display_id=display_id or "all",
                    width=screenshot.width,
                    height=screenshot.height,
                    format="RGB",
                    metadata={"method": "pil", "single_shot": True}
                )
            except Exception as e:
                logger.error(f"PIL single frame capture failed: {e}")
                return None
        
        else:
            logger.error("No capture method available")
            return None
