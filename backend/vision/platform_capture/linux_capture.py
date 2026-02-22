"""
Linux Screen Capture Implementation

Supports both X11 and Wayland display servers.
Uses mss for X11 and fallback methods for Wayland.

Created: 2026-02-23
Purpose: Windows/Linux porting - Phase 3 (Screen Capture)
"""

import asyncio
import logging
import os
import subprocess
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
)

logger = logging.getLogger(__name__)

# Try importing Linux-specific libraries
MSS_AVAILABLE = False
PIL_AVAILABLE = False
PYAUTOGUI_AVAILABLE = False

try:
    import mss
    MSS_AVAILABLE = True
    logger.info("✅ mss library available for Linux X11 screen capture")
except ImportError:
    logger.warning("⚠️ mss library not available (install: pip install mss)")

try:
    from PIL import Image, ImageGrab
    PIL_AVAILABLE = True
    logger.debug("PIL available for Linux screen capture")
except ImportError:
    logger.debug("PIL not available")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    logger.debug("pyautogui available as fallback")
except ImportError:
    logger.debug("pyautogui not available")


def _detect_display_server() -> str:
    """
    Detect if running on X11 or Wayland.
    
    Returns:
        "x11", "wayland", or "unknown"
    """
    # Check XDG_SESSION_TYPE first (most reliable)
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session_type in ("x11", "wayland"):
        return session_type
    
    # Check WAYLAND_DISPLAY
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    
    # Check DISPLAY (X11)
    if os.environ.get("DISPLAY"):
        return "x11"
    
    return "unknown"


class LinuxScreenCapture(ScreenCaptureInterface):
    """
    Linux screen capture implementation.
    
    Features:
    - X11 support via mss (fast)
    - Wayland support via grim/slurp (if available)
    - Multi-monitor support
    - Fallback to scrot or ImageMagick import
    - PIL and pyautogui as final fallbacks
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """
        Initialize Linux screen capture.
        
        Args:
            config: Capture configuration
        """
        super().__init__(config)
        self._sct: Optional[mss.mss] = None
        self._display_server = _detect_display_server()
        self._capture_task: Optional[asyncio.Task] = None
        self._frame_queue: Optional[asyncio.Queue] = None
        self._selected_monitor: Optional[Dict[str, Any]] = None
        
        logger.info(f"Linux display server detected: {self._display_server}")
        
        # Determine capture method priority
        self._capture_methods = self._detect_available_methods()
        if not self._capture_methods:
            raise CaptureNotSupportedError(
                "No screen capture methods available on Linux. "
                "Install mss: pip install mss (for X11) or grim (for Wayland)"
            )
        
        logger.info(f"Linux capture methods available: {[m.value for m in self._capture_methods]}")
    
    def _detect_available_methods(self) -> List[CaptureMethod]:
        """Detect available capture methods in priority order."""
        methods = []
        
        if self._display_server == "x11" and MSS_AVAILABLE:
            methods.append(CaptureMethod.X11)
        
        if self._display_server == "wayland":
            # Check for grim (Wayland screenshot tool)
            if self._command_exists("grim"):
                methods.append(CaptureMethod.WAYLAND)
        
        # Fallback methods (work on both X11 and Wayland)
        if self._command_exists("scrot"):
            methods.append(CaptureMethod.LINUX_SCROT)
        
        if PIL_AVAILABLE:
            methods.append(CaptureMethod.PIL_SCREENSHOT)
        
        if PYAUTOGUI_AVAILABLE:
            methods.append(CaptureMethod.PIL_SCREENSHOT)
        
        return methods
    
    @staticmethod
    def _command_exists(command: str) -> bool:
        """Check if a command exists in PATH."""
        try:
            subprocess.run(
                ["which", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def get_capture_methods(self) -> List[CaptureMethod]:
        """Get list of available capture methods."""
        return self._capture_methods
    
    def get_available_displays(self) -> List[Dict[str, Any]]:
        """
        Get list of available displays/monitors on Linux.
        
        Returns:
            List of display info dicts
        """
        displays = []
        
        if self._display_server == "x11" and MSS_AVAILABLE:
            try:
                with mss.mss() as sct:
                    for i, monitor in enumerate(sct.monitors[1:], start=1):
                        display_info = {
                            "id": str(i),
                            "name": f"Display {i}",
                            "width": monitor["width"],
                            "height": monitor["height"],
                            "x": monitor["left"],
                            "y": monitor["top"],
                            "is_primary": i == 1,
                        }
                        displays.append(display_info)
                        logger.debug(f"Detected X11 monitor {i}: {monitor['width']}x{monitor['height']}")
            except Exception as e:
                logger.error(f"Failed to enumerate X11 displays: {e}")
        
        elif self._display_server == "wayland":
            # Wayland doesn't expose monitor info easily
            # Use a default primary display
            displays.append({
                "id": "1",
                "name": "Wayland Display",
                "width": 1920,  # Default, will be updated on capture
                "height": 1080,
                "x": 0,
                "y": 0,
                "is_primary": True,
            })
        
        return displays
    
    async def start(self) -> bool:
        """
        Start screen capture on Linux.
        
        Returns:
            bool: True if started successfully
        """
        if self._running:
            logger.warning("Capture already running")
            return True
        
        try:
            # Initialize based on display server
            if self._display_server == "x11" and MSS_AVAILABLE:
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
                    self._selected_monitor = self._sct.monitors[1]
                
                logger.info(
                    f"Linux X11 capture started on monitor: "
                    f"{self._selected_monitor['width']}x{self._selected_monitor['height']}"
                )
            
            elif self._display_server == "wayland":
                logger.info("Linux Wayland capture started (using grim)")
                self._selected_monitor = {"width": 1920, "height": 1080}  # Will be updated
            
            else:
                logger.warning(f"Unknown display server: {self._display_server}, using fallback")
            
            # Create frame queue
            self._frame_queue = asyncio.Queue(maxsize=self.config.buffer_size)
            
            # Start capture loop
            self._running = True
            self._capture_task = asyncio.create_task(self._capture_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Linux capture: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> None:
        """Stop screen capture and cleanup resources."""
        if not self._running:
            return
        
        logger.info("Stopping Linux screen capture...")
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
        
        logger.info("Linux screen capture stopped")
    
    async def _capture_loop(self) -> None:
        """Main capture loop running in background."""
        frame_interval = 1.0 / self.config.fps_target
        last_frame_time = 0.0
        
        logger.info(f"Capture loop started (target: {self.config.fps_target} FPS)")
        
        while self._running:
            try:
                current_time = time.time()
                
                # Rate limiting
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
                    
                    # Put in queue
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                            self.stats.frames_dropped += 1
                        except asyncio.QueueEmpty:
                            pass
                    
                    await self._frame_queue.put(frame)
                    self._notify_callbacks(frame)
                else:
                    self.stats.capture_errors += 1
                    await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self.stats.capture_errors += 1
                await asyncio.sleep(0.1)
        
        logger.info("Capture loop exited")
    
    async def _capture_frame_internal(self) -> Optional[CaptureFrame]:
        """Internal method to capture a single frame."""
        try:
            # X11 with mss (fast path)
            if self._display_server == "x11" and self._sct and self._selected_monitor:
                screenshot = self._sct.grab(self._selected_monitor)
                img_array = np.array(screenshot, dtype=np.uint8)
                
                return CaptureFrame(
                    data=img_array,
                    timestamp=datetime.now(),
                    display_id=self.config.display_id or "1",
                    width=screenshot.width,
                    height=screenshot.height,
                    format="BGRA",
                    metadata={"method": "mss_x11"}
                )
            
            # Wayland with grim (slower)
            elif self._display_server == "wayland" and self._command_exists("grim"):
                return await self._capture_wayland_grim()
            
            # Fallback to pyautogui
            elif PYAUTOGUI_AVAILABLE:
                screenshot = pyautogui.screenshot()
                img_array = np.array(screenshot, dtype=np.uint8)
                
                return CaptureFrame(
                    data=img_array,
                    timestamp=datetime.now(),
                    display_id=self.config.display_id or "1",
                    width=screenshot.width,
                    height=screenshot.height,
                    format="RGB",
                    metadata={"method": "pyautogui"}
                )
            
            else:
                logger.error("No capture method available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    async def _capture_wayland_grim(self) -> Optional[CaptureFrame]:
        """Capture frame on Wayland using grim tool."""
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            
            # Run grim to capture screenshot
            process = await asyncio.create_subprocess_exec(
                "grim", tmp_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await process.wait()
            
            if process.returncode == 0 and PIL_AVAILABLE:
                # Load image
                img = Image.open(tmp_path)
                img_array = np.array(img, dtype=np.uint8)
                
                frame = CaptureFrame(
                    data=img_array,
                    timestamp=datetime.now(),
                    display_id=self.config.display_id or "1",
                    width=img.width,
                    height=img.height,
                    format="RGB" if img.mode == "RGB" else "RGBA",
                    metadata={"method": "grim_wayland"}
                )
                
                # Cleanup
                os.unlink(tmp_path)
                return frame
            else:
                logger.error("grim capture failed")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                return None
                
        except Exception as e:
            logger.error(f"Wayland grim capture failed: {e}")
            return None
    
    async def get_frame(self, timeout: Optional[float] = None) -> Optional[CaptureFrame]:
        """Get the next captured frame from queue."""
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
        """Capture a single frame without starting continuous capture."""
        if self._display_server == "x11" and MSS_AVAILABLE:
            try:
                with mss.mss() as sct:
                    monitor_idx = int(display_id) if display_id else 1
                    monitor = sct.monitors[monitor_idx] if 0 < monitor_idx < len(sct.monitors) else sct.monitors[1]
                    
                    screenshot = sct.grab(monitor)
                    img_array = np.array(screenshot, dtype=np.uint8)
                    
                    return CaptureFrame(
                        data=img_array,
                        timestamp=datetime.now(),
                        display_id=display_id or "1",
                        width=screenshot.width,
                        height=screenshot.height,
                        format="BGRA",
                        metadata={"method": "mss_x11", "single_shot": True}
                    )
            except Exception as e:
                logger.error(f"X11 single frame capture failed: {e}")
                return None
        
        elif PYAUTOGUI_AVAILABLE:
            try:
                screenshot = pyautogui.screenshot()
                img_array = np.array(screenshot, dtype=np.uint8)
                
                return CaptureFrame(
                    data=img_array,
                    timestamp=datetime.now(),
                    display_id=display_id or "1",
                    width=screenshot.width,
                    height=screenshot.height,
                    format="RGB",
                    metadata={"method": "pyautogui", "single_shot": True}
                )
            except Exception as e:
                logger.error(f"pyautogui single frame capture failed: {e}")
                return None
        
        else:
            logger.error("No capture method available")
            return None
