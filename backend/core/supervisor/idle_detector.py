#!/usr/bin/env python3
"""
JARVIS Idle Detector
=====================

System activity monitoring for the Self-Updating Lifecycle Manager.
Detects user idle state to trigger silent updates.

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

from .supervisor_config import SupervisorConfig, get_supervisor_config

logger = logging.getLogger(__name__)


class ActivityLevel(str, Enum):
    """System activity levels."""
    ACTIVE = "active"
    IDLE = "idle"
    DEEP_IDLE = "deep_idle"
    UNKNOWN = "unknown"


@dataclass
class IdleState:
    """Current idle state information."""
    level: ActivityLevel = ActivityLevel.UNKNOWN
    idle_seconds: float = 0.0
    last_activity: Optional[datetime] = None
    threshold_seconds: int = 0
    is_idle: bool = False
    checked_at: datetime = None
    
    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now()


class IdleDetector:
    """
    System activity monitoring.
    
    Features:
    - macOS IOKit integration (async)
    - Configurable idle threshold
    - Activity pattern detection
    - Do-not-disturb window support
    
    Example:
        >>> detector = IdleDetector(config)
        >>> is_idle = await detector.is_system_idle()
        >>> if is_idle:
        ...     print("User is idle, can perform silent update")
    """
    
    def __init__(self, config: Optional[SupervisorConfig] = None):
        """
        Initialize the idle detector.
        
        Args:
            config: Supervisor configuration
        """
        self.config = config or get_supervisor_config()
        
        self._last_active: datetime = datetime.now()
        self._idle_start: Optional[datetime] = None
        self._consecutive_idle_seconds: float = 0.0
        
        self._on_idle: list[Callable[[], None]] = []
        self._on_active: list[Callable[[], None]] = []
        
        # Load macOS frameworks if available
        self._iokit: Optional[ctypes.CDLL] = None
        self._core_foundation: Optional[ctypes.CDLL] = None
        self._initialized = False
        
        if platform.system() == "Darwin":
            self._init_macos()
        
        logger.info("🔧 Idle detector initialized")
    
    def _init_macos(self) -> None:
        """Initialize macOS IOKit for HID idle detection."""
        try:
            # Load IOKit
            iokit_path = ctypes.util.find_library("IOKit")
            if iokit_path:
                self._iokit = ctypes.CDLL(iokit_path)
            
            # Load CoreFoundation
            cf_path = ctypes.util.find_library("CoreFoundation")
            if cf_path:
                self._core_foundation = ctypes.CDLL(cf_path)
            
            if self._iokit and self._core_foundation:
                self._initialized = True
                logger.debug("✅ macOS IOKit initialized")
            else:
                logger.warning("⚠️ Could not load IOKit/CoreFoundation")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize macOS idle detection: {e}")
    
    def _get_macos_idle_time(self) -> float:
        """
        Get idle time in seconds using macOS IOKit.
        
        Returns:
            Idle time in seconds, or -1 if unavailable
        """
        if not self._initialized or not self._iokit:
            return -1.0
        
        try:
            # Define IOKit functions
            IOServiceGetMatchingService = self._iokit.IOServiceGetMatchingService
            IOServiceMatching = self._iokit.IOServiceMatching
            IORegistryEntryCreateCFProperty = self._iokit.IORegistryEntryCreateCFProperty
            IOObjectRelease = self._iokit.IOObjectRelease
            
            # Define CoreFoundation functions
            CFStringCreateWithCString = self._core_foundation.CFStringCreateWithCString
            CFNumberGetValue = self._core_foundation.CFNumberGetValue
            CFRelease = self._core_foundation.CFRelease
            
            # Set return types
            IOServiceMatching.restype = ctypes.c_void_p
            IOServiceGetMatchingService.restype = ctypes.c_uint
            IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
            CFStringCreateWithCString.restype = ctypes.c_void_p
            
            # Get HID idle time service
            kCFAllocatorDefault = ctypes.c_void_p(0)
            kCFStringEncodingASCII = 0x0600
            
            matching = IOServiceMatching(b"IOHIDSystem")
            if not matching:
                return -1.0
            
            service = IOServiceGetMatchingService(0, matching)
            if not service:
                return -1.0
            
            try:
                # Create key string
                key = CFStringCreateWithCString(
                    kCFAllocatorDefault,
                    b"HIDIdleTime",
                    kCFStringEncodingASCII,
                )
                
                if not key:
                    return -1.0
                
                try:
                    # Get idle time property
                    idle_time_ref = IORegistryEntryCreateCFProperty(
                        service,
                        key,
                        kCFAllocatorDefault,
                        0,
                    )
                    
                    if not idle_time_ref:
                        return -1.0
                    
                    try:
                        # Extract value (nanoseconds)
                        idle_ns = ctypes.c_int64()
                        kCFNumberSInt64Type = 4
                        
                        success = CFNumberGetValue(
                            idle_time_ref,
                            kCFNumberSInt64Type,
                            ctypes.byref(idle_ns),
                        )
                        
                        if success:
                            return idle_ns.value / 1_000_000_000.0
                        return -1.0
                        
                    finally:
                        CFRelease(idle_time_ref)
                        
                finally:
                    CFRelease(key)
                    
            finally:
                IOObjectRelease(service)
                
        except Exception as e:
            logger.debug(f"Failed to get macOS idle time: {e}")
            return -1.0
    
    async def _get_idle_time_async(self) -> float:
        """Get idle time asynchronously."""
        loop = asyncio.get_event_loop()
        
        if platform.system() == "Darwin" and self._initialized:
            # Run in executor to not block
            idle_time = await loop.run_in_executor(
                None,
                self._get_macos_idle_time,
            )
            if idle_time >= 0:
                return idle_time
        
        # Fallback: use tracking-based approach
        return self._consecutive_idle_seconds
    
    def _is_in_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        if not self.config.idle.enabled:
            return False
        
        # Note: quiet_hours config would need to be added to IdleConfig
        # For now, always return False
        return False
    
    def record_activity(self) -> None:
        """Record user activity (call from input handlers)."""
        self._last_active = datetime.now()
        self._idle_start = None
        self._consecutive_idle_seconds = 0.0
    
    def get_idle_state(self) -> IdleState:
        """Get current idle state synchronously."""
        idle_time = self._get_macos_idle_time()
        
        if idle_time < 0:
            # Fallback
            idle_time = self._consecutive_idle_seconds
        
        threshold = self.config.idle.threshold_seconds
        min_consecutive = self.config.idle.min_consecutive_seconds
        
        is_idle = idle_time >= threshold
        
        # Determine activity level
        if idle_time < 60:
            level = ActivityLevel.ACTIVE
        elif idle_time < threshold:
            level = ActivityLevel.IDLE
        else:
            level = ActivityLevel.DEEP_IDLE
        
        return IdleState(
            level=level,
            idle_seconds=idle_time,
            last_activity=self._last_active,
            threshold_seconds=threshold,
            is_idle=is_idle and idle_time >= min_consecutive,
        )
    
    async def is_system_idle(self) -> bool:
        """
        Check if the system is in idle state.
        
        Returns:
            True if system has been idle longer than threshold
        """
        if not self.config.idle.enabled:
            return False
        
        idle_time = await self._get_idle_time_async()
        threshold = self.config.idle.threshold_seconds
        min_consecutive = self.config.idle.min_consecutive_seconds
        
        # Track consecutive idle
        if idle_time < 60:
            # Activity detected (< 1 minute idle)
            if self._idle_start is not None:
                self._idle_start = None
                self._consecutive_idle_seconds = 0.0
                
                # Notify active callbacks
                for callback in self._on_active:
                    try:
                        callback()
                    except Exception as e:
                        logger.debug(f"Active callback error: {e}")
        else:
            # Currently idle
            if self._idle_start is None:
                self._idle_start = datetime.now()
                self._consecutive_idle_seconds = 0.0
            else:
                self._consecutive_idle_seconds = (
                    datetime.now() - self._idle_start
                ).total_seconds()
        
        # Check if idle long enough
        is_idle = (
            idle_time >= threshold and
            self._consecutive_idle_seconds >= min_consecutive
        )
        
        if is_idle:
            # First time crossing threshold
            if self._consecutive_idle_seconds < threshold + 60:
                for callback in self._on_idle:
                    try:
                        callback()
                    except Exception as e:
                        logger.debug(f"Idle callback error: {e}")
        
        return is_idle
    
    async def wait_for_idle(
        self,
        timeout: Optional[float] = None,
        check_interval: float = 60.0,
    ) -> bool:
        """
        Wait for system to become idle.
        
        Args:
            timeout: Maximum time to wait (None = wait forever)
            check_interval: Seconds between checks
            
        Returns:
            True if system became idle, False if timeout
        """
        start = time.time()
        
        while True:
            if await self.is_system_idle():
                return True
            
            if timeout and (time.time() - start) >= timeout:
                return False
            
            await asyncio.sleep(check_interval)
    
    def on_idle(self, callback: Callable[[], None]) -> None:
        """Register callback for when system becomes idle."""
        self._on_idle.append(callback)
    
    def on_active(self, callback: Callable[[], None]) -> None:
        """Register callback for when user becomes active."""
        self._on_active.append(callback)


async def check_idle(config: Optional[SupervisorConfig] = None) -> bool:
    """Quick utility to check if system is idle."""
    detector = IdleDetector(config)
    return await detector.is_system_idle()
