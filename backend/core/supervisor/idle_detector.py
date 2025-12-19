#!/usr/bin/env python3
"""
JARVIS Idle Detector
=====================

System activity monitoring for the Self-Updating Lifecycle Manager.
Detects user idle state to trigger silent updates.

Uses subprocess-based approach with `ioreg` for macOS idle detection,
which is safer and more reliable than direct IOKit ctypes bindings.

Author: JARVIS System
Version: 1.1.0
"""

from __future__ import annotations

import asyncio
import logging
import platform
import re
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
    System activity monitoring using subprocess-based detection.
    
    Uses macOS `ioreg` command for safe, reliable idle time detection
    without the instability of direct ctypes IOKit bindings.
    
    Features:
    - Safe subprocess-based detection (no segfaults)
    - Configurable idle threshold
    - Activity pattern tracking
    - Async-first design
    
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
        self._last_idle_check: float = 0.0
        self._cached_idle_time: float = 0.0
        self._cache_ttl: float = 5.0  # Cache for 5 seconds
        
        self._on_idle: list[Callable[[], None]] = []
        self._on_active: list[Callable[[], None]] = []
        
        self._is_macos = platform.system() == "Darwin"
        
        logger.info("🔧 Idle detector initialized")
    
    async def _get_macos_idle_time_subprocess(self) -> float:
        """
        Get idle time using ioreg subprocess (safe alternative to ctypes).
        
        Returns:
            Idle time in seconds, or -1 if unavailable
        """
        try:
            # Use ioreg to get HIDIdleTime - this is safe and reliable
            process = await asyncio.create_subprocess_exec(
                "ioreg",
                "-c", "IOHIDSystem",
                "-d", "4",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=5.0,
            )
            
            if process.returncode != 0:
                return -1.0
            
            # Parse output for HIDIdleTime
            output = stdout.decode("utf-8", errors="ignore")
            
            # Look for HIDIdleTime = <number>
            match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', output)
            if match:
                idle_ns = int(match.group(1))
                return idle_ns / 1_000_000_000.0  # Convert nanoseconds to seconds
            
            return -1.0
            
        except asyncio.TimeoutError:
            logger.debug("ioreg command timed out")
            return -1.0
        except FileNotFoundError:
            logger.debug("ioreg command not found")
            return -1.0
        except Exception as e:
            logger.debug(f"Failed to get macOS idle time: {e}")
            return -1.0
    
    async def _get_idle_time_async(self) -> float:
        """Get idle time asynchronously with caching."""
        current_time = time.time()
        
        # Return cached value if still valid
        if current_time - self._last_idle_check < self._cache_ttl:
            return self._cached_idle_time
        
        if self._is_macos:
            idle_time = await self._get_macos_idle_time_subprocess()
            if idle_time >= 0:
                self._cached_idle_time = idle_time
                self._last_idle_check = current_time
                return idle_time
        
        # Fallback: use tracking-based approach
        return self._consecutive_idle_seconds
    
    def _is_in_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours."""
        if not self.config.idle.enabled:
            return False
        
        # Could add quiet hours support here
        return False
    
    def record_activity(self) -> None:
        """Record user activity (call from input handlers)."""
        self._last_active = datetime.now()
        self._idle_start = None
        self._consecutive_idle_seconds = 0.0
    
    async def get_idle_state(self) -> IdleState:
        """Get current idle state."""
        idle_time = await self._get_idle_time_async()
        
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
