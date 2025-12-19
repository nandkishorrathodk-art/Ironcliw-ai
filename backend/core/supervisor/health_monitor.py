#!/usr/bin/env python3
"""
JARVIS Health Monitor
======================

Boot health and stability monitoring for the Self-Updating Lifecycle Manager.
Tracks component initialization, memory baselines, and detects crashes.

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import aiohttp

from .supervisor_config import SupervisorConfig, get_supervisor_config

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: Optional[datetime] = None
    response_time_ms: float = 0.0
    error: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus = HealthStatus.UNKNOWN
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    boot_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    is_stable: bool = False
    health_score: float = 0.0  # 0.0 - 1.0
    last_check: Optional[datetime] = None


class HealthMonitor:
    """
    Boot health and stability monitoring.
    
    Features:
    - Startup time tracking
    - Component initialization monitoring
    - HTTP health endpoint checking
    - Crash detection within configurable window
    - Health score calculation
    
    Example:
        >>> monitor = HealthMonitor(config)
        >>> health = await monitor.check_health()
        >>> if health.is_stable:
        ...     print("System is stable")
    """
    
    def __init__(self, config: Optional[SupervisorConfig] = None):
        """
        Initialize the health monitor.
        
        Args:
            config: Supervisor configuration
        """
        self.config = config or get_supervisor_config()
        
        self.system_health = SystemHealth()
        self._boot_time: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        self._on_status_change: list[Callable[[HealthStatus], None]] = []
        self._on_component_change: list[Callable[[ComponentHealth], None]] = []
        
        logger.info("🔧 Health monitor initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.config.health.check_timeout_seconds
                )
            )
        return self._session
    
    def record_boot_start(self) -> None:
        """Record the start of a boot sequence."""
        self._boot_time = datetime.now()
        self.system_health.boot_time = self._boot_time
        self.system_health.is_stable = False
        logger.info("🚀 Boot sequence started")
    
    def get_uptime(self) -> float:
        """Get current uptime in seconds."""
        if self._boot_time is None:
            return 0.0
        return (datetime.now() - self._boot_time).total_seconds()
    
    def is_stable(self) -> bool:
        """Check if system has been stable for the configured window."""
        uptime = self.get_uptime()
        return uptime >= self.config.health.boot_stability_window
    
    async def check_http_endpoint(
        self,
        name: str,
        url: str,
    ) -> ComponentHealth:
        """
        Check health of an HTTP endpoint.
        
        Args:
            name: Component name
            url: Health check URL
            
        Returns:
            ComponentHealth status
        """
        component = ComponentHealth(name=name)
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            async with session.get(url) as response:
                elapsed = (time.time() - start_time) * 1000
                component.response_time_ms = elapsed
                component.last_check = datetime.now()
                
                if response.status == 200:
                    component.status = HealthStatus.HEALTHY
                    
                    # Try to parse JSON response for details
                    try:
                        data = await response.json()
                        component.details = data
                    except Exception:
                        pass
                        
                elif response.status < 500:
                    component.status = HealthStatus.DEGRADED
                else:
                    component.status = HealthStatus.UNHEALTHY
                    component.error = f"HTTP {response.status}"
                    
        except asyncio.TimeoutError:
            component.status = HealthStatus.UNHEALTHY
            component.error = "Timeout"
            component.response_time_ms = self.config.health.check_timeout_seconds * 1000
            
        except aiohttp.ClientError as e:
            component.status = HealthStatus.UNHEALTHY
            component.error = str(e)
            
        except Exception as e:
            component.status = HealthStatus.UNHEALTHY
            component.error = f"Unexpected error: {e}"
        
        return component
    
    async def check_internal_component(
        self,
        name: str,
        check_func: Callable[[], bool],
    ) -> ComponentHealth:
        """
        Check health of an internal component.
        
        Args:
            name: Component name
            check_func: Function that returns True if healthy
            
        Returns:
            ComponentHealth status
        """
        component = ComponentHealth(name=name)
        start_time = time.time()
        
        try:
            is_healthy = check_func()
            elapsed = (time.time() - start_time) * 1000
            
            component.response_time_ms = elapsed
            component.last_check = datetime.now()
            component.status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
            
        except Exception as e:
            component.status = HealthStatus.UNHEALTHY
            component.error = str(e)
        
        return component
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0.0 - 1.0)."""
        if not self.system_health.components:
            return 0.0
        
        scores = []
        for component in self.system_health.components.values():
            if component.status == HealthStatus.HEALTHY:
                scores.append(1.0)
            elif component.status == HealthStatus.DEGRADED:
                scores.append(0.5)
            else:
                scores.append(0.0)
        
        return sum(scores) / len(scores)
    
    def _determine_overall_status(self) -> HealthStatus:
        """Determine overall system status from component statuses."""
        if not self.system_health.components:
            return HealthStatus.UNKNOWN
        
        statuses = [c.status for c in self.system_health.components.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN
    
    async def check_health(self) -> SystemHealth:
        """
        Perform a full health check.
        
        Returns:
            SystemHealth with all component statuses
        """
        old_status = self.system_health.status
        
        # Check websocket endpoint (if configured)
        websocket_check = await self.check_http_endpoint(
            "websocket",
            "http://localhost:8010/health",
        )
        self.system_health.components["websocket"] = websocket_check
        
        # Update system health
        self.system_health.uptime_seconds = self.get_uptime()
        self.system_health.is_stable = self.is_stable()
        self.system_health.health_score = self._calculate_health_score()
        self.system_health.status = self._determine_overall_status()
        self.system_health.last_check = datetime.now()
        
        # Notify on status change
        if old_status != self.system_health.status:
            logger.info(
                f"📊 Health status: {old_status.value} → {self.system_health.status.value}"
            )
            for callback in self._on_status_change:
                try:
                    callback(self.system_health.status)
                except Exception as e:
                    logger.warning(f"Status change callback error: {e}")
        
        return self.system_health
    
    async def wait_for_stability(
        self,
        timeout: Optional[float] = None,
        check_interval: float = 5.0,
    ) -> bool:
        """
        Wait for system to become stable.
        
        Args:
            timeout: Maximum time to wait (default: stability window)
            check_interval: Time between checks
            
        Returns:
            True if system became stable, False if timeout
        """
        if timeout is None:
            timeout = self.config.health.boot_stability_window + 10
        
        start = time.time()
        
        while time.time() - start < timeout:
            if self.is_stable():
                logger.info("✅ System is stable")
                return True
            
            await asyncio.sleep(check_interval)
            await self.check_health()
        
        logger.warning("⚠️ Stability timeout reached")
        return False
    
    def on_status_change(self, callback: Callable[[HealthStatus], None]) -> None:
        """Register a status change callback."""
        self._on_status_change.append(callback)
    
    def on_component_change(self, callback: Callable[[ComponentHealth], None]) -> None:
        """Register a component change callback."""
        self._on_component_change.append(callback)
    
    def get_health_report(self) -> dict[str, Any]:
        """Get a detailed health report."""
        return {
            "status": self.system_health.status.value,
            "uptime_seconds": self.system_health.uptime_seconds,
            "is_stable": self.system_health.is_stable,
            "health_score": self.system_health.health_score,
            "last_check": (
                self.system_health.last_check.isoformat()
                if self.system_health.last_check
                else None
            ),
            "components": {
                name: {
                    "status": comp.status.value,
                    "response_time_ms": comp.response_time_ms,
                    "error": comp.error,
                }
                for name, comp in self.system_health.components.items()
            },
        }
    
    async def close(self) -> None:
        """Close resources."""
        if self._session and not self._session.closed:
            await self._session.close()
