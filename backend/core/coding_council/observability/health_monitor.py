"""
v77.0: Health Monitor - Gap #31
================================

Health monitoring and alerting:
- Component health checks
- Dependency monitoring
- Alert thresholds
- Health aggregation
- Readiness/liveness probes

Author: Ironcliw v77.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status of a component."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    last_check: float = 0.0
    check_duration_ms: float = 0.0
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check,
            "check_duration_ms": self.check_duration_ms,
            "consecutive_failures": self.consecutive_failures,
            "metadata": self.metadata,
        }


@dataclass
class HealthCheck:
    """A health check definition."""
    name: str
    check_fn: Callable[[], Coroutine[Any, Any, bool]]
    interval: float = 30.0  # Check interval in seconds
    timeout: float = 10.0   # Check timeout
    failure_threshold: int = 3  # Failures before unhealthy
    success_threshold: int = 1  # Successes to recover
    critical: bool = True   # If critical, affects overall health


class HealthMonitor:
    """
    Health monitoring system.

    Features:
    - Component health tracking
    - Periodic health checks
    - Health aggregation
    - Alert callbacks
    - Readiness/liveness endpoints
    """

    def __init__(self, service_name: str = "coding_council"):
        self.service_name = service_name
        self._checks: Dict[str, HealthCheck] = {}
        self._health: Dict[str, ComponentHealth] = {}
        self._alert_callbacks: List[Callable[[str, HealthStatus, str], Coroutine]] = []
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the health monitor."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[HealthMonitor] Started")

    async def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[HealthMonitor] Stopped")

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, bool]],
        interval: float = 30.0,
        timeout: float = 10.0,
        failure_threshold: int = 3,
        critical: bool = True,
    ) -> None:
        """Register a health check."""
        check = HealthCheck(
            name=name,
            check_fn=check_fn,
            interval=interval,
            timeout=timeout,
            failure_threshold=failure_threshold,
            critical=critical,
        )
        self._checks[name] = check
        self._health[name] = ComponentHealth(name=name)
        logger.info(f"[HealthMonitor] Registered check: {name}")

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._health.pop(name, None)

    def on_alert(self, callback: Callable[[str, HealthStatus, str], Coroutine]) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    async def check_health(self, name: str) -> ComponentHealth:
        """Run a specific health check."""
        if name not in self._checks:
            return ComponentHealth(name=name, status=HealthStatus.UNKNOWN, message="Check not found")

        check = self._checks[name]
        health = self._health[name]
        old_status = health.status

        start_time = time.time()

        try:
            result = await asyncio.wait_for(check.check_fn(), timeout=check.timeout)

            health.check_duration_ms = (time.time() - start_time) * 1000
            health.last_check = time.time()

            if result:
                health.consecutive_failures = 0
                health.status = HealthStatus.HEALTHY
                health.message = "OK"
            else:
                health.consecutive_failures += 1
                if health.consecutive_failures >= check.failure_threshold:
                    health.status = HealthStatus.UNHEALTHY
                else:
                    health.status = HealthStatus.DEGRADED
                health.message = f"Check returned false (failures: {health.consecutive_failures})"

        except asyncio.TimeoutError:
            health.consecutive_failures += 1
            health.check_duration_ms = (time.time() - start_time) * 1000
            health.last_check = time.time()

            if health.consecutive_failures >= check.failure_threshold:
                health.status = HealthStatus.UNHEALTHY
            else:
                health.status = HealthStatus.DEGRADED
            health.message = f"Check timed out after {check.timeout}s"

        except Exception as e:
            health.consecutive_failures += 1
            health.check_duration_ms = (time.time() - start_time) * 1000
            health.last_check = time.time()

            if health.consecutive_failures >= check.failure_threshold:
                health.status = HealthStatus.UNHEALTHY
            else:
                health.status = HealthStatus.DEGRADED
            health.message = f"Check error: {str(e)}"

        # Trigger alerts on status change
        if health.status != old_status:
            await self._trigger_alert(name, health.status, health.message)

        return health

    async def check_all(self) -> Dict[str, ComponentHealth]:
        """Run all health checks."""
        tasks = [self.check_health(name) for name in self._checks]
        await asyncio.gather(*tasks, return_exceptions=True)
        return self._health.copy()

    def get_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health status for a component."""
        return self._health.get(name)

    def get_all_health(self) -> Dict[str, ComponentHealth]:
        """Get health status for all components."""
        return self._health.copy()

    def get_overall_health(self) -> HealthStatus:
        """
        Get overall system health.

        Returns UNHEALTHY if any critical component is unhealthy.
        Returns DEGRADED if any component is degraded.
        Returns HEALTHY if all components are healthy.
        """
        has_degraded = False

        for name, health in self._health.items():
            check = self._checks.get(name)
            if not check:
                continue

            if health.status == HealthStatus.UNHEALTHY and check.critical:
                return HealthStatus.UNHEALTHY

            if health.status == HealthStatus.DEGRADED:
                has_degraded = True

        return HealthStatus.DEGRADED if has_degraded else HealthStatus.HEALTHY

    def is_ready(self) -> bool:
        """
        Check if service is ready to receive traffic.

        Used for Kubernetes readiness probes.
        """
        overall = self.get_overall_health()
        return overall in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def is_live(self) -> bool:
        """
        Check if service is alive.

        Used for Kubernetes liveness probes.
        Returns False only if critical components are unhealthy.
        """
        for name, health in self._health.items():
            check = self._checks.get(name)
            if check and check.critical and health.status == HealthStatus.UNHEALTHY:
                return False
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        components = {
            name: health.to_dict()
            for name, health in self._health.items()
        }

        by_status = {}
        for health in self._health.values():
            status = health.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "service": self.service_name,
            "overall_status": self.get_overall_health().value,
            "is_ready": self.is_ready(),
            "is_live": self.is_live(),
            "timestamp": time.time(),
            "by_status": by_status,
            "components": components,
        }

    async def _trigger_alert(self, name: str, status: HealthStatus, message: str) -> None:
        """Trigger alert callbacks."""
        logger.warning(f"[HealthMonitor] Alert: {name} is {status.value}: {message}")

        for callback in self._alert_callbacks:
            try:
                await callback(name, status, message)
            except Exception as e:
                logger.error(f"[HealthMonitor] Alert callback error: {e}")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        # Track last check times
        last_checks: Dict[str, float] = {}

        while self._running:
            try:
                now = time.time()

                for name, check in self._checks.items():
                    last_check = last_checks.get(name, 0)

                    if now - last_check >= check.interval:
                        await self.check_health(name)
                        last_checks[name] = now

                await asyncio.sleep(1)  # Check every second for due checks

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[HealthMonitor] Monitor loop error: {e}")
                await asyncio.sleep(5)


# Global health monitor
_monitor: Optional[HealthMonitor] = None


def get_health_monitor(service_name: str = "coding_council") -> HealthMonitor:
    """Get global health monitor."""
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor(service_name)
    return _monitor


def health_check(
    name: str,
    interval: float = 30.0,
    timeout: float = 10.0,
    critical: bool = True,
):
    """
    Decorator to register a function as a health check.

    Usage:
        @health_check("database")
        async def check_database():
            # return True if healthy
            return await db.ping()
    """
    def decorator(func: Callable[[], Coroutine[Any, Any, bool]]) -> Callable[[], Coroutine[Any, Any, bool]]:
        monitor = get_health_monitor()
        monitor.register_check(
            name=name,
            check_fn=func,
            interval=interval,
            timeout=timeout,
            critical=critical,
        )
        return func

    return decorator
