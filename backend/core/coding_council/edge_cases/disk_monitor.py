"""
v77.0: Disk Monitor - Gap #32
==============================

Disk space monitoring and management:
- Real-time disk usage tracking
- Alert thresholds
- Automatic cleanup triggers
- Per-path monitoring
- Trend analysis

Author: Ironcliw v77.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class DiskAlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DiskUsage:
    """Disk usage information."""
    path: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent_used: float
    timestamp: float = field(default_factory=time.time)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "total_bytes": self.total_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "percent_used": self.percent_used,
            "total_gb": round(self.total_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class DiskAlert:
    """A disk space alert."""
    level: DiskAlertLevel
    path: str
    message: str
    usage: DiskUsage
    timestamp: float = field(default_factory=time.time)


@dataclass
class DiskThreshold:
    """Threshold configuration for disk alerts."""
    warning_percent: float = 80.0
    critical_percent: float = 90.0
    emergency_percent: float = 95.0
    min_free_gb: float = 1.0  # Minimum free space in GB


class DiskMonitor:
    """
    Disk space monitoring system.

    Features:
    - Multi-path monitoring
    - Configurable thresholds
    - Alert callbacks
    - Trend analysis
    - Cleanup triggers
    """

    def __init__(
        self,
        paths: Optional[List[str]] = None,
        threshold: Optional[DiskThreshold] = None,
        check_interval: float = 60.0,
    ):
        self.paths = paths or [str(Path.home())]
        self.threshold = threshold or DiskThreshold()
        self.check_interval = check_interval

        self._usage_history: Dict[str, List[DiskUsage]] = {}
        self._alerts: List[DiskAlert] = []
        self._alert_callbacks: List[Callable[[DiskAlert], Coroutine]] = []
        self._cleanup_callbacks: List[Callable[[str, float], Coroutine]] = []
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._max_history = 100

    async def start(self) -> None:
        """Start the disk monitor."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"[DiskMonitor] Started monitoring: {self.paths}")

    async def stop(self) -> None:
        """Stop the disk monitor."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[DiskMonitor] Stopped")

    def add_path(self, path: str) -> None:
        """Add a path to monitor."""
        if path not in self.paths:
            self.paths.append(path)
            logger.info(f"[DiskMonitor] Added path: {path}")

    def remove_path(self, path: str) -> None:
        """Remove a path from monitoring."""
        if path in self.paths:
            self.paths.remove(path)
            self._usage_history.pop(path, None)

    def on_alert(self, callback: Callable[[DiskAlert], Coroutine]) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def on_cleanup_needed(self, callback: Callable[[str, float], Coroutine]) -> None:
        """Register cleanup callback (path, target_free_gb)."""
        self._cleanup_callbacks.append(callback)

    async def check_disk(self, path: str) -> DiskUsage:
        """Check disk usage for a path."""
        try:
            # Get disk usage
            if os.path.exists(path):
                stat = shutil.disk_usage(path)
            else:
                # Use parent directory if path doesn't exist
                parent = str(Path(path).parent)
                stat = shutil.disk_usage(parent)

            usage = DiskUsage(
                path=path,
                total_bytes=stat.total,
                used_bytes=stat.used,
                free_bytes=stat.free,
                percent_used=(stat.used / stat.total) * 100 if stat.total > 0 else 0,
            )

            # Store in history
            if path not in self._usage_history:
                self._usage_history[path] = []
            self._usage_history[path].append(usage)

            # Trim history
            if len(self._usage_history[path]) > self._max_history:
                self._usage_history[path] = self._usage_history[path][-self._max_history:]

            # Check thresholds
            await self._check_thresholds(usage)

            return usage

        except Exception as e:
            logger.error(f"[DiskMonitor] Error checking {path}: {e}")
            raise

    async def check_all(self) -> Dict[str, DiskUsage]:
        """Check disk usage for all monitored paths."""
        results = {}
        for path in self.paths:
            try:
                results[path] = await self.check_disk(path)
            except Exception:
                pass
        return results

    def get_usage(self, path: str) -> Optional[DiskUsage]:
        """Get latest usage for a path."""
        history = self._usage_history.get(path, [])
        return history[-1] if history else None

    def get_all_usage(self) -> Dict[str, Optional[DiskUsage]]:
        """Get latest usage for all paths."""
        return {path: self.get_usage(path) for path in self.paths}

    def get_usage_trend(self, path: str, hours: float = 24.0) -> Dict[str, Any]:
        """Get usage trend for a path."""
        history = self._usage_history.get(path, [])
        if not history:
            return {"path": path, "trend": "unknown"}

        cutoff = time.time() - (hours * 3600)
        recent = [u for u in history if u.timestamp >= cutoff]

        if len(recent) < 2:
            return {"path": path, "trend": "insufficient_data"}

        # Calculate trend
        first = recent[0]
        last = recent[-1]
        change_gb = (last.used_bytes - first.used_bytes) / (1024 ** 3)
        time_diff_hours = (last.timestamp - first.timestamp) / 3600

        if time_diff_hours > 0:
            rate_gb_per_hour = change_gb / time_diff_hours
        else:
            rate_gb_per_hour = 0

        # Estimate time to full
        if rate_gb_per_hour > 0 and last.free_gb > 0:
            hours_to_full = last.free_gb / rate_gb_per_hour
        else:
            hours_to_full = float("inf")

        return {
            "path": path,
            "trend": "increasing" if change_gb > 0 else "decreasing" if change_gb < 0 else "stable",
            "change_gb": round(change_gb, 2),
            "rate_gb_per_hour": round(rate_gb_per_hour, 4),
            "hours_to_full": round(hours_to_full, 1) if hours_to_full != float("inf") else None,
            "data_points": len(recent),
        }

    def get_recent_alerts(self, count: int = 10) -> List[DiskAlert]:
        """Get recent alerts."""
        return self._alerts[-count:]

    async def _check_thresholds(self, usage: DiskUsage) -> None:
        """Check usage against thresholds and trigger alerts."""
        level = None
        message = ""

        # Check percentage thresholds
        if usage.percent_used >= self.threshold.emergency_percent:
            level = DiskAlertLevel.EMERGENCY
            message = f"EMERGENCY: Disk {usage.percent_used:.1f}% full at {usage.path}"
        elif usage.percent_used >= self.threshold.critical_percent:
            level = DiskAlertLevel.CRITICAL
            message = f"Critical: Disk {usage.percent_used:.1f}% full at {usage.path}"
        elif usage.percent_used >= self.threshold.warning_percent:
            level = DiskAlertLevel.WARNING
            message = f"Warning: Disk {usage.percent_used:.1f}% full at {usage.path}"

        # Check minimum free space
        if usage.free_gb < self.threshold.min_free_gb:
            if level is None or level.value < DiskAlertLevel.CRITICAL.value:
                level = DiskAlertLevel.CRITICAL
                message = f"Critical: Only {usage.free_gb:.2f}GB free at {usage.path}"

        # Trigger alert if threshold exceeded
        if level:
            alert = DiskAlert(
                level=level,
                path=usage.path,
                message=message,
                usage=usage,
            )
            self._alerts.append(alert)

            # Trim alerts
            if len(self._alerts) > 100:
                self._alerts = self._alerts[-100:]

            await self._trigger_alert(alert)

            # Trigger cleanup if critical
            if level in (DiskAlertLevel.CRITICAL, DiskAlertLevel.EMERGENCY):
                await self._trigger_cleanup(usage.path, self.threshold.min_free_gb * 2)

    async def _trigger_alert(self, alert: DiskAlert) -> None:
        """Trigger alert callbacks."""
        logger.warning(f"[DiskMonitor] {alert.message}")

        for callback in self._alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"[DiskMonitor] Alert callback error: {e}")

    async def _trigger_cleanup(self, path: str, target_free_gb: float) -> None:
        """Trigger cleanup callbacks."""
        logger.info(f"[DiskMonitor] Triggering cleanup for {path}, target: {target_free_gb}GB free")

        for callback in self._cleanup_callbacks:
            try:
                await callback(path, target_free_gb)
            except Exception as e:
                logger.error(f"[DiskMonitor] Cleanup callback error: {e}")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[DiskMonitor] Monitor loop error: {e}")
                await asyncio.sleep(5)

    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        all_usage = self.get_all_usage()

        return {
            "monitored_paths": len(self.paths),
            "paths": {
                path: usage.to_dict() if usage else None
                for path, usage in all_usage.items()
            },
            "thresholds": {
                "warning_percent": self.threshold.warning_percent,
                "critical_percent": self.threshold.critical_percent,
                "min_free_gb": self.threshold.min_free_gb,
            },
            "recent_alerts": len(self._alerts),
        }


# Global disk monitor
_monitor: Optional[DiskMonitor] = None


def get_disk_monitor() -> DiskMonitor:
    """Get global disk monitor."""
    global _monitor
    if _monitor is None:
        _monitor = DiskMonitor()
    return _monitor
