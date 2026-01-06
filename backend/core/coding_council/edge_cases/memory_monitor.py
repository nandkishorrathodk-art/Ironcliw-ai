"""
v77.0: Memory Monitor - Gap #34
================================

Memory pressure monitoring:
- Process memory tracking
- System memory monitoring
- Alert thresholds
- Garbage collection triggers
- Memory leak detection

Author: JARVIS v77.0
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryAlertLevel(Enum):
    """Memory alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MemoryUsage:
    """Memory usage information."""
    timestamp: float = field(default_factory=time.time)
    # Process memory
    rss_bytes: int = 0  # Resident Set Size
    vms_bytes: int = 0  # Virtual Memory Size
    heap_bytes: int = 0  # Python heap
    # System memory
    system_total: int = 0
    system_available: int = 0
    system_percent: float = 0.0
    # GC stats
    gc_gen0_count: int = 0
    gc_gen1_count: int = 0
    gc_gen2_count: int = 0
    gc_collected: int = 0

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 ** 2)

    @property
    def vms_mb(self) -> float:
        return self.vms_bytes / (1024 ** 2)

    @property
    def system_total_gb(self) -> float:
        return self.system_total / (1024 ** 3)

    @property
    def system_available_gb(self) -> float:
        return self.system_available / (1024 ** 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "process": {
                "rss_mb": round(self.rss_mb, 2),
                "vms_mb": round(self.vms_mb, 2),
                "heap_bytes": self.heap_bytes,
            },
            "system": {
                "total_gb": round(self.system_total_gb, 2),
                "available_gb": round(self.system_available_gb, 2),
                "percent_used": round(self.system_percent, 1),
            },
            "gc": {
                "gen0_count": self.gc_gen0_count,
                "gen1_count": self.gc_gen1_count,
                "gen2_count": self.gc_gen2_count,
                "collected": self.gc_collected,
            },
        }


@dataclass
class MemoryAlert:
    """A memory alert."""
    level: MemoryAlertLevel
    message: str
    usage: MemoryUsage
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryThreshold:
    """Threshold configuration for memory alerts."""
    warning_percent: float = 70.0   # Process memory % of limit
    critical_percent: float = 85.0
    emergency_percent: float = 95.0
    system_warning_percent: float = 80.0  # System memory
    system_critical_percent: float = 90.0
    max_rss_mb: float = 2048.0  # Max process RSS


@dataclass
class MemoryPressure:
    """Memory pressure state."""
    level: int = 0  # 0=normal, 1=low, 2=high, 3=critical
    should_gc: bool = False
    should_reduce_cache: bool = False
    should_shed_load: bool = False


class MemoryMonitor:
    """
    Memory monitoring system.

    Features:
    - Process memory tracking
    - System memory monitoring
    - GC statistics and triggers
    - Alert callbacks
    - Memory pressure management
    """

    def __init__(
        self,
        threshold: Optional[MemoryThreshold] = None,
        check_interval: float = 30.0,
        enable_auto_gc: bool = True,
    ):
        self.threshold = threshold or MemoryThreshold()
        self.check_interval = check_interval
        self.enable_auto_gc = enable_auto_gc

        self._usage_history: List[MemoryUsage] = []
        self._alerts: List[MemoryAlert] = []
        self._alert_callbacks: List[Callable[[MemoryAlert], Coroutine]] = []
        self._pressure_callbacks: List[Callable[[MemoryPressure], Coroutine]] = []
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._max_history = 100
        self._last_gc_time = 0.0
        self._gc_cooldown = 60.0  # Minimum seconds between GCs

    async def start(self) -> None:
        """Start the memory monitor."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[MemoryMonitor] Started")

    async def stop(self) -> None:
        """Stop the memory monitor."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[MemoryMonitor] Stopped")

    def on_alert(self, callback: Callable[[MemoryAlert], Coroutine]) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def on_pressure(self, callback: Callable[[MemoryPressure], Coroutine]) -> None:
        """Register pressure callback."""
        self._pressure_callbacks.append(callback)

    async def check_memory(self) -> MemoryUsage:
        """Check current memory usage."""
        usage = MemoryUsage()

        try:
            # Process memory using resource module
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            usage.rss_bytes = rusage.ru_maxrss
            # On macOS ru_maxrss is in bytes, on Linux in KB
            if sys.platform != "darwin":
                usage.rss_bytes *= 1024

            # Try to get more detailed info
            try:
                import psutil
                proc = psutil.Process()
                mem_info = proc.memory_info()
                usage.rss_bytes = mem_info.rss
                usage.vms_bytes = mem_info.vms

                # System memory
                sys_mem = psutil.virtual_memory()
                usage.system_total = sys_mem.total
                usage.system_available = sys_mem.available
                usage.system_percent = sys_mem.percent
            except ImportError:
                # psutil not available, use basic info
                pass

            # GC stats
            gc_stats = gc.get_stats()
            if len(gc_stats) >= 3:
                usage.gc_gen0_count = gc_stats[0].get("collections", 0)
                usage.gc_gen1_count = gc_stats[1].get("collections", 0)
                usage.gc_gen2_count = gc_stats[2].get("collections", 0)
                usage.gc_collected = sum(s.get("collected", 0) for s in gc_stats)

            # Python heap (approximate)
            usage.heap_bytes = sys.getsizeof(gc.get_objects())

        except Exception as e:
            logger.error(f"[MemoryMonitor] Error getting memory info: {e}")

        # Store in history
        self._usage_history.append(usage)
        if len(self._usage_history) > self._max_history:
            self._usage_history = self._usage_history[-self._max_history:]

        # Check thresholds
        await self._check_thresholds(usage)

        return usage

    def get_current_usage(self) -> Optional[MemoryUsage]:
        """Get most recent memory usage."""
        return self._usage_history[-1] if self._usage_history else None

    def get_pressure(self) -> MemoryPressure:
        """Get current memory pressure state."""
        usage = self.get_current_usage()
        if not usage:
            return MemoryPressure()

        pressure = MemoryPressure()

        # Check process memory
        process_percent = (usage.rss_mb / self.threshold.max_rss_mb) * 100

        if process_percent >= self.threshold.emergency_percent:
            pressure.level = 3
            pressure.should_gc = True
            pressure.should_reduce_cache = True
            pressure.should_shed_load = True
        elif process_percent >= self.threshold.critical_percent:
            pressure.level = 2
            pressure.should_gc = True
            pressure.should_reduce_cache = True
        elif process_percent >= self.threshold.warning_percent:
            pressure.level = 1
            pressure.should_gc = True

        # Check system memory
        if usage.system_percent >= self.threshold.system_critical_percent:
            pressure.level = max(pressure.level, 2)
            pressure.should_reduce_cache = True
        elif usage.system_percent >= self.threshold.system_warning_percent:
            pressure.level = max(pressure.level, 1)

        return pressure

    async def force_gc(self) -> Dict[str, int]:
        """Force garbage collection."""
        now = time.time()

        # Check cooldown
        if now - self._last_gc_time < self._gc_cooldown:
            return {"skipped": True, "reason": "cooldown"}

        self._last_gc_time = now

        # Collect stats before
        before = gc.get_count()

        # Force full collection
        collected = gc.collect(generation=2)

        # Stats after
        after = gc.get_count()

        result = {
            "collected": collected,
            "before": before,
            "after": after,
        }

        logger.info(f"[MemoryMonitor] GC collected {collected} objects")
        return result

    def get_memory_trend(self, minutes: float = 30.0) -> Dict[str, Any]:
        """Get memory usage trend."""
        if not self._usage_history:
            return {"trend": "unknown"}

        cutoff = time.time() - (minutes * 60)
        recent = [u for u in self._usage_history if u.timestamp >= cutoff]

        if len(recent) < 2:
            return {"trend": "insufficient_data"}

        first = recent[0]
        last = recent[-1]

        rss_change = last.rss_mb - first.rss_mb
        time_diff = (last.timestamp - first.timestamp) / 60  # minutes

        if time_diff > 0:
            rate_mb_per_min = rss_change / time_diff
        else:
            rate_mb_per_min = 0

        # Detect potential leak
        potential_leak = rate_mb_per_min > 1.0 and len(recent) >= 10

        return {
            "trend": "increasing" if rss_change > 0 else "decreasing" if rss_change < 0 else "stable",
            "rss_change_mb": round(rss_change, 2),
            "rate_mb_per_min": round(rate_mb_per_min, 4),
            "potential_leak": potential_leak,
            "data_points": len(recent),
        }

    def get_recent_alerts(self, count: int = 10) -> List[MemoryAlert]:
        """Get recent alerts."""
        return self._alerts[-count:]

    async def _check_thresholds(self, usage: MemoryUsage) -> None:
        """Check usage against thresholds."""
        process_percent = (usage.rss_mb / self.threshold.max_rss_mb) * 100
        level = None
        message = ""

        # Process memory thresholds
        if process_percent >= self.threshold.emergency_percent:
            level = MemoryAlertLevel.EMERGENCY
            message = f"EMERGENCY: Process using {usage.rss_mb:.0f}MB ({process_percent:.1f}% of limit)"
        elif process_percent >= self.threshold.critical_percent:
            level = MemoryAlertLevel.CRITICAL
            message = f"Critical: Process using {usage.rss_mb:.0f}MB ({process_percent:.1f}% of limit)"
        elif process_percent >= self.threshold.warning_percent:
            level = MemoryAlertLevel.WARNING
            message = f"Warning: Process using {usage.rss_mb:.0f}MB ({process_percent:.1f}% of limit)"

        # System memory thresholds
        if usage.system_percent >= self.threshold.system_critical_percent:
            if level is None or level.value < MemoryAlertLevel.CRITICAL.value:
                level = MemoryAlertLevel.CRITICAL
                message = f"Critical: System memory {usage.system_percent:.1f}% used"
        elif usage.system_percent >= self.threshold.system_warning_percent:
            if level is None:
                level = MemoryAlertLevel.WARNING
                message = f"Warning: System memory {usage.system_percent:.1f}% used"

        if level:
            alert = MemoryAlert(level=level, message=message, usage=usage)
            self._alerts.append(alert)

            if len(self._alerts) > 100:
                self._alerts = self._alerts[-100:]

            await self._trigger_alert(alert)

            # Auto GC on high memory
            if level in (MemoryAlertLevel.CRITICAL, MemoryAlertLevel.EMERGENCY):
                if self.enable_auto_gc:
                    await self.force_gc()

        # Check and notify pressure
        pressure = self.get_pressure()
        if pressure.level > 0:
            await self._trigger_pressure(pressure)

    async def _trigger_alert(self, alert: MemoryAlert) -> None:
        """Trigger alert callbacks."""
        logger.warning(f"[MemoryMonitor] {alert.message}")

        for callback in self._alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"[MemoryMonitor] Alert callback error: {e}")

    async def _trigger_pressure(self, pressure: MemoryPressure) -> None:
        """Trigger pressure callbacks."""
        for callback in self._pressure_callbacks:
            try:
                await callback(pressure)
            except Exception as e:
                logger.error(f"[MemoryMonitor] Pressure callback error: {e}")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self.check_memory()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[MemoryMonitor] Monitor loop error: {e}")
                await asyncio.sleep(5)

    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        usage = self.get_current_usage()
        pressure = self.get_pressure()

        return {
            "current": usage.to_dict() if usage else None,
            "pressure_level": pressure.level,
            "trend": self.get_memory_trend(),
            "thresholds": {
                "warning_percent": self.threshold.warning_percent,
                "critical_percent": self.threshold.critical_percent,
                "max_rss_mb": self.threshold.max_rss_mb,
            },
            "recent_alerts": len(self._alerts),
        }


# Global memory monitor
_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor."""
    global _monitor
    if _monitor is None:
        _monitor = MemoryMonitor()
    return _monitor
