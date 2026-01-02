"""
Adaptive Resource Governor - Dynamic FPS Throttling Under Load
==============================================================

Fixes the "Meltdown" Risk:
- When JARVIS watches 11 windows at 60 FPS during heavy CPU load,
  the system can become unresponsive as ScreenCaptureKit competes
  for resources.

Solution:
- Monitor system load (CPU, Memory, GPU) in real-time
- Dynamically throttle watcher FPS when load is high
- Restore FPS when load decreases
- Intelligent prioritization (high-confidence watchers get more FPS)

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  AdaptiveResourceGovernor                                           │
    │  ├── LoadMonitor (tracks CPU, Memory, GPU usage)                    │
    │  ├── ThrottleCalculator (determines target FPS based on load)       │
    │  ├── WatcherPrioritizer (assigns FPS budgets to watchers)           │
    │  └── AdaptiveEnforcer (applies throttling to active watchers)       │
    └─────────────────────────────────────────────────────────────────────┘

Throttle Levels:
    Level 0 (NORMAL):      < 50% CPU → Full FPS (60)
    Level 1 (LIGHT):       50-70% CPU → Reduced FPS (30)
    Level 2 (MODERATE):    70-85% CPU → Low FPS (15)
    Level 3 (HEAVY):       85-95% CPU → Minimal FPS (5)
    Level 4 (CRITICAL):    > 95% CPU → Emergency FPS (1)

Author: JARVIS v27.0 - Resource Governance
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ResourceGovernorConfig:
    """Configuration for adaptive resource governance."""

    # Load thresholds (CPU percentage)
    normal_threshold: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_LOAD_NORMAL", "50"))
    )
    light_threshold: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_LOAD_LIGHT", "70"))
    )
    moderate_threshold: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_LOAD_MODERATE", "85"))
    )
    heavy_threshold: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_LOAD_HEAVY", "95"))
    )

    # FPS targets for each level
    fps_full: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_FPS_FULL", "60"))
    )
    fps_reduced: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_FPS_REDUCED", "30"))
    )
    fps_low: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_FPS_LOW", "15"))
    )
    fps_minimal: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_FPS_MINIMAL", "5"))
    )
    fps_emergency: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_FPS_EMERGENCY", "1"))
    )

    # Memory thresholds
    memory_warning_threshold: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_MEM_WARN", "80"))
    )
    memory_critical_threshold: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_MEM_CRITICAL", "90"))
    )

    # Monitoring settings
    sample_interval: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_LOAD_SAMPLE_INTERVAL", "1.0"))
    )
    sample_window: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_LOAD_SAMPLE_WINDOW", "5"))
    )

    # Hysteresis (prevent rapid oscillation)
    hysteresis_up: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_HYSTERESIS_UP", "5.0"))
    )
    hysteresis_down: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_HYSTERESIS_DOWN", "10.0"))
    )
    cooldown_seconds: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_THROTTLE_COOLDOWN", "3.0"))
    )

    # Priority boost for high-confidence watchers
    priority_boost_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_PRIORITY_BOOST", "true").lower() == "true"
    )
    priority_boost_multiplier: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_PRIORITY_MULTIPLIER", "1.5"))
    )


# =============================================================================
# Enums
# =============================================================================

class ThrottleLevel(enum.Enum):
    """System load throttle levels."""
    NORMAL = 0      # Full performance
    LIGHT = 1       # Slight reduction
    MODERATE = 2    # Noticeable reduction
    HEAVY = 3       # Significant reduction
    CRITICAL = 4    # Emergency mode


class WatcherPriority(enum.Enum):
    """Watcher priority levels for FPS allocation."""
    LOW = 1         # Background, low confidence
    NORMAL = 2      # Standard monitoring
    HIGH = 3        # High confidence or active detection
    CRITICAL = 4    # Trigger detected, need full FPS


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SystemLoad:
    """Current system load metrics."""
    cpu_percent: float
    memory_percent: float
    cpu_per_core: List[float]
    swap_percent: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_overloaded(self) -> bool:
        """Check if system is overloaded."""
        return self.cpu_percent > 90 or self.memory_percent > 95


@dataclass
class ThrottleState:
    """Current throttle state."""
    level: ThrottleLevel
    target_fps: int
    reason: str
    changed_at: datetime = field(default_factory=datetime.now)
    load_at_change: float = 0.0


@dataclass
class WatcherFPSAllocation:
    """FPS allocation for a specific watcher."""
    watcher_id: str
    priority: WatcherPriority
    base_fps: int
    allocated_fps: int
    is_throttled: bool = False
    throttle_reason: str = ""


# =============================================================================
# Adaptive Resource Governor
# =============================================================================

class AdaptiveResourceGovernor:
    """
    Intelligent resource governor for dynamic FPS throttling.

    Key Features:
    1. Real-time load monitoring (CPU, Memory)
    2. Adaptive FPS throttling based on load
    3. Hysteresis to prevent rapid oscillation
    4. Priority-based FPS allocation
    5. Callback-based throttle notifications
    """

    def __init__(
        self,
        config: Optional[ResourceGovernorConfig] = None,
        throttle_callback: Optional[Callable[[ThrottleState], None]] = None,
    ):
        self.config = config or ResourceGovernorConfig()
        self._throttle_callback = throttle_callback

        # State
        self._current_level = ThrottleLevel.NORMAL
        self._current_fps = self.config.fps_full
        self._last_level_change = datetime.now()
        self._load_history: List[SystemLoad] = []

        # Watcher tracking
        self._watcher_priorities: Dict[str, WatcherPriority] = {}
        self._watcher_allocations: Dict[str, WatcherFPSAllocation] = {}

        # Monitoring
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        logger.info(
            f"[ResourceGovernor] Initialized with thresholds: "
            f"normal={self.config.normal_threshold}%, "
            f"light={self.config.light_threshold}%, "
            f"moderate={self.config.moderate_threshold}%, "
            f"heavy={self.config.heavy_threshold}%"
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[ResourceGovernor] Started monitoring")

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("[ResourceGovernor] Stopped monitoring")

    # =========================================================================
    # Load Monitoring
    # =========================================================================

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Sample system load
                load = await self._sample_load()

                # Add to history
                async with self._lock:
                    self._load_history.append(load)
                    # Keep only last N samples
                    if len(self._load_history) > self.config.sample_window:
                        self._load_history = self._load_history[-self.config.sample_window:]

                # Calculate average load
                avg_cpu = sum(l.cpu_percent for l in self._load_history) / len(self._load_history)
                avg_mem = sum(l.memory_percent for l in self._load_history) / len(self._load_history)

                # Determine throttle level
                new_level = self._calculate_throttle_level(avg_cpu, avg_mem)

                # Apply throttle if changed (with hysteresis)
                await self._maybe_apply_throttle(new_level, avg_cpu)

                # Wait for next sample
                await asyncio.sleep(self.config.sample_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ResourceGovernor] Monitor error: {e}")
                await asyncio.sleep(self.config.sample_interval)

    async def _sample_load(self) -> SystemLoad:
        """Sample current system load."""
        if not PSUTIL_AVAILABLE:
            return SystemLoad(
                cpu_percent=50.0,  # Assume moderate load
                memory_percent=50.0,
                cpu_per_core=[50.0],
                swap_percent=0.0,
            )

        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()

            def get_metrics():
                cpu = psutil.cpu_percent(interval=0.1)
                cpu_per_core = psutil.cpu_percent(percpu=True)
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                return cpu, cpu_per_core, mem.percent, swap.percent

            cpu, cpu_per_core, mem, swap = await loop.run_in_executor(
                None, get_metrics
            )

            return SystemLoad(
                cpu_percent=cpu,
                memory_percent=mem,
                cpu_per_core=cpu_per_core,
                swap_percent=swap,
            )

        except Exception as e:
            logger.warning(f"[ResourceGovernor] Failed to sample load: {e}")
            return SystemLoad(
                cpu_percent=50.0,
                memory_percent=50.0,
                cpu_per_core=[50.0],
                swap_percent=0.0,
            )

    def _calculate_throttle_level(
        self,
        cpu_percent: float,
        memory_percent: float,
    ) -> ThrottleLevel:
        """Calculate throttle level based on load."""
        # Check memory first (more critical)
        if memory_percent >= self.config.memory_critical_threshold:
            return ThrottleLevel.CRITICAL
        elif memory_percent >= self.config.memory_warning_threshold:
            # Bump up throttle by 1 level
            cpu_percent += 10

        # Determine level based on CPU
        if cpu_percent >= self.config.heavy_threshold:
            return ThrottleLevel.CRITICAL
        elif cpu_percent >= self.config.moderate_threshold:
            return ThrottleLevel.HEAVY
        elif cpu_percent >= self.config.light_threshold:
            return ThrottleLevel.MODERATE
        elif cpu_percent >= self.config.normal_threshold:
            return ThrottleLevel.LIGHT
        else:
            return ThrottleLevel.NORMAL

    async def _maybe_apply_throttle(
        self,
        new_level: ThrottleLevel,
        current_load: float,
    ) -> None:
        """Apply throttle if appropriate (with hysteresis)."""
        async with self._lock:
            if new_level == self._current_level:
                return

            # Check cooldown
            elapsed = (datetime.now() - self._last_level_change).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return

            # Apply hysteresis
            current_value = self._current_level.value
            new_value = new_level.value

            if new_value > current_value:
                # Throttling up - use smaller hysteresis
                threshold_adjustment = self.config.hysteresis_up
            else:
                # Throttling down - use larger hysteresis
                threshold_adjustment = self.config.hysteresis_down

            # Check if change is significant enough
            load_at_current = self._get_threshold_for_level(self._current_level)
            load_at_new = self._get_threshold_for_level(new_level)

            if new_value > current_value:
                # Going up - need to exceed threshold
                if current_load < load_at_new:
                    return
            else:
                # Going down - need to be well below threshold
                if current_load > load_at_current - threshold_adjustment:
                    return

            # Apply new level
            old_level = self._current_level
            self._current_level = new_level
            self._current_fps = self._get_fps_for_level(new_level)
            self._last_level_change = datetime.now()

            state = ThrottleState(
                level=new_level,
                target_fps=self._current_fps,
                reason=f"Load changed: {current_load:.1f}%",
                load_at_change=current_load,
            )

            logger.info(
                f"[ResourceGovernor] Throttle level changed: "
                f"{old_level.name} → {new_level.name} "
                f"(FPS: {self._get_fps_for_level(old_level)} → {self._current_fps}, "
                f"load: {current_load:.1f}%)"
            )

            # Update watcher allocations
            await self._update_all_allocations()

            # Notify callback
            if self._throttle_callback:
                try:
                    if asyncio.iscoroutinefunction(self._throttle_callback):
                        await self._throttle_callback(state)
                    else:
                        self._throttle_callback(state)
                except Exception as e:
                    logger.error(f"[ResourceGovernor] Callback error: {e}")

    def _get_threshold_for_level(self, level: ThrottleLevel) -> float:
        """Get CPU threshold for a throttle level."""
        thresholds = {
            ThrottleLevel.NORMAL: 0,
            ThrottleLevel.LIGHT: self.config.normal_threshold,
            ThrottleLevel.MODERATE: self.config.light_threshold,
            ThrottleLevel.HEAVY: self.config.moderate_threshold,
            ThrottleLevel.CRITICAL: self.config.heavy_threshold,
        }
        return thresholds.get(level, 0)

    def _get_fps_for_level(self, level: ThrottleLevel) -> int:
        """Get target FPS for a throttle level."""
        fps_map = {
            ThrottleLevel.NORMAL: self.config.fps_full,
            ThrottleLevel.LIGHT: self.config.fps_reduced,
            ThrottleLevel.MODERATE: self.config.fps_low,
            ThrottleLevel.HEAVY: self.config.fps_minimal,
            ThrottleLevel.CRITICAL: self.config.fps_emergency,
        }
        return fps_map.get(level, self.config.fps_full)

    # =========================================================================
    # Watcher Management
    # =========================================================================

    async def register_watcher(
        self,
        watcher_id: str,
        priority: WatcherPriority = WatcherPriority.NORMAL,
        base_fps: int = 60,
    ) -> WatcherFPSAllocation:
        """Register a watcher and get its FPS allocation."""
        async with self._lock:
            self._watcher_priorities[watcher_id] = priority

            allocation = self._calculate_allocation(watcher_id, priority, base_fps)
            self._watcher_allocations[watcher_id] = allocation

            logger.debug(
                f"[ResourceGovernor] Registered watcher {watcher_id}: "
                f"priority={priority.name}, allocated_fps={allocation.allocated_fps}"
            )

            return allocation

    async def unregister_watcher(self, watcher_id: str) -> None:
        """Unregister a watcher."""
        async with self._lock:
            self._watcher_priorities.pop(watcher_id, None)
            self._watcher_allocations.pop(watcher_id, None)
            logger.debug(f"[ResourceGovernor] Unregistered watcher {watcher_id}")

    async def update_watcher_priority(
        self,
        watcher_id: str,
        priority: WatcherPriority,
    ) -> Optional[WatcherFPSAllocation]:
        """Update watcher priority and recalculate allocation."""
        async with self._lock:
            if watcher_id not in self._watcher_priorities:
                return None

            old_priority = self._watcher_priorities[watcher_id]
            self._watcher_priorities[watcher_id] = priority

            allocation = self._watcher_allocations.get(watcher_id)
            if allocation:
                new_allocation = self._calculate_allocation(
                    watcher_id, priority, allocation.base_fps
                )
                self._watcher_allocations[watcher_id] = new_allocation

                logger.debug(
                    f"[ResourceGovernor] Updated priority {watcher_id}: "
                    f"{old_priority.name} → {priority.name}, "
                    f"fps: {allocation.allocated_fps} → {new_allocation.allocated_fps}"
                )

                return new_allocation

            return None

    def _calculate_allocation(
        self,
        watcher_id: str,
        priority: WatcherPriority,
        base_fps: int,
    ) -> WatcherFPSAllocation:
        """Calculate FPS allocation for a watcher."""
        # Start with current level's FPS
        target_fps = self._current_fps

        # Apply priority boost if enabled
        if self.config.priority_boost_enabled:
            if priority == WatcherPriority.HIGH:
                target_fps = int(target_fps * self.config.priority_boost_multiplier)
            elif priority == WatcherPriority.CRITICAL:
                target_fps = base_fps  # Full FPS for critical watchers
            elif priority == WatcherPriority.LOW:
                target_fps = max(1, target_fps // 2)

        # Cap at base FPS
        target_fps = min(target_fps, base_fps)

        is_throttled = target_fps < base_fps
        throttle_reason = f"Level: {self._current_level.name}" if is_throttled else ""

        return WatcherFPSAllocation(
            watcher_id=watcher_id,
            priority=priority,
            base_fps=base_fps,
            allocated_fps=target_fps,
            is_throttled=is_throttled,
            throttle_reason=throttle_reason,
        )

    async def _update_all_allocations(self) -> None:
        """Update FPS allocations for all registered watchers."""
        for watcher_id, priority in list(self._watcher_priorities.items()):
            allocation = self._watcher_allocations.get(watcher_id)
            if allocation:
                new_allocation = self._calculate_allocation(
                    watcher_id, priority, allocation.base_fps
                )
                self._watcher_allocations[watcher_id] = new_allocation

    # =========================================================================
    # Status & Queries
    # =========================================================================

    def get_current_fps(self) -> int:
        """Get current target FPS."""
        return self._current_fps

    def get_throttle_level(self) -> ThrottleLevel:
        """Get current throttle level."""
        return self._current_level

    def get_allocation(self, watcher_id: str) -> Optional[WatcherFPSAllocation]:
        """Get FPS allocation for a specific watcher."""
        return self._watcher_allocations.get(watcher_id)

    def get_all_allocations(self) -> Dict[str, WatcherFPSAllocation]:
        """Get all watcher allocations."""
        return dict(self._watcher_allocations)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive governor status."""
        recent_load = self._load_history[-1] if self._load_history else None

        return {
            "current_level": self._current_level.name,
            "current_fps": self._current_fps,
            "monitoring": self._monitoring,
            "watcher_count": len(self._watcher_priorities),
            "recent_load": {
                "cpu": recent_load.cpu_percent if recent_load else None,
                "memory": recent_load.memory_percent if recent_load else None,
            } if recent_load else None,
            "thresholds": {
                "normal": self.config.normal_threshold,
                "light": self.config.light_threshold,
                "moderate": self.config.moderate_threshold,
                "heavy": self.config.heavy_threshold,
            },
            "fps_targets": {
                "full": self.config.fps_full,
                "reduced": self.config.fps_reduced,
                "low": self.config.fps_low,
                "minimal": self.config.fps_minimal,
                "emergency": self.config.fps_emergency,
            },
        }


# =============================================================================
# Module-level convenience functions
# =============================================================================

_governor_instance: Optional[AdaptiveResourceGovernor] = None


def get_resource_governor(**kwargs) -> AdaptiveResourceGovernor:
    """Get or create the global resource governor instance."""
    global _governor_instance

    if _governor_instance is None:
        _governor_instance = AdaptiveResourceGovernor(**kwargs)

    return _governor_instance


async def start_resource_monitoring() -> None:
    """Convenience function to start resource monitoring."""
    governor = get_resource_governor()
    await governor.start_monitoring()


async def stop_resource_monitoring() -> None:
    """Convenience function to stop resource monitoring."""
    governor = get_resource_governor()
    await governor.stop_monitoring()


async def get_watcher_fps(watcher_id: str) -> int:
    """Get current FPS allocation for a watcher."""
    governor = get_resource_governor()
    allocation = governor.get_allocation(watcher_id)
    return allocation.allocated_fps if allocation else governor.get_current_fps()
