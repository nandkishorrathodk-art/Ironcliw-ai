"""
System State Monitor - Tracks system-wide state for context-aware operations

This module provides system state monitoring capabilities for JARVIS,
enabling context-aware decision making based on current system conditions.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System state categories"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    LOCKED = "locked"
    UNKNOWN = "unknown"


@dataclass
class StateSnapshot:
    """Snapshot of system state at a point in time"""
    timestamp: datetime
    screen_locked: bool = False
    active_apps: List[str] = field(default_factory=list)
    network_connected: bool = True
    battery_level: Optional[int] = None
    audio_playing: bool = False
    active_window: Optional[str] = None
    system_load: float = 0.0
    overall_state: SystemState = SystemState.UNKNOWN


class SystemStateMonitor:
    """Monitors system state for context-aware operations.

    Provides real-time system state information to enable intelligent
    context-aware command execution.
    """

    _instance: Optional["SystemStateMonitor"] = None

    def __init__(self):
        self._state_detectors: Dict[str, Callable] = {}
        self._current_state: StateSnapshot = StateSnapshot(
            timestamp=datetime.now(),
            overall_state=SystemState.UNKNOWN
        )
        self._monitoring_active: bool = False
        self._update_interval: float = 5.0  # seconds
        self._monitor_task: Optional[asyncio.Task] = None

        # Register default detectors
        self._register_default_detectors()

    @classmethod
    def get_instance(cls) -> "SystemStateMonitor":
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_default_detectors(self):
        """Register default state detectors"""
        # Basic detectors that return safe defaults
        self._state_detectors["screen_locked"] = self._detect_screen_locked
        self._state_detectors["active_apps"] = self._detect_active_apps
        self._state_detectors["network_connected"] = self._detect_network
        self._state_detectors["battery_level"] = self._detect_battery
        self._state_detectors["audio_playing"] = self._detect_audio
        self._state_detectors["active_window"] = self._detect_active_window
        self._state_detectors["system_load"] = self._detect_system_load

        for name in self._state_detectors:
            logger.info(f"Registered state detector: {name}")

    async def _detect_screen_locked(self) -> bool:
        """Detect if screen is locked"""
        try:
            from context_intelligence.detectors.screen_lock_detector import get_screen_lock_detector
            detector = get_screen_lock_detector()
            if detector:
                return await detector.is_screen_locked()
        except Exception:
            pass
        return False

    async def _detect_active_apps(self) -> List[str]:
        """Detect active applications"""
        # Return empty list as default
        return []

    async def _detect_network(self) -> bool:
        """Detect network connectivity"""
        return True

    async def _detect_battery(self) -> Optional[int]:
        """Detect battery level"""
        return None

    async def _detect_audio(self) -> bool:
        """Detect if audio is playing"""
        return False

    async def _detect_active_window(self) -> Optional[str]:
        """Detect active window title"""
        return None

    async def _detect_system_load(self) -> float:
        """Detect system load"""
        return 0.0

    async def update_state(self) -> StateSnapshot:
        """Update and return current system state"""
        try:
            screen_locked = await self._detect_screen_locked()
            active_apps = await self._detect_active_apps()
            network_connected = await self._detect_network()
            battery_level = await self._detect_battery()
            audio_playing = await self._detect_audio()
            active_window = await self._detect_active_window()
            system_load = await self._detect_system_load()

            # Determine overall state
            if screen_locked:
                overall_state = SystemState.LOCKED
            elif system_load > 0.8:
                overall_state = SystemState.BUSY
            elif len(active_apps) > 0:
                overall_state = SystemState.ACTIVE
            else:
                overall_state = SystemState.IDLE

            self._current_state = StateSnapshot(
                timestamp=datetime.now(),
                screen_locked=screen_locked,
                active_apps=active_apps,
                network_connected=network_connected,
                battery_level=battery_level,
                audio_playing=audio_playing,
                active_window=active_window,
                system_load=system_load,
                overall_state=overall_state
            )

        except Exception as e:
            logger.warning(f"Error updating system state: {e}")

        return self._current_state

    def get_current_state(self) -> StateSnapshot:
        """Get the current state snapshot"""
        return self._current_state

    async def get_states(self, states_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get system states as a dictionary.

        Returns a dictionary of monitored states, optionally filtered
        to only include specified state names.

        Args:
            states_filter: Optional list of state names to include.
                          If None, returns all states.

        Returns:
            Dictionary with state names as keys and current values
        """
        # Update state before returning
        await self.update_state()

        state = self._current_state
        all_states = {
            "screen_locked": state.screen_locked,
            "active_apps": state.active_apps,
            "network_connected": state.network_connected,
            "battery_level": state.battery_level,
            "audio_playing": state.audio_playing,
            "active_window": state.active_window,
            "system_load": state.system_load,
            "overall_state": state.overall_state.value,
            "timestamp": state.timestamp.isoformat() if state.timestamp else None,
        }

        if states_filter:
            return {k: v for k, v in all_states.items() if k in states_filter}
        return all_states

    async def start_monitoring(self):
        """Start background state monitoring"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        logger.info("Started system state monitoring")

        async def monitor_loop():
            while self._monitoring_active:
                try:
                    await self.update_state()
                except Exception as e:
                    logger.warning(f"State monitoring error: {e}")
                await asyncio.sleep(self._update_interval)

        self._monitor_task = asyncio.create_task(monitor_loop())

    async def stop_monitoring(self):
        """Stop background state monitoring"""
        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped system state monitoring")


# Global accessor
_global_monitor: Optional[SystemStateMonitor] = None


def get_system_monitor() -> SystemStateMonitor:
    """Get the global system state monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemStateMonitor.get_instance()
    return _global_monitor
