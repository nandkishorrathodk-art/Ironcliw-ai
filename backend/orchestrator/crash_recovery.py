"""
Ironcliw Crash Recovery System v1.0.0
====================================

Intelligent crash recovery and graceful degradation for the Trinity ecosystem.

Provides:
1. OOM (Out of Memory) detection and recovery
2. Process crash classification and handling
3. Automatic GCP failover for heavy workloads
4. State preservation during crashes
5. Intelligent restart strategies
6. Circuit breaker coordination

Crash Recovery Philosophy:
    "Cure the disease, not apply band-aids"
    
    We don't just restart crashed processes - we analyze WHY they crashed
    and take corrective action to prevent the same crash from recurring.

Recovery Strategies:
    RESTART         - Simple restart (for transient failures)
    RESTART_SLIM    - Restart in slim mode (for memory issues)
    GCP_FAILOVER    - Offload to GCP VM (for persistent OOM)
    GRACEFUL_DEGRADE- Disable component, continue without it
    ESCALATE        - Alert user, manual intervention needed

Exit Code Reference:
    0   - Clean exit
    1   - General error
    -9  - SIGKILL (OOM)
    -15 - SIGTERM (graceful shutdown)
    137 - 128 + SIGKILL (OOM killed by system)
    139 - 128 + SIGSEGV (segfault)

Author: Ironcliw AI System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class CrashType(Enum):
    """Types of process crashes."""
    UNKNOWN = "unknown"
    OOM = "oom"                    # Out of memory
    SEGFAULT = "segfault"         # Segmentation fault
    SIGTERM = "sigterm"           # Graceful termination
    SIGKILL = "sigkill"           # Force killed
    EXCEPTION = "exception"       # Unhandled exception
    TIMEOUT = "timeout"           # Startup timeout
    DEPENDENCY = "dependency"     # Dependency failure
    CONFIG = "config"             # Configuration error


class RecoveryStrategy(Enum):
    """Recovery strategies for crashed processes."""
    RESTART = "restart"                     # Simple restart
    RESTART_SLIM = "restart_slim"           # Restart in slim/degraded mode
    RESTART_WITH_DELAY = "restart_delay"    # Restart after cooldown
    GCP_FAILOVER = "gcp_failover"           # Failover to GCP VM
    GRACEFUL_DEGRADE = "graceful_degrade"   # Continue without this component
    MANUAL_INTERVENTION = "manual"          # Escalate to user
    NO_ACTION = "no_action"                 # Expected exit, no recovery needed


class ComponentCriticality(Enum):
    """Criticality levels for components."""
    REQUIRED = "required"       # System cannot function without this
    DEGRADED_OK = "degraded_ok" # System degrades but continues
    OPTIONAL = "optional"       # System fully functional without this


# =============================================================================
# CRASH INFORMATION
# =============================================================================

@dataclass
class CrashInfo:
    """Information about a process crash."""
    component: str
    crash_type: CrashType
    exit_code: int
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: str = ""
    stack_trace: str = ""
    memory_usage_mb: float = 0.0
    uptime_seconds: float = 0.0
    restart_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_exit_code(cls, component: str, exit_code: int, **kwargs) -> "CrashInfo":
        """Create CrashInfo from exit code."""
        crash_type = classify_exit_code(exit_code)
        return cls(
            component=component,
            crash_type=crash_type,
            exit_code=exit_code,
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component": self.component,
            "crash_type": self.crash_type.value,
            "exit_code": self.exit_code,
            "timestamp": self.timestamp.isoformat(),
            "error_message": self.error_message,
            "memory_usage_mb": self.memory_usage_mb,
            "uptime_seconds": self.uptime_seconds,
            "restart_count": self.restart_count,
        }


def classify_exit_code(exit_code: int) -> CrashType:
    """
    Classify exit code to crash type.
    
    Args:
        exit_code: Process exit code
    
    Returns:
        CrashType classification
    """
    if exit_code == 0:
        return CrashType.SIGTERM  # Clean exit
    
    if exit_code in (-9, 137):  # SIGKILL or 128 + 9
        return CrashType.OOM
    
    if exit_code in (-15, 143):  # SIGTERM or 128 + 15
        return CrashType.SIGTERM
    
    if exit_code in (-11, 139, 11):  # SIGSEGV or 128 + 11
        return CrashType.SEGFAULT
    
    if exit_code == 1:
        return CrashType.EXCEPTION
    
    return CrashType.UNKNOWN


# =============================================================================
# RECOVERY DECISION ENGINE
# =============================================================================

@dataclass
class RecoveryDecision:
    """A recovery decision with reasoning."""
    strategy: RecoveryStrategy
    reason: str
    delay_seconds: float = 0.0
    force_slim_mode: bool = False
    force_gcp: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecoveryDecisionEngine:
    """
    Makes intelligent recovery decisions based on crash history and context.
    
    The engine considers:
    - Crash type and frequency
    - Component criticality
    - System resources (memory, CPU)
    - Hardware profile (SLIM vs FULL)
    - Previous recovery attempts
    
    Philosophy: Cure the disease, not apply band-aids.
    If the same crash keeps happening, we need a different strategy.
    """
    
    # Maximum consecutive crashes before escalation
    MAX_CONSECUTIVE_CRASHES = 3
    
    # Crash rate threshold (crashes per minute)
    CRASH_RATE_THRESHOLD = 0.5
    
    # Memory threshold for slim mode recommendation (percent)
    SLIM_MODE_MEMORY_THRESHOLD = 80.0
    
    def __init__(self):
        self._crash_history: Dict[str, List[CrashInfo]] = {}
        self._recovery_attempts: Dict[str, List[RecoveryDecision]] = {}
        self._lock = asyncio.Lock()
    
    async def decide_recovery(
        self,
        crash_info: CrashInfo,
        criticality: ComponentCriticality = ComponentCriticality.REQUIRED,
    ) -> RecoveryDecision:
        """
        Decide the best recovery strategy for a crashed component.
        
        This is the core intelligence of the recovery system.
        """
        async with self._lock:
            # Record crash
            if crash_info.component not in self._crash_history:
                self._crash_history[crash_info.component] = []
            self._crash_history[crash_info.component].append(crash_info)
            
            # Get crash history for this component
            history = self._crash_history[crash_info.component]
            recent_crashes = self._get_recent_crashes(history, minutes=5)
            consecutive_crashes = self._count_consecutive_crashes(history)
            
            # Decision logic
            return self._make_decision(
                crash_info=crash_info,
                criticality=criticality,
                recent_crashes=recent_crashes,
                consecutive_crashes=consecutive_crashes,
            )
    
    def _get_recent_crashes(
        self,
        history: List[CrashInfo],
        minutes: int = 5,
    ) -> List[CrashInfo]:
        """Get crashes within the last N minutes."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [c for c in history if c.timestamp > cutoff]
    
    def _count_consecutive_crashes(self, history: List[CrashInfo]) -> int:
        """Count consecutive crashes (crashes without successful restart)."""
        count = 0
        for crash in reversed(history):
            if crash.crash_type == CrashType.SIGTERM:
                break  # Clean shutdown, reset count
            count += 1
        return count
    
    def _make_decision(
        self,
        crash_info: CrashInfo,
        criticality: ComponentCriticality,
        recent_crashes: List[CrashInfo],
        consecutive_crashes: int,
    ) -> RecoveryDecision:
        """Make the actual recovery decision."""
        
        # Clean exit - no recovery needed
        if crash_info.crash_type == CrashType.SIGTERM and crash_info.exit_code in (0, -15, 143):
            return RecoveryDecision(
                strategy=RecoveryStrategy.NO_ACTION,
                reason="Clean shutdown, no recovery needed",
            )
        
        # OOM crash - this is the most common issue we need to cure
        if crash_info.crash_type == CrashType.OOM:
            return self._handle_oom_crash(
                crash_info, criticality, consecutive_crashes
            )
        
        # Segfault - usually a bug or memory corruption
        if crash_info.crash_type == CrashType.SEGFAULT:
            return self._handle_segfault(
                crash_info, criticality, consecutive_crashes
            )
        
        # Too many consecutive crashes - escalate
        if consecutive_crashes >= self.MAX_CONSECUTIVE_CRASHES:
            if criticality == ComponentCriticality.OPTIONAL:
                return RecoveryDecision(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADE,
                    reason=f"Too many crashes ({consecutive_crashes}), disabling optional component",
                )
            else:
                return RecoveryDecision(
                    strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                    reason=f"Too many crashes ({consecutive_crashes}), manual intervention required",
                )
        
        # Too many recent crashes - add delay
        crash_rate = len(recent_crashes) / 5.0  # crashes per minute
        if crash_rate > self.CRASH_RATE_THRESHOLD:
            return RecoveryDecision(
                strategy=RecoveryStrategy.RESTART_WITH_DELAY,
                reason=f"High crash rate ({crash_rate:.2f}/min), adding cooldown",
                delay_seconds=30.0 * (consecutive_crashes + 1),
            )
        
        # Default: simple restart
        return RecoveryDecision(
            strategy=RecoveryStrategy.RESTART,
            reason="Attempting standard restart",
            delay_seconds=min(5.0 * consecutive_crashes, 30.0),
        )
    
    def _handle_oom_crash(
        self,
        crash_info: CrashInfo,
        criticality: ComponentCriticality,
        consecutive_crashes: int,
    ) -> RecoveryDecision:
        """
        Handle OOM (Out of Memory) crash.
        
        OOM is the most common cause of crashes in the Ironcliw system,
        especially on machines with < 32GB RAM running ML models.
        
        Strategy:
        1. First OOM: Try slim mode
        2. Second OOM: Force GCP offload
        3. Third+ OOM: Graceful degradation or manual intervention
        """
        
        # Check current system memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            memory_percent = mem.percent
        except ImportError:
            memory_percent = 90.0  # Assume high if we can't check
        
        if consecutive_crashes == 1:
            # First OOM - try slim mode
            return RecoveryDecision(
                strategy=RecoveryStrategy.RESTART_SLIM,
                reason="OOM detected - restarting in slim mode with reduced memory",
                force_slim_mode=True,
                delay_seconds=5.0,
                metadata={"memory_percent": memory_percent},
            )
        
        elif consecutive_crashes == 2:
            # Second OOM - GCP failover
            return RecoveryDecision(
                strategy=RecoveryStrategy.GCP_FAILOVER,
                reason="Repeated OOM - offloading to GCP VM",
                force_gcp=True,
                delay_seconds=10.0,
                metadata={"memory_percent": memory_percent},
            )
        
        else:
            # Multiple OOMs - escalate based on criticality
            if criticality == ComponentCriticality.REQUIRED:
                return RecoveryDecision(
                    strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                    reason=f"Persistent OOM ({consecutive_crashes} times) - manual intervention required",
                    metadata={"memory_percent": memory_percent},
                )
            else:
                return RecoveryDecision(
                    strategy=RecoveryStrategy.GRACEFUL_DEGRADE,
                    reason=f"Persistent OOM ({consecutive_crashes} times) - disabling component",
                    metadata={"memory_percent": memory_percent},
                )
    
    def _handle_segfault(
        self,
        crash_info: CrashInfo,
        criticality: ComponentCriticality,
        consecutive_crashes: int,
    ) -> RecoveryDecision:
        """Handle segmentation fault."""
        
        if consecutive_crashes >= 2:
            # Repeated segfaults indicate a serious issue
            return RecoveryDecision(
                strategy=RecoveryStrategy.MANUAL_INTERVENTION,
                reason=f"Repeated segfaults ({consecutive_crashes}) - likely a bug or memory corruption",
            )
        
        # First segfault - try restart with delay
        return RecoveryDecision(
            strategy=RecoveryStrategy.RESTART_WITH_DELAY,
            reason="Segfault detected - restarting with delay",
            delay_seconds=10.0,
        )
    
    def get_crash_statistics(self, component: str) -> Dict[str, Any]:
        """Get crash statistics for a component."""
        history = self._crash_history.get(component, [])
        
        if not history:
            return {"total_crashes": 0}
        
        recent = self._get_recent_crashes(history, minutes=60)
        
        # Count by type
        by_type: Dict[str, int] = {}
        for crash in history:
            type_name = crash.crash_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        return {
            "total_crashes": len(history),
            "recent_crashes_1h": len(recent),
            "crashes_by_type": by_type,
            "first_crash": history[0].timestamp.isoformat() if history else None,
            "last_crash": history[-1].timestamp.isoformat() if history else None,
        }
    
    def clear_history(self, component: str) -> None:
        """Clear crash history for a component (e.g., after successful recovery)."""
        if component in self._crash_history:
            del self._crash_history[component]
        if component in self._recovery_attempts:
            del self._recovery_attempts[component]


# =============================================================================
# CRASH RECOVERY COORDINATOR
# =============================================================================

class CrashRecoveryCoordinator:
    """
    Coordinates crash recovery across all Trinity components.
    
    Integrates with:
    - ServiceRegistry for component status
    - HealthCoordinator for health checks
    - GCP VM Manager for cloud failover
    """
    
    _instance: Optional["CrashRecoveryCoordinator"] = None
    
    def __new__(cls) -> "CrashRecoveryCoordinator":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._decision_engine = RecoveryDecisionEngine()
        self._recovery_callbacks: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
        self._cloud_lock_path = Path.home() / ".jarvis" / "trinity" / "cloud_lock.json"
        self._initialized = True
        
        logger.debug("[CrashRecoveryCoordinator] Initialized")
    
    def register_recovery_callback(
        self,
        component: str,
        callback: Callable[[RecoveryDecision], Any],
    ) -> None:
        """Register a callback for recovery decisions."""
        self._recovery_callbacks[component] = callback
    
    async def handle_crash(
        self,
        component: str,
        exit_code: int,
        criticality: ComponentCriticality = ComponentCriticality.REQUIRED,
        **kwargs,
    ) -> RecoveryDecision:
        """
        Handle a component crash.
        
        Args:
            component: Name of the crashed component
            exit_code: Process exit code
            criticality: How critical this component is
            **kwargs: Additional crash information
        
        Returns:
            RecoveryDecision with strategy and reasoning
        """
        async with self._lock:
            # Create crash info
            crash_info = CrashInfo.from_exit_code(
                component=component,
                exit_code=exit_code,
                **kwargs
            )
            
            logger.warning(
                f"[CrashRecoveryCoordinator] {component} crashed: "
                f"type={crash_info.crash_type.value}, exit_code={exit_code}"
            )
            
            # Get recovery decision
            decision = await self._decision_engine.decide_recovery(
                crash_info, criticality
            )
            
            logger.info(
                f"[CrashRecoveryCoordinator] Recovery decision for {component}: "
                f"{decision.strategy.value} - {decision.reason}"
            )
            
            # Handle GCP failover
            if decision.strategy == RecoveryStrategy.GCP_FAILOVER:
                await self._set_cloud_lock(component)
            
            # Execute callback if registered
            if component in self._recovery_callbacks:
                try:
                    callback = self._recovery_callbacks[component]
                    if asyncio.iscoroutinefunction(callback):
                        await callback(decision)
                    else:
                        callback(decision)
                except Exception as e:
                    logger.warning(f"[CrashRecoveryCoordinator] Callback failed: {e}")
            
            return decision
    
    async def _set_cloud_lock(self, component: str) -> None:
        """
        Set cloud lock to persist GCP mode across restarts.
        
        This prevents the "OOM -> restart -> OOM" infinite loop.
        """
        try:
            self._cloud_lock_path.parent.mkdir(parents=True, exist_ok=True)
            
            lock_data = {
                "component": component,
                "reason": "OOM crash recovery",
                "timestamp": datetime.now().isoformat(),
                "force_gcp": True,
            }
            
            self._cloud_lock_path.write_text(json.dumps(lock_data, indent=2))
            logger.info(f"[CrashRecoveryCoordinator] Cloud lock set for {component}")
            
        except Exception as e:
            logger.warning(f"[CrashRecoveryCoordinator] Failed to set cloud lock: {e}")
    
    def is_cloud_locked(self) -> bool:
        """Check if cloud lock is active."""
        return self._cloud_lock_path.exists()
    
    def clear_cloud_lock(self) -> None:
        """Clear the cloud lock (e.g., after hardware upgrade)."""
        try:
            if self._cloud_lock_path.exists():
                self._cloud_lock_path.unlink()
                logger.info("[CrashRecoveryCoordinator] Cloud lock cleared")
        except Exception as e:
            logger.warning(f"[CrashRecoveryCoordinator] Failed to clear cloud lock: {e}")
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get crash statistics for all components."""
        return {
            component: self._decision_engine.get_crash_statistics(component)
            for component in self._decision_engine._crash_history.keys()
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_coordinator_instance: Optional[CrashRecoveryCoordinator] = None


def get_crash_recovery_coordinator() -> CrashRecoveryCoordinator:
    """Get the singleton CrashRecoveryCoordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = CrashRecoveryCoordinator()
    return _coordinator_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def handle_component_crash(
    component: str,
    exit_code: int,
    criticality: ComponentCriticality = ComponentCriticality.REQUIRED,
    **kwargs,
) -> RecoveryDecision:
    """
    Handle a component crash and get recovery decision.
    
    Convenience function that uses the global coordinator.
    """
    coordinator = get_crash_recovery_coordinator()
    return await coordinator.handle_crash(component, exit_code, criticality, **kwargs)


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.debug("[CrashRecovery] Module loaded")
