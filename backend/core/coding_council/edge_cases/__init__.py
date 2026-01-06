"""
v77.0: Edge Cases Module - Gaps #32-40
=======================================

Edge case handling and resilience:
- Gap #32: Disk space monitoring
- Gap #33: Network resilience and reconnection
- Gap #34: Memory pressure handling
- Gap #35: Process health monitoring
- Gap #36: Graceful shutdown handling
- Gap #37: Signal handling
- Gap #38: Crash recovery
- Gap #39: Data integrity checks
- Gap #40: Emergency fallbacks

Author: JARVIS v77.0
"""

from .disk_monitor import (
    DiskMonitor,
    DiskUsage,
    DiskAlert,
)
from .network_resilience import (
    NetworkResilience,
    ConnectionPool,
    ReconnectionStrategy,
)
from .memory_monitor import (
    MemoryMonitor,
    MemoryPressure,
    MemoryAlert,
)
from .crash_recovery import (
    CrashRecovery,
    RecoveryPoint,
    StateCheckpoint,
)
from .graceful_shutdown import (
    GracefulShutdown,
    ShutdownHandler,
    shutdown_handler,
)

__all__ = [
    # Disk
    "DiskMonitor",
    "DiskUsage",
    "DiskAlert",
    # Network
    "NetworkResilience",
    "ConnectionPool",
    "ReconnectionStrategy",
    # Memory
    "MemoryMonitor",
    "MemoryPressure",
    "MemoryAlert",
    # Recovery
    "CrashRecovery",
    "RecoveryPoint",
    "StateCheckpoint",
    # Shutdown
    "GracefulShutdown",
    "ShutdownHandler",
    "shutdown_handler",
]
