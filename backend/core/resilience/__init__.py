"""
Cross-Repo Resilience Utilities - Production-Hardened Components
================================================================

Provides distributed coordination primitives for robust cross-repo communication.

Features:
    - Redis-backed distributed locks with fencing tokens
    - Atomic file operations with retry and temp file safety
    - Auto-reconnecting Redis client with exponential backoff
    - Correlation context for cross-repo request tracing
    - File watch guard with event deduplication and recovery
    - Enhanced circuit breaker with failure classification
"""

from backend.core.resilience.distributed_lock import (
    DistributedLock,
    DistributedLockConfig,
    LockAcquisitionError,
    LockNotHeldError,
)
from backend.core.resilience.atomic_file_ops import (
    AtomicFileOps,
    AtomicWriteError,
    AtomicReadError,
)
from backend.core.resilience.redis_reconnector import (
    ResilientRedisClient,
    RedisConnectionConfig,
    RedisNotAvailableError,
)
from backend.core.resilience.correlation_context import (
    CorrelationContext,
    get_current_correlation_id,
    with_correlation,
)
from backend.core.resilience.file_watch_guard import (
    FileWatchGuard,
    FileWatchConfig,
    FileEvent,
)
from backend.core.resilience.cross_repo_circuit_breaker import (
    CrossRepoCircuitBreaker,
    FailureType,
    TierHealth,
)

__all__ = [
    # Distributed Lock
    "DistributedLock",
    "DistributedLockConfig",
    "LockAcquisitionError",
    "LockNotHeldError",
    # Atomic File Ops
    "AtomicFileOps",
    "AtomicWriteError",
    "AtomicReadError",
    # Redis Reconnector
    "ResilientRedisClient",
    "RedisConnectionConfig",
    "RedisNotAvailableError",
    # Correlation Context
    "CorrelationContext",
    "get_current_correlation_id",
    "with_correlation",
    # File Watch Guard
    "FileWatchGuard",
    "FileWatchConfig",
    "FileEvent",
    # Circuit Breaker
    "CrossRepoCircuitBreaker",
    "FailureType",
    "TierHealth",
]
