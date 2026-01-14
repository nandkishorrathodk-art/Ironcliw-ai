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
    - Vector clocks for distributed event ordering and causality
    - Distributed deduplication with Bloom filters and Redis TTL
    - CRDTs for conflict-free distributed state consistency
    - Adaptive rate limiting with AIMD congestion control
    - Distributed circuit breakers with cross-instance coordination
    - Graceful shutdown with request draining and coordination
    - Predictive VM provisioning with time-series forecasting
    - Model versioning with automatic rollback
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
from backend.core.resilience.vector_clock import (
    LamportTimestamp,
    VectorClock,
    CausalEvent,
    CausalEventManager,
    CausalBarrier,
    CausalOrdering,
)
from backend.core.resilience.distributed_dedup import (
    BloomFilter,
    LRUCache,
    IdempotencyKey,
    DistributedDedup,
    get_distributed_dedup,
    shutdown_distributed_dedup,
)
from backend.core.resilience.crdt import (
    GCounter,
    PNCounter,
    LWWRegister,
    ORSet,
    LWWMap,
    CRDTStateManager,
    get_crdt_manager,
    shutdown_crdt_manager,
)
from backend.core.resilience.adaptive_rate_limiter import (
    AdaptiveRateLimiter,
    TierRateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    BackpressureSignal,
    TokenBucket,
    SlidingWindowCounter,
    PriorityRequestQueue,
    get_adaptive_rate_limiter,
    get_tier_rate_limiter,
    shutdown_rate_limiters,
)
from backend.core.resilience.distributed_circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerGroup,
    DistributedCircuitBreakerManager,
    CircuitConfig,
    CircuitState,
    FailureCategory,
    CircuitOpenError,
    SlidingWindowTracker,
    get_distributed_circuit_breaker,
    shutdown_distributed_circuit_breaker,
)
from backend.core.resilience.graceful_shutdown import (
    ShutdownCoordinator,
    ShutdownPhase,
    ShutdownReason,
    ShutdownHandler,
    InFlightTracker,
    TrackedRequest,
    ShutdownInProgressError,
    get_shutdown_coordinator,
    shutdown_coordinator,
    track_request,
)
from backend.core.resilience.predictive_provisioning import (
    PredictiveProvisioner,
    HoltWintersForecaster,
    AnomalyDetector,
    PatternDetector,
    Forecast,
    ScalingRecommendation,
    ScalingAction,
    SeasonalPattern,
    get_predictive_provisioner,
    shutdown_predictive_provisioner,
)
from backend.core.resilience.model_versioning import (
    ModelVersionManager,
    ModelVersion,
    SemanticVersion,
    DeploymentType,
    DeploymentStatus,
    RollbackReason,
    RollbackEvent,
    VersionMetricTracker,
    get_model_version_manager,
    shutdown_model_version_manager,
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
    # Cross-Repo Circuit Breaker
    "CrossRepoCircuitBreaker",
    "FailureType",
    "TierHealth",
    # Vector Clocks
    "LamportTimestamp",
    "VectorClock",
    "CausalEvent",
    "CausalEventManager",
    "CausalBarrier",
    "CausalOrdering",
    # Distributed Deduplication
    "BloomFilter",
    "LRUCache",
    "IdempotencyKey",
    "DistributedDedup",
    "get_distributed_dedup",
    "shutdown_distributed_dedup",
    # CRDTs
    "GCounter",
    "PNCounter",
    "LWWRegister",
    "ORSet",
    "LWWMap",
    "CRDTStateManager",
    "get_crdt_manager",
    "shutdown_crdt_manager",
    # Adaptive Rate Limiter
    "AdaptiveRateLimiter",
    "TierRateLimiter",
    "RateLimitConfig",
    "RateLimitStrategy",
    "BackpressureSignal",
    "TokenBucket",
    "SlidingWindowCounter",
    "PriorityRequestQueue",
    "get_adaptive_rate_limiter",
    "get_tier_rate_limiter",
    "shutdown_rate_limiters",
    # Distributed Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerGroup",
    "DistributedCircuitBreakerManager",
    "CircuitConfig",
    "CircuitState",
    "FailureCategory",
    "CircuitOpenError",
    "SlidingWindowTracker",
    "get_distributed_circuit_breaker",
    "shutdown_distributed_circuit_breaker",
    # Graceful Shutdown
    "ShutdownCoordinator",
    "ShutdownPhase",
    "ShutdownReason",
    "ShutdownHandler",
    "InFlightTracker",
    "TrackedRequest",
    "ShutdownInProgressError",
    "get_shutdown_coordinator",
    "shutdown_coordinator",
    "track_request",
    # Predictive Provisioning
    "PredictiveProvisioner",
    "HoltWintersForecaster",
    "AnomalyDetector",
    "PatternDetector",
    "Forecast",
    "ScalingRecommendation",
    "ScalingAction",
    "SeasonalPattern",
    "get_predictive_provisioner",
    "shutdown_predictive_provisioner",
    # Model Versioning
    "ModelVersionManager",
    "ModelVersion",
    "SemanticVersion",
    "DeploymentType",
    "DeploymentStatus",
    "RollbackReason",
    "RollbackEvent",
    "VersionMetricTracker",
    "get_model_version_manager",
    "shutdown_model_version_manager",
]
