"""
v77.2: Advanced Coding Council Module
======================================

Advanced patterns for production-grade multi-repo evolution:

v77.1 Gaps Addressed:
- Gap #41: Partial success handling
- Gap #43: Cross-repo dependency tracking
- Gap #44: Distributed transaction coordination
- Gap #57: Evolution state persistence

v77.2 Gaps Addressed:
- Gap #61: Atomic file locking (fcntl-based)
- Gap #62: Git conflict resolution
- Gap #64: Memory management / bounded collections
- Gap #65: Async lock with timeout
- Gap #66: Circuit breaker pattern
- Gap #67: Retention policies
- Gap #71: Graceful shutdown
- Gap #72: Error classification (transient vs permanent)
- Gap #80: Handler lifecycle management

Advanced Patterns:
- Saga pattern for long-running operations
- Event sourcing for complete audit trail
- Adaptive framework selection (ML-based)
- Two-phase commit protocol
- Circuit breaker for fault tolerance
- Exponential backoff with jitter
- Bounded collections to prevent memory leaks

Author: JARVIS v77.2
"""

# v77.1 Modules
from .distributed_transaction import (
    DistributedTransactionCoordinator,
    TransactionPhase,
    TransactionState,
    TransactionResult,
    TwoPhaseCommit,
)
from .saga_coordinator import (
    SagaCoordinator,
    SagaStep,
    SagaState,
    SagaResult,
    CompensatingAction,
)
from .event_store import (
    EvolutionEventStore,
    EvolutionEvent,
    EventType,
    EventStream,
)
from .dependency_tracker import (
    CrossRepoDependencyTracker,
    Dependency,
    DependencyGraph,
    ImportAnalyzer,
)
from .state_machine import (
    EvolutionStateMachine,
    EvolutionState,
    StateTransition,
    Checkpoint,
)
from .partial_success import (
    PartialSuccessHandler,
    FrameworkOutcome,
    RecoveryStrategy,
    MergeResult,
)
from .adaptive_selector import (
    AdaptiveFrameworkSelector,
    FrameworkScore,
    SelectionContext,
    LearningModel,
)

# v77.2 Modules - Resilience
from .resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    ErrorClassifier,
    ErrorCategory,
    ClassifiedError,
    RetryPolicy,
    RetryConfig,
    with_retry,
    AsyncLockWithTimeout,
    ShutdownHandler,
    HealthChecker,
    HealthStatus,
    get_shutdown_handler,
    get_health_checker,
    get_circuit_breaker_registry,
)

# v77.2 Modules - Atomic Locking
from .atomic_locking import (
    AtomicFileLock,
    LockManager,
    LockType,
    LockInfo,
    LockAcquisitionError,
    get_lock_manager,
)

# v77.2 Modules - Git Conflict Handler
from .git_conflict_handler import (
    GitConflictHandler,
    ConflictDetector,
    ConflictResolver,
    ConflictParser,
    FileConflict,
    ConflictHunk,
    ConflictType,
    ResolutionStrategy,
    MergeConflictResult,
)

# v77.2 Modules - Memory Management
from .memory_management import (
    BoundedDict,
    BoundedList,
    EvictionPolicy,
    WeakHandlerSet,
    RetentionPolicy,
    RetentionManager,
    MemoryMonitor,
    CleanupService,
    get_memory_monitor,
    get_cleanup_service,
)

__all__ = [
    # Distributed Transactions (v77.1)
    "DistributedTransactionCoordinator",
    "TransactionPhase",
    "TransactionState",
    "TransactionResult",
    "TwoPhaseCommit",
    # Saga Pattern (v77.1)
    "SagaCoordinator",
    "SagaStep",
    "SagaState",
    "SagaResult",
    "CompensatingAction",
    # Event Sourcing (v77.1)
    "EvolutionEventStore",
    "EvolutionEvent",
    "EventType",
    "EventStream",
    # Dependency Tracking (v77.1)
    "CrossRepoDependencyTracker",
    "Dependency",
    "DependencyGraph",
    "ImportAnalyzer",
    # State Machine (v77.1)
    "EvolutionStateMachine",
    "EvolutionState",
    "StateTransition",
    "Checkpoint",
    # Partial Success (v77.1)
    "PartialSuccessHandler",
    "FrameworkOutcome",
    "RecoveryStrategy",
    "MergeResult",
    # Adaptive Selection (v77.1)
    "AdaptiveFrameworkSelector",
    "FrameworkScore",
    "SelectionContext",
    "LearningModel",
    # Resilience (v77.2)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    "ErrorClassifier",
    "ErrorCategory",
    "ClassifiedError",
    "RetryPolicy",
    "RetryConfig",
    "with_retry",
    "AsyncLockWithTimeout",
    "ShutdownHandler",
    "HealthChecker",
    "HealthStatus",
    "get_shutdown_handler",
    "get_health_checker",
    "get_circuit_breaker_registry",
    # Atomic Locking (v77.2)
    "AtomicFileLock",
    "LockManager",
    "LockType",
    "LockInfo",
    "LockAcquisitionError",
    "get_lock_manager",
    # Git Conflict Handler (v77.2)
    "GitConflictHandler",
    "ConflictDetector",
    "ConflictResolver",
    "ConflictParser",
    "FileConflict",
    "ConflictHunk",
    "ConflictType",
    "ResolutionStrategy",
    "MergeConflictResult",
    # Memory Management (v77.2)
    "BoundedDict",
    "BoundedList",
    "EvictionPolicy",
    "WeakHandlerSet",
    "RetentionPolicy",
    "RetentionManager",
    "MemoryMonitor",
    "CleanupService",
    "get_memory_monitor",
    "get_cleanup_service",
]
