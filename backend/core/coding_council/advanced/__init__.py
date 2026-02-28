"""
v78.0: Advanced Coding Council Module
======================================

Advanced patterns for production-grade multi-repo evolution.

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

v78.0 Gaps Addressed:
- Gap #90: Unified Process Tree Management
- Gap #91: Command Buffer for early commands
- Gap #92: Atomic Command Queue (race-condition free)
- Gap #93: Cross-Repo Transaction Coordinator (2PC)
- Gap #94: Adaptive Timeout Manager
- Gap #95: Intelligent Retry Manager

Advanced Patterns:
- Saga pattern for long-running operations
- Event sourcing for complete audit trail
- Adaptive framework selection (ML-based)
- Two-phase commit protocol
- Circuit breaker for fault tolerance
- Exponential backoff with jitter
- Bounded collections to prevent memory leaks
- Process tree tracking with cascading shutdown
- File-based atomic queues with deduplication
- Context-aware retry strategies

Author: Ironcliw v78.0
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

# v78.0 Modules - Process Tree Management
from .unified_process_tree import (
    UnifiedProcessTree,
    ProcessNode,
    ProcessRole,
    ProcessState,
    ProcessMetrics,
    ShutdownStrategy,
    TreeSnapshot,
    get_process_tree,
    get_process_tree_sync,
)

# v78.0 Modules - Command Buffer
from .command_buffer import (
    CommandBuffer,
    BufferedCommand,
    CommandPriority,
    CommandState,
    CommandType,
    BufferStats,
    get_command_buffer,
    get_command_buffer_sync,
)

# v78.0 Modules - Atomic Command Queue
from .atomic_command_queue import (
    AtomicCommandQueue,
    AtomicQueueEntry,
    AtomicFileLock as AtomicQueueLock,
    LockType as AtomicLockType,
    EntryState,
    QueueStats,
    AtomicQueueError,
    LockAcquisitionError as AtomicLockError,
    get_atomic_queue,
    get_atomic_queue_sync,
    Ironcliw_TO_JPRIME,
    JPRIME_TO_Ironcliw,
    Ironcliw_TO_REACTOR,
    REACTOR_TO_Ironcliw,
)

# v78.0 Modules - Cross-Repo Transaction Coordinator
from .cross_repo_coordinator import (
    CrossRepoTransactionCoordinator,
    Transaction,
    RepoScope,
    TransactionState as CrossRepoTxState,
    VoteResult,
    ParticipantVote,
    CoordinatorStats,
    get_transaction_coordinator,
    get_transaction_coordinator_sync,
)

# v78.0 Modules - Adaptive Timeout Manager
from .adaptive_timeout_manager import (
    AdaptiveTimeoutManager,
    OperationType,
    TimeoutStrategy,
    TimeoutConfig,
    TimeoutBudget,
    OperationStats as TimeoutOperationStats,
    LoadLevel,
    ComplexityEstimator,
    get_timeout_manager,
    get_timeout_manager_sync,
)

# v78.0 Modules - Intelligent Retry Manager
from .intelligent_retry_manager import (
    IntelligentRetryManager,
    RetryStrategy,
    RetryConfig as IntelligentRetryConfig,
    RetryResult,
    RetryAttempt,
    CircuitBreaker as RetryCircuitBreaker,
    CircuitBreakerConfig as RetryCircuitConfig,
    CircuitState as RetryCircuitState,
    ErrorCategory as RetryErrorCategory,
    ErrorClassifier as RetryErrorClassifier,
    DelayCalculator,
    RetryStats,
    with_retry as with_intelligent_retry,
    get_retry_manager,
    get_retry_manager_sync,
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
    # Process Tree Management (v78.0)
    "UnifiedProcessTree",
    "ProcessNode",
    "ProcessRole",
    "ProcessState",
    "ProcessMetrics",
    "ShutdownStrategy",
    "TreeSnapshot",
    "get_process_tree",
    "get_process_tree_sync",
    # Command Buffer (v78.0)
    "CommandBuffer",
    "BufferedCommand",
    "CommandPriority",
    "CommandState",
    "CommandType",
    "BufferStats",
    "get_command_buffer",
    "get_command_buffer_sync",
    # Atomic Command Queue (v78.0)
    "AtomicCommandQueue",
    "AtomicQueueEntry",
    "AtomicQueueLock",
    "AtomicLockType",
    "EntryState",
    "QueueStats",
    "AtomicQueueError",
    "AtomicLockError",
    "get_atomic_queue",
    "get_atomic_queue_sync",
    "Ironcliw_TO_JPRIME",
    "JPRIME_TO_Ironcliw",
    "Ironcliw_TO_REACTOR",
    "REACTOR_TO_Ironcliw",
    # Cross-Repo Transaction Coordinator (v78.0)
    "CrossRepoTransactionCoordinator",
    "Transaction",
    "RepoScope",
    "CrossRepoTxState",
    "VoteResult",
    "ParticipantVote",
    "CoordinatorStats",
    "get_transaction_coordinator",
    "get_transaction_coordinator_sync",
    # Adaptive Timeout Manager (v78.0)
    "AdaptiveTimeoutManager",
    "OperationType",
    "TimeoutStrategy",
    "TimeoutConfig",
    "TimeoutBudget",
    "TimeoutOperationStats",
    "LoadLevel",
    "ComplexityEstimator",
    "get_timeout_manager",
    "get_timeout_manager_sync",
    # Intelligent Retry Manager (v78.0)
    "IntelligentRetryManager",
    "RetryStrategy",
    "IntelligentRetryConfig",
    "RetryResult",
    "RetryAttempt",
    "RetryCircuitBreaker",
    "RetryCircuitConfig",
    "RetryCircuitState",
    "RetryErrorCategory",
    "RetryErrorClassifier",
    "DelayCalculator",
    "RetryStats",
    "with_intelligent_retry",
    "get_retry_manager",
    "get_retry_manager_sync",
]
