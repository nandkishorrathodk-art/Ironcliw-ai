"""
Ironcliw API Clients - Trinity Cross-Repo Integration.
=====================================================

Provides async clients for external service integration:
- ReactorCoreClient: Training pipeline trigger (the "Ignition Key")
- IroncliwPrimeClient: Cognitive mind integration
- TrinityBaseClient: Base class with circuit breaker, retry, DLQ

Author: Ironcliw Trinity v81.0
"""

from backend.clients.reactor_core_client import (
    ReactorCoreClient,
    ReactorCoreConfig,
    TrainingPriority,
    TrainingJob,
    PipelineStage,
    get_reactor_client,
    initialize_reactor_client,
    shutdown_reactor_client,
    check_and_trigger_training,
)

from backend.clients.jarvis_prime_client import (
    IroncliwPrimeClient,
    IroncliwPrimeConfig,
    InferenceMode,
    CognitiveTaskType,
    ModelStatus,
    InferenceRequest,
    InferenceResponse,
    CognitiveTask,
    CognitiveResult,
    ModelInfo,
    get_jarvis_prime_client,
    close_jarvis_prime_client,
    inference,
    reason,
)

from backend.clients.trinity_base_client import (
    TrinityBaseClient,
    ClientConfig,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    RetryPolicy,
    DeadLetterQueue,
    DLQEntry,
    RequestDeduplicator,
)

__all__ = [
    # Reactor-Core Client
    "ReactorCoreClient",
    "ReactorCoreConfig",
    "TrainingPriority",
    "TrainingJob",
    "PipelineStage",
    "get_reactor_client",
    "initialize_reactor_client",
    "shutdown_reactor_client",
    "check_and_trigger_training",
    # Ironcliw Prime Client
    "IroncliwPrimeClient",
    "IroncliwPrimeConfig",
    "InferenceMode",
    "CognitiveTaskType",
    "ModelStatus",
    "InferenceRequest",
    "InferenceResponse",
    "CognitiveTask",
    "CognitiveResult",
    "ModelInfo",
    "get_jarvis_prime_client",
    "close_jarvis_prime_client",
    "inference",
    "reason",
    # Base Client Components
    "TrinityBaseClient",
    "ClientConfig",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitState",
    "RetryConfig",
    "RetryPolicy",
    "DeadLetterQueue",
    "DLQEntry",
    "RequestDeduplicator",
]
