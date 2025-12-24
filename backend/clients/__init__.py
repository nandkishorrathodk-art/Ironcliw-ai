"""
JARVIS API Clients
==================

Provides async clients for external service integration:
- ReactorCoreClient: Training pipeline trigger (the "Ignition Key")
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
]
