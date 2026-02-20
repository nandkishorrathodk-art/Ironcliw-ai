"""
v77.0: Trinity Module - Gaps #1-7
==================================

Cross-repo Trinity integration for Coding Council:
- Gap #1: Multi-transport fallback (Redis → WebSocket → File)
- Gap #2: Heartbeat verification with staleness detection
- Gap #3: PID validation for heartbeats
- Gap #4: Persistent message queue
- Gap #5: Cross-repo state synchronization
- Gap #6: Component discovery and health
- Gap #7: Graceful degradation on component failure

Author: JARVIS v77.0
"""

from .multi_transport import (
    MultiTransport,
    TransportType,
    TransportStatus,
    TransportMessage,
)
from .message_queue import (
    PersistentMessageQueue,
    QueueMessage,
    MessagePriority,
)
from .heartbeat_validator import (
    HeartbeatValidator,
    HeartbeatStatus,
    ComponentHealth,
    acquire_shared_heartbeat_validator,
    release_shared_heartbeat_validator,
    get_shared_heartbeat_validator_refcount,
)
from .cross_repo_sync import (
    CrossRepoSync,
    RepoState,
    SyncStatus,
)

__all__ = [
    "MultiTransport",
    "TransportType",
    "TransportStatus",
    "TransportMessage",
    "PersistentMessageQueue",
    "QueueMessage",
    "MessagePriority",
    "HeartbeatValidator",
    "HeartbeatStatus",
    "ComponentHealth",
    "acquire_shared_heartbeat_validator",
    "release_shared_heartbeat_validator",
    "get_shared_heartbeat_validator_refcount",
    "CrossRepoSync",
    "RepoState",
    "SyncStatus",
]
