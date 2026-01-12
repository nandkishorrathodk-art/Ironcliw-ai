"""
JARVIS Distributed State Module
================================

Provides unified distributed state management with:
- Transactional state updates with atomicity guarantees
- Redis-backed distributed state with local fallback
- Leader election for coordination tasks
- State snapshots and recovery
- Pub/sub for state change notifications

Usage:
    from backend.core.state import get_state_manager, StateNamespace

    state = await get_state_manager()

    # Basic operations
    await state.set("config", {"debug": True}, StateNamespace.CONFIG)
    value = await state.get("config", StateNamespace.CONFIG)

    # Transactions
    tx = await state.begin_transaction()
    await state.tx_set(tx, "key1", "value1")
    await state.tx_set(tx, "key2", "value2")
    await state.commit(tx)  # Atomic update

    # Leadership
    if state.is_leader():
        # Perform leader-only tasks
        pass
"""

from .distributed_state_manager import (
    DistributedStateManager,
    StateNamespace,
    StateEntry,
    StateTransaction,
    StateSnapshot,
    TransactionState,
    ConflictResolution,
    get_state_manager,
    shutdown_state_manager,
)

__all__ = [
    "DistributedStateManager",
    "StateNamespace",
    "StateEntry",
    "StateTransaction",
    "StateSnapshot",
    "TransactionState",
    "ConflictResolution",
    "get_state_manager",
    "shutdown_state_manager",
]
