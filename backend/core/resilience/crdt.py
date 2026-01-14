"""
Conflict-Free Replicated Data Types (CRDTs) for Distributed State
================================================================

Provides eventually consistent data structures that automatically resolve
conflicts without coordination.

Features:
    - G-Counter: Grow-only counter (only increments)
    - PN-Counter: Positive-Negative counter (increments and decrements)
    - LWW-Register: Last-Writer-Wins register with timestamps
    - OR-Set: Observed-Remove set (add/remove operations)
    - LWW-Map: Last-Writer-Wins map for key-value state
    - Redis-backed synchronization
    - Automatic conflict resolution

Theory:
    CRDTs are data structures that can be replicated across multiple nodes
    and updated independently. They are designed to converge to a consistent
    state without coordination, making them ideal for distributed systems
    where network partitions may occur.

    Key properties:
    - Commutativity: order of operations doesn't matter
    - Associativity: grouping of operations doesn't matter
    - Idempotency: repeated operations have same effect

Usage:
    # Counter example
    counter = PNCounter(repo_id="jarvis")
    counter.increment()
    counter.decrement()
    print(counter.value)  # Works across repos with eventual consistency

    # Map example
    state = LWWMap(repo_id="jarvis")
    state.set("model_version", "1.2.3")
    merged = state.merge(other_state)  # Automatic conflict resolution

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar

logger = logging.getLogger("CRDT")


# =============================================================================
# Configuration
# =============================================================================

CRDT_SYNC_INTERVAL = float(os.getenv("CRDT_SYNC_INTERVAL", "5.0"))
CRDT_REDIS_PREFIX = os.getenv("CRDT_REDIS_PREFIX", "crdt:")


# =============================================================================
# Base CRDT Interface
# =============================================================================

T = TypeVar('T')


class CRDT(ABC, Generic[T]):
    """Abstract base class for CRDTs."""

    @abstractmethod
    def value(self) -> T:
        """Get current value."""
        pass

    @abstractmethod
    def merge(self, other: 'CRDT[T]') -> 'CRDT[T]':
        """Merge with another CRDT instance."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CRDT[T]':
        """Deserialize from dictionary."""
        pass


# =============================================================================
# G-Counter (Grow-only Counter)
# =============================================================================

@dataclass
class GCounter(CRDT[int]):
    """
    Grow-only counter CRDT.

    Each replica maintains a local counter. Global value is sum of all counters.
    Only supports increment operations (no decrement).
    """
    counts: Dict[str, int] = field(default_factory=dict)
    repo_id: str = ""

    def increment(self, amount: int = 1) -> 'GCounter':
        """Increment counter for this repo."""
        if amount < 0:
            raise ValueError("G-Counter only supports positive increments")

        new_counts = self.counts.copy()
        new_counts[self.repo_id] = new_counts.get(self.repo_id, 0) + amount
        return GCounter(counts=new_counts, repo_id=self.repo_id)

    def value(self) -> int:
        """Get total count across all replicas."""
        return sum(self.counts.values())

    def merge(self, other: 'GCounter') -> 'GCounter':
        """Merge by taking max count per replica."""
        merged = self.counts.copy()
        for repo, count in other.counts.items():
            merged[repo] = max(merged.get(repo, 0), count)
        return GCounter(counts=merged, repo_id=self.repo_id)

    def to_dict(self) -> Dict[str, Any]:
        return {"counts": self.counts, "repo_id": self.repo_id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GCounter':
        return cls(counts=data.get("counts", {}), repo_id=data.get("repo_id", ""))


# =============================================================================
# PN-Counter (Positive-Negative Counter)
# =============================================================================

@dataclass
class PNCounter(CRDT[int]):
    """
    Positive-Negative counter CRDT.

    Supports both increment and decrement by using two G-Counters.
    Value = P-counter - N-counter.
    """
    p_counter: GCounter = field(default_factory=GCounter)
    n_counter: GCounter = field(default_factory=GCounter)
    repo_id: str = ""

    def __post_init__(self):
        if not self.p_counter.repo_id:
            self.p_counter = GCounter(counts={}, repo_id=self.repo_id)
        if not self.n_counter.repo_id:
            self.n_counter = GCounter(counts={}, repo_id=self.repo_id)

    def increment(self, amount: int = 1) -> 'PNCounter':
        """Increment counter."""
        return PNCounter(
            p_counter=self.p_counter.increment(amount),
            n_counter=self.n_counter,
            repo_id=self.repo_id,
        )

    def decrement(self, amount: int = 1) -> 'PNCounter':
        """Decrement counter."""
        return PNCounter(
            p_counter=self.p_counter,
            n_counter=self.n_counter.increment(amount),
            repo_id=self.repo_id,
        )

    def value(self) -> int:
        """Get current value (P - N)."""
        return self.p_counter.value() - self.n_counter.value()

    def merge(self, other: 'PNCounter') -> 'PNCounter':
        """Merge both P and N counters."""
        return PNCounter(
            p_counter=self.p_counter.merge(other.p_counter),
            n_counter=self.n_counter.merge(other.n_counter),
            repo_id=self.repo_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_counter": self.p_counter.to_dict(),
            "n_counter": self.n_counter.to_dict(),
            "repo_id": self.repo_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PNCounter':
        return cls(
            p_counter=GCounter.from_dict(data.get("p_counter", {})),
            n_counter=GCounter.from_dict(data.get("n_counter", {})),
            repo_id=data.get("repo_id", ""),
        )


# =============================================================================
# LWW-Register (Last-Writer-Wins Register)
# =============================================================================

@dataclass
class LWWRegister(CRDT[Any]):
    """
    Last-Writer-Wins register CRDT.

    Stores a single value with timestamp. Merge takes value with later timestamp.
    Tie-breaking uses repo_id for deterministic ordering.
    """
    _value: Any = None
    timestamp: float = 0.0
    repo_id: str = ""

    def set(self, value: Any) -> 'LWWRegister':
        """Set value with current timestamp."""
        return LWWRegister(
            _value=value,
            timestamp=time.time(),
            repo_id=self.repo_id,
        )

    def value(self) -> Any:
        """Get current value."""
        return self._value

    def merge(self, other: 'LWWRegister') -> 'LWWRegister':
        """Take value with later timestamp (tie-break by repo_id)."""
        if other.timestamp > self.timestamp:
            return other
        elif other.timestamp < self.timestamp:
            return self
        else:
            # Tie-break by repo_id
            if other.repo_id > self.repo_id:
                return other
            return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self._value,
            "timestamp": self.timestamp,
            "repo_id": self.repo_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LWWRegister':
        return cls(
            _value=data.get("value"),
            timestamp=data.get("timestamp", 0.0),
            repo_id=data.get("repo_id", ""),
        )


# =============================================================================
# OR-Set (Observed-Remove Set)
# =============================================================================

@dataclass
class ORSet(CRDT[Set[Any]]):
    """
    Observed-Remove Set CRDT.

    Supports add and remove operations with proper semantics:
    - Add wins over concurrent remove
    - Remove only removes elements that were observed

    Uses unique tags for each add operation.
    """
    elements: Dict[Any, Set[str]] = field(default_factory=dict)  # value -> set of tags
    tombstones: Dict[Any, Set[str]] = field(default_factory=dict)  # removed tags
    repo_id: str = ""

    def add(self, element: Any) -> 'ORSet':
        """Add element with unique tag."""
        tag = f"{self.repo_id}:{uuid.uuid4().hex[:12]}"

        new_elements = {k: v.copy() for k, v in self.elements.items()}
        if element not in new_elements:
            new_elements[element] = set()
        new_elements[element].add(tag)

        return ORSet(
            elements=new_elements,
            tombstones={k: v.copy() for k, v in self.tombstones.items()},
            repo_id=self.repo_id,
        )

    def remove(self, element: Any) -> 'ORSet':
        """Remove element by tombstoning all its tags."""
        new_tombstones = {k: v.copy() for k, v in self.tombstones.items()}

        if element in self.elements:
            if element not in new_tombstones:
                new_tombstones[element] = set()
            new_tombstones[element].update(self.elements[element])

        return ORSet(
            elements={k: v.copy() for k, v in self.elements.items()},
            tombstones=new_tombstones,
            repo_id=self.repo_id,
        )

    def contains(self, element: Any) -> bool:
        """Check if element is in the set."""
        if element not in self.elements:
            return False

        # Element exists if it has tags not in tombstones
        tags = self.elements[element]
        tombstoned = self.tombstones.get(element, set())
        return bool(tags - tombstoned)

    def value(self) -> Set[Any]:
        """Get current set elements."""
        result = set()
        for element, tags in self.elements.items():
            tombstoned = self.tombstones.get(element, set())
            if tags - tombstoned:
                result.add(element)
        return result

    def merge(self, other: 'ORSet') -> 'ORSet':
        """Merge by union of elements and tombstones."""
        # Merge elements
        merged_elements: Dict[Any, Set[str]] = {}
        all_elements = set(self.elements.keys()) | set(other.elements.keys())
        for element in all_elements:
            tags = self.elements.get(element, set()) | other.elements.get(element, set())
            if tags:
                merged_elements[element] = tags

        # Merge tombstones
        merged_tombstones: Dict[Any, Set[str]] = {}
        all_tombstoned = set(self.tombstones.keys()) | set(other.tombstones.keys())
        for element in all_tombstoned:
            tags = self.tombstones.get(element, set()) | other.tombstones.get(element, set())
            if tags:
                merged_tombstones[element] = tags

        return ORSet(
            elements=merged_elements,
            tombstones=merged_tombstones,
            repo_id=self.repo_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elements": {str(k): list(v) for k, v in self.elements.items()},
            "tombstones": {str(k): list(v) for k, v in self.tombstones.items()},
            "repo_id": self.repo_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ORSet':
        return cls(
            elements={k: set(v) for k, v in data.get("elements", {}).items()},
            tombstones={k: set(v) for k, v in data.get("tombstones", {}).items()},
            repo_id=data.get("repo_id", ""),
        )


# =============================================================================
# LWW-Map (Last-Writer-Wins Map)
# =============================================================================

@dataclass
class LWWMap(CRDT[Dict[str, Any]]):
    """
    Last-Writer-Wins Map CRDT.

    Each key is an LWW-Register. Supports set, delete, and merge operations.
    Perfect for distributed configuration/state management.
    """
    registers: Dict[str, LWWRegister] = field(default_factory=dict)
    repo_id: str = ""

    def set(self, key: str, value: Any) -> 'LWWMap':
        """Set key to value."""
        new_registers = self.registers.copy()
        current = self.registers.get(key, LWWRegister(repo_id=self.repo_id))
        new_registers[key] = current.set(value)
        return LWWMap(registers=new_registers, repo_id=self.repo_id)

    def delete(self, key: str) -> 'LWWMap':
        """Delete key by setting to None with new timestamp."""
        return self.set(key, None)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key."""
        if key not in self.registers:
            return default
        value = self.registers[key].value()
        return default if value is None else value

    def value(self) -> Dict[str, Any]:
        """Get current map (excluding None values)."""
        return {
            k: r.value()
            for k, r in self.registers.items()
            if r.value() is not None
        }

    def merge(self, other: 'LWWMap') -> 'LWWMap':
        """Merge all registers."""
        merged = self.registers.copy()
        for key, register in other.registers.items():
            if key in merged:
                merged[key] = merged[key].merge(register)
            else:
                merged[key] = register
        return LWWMap(registers=merged, repo_id=self.repo_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registers": {k: r.to_dict() for k, r in self.registers.items()},
            "repo_id": self.repo_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LWWMap':
        return cls(
            registers={
                k: LWWRegister.from_dict(v)
                for k, v in data.get("registers", {}).items()
            },
            repo_id=data.get("repo_id", ""),
        )


# =============================================================================
# CRDT State Manager
# =============================================================================

class CRDTStateManager:
    """
    Manages distributed state using CRDTs with Redis synchronization.

    Features:
    - Automatic background synchronization
    - Conflict-free merges across repos
    - Local state caching
    - Event notifications on changes
    """

    def __init__(
        self,
        repo_id: str,
        redis_client: Optional[Any] = None,
        sync_interval: float = CRDT_SYNC_INTERVAL,
    ):
        self.repo_id = repo_id
        self._redis = redis_client
        self._sync_interval = sync_interval

        # State stores
        self._counters: Dict[str, PNCounter] = {}
        self._registers: Dict[str, LWWRegister] = {}
        self._sets: Dict[str, ORSet] = {}
        self._maps: Dict[str, LWWMap] = {}

        # Change callbacks
        self._on_change: Optional[Callable[[str, str, Any], Any]] = None

        # Background sync task
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._metrics = {
            "syncs": 0,
            "conflicts_resolved": 0,
            "local_updates": 0,
            "remote_updates": 0,
        }

        logger.info(f"CRDTStateManager initialized for repo: {repo_id}")

    async def start(self) -> None:
        """Start background synchronization."""
        self._running = True
        self._sync_task = asyncio.create_task(
            self._sync_loop(),
            name="crdt_sync",
        )

        # Initial sync from Redis
        if self._redis:
            await self._sync_from_redis()

    async def stop(self) -> None:
        """Stop background synchronization."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

    def on_change(self, callback: Callable[[str, str, Any], Any]) -> None:
        """Register callback for state changes. Args: (type, key, value)"""
        self._on_change = callback

    # Counter operations
    def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter and return new value."""
        if key not in self._counters:
            self._counters[key] = PNCounter(repo_id=self.repo_id)
        self._counters[key] = self._counters[key].increment(amount)
        self._metrics["local_updates"] += 1
        return self._counters[key].value()

    def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement counter and return new value."""
        if key not in self._counters:
            self._counters[key] = PNCounter(repo_id=self.repo_id)
        self._counters[key] = self._counters[key].decrement(amount)
        self._metrics["local_updates"] += 1
        return self._counters[key].value()

    def get_counter(self, key: str) -> int:
        """Get counter value."""
        if key not in self._counters:
            return 0
        return self._counters[key].value()

    # Register operations
    def set_value(self, key: str, value: Any) -> None:
        """Set register value."""
        if key not in self._registers:
            self._registers[key] = LWWRegister(repo_id=self.repo_id)
        self._registers[key] = self._registers[key].set(value)
        self._metrics["local_updates"] += 1

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get register value."""
        if key not in self._registers:
            return default
        return self._registers[key].value()

    # Set operations
    def add_to_set(self, key: str, element: Any) -> None:
        """Add element to set."""
        if key not in self._sets:
            self._sets[key] = ORSet(repo_id=self.repo_id)
        self._sets[key] = self._sets[key].add(element)
        self._metrics["local_updates"] += 1

    def remove_from_set(self, key: str, element: Any) -> None:
        """Remove element from set."""
        if key not in self._sets:
            return
        self._sets[key] = self._sets[key].remove(element)
        self._metrics["local_updates"] += 1

    def get_set(self, key: str) -> Set[Any]:
        """Get set elements."""
        if key not in self._sets:
            return set()
        return self._sets[key].value()

    # Map operations
    def set_map_value(self, map_key: str, key: str, value: Any) -> None:
        """Set value in map."""
        if map_key not in self._maps:
            self._maps[map_key] = LWWMap(repo_id=self.repo_id)
        self._maps[map_key] = self._maps[map_key].set(key, value)
        self._metrics["local_updates"] += 1

    def get_map_value(self, map_key: str, key: str, default: Any = None) -> Any:
        """Get value from map."""
        if map_key not in self._maps:
            return default
        return self._maps[map_key].get(key, default)

    def get_map(self, map_key: str) -> Dict[str, Any]:
        """Get entire map."""
        if map_key not in self._maps:
            return {}
        return self._maps[map_key].value()

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                if self._redis:
                    await self._sync_to_redis()
                    await self._sync_from_redis()
                    self._metrics["syncs"] += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"CRDT sync error: {e}")
                await asyncio.sleep(5.0)

    async def _sync_to_redis(self) -> None:
        """Sync local state to Redis."""
        if not self._redis:
            return

        try:
            pipe = self._redis.pipeline()

            # Sync counters
            for key, counter in self._counters.items():
                redis_key = f"{CRDT_REDIS_PREFIX}counter:{key}:{self.repo_id}"
                pipe.set(redis_key, json.dumps(counter.to_dict()), ex=3600)

            # Sync registers
            for key, register in self._registers.items():
                redis_key = f"{CRDT_REDIS_PREFIX}register:{key}:{self.repo_id}"
                pipe.set(redis_key, json.dumps(register.to_dict()), ex=3600)

            # Sync sets
            for key, orset in self._sets.items():
                redis_key = f"{CRDT_REDIS_PREFIX}set:{key}:{self.repo_id}"
                pipe.set(redis_key, json.dumps(orset.to_dict()), ex=3600)

            # Sync maps
            for key, lwwmap in self._maps.items():
                redis_key = f"{CRDT_REDIS_PREFIX}map:{key}:{self.repo_id}"
                pipe.set(redis_key, json.dumps(lwwmap.to_dict()), ex=3600)

            await pipe.execute()

        except Exception as e:
            logger.warning(f"Failed to sync to Redis: {e}")

    async def _sync_from_redis(self) -> None:
        """Sync state from other repos via Redis."""
        if not self._redis:
            return

        try:
            # Get all keys
            keys = await self._redis.keys(f"{CRDT_REDIS_PREFIX}*")

            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key

                # Skip our own keys
                if key_str.endswith(f":{self.repo_id}"):
                    continue

                data = await self._redis.get(key)
                if not data:
                    continue

                parsed = json.loads(data)

                # Determine type and merge
                if ":counter:" in key_str:
                    await self._merge_counter(key_str, parsed)
                elif ":register:" in key_str:
                    await self._merge_register(key_str, parsed)
                elif ":set:" in key_str:
                    await self._merge_set(key_str, parsed)
                elif ":map:" in key_str:
                    await self._merge_map(key_str, parsed)

        except Exception as e:
            logger.warning(f"Failed to sync from Redis: {e}")

    async def _merge_counter(self, key_str: str, data: Dict[str, Any]) -> None:
        """Merge counter from another repo."""
        # Extract key name
        parts = key_str.split(":")
        if len(parts) >= 3:
            key = parts[2]
            other = PNCounter.from_dict(data)
            if key in self._counters:
                old_value = self._counters[key].value()
                self._counters[key] = self._counters[key].merge(other)
                if self._counters[key].value() != old_value:
                    self._metrics["conflicts_resolved"] += 1
                    self._metrics["remote_updates"] += 1
            else:
                self._counters[key] = other
                self._metrics["remote_updates"] += 1

    async def _merge_register(self, key_str: str, data: Dict[str, Any]) -> None:
        """Merge register from another repo."""
        parts = key_str.split(":")
        if len(parts) >= 3:
            key = parts[2]
            other = LWWRegister.from_dict(data)
            if key in self._registers:
                old_value = self._registers[key].value()
                self._registers[key] = self._registers[key].merge(other)
                if self._registers[key].value() != old_value:
                    self._metrics["conflicts_resolved"] += 1
                    self._metrics["remote_updates"] += 1
            else:
                self._registers[key] = other
                self._metrics["remote_updates"] += 1

    async def _merge_set(self, key_str: str, data: Dict[str, Any]) -> None:
        """Merge set from another repo."""
        parts = key_str.split(":")
        if len(parts) >= 3:
            key = parts[2]
            other = ORSet.from_dict(data)
            if key in self._sets:
                old_value = self._sets[key].value()
                self._sets[key] = self._sets[key].merge(other)
                if self._sets[key].value() != old_value:
                    self._metrics["conflicts_resolved"] += 1
                    self._metrics["remote_updates"] += 1
            else:
                self._sets[key] = other
                self._metrics["remote_updates"] += 1

    async def _merge_map(self, key_str: str, data: Dict[str, Any]) -> None:
        """Merge map from another repo."""
        parts = key_str.split(":")
        if len(parts) >= 3:
            key = parts[2]
            other = LWWMap.from_dict(data)
            if key in self._maps:
                old_value = self._maps[key].value()
                self._maps[key] = self._maps[key].merge(other)
                if self._maps[key].value() != old_value:
                    self._metrics["conflicts_resolved"] += 1
                    self._metrics["remote_updates"] += 1
            else:
                self._maps[key] = other
                self._metrics["remote_updates"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get state manager metrics."""
        return {
            **self._metrics,
            "counters_count": len(self._counters),
            "registers_count": len(self._registers),
            "sets_count": len(self._sets),
            "maps_count": len(self._maps),
            "redis_available": self._redis is not None,
        }


# =============================================================================
# Global Factory
# =============================================================================

_state_manager: Optional[CRDTStateManager] = None
_manager_lock = asyncio.Lock()


async def get_crdt_state_manager(
    repo_id: Optional[str] = None,
    redis_client: Optional[Any] = None,
) -> CRDTStateManager:
    """Get or create the global CRDTStateManager."""
    global _state_manager

    async with _manager_lock:
        if _state_manager is None:
            if repo_id is None:
                repo_id = os.getenv("REPO_ID", "jarvis")

            _state_manager = CRDTStateManager(
                repo_id=repo_id,
                redis_client=redis_client,
            )
            await _state_manager.start()

        return _state_manager


async def shutdown_crdt_state_manager() -> None:
    """Shutdown the global CRDTStateManager."""
    global _state_manager

    if _state_manager:
        await _state_manager.stop()
        _state_manager = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "CRDT",
    # Data types
    "GCounter",
    "PNCounter",
    "LWWRegister",
    "ORSet",
    "LWWMap",
    # Manager
    "CRDTStateManager",
    "get_crdt_state_manager",
    "shutdown_crdt_state_manager",
]
