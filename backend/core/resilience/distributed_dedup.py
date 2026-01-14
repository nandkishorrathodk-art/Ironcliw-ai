"""
Distributed Event Deduplication with Redis TTL
==============================================

Provides exactly-once event processing across distributed repos.

Features:
    - Redis-backed deduplication with TTL
    - Bloom filter for efficient membership testing
    - Idempotency keys for request deduplication
    - Configurable deduplication windows
    - Local cache fallback when Redis unavailable
    - Metrics and monitoring

Theory:
    Events in a distributed system may be delivered multiple times due to:
    - Network retries
    - Multiple watchers on same directory
    - Restart recovery replays

    This module ensures exactly-once processing by tracking event IDs
    in Redis with TTL-based expiration.

Usage:
    dedup = await get_distributed_dedup()

    if await dedup.is_duplicate("event-123"):
        return  # Already processed

    # Process event...
    await dedup.mark_processed("event-123")

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import OrderedDict
import struct

logger = logging.getLogger("DistributedDedup")


# =============================================================================
# Configuration
# =============================================================================

DEDUP_TTL_SECONDS = int(os.getenv("DEDUP_TTL_SECONDS", "3600"))  # 1 hour
DEDUP_REDIS_PREFIX = os.getenv("DEDUP_REDIS_PREFIX", "dedup:")
LOCAL_CACHE_SIZE = int(os.getenv("DEDUP_LOCAL_CACHE_SIZE", "10000"))
BLOOM_FILTER_SIZE = int(os.getenv("BLOOM_FILTER_SIZE", "100000"))
BLOOM_HASH_COUNT = int(os.getenv("BLOOM_HASH_COUNT", "7"))


# =============================================================================
# Bloom Filter for Efficient Deduplication
# =============================================================================

class BloomFilter:
    """
    Space-efficient probabilistic data structure for membership testing.

    False positives are possible, but false negatives are not.
    Used as a fast first-pass check before hitting Redis.
    """

    def __init__(self, size: int = BLOOM_FILTER_SIZE, hash_count: int = BLOOM_HASH_COUNT):
        self._size = size
        self._hash_count = hash_count
        self._bits = bytearray((size + 7) // 8)
        self._count = 0

    def _hashes(self, item: str) -> List[int]:
        """Generate hash positions for item."""
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)

        positions = []
        for i in range(self._hash_count):
            pos = (h1 + i * h2) % self._size
            positions.append(pos)
        return positions

    def add(self, item: str) -> None:
        """Add item to bloom filter."""
        for pos in self._hashes(item):
            byte_idx = pos // 8
            bit_idx = pos % 8
            self._bits[byte_idx] |= (1 << bit_idx)
        self._count += 1

    def might_contain(self, item: str) -> bool:
        """Check if item might be in the filter (may have false positives)."""
        for pos in self._hashes(item):
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def clear(self) -> None:
        """Clear the bloom filter."""
        self._bits = bytearray(len(self._bits))
        self._count = 0

    @property
    def false_positive_rate(self) -> float:
        """Estimate current false positive rate."""
        if self._count == 0:
            return 0.0
        # FPR ≈ (1 - e^(-kn/m))^k
        import math
        k = self._hash_count
        n = self._count
        m = self._size
        return (1 - math.exp(-k * n / m)) ** k


# =============================================================================
# LRU Cache for Local Fallback
# =============================================================================

class LRUCache:
    """
    Least Recently Used cache for local deduplication fallback.

    Used when Redis is unavailable.
    """

    def __init__(self, max_size: int = LOCAL_CACHE_SIZE):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._max_size = max_size

    def add(self, key: str, ttl: float) -> None:
        """Add key with expiration time."""
        expiry = time.time() + ttl

        # Remove if exists (to update position)
        if key in self._cache:
            del self._cache[key]

        self._cache[key] = expiry

        # Evict oldest if over capacity
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def contains(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key not in self._cache:
            return False

        expiry = self._cache[key]
        if time.time() > expiry:
            del self._cache[key]
            return False

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return True

    def remove(self, key: str) -> bool:
        """Remove key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = time.time()
        expired = [k for k, v in self._cache.items() if v < now]
        for k in expired:
            del self._cache[k]
        return len(expired)


# =============================================================================
# Idempotency Key Generator
# =============================================================================

@dataclass
class IdempotencyKey:
    """
    Generates idempotency keys for request deduplication.

    Keys are deterministic based on request content, ensuring
    same request always generates same key.
    """
    namespace: str = "default"
    version: int = 1

    def generate(
        self,
        operation: str,
        params: Dict[str, Any],
        include_timestamp: bool = False,
    ) -> str:
        """
        Generate idempotency key for operation.

        Args:
            operation: Operation name (e.g., "create_vm")
            params: Operation parameters
            include_timestamp: If True, includes hour-level timestamp

        Returns:
            Deterministic key string
        """
        content = {
            "ns": self.namespace,
            "v": self.version,
            "op": operation,
            "params": self._normalize_params(params),
        }

        if include_timestamp:
            # Round to hour for time-based grouping
            content["hour"] = int(time.time() // 3600)

        key_str = json.dumps(content, sort_keys=True)
        hash_val = hashlib.sha256(key_str.encode()).hexdigest()[:24]

        return f"{self.namespace}:{operation}:{hash_val}"

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for consistent hashing."""
        normalized = {}
        for k, v in sorted(params.items()):
            if isinstance(v, dict):
                normalized[k] = self._normalize_params(v)
            elif isinstance(v, (list, tuple)):
                normalized[k] = sorted(str(x) for x in v)
            else:
                normalized[k] = str(v)
        return normalized


# =============================================================================
# Distributed Deduplication Manager
# =============================================================================

class DistributedDedup:
    """
    Distributed event/request deduplication manager.

    Features:
    - Redis-backed with TTL for automatic cleanup
    - Bloom filter for fast negative lookups
    - LRU cache fallback when Redis unavailable
    - Idempotency key generation
    - Batch operations for efficiency
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        ttl_seconds: int = DEDUP_TTL_SECONDS,
        prefix: str = DEDUP_REDIS_PREFIX,
    ):
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._prefix = prefix

        # Bloom filter for fast negative lookups
        self._bloom = BloomFilter()

        # Local cache fallback
        self._local_cache = LRUCache()

        # Idempotency key generator
        self._idem_key = IdempotencyKey()

        # Pending operations (for batch)
        self._pending_marks: Set[str] = set()
        self._batch_lock = asyncio.Lock()

        # Metrics
        self._metrics = {
            "checks": 0,
            "duplicates_detected": 0,
            "new_entries": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "bloom_filter_hits": 0,
            "local_cache_hits": 0,
            "batch_flushes": 0,
        }

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(f"DistributedDedup initialized (TTL: {ttl_seconds}s, prefix: {prefix})")

    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="dedup_cleanup",
        )

    async def stop(self) -> None:
        """Stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def is_duplicate(self, event_id: str) -> bool:
        """
        Check if event has already been processed.

        Uses bloom filter for fast negative check, then Redis/local cache.

        Args:
            event_id: Unique event identifier

        Returns:
            True if duplicate, False if new
        """
        self._metrics["checks"] += 1
        key = f"{self._prefix}{event_id}"

        # Fast path: bloom filter negative check
        if not self._bloom.might_contain(key):
            return False

        self._metrics["bloom_filter_hits"] += 1

        # Check Redis if available
        if self._redis:
            try:
                exists = await self._redis.exists(key)
                if exists:
                    self._metrics["redis_hits"] += 1
                    self._metrics["duplicates_detected"] += 1
                    return True
                self._metrics["redis_misses"] += 1
                return False
            except Exception as e:
                logger.warning(f"Redis check failed: {e}")

        # Fallback to local cache
        if self._local_cache.contains(key):
            self._metrics["local_cache_hits"] += 1
            self._metrics["duplicates_detected"] += 1
            return True

        return False

    async def mark_processed(
        self,
        event_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Mark event as processed.

        Args:
            event_id: Unique event identifier
            metadata: Optional metadata to store with the entry
            ttl: Optional custom TTL (defaults to configured TTL)

        Returns:
            True if newly marked, False if already existed
        """
        key = f"{self._prefix}{event_id}"
        ttl = ttl or self._ttl

        # Add to bloom filter
        self._bloom.add(key)

        # Add to local cache (always, for fallback)
        self._local_cache.add(key, ttl)

        # Add to Redis if available
        if self._redis:
            try:
                value = json.dumps({
                    "processed_at": time.time(),
                    "metadata": metadata or {},
                })

                # Use SET NX (only set if not exists) with TTL
                result = await self._redis.set(key, value, ex=ttl, nx=True)

                if result:
                    self._metrics["new_entries"] += 1
                    return True
                else:
                    # Already existed
                    self._metrics["duplicates_detected"] += 1
                    return False

            except Exception as e:
                logger.warning(f"Redis mark failed: {e}")

        self._metrics["new_entries"] += 1
        return True

    async def mark_processed_batch(self, event_ids: List[str]) -> int:
        """
        Mark multiple events as processed in a batch.

        More efficient than individual marks for bulk operations.

        Args:
            event_ids: List of event identifiers

        Returns:
            Count of newly marked events
        """
        if not event_ids:
            return 0

        async with self._batch_lock:
            # Add all to bloom filter and local cache
            for event_id in event_ids:
                key = f"{self._prefix}{event_id}"
                self._bloom.add(key)
                self._local_cache.add(key, self._ttl)

            # Batch add to Redis if available
            if self._redis:
                try:
                    pipe = self._redis.pipeline()
                    for event_id in event_ids:
                        key = f"{self._prefix}{event_id}"
                        value = json.dumps({"processed_at": time.time()})
                        pipe.set(key, value, ex=self._ttl, nx=True)

                    results = await pipe.execute()
                    new_count = sum(1 for r in results if r)
                    self._metrics["new_entries"] += new_count
                    self._metrics["batch_flushes"] += 1
                    return new_count

                except Exception as e:
                    logger.warning(f"Redis batch mark failed: {e}")

            self._metrics["new_entries"] += len(event_ids)
            return len(event_ids)

    async def remove(self, event_id: str) -> bool:
        """
        Remove event from deduplication tracking.

        Useful for retrying failed events.

        Args:
            event_id: Event identifier to remove

        Returns:
            True if removed, False if not found
        """
        key = f"{self._prefix}{event_id}"

        # Remove from local cache
        self._local_cache.remove(key)

        # Note: Cannot remove from bloom filter (by design)

        # Remove from Redis if available
        if self._redis:
            try:
                result = await self._redis.delete(key)
                return result > 0
            except Exception as e:
                logger.warning(f"Redis remove failed: {e}")

        return True

    def generate_idempotency_key(
        self,
        operation: str,
        params: Dict[str, Any],
        include_timestamp: bool = False,
    ) -> str:
        """Generate idempotency key for operation."""
        return self._idem_key.generate(operation, params, include_timestamp)

    async def check_and_mark(
        self,
        event_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Atomic check-and-mark operation.

        Returns True if this is the first time seeing the event.

        Args:
            event_id: Event identifier
            metadata: Optional metadata

        Returns:
            True if newly processed, False if duplicate
        """
        key = f"{self._prefix}{event_id}"

        # Fast negative check
        if not self._bloom.might_contain(key):
            # Definitely new - mark it
            await self.mark_processed(event_id, metadata)
            return True

        # Need to check Redis/cache
        if await self.is_duplicate(event_id):
            return False

        # Not a duplicate - mark it
        return await self.mark_processed(event_id, metadata)

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Cleanup local cache
                expired = self._local_cache.cleanup_expired()
                if expired > 0:
                    logger.debug(f"Cleaned up {expired} expired local cache entries")

                # Optionally reset bloom filter if too full
                if self._bloom.false_positive_rate > 0.1:
                    logger.info("Resetting bloom filter (FPR > 10%)")
                    self._bloom.clear()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)

    def get_metrics(self) -> Dict[str, Any]:
        """Get deduplication metrics."""
        return {
            **self._metrics,
            "bloom_filter_fpr": self._bloom.false_positive_rate,
            "local_cache_size": len(self._local_cache._cache),
            "redis_available": self._redis is not None,
        }


# =============================================================================
# Global Factory
# =============================================================================

_dedup_instance: Optional[DistributedDedup] = None
_dedup_lock = asyncio.Lock()


async def get_distributed_dedup(
    redis_client: Optional[Any] = None,
) -> DistributedDedup:
    """Get or create the global DistributedDedup instance."""
    global _dedup_instance

    async with _dedup_lock:
        if _dedup_instance is None:
            _dedup_instance = DistributedDedup(redis_client=redis_client)
            await _dedup_instance.start()

        return _dedup_instance


async def shutdown_distributed_dedup() -> None:
    """Shutdown the global DistributedDedup instance."""
    global _dedup_instance

    if _dedup_instance:
        await _dedup_instance.stop()
        _dedup_instance = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DistributedDedup",
    "BloomFilter",
    "LRUCache",
    "IdempotencyKey",
    "get_distributed_dedup",
    "shutdown_distributed_dedup",
]
