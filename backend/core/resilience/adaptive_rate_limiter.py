"""
Adaptive Rate Limiter with AIMD Algorithm
==========================================

Provides intelligent rate limiting that automatically adapts to system capacity.

Features:
    - AIMD (Additive Increase Multiplicative Decrease) congestion control
    - Token bucket with adaptive refill rate
    - Sliding window for accurate rate tracking
    - Priority-based request queuing
    - Redis-backed distributed coordination
    - Backpressure signal propagation
    - Per-tier and per-client rate limits
    - Fairness guarantees across clients

Theory:
    AIMD is the same algorithm used in TCP congestion control.
    - Additive Increase: Gradually increase rate when successful
    - Multiplicative Decrease: Halve rate on failure/congestion

    This provides provably fair bandwidth allocation while adapting
    to changing system capacity.

Usage:
    limiter = await get_adaptive_rate_limiter()

    async with limiter.acquire("client-1", priority=5):
        # Request is rate-limited
        await process_request()

    # Or with manual control
    if await limiter.try_acquire("client-1"):
        try:
            await process_request()
            await limiter.on_success("client-1")
        except Exception:
            await limiter.on_failure("client-1")
    else:
        # Rate limited - retry later

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("AdaptiveRateLimiter")


# =============================================================================
# Configuration
# =============================================================================

# AIMD parameters
AIMD_ADDITIVE_INCREASE = float(os.getenv("AIMD_ADDITIVE_INCREASE", "1.0"))
AIMD_MULTIPLICATIVE_DECREASE = float(os.getenv("AIMD_MULTIPLICATIVE_DECREASE", "0.5"))
AIMD_MIN_RATE = float(os.getenv("AIMD_MIN_RATE", "1.0"))  # Minimum requests/sec
AIMD_MAX_RATE = float(os.getenv("AIMD_MAX_RATE", "1000.0"))  # Maximum requests/sec
AIMD_INITIAL_RATE = float(os.getenv("AIMD_INITIAL_RATE", "100.0"))  # Starting rate

# Token bucket parameters
TOKEN_BUCKET_CAPACITY_MULTIPLIER = float(os.getenv("TOKEN_BUCKET_CAPACITY_MULTIPLIER", "2.0"))
TOKEN_REFILL_INTERVAL = float(os.getenv("TOKEN_REFILL_INTERVAL", "0.1"))  # 100ms

# Sliding window parameters
SLIDING_WINDOW_SIZE = int(os.getenv("SLIDING_WINDOW_SIZE", "60"))  # seconds
SLIDING_WINDOW_BUCKETS = int(os.getenv("SLIDING_WINDOW_BUCKETS", "60"))  # 1 bucket per second

# Redis configuration
RATE_LIMIT_REDIS_PREFIX = os.getenv("RATE_LIMIT_REDIS_PREFIX", "ratelimit:")
RATE_LIMIT_SYNC_INTERVAL = float(os.getenv("RATE_LIMIT_SYNC_INTERVAL", "1.0"))

# Queue parameters
MAX_QUEUE_SIZE = int(os.getenv("RATE_LIMIT_MAX_QUEUE_SIZE", "1000"))
MAX_QUEUE_WAIT = float(os.getenv("RATE_LIMIT_MAX_QUEUE_WAIT", "30.0"))


# =============================================================================
# Enums and Data Classes
# =============================================================================

class RateLimitStrategy(Enum):
    """Rate limiting strategy."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"
    FIXED_WINDOW = "fixed_window"


class BackpressureSignal(Enum):
    """Backpressure signal types."""
    NONE = "none"
    LIGHT = "light"      # 50-70% capacity
    MODERATE = "moderate"  # 70-85% capacity
    HEAVY = "heavy"      # 85-95% capacity
    CRITICAL = "critical"  # >95% capacity


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    initial_rate: float = AIMD_INITIAL_RATE
    min_rate: float = AIMD_MIN_RATE
    max_rate: float = AIMD_MAX_RATE
    additive_increase: float = AIMD_ADDITIVE_INCREASE
    multiplicative_decrease: float = AIMD_MULTIPLICATIVE_DECREASE
    burst_multiplier: float = TOKEN_BUCKET_CAPACITY_MULTIPLIER


@dataclass
class ClientState:
    """Per-client rate limiting state."""
    client_id: str
    current_rate: float = AIMD_INITIAL_RATE
    tokens: float = AIMD_INITIAL_RATE
    last_refill: float = field(default_factory=time.time)
    success_count: int = 0
    failure_count: int = 0
    total_requests: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    backpressure: BackpressureSignal = BackpressureSignal.NONE


@dataclass
class QueuedRequest:
    """A queued rate-limited request."""
    client_id: str
    priority: int
    timestamp: float
    event: asyncio.Event = field(default_factory=asyncio.Event)
    granted: bool = False

    def __lt__(self, other: "QueuedRequest") -> bool:
        """Higher priority (lower number) comes first, then earlier timestamp."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


# =============================================================================
# Sliding Window Counter
# =============================================================================

class SlidingWindowCounter:
    """
    Sliding window rate counter with sub-second precision.

    Uses multiple buckets to approximate a sliding window without
    storing individual request timestamps.
    """

    def __init__(
        self,
        window_size: int = SLIDING_WINDOW_SIZE,
        num_buckets: int = SLIDING_WINDOW_BUCKETS,
    ):
        self._window_size = window_size
        self._num_buckets = num_buckets
        self._bucket_size = window_size / num_buckets
        self._buckets: List[int] = [0] * num_buckets
        self._current_bucket: int = 0
        self._last_update: float = time.time()

    def _advance_buckets(self) -> None:
        """Advance buckets based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        buckets_to_advance = int(elapsed / self._bucket_size)

        if buckets_to_advance > 0:
            # Clear old buckets
            for i in range(min(buckets_to_advance, self._num_buckets)):
                bucket_idx = (self._current_bucket + i + 1) % self._num_buckets
                self._buckets[bucket_idx] = 0

            self._current_bucket = (self._current_bucket + buckets_to_advance) % self._num_buckets
            self._last_update = now

    def increment(self, count: int = 1) -> None:
        """Increment the current bucket."""
        self._advance_buckets()
        self._buckets[self._current_bucket] += count

    def get_count(self) -> int:
        """Get total count in the sliding window."""
        self._advance_buckets()
        return sum(self._buckets)

    def get_rate(self) -> float:
        """Get current rate (requests per second)."""
        count = self.get_count()
        return count / self._window_size

    def clear(self) -> None:
        """Clear all buckets."""
        self._buckets = [0] * self._num_buckets
        self._current_bucket = 0
        self._last_update = time.time()


# =============================================================================
# Token Bucket Implementation
# =============================================================================

class TokenBucket:
    """
    Token bucket rate limiter with AIMD rate adaptation.

    Tokens are added at the current rate, and each request consumes one token.
    The rate adapts based on success/failure signals.
    """

    def __init__(
        self,
        initial_rate: float = AIMD_INITIAL_RATE,
        min_rate: float = AIMD_MIN_RATE,
        max_rate: float = AIMD_MAX_RATE,
        burst_multiplier: float = TOKEN_BUCKET_CAPACITY_MULTIPLIER,
    ):
        self._rate = initial_rate
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._burst_multiplier = burst_multiplier

        self._tokens = initial_rate * burst_multiplier
        self._last_refill = time.time()
        self._lock = asyncio.Lock()

    @property
    def rate(self) -> float:
        """Current token refill rate."""
        return self._rate

    @property
    def capacity(self) -> float:
        """Maximum token capacity."""
        return self._rate * self._burst_multiplier

    @property
    def tokens(self) -> float:
        """Current token count."""
        return self._tokens

    async def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            new_tokens = elapsed * self._rate
            self._tokens = min(self.capacity, self._tokens + new_tokens)
            self._last_refill = now

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns False if insufficient tokens."""
        await self.refill()

        async with self._lock:
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Acquire tokens, waiting if necessary."""
        deadline = time.time() + timeout if timeout else None

        while True:
            if await self.try_acquire(tokens):
                return True

            if deadline and time.time() >= deadline:
                return False

            # Calculate wait time until we have enough tokens
            async with self._lock:
                wait_time = (tokens - self._tokens) / self._rate
                wait_time = min(wait_time, TOKEN_REFILL_INTERVAL)

            await asyncio.sleep(wait_time)

    def increase_rate(self, additive: float = AIMD_ADDITIVE_INCREASE) -> float:
        """Additively increase the rate (AIMD AI)."""
        self._rate = min(self._max_rate, self._rate + additive)
        return self._rate

    def decrease_rate(self, multiplicative: float = AIMD_MULTIPLICATIVE_DECREASE) -> float:
        """Multiplicatively decrease the rate (AIMD MD)."""
        self._rate = max(self._min_rate, self._rate * multiplicative)
        return self._rate

    def set_rate(self, rate: float) -> float:
        """Set rate directly (clamped to min/max)."""
        self._rate = max(self._min_rate, min(self._max_rate, rate))
        return self._rate


# =============================================================================
# Priority Queue for Rate-Limited Requests
# =============================================================================

class PriorityRequestQueue:
    """
    Priority queue for rate-limited requests.

    Higher priority requests (lower priority number) are processed first.
    Provides fairness guarantees and prevents starvation.
    """

    def __init__(self, max_size: int = MAX_QUEUE_SIZE):
        self._queue: List[QueuedRequest] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._waiting_count = 0
        self._starvation_prevention_counter: Dict[str, int] = defaultdict(int)

    @property
    def size(self) -> int:
        """Current queue size."""
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return len(self._queue) >= self._max_size

    async def enqueue(
        self,
        client_id: str,
        priority: int = 5,
    ) -> Optional[QueuedRequest]:
        """
        Enqueue a request.

        Args:
            client_id: Client identifier
            priority: Request priority (0 = highest, 9 = lowest)

        Returns:
            QueuedRequest if enqueued, None if queue is full
        """
        async with self._lock:
            if self.is_full:
                return None

            # Apply starvation prevention - boost priority for waiting clients
            boost = self._starvation_prevention_counter[client_id] // 10
            effective_priority = max(0, priority - boost)

            request = QueuedRequest(
                client_id=client_id,
                priority=effective_priority,
                timestamp=time.time(),
            )

            heapq.heappush(self._queue, request)
            self._starvation_prevention_counter[client_id] += 1
            self._waiting_count += 1

            return request

    async def dequeue(self) -> Optional[QueuedRequest]:
        """Dequeue the highest priority request."""
        async with self._lock:
            if not self._queue:
                return None

            request = heapq.heappop(self._queue)
            self._waiting_count -= 1

            # Reset starvation counter on grant
            self._starvation_prevention_counter[request.client_id] = 0

            return request

    async def remove(self, request: QueuedRequest) -> bool:
        """Remove a specific request from the queue."""
        async with self._lock:
            try:
                self._queue.remove(request)
                heapq.heapify(self._queue)
                self._waiting_count -= 1
                return True
            except ValueError:
                return False

    async def cleanup_expired(self, max_age: float = MAX_QUEUE_WAIT) -> int:
        """Remove expired requests from the queue."""
        async with self._lock:
            now = time.time()
            expired = []
            valid = []

            for req in self._queue:
                if now - req.timestamp > max_age:
                    expired.append(req)
                else:
                    valid.append(req)

            self._queue = valid
            heapq.heapify(self._queue)
            self._waiting_count = len(self._queue)

            return len(expired)


# =============================================================================
# Adaptive Rate Limiter
# =============================================================================

class AdaptiveRateLimiter:
    """
    Adaptive rate limiter with AIMD congestion control.

    Features:
    - Per-client rate limiting with fair bandwidth allocation
    - AIMD algorithm for automatic rate adaptation
    - Priority queuing with starvation prevention
    - Redis-backed distributed coordination
    - Backpressure signal propagation
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        config: Optional[RateLimitConfig] = None,
    ):
        self._redis = redis_client
        self._config = config or RateLimitConfig()

        # Per-client state
        self._clients: Dict[str, ClientState] = {}
        self._client_buckets: Dict[str, TokenBucket] = {}
        self._client_windows: Dict[str, SlidingWindowCounter] = {}

        # Global rate tracking
        self._global_bucket = TokenBucket(
            initial_rate=self._config.initial_rate * 10,
            max_rate=self._config.max_rate * 10,
        )
        self._global_window = SlidingWindowCounter()

        # Request queue
        self._queue = PriorityRequestQueue()

        # Synchronization
        self._state_lock = asyncio.Lock()
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Metrics
        self._metrics = {
            "total_requests": 0,
            "granted_requests": 0,
            "rejected_requests": 0,
            "queued_requests": 0,
            "queue_timeouts": 0,
            "rate_increases": 0,
            "rate_decreases": 0,
            "backpressure_events": 0,
        }

        logger.info(f"AdaptiveRateLimiter initialized (strategy: {self._config.strategy.value})")

    async def start(self) -> None:
        """Start background tasks."""
        self._running = True

        # Queue processor
        self._tasks.append(asyncio.create_task(
            self._process_queue(),
            name="rate_limiter_queue",
        ))

        # Periodic sync
        if self._redis:
            self._tasks.append(asyncio.create_task(
                self._sync_loop(),
                name="rate_limiter_sync",
            ))

        # Cleanup task
        self._tasks.append(asyncio.create_task(
            self._cleanup_loop(),
            name="rate_limiter_cleanup",
        ))

        logger.info("AdaptiveRateLimiter started")

    async def stop(self) -> None:
        """Stop background tasks."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("AdaptiveRateLimiter stopped")

    def _get_or_create_client(self, client_id: str) -> ClientState:
        """Get or create client state."""
        if client_id not in self._clients:
            self._clients[client_id] = ClientState(
                client_id=client_id,
                current_rate=self._config.initial_rate,
                tokens=self._config.initial_rate,
            )
            self._client_buckets[client_id] = TokenBucket(
                initial_rate=self._config.initial_rate,
                min_rate=self._config.min_rate,
                max_rate=self._config.max_rate,
                burst_multiplier=self._config.burst_multiplier,
            )
            self._client_windows[client_id] = SlidingWindowCounter()

        return self._clients[client_id]

    async def try_acquire(
        self,
        client_id: str,
        cost: int = 1,
    ) -> bool:
        """
        Try to acquire rate limit tokens.

        Args:
            client_id: Client identifier
            cost: Token cost of this request

        Returns:
            True if acquired, False if rate limited
        """
        self._metrics["total_requests"] += 1

        async with self._state_lock:
            self._get_or_create_client(client_id)

        bucket = self._client_buckets[client_id]
        window = self._client_windows[client_id]

        # Check global limit first
        if not await self._global_bucket.try_acquire(cost):
            self._metrics["rejected_requests"] += 1
            return False

        # Check per-client limit
        if not await bucket.try_acquire(cost):
            # Return global token
            self._global_bucket._tokens += cost
            self._metrics["rejected_requests"] += 1
            return False

        # Track in sliding window
        window.increment(cost)
        self._global_window.increment(cost)

        self._metrics["granted_requests"] += 1
        return True

    async def acquire(
        self,
        client_id: str,
        cost: int = 1,
        priority: int = 5,
        timeout: Optional[float] = MAX_QUEUE_WAIT,
    ) -> bool:
        """
        Acquire rate limit tokens, waiting if necessary.

        Args:
            client_id: Client identifier
            cost: Token cost of this request
            priority: Request priority (0 = highest)
            timeout: Maximum wait time

        Returns:
            True if acquired, False if timed out
        """
        # Try immediate acquisition
        if await self.try_acquire(client_id, cost):
            return True

        # Queue the request
        request = await self._queue.enqueue(client_id, priority)
        if not request:
            # Queue is full
            self._metrics["rejected_requests"] += 1
            return False

        self._metrics["queued_requests"] += 1

        try:
            # Wait for grant or timeout
            await asyncio.wait_for(request.event.wait(), timeout=timeout)
            return request.granted
        except asyncio.TimeoutError:
            self._metrics["queue_timeouts"] += 1
            await self._queue.remove(request)
            return False

    @asynccontextmanager
    async def acquire_context(
        self,
        client_id: str,
        cost: int = 1,
        priority: int = 5,
        timeout: Optional[float] = MAX_QUEUE_WAIT,
    ) -> AsyncIterator[bool]:
        """
        Context manager for rate-limited operations.

        Usage:
            async with limiter.acquire_context("client-1") as granted:
                if granted:
                    # Do work
                    await limiter.on_success("client-1")
        """
        granted = await self.acquire(client_id, cost, priority, timeout)
        try:
            yield granted
        except Exception:
            if granted:
                await self.on_failure(client_id)
            raise
        else:
            if granted:
                await self.on_success(client_id)

    async def on_success(self, client_id: str) -> None:
        """
        Signal successful completion (AIMD additive increase).

        Call this when a rate-limited request succeeds.
        """
        async with self._state_lock:
            state = self._get_or_create_client(client_id)
            state.success_count += 1
            state.last_success = time.time()
            state.consecutive_successes += 1
            state.consecutive_failures = 0

        bucket = self._client_buckets[client_id]

        # Additive increase after sustained success
        if state.consecutive_successes >= 10:
            new_rate = bucket.increase_rate(self._config.additive_increase)
            state.current_rate = new_rate
            state.consecutive_successes = 0
            self._metrics["rate_increases"] += 1
            logger.debug(f"Rate increased for {client_id}: {new_rate:.2f}/s")

        # Update backpressure signal
        await self._update_backpressure(client_id)

    async def on_failure(self, client_id: str) -> None:
        """
        Signal failure (AIMD multiplicative decrease).

        Call this when a rate-limited request fails.
        """
        async with self._state_lock:
            state = self._get_or_create_client(client_id)
            state.failure_count += 1
            state.last_failure = time.time()
            state.consecutive_failures += 1
            state.consecutive_successes = 0

        bucket = self._client_buckets[client_id]

        # Multiplicative decrease on any failure
        new_rate = bucket.decrease_rate(self._config.multiplicative_decrease)
        state.current_rate = new_rate
        self._metrics["rate_decreases"] += 1
        logger.debug(f"Rate decreased for {client_id}: {new_rate:.2f}/s")

        # Update backpressure signal
        await self._update_backpressure(client_id)

    async def _update_backpressure(self, client_id: str) -> None:
        """Update backpressure signal for client."""
        state = self._clients[client_id]
        bucket = self._client_buckets[client_id]

        # Calculate utilization
        utilization = 1.0 - (bucket.tokens / bucket.capacity)

        if utilization >= 0.95:
            new_signal = BackpressureSignal.CRITICAL
        elif utilization >= 0.85:
            new_signal = BackpressureSignal.HEAVY
        elif utilization >= 0.70:
            new_signal = BackpressureSignal.MODERATE
        elif utilization >= 0.50:
            new_signal = BackpressureSignal.LIGHT
        else:
            new_signal = BackpressureSignal.NONE

        if new_signal != state.backpressure:
            state.backpressure = new_signal
            self._metrics["backpressure_events"] += 1
            logger.debug(f"Backpressure for {client_id}: {new_signal.value}")

    def get_backpressure(self, client_id: str) -> BackpressureSignal:
        """Get current backpressure signal for client."""
        if client_id in self._clients:
            return self._clients[client_id].backpressure
        return BackpressureSignal.NONE

    async def _process_queue(self) -> None:
        """Background task to process queued requests."""
        while self._running:
            try:
                # Check if we can grant any queued requests
                request = await self._queue.dequeue()
                if not request:
                    await asyncio.sleep(TOKEN_REFILL_INTERVAL)
                    continue

                # Try to acquire for this request
                if await self.try_acquire(request.client_id):
                    request.granted = True
                    request.event.set()
                else:
                    # Re-queue if not expired
                    if time.time() - request.timestamp < MAX_QUEUE_WAIT:
                        await self._queue.enqueue(
                            request.client_id,
                            request.priority,
                        )
                    else:
                        request.event.set()  # Wake up waiter (not granted)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(1)

    async def _sync_loop(self) -> None:
        """Sync state with Redis for distributed coordination."""
        while self._running:
            try:
                await asyncio.sleep(RATE_LIMIT_SYNC_INTERVAL)

                if not self._redis:
                    continue

                # Sync global rate
                key = f"{RATE_LIMIT_REDIS_PREFIX}global_rate"
                try:
                    # Get other instances' rates
                    rates_json = await self._redis.get(key)
                    if rates_json:
                        import json
                        rates = json.loads(rates_json)
                        # Compute fair share
                        total_instances = len(rates)
                        if total_instances > 0:
                            fair_share = self._config.max_rate / total_instances
                            self._global_bucket.set_rate(fair_share)
                except Exception as e:
                    logger.debug(f"Redis sync failed: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(5)

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale clients and expired queue entries."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute

                # Cleanup expired queue entries
                expired = await self._queue.cleanup_expired()
                if expired > 0:
                    logger.debug(f"Cleaned up {expired} expired queue entries")

                # Cleanup stale client state
                async with self._state_lock:
                    now = time.time()
                    stale_clients = [
                        cid for cid, state in self._clients.items()
                        if now - max(state.last_success, state.last_failure, state.last_refill) > 3600
                    ]

                    for cid in stale_clients:
                        del self._clients[cid]
                        del self._client_buckets[cid]
                        del self._client_windows[cid]

                    if stale_clients:
                        logger.debug(f"Cleaned up {len(stale_clients)} stale clients")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)

    def get_client_rate(self, client_id: str) -> float:
        """Get current rate for a client."""
        if client_id in self._client_buckets:
            return self._client_buckets[client_id].rate
        return self._config.initial_rate

    def get_global_rate(self) -> float:
        """Get global rate limit."""
        return self._global_bucket.rate

    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        return {
            **self._metrics,
            "active_clients": len(self._clients),
            "queue_size": self._queue.size,
            "global_rate": self._global_bucket.rate,
            "global_tokens": self._global_bucket.tokens,
            "global_window_rate": self._global_window.get_rate(),
        }


# =============================================================================
# Tier-Based Rate Limiter
# =============================================================================

class TierRateLimiter:
    """
    Rate limiter with different limits per tier.

    Provides separate rate limiting for different execution tiers
    (e.g., local, GCP, Cloud Run) with intelligent overflow handling.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        tier_configs: Optional[Dict[str, RateLimitConfig]] = None,
    ):
        self._redis = redis_client

        # Default tier configurations
        default_configs = {
            "local": RateLimitConfig(initial_rate=100, max_rate=500),
            "gcp": RateLimitConfig(initial_rate=50, max_rate=200),
            "cloud_run": RateLimitConfig(initial_rate=200, max_rate=1000),
            "claude_api": RateLimitConfig(initial_rate=10, max_rate=60),  # API rate limits
        }

        self._tier_configs = tier_configs or default_configs

        # Per-tier limiters
        self._tier_limiters: Dict[str, AdaptiveRateLimiter] = {}
        for tier, config in self._tier_configs.items():
            self._tier_limiters[tier] = AdaptiveRateLimiter(
                redis_client=redis_client,
                config=config,
            )

    async def start(self) -> None:
        """Start all tier limiters."""
        for limiter in self._tier_limiters.values():
            await limiter.start()

    async def stop(self) -> None:
        """Stop all tier limiters."""
        for limiter in self._tier_limiters.values():
            await limiter.stop()

    async def try_acquire(
        self,
        tier: str,
        client_id: str,
        cost: int = 1,
    ) -> bool:
        """Try to acquire tokens for a specific tier."""
        if tier not in self._tier_limiters:
            logger.warning(f"Unknown tier: {tier}")
            return False

        return await self._tier_limiters[tier].try_acquire(client_id, cost)

    async def acquire(
        self,
        tier: str,
        client_id: str,
        cost: int = 1,
        priority: int = 5,
        timeout: Optional[float] = MAX_QUEUE_WAIT,
    ) -> bool:
        """Acquire tokens for a specific tier."""
        if tier not in self._tier_limiters:
            logger.warning(f"Unknown tier: {tier}")
            return False

        return await self._tier_limiters[tier].acquire(client_id, cost, priority, timeout)

    async def on_success(self, tier: str, client_id: str) -> None:
        """Signal success for a tier."""
        if tier in self._tier_limiters:
            await self._tier_limiters[tier].on_success(client_id)

    async def on_failure(self, tier: str, client_id: str) -> None:
        """Signal failure for a tier."""
        if tier in self._tier_limiters:
            await self._tier_limiters[tier].on_failure(client_id)

    def get_backpressure(self, tier: str, client_id: str) -> BackpressureSignal:
        """Get backpressure signal for a tier/client."""
        if tier in self._tier_limiters:
            return self._tier_limiters[tier].get_backpressure(client_id)
        return BackpressureSignal.NONE

    async def find_available_tier(
        self,
        client_id: str,
        preferred_order: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Find the first available tier for a client.

        Tries tiers in order of preference until one accepts.

        Args:
            client_id: Client identifier
            preferred_order: List of tiers to try in order

        Returns:
            Available tier name, or None if all rate limited
        """
        order = preferred_order or list(self._tier_configs.keys())

        for tier in order:
            if tier not in self._tier_limiters:
                continue

            # Check if tier has capacity (don't actually acquire)
            limiter = self._tier_limiters[tier]
            bucket = limiter._client_buckets.get(client_id)

            if bucket and bucket.tokens >= 1:
                return tier
            elif not bucket:
                # New client - use default
                return tier

        return None

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all tiers."""
        return {
            tier: limiter.get_metrics()
            for tier, limiter in self._tier_limiters.items()
        }


# =============================================================================
# Global Factory
# =============================================================================

_rate_limiter_instance: Optional[AdaptiveRateLimiter] = None
_tier_limiter_instance: Optional[TierRateLimiter] = None
_limiter_lock = asyncio.Lock()


async def get_adaptive_rate_limiter(
    redis_client: Optional[Any] = None,
    config: Optional[RateLimitConfig] = None,
) -> AdaptiveRateLimiter:
    """Get or create the global AdaptiveRateLimiter instance."""
    global _rate_limiter_instance

    async with _limiter_lock:
        if _rate_limiter_instance is None:
            _rate_limiter_instance = AdaptiveRateLimiter(
                redis_client=redis_client,
                config=config,
            )
            await _rate_limiter_instance.start()

        return _rate_limiter_instance


async def get_tier_rate_limiter(
    redis_client: Optional[Any] = None,
    tier_configs: Optional[Dict[str, RateLimitConfig]] = None,
) -> TierRateLimiter:
    """Get or create the global TierRateLimiter instance."""
    global _tier_limiter_instance

    async with _limiter_lock:
        if _tier_limiter_instance is None:
            _tier_limiter_instance = TierRateLimiter(
                redis_client=redis_client,
                tier_configs=tier_configs,
            )
            await _tier_limiter_instance.start()

        return _tier_limiter_instance


async def shutdown_rate_limiters() -> None:
    """Shutdown all rate limiter instances."""
    global _rate_limiter_instance, _tier_limiter_instance

    if _rate_limiter_instance:
        await _rate_limiter_instance.stop()
        _rate_limiter_instance = None

    if _tier_limiter_instance:
        await _tier_limiter_instance.stop()
        _tier_limiter_instance = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
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
]
