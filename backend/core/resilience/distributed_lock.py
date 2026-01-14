"""
Redis-Backed Distributed Lock with Fencing Tokens
=================================================

Production-grade distributed locking for multi-process coordination.

Features:
    - Redis-backed with automatic expiry (prevents deadlocks)
    - Fencing tokens to prevent stale lock holders
    - Automatic renewal with background task
    - Lock inheritance for nested acquisitions
    - Graceful degradation to local lock when Redis unavailable
    - Comprehensive metrics and telemetry

Author: JARVIS Cross-Repo Resilience
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


class LockAcquisitionError(Exception):
    """Raised when lock cannot be acquired within timeout."""

    def __init__(self, lock_name: str, timeout: float, holder: Optional[str] = None):
        self.lock_name = lock_name
        self.timeout = timeout
        self.holder = holder
        super().__init__(
            f"Failed to acquire lock '{lock_name}' within {timeout}s"
            + (f" (held by {holder})" if holder else "")
        )


class LockNotHeldError(Exception):
    """Raised when trying to release a lock not held by current process."""

    def __init__(self, lock_name: str, expected_token: str, actual_token: Optional[str]):
        self.lock_name = lock_name
        self.expected_token = expected_token
        self.actual_token = actual_token
        super().__init__(
            f"Cannot release lock '{lock_name}': token mismatch "
            f"(expected={expected_token}, actual={actual_token})"
        )


@dataclass
class DistributedLockConfig:
    """Configuration for distributed lock."""

    # Timing
    default_timeout: float = 30.0  # Max wait time to acquire
    lock_ttl: float = 60.0  # Lock expiry time in Redis
    renewal_interval: float = 20.0  # Renew lock every N seconds (< TTL)
    retry_interval: float = 0.1  # Time between acquisition attempts

    # Behavior
    auto_renew: bool = True  # Automatically extend lock while held
    allow_reentry: bool = True  # Same holder can acquire again
    fallback_to_local: bool = True  # Use local lock if Redis fails

    # Identification
    holder_id: str = field(default_factory=lambda: f"{os.getpid()}-{uuid.uuid4().hex[:8]}")


@dataclass
class LockMetrics:
    """Metrics for lock operations."""

    acquisitions: int = 0
    acquisition_failures: int = 0
    releases: int = 0
    renewals: int = 0
    renewal_failures: int = 0
    fallback_to_local: int = 0
    contention_events: int = 0  # Times we had to wait for lock
    avg_wait_time_ms: float = 0.0
    max_hold_time_ms: float = 0.0

    def record_acquisition(self, wait_time_ms: float) -> None:
        self.acquisitions += 1
        total_wait = self.avg_wait_time_ms * (self.acquisitions - 1) + wait_time_ms
        self.avg_wait_time_ms = total_wait / self.acquisitions

    def record_hold_time(self, hold_time_ms: float) -> None:
        self.max_hold_time_ms = max(self.max_hold_time_ms, hold_time_ms)


class DistributedLock:
    """
    Redis-backed distributed lock with fencing tokens.

    Ensures only one process across multiple machines can hold the lock.
    Uses fencing tokens to prevent stale lock holders from causing issues.

    Usage:
        lock = DistributedLock("my_resource", redis_client)

        async with lock.acquire():
            # Critical section - only one holder at a time
            await do_exclusive_work()

        # Or with explicit token handling
        token = await lock.acquire_lock()
        try:
            # Use token for operations
            await guarded_operation(fencing_token=token)
        finally:
            await lock.release_lock(token)
    """

    # Lua script for atomic lock acquisition (CAS operation)
    _ACQUIRE_SCRIPT = """
    local lock_key = KEYS[1]
    local holder_id = ARGV[1]
    local fencing_token = ARGV[2]
    local ttl_ms = tonumber(ARGV[3])
    local allow_reentry = ARGV[4] == "1"

    -- Check if lock exists
    local current = redis.call('HGETALL', lock_key)

    if #current == 0 then
        -- Lock is free, acquire it
        redis.call('HSET', lock_key,
            'holder', holder_id,
            'token', fencing_token,
            'acquired_at', ARGV[5],
            'reentry_count', 1
        )
        redis.call('PEXPIRE', lock_key, ttl_ms)
        return {1, fencing_token}
    end

    -- Lock exists, check if we already hold it
    local current_holder = ''
    local current_token = ''
    local reentry_count = 0

    for i = 1, #current, 2 do
        if current[i] == 'holder' then
            current_holder = current[i+1]
        elseif current[i] == 'token' then
            current_token = current[i+1]
        elseif current[i] == 'reentry_count' then
            reentry_count = tonumber(current[i+1])
        end
    end

    if current_holder == holder_id and allow_reentry then
        -- Re-entry: increment count and extend TTL
        redis.call('HSET', lock_key, 'reentry_count', reentry_count + 1)
        redis.call('PEXPIRE', lock_key, ttl_ms)
        return {1, current_token}
    end

    -- Lock is held by someone else
    return {0, current_holder}
    """

    # Lua script for atomic lock release
    _RELEASE_SCRIPT = """
    local lock_key = KEYS[1]
    local expected_token = ARGV[1]

    local current = redis.call('HGETALL', lock_key)
    if #current == 0 then
        return {0, 'not_held'}
    end

    local current_token = ''
    local reentry_count = 0

    for i = 1, #current, 2 do
        if current[i] == 'token' then
            current_token = current[i+1]
        elseif current[i] == 'reentry_count' then
            reentry_count = tonumber(current[i+1])
        end
    end

    if current_token ~= expected_token then
        return {0, 'token_mismatch'}
    end

    if reentry_count > 1 then
        -- Decrement reentry count
        redis.call('HSET', lock_key, 'reentry_count', reentry_count - 1)
        return {1, 'decremented'}
    end

    -- Final release
    redis.call('DEL', lock_key)
    return {1, 'released'}
    """

    # Lua script for lock renewal
    _RENEW_SCRIPT = """
    local lock_key = KEYS[1]
    local expected_token = ARGV[1]
    local ttl_ms = tonumber(ARGV[2])

    local current_token = redis.call('HGET', lock_key, 'token')
    if current_token ~= expected_token then
        return 0
    end

    redis.call('PEXPIRE', lock_key, ttl_ms)
    return 1
    """

    def __init__(
        self,
        name: str,
        redis_client: Optional[Any] = None,  # ResilientRedisClient or redis.asyncio.Redis
        config: Optional[DistributedLockConfig] = None,
    ):
        self.name = name
        self._redis = redis_client
        self.config = config or DistributedLockConfig()

        self._lock_key = f"jarvis:lock:{name}"
        self._current_token: Optional[str] = None
        self._acquired_at: float = 0.0
        self._renewal_task: Optional[asyncio.Task] = None
        self._local_lock = asyncio.Lock()  # Fallback
        self._using_local_lock = False

        self.metrics = LockMetrics()

        # Script hashes (cached after first use)
        self._acquire_sha: Optional[str] = None
        self._release_sha: Optional[str] = None
        self._renew_sha: Optional[str] = None

    async def _load_scripts(self) -> None:
        """Load Lua scripts into Redis."""
        if not self._redis or self._acquire_sha:
            return

        try:
            self._acquire_sha = await self._redis.script_load(self._ACQUIRE_SCRIPT)
            self._release_sha = await self._redis.script_load(self._RELEASE_SCRIPT)
            self._renew_sha = await self._redis.script_load(self._RENEW_SCRIPT)
        except Exception as e:
            logger.warning(f"[DistributedLock:{self.name}] Failed to load scripts: {e}")
            self._acquire_sha = None

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Context manager for lock acquisition.

        Args:
            timeout: Max wait time (default from config)

        Yields:
            str: The fencing token for this acquisition

        Raises:
            LockAcquisitionError: If lock cannot be acquired
        """
        token = await self.acquire_lock(timeout)
        try:
            yield token
        finally:
            await self.release_lock(token)

    async def acquire_lock(self, timeout: Optional[float] = None) -> str:
        """
        Acquire the distributed lock.

        Args:
            timeout: Max wait time (default from config)

        Returns:
            str: Fencing token for this acquisition

        Raises:
            LockAcquisitionError: If lock cannot be acquired
        """
        timeout = timeout if timeout is not None else self.config.default_timeout
        start_time = time.time()
        fencing_token = f"{self.config.holder_id}-{uuid.uuid4().hex[:12]}-{int(time.time() * 1000)}"

        # Try Redis lock first
        if self._redis:
            await self._load_scripts()

            while time.time() - start_time < timeout:
                try:
                    result = await self._try_redis_acquire(fencing_token)

                    if result[0] == 1:
                        # Acquired
                        self._current_token = result[1]
                        self._acquired_at = time.time()
                        self._using_local_lock = False

                        wait_time_ms = (time.time() - start_time) * 1000
                        self.metrics.record_acquisition(wait_time_ms)

                        if wait_time_ms > 100:
                            self.metrics.contention_events += 1

                        # Start renewal task
                        if self.config.auto_renew:
                            self._start_renewal_task()

                        logger.debug(
                            f"[DistributedLock:{self.name}] Acquired (token={self._current_token[:16]}...)"
                        )
                        return self._current_token

                    # Lock held by someone else
                    holder = result[1]
                    await asyncio.sleep(self.config.retry_interval)

                except Exception as e:
                    logger.warning(f"[DistributedLock:{self.name}] Redis error: {e}")
                    if self.config.fallback_to_local:
                        break
                    raise

            # Timeout reached
            if not self.config.fallback_to_local:
                self.metrics.acquisition_failures += 1
                raise LockAcquisitionError(self.name, timeout)

        # Fallback to local lock
        if self.config.fallback_to_local:
            logger.info(f"[DistributedLock:{self.name}] Falling back to local lock")
            self.metrics.fallback_to_local += 1

            remaining = max(0, timeout - (time.time() - start_time))
            try:
                acquired = await asyncio.wait_for(
                    self._local_lock.acquire(),
                    timeout=remaining
                )
                if acquired:
                    self._current_token = fencing_token
                    self._acquired_at = time.time()
                    self._using_local_lock = True

                    wait_time_ms = (time.time() - start_time) * 1000
                    self.metrics.record_acquisition(wait_time_ms)

                    return self._current_token
            except asyncio.TimeoutError:
                pass

        self.metrics.acquisition_failures += 1
        raise LockAcquisitionError(self.name, timeout)

    async def _try_redis_acquire(self, fencing_token: str) -> tuple:
        """Attempt to acquire lock via Redis."""
        ttl_ms = int(self.config.lock_ttl * 1000)
        acquired_at = str(time.time())
        allow_reentry = "1" if self.config.allow_reentry else "0"

        if self._acquire_sha:
            try:
                return await self._redis.evalsha(
                    self._acquire_sha,
                    1,
                    self._lock_key,
                    self.config.holder_id,
                    fencing_token,
                    str(ttl_ms),
                    allow_reentry,
                    acquired_at,
                )
            except Exception:
                # Script might have been flushed
                self._acquire_sha = None

        # Fallback to eval
        return await self._redis.eval(
            self._ACQUIRE_SCRIPT,
            1,
            self._lock_key,
            self.config.holder_id,
            fencing_token,
            str(ttl_ms),
            allow_reentry,
            acquired_at,
        )

    async def release_lock(self, token: Optional[str] = None) -> None:
        """
        Release the distributed lock.

        Args:
            token: Fencing token (uses current if not provided)

        Raises:
            LockNotHeldError: If lock is not held by this token
        """
        token = token or self._current_token
        if not token:
            logger.warning(f"[DistributedLock:{self.name}] Release called but no token")
            return

        # Stop renewal
        self._stop_renewal_task()

        # Record hold time
        if self._acquired_at:
            hold_time_ms = (time.time() - self._acquired_at) * 1000
            self.metrics.record_hold_time(hold_time_ms)

        if self._using_local_lock:
            if self._local_lock.locked():
                self._local_lock.release()
            self._current_token = None
            self._acquired_at = 0.0
            self.metrics.releases += 1
            logger.debug(f"[DistributedLock:{self.name}] Released local lock")
            return

        if not self._redis:
            self._current_token = None
            return

        try:
            if self._release_sha:
                try:
                    result = await self._redis.evalsha(
                        self._release_sha,
                        1,
                        self._lock_key,
                        token,
                    )
                except Exception:
                    self._release_sha = None
                    result = await self._redis.eval(
                        self._RELEASE_SCRIPT,
                        1,
                        self._lock_key,
                        token,
                    )
            else:
                result = await self._redis.eval(
                    self._RELEASE_SCRIPT,
                    1,
                    self._lock_key,
                    token,
                )

            if result[0] == 0:
                if result[1] == "token_mismatch":
                    actual = await self._redis.hget(self._lock_key, "token")
                    raise LockNotHeldError(self.name, token, actual)

            self._current_token = None
            self._acquired_at = 0.0
            self.metrics.releases += 1

            logger.debug(
                f"[DistributedLock:{self.name}] Released (result={result[1]})"
            )

        except LockNotHeldError:
            raise
        except Exception as e:
            logger.error(f"[DistributedLock:{self.name}] Release error: {e}")
            # Best effort - clear local state
            self._current_token = None
            self._acquired_at = 0.0

    def _start_renewal_task(self) -> None:
        """Start background lock renewal."""
        if self._renewal_task and not self._renewal_task.done():
            return

        self._renewal_task = asyncio.create_task(self._renewal_loop())

    def _stop_renewal_task(self) -> None:
        """Stop background lock renewal."""
        if self._renewal_task:
            self._renewal_task.cancel()
            self._renewal_task = None

    async def _renewal_loop(self) -> None:
        """Background task to renew lock TTL."""
        while True:
            try:
                await asyncio.sleep(self.config.renewal_interval)

                if not self._current_token:
                    break

                success = await self._renew_lock()
                if not success:
                    logger.warning(
                        f"[DistributedLock:{self.name}] Renewal failed - lock may have been stolen"
                    )
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[DistributedLock:{self.name}] Renewal error: {e}")
                self.metrics.renewal_failures += 1

    async def _renew_lock(self) -> bool:
        """Renew lock TTL in Redis."""
        if not self._redis or not self._current_token:
            return False

        ttl_ms = int(self.config.lock_ttl * 1000)

        try:
            if self._renew_sha:
                try:
                    result = await self._redis.evalsha(
                        self._renew_sha,
                        1,
                        self._lock_key,
                        self._current_token,
                        str(ttl_ms),
                    )
                except Exception:
                    self._renew_sha = None
                    result = await self._redis.eval(
                        self._RENEW_SCRIPT,
                        1,
                        self._lock_key,
                        self._current_token,
                        str(ttl_ms),
                    )
            else:
                result = await self._redis.eval(
                    self._RENEW_SCRIPT,
                    1,
                    self._lock_key,
                    self._current_token,
                    str(ttl_ms),
                )

            if result == 1:
                self.metrics.renewals += 1
                return True
            return False

        except Exception as e:
            logger.error(f"[DistributedLock:{self.name}] Renew error: {e}")
            self.metrics.renewal_failures += 1
            return False

    @property
    def is_held(self) -> bool:
        """Check if lock is currently held by this instance."""
        return self._current_token is not None

    @property
    def fencing_token(self) -> Optional[str]:
        """Get current fencing token."""
        return self._current_token

    async def get_holder_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current lock holder."""
        if not self._redis:
            return None

        try:
            data = await self._redis.hgetall(self._lock_key)
            if not data:
                return None

            # Decode bytes if needed
            return {
                k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) else v
                for k, v in data.items()
            }
        except Exception as e:
            logger.error(f"[DistributedLock:{self.name}] Failed to get holder info: {e}")
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get lock metrics."""
        return {
            "name": self.name,
            "is_held": self.is_held,
            "using_local_fallback": self._using_local_lock,
            "acquisitions": self.metrics.acquisitions,
            "acquisition_failures": self.metrics.acquisition_failures,
            "releases": self.metrics.releases,
            "renewals": self.metrics.renewals,
            "renewal_failures": self.metrics.renewal_failures,
            "fallback_to_local": self.metrics.fallback_to_local,
            "contention_events": self.metrics.contention_events,
            "avg_wait_time_ms": round(self.metrics.avg_wait_time_ms, 2),
            "max_hold_time_ms": round(self.metrics.max_hold_time_ms, 2),
        }
