"""
HealthProbe - Cached Health Checks with Failure Tracking
=========================================================

This module provides a health probe implementation that caches health check
results and tracks consecutive failures for determining unhealthy state.

Features:
- Configurable cache TTL to reduce expensive health check calls
- Per-check timeout to prevent slow checks from blocking
- Consecutive failure tracking with configurable threshold
- Async callbacks for unhealthy/healthy state transitions
- Force bypass for cache when immediate status is needed
- Thread-safe with asyncio.Lock

The health probe pattern helps with:
- Reducing load on health check endpoints
- Detecting persistent failures vs transient issues
- Triggering alerts or fallback behavior when unhealthy
- Automatic recovery detection when service returns

Example usage:
    from backend.core.resilience.health import HealthProbe

    # Basic usage
    async def check_database():
        try:
            await db.execute("SELECT 1")
            return True
        except Exception:
            return False

    probe = HealthProbe(check_fn=check_database)
    is_healthy = await probe.check()

    # With callbacks
    async def alert_team():
        await send_slack_message("Database is unhealthy!")

    async def notify_recovery():
        await send_slack_message("Database recovered!")

    probe = HealthProbe(
        check_fn=check_database,
        cache_ttl=30.0,
        timeout=5.0,
        unhealthy_threshold=3,
        on_unhealthy=alert_team,
        on_healthy=notify_recovery,
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import (
    Awaitable,
    Callable,
)


@dataclass
class HealthProbe:
    """
    Cached health probe with failure tracking and state callbacks.

    This class provides a health check mechanism that caches results to reduce
    load on health endpoints, tracks consecutive failures to determine unhealthy
    state, and provides async callbacks for state transitions.

    The probe has two main states:
    - Healthy: consecutive_failures < unhealthy_threshold
    - Unhealthy: consecutive_failures >= unhealthy_threshold

    State Transitions:
        Healthy -> Unhealthy: When consecutive_failures reaches threshold
        Unhealthy -> Healthy: When a check succeeds (consecutive_failures reset)

    Attributes:
        check_fn: Async function that returns True if healthy, False otherwise.
        cache_ttl: Seconds to cache the result. Default is 30.0.
        timeout: Per-check timeout in seconds. Default is 10.0.
        unhealthy_threshold: Consecutive failures before unhealthy. Default is 3.
        on_unhealthy: Optional async callback called when becoming unhealthy.
        on_healthy: Optional async callback called when recovering to healthy.

    Example:
        # Basic usage
        probe = HealthProbe(check_fn=my_health_check)
        is_healthy = await probe.check()

        # Customized probe with callbacks
        async def db_check():
            try:
                await pool.execute("SELECT 1")
                return True
            except Exception:
                return False

        async def on_db_unhealthy():
            await switch_to_readonly_mode()

        probe = HealthProbe(
            check_fn=db_check,
            cache_ttl=15.0,
            timeout=5.0,
            unhealthy_threshold=5,
            on_unhealthy=on_db_unhealthy,
        )
    """

    check_fn: Callable[[], Awaitable[bool]]
    cache_ttl: float = 30.0
    timeout: float = 10.0
    unhealthy_threshold: int = 3
    on_unhealthy: Callable[[], Awaitable[None]] | None = None
    on_healthy: Callable[[], Awaitable[None]] | None = None

    # Internal state (not exposed as constructor params)
    _cached_result: bool | None = field(default=None, init=False, repr=False)
    _cache_time: float = field(default=0.0, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _is_unhealthy: bool = field(default=False, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def consecutive_failures(self) -> int:
        """
        Get the current consecutive failure count.

        Returns:
            The number of consecutive failed health checks since last success.
        """
        return self._consecutive_failures

    @property
    def is_unhealthy(self) -> bool:
        """
        Get whether the probe is in unhealthy state.

        Returns:
            True if consecutive_failures >= unhealthy_threshold, False otherwise.
        """
        return self._is_unhealthy

    async def check(self, force: bool = False) -> bool:
        """
        Perform a health check with caching.

        If a cached result exists and is within the TTL, returns the cached
        value. Otherwise, calls the check_fn to get a fresh result. The
        force parameter can be used to bypass the cache.

        On success (True result):
        - Resets consecutive_failures to 0
        - If was unhealthy, calls on_healthy callback and sets is_unhealthy to False

        On failure (False result or exception/timeout):
        - Increments consecutive_failures
        - If consecutive_failures reaches threshold, sets is_unhealthy to True
          and calls on_unhealthy callback (once)

        Args:
            force: If True, bypass the cache and perform a fresh check.

        Returns:
            True if healthy, False if unhealthy or on error/timeout.
            Never raises exceptions - always returns a boolean.

        Example:
            # Normal check (uses cache)
            is_healthy = await probe.check()

            # Force fresh check (bypasses cache)
            is_healthy = await probe.check(force=True)
        """
        async with self._lock:
            # Check if we can use cached result
            if not force and self._cached_result is not None:
                elapsed = time.monotonic() - self._cache_time
                if elapsed < self.cache_ttl:
                    return self._cached_result

            # Perform the actual health check
            result = await self._perform_check()

            # Update cache
            self._cached_result = result
            self._cache_time = time.monotonic()

            # Update failure tracking and state
            if result:
                await self._handle_success()
            else:
                await self._handle_failure()

            return result

    async def _perform_check(self) -> bool:
        """
        Perform the actual health check with timeout.

        Calls check_fn with the configured timeout. Returns False on any
        exception or timeout - never raises.

        Returns:
            The result of check_fn, or False on exception/timeout.
        """
        try:
            result = await asyncio.wait_for(
                self.check_fn(),
                timeout=self.timeout,
            )
            return result
        except Exception:
            # Timeout, exception from check_fn, or any other error
            return False

    async def _handle_success(self) -> None:
        """
        Handle a successful health check.

        Resets consecutive failures and transitions from unhealthy to healthy
        if applicable, calling the on_healthy callback.

        Note: This method should be called with the lock held.
        """
        was_unhealthy = self._is_unhealthy

        # Reset failure tracking
        self._consecutive_failures = 0

        # Transition to healthy if was unhealthy
        if was_unhealthy:
            self._is_unhealthy = False
            if self.on_healthy is not None:
                await self.on_healthy()

    async def _handle_failure(self) -> None:
        """
        Handle a failed health check.

        Increments consecutive failures and transitions to unhealthy state
        if threshold is reached, calling the on_unhealthy callback (once).

        Note: This method should be called with the lock held.
        """
        was_unhealthy = self._is_unhealthy

        # Increment failure count
        self._consecutive_failures += 1

        # Transition to unhealthy if threshold reached (and wasn't already)
        if not was_unhealthy and self._consecutive_failures >= self.unhealthy_threshold:
            self._is_unhealthy = True
            if self.on_unhealthy is not None:
                await self.on_unhealthy()

    def reset(self) -> None:
        """
        Reset the health probe to initial state.

        Clears the cached result, resets consecutive failures to 0,
        and sets is_unhealthy to False. Does not call any callbacks.

        This is useful for manual reset after an operator has verified
        the service has recovered, or during initialization.

        Example:
            # Reset probe after manual intervention
            probe.reset()

            # Now perform a fresh check
            is_healthy = await probe.check()
        """
        self._cached_result = None
        self._cache_time = 0.0
        self._consecutive_failures = 0
        self._is_unhealthy = False


__all__ = [
    "HealthProbe",
]
