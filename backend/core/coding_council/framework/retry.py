"""
v77.0: Retry with Exponential Backoff - Gap #12
================================================

Retry mechanisms:
- Configurable retry policies
- Exponential backoff with jitter
- Per-exception retry rules
- Maximum attempt limits
- Retry budgets

Author: JARVIS v77.0
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1  # Random jitter factor (0-1)
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    retry_on_result: Optional[Callable[[Any], bool]] = None  # Retry if returns True

    def should_retry(self, exception: Optional[Exception], result: Any) -> bool:
        """Determine if operation should be retried."""
        if exception:
            # Check non-retryable first
            if isinstance(exception, self.non_retryable_exceptions):
                return False
            return isinstance(exception, self.retryable_exceptions)

        # Check result-based retry
        if self.retry_on_result:
            return self.retry_on_result(result)

        return False


@dataclass
class RetryStats:
    """Statistics for retry operations."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retries: int = 0
    total_delay: float = 0.0
    last_error: Optional[str] = None


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
) -> float:
    """
    Calculate exponential backoff delay with jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential growth
        jitter: Random jitter factor (0-1)

    Returns:
        Delay in seconds
    """
    # Calculate base exponential delay
    delay = base_delay * (exponential_base ** attempt)

    # Apply cap
    delay = min(delay, max_delay)

    # Add jitter (±jitter%)
    if jitter > 0:
        jitter_amount = delay * jitter
        delay += random.uniform(-jitter_amount, jitter_amount)

    return max(0, delay)


class RetryContext:
    """Context for a retry operation."""

    def __init__(self, policy: RetryPolicy, operation_name: str = "operation"):
        self.policy = policy
        self.operation_name = operation_name
        self.attempt = 0
        self.errors: List[Exception] = []
        self.start_time = time.time()
        self.stats = RetryStats()

    @property
    def is_last_attempt(self) -> bool:
        return self.attempt >= self.policy.max_attempts - 1

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def get_delay(self) -> float:
        """Get delay for next retry."""
        return exponential_backoff(
            self.attempt,
            self.policy.base_delay,
            self.policy.max_delay,
            self.policy.exponential_base,
            self.policy.jitter,
        )

    def record_error(self, error: Exception) -> None:
        """Record an error for this context."""
        self.errors.append(error)
        self.stats.failed_attempts += 1
        self.stats.last_error = str(error)


async def retry_async(
    func: Callable[..., Coroutine],
    *args,
    policy: Optional[RetryPolicy] = None,
    operation_name: str = "operation",
    on_retry: Optional[Callable[[int, Exception], Coroutine]] = None,
    **kwargs,
) -> Any:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        policy: Retry policy configuration
        operation_name: Name for logging
        on_retry: Optional callback before each retry
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        Last exception if all retries fail
    """
    policy = policy or RetryPolicy()
    ctx = RetryContext(policy, operation_name)

    while ctx.attempt < policy.max_attempts:
        ctx.stats.total_attempts += 1

        try:
            result = await func(*args, **kwargs)

            # Check if result triggers retry
            if policy.should_retry(None, result):
                if ctx.is_last_attempt:
                    return result

                delay = ctx.get_delay()
                logger.debug(
                    f"[Retry] {operation_name} result triggered retry, "
                    f"attempt {ctx.attempt + 1}/{policy.max_attempts}, "
                    f"waiting {delay:.2f}s"
                )
                ctx.stats.total_delay += delay
                await asyncio.sleep(delay)
                ctx.attempt += 1
                ctx.stats.total_retries += 1
                continue

            ctx.stats.successful_attempts += 1
            return result

        except Exception as e:
            ctx.record_error(e)

            if not policy.should_retry(e, None):
                logger.warning(f"[Retry] {operation_name} non-retryable error: {e}")
                raise

            if ctx.is_last_attempt:
                logger.error(
                    f"[Retry] {operation_name} failed after {policy.max_attempts} attempts: {e}"
                )
                raise

            # Call retry callback if provided
            if on_retry:
                try:
                    await on_retry(ctx.attempt, e)
                except Exception as callback_error:
                    logger.error(f"[Retry] on_retry callback error: {callback_error}")

            delay = ctx.get_delay()
            logger.warning(
                f"[Retry] {operation_name} attempt {ctx.attempt + 1}/{policy.max_attempts} "
                f"failed: {e}, retrying in {delay:.2f}s"
            )

            ctx.stats.total_delay += delay
            await asyncio.sleep(delay)
            ctx.attempt += 1
            ctx.stats.total_retries += 1

    # Should not reach here
    raise RuntimeError(f"{operation_name} failed: max retries exceeded")


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
    operation_name: Optional[str] = None,
):
    """
    Decorator to add retry logic to async functions.

    Usage:
        @retry(max_attempts=3, base_delay=1.0)
        async def flaky_operation():
            ...
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
    )

    def decorator(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            return await retry_async(func, *args, policy=policy, operation_name=name, **kwargs)
        return wrapper

    return decorator


class RetryBudget:
    """
    Budget-based retry limiting.

    Limits total retries across operations to prevent
    cascading failures during outages.
    """

    def __init__(
        self,
        budget_per_second: float = 10.0,
        min_retries_per_second: float = 1.0,
        window_seconds: float = 10.0,
    ):
        self.budget_per_second = budget_per_second
        self.min_retries_per_second = min_retries_per_second
        self.window_seconds = window_seconds
        self._requests: List[float] = []
        self._retries: List[float] = []
        self._lock = asyncio.Lock()

    async def record_request(self) -> None:
        """Record a request."""
        async with self._lock:
            now = time.time()
            self._requests.append(now)
            self._cleanup(now)

    async def try_retry(self) -> bool:
        """
        Try to perform a retry within budget.

        Returns True if retry is allowed.
        """
        async with self._lock:
            now = time.time()
            self._cleanup(now)

            # Calculate allowed retries
            request_count = len(self._requests)
            current_retries = len(self._retries)

            # Allow at least min_retries_per_second
            min_allowed = self.min_retries_per_second * self.window_seconds

            # Budget is percentage of requests
            budget_allowed = request_count * (self.budget_per_second / 100.0)

            max_allowed = max(min_allowed, budget_allowed)

            if current_retries < max_allowed:
                self._retries.append(now)
                return True

            return False

    def _cleanup(self, now: float) -> None:
        """Remove old entries outside window."""
        cutoff = now - self.window_seconds
        self._requests = [t for t in self._requests if t >= cutoff]
        self._retries = [t for t in self._retries if t >= cutoff]

    def get_stats(self) -> Dict[str, Any]:
        """Get budget statistics."""
        return {
            "requests_in_window": len(self._requests),
            "retries_in_window": len(self._retries),
            "budget_per_second": self.budget_per_second,
            "window_seconds": self.window_seconds,
        }


# Global retry budget
_retry_budget: Optional[RetryBudget] = None


def get_retry_budget() -> RetryBudget:
    """Get global retry budget."""
    global _retry_budget
    if _retry_budget is None:
        _retry_budget = RetryBudget()
    return _retry_budget
