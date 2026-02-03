"""
RetryPolicy - Exponential Backoff with Jitter for Resilient Operations
======================================================================

This module provides a configurable retry policy with exponential backoff
and optional jitter for handling transient failures in async operations.

Features:
- Configurable maximum attempts
- Exponential backoff with customizable base and exponent
- Maximum delay cap to prevent infinite backoff
- Jitter to prevent thundering herd problems
- Per-attempt timeout support
- Selective exception filtering (only retry specified exceptions)
- Callback hook for retry events (logging, metrics, etc.)

Example usage:
    from backend.core.resilience.retry import RetryPolicy, RetryExhausted

    # Basic usage
    policy = RetryPolicy(max_attempts=5)
    result = await policy.execute(my_async_function, arg1, arg2, kwarg=value)

    # With callback
    async def log_retry(exc: Exception, attempt: int) -> None:
        logger.warning(f"Retry attempt {attempt} after {exc}")

    policy = RetryPolicy(
        max_attempts=5,
        base_delay=1.0,
        max_delay=30.0,
        jitter=0.1,
        timeout=10.0,
        retry_on=(ConnectionError, TimeoutError),
        on_retry=log_retry,
    )

    try:
        result = await policy.execute(api_call)
    except RetryExhausted as e:
        logger.error(f"All retries exhausted: {e.last_exception}")
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Tuple,
    Type,
    TypeVar,
)

T = TypeVar("T")


class RetryExhausted(Exception):
    """
    Raised when all retry attempts have been exhausted.

    Attributes:
        last_exception: The last exception that caused the final failure.
                       May be None if the failure was due to some other reason.

    Example:
        try:
            result = await policy.execute(unreliable_function)
        except RetryExhausted as e:
            logger.error(f"All retries failed. Last error: {e.last_exception}")
            raise
    """

    def __init__(self, message: str, last_exception: Exception | None = None) -> None:
        """
        Initialize RetryExhausted exception.

        Args:
            message: Human-readable description of the failure
            last_exception: The exception that caused the final failure
        """
        super().__init__(message)
        self.last_exception = last_exception


@dataclass
class RetryPolicy:
    """
    Configurable retry policy with exponential backoff and jitter.

    This class provides a flexible retry mechanism for async operations
    that may fail due to transient issues. It supports exponential backoff
    to reduce load on failing services, jitter to prevent synchronized
    retry storms, and selective exception filtering.

    Delay formula: base_delay * exponential_base^(attempt-1), capped at max_delay
    Jitter: random value between delay*(1-jitter) and delay*(1+jitter)

    Attributes:
        max_attempts: Maximum number of attempts (including the first try).
                     Must be >= 1. Default is 3.
        base_delay: Initial delay between retries in seconds. Default is 1.0.
        max_delay: Maximum delay cap in seconds. Default is 60.0.
        exponential_base: Multiplier for exponential backoff. Default is 2.0.
        jitter: Random +/- percentage (0.0 to 1.0). Default is 0.1 (10%).
        timeout: Per-attempt timeout in seconds. None means no timeout.
        retry_on: Tuple of exception types to retry on. Default is (Exception,).
        on_retry: Optional async callback(exception, attempt) called before each retry.

    Example:
        # Simple retry with defaults
        policy = RetryPolicy()
        result = await policy.execute(my_async_func)

        # Customized retry
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=0.2,
            timeout=10.0,
            retry_on=(ConnectionError, TimeoutError),
        )
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    timeout: float | None = None
    retry_on: Tuple[Type[Exception], ...] = field(default=(Exception,))
    on_retry: Callable[[Exception, int], Awaitable[None]] | None = None

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate the delay before the next retry attempt.

        Uses exponential backoff: base_delay * exponential_base^(attempt-1)
        with the result capped at max_delay. Jitter is then applied as a
        random multiplier between (1-jitter) and (1+jitter).

        Args:
            attempt: The attempt number (1-indexed). First retry is attempt 1.

        Returns:
            The delay in seconds to wait before the next attempt.

        Example:
            # With base_delay=1.0, exponential_base=2.0, max_delay=10.0:
            # attempt 1: 1.0 * 2^0 = 1.0
            # attempt 2: 1.0 * 2^1 = 2.0
            # attempt 3: 1.0 * 2^2 = 4.0
            # attempt 4: 1.0 * 2^3 = 8.0
            # attempt 5: 1.0 * 2^4 = 16.0 -> capped to 10.0
        """
        # Calculate base exponential delay
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))

        # Cap at max_delay before applying jitter
        delay = min(delay, self.max_delay)

        # Apply jitter as a random multiplier
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay = delay + random.uniform(-jitter_range, jitter_range)
            # Ensure we don't go negative or exceed max_delay after jitter
            delay = max(0.0, min(delay, self.max_delay))

        return delay

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with the configured retry policy.

        Attempts to execute the given async function up to max_attempts times.
        On failure (if the exception matches retry_on), waits with exponential
        backoff before retrying. If all attempts fail, raises RetryExhausted
        with the last exception.

        Args:
            func: The async function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value from the successful function call.

        Raises:
            RetryExhausted: If all retry attempts fail.
            Exception: If an exception not in retry_on is raised (propagated immediately).

        Example:
            async def fetch_data(url: str) -> dict:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        return await response.json()

            policy = RetryPolicy(max_attempts=3, retry_on=(aiohttp.ClientError,))
            data = await policy.execute(fetch_data, "https://api.example.com/data")
        """
        last_exception: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                # Apply per-attempt timeout if configured
                if self.timeout is not None:
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.timeout,
                    )
                else:
                    result = await func(*args, **kwargs)

                # Success - return the result
                return result

            except Exception as exc:
                last_exception = exc

                # Check if this exception type should be retried
                if not isinstance(exc, self.retry_on):
                    # Not a retryable exception - propagate immediately
                    raise

                # Check if we have more attempts
                if attempt >= self.max_attempts:
                    # No more attempts - will raise RetryExhausted
                    break

                # Call on_retry callback if provided
                if self.on_retry is not None:
                    await self.on_retry(exc, attempt)

                # Calculate and apply delay before next attempt
                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)

        # All attempts exhausted
        raise RetryExhausted(
            f"All {self.max_attempts} retry attempts exhausted",
            last_exception=last_exception,
        )


__all__ = [
    "RetryPolicy",
    "RetryExhausted",
]
