"""
Comprehensive tests for RetryPolicy with exponential backoff.

Tests cover:
- Succeeds on first try
- Retries on failure then succeeds
- Raises RetryExhausted after max_attempts
- Respects retry_on filter (only retry specified exceptions)
- Exponential backoff delays (mock asyncio.sleep)
- max_delay caps backoff
- Jitter adds randomness
- Per-attempt timeout cancels slow attempts
- on_retry callback called with exception and attempt number
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import random

from backend.core.resilience.retry import (
    RetryPolicy,
    RetryExhausted,
)


class TestRetryExhausted:
    """Tests for RetryExhausted exception."""

    def test_exception_has_last_exception(self):
        """RetryExhausted should store the last exception that caused failure."""
        original = ValueError("original error")
        exc = RetryExhausted("All attempts failed", last_exception=original)
        assert exc.last_exception is original

    def test_exception_inherits_from_exception(self):
        """RetryExhausted should inherit from Exception."""
        assert issubclass(RetryExhausted, Exception)

    def test_exception_message_preserved(self):
        """RetryExhausted should preserve the error message."""
        exc = RetryExhausted("Custom message", last_exception=None)
        assert "Custom message" in str(exc)

    def test_exception_without_last_exception(self):
        """RetryExhausted should work with last_exception=None."""
        exc = RetryExhausted("No last exception", last_exception=None)
        assert exc.last_exception is None


class TestRetryPolicyDefaults:
    """Tests for RetryPolicy default values."""

    def test_default_max_attempts(self):
        """Default max_attempts should be 3."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3

    def test_default_base_delay(self):
        """Default base_delay should be 1.0."""
        policy = RetryPolicy()
        assert policy.base_delay == 1.0

    def test_default_max_delay(self):
        """Default max_delay should be 60.0."""
        policy = RetryPolicy()
        assert policy.max_delay == 60.0

    def test_default_exponential_base(self):
        """Default exponential_base should be 2.0."""
        policy = RetryPolicy()
        assert policy.exponential_base == 2.0

    def test_default_jitter(self):
        """Default jitter should be 0.1."""
        policy = RetryPolicy()
        assert policy.jitter == 0.1

    def test_default_timeout(self):
        """Default timeout should be None."""
        policy = RetryPolicy()
        assert policy.timeout is None

    def test_default_retry_on(self):
        """Default retry_on should be (Exception,)."""
        policy = RetryPolicy()
        assert policy.retry_on == (Exception,)

    def test_default_on_retry(self):
        """Default on_retry should be None."""
        policy = RetryPolicy()
        assert policy.on_retry is None


class TestRetryPolicyCalculateDelay:
    """Tests for _calculate_delay method."""

    def test_first_attempt_uses_base_delay(self):
        """First retry (attempt 1) should use base_delay as base."""
        policy = RetryPolicy(base_delay=1.0, jitter=0.0)
        delay = policy._calculate_delay(1)
        assert delay == 1.0

    def test_exponential_growth(self):
        """Delay should grow exponentially with attempt number."""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, jitter=0.0)
        # Attempt 1: 1.0 * 2^0 = 1.0
        assert policy._calculate_delay(1) == 1.0
        # Attempt 2: 1.0 * 2^1 = 2.0
        assert policy._calculate_delay(2) == 2.0
        # Attempt 3: 1.0 * 2^2 = 4.0
        assert policy._calculate_delay(3) == 4.0

    def test_max_delay_caps_growth(self):
        """Delay should be capped at max_delay."""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, max_delay=3.0, jitter=0.0)
        # Attempt 3: 1.0 * 2^2 = 4.0, capped to 3.0
        assert policy._calculate_delay(3) == 3.0

    def test_jitter_adds_randomness(self):
        """Jitter should add randomness to the delay."""
        policy = RetryPolicy(base_delay=1.0, jitter=0.5)
        # With 0.5 jitter, delay should be in range [0.5, 1.5]
        delays = [policy._calculate_delay(1) for _ in range(100)]
        assert min(delays) >= 0.5
        assert max(delays) <= 1.5
        # Verify there's actual variation
        assert len(set(delays)) > 1

    def test_zero_jitter_deterministic(self):
        """With zero jitter, delay should be deterministic."""
        policy = RetryPolicy(base_delay=1.0, jitter=0.0)
        delays = [policy._calculate_delay(1) for _ in range(10)]
        assert all(d == 1.0 for d in delays)

    def test_jitter_respects_max_delay(self):
        """Jitter should not push delay above max_delay."""
        policy = RetryPolicy(base_delay=1.0, max_delay=1.0, jitter=0.5)
        # Even with jitter that could push to 1.5, cap to 1.0
        for _ in range(100):
            assert policy._calculate_delay(1) <= 1.0


class TestRetryPolicyExecute:
    """Tests for execute method."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        """Function that succeeds immediately should not retry."""
        policy = RetryPolicy()
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await policy.execute(success_func)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        """Function should be retried on failure until success."""
        policy = RetryPolicy(max_attempts=5, base_delay=0.001, jitter=0.0)
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await policy.execute(fail_then_succeed)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_retry_exhausted_after_max_attempts(self):
        """Should raise RetryExhausted after max_attempts failures."""
        policy = RetryPolicy(max_attempts=3, base_delay=0.001, jitter=0.0)
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RetryExhausted) as exc_info:
                await policy.execute(always_fails)

        assert call_count == 3
        assert isinstance(exc_info.value.last_exception, ValueError)
        assert "always fails" in str(exc_info.value.last_exception)

    @pytest.mark.asyncio
    async def test_respects_retry_on_filter(self):
        """Should only retry specified exception types."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.001,
            jitter=0.0,
            retry_on=(ValueError,),
        )
        call_count = 0

        async def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        # TypeError is not in retry_on, so it should propagate immediately
        with pytest.raises(TypeError):
            await policy.execute(raises_type_error)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_specified_exception_types(self):
        """Should retry the specified exception types."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.001,
            jitter=0.0,
            retry_on=(ValueError, KeyError),
        )
        call_count = 0

        async def raises_value_error_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("retryable")
            return "success"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await policy.execute(raises_value_error_twice)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Should apply exponential backoff between retries."""
        policy = RetryPolicy(
            max_attempts=4,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=0.0,
        )

        async def always_fails():
            raise ValueError("fail")

        sleep_mock = AsyncMock()
        with patch("asyncio.sleep", sleep_mock):
            with pytest.raises(RetryExhausted):
                await policy.execute(always_fails)

        # Check sleep calls: 3 sleeps for 4 attempts
        assert sleep_mock.call_count == 3
        # Delays: 1.0, 2.0, 4.0
        sleep_mock.assert_any_call(1.0)
        sleep_mock.assert_any_call(2.0)
        sleep_mock.assert_any_call(4.0)

    @pytest.mark.asyncio
    async def test_per_attempt_timeout(self):
        """Per-attempt timeout should cancel slow attempts."""
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.001,
            jitter=0.0,
            timeout=0.05,  # 50ms timeout
        )
        call_count = 0

        async def slow_func():
            nonlocal call_count
            call_count += 1
            # Use a very long sleep that will definitely be cancelled by timeout
            await asyncio.sleep(10.0)
            return "success"

        with pytest.raises(RetryExhausted) as exc_info:
            await policy.execute(slow_func)

        assert call_count == 3
        assert isinstance(exc_info.value.last_exception, asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_on_retry_callback_called(self):
        """on_retry callback should be called with exception and attempt number."""
        callback = AsyncMock()
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.001,
            jitter=0.0,
            on_retry=callback,
        )

        async def fails_twice():
            if callback.call_count < 2:
                raise ValueError("fail")
            return "success"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await policy.execute(fails_twice)

        assert result == "success"
        assert callback.call_count == 2
        # First call: exception, attempt 1
        first_call_args = callback.call_args_list[0][0]
        assert isinstance(first_call_args[0], ValueError)
        assert first_call_args[1] == 1
        # Second call: exception, attempt 2
        second_call_args = callback.call_args_list[1][0]
        assert isinstance(second_call_args[0], ValueError)
        assert second_call_args[1] == 2

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self):
        """execute should pass *args and **kwargs to the function."""
        policy = RetryPolicy()

        async def func_with_args(a, b, c=None):
            return (a, b, c)

        result = await policy.execute(func_with_args, 1, 2, c=3)
        assert result == (1, 2, 3)

    @pytest.mark.asyncio
    async def test_returns_generic_type(self):
        """execute should return the generic type T."""
        policy = RetryPolicy()

        async def returns_dict() -> dict:
            return {"key": "value"}

        result = await policy.execute(returns_dict)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_max_attempts_of_one(self):
        """With max_attempts=1, should fail immediately without retry."""
        policy = RetryPolicy(max_attempts=1)
        call_count = 0

        async def fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        with pytest.raises(RetryExhausted):
            await policy.execute(fails)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_sync_function_wrapped(self):
        """Should handle sync functions properly (if wrapped)."""
        policy = RetryPolicy()
        call_count = 0

        async def sync_wrapper():
            nonlocal call_count
            call_count += 1
            return "sync_result"

        result = await policy.execute(sync_wrapper)
        assert result == "sync_result"
        assert call_count == 1


class TestRetryPolicyEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_large_max_delay(self):
        """Should handle very large max_delay values."""
        policy = RetryPolicy(
            max_attempts=2,
            base_delay=1000.0,
            max_delay=1000000.0,
            jitter=0.0,
        )
        delay = policy._calculate_delay(1)
        assert delay == 1000.0

    @pytest.mark.asyncio
    async def test_retry_on_subclass_exception(self):
        """Should retry on subclasses of specified exceptions."""
        class CustomError(ValueError):
            pass

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.001,
            jitter=0.0,
            retry_on=(ValueError,),  # Parent class
        )
        call_count = 0

        async def raises_custom_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise CustomError("subclass error")
            return "success"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await policy.execute(raises_custom_error)

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_jitter_boundary_values(self):
        """Test jitter at boundary values 0.0 and 1.0."""
        # Zero jitter
        policy_no_jitter = RetryPolicy(base_delay=1.0, jitter=0.0)
        assert policy_no_jitter._calculate_delay(1) == 1.0

        # Max jitter (1.0)
        policy_max_jitter = RetryPolicy(base_delay=1.0, jitter=1.0)
        delays = [policy_max_jitter._calculate_delay(1) for _ in range(100)]
        # Range should be [0.0, 2.0]
        assert min(delays) >= 0.0
        assert max(delays) <= 2.0

    @pytest.mark.asyncio
    async def test_callback_not_called_on_success(self):
        """on_retry callback should not be called if first attempt succeeds."""
        callback = AsyncMock()
        policy = RetryPolicy(on_retry=callback)

        async def succeeds():
            return "success"

        result = await policy.execute(succeeds)
        assert result == "success"
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_retry_on_tuple(self):
        """Empty retry_on tuple should not retry any exceptions."""
        policy = RetryPolicy(
            max_attempts=3,
            retry_on=(),
        )

        async def raises():
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            await policy.execute(raises)


class TestModuleExports:
    """Tests for module exports and structure."""

    def test_retry_policy_importable(self):
        """RetryPolicy should be importable from retry module."""
        from backend.core.resilience.retry import RetryPolicy
        assert RetryPolicy is not None

    def test_retry_exhausted_importable(self):
        """RetryExhausted should be importable from retry module."""
        from backend.core.resilience.retry import RetryExhausted
        assert RetryExhausted is not None

    def test_exports_from_resilience_package(self):
        """RetryPolicy and RetryExhausted should be exported from resilience package."""
        from backend.core.resilience import RetryPolicy, RetryExhausted
        assert RetryPolicy is not None
        assert RetryExhausted is not None
