"""
Comprehensive tests for CircuitBreaker with state transitions.

Tests cover:
- Starts in CLOSED state
- Passes through when CLOSED
- Opens after threshold failures
- Rejects with CircuitOpen when OPEN
- Transitions to HALF_OPEN after recovery_timeout
- HALF_OPEN success closes circuit
- HALF_OPEN failure reopens circuit
- on_open/on_close callbacks called appropriately
- reset() manually closes circuit
- Double start/stop is safe
"""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest
import time

from backend.core.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpen,
)
from backend.core.resilience.types import CircuitState


class TestCircuitOpen:
    """Tests for CircuitOpen exception."""

    def test_exception_inherits_from_exception(self):
        """CircuitOpen should inherit from Exception."""
        assert issubclass(CircuitOpen, Exception)

    def test_exception_message_preserved(self):
        """CircuitOpen should preserve the error message."""
        exc = CircuitOpen("Circuit is open")
        assert "Circuit is open" in str(exc)

    def test_exception_with_custom_message(self):
        """CircuitOpen should work with custom messages."""
        exc = CircuitOpen("Service unavailable: database circuit open")
        assert "database circuit open" in str(exc)


class TestCircuitBreakerDefaults:
    """Tests for CircuitBreaker default values."""

    def test_default_failure_threshold(self):
        """Default failure_threshold should be 5."""
        breaker = CircuitBreaker()
        assert breaker.failure_threshold == 5

    def test_default_recovery_timeout(self):
        """Default recovery_timeout should be 30.0."""
        breaker = CircuitBreaker()
        assert breaker.recovery_timeout == 30.0

    def test_default_on_open(self):
        """Default on_open should be None."""
        breaker = CircuitBreaker()
        assert breaker.on_open is None

    def test_default_on_close(self):
        """Default on_close should be None."""
        breaker = CircuitBreaker()
        assert breaker.on_close is None


class TestCircuitBreakerInitialState:
    """Tests for CircuitBreaker initial state."""

    def test_starts_in_closed_state(self):
        """CircuitBreaker should start in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    def test_initial_failure_count_is_zero(self):
        """Initial failure_count should be 0."""
        breaker = CircuitBreaker()
        assert breaker.failure_count == 0


class TestCircuitBreakerClosedState:
    """Tests for CircuitBreaker in CLOSED state."""

    @pytest.mark.asyncio
    async def test_passes_through_when_closed(self):
        """Calls should pass through when circuit is CLOSED."""
        breaker = CircuitBreaker()
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert call_count == 1
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self):
        """call should pass *args and **kwargs to the function."""
        breaker = CircuitBreaker()

        async def func_with_args(a, b, c=None):
            return (a, b, c)

        result = await breaker.call(func_with_args, 1, 2, c=3)
        assert result == (1, 2, 3)

    @pytest.mark.asyncio
    async def test_increments_failure_count_on_error(self):
        """Failure count should increment when function raises."""
        breaker = CircuitBreaker(failure_threshold=5)

        async def fails():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await breaker.call(fails)

        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_resets_failure_count_on_success(self):
        """Failure count should reset to 0 on successful call."""
        breaker = CircuitBreaker(failure_threshold=5)

        async def fails():
            raise ValueError("fail")

        async def succeeds():
            return "success"

        # Accumulate some failures
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.failure_count == 3

        # Successful call resets counter
        await breaker.call(succeeds)
        assert breaker.failure_count == 0


class TestCircuitBreakerOpenState:
    """Tests for CircuitBreaker opening on threshold failures."""

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        """Circuit should open after failure_threshold consecutive failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def fails():
            raise ValueError("fail")

        # Fail 3 times to reach threshold
        for i in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_rejects_with_circuit_open_when_open(self):
        """Calls should be rejected with CircuitOpen when circuit is OPEN."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def fails():
            raise ValueError("fail")

        async def succeeds():
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.state == CircuitState.OPEN

        # Now calls should be rejected
        with pytest.raises(CircuitOpen):
            await breaker.call(succeeds)

    @pytest.mark.asyncio
    async def test_on_open_callback_called(self):
        """on_open callback should be called when circuit opens."""
        callback = AsyncMock()
        breaker = CircuitBreaker(failure_threshold=3, on_open=callback)

        async def fails():
            raise ValueError("fail")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_open_not_called_before_threshold(self):
        """on_open callback should not be called before threshold is reached."""
        callback = AsyncMock()
        breaker = CircuitBreaker(failure_threshold=3, on_open=callback)

        async def fails():
            raise ValueError("fail")

        # Fail twice (below threshold)
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        callback.assert_not_called()
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerHalfOpenState:
    """Tests for CircuitBreaker HALF_OPEN state transitions."""

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_recovery_timeout(self):
        """Circuit should transition to HALF_OPEN after recovery_timeout."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        async def fails():
            raise ValueError("fail")

        async def succeeds():
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should allow one request through (HALF_OPEN)
        result = await breaker.call(succeeds)
        assert result == "success"
        # After success, circuit should be CLOSED
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self):
        """Successful call in HALF_OPEN state should close circuit."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        callback = AsyncMock()
        breaker.on_close = callback

        async def fails():
            raise ValueError("fail")

        async def succeeds():
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.1)

        # Successful call should close circuit
        result = await breaker.call(succeeds)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        """Failed call in HALF_OPEN state should reopen circuit."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        on_open = AsyncMock()
        breaker.on_open = on_open

        async def fails():
            raise ValueError("fail")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.state == CircuitState.OPEN
        on_open.assert_called_once()
        on_open.reset_mock()

        # Wait for recovery timeout
        await asyncio.sleep(0.1)

        # Failed call should reopen circuit
        with pytest.raises(ValueError):
            await breaker.call(fails)

        assert breaker.state == CircuitState.OPEN
        on_open.assert_called_once()

    @pytest.mark.asyncio
    async def test_stays_open_before_recovery_timeout(self):
        """Circuit should stay OPEN before recovery_timeout elapses."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)

        async def fails():
            raise ValueError("fail")

        async def succeeds():
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.state == CircuitState.OPEN

        # Without waiting, should still reject
        with pytest.raises(CircuitOpen):
            await breaker.call(succeeds)


class TestCircuitBreakerReset:
    """Tests for CircuitBreaker reset() method."""

    @pytest.mark.asyncio
    async def test_reset_closes_circuit_from_open(self):
        """reset() should close circuit from OPEN state."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def fails():
            raise ValueError("fail")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.state == CircuitState.OPEN

        # Reset should close it
        await breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_reset_calls_on_close_callback(self):
        """reset() should call on_close callback."""
        callback = AsyncMock()
        breaker = CircuitBreaker(failure_threshold=2, on_close=callback)

        async def fails():
            raise ValueError("fail")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        # Reset should call on_close
        await breaker.reset()
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_is_idempotent(self):
        """Multiple reset() calls should be safe."""
        callback = AsyncMock()
        breaker = CircuitBreaker(failure_threshold=2, on_close=callback)

        async def fails():
            raise ValueError("fail")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        # Multiple resets should be safe
        await breaker.reset()
        await breaker.reset()
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        # on_close should only be called once (first reset)
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_reset_from_closed_is_safe(self):
        """reset() from CLOSED state should be safe and not call on_close."""
        callback = AsyncMock()
        breaker = CircuitBreaker(on_close=callback)

        # Already closed
        assert breaker.state == CircuitState.CLOSED

        # Reset should be a no-op
        await breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        callback.assert_not_called()


class TestCircuitBreakerThreadSafety:
    """Tests for thread safety with concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_safe(self):
        """Concurrent calls should be thread-safe."""
        breaker = CircuitBreaker(failure_threshold=100)  # High threshold
        call_count = 0

        async def increment():
            nonlocal call_count
            call_count += 1
            return "success"

        # Run many concurrent calls
        tasks = [breaker.call(increment) for _ in range(50)]
        results = await asyncio.gather(*tasks)

        assert all(r == "success" for r in results)
        assert call_count == 50

    @pytest.mark.asyncio
    async def test_concurrent_failures_open_circuit_once(self):
        """Concurrent failures should open circuit only once."""
        on_open = AsyncMock()
        breaker = CircuitBreaker(failure_threshold=5, on_open=on_open)

        async def fails():
            raise ValueError("fail")

        # Run concurrent failures
        tasks = [breaker.call(fails) for _ in range(10)]

        # Gather with return_exceptions to avoid propagation
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some should be ValueError, some should be CircuitOpen
        value_errors = sum(1 for r in results if isinstance(r, ValueError))
        circuit_opens = sum(1 for r in results if isinstance(r, CircuitOpen))

        # At least threshold should be ValueErrors, rest should be CircuitOpen
        assert value_errors >= 5
        assert circuit_opens + value_errors == 10
        assert breaker.state == CircuitState.OPEN
        # on_open should be called exactly once
        on_open.assert_called_once()


class TestCircuitBreakerCallbacks:
    """Tests for callback behavior."""

    @pytest.mark.asyncio
    async def test_callbacks_are_optional(self):
        """Circuit breaker should work without callbacks."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def fails():
            raise ValueError("fail")

        async def succeeds():
            return "success"

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.state == CircuitState.OPEN

        # Reset without on_close
        await breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_callback_exception_propagates(self):
        """Exception in callback should propagate."""
        async def bad_callback():
            raise RuntimeError("callback error")

        breaker = CircuitBreaker(failure_threshold=2, on_open=bad_callback)

        async def fails():
            raise ValueError("fail")

        # First failure
        with pytest.raises(ValueError):
            await breaker.call(fails)

        # Second failure triggers on_open which raises
        with pytest.raises(RuntimeError):
            await breaker.call(fails)


class TestCircuitBreakerProperties:
    """Tests for property access."""

    def test_state_property_returns_circuit_state(self):
        """state property should return CircuitState enum."""
        breaker = CircuitBreaker()
        assert isinstance(breaker.state, CircuitState)
        assert breaker.state == CircuitState.CLOSED

    def test_failure_count_property_returns_int(self):
        """failure_count property should return integer."""
        breaker = CircuitBreaker()
        assert isinstance(breaker.failure_count, int)
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_properties_reflect_current_state(self):
        """Properties should reflect current internal state."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def fails():
            raise ValueError("fail")

        # Accumulate failures
        for i in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fails)

        assert breaker.failure_count == 2
        assert breaker.state == CircuitState.CLOSED

        # One more to open
        with pytest.raises(ValueError):
            await breaker.call(fails)

        assert breaker.failure_count == 3
        assert breaker.state == CircuitState.OPEN


class TestModuleExports:
    """Tests for module exports and structure."""

    def test_circuit_breaker_importable(self):
        """CircuitBreaker should be importable from circuit_breaker module."""
        from backend.core.resilience.circuit_breaker import CircuitBreaker
        assert CircuitBreaker is not None

    def test_circuit_open_importable(self):
        """CircuitOpen should be importable from circuit_breaker module."""
        from backend.core.resilience.circuit_breaker import CircuitOpen
        assert CircuitOpen is not None

    def test_exports_from_resilience_package(self):
        """CircuitBreaker and CircuitOpen should be exported from resilience package."""
        from backend.core.resilience import (
            PrimitivesCircuitBreaker,
            PrimitivesCircuitOpen,
        )
        assert PrimitivesCircuitBreaker is not None
        assert PrimitivesCircuitOpen is not None

    def test_primitives_circuit_state_exported(self):
        """PrimitivesCircuitState should be exported from resilience package."""
        from backend.core.resilience import PrimitivesCircuitState
        assert PrimitivesCircuitState is not None
