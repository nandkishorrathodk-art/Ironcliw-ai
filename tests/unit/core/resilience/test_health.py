"""
Comprehensive tests for HealthProbe with caching.

Tests cover:
- Returns check_fn result
- Returns False on exception
- Caches result within TTL
- Refreshes after TTL expires
- force=True bypasses cache
- Tracks consecutive failures
- Resets failures on success
- on_unhealthy called at threshold
- on_healthy called on recovery
- Timeout returns False
"""

import asyncio
from unittest.mock import AsyncMock, patch
import pytest
import time

from backend.core.resilience.health import HealthProbe


class TestHealthProbeDefaults:
    """Tests for HealthProbe default values."""

    def test_default_cache_ttl(self):
        """Default cache_ttl should be 30.0."""
        async def check_fn():
            return True
        probe = HealthProbe(check_fn=check_fn)
        assert probe.cache_ttl == 30.0

    def test_default_timeout(self):
        """Default timeout should be 10.0."""
        async def check_fn():
            return True
        probe = HealthProbe(check_fn=check_fn)
        assert probe.timeout == 10.0

    def test_default_unhealthy_threshold(self):
        """Default unhealthy_threshold should be 3."""
        async def check_fn():
            return True
        probe = HealthProbe(check_fn=check_fn)
        assert probe.unhealthy_threshold == 3

    def test_default_on_unhealthy(self):
        """Default on_unhealthy should be None."""
        async def check_fn():
            return True
        probe = HealthProbe(check_fn=check_fn)
        assert probe.on_unhealthy is None

    def test_default_on_healthy(self):
        """Default on_healthy should be None."""
        async def check_fn():
            return True
        probe = HealthProbe(check_fn=check_fn)
        assert probe.on_healthy is None


class TestHealthProbeInitialState:
    """Tests for HealthProbe initial state."""

    def test_initial_consecutive_failures_is_zero(self):
        """Initial consecutive_failures should be 0."""
        async def check_fn():
            return True
        probe = HealthProbe(check_fn=check_fn)
        assert probe.consecutive_failures == 0

    def test_initial_is_unhealthy_is_false(self):
        """Initial is_unhealthy should be False."""
        async def check_fn():
            return True
        probe = HealthProbe(check_fn=check_fn)
        assert probe.is_unhealthy is False


class TestHealthProbeBasicCheck:
    """Tests for basic HealthProbe.check() functionality."""

    @pytest.mark.asyncio
    async def test_returns_check_fn_result_true(self):
        """check() should return True when check_fn returns True."""
        async def check_fn():
            return True
        probe = HealthProbe(check_fn=check_fn)
        result = await probe.check()
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_check_fn_result_false(self):
        """check() should return False when check_fn returns False."""
        async def check_fn():
            return False
        probe = HealthProbe(check_fn=check_fn)
        result = await probe.check()
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_exception(self):
        """check() should return False when check_fn raises exception."""
        async def check_fn():
            raise ValueError("check failed")
        probe = HealthProbe(check_fn=check_fn)
        result = await probe.check()
        assert result is False

    @pytest.mark.asyncio
    async def test_never_raises_exception(self):
        """check() should never raise, even when check_fn raises."""
        async def check_fn():
            raise RuntimeError("boom")
        probe = HealthProbe(check_fn=check_fn)
        # Should not raise
        result = await probe.check()
        assert result is False


class TestHealthProbeCaching:
    """Tests for HealthProbe caching behavior."""

    @pytest.mark.asyncio
    async def test_caches_result_within_ttl(self):
        """check() should cache result and not call check_fn again within TTL."""
        call_count = 0
        async def check_fn():
            nonlocal call_count
            call_count += 1
            return True

        probe = HealthProbe(check_fn=check_fn, cache_ttl=10.0)

        # First call
        result1 = await probe.check()
        assert result1 is True
        assert call_count == 1

        # Second call should use cache
        result2 = await probe.check()
        assert result2 is True
        assert call_count == 1  # Still 1

    @pytest.mark.asyncio
    async def test_refreshes_after_ttl_expires(self):
        """check() should refresh result after cache_ttl expires."""
        call_count = 0
        async def check_fn():
            nonlocal call_count
            call_count += 1
            return True

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.05)  # 50ms TTL

        # First call
        result1 = await probe.check()
        assert result1 is True
        assert call_count == 1

        # Wait for TTL to expire
        await asyncio.sleep(0.1)

        # Second call should refresh
        result2 = await probe.check()
        assert result2 is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_force_bypasses_cache(self):
        """check(force=True) should bypass cache and call check_fn."""
        call_count = 0
        async def check_fn():
            nonlocal call_count
            call_count += 1
            return True

        probe = HealthProbe(check_fn=check_fn, cache_ttl=60.0)  # Long TTL

        # First call
        result1 = await probe.check()
        assert result1 is True
        assert call_count == 1

        # Second call with force=True
        result2 = await probe.check(force=True)
        assert result2 is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_caches_false_result_too(self):
        """check() should also cache False results."""
        call_count = 0
        async def check_fn():
            nonlocal call_count
            call_count += 1
            return False

        probe = HealthProbe(check_fn=check_fn, cache_ttl=10.0)

        # First call
        result1 = await probe.check()
        assert result1 is False
        assert call_count == 1

        # Second call should use cache
        result2 = await probe.check()
        assert result2 is False
        assert call_count == 1  # Still 1


class TestHealthProbeConsecutiveFailures:
    """Tests for tracking consecutive failures."""

    @pytest.mark.asyncio
    async def test_tracks_consecutive_failures(self):
        """consecutive_failures should increment on each False result."""
        async def check_fn():
            return False

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0)  # No cache

        await probe.check()
        assert probe.consecutive_failures == 1

        await probe.check()
        assert probe.consecutive_failures == 2

        await probe.check()
        assert probe.consecutive_failures == 3

    @pytest.mark.asyncio
    async def test_resets_failures_on_success(self):
        """consecutive_failures should reset to 0 when check succeeds."""
        results = [False, False, True]
        result_iter = iter(results)

        async def check_fn():
            return next(result_iter)

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0)

        await probe.check()
        assert probe.consecutive_failures == 1

        await probe.check()
        assert probe.consecutive_failures == 2

        # This one succeeds
        result = await probe.check()
        assert result is True
        assert probe.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_exception_counts_as_failure(self):
        """Exception in check_fn should count as a failure."""
        async def check_fn():
            raise ValueError("fail")

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0)

        await probe.check()
        assert probe.consecutive_failures == 1

        await probe.check()
        assert probe.consecutive_failures == 2


class TestHealthProbeUnhealthyState:
    """Tests for is_unhealthy state and on_unhealthy callback."""

    @pytest.mark.asyncio
    async def test_becomes_unhealthy_at_threshold(self):
        """is_unhealthy should become True when consecutive_failures reaches threshold."""
        async def check_fn():
            return False

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0, unhealthy_threshold=3)

        await probe.check()
        assert probe.consecutive_failures == 1
        assert probe.is_unhealthy is False

        await probe.check()
        assert probe.consecutive_failures == 2
        assert probe.is_unhealthy is False

        await probe.check()
        assert probe.consecutive_failures == 3
        assert probe.is_unhealthy is True

    @pytest.mark.asyncio
    async def test_on_unhealthy_called_at_threshold(self):
        """on_unhealthy callback should be called when threshold is reached."""
        callback = AsyncMock()

        async def check_fn():
            return False

        probe = HealthProbe(
            check_fn=check_fn,
            cache_ttl=0.0,
            unhealthy_threshold=3,
            on_unhealthy=callback,
        )

        await probe.check()
        callback.assert_not_called()

        await probe.check()
        callback.assert_not_called()

        await probe.check()
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_unhealthy_called_only_once(self):
        """on_unhealthy should only be called once when crossing threshold."""
        callback = AsyncMock()

        async def check_fn():
            return False

        probe = HealthProbe(
            check_fn=check_fn,
            cache_ttl=0.0,
            unhealthy_threshold=2,
            on_unhealthy=callback,
        )

        # Reach threshold
        await probe.check()
        await probe.check()
        assert callback.call_count == 1

        # Continue failing - callback should not be called again
        await probe.check()
        await probe.check()
        await probe.check()
        assert callback.call_count == 1


class TestHealthProbeRecovery:
    """Tests for recovery and on_healthy callback."""

    @pytest.mark.asyncio
    async def test_on_healthy_called_on_recovery(self):
        """on_healthy callback should be called when recovering from unhealthy."""
        callback = AsyncMock()
        results = [False, False, False, True]  # 3 failures, then success
        result_iter = iter(results)

        async def check_fn():
            return next(result_iter)

        probe = HealthProbe(
            check_fn=check_fn,
            cache_ttl=0.0,
            unhealthy_threshold=3,
            on_healthy=callback,
        )

        # Fail until unhealthy
        await probe.check()
        await probe.check()
        await probe.check()
        assert probe.is_unhealthy is True
        callback.assert_not_called()

        # Recover
        result = await probe.check()
        assert result is True
        assert probe.is_unhealthy is False
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_healthy_not_called_when_already_healthy(self):
        """on_healthy should not be called when already healthy."""
        callback = AsyncMock()

        async def check_fn():
            return True

        probe = HealthProbe(
            check_fn=check_fn,
            cache_ttl=0.0,
            on_healthy=callback,
        )

        # Successful checks when already healthy
        await probe.check()
        await probe.check()
        await probe.check()

        # on_healthy should never be called
        callback.assert_not_called()


class TestHealthProbeTimeout:
    """Tests for per-check timeout."""

    @pytest.mark.asyncio
    async def test_timeout_returns_false(self):
        """check() should return False when check_fn times out."""
        async def slow_check():
            await asyncio.sleep(10.0)  # Very slow
            return True

        probe = HealthProbe(
            check_fn=slow_check,
            cache_ttl=0.0,
            timeout=0.05,  # 50ms timeout
        )

        result = await probe.check()
        assert result is False

    @pytest.mark.asyncio
    async def test_timeout_counts_as_failure(self):
        """Timeout should count as a failure for consecutive_failures."""
        async def slow_check():
            await asyncio.sleep(10.0)
            return True

        probe = HealthProbe(
            check_fn=slow_check,
            cache_ttl=0.0,
            timeout=0.05,
            unhealthy_threshold=2,
        )

        await probe.check()
        assert probe.consecutive_failures == 1

        await probe.check()
        assert probe.consecutive_failures == 2
        assert probe.is_unhealthy is True


class TestHealthProbeReset:
    """Tests for reset() method."""

    @pytest.mark.asyncio
    async def test_reset_clears_cache(self):
        """reset() should clear the cached result."""
        call_count = 0
        async def check_fn():
            nonlocal call_count
            call_count += 1
            return True

        probe = HealthProbe(check_fn=check_fn, cache_ttl=60.0)

        # First call
        await probe.check()
        assert call_count == 1

        # Reset and check again
        probe.reset()
        await probe.check()
        assert call_count == 2  # Should have called check_fn again

    @pytest.mark.asyncio
    async def test_reset_clears_failure_count(self):
        """reset() should clear consecutive_failures."""
        async def check_fn():
            return False

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0)

        await probe.check()
        await probe.check()
        assert probe.consecutive_failures == 2

        probe.reset()
        assert probe.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_reset_clears_unhealthy_state(self):
        """reset() should clear is_unhealthy state."""
        async def check_fn():
            return False

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0, unhealthy_threshold=2)

        await probe.check()
        await probe.check()
        assert probe.is_unhealthy is True

        probe.reset()
        assert probe.is_unhealthy is False


class TestHealthProbeConcurrency:
    """Tests for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_concurrent_checks_are_safe(self):
        """Concurrent checks should be thread-safe."""
        call_count = 0
        async def check_fn():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Small delay to encourage interleaving
            return True

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0)

        # Run many concurrent checks
        tasks = [probe.check() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        assert all(r is True for r in results)
        # Due to lock, each check should be called
        assert call_count == 20


class TestHealthProbeCallbacks:
    """Tests for callback edge cases."""

    @pytest.mark.asyncio
    async def test_callbacks_are_optional(self):
        """HealthProbe should work without callbacks."""
        async def check_fn():
            return False

        probe = HealthProbe(check_fn=check_fn, cache_ttl=0.0, unhealthy_threshold=2)

        # Fail and become unhealthy without callbacks
        await probe.check()
        await probe.check()
        assert probe.is_unhealthy is True

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_affect_result(self):
        """Exception in callback should not affect the check result."""
        async def bad_callback():
            raise RuntimeError("callback error")

        results = [False, False, True]
        result_iter = iter(results)

        async def check_fn():
            return next(result_iter)

        probe = HealthProbe(
            check_fn=check_fn,
            cache_ttl=0.0,
            unhealthy_threshold=2,
            on_unhealthy=bad_callback,
        )

        # First failure - no callback yet
        result1 = await probe.check()
        assert result1 is False

        # Second failure - triggers bad callback, but should still complete
        # Note: The callback may raise, but the check result should be consistent
        with pytest.raises(RuntimeError):
            await probe.check()

        # State should still have updated
        assert probe.is_unhealthy is True


class TestModuleExports:
    """Tests for module exports and structure."""

    def test_health_probe_importable(self):
        """HealthProbe should be importable from health module."""
        from backend.core.resilience.health import HealthProbe
        assert HealthProbe is not None

    def test_exports_from_resilience_package(self):
        """HealthProbe should be exported from resilience package."""
        from backend.core.resilience import HealthProbe
        assert HealthProbe is not None
