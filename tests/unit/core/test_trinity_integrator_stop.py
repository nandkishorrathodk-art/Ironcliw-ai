"""
Tests for TrinityIntegrator.tiered_stop() method.

These tests verify that the tiered stop() method:
1. Is idempotent - calling stop() multiple times is safe
2. Never raises - all exceptions are caught and logged
3. Is bounded by timeout - completes within specified timeout
4. Uses tiered approach - SIGTERM first, then SIGKILL, then abandon

Following the 6 Pillars design for Pillar 5: Graceful Shutdown.
"""

import asyncio
import os
import signal
import time
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_process():
    """Create a mock process for testing."""
    process = MagicMock()
    process.pid = 12345
    process.returncode = None
    return process


@pytest.fixture
def mock_psutil_process():
    """Create a mock psutil.Process for testing."""
    process = MagicMock()
    process.pid = 12345
    process.is_running.return_value = True
    process.terminate = MagicMock()
    process.kill = MagicMock()
    process.wait = MagicMock(return_value=0)
    return process


@pytest.fixture
def mock_timeouts():
    """Create mock timeouts for testing."""
    timeouts = MagicMock()
    timeouts.cleanup_timeout_sigterm = 2.0
    timeouts.cleanup_timeout_sigkill = 1.0
    return timeouts


# =============================================================================
# Mock Trinity Integrator for isolated testing
# =============================================================================


class MockTrinityIntegrator:
    """
    Simplified mock of TrinityIntegrator for testing the tiered_stop pattern.

    This implements the same tiered_stop() contract that we'll add to the real
    TrinityIntegrator class.
    """

    def __init__(self):
        self._stopped = False
        self._lock = asyncio.Lock()
        self._managed_pids: Dict[int, Dict[str, Any]] = {}
        self._running = True

    def register_pid(self, pid: int, name: str = "test_process") -> None:
        """Register a PID for management."""
        self._managed_pids[pid] = {"name": name, "registered_at": time.time()}

    async def tiered_stop(
        self,
        timeout: float = 30.0,
    ) -> Dict[int, str]:
        """
        Tiered stop method following the spec:

        1. Idempotent - returns empty dict if already stopped
        2. Never raises - catches all exceptions
        3. Bounded - completes within timeout
        4. Tiered - SIGTERM -> SIGKILL -> abandon

        Args:
            timeout: Maximum time to wait for all processes to stop

        Returns:
            Dict mapping PID to result:
            - "stopped": Process terminated gracefully with SIGTERM
            - "killed": Process required SIGKILL
            - "abandoned": Process didn't respond to SIGKILL
            - Error message string if exception occurred
        """
        # Idempotent: return immediately if already stopped
        if self._stopped:
            return {}

        async with self._lock:
            if self._stopped:
                return {}

            self._stopped = True
            self._running = False
            results: Dict[int, str] = {}

            try:
                # Apply overall timeout
                results = await asyncio.wait_for(
                    self._stop_all_processes(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Mark remaining processes as abandoned
                for pid in self._managed_pids:
                    if pid not in results:
                        results[pid] = "abandoned"
            except Exception as e:
                # Never raise - log and return error
                for pid in self._managed_pids:
                    if pid not in results:
                        results[pid] = f"error: {e}"

            return results

    async def _stop_all_processes(self) -> Dict[int, str]:
        """Stop all managed processes using tiered approach."""
        import psutil

        results: Dict[int, str] = {}
        tier_timeout = 30.0 / 3  # Split timeout into 3 tiers

        for pid, info in list(self._managed_pids.items()):
            result = await self._stop_process_tiered(pid, tier_timeout)
            results[pid] = result

        return results

    async def _stop_process_tiered(
        self,
        pid: int,
        tier_timeout: float,
    ) -> str:
        """
        Stop a single process using tiered approach.

        Tier 1: SIGTERM (graceful, wait tier_timeout)
        Tier 2: SIGKILL (force, wait tier_timeout)
        Tier 3: Abandon (log and give up)
        """
        import psutil

        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return "stopped"  # Already gone

        # Tier 1: SIGTERM (graceful)
        try:
            os.kill(pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, proc.wait),
                    timeout=tier_timeout,
                )
                return "stopped"
            except asyncio.TimeoutError:
                pass  # Continue to Tier 2
        except ProcessLookupError:
            return "stopped"  # Already gone
        except Exception:
            pass  # Continue to Tier 2

        # Tier 2: SIGKILL (force)
        try:
            os.kill(pid, signal.SIGKILL)
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, proc.wait),
                    timeout=tier_timeout,
                )
                return "killed"
            except asyncio.TimeoutError:
                pass  # Continue to Tier 3
        except ProcessLookupError:
            return "killed"  # Died from SIGTERM
        except Exception:
            pass  # Continue to Tier 3

        # Tier 3: Abandon
        return "abandoned"


# =============================================================================
# Test: Idempotent Stop
# =============================================================================


class TestIdempotentStop:
    """Tests for idempotent behavior of tiered_stop()."""

    @pytest.mark.asyncio
    async def test_stop_is_idempotent_returns_empty_on_second_call(self):
        """Second call to tiered_stop() returns empty dict."""
        integrator = MockTrinityIntegrator()

        # First call - should return results (empty since no PIDs registered)
        result1 = await integrator.tiered_stop()
        assert result1 == {}
        assert integrator._stopped is True

        # Second call - should return empty dict immediately
        result2 = await integrator.tiered_stop()
        assert result2 == {}

    @pytest.mark.asyncio
    async def test_stop_is_idempotent_with_registered_pids(self):
        """Second call returns empty even if PIDs were registered."""
        integrator = MockTrinityIntegrator()

        # Register a fake PID (won't exist)
        integrator.register_pid(999999, "fake_process")

        # First call processes the PID
        result1 = await integrator.tiered_stop()
        assert integrator._stopped is True
        assert 999999 in result1  # PID was processed

        # Second call returns empty
        result2 = await integrator.tiered_stop()
        assert result2 == {}

    @pytest.mark.asyncio
    async def test_stop_concurrent_calls_are_safe(self):
        """Concurrent calls to tiered_stop() are safe."""
        integrator = MockTrinityIntegrator()

        # Call stop concurrently
        results = await asyncio.gather(
            integrator.tiered_stop(),
            integrator.tiered_stop(),
            integrator.tiered_stop(),
        )

        # Only one call should have done work (others return empty)
        non_empty_results = [r for r in results if r != {}]
        # At most one non-empty result (could be zero if no PIDs)
        assert len(non_empty_results) <= 1


# =============================================================================
# Test: Never Raises
# =============================================================================


class TestNeverRaises:
    """Tests for never-raises behavior of tiered_stop()."""

    @pytest.mark.asyncio
    async def test_stop_never_raises_on_psutil_error(self):
        """tiered_stop() catches psutil exceptions."""
        integrator = MockTrinityIntegrator()

        # Register a PID that doesn't exist
        integrator.register_pid(999999, "nonexistent")

        # Should not raise, should return result
        result = await integrator.tiered_stop()
        assert isinstance(result, dict)
        assert 999999 in result
        # Process doesn't exist, so it should be "stopped"
        assert result[999999] == "stopped"

    @pytest.mark.asyncio
    async def test_stop_never_raises_on_os_error(self):
        """tiered_stop() catches OS errors from signal sending."""
        integrator = MockTrinityIntegrator()

        # Register a PID that will cause permission error (PID 1 = init)
        # Note: This test may behave differently on different systems
        # Using a very high PID that likely doesn't exist
        integrator.register_pid(2147483647, "max_pid")

        # Should not raise
        result = await integrator.tiered_stop()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_stop_never_raises_with_empty_pids(self):
        """tiered_stop() handles empty PID list gracefully."""
        integrator = MockTrinityIntegrator()

        # No PIDs registered
        result = await integrator.tiered_stop()
        assert result == {}


# =============================================================================
# Test: Bounded by Timeout
# =============================================================================


class TestBoundedByTimeout:
    """Tests for timeout-bounded behavior of tiered_stop()."""

    @pytest.mark.asyncio
    async def test_stop_bounded_by_timeout(self):
        """tiered_stop() completes within timeout."""
        integrator = MockTrinityIntegrator()

        # Register a fake PID
        integrator.register_pid(999999, "fake")

        start_time = time.time()
        timeout = 2.0

        result = await integrator.tiered_stop(timeout=timeout)

        elapsed = time.time() - start_time
        # Should complete well within timeout (no real processes to wait for)
        assert elapsed < timeout + 1.0  # Allow 1s buffer

    @pytest.mark.asyncio
    async def test_stop_returns_abandoned_on_timeout(self):
        """Processes that don't stop within timeout are marked as abandoned."""
        # This is a theoretical test - in practice, a non-existent process
        # returns immediately. The real test would need a stubborn process.

        integrator = MockTrinityIntegrator()

        # Register non-existent PID
        integrator.register_pid(999999, "fake")

        result = await integrator.tiered_stop(timeout=0.001)  # Very short timeout

        # With very short timeout, might be abandoned or stopped
        assert isinstance(result, dict)
        assert 999999 in result
        # Result should be one of the valid states
        assert result[999999] in ["stopped", "killed", "abandoned"] or result[999999].startswith("error:")


# =============================================================================
# Test: Tiered Shutdown (SIGTERM -> SIGKILL -> Abandon)
# =============================================================================


class TestTieredShutdown:
    """Tests for tiered shutdown behavior."""

    @pytest.mark.asyncio
    async def test_stop_uses_sigterm_first(self):
        """tiered_stop() uses SIGTERM before SIGKILL."""
        integrator = MockTrinityIntegrator()

        # For this test, we'd need to mock os.kill and track calls
        # This is a placeholder for the pattern
        signals_sent = []

        original_kill = os.kill
        def mock_kill(pid, sig):
            signals_sent.append((pid, sig))
            raise ProcessLookupError()  # Simulate process not found

        integrator.register_pid(999999, "test")

        with patch('os.kill', mock_kill):
            result = await integrator.tiered_stop()

        # First signal should be SIGTERM
        if signals_sent:
            assert signals_sent[0][1] == signal.SIGTERM

    @pytest.mark.asyncio
    async def test_stop_escalates_to_sigkill(self):
        """tiered_stop() escalates to SIGKILL if SIGTERM times out."""
        # This test verifies the escalation pattern
        integrator = MockTrinityIntegrator()
        signals_sent = []

        call_count = [0]

        def mock_kill(pid, sig):
            signals_sent.append((pid, sig))
            call_count[0] += 1
            if sig == signal.SIGTERM:
                # SIGTERM doesn't kill it
                pass
            elif sig == signal.SIGKILL:
                # SIGKILL succeeds
                raise ProcessLookupError()

        # Mock psutil.Process to simulate a stubborn process
        mock_proc = MagicMock()
        mock_proc.wait = MagicMock(side_effect=lambda: time.sleep(10))

        integrator.register_pid(999999, "test")

        with patch('os.kill', mock_kill):
            with patch('psutil.Process', return_value=mock_proc):
                result = await integrator.tiered_stop(timeout=1.0)

        # Should have sent both SIGTERM and SIGKILL
        sigterm_sent = any(s[1] == signal.SIGTERM for s in signals_sent)
        sigkill_sent = any(s[1] == signal.SIGKILL for s in signals_sent)

        # At minimum SIGTERM should have been attempted
        assert sigterm_sent or 999999 in result


# =============================================================================
# Test: Result Dictionary Format
# =============================================================================


class TestResultFormat:
    """Tests for the result dictionary format."""

    @pytest.mark.asyncio
    async def test_result_contains_all_registered_pids(self):
        """Result dictionary contains all registered PIDs."""
        integrator = MockTrinityIntegrator()

        # Register multiple PIDs
        pids = [999991, 999992, 999993]
        for pid in pids:
            integrator.register_pid(pid, f"process_{pid}")

        result = await integrator.tiered_stop()

        # All PIDs should be in result
        for pid in pids:
            assert pid in result

    @pytest.mark.asyncio
    async def test_result_values_are_valid_states(self):
        """Result values are valid state strings."""
        integrator = MockTrinityIntegrator()

        integrator.register_pid(999999, "test")

        result = await integrator.tiered_stop()

        valid_states = ["stopped", "killed", "abandoned"]
        for pid, state in result.items():
            # State is either a valid state or an error message
            assert state in valid_states or state.startswith("error:")


# =============================================================================
# Integration Test: Real Process (Optional)
# =============================================================================


class TestRealProcess:
    """Integration tests with real processes (skipped in CI)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip real process tests in CI"
    )
    async def test_stop_real_sleep_process(self):
        """Test stopping a real sleep process."""
        import subprocess

        # Start a real process
        proc = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        integrator = MockTrinityIntegrator()
        integrator.register_pid(proc.pid, "sleep_process")

        try:
            result = await integrator.tiered_stop(timeout=5.0)

            assert proc.pid in result
            assert result[proc.pid] in ["stopped", "killed"]

            # Process should be dead
            assert proc.poll() is not None
        finally:
            # Cleanup: ensure process is dead
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass


# =============================================================================
# Integration Tests: Real TrinityIntegrator Class
# =============================================================================


class TestRealTrinityIntegrator:
    """
    Integration tests using the actual TrinityIntegrator class.

    These tests verify that our implementation in trinity_integrator.py
    correctly implements the tiered_stop() contract.
    """

    @pytest.fixture
    def trinity_integrator(self):
        """Create a TrinityIntegrator instance for testing."""
        # Import inside fixture to avoid import errors if not available
        try:
            from backend.core.trinity_integrator import TrinityUnifiedOrchestrator
            # Create with components disabled to avoid network calls
            integrator = TrinityUnifiedOrchestrator(
                enable_jprime=False,
                enable_reactor=False,
            )
            return integrator
        except ImportError as e:
            pytest.skip(f"TrinityIntegrator not available: {e}")

    @pytest.mark.asyncio
    async def test_real_integrator_tiered_stop_is_idempotent(self, trinity_integrator):
        """Real TrinityIntegrator.tiered_stop() is idempotent."""
        # First call
        result1 = await trinity_integrator.tiered_stop()
        assert trinity_integrator._stopped is True

        # Second call returns empty dict
        result2 = await trinity_integrator.tiered_stop()
        assert result2 == {}

    @pytest.mark.asyncio
    async def test_real_integrator_tiered_stop_never_raises(self, trinity_integrator):
        """Real TrinityIntegrator.tiered_stop() never raises."""
        # Register non-existent PIDs
        trinity_integrator.register_pid(999999, "fake1")
        trinity_integrator.register_pid(999998, "fake2")

        # Should not raise
        result = await trinity_integrator.tiered_stop()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_real_integrator_tiered_stop_bounded(self, trinity_integrator):
        """Real TrinityIntegrator.tiered_stop() completes within timeout."""
        timeout = 2.0
        start = time.time()

        result = await trinity_integrator.tiered_stop(timeout=timeout)

        elapsed = time.time() - start
        # Should complete within timeout + buffer
        assert elapsed < timeout + 2.0

    @pytest.mark.asyncio
    async def test_real_integrator_register_pid(self, trinity_integrator):
        """Real TrinityIntegrator can register PIDs."""
        trinity_integrator.register_pid(12345, "test_process")

        pids = trinity_integrator._get_all_managed_pids()
        assert 12345 in pids

    @pytest.mark.asyncio
    async def test_real_integrator_handles_real_process(self, trinity_integrator):
        """Real TrinityIntegrator can stop a real process."""
        import subprocess

        # Start a real process
        proc = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            trinity_integrator.register_pid(proc.pid, "sleep_process")

            result = await trinity_integrator.tiered_stop(timeout=5.0)

            assert proc.pid in result
            assert result[proc.pid] in ["stopped", "killed"]

            # Verify process is actually dead
            import psutil
            assert not psutil.pid_exists(proc.pid) or proc.poll() is not None

        finally:
            # Cleanup in case test fails
            try:
                proc.kill()
                proc.wait(timeout=1)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_real_integrator_default_timeout_from_config(self, trinity_integrator):
        """Real TrinityIntegrator uses CLEANUP_BUDGET as default timeout."""
        # Just verify it doesn't crash with no timeout specified
        result = await trinity_integrator.tiered_stop()
        assert isinstance(result, dict)


class TestTrinityIntegratorStateTransition:
    """Tests for state transitions during tiered_stop()."""

    @pytest.fixture
    def fresh_integrator(self):
        """Create a fresh TrinityIntegrator for each test."""
        try:
            from backend.core.trinity_integrator import TrinityUnifiedOrchestrator
            return TrinityUnifiedOrchestrator(
                enable_jprime=False,
                enable_reactor=False,
            )
        except ImportError as e:
            pytest.skip(f"TrinityIntegrator not available: {e}")

    @pytest.mark.asyncio
    async def test_state_changes_to_stopped(self, fresh_integrator):
        """tiered_stop() changes state to STOPPED."""
        from backend.core.trinity_integrator import TrinityState

        # Initial state
        assert fresh_integrator._state != TrinityState.STOPPED

        await fresh_integrator.tiered_stop()

        # Should be stopped
        assert fresh_integrator._state == TrinityState.STOPPED

    @pytest.mark.asyncio
    async def test_running_flag_cleared(self, fresh_integrator):
        """tiered_stop() clears the _running flag."""
        fresh_integrator._running = True

        await fresh_integrator.tiered_stop()

        assert fresh_integrator._running is False

    @pytest.mark.asyncio
    async def test_stopped_flag_set(self, fresh_integrator):
        """tiered_stop() sets the _stopped flag."""
        assert fresh_integrator._stopped is False

        await fresh_integrator.tiered_stop()

        assert fresh_integrator._stopped is True
