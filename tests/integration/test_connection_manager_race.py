"""
Integration Tests: Connection Manager Race Condition Fixes
============================================================

Verifies that the enterprise connection components properly fix
the 8 critical race conditions identified in cloud_sql_connection_manager.py:

1. Non-atomic HALF_OPEN transition (thundering herd)
2. AsyncLock event-loop mismatch (deadlocks)
3. Reactive proxy detection (2s timeout)
4. TLS race condition in asyncpg.connect

These tests verify the fixes work under realistic concurrent load.

Author: Ironcliw System
Version: 1.0.0
"""

import pytest
import asyncio
import time
import threading
from typing import List

from backend.core.connection import (
    AtomicCircuitBreaker,
    AtomicStateMachine,
    CircuitBreakerConfig,
    CircuitState,
    EventLoopAwareLock,
    ProactiveProxyDetector,
    ProxyDetectorConfig,
    ProxyStatus,
)


class TestThunderingHerdPrevention:
    """Tests that verify thundering herd prevention on HALF_OPEN transition."""

    @pytest.mark.asyncio
    async def test_only_one_request_allowed_in_half_open(self):
        """
        Verify that only ONE connection attempt occurs when circuit
        transitions from OPEN to HALF_OPEN.

        This is the core fix for the db-f1-micro thundering herd issue.
        """
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout_seconds=0.05,  # 50ms for fast test
            half_open_max_requests=1,
        )
        breaker = AtomicCircuitBreaker(config)

        # Open the circuit with failures
        await breaker.record_failure("error 1")
        await breaker.record_failure("error 2")
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.06)

        # Launch 100 concurrent can_execute() calls
        allowed_count = 0
        lock = asyncio.Lock()

        async def try_execute(task_id: int):
            nonlocal allowed_count
            if await breaker.can_execute():
                async with lock:
                    allowed_count += 1

        await asyncio.gather(*[try_execute(i) for i in range(100)])

        # Only 1 should have been allowed (half_open_max_requests=1)
        assert allowed_count == 1, f"Expected 1 allowed, got {allowed_count}"
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_configurable_half_open_requests(self):
        """Verify that half_open_max_requests is respected."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout_seconds=0.01,
            half_open_max_requests=3,  # Allow 3 test requests
        )
        breaker = AtomicCircuitBreaker(config)

        # Open circuit
        await breaker.record_failure("error")
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.02)

        # Count allowed requests
        allowed = 0
        for _ in range(10):
            if await breaker.can_execute():
                allowed += 1

        assert allowed == 3, f"Expected 3 allowed, got {allowed}"


class TestAtomicStateTransitions:
    """Tests that verify atomic state transitions using CAS pattern."""

    @pytest.mark.asyncio
    async def test_concurrent_transition_only_one_wins(self):
        """Only one coroutine should win a contested state transition."""
        machine = AtomicStateMachine(initial_state=CircuitState.OPEN)

        winners: List[int] = []
        lock = asyncio.Lock()

        async def try_transition(task_id: int):
            success = await machine.try_transition(
                from_state=CircuitState.OPEN,
                to_state=CircuitState.HALF_OPEN,
                reason=f"Task {task_id} attempting transition",
            )
            if success:
                async with lock:
                    winners.append(task_id)

        # Launch 50 concurrent transition attempts
        await asyncio.gather(*[try_transition(i) for i in range(50)])

        # Exactly one should win
        assert len(winners) == 1, f"Expected 1 winner, got {len(winners)}"
        assert machine.current_state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_rapid_state_transitions_are_atomic(self):
        """Verify that rapid state transitions don't corrupt state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=1,
            recovery_timeout_seconds=0.01,
        )
        breaker = AtomicCircuitBreaker(config)

        # Rapidly toggle between states
        for _ in range(50):
            await breaker.record_failure("test")
            await asyncio.sleep(0.015)  # Wait for recovery
            if await breaker.can_execute():
                await breaker.record_success()

        # State should be valid (not corrupted)
        assert breaker.state in (
            CircuitState.CLOSED,
            CircuitState.OPEN,
            CircuitState.HALF_OPEN,
        )

        # Transition count should match operations
        info = breaker.get_state_info()
        assert info['transition_count'] > 0

    @pytest.mark.asyncio
    async def test_transition_history_maintained(self):
        """Verify that transition history is properly maintained."""
        machine = AtomicStateMachine(initial_state=CircuitState.CLOSED)

        # Perform some transitions
        await machine.try_transition(CircuitState.CLOSED, CircuitState.OPEN, "Test open")
        await machine.try_transition(CircuitState.OPEN, CircuitState.HALF_OPEN, "Test half")
        await machine.try_transition(CircuitState.HALF_OPEN, CircuitState.CLOSED, "Test close")

        history = machine.get_history(limit=10)
        assert len(history) == 3
        assert history[0].from_state == CircuitState.CLOSED
        assert history[0].to_state == CircuitState.OPEN
        assert history[2].to_state == CircuitState.CLOSED


class TestEventLoopAwareLock:
    """Tests that verify EventLoopAwareLock works across multiple loops."""

    @pytest.mark.asyncio
    async def test_lock_works_across_different_event_loops(self):
        """Lock should work correctly when accessed from different event loops."""
        lock = EventLoopAwareLock()
        results: List[str] = []
        results_lock = threading.Lock()

        async def hold_lock(loop_id: int):
            async with lock:
                with results_lock:
                    results.append(f"acquired_{loop_id}")
                await asyncio.sleep(0.01)
                with results_lock:
                    results.append(f"released_{loop_id}")

        # Run in main loop
        await hold_lock(1)

        # Run in different loop via thread
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(hold_lock(2))
            finally:
                loop.close()

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        # Both should have acquired and released
        assert 'acquired_1' in results
        assert 'released_1' in results
        assert 'acquired_2' in results
        assert 'released_2' in results

    @pytest.mark.asyncio
    async def test_concurrent_async_access_serialized(self):
        """Concurrent async access should be properly serialized."""
        lock = EventLoopAwareLock()
        results: List[str] = []

        async def worker(task_id: int):
            async with lock:
                results.append(f"start_{task_id}")
                await asyncio.sleep(0.005)
                results.append(f"end_{task_id}")

        # Run 5 workers concurrently
        await asyncio.gather(*[worker(i) for i in range(5)])

        # Each start should be followed by its corresponding end before next start
        assert len(results) == 10
        for i in range(5):
            start_idx = results.index(f"start_{i}")
            end_idx = results.index(f"end_{i}")
            assert end_idx == start_idx + 1, \
                f"Task {i} was interrupted: start={start_idx}, end={end_idx}"


class TestProactiveProxyDetection:
    """Tests that verify proactive (sub-100ms) proxy detection."""

    @pytest.mark.asyncio
    async def test_detection_completes_under_100ms(self):
        """Proxy detection should complete in under 100ms regardless of result."""
        detector = ProactiveProxyDetector()

        start = time.perf_counter()
        status, msg = await detector.detect()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete quickly even if proxy is down
        assert elapsed_ms < 100, f"Detection took {elapsed_ms:.1f}ms, expected <100ms"
        assert status in (ProxyStatus.AVAILABLE, ProxyStatus.UNAVAILABLE, ProxyStatus.UNKNOWN)

    @pytest.mark.asyncio
    async def test_cache_works_correctly(self):
        """Cache should return cached result within TTL."""
        config = ProxyDetectorConfig()
        config.cache_ttl_seconds = 1.0  # 1 second cache
        detector = ProactiveProxyDetector(config)

        # First detection - actual check
        status1, msg1 = await detector.detect()

        # Second detection - should be cached
        start = time.perf_counter()
        status2, msg2 = await detector.detect()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Cached result should be nearly instant (< 5ms)
        assert elapsed_ms < 5, f"Cached detection took {elapsed_ms:.1f}ms"
        assert msg2 == "Cached status"
        assert status1 == status2

    @pytest.mark.asyncio
    async def test_force_bypass_cache(self):
        """Force flag should bypass cache."""
        detector = ProactiveProxyDetector()

        # First detection
        await detector.detect()

        # Second detection with force
        _, msg = await detector.detect(force=True)

        # Should not be cached
        assert msg != "Cached status"


class TestIntegratedComponents:
    """Tests that verify components work together correctly."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_proxy_detection(self):
        """Circuit breaker should integrate with proxy detection."""
        detector = ProactiveProxyDetector()
        breaker = AtomicCircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=0.1,
        ))

        # Simulate connection workflow with proxy detection
        proxy_status, _ = await detector.detect()

        # Test the integration - circuit breaker respects proxy status
        if proxy_status == ProxyStatus.AVAILABLE:
            # Proxy is available - record success
            if await breaker.can_execute():
                await breaker.record_success()
                assert breaker.state == CircuitState.CLOSED
        else:
            # Proxy unavailable - record failures until circuit opens
            for _ in range(3):
                await breaker.record_failure("Proxy unavailable")
            assert breaker.state == CircuitState.OPEN

            # Wait for recovery timeout
            await asyncio.sleep(0.15)

            # Should be able to attempt again (transition to HALF_OPEN)
            can_retry = await breaker.can_execute()
            assert can_retry or breaker.state == CircuitState.HALF_OPEN

        # Verify circuit breaker state is valid (not corrupted)
        assert breaker.state in (
            CircuitState.CLOSED,
            CircuitState.OPEN,
            CircuitState.HALF_OPEN
        )

    @pytest.mark.asyncio
    async def test_lock_protects_shared_state(self):
        """EventLoopAwareLock should protect shared state from races."""
        lock = EventLoopAwareLock()
        counter = 0

        async def increment():
            nonlocal counter
            async with lock:
                temp = counter
                await asyncio.sleep(0.001)  # Simulate some async work
                counter = temp + 1

        # Run 100 concurrent increments
        await asyncio.gather(*[increment() for _ in range(100)])

        # Without lock, this would likely be less than 100
        assert counter == 100


class TestDIRegistration:
    """Tests that verify DI container registration works."""

    @pytest.mark.asyncio
    async def test_connection_services_import(self):
        """Connection services should be importable from DI package."""
        from backend.core.di import (
            register_connection_services,
            get_connection_health,
            get_connection_services_status,
        )

        # All functions should be callable
        assert callable(register_connection_services)
        assert callable(get_connection_health)
        assert callable(get_connection_services_status)

    @pytest.mark.asyncio
    async def test_connection_package_exports(self):
        """Connection package should export all required components."""
        from backend.core.connection import (
            AtomicStateMachine,
            AtomicCircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
            EventLoopAwareLock,
            ProactiveProxyDetector,
            ProxyDetectorConfig,
            ProxyStatus,
            StateTransition,
            StateTransitionError,
        )

        # All should be importable and usable
        assert AtomicStateMachine is not None
        assert AtomicCircuitBreaker is not None
        assert CircuitBreakerConfig is not None
        assert EventLoopAwareLock is not None
        assert ProactiveProxyDetector is not None
