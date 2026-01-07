#!/usr/bin/env python3
"""
Trinity Integration Tests
=========================

Tests for the Trinity ecosystem integration:
- Prime Client
- Prime Router
- Graceful Degradation
- Query Handler with Trinity routing

Run with: python3 -m pytest tests/test_trinity_integration.py -v
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPrimeClient:
    """Tests for PrimeClient."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test that PrimeClient initializes correctly."""
        from core.prime_client import PrimeClient, PrimeClientConfig

        config = PrimeClientConfig(
            prime_host="localhost",
            prime_port=8020,
        )
        client = PrimeClient(config)

        assert client._config.prime_host == "localhost"
        assert client._config.prime_port == 8020
        assert not client._initialized

    @pytest.mark.asyncio
    async def test_client_status(self):
        """Test client status reporting."""
        from core.prime_client import PrimeClient

        client = PrimeClient()
        status = client.get_status()

        assert "status" in status
        assert "initialized" in status
        assert "circuit_breaker" in status

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failure(self):
        """Test that circuit breaker records failures correctly."""
        from core.prime_client import PrimeCircuitBreaker, PrimeClientConfig

        config = PrimeClientConfig()
        circuit = PrimeCircuitBreaker(config)

        # Should be able to execute initially
        assert await circuit.can_execute()

        # Record failures
        for _ in range(5):
            await circuit.record_failure()

        # Circuit should be open after 5 failures (default threshold)
        state = circuit.get_state()
        assert state["state"] == "open"


class TestPrimeRouter:
    """Tests for PrimeRouter."""

    @pytest.mark.asyncio
    async def test_router_config(self):
        """Test router configuration."""
        from core.prime_router import PrimeRouter, PrimeRouterConfig

        config = PrimeRouterConfig(
            prefer_local=True,
            enable_cloud_fallback=True,
        )
        router = PrimeRouter(config)

        assert router._config.prefer_local
        assert router._config.enable_cloud_fallback

    @pytest.mark.asyncio
    async def test_router_status(self):
        """Test router status reporting."""
        from core.prime_router import PrimeRouter

        router = PrimeRouter()
        status = router.get_status()

        assert "initialized" in status
        assert "config" in status
        assert "metrics" in status

    @pytest.mark.asyncio
    async def test_routing_decision_hybrid_when_prefer_local(self):
        """Test that routing defaults to hybrid when prefer_local is True."""
        from core.prime_router import PrimeRouter, RoutingDecision

        router = PrimeRouter()
        router._initialized = True
        router._prime_client = MagicMock()
        router._prime_client.is_available = True

        decision = router._decide_route()
        assert decision == RoutingDecision.HYBRID


class TestGracefulDegradation:
    """Tests for Graceful Degradation system."""

    @pytest.mark.asyncio
    async def test_degradation_initialization(self):
        """Test graceful degradation initialization."""
        from core.graceful_degradation import GracefulDegradation, InferenceTarget

        degradation = GracefulDegradation()

        # All targets should be enabled by default
        assert degradation._targets[InferenceTarget.LOCAL_PRIME].enabled
        assert degradation._targets[InferenceTarget.CLOUD_CLAUDE].enabled
        assert degradation._targets[InferenceTarget.CACHED].enabled

    @pytest.mark.asyncio
    async def test_best_target_prefers_local(self):
        """Test that best target prefers local when healthy."""
        from core.graceful_degradation import GracefulDegradation, InferenceTarget

        degradation = GracefulDegradation()

        decision = await degradation.get_best_target(prefer_local=True)

        # Should prefer local_prime
        assert decision.target == InferenceTarget.LOCAL_PRIME

    @pytest.mark.asyncio
    async def test_fallback_chain(self):
        """Test fallback chain order."""
        from core.graceful_degradation import GracefulDegradation, InferenceTarget

        degradation = GracefulDegradation()

        assert degradation._fallback_chain == [
            InferenceTarget.LOCAL_PRIME,
            InferenceTarget.CLOUD_CLAUDE,
            InferenceTarget.CLOUD_OPENAI,
            InferenceTarget.CACHED,
            InferenceTarget.DEGRADED,
        ]

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self):
        """Test that circuit opens after consecutive failures."""
        from core.graceful_degradation import GracefulDegradation, InferenceTarget, TargetHealth

        degradation = GracefulDegradation()
        target = InferenceTarget.LOCAL_PRIME

        # Record 5 failures
        for _ in range(5):
            degradation._targets[target].record_failure()

        # Check circuit is open
        assert degradation._targets[target].circuit_open
        assert degradation._targets[target].health == TargetHealth.UNHEALTHY

    @pytest.mark.asyncio
    async def test_status_reporting(self):
        """Test status reporting."""
        from core.graceful_degradation import GracefulDegradation

        degradation = GracefulDegradation()
        status = degradation.get_status()

        assert "targets" in status
        assert "fallback_chain" in status
        assert "manual_override" in status


class TestQueryHandler:
    """Tests for Query Handler with Trinity integration."""

    @pytest.mark.asyncio
    async def test_fallback_to_cloud(self):
        """Test cloud fallback function."""
        from api.query_handler import _fallback_to_cloud

        # Without API key, should return degraded
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            result = await _fallback_to_cloud("test query")
            assert result["source"] == "degraded"


class TestTrinityHealthAPI:
    """Tests for Trinity Health API endpoints."""

    @pytest.mark.asyncio
    async def test_health_helpers(self):
        """Test health check helper functions."""
        from api.trinity_health_api import _check_reactor_health

        # Reactor health should return unknown if no heartbeat file
        status = await _check_reactor_health()
        assert status.name == "Reactor-Core (Nerves)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
