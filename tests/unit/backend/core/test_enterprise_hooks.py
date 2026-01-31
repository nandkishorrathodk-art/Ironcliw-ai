"""
Tests for Enterprise Hooks module.

Tests the integration layer between enterprise modules and the existing
supervisor infrastructure.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from backend.core.enterprise_hooks import (
    # Initialization
    enterprise_init,
    enterprise_shutdown,
    is_enterprise_available,
    get_enterprise_status,
    # Failure handling
    handle_gcp_failure,
    handle_memory_pressure,
    GCPErrorContext,
    classify_gcp_error,
    # Routing
    get_routing_decision,
    route_with_fallback,
    record_provider_success,
    record_provider_failure,
    get_circuit_breaker_status,
    # Health
    aggregate_health,
    get_component_health,
    update_component_health,
    # Subprocess management
    start_cross_repo_process,
    stop_cross_repo_process,
    restart_cross_repo_process,
    shutdown_all_processes,
    is_process_running,
    get_process_handle,
    # Constants
    GCP_ERROR_PATTERNS,
    TRINITY_FALLBACK_CHAIN,
)


# =============================================================================
# Test Constants
# =============================================================================

class TestGCPErrorPatterns:
    """Tests for GCP_ERROR_PATTERNS constant."""

    def test_patterns_has_timeout_category(self):
        """Test that timeout patterns exist."""
        assert "gcp_startup_timeout" in GCP_ERROR_PATTERNS
        assert len(GCP_ERROR_PATTERNS["gcp_startup_timeout"]) > 0

    def test_patterns_has_network_category(self):
        """Test that network patterns exist."""
        assert "gcp_network_error" in GCP_ERROR_PATTERNS
        assert "Connection refused" in GCP_ERROR_PATTERNS["gcp_network_error"]

    def test_patterns_has_vm_state_category(self):
        """Test that VM state patterns exist."""
        assert "gcp_vm_state_error" in GCP_ERROR_PATTERNS
        assert "VM is in TERMINATED state" in GCP_ERROR_PATTERNS["gcp_vm_state_error"]

    def test_patterns_has_resource_category(self):
        """Test that resource patterns exist."""
        assert "gcp_resource_error" in GCP_ERROR_PATTERNS
        assert "Quota exceeded" in GCP_ERROR_PATTERNS["gcp_resource_error"]

    def test_patterns_has_auth_category(self):
        """Test that auth patterns exist."""
        assert "gcp_auth_error" in GCP_ERROR_PATTERNS
        assert "Permission denied" in GCP_ERROR_PATTERNS["gcp_auth_error"]


class TestTrinityFallbackChain:
    """Tests for TRINITY_FALLBACK_CHAIN constant."""

    def test_llm_inference_chain(self):
        """Test LLM inference fallback chain."""
        assert "llm_inference" in TRINITY_FALLBACK_CHAIN
        chain = TRINITY_FALLBACK_CHAIN["llm_inference"]
        assert "gcp_vm" in chain
        assert "claude_api" in chain
        assert chain.index("gcp_vm") < chain.index("claude_api")

    def test_voice_synthesis_chain(self):
        """Test voice synthesis fallback chain."""
        assert "voice_synthesis" in TRINITY_FALLBACK_CHAIN
        chain = TRINITY_FALLBACK_CHAIN["voice_synthesis"]
        assert len(chain) >= 2

    def test_voice_recognition_chain(self):
        """Test voice recognition fallback chain."""
        assert "voice_recognition" in TRINITY_FALLBACK_CHAIN

    def test_memory_retrieval_chain(self):
        """Test memory retrieval fallback chain."""
        assert "memory_retrieval" in TRINITY_FALLBACK_CHAIN


# =============================================================================
# Test GCPErrorContext
# =============================================================================

class TestGCPErrorContext:
    """Tests for GCPErrorContext dataclass."""

    def test_basic_creation(self):
        """Test basic context creation."""
        error = RuntimeError("Test error")
        ctx = GCPErrorContext(error=error, error_message=str(error))
        assert ctx.error == error
        assert ctx.error_message == "Test error"
        assert ctx.vm_ip is None
        assert ctx.gcp_attempts == 0

    def test_with_all_fields(self):
        """Test context with all fields."""
        error = RuntimeError("GCP timeout")
        ctx = GCPErrorContext(
            error=error,
            error_message=str(error),
            vm_ip="10.0.0.1",
            timeout_seconds=180.0,
            gcp_attempts=3,
            component="gcp_vm",
            additional_context={"region": "us-central1"},
        )
        assert ctx.vm_ip == "10.0.0.1"
        assert ctx.timeout_seconds == 180.0
        assert ctx.gcp_attempts == 3
        assert ctx.additional_context["region"] == "us-central1"

    def test_timestamp_auto_set(self):
        """Test that timestamp is auto-set."""
        ctx = GCPErrorContext(
            error=RuntimeError("test"),
            error_message="test",
        )
        assert ctx.timestamp is not None
        assert isinstance(ctx.timestamp, datetime)


# =============================================================================
# Test Error Classification
# =============================================================================

class TestClassifyGcpError:
    """Tests for classify_gcp_error function."""

    def test_classify_timeout_error(self):
        """Test timeout error classification."""
        error = RuntimeError("GCP VM not ready after 180s timeout")
        category, error_class = classify_gcp_error(error)
        assert category == "gcp_startup_timeout"
        # Handle both real and fallback enum
        value = error_class.value if hasattr(error_class, 'value') else str(error_class)
        assert value == "timeout"

    def test_classify_network_error(self):
        """Test network error classification."""
        error = ConnectionRefusedError("Connection refused")
        category, error_class = classify_gcp_error(error)
        assert category == "gcp_network_error"
        value = error_class.value if hasattr(error_class, 'value') else str(error_class)
        assert value == "network"

    def test_classify_vm_state_error(self):
        """Test VM state error classification."""
        error = RuntimeError("VM is in TERMINATED state")
        category, error_class = classify_gcp_error(error)
        assert category == "gcp_vm_state_error"
        value = error_class.value if hasattr(error_class, 'value') else str(error_class)
        assert value == "transient"

    def test_classify_resource_error(self):
        """Test resource error classification."""
        error = RuntimeError("Quota exceeded for region")
        category, error_class = classify_gcp_error(error)
        assert category == "gcp_resource_error"
        value = error_class.value if hasattr(error_class, 'value') else str(error_class)
        assert value == "resource_exhausted"

    def test_classify_auth_error(self):
        """Test auth error classification."""
        error = PermissionError("Permission denied")
        category, error_class = classify_gcp_error(error)
        assert category == "gcp_auth_error"
        value = error_class.value if hasattr(error_class, 'value') else str(error_class)
        assert value == "configuration"

    def test_classify_unknown_error(self):
        """Test unknown error classification."""
        error = RuntimeError("Some random error")
        category, error_class = classify_gcp_error(error)
        assert category == "gcp_unknown"
        value = error_class.value if hasattr(error_class, 'value') else str(error_class)
        assert value == "transient"

    def test_classify_case_insensitive(self):
        """Test classification is case insensitive."""
        error = RuntimeError("GCP READY WAIT TIMED OUT")
        category, error_class = classify_gcp_error(error)
        assert category == "gcp_startup_timeout"


# =============================================================================
# Test Enterprise Availability
# =============================================================================

class TestIsEnterpriseAvailable:
    """Tests for is_enterprise_available function."""

    def test_returns_bool(self):
        """Test that function returns boolean."""
        result = is_enterprise_available()
        assert isinstance(result, bool)


class TestGetEnterpriseStatus:
    """Tests for get_enterprise_status function."""

    def test_returns_dict(self):
        """Test that function returns dict."""
        status = get_enterprise_status()
        assert isinstance(status, dict)

    def test_has_all_keys(self):
        """Test that status has all expected keys."""
        status = get_enterprise_status()
        expected_keys = [
            "available",
            "recovery_engine",
            "capability_router",
            "health_aggregator",
            "subprocess_manager",
            "component_registry",
        ]
        for key in expected_keys:
            assert key in status


# =============================================================================
# Test Enterprise Initialization
# =============================================================================

class TestEnterpriseInit:
    """Tests for enterprise_init function."""

    @pytest.mark.asyncio
    async def test_init_returns_bool(self):
        """Test that init returns boolean."""
        # Reset module state first
        import backend.core.enterprise_hooks as hooks
        hooks._RECOVERY_ENGINE = None
        hooks._CAPABILITY_ROUTER = None
        hooks._HEALTH_AGGREGATOR = None
        hooks._SUBPROCESS_MANAGER = None
        hooks._COMPONENT_REGISTRY = None

        result = await enterprise_init()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_init_with_all_disabled(self):
        """Test init with all modules disabled."""
        result = await enterprise_init(
            enable_recovery=False,
            enable_routing=False,
            enable_health=False,
            enable_subprocess=False,
        )
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_init_sets_status(self):
        """Test that init sets enterprise status."""
        await enterprise_init()
        status = get_enterprise_status()
        # At least available should be set
        assert "available" in status


class TestEnterpriseShutdown:
    """Tests for enterprise_shutdown function."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_state(self):
        """Test that shutdown clears module state."""
        # First init
        await enterprise_init()

        # Then shutdown
        await enterprise_shutdown()

        # Check state is cleared
        import backend.core.enterprise_hooks as hooks
        assert hooks._RECOVERY_ENGINE is None
        assert hooks._CAPABILITY_ROUTER is None


# =============================================================================
# Test Failure Handling
# =============================================================================

class TestHandleGcpFailure:
    """Tests for handle_gcp_failure function."""

    @pytest.mark.asyncio
    async def test_handle_without_engine_returns_retry(self):
        """Test handling without recovery engine returns RETRY."""
        import backend.core.enterprise_hooks as hooks
        hooks._RECOVERY_ENGINE = None

        error = RuntimeError("GCP timeout")
        strategy = await handle_gcp_failure(error)
        # Handle both real enum and fallback enum
        value = strategy.value if hasattr(strategy, 'value') else str(strategy)
        assert value == "retry"

    @pytest.mark.asyncio
    async def test_handle_with_context(self):
        """Test handling with context."""
        import backend.core.enterprise_hooks as hooks
        hooks._RECOVERY_ENGINE = None

        error = RuntimeError("GCP VM not ready after 180s")
        ctx = GCPErrorContext(
            error=error,
            error_message=str(error),
            vm_ip="10.0.0.1",
            gcp_attempts=2,
        )
        strategy = await handle_gcp_failure(error, ctx)
        assert strategy is not None


class TestHandleMemoryPressure:
    """Tests for handle_memory_pressure function."""

    @pytest.mark.asyncio
    async def test_handle_normal_memory(self):
        """Test handling normal memory returns RETRY."""
        import backend.core.enterprise_hooks as hooks
        hooks._RECOVERY_ENGINE = None

        strategy = await handle_memory_pressure(50.0)
        # Handle both real enum and fallback enum
        value = strategy.value if hasattr(strategy, 'value') else str(strategy)
        assert value == "retry"

    @pytest.mark.asyncio
    async def test_handle_warning_memory(self):
        """Test handling warning level memory."""
        import backend.core.enterprise_hooks as hooks
        hooks._RECOVERY_ENGINE = None

        strategy = await handle_memory_pressure(87.0, trend="increasing", slope=0.5)
        assert strategy is not None

    @pytest.mark.asyncio
    async def test_handle_critical_memory(self):
        """Test handling critical memory."""
        import backend.core.enterprise_hooks as hooks
        hooks._RECOVERY_ENGINE = None

        strategy = await handle_memory_pressure(96.0, trend="increasing", slope=1.0)
        assert strategy is not None


# =============================================================================
# Test Routing Hooks
# =============================================================================

class TestGetRoutingDecision:
    """Tests for get_routing_decision function."""

    @pytest.mark.asyncio
    async def test_without_router_returns_none(self):
        """Test that function returns None without router."""
        import backend.core.enterprise_hooks as hooks
        hooks._CAPABILITY_ROUTER = None

        result = await get_routing_decision("llm_inference")
        assert result is None

    @pytest.mark.asyncio
    async def test_with_preferred_provider(self):
        """Test routing with preferred provider."""
        import backend.core.enterprise_hooks as hooks
        hooks._CAPABILITY_ROUTER = None

        result = await get_routing_decision(
            "llm_inference",
            preferred_provider="claude_api",
        )
        assert result is None  # No router available


class TestRouteWithFallback:
    """Tests for route_with_fallback function."""

    @pytest.mark.asyncio
    async def test_without_router_executes_directly(self):
        """Test that operation executes directly without router."""
        import backend.core.enterprise_hooks as hooks
        hooks._CAPABILITY_ROUTER = None

        async def my_operation(x: int) -> int:
            return x * 2

        result = await route_with_fallback("test", my_operation, 5)
        assert result == 10


class TestRecordProviderSuccess:
    """Tests for record_provider_success function."""

    def test_without_router_is_safe(self):
        """Test that function is safe without router."""
        import backend.core.enterprise_hooks as hooks
        hooks._CAPABILITY_ROUTER = None

        # Should not raise
        record_provider_success("test", "provider")


class TestRecordProviderFailure:
    """Tests for record_provider_failure function."""

    def test_without_router_is_safe(self):
        """Test that function is safe without router."""
        import backend.core.enterprise_hooks as hooks
        hooks._CAPABILITY_ROUTER = None

        # Should not raise
        record_provider_failure("test", "provider", RuntimeError("fail"))


class TestGetCircuitBreakerStatus:
    """Tests for get_circuit_breaker_status function."""

    def test_without_router_returns_none(self):
        """Test that function returns None without router."""
        import backend.core.enterprise_hooks as hooks
        hooks._CAPABILITY_ROUTER = None

        result = get_circuit_breaker_status("test")
        assert result is None


# =============================================================================
# Test Health Hooks
# =============================================================================

class TestAggregateHealth:
    """Tests for aggregate_health function."""

    @pytest.mark.asyncio
    async def test_without_aggregator_returns_none(self):
        """Test that function returns None without aggregator."""
        import backend.core.enterprise_hooks as hooks
        hooks._HEALTH_AGGREGATOR = None

        result = await aggregate_health()
        assert result is None


class TestGetComponentHealth:
    """Tests for get_component_health function."""

    @pytest.mark.asyncio
    async def test_without_registry_returns_none(self):
        """Test that function returns None without registry."""
        import backend.core.enterprise_hooks as hooks
        hooks._COMPONENT_REGISTRY = None

        result = await get_component_health("test")
        assert result is None


class TestUpdateComponentHealth:
    """Tests for update_component_health function."""

    def test_without_registry_is_safe(self):
        """Test that function is safe without registry."""
        import backend.core.enterprise_hooks as hooks
        hooks._COMPONENT_REGISTRY = None

        from backend.core.health_contracts import HealthStatus

        # Should not raise
        update_component_health("test", HealthStatus.HEALTHY)


# =============================================================================
# Test Subprocess Management
# =============================================================================

class TestStartCrossRepoProcess:
    """Tests for start_cross_repo_process function."""

    @pytest.mark.asyncio
    async def test_without_manager_returns_none(self):
        """Test that function returns None without manager."""
        import backend.core.enterprise_hooks as hooks
        hooks._SUBPROCESS_MANAGER = None
        hooks._COMPONENT_REGISTRY = None

        result = await start_cross_repo_process(
            name="test",
            repo_path="/tmp/test",
        )
        assert result is None


class TestStopCrossRepoProcess:
    """Tests for stop_cross_repo_process function."""

    @pytest.mark.asyncio
    async def test_without_manager_returns_true(self):
        """Test that function returns True without manager."""
        import backend.core.enterprise_hooks as hooks
        hooks._SUBPROCESS_MANAGER = None

        result = await stop_cross_repo_process("test")
        assert result is True


class TestRestartCrossRepoProcess:
    """Tests for restart_cross_repo_process function."""

    @pytest.mark.asyncio
    async def test_without_manager_returns_none(self):
        """Test that function returns None without manager."""
        import backend.core.enterprise_hooks as hooks
        hooks._SUBPROCESS_MANAGER = None

        result = await restart_cross_repo_process("test")
        assert result is None


class TestShutdownAllProcesses:
    """Tests for shutdown_all_processes function."""

    @pytest.mark.asyncio
    async def test_without_manager_is_safe(self):
        """Test that function is safe without manager."""
        import backend.core.enterprise_hooks as hooks
        hooks._SUBPROCESS_MANAGER = None

        # Should not raise
        await shutdown_all_processes()


class TestIsProcessRunning:
    """Tests for is_process_running function."""

    def test_without_manager_returns_false(self):
        """Test that function returns False without manager."""
        import backend.core.enterprise_hooks as hooks
        hooks._SUBPROCESS_MANAGER = None

        result = is_process_running("test")
        assert result is False


class TestGetProcessHandle:
    """Tests for get_process_handle function."""

    def test_without_manager_returns_none(self):
        """Test that function returns None without manager."""
        import backend.core.enterprise_hooks as hooks
        hooks._SUBPROCESS_MANAGER = None

        result = get_process_handle("test")
        assert result is None


# =============================================================================
# Test Module Exports
# =============================================================================

class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_defined(self):
        """Test that all exports in __all__ are defined."""
        import backend.core.enterprise_hooks as hooks

        for name in hooks.__all__:
            assert hasattr(hooks, name), f"Missing export: {name}"

    def test_key_functions_exported(self):
        """Test that key functions are exported."""
        import backend.core.enterprise_hooks as hooks

        key_exports = [
            "enterprise_init",
            "enterprise_shutdown",
            "handle_gcp_failure",
            "handle_memory_pressure",
            "get_routing_decision",
            "aggregate_health",
            "start_cross_repo_process",
        ]
        for name in key_exports:
            assert name in hooks.__all__


# =============================================================================
# Integration Tests
# =============================================================================

class TestEnterpriseIntegration:
    """Integration tests for enterprise hooks."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full init/use/shutdown lifecycle."""
        # Init
        result = await enterprise_init()
        assert isinstance(result, bool)

        # Check status
        status = get_enterprise_status()
        assert status["available"] == is_enterprise_available()

        # Shutdown
        await enterprise_shutdown()

        # Verify cleanup
        import backend.core.enterprise_hooks as hooks
        assert hooks._RECOVERY_ENGINE is None

    @pytest.mark.asyncio
    async def test_error_handling_flow(self):
        """Test error handling through enterprise hooks."""
        await enterprise_init()

        try:
            # Simulate GCP failure
            error = RuntimeError("GCP VM not ready after 180s timeout")
            strategy = await handle_gcp_failure(error)
            assert strategy is not None

            # Simulate memory pressure
            strategy = await handle_memory_pressure(92.0)
            assert strategy is not None
        finally:
            await enterprise_shutdown()

    @pytest.mark.asyncio
    async def test_safe_without_initialization(self):
        """Test that all functions are safe without initialization."""
        import backend.core.enterprise_hooks as hooks

        # Ensure nothing is initialized
        hooks._RECOVERY_ENGINE = None
        hooks._CAPABILITY_ROUTER = None
        hooks._HEALTH_AGGREGATOR = None
        hooks._SUBPROCESS_MANAGER = None
        hooks._COMPONENT_REGISTRY = None

        # All these should be safe and not raise
        strategy = await handle_gcp_failure(RuntimeError("test"))
        assert strategy is not None  # Should return fallback RETRY

        strategy = await handle_memory_pressure(90.0)
        assert strategy is not None

        await get_routing_decision("test")

        async def dummy_op():
            await asyncio.sleep(0)
            return "done"

        result = await route_with_fallback("test", dummy_op)
        assert result == "done"

        record_provider_success("test", "provider")
        record_provider_failure("test", "provider", RuntimeError("test"))
        get_circuit_breaker_status("test")
        await aggregate_health()
        await get_component_health("test")
        await start_cross_repo_process("test", "/tmp")
        await stop_cross_repo_process("test")
        await restart_cross_repo_process("test")
        await shutdown_all_processes()
        is_process_running("test")
        get_process_handle("test")
