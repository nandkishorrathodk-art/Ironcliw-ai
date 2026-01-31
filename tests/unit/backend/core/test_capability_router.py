"""Tests for CapabilityRouter - routes requests based on available capabilities."""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_circuit_state_values(self):
        from backend.core.capability_router import CircuitState
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreaker:
    """Tests for CircuitBreaker state machine."""

    def test_initial_state(self):
        """New circuit breaker starts closed."""
        from backend.core.capability_router import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(provider="test-provider")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
        assert breaker.can_execute() is True

    def test_record_success_resets_failure_count(self):
        """Recording success resets failure count."""
        from backend.core.capability_router import CircuitBreaker
        breaker = CircuitBreaker(provider="test-provider")
        breaker.failure_count = 3
        breaker.record_success()
        assert breaker.failure_count == 0
        assert breaker.success_count == 1

    def test_record_failure_increments_count(self):
        """Recording failure increments failure count."""
        from backend.core.capability_router import CircuitBreaker
        breaker = CircuitBreaker(provider="test-provider")
        breaker.record_failure()
        assert breaker.failure_count == 1
        assert breaker.success_count == 0

    def test_circuit_opens_after_threshold_failures(self):
        """Circuit opens after reaching failure threshold."""
        from backend.core.capability_router import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(provider="test-provider", failure_threshold=3)

        # First two failures don't open circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        # Third failure opens circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.can_execute() is False

    def test_circuit_half_opens_after_timeout(self):
        """Circuit transitions to half-open after timeout."""
        from backend.core.capability_router import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(
            provider="test-provider",
            failure_threshold=3,
            timeout_seconds=0.1  # Very short timeout for testing
        )

        # Open the circuit
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        import time
        time.sleep(0.15)

        # Next check should transition to half-open
        assert breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN

    def test_circuit_closes_after_success_threshold_in_half_open(self):
        """Circuit closes after enough successes in half-open state."""
        from backend.core.capability_router import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(
            provider="test-provider",
            success_threshold=2
        )
        breaker.state = CircuitState.HALF_OPEN
        breaker.last_state_change = datetime.now()

        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN  # Not closed yet

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED  # Now closed

    def test_circuit_reopens_on_failure_in_half_open(self):
        """Circuit reopens on failure in half-open state."""
        from backend.core.capability_router import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(
            provider="test-provider",
            failure_threshold=1
        )
        breaker.state = CircuitState.HALF_OPEN
        breaker.last_state_change = datetime.now()

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_reset_clears_state(self):
        """Reset returns circuit to initial state."""
        from backend.core.capability_router import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(provider="test-provider")
        breaker.state = CircuitState.OPEN
        breaker.failure_count = 10
        breaker.success_count = 5
        breaker.last_failure = datetime.now()

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
        assert breaker.last_failure is None
        assert breaker.last_state_change is not None


class TestRoutingDecision:
    """Tests for RoutingDecision dataclass."""

    def test_routing_decision_creation(self):
        """RoutingDecision can be created with required fields."""
        from backend.core.capability_router import RoutingDecision
        decision = RoutingDecision(
            capability="inference",
            provider="jarvis-prime",
            is_fallback=False
        )
        assert decision.capability == "inference"
        assert decision.provider == "jarvis-prime"
        assert decision.is_fallback is False
        assert decision.fallback_reason is None

    def test_routing_decision_with_fallback(self):
        """RoutingDecision can indicate fallback usage."""
        from backend.core.capability_router import RoutingDecision, CircuitState
        decision = RoutingDecision(
            capability="inference",
            provider="claude-api",
            is_fallback=True,
            fallback_reason="Primary provider jarvis-prime is failed",
            circuit_state=CircuitState.OPEN
        )
        assert decision.is_fallback is True
        assert decision.fallback_reason is not None
        assert decision.circuit_state == CircuitState.OPEN


class TestCapabilityRouter:
    """Tests for CapabilityRouter main class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock ComponentRegistry."""
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentState,
            Criticality, ProcessType, ComponentStatus
        )
        registry = ComponentRegistry()
        registry._reset_for_testing()
        return registry

    @pytest.fixture
    def router_with_registry(self, mock_registry):
        """Create a CapabilityRouter with mock registry."""
        from backend.core.capability_router import CapabilityRouter
        return CapabilityRouter(mock_registry)

    def test_is_capability_available_returns_false_when_not_registered(self, router_with_registry):
        """is_capability_available returns False when capability not registered."""
        assert router_with_registry.is_capability_available("nonexistent") is False

    def test_is_capability_available_returns_true_when_healthy(self, mock_registry, router_with_registry):
        """is_capability_available returns True when provider is healthy."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        defn = ComponentDefinition(
            name="test-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        mock_registry.register(defn)
        mock_registry.mark_status("test-provider", ComponentStatus.HEALTHY)

        assert router_with_registry.is_capability_available("inference") is True

    def test_is_capability_available_returns_true_when_degraded(self, mock_registry, router_with_registry):
        """is_capability_available returns True when provider is degraded."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        defn = ComponentDefinition(
            name="test-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        mock_registry.register(defn)
        mock_registry.mark_status("test-provider", ComponentStatus.DEGRADED, "Reduced capacity")

        assert router_with_registry.is_capability_available("inference") is True

    def test_is_capability_available_returns_false_when_failed(self, mock_registry, router_with_registry):
        """is_capability_available returns False when provider is failed."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        defn = ComponentDefinition(
            name="test-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        mock_registry.register(defn)
        mock_registry.mark_status("test-provider", ComponentStatus.FAILED, "Connection error")

        assert router_with_registry.is_capability_available("inference") is False

    @pytest.mark.asyncio
    async def test_route_returns_provider(self, mock_registry, router_with_registry):
        """route() returns provider name for available capability."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        defn = ComponentDefinition(
            name="test-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        mock_registry.register(defn)
        mock_registry.mark_status("test-provider", ComponentStatus.HEALTHY)

        provider = await router_with_registry.route("inference")
        assert provider == "test-provider"

    @pytest.mark.asyncio
    async def test_route_returns_none_when_unavailable(self, router_with_registry):
        """route() returns None when capability not available."""
        provider = await router_with_registry.route("nonexistent")
        assert provider is None

    @pytest.mark.asyncio
    async def test_get_routing_decision_with_healthy_provider(self, mock_registry, router_with_registry):
        """get_routing_decision returns correct decision for healthy provider."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        from backend.core.capability_router import CircuitState

        defn = ComponentDefinition(
            name="test-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        mock_registry.register(defn)
        mock_registry.mark_status("test-provider", ComponentStatus.HEALTHY)

        decision = await router_with_registry.get_routing_decision("inference")
        assert decision.provider == "test-provider"
        assert decision.is_fallback is False
        assert decision.circuit_state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_get_routing_decision_with_unhealthy_provider(self, mock_registry, router_with_registry):
        """get_routing_decision returns fallback for unhealthy provider."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        defn = ComponentDefinition(
            name="test-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        mock_registry.register(defn)
        mock_registry.mark_status("test-provider", ComponentStatus.FAILED, "Error")

        decision = await router_with_registry.get_routing_decision("inference")
        assert decision.is_fallback is True
        assert "failed" in decision.fallback_reason.lower()

    @pytest.mark.asyncio
    async def test_get_routing_decision_with_circuit_breaker_open(self, mock_registry, router_with_registry):
        """get_routing_decision respects circuit breaker state."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        from backend.core.capability_router import CircuitState

        defn = ComponentDefinition(
            name="test-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        mock_registry.register(defn)
        mock_registry.mark_status("test-provider", ComponentStatus.HEALTHY)

        # Manually open the circuit breaker
        breaker = router_with_registry._get_or_create_breaker("test-provider")
        breaker.state = CircuitState.OPEN
        breaker.last_state_change = datetime.now()

        decision = await router_with_registry.get_routing_decision("inference")
        assert decision.is_fallback is True
        assert "circuit breaker" in decision.fallback_reason.lower()

    @pytest.mark.asyncio
    async def test_call_with_fallback_success(self, mock_registry, router_with_registry):
        """call_with_fallback executes primary callable on success."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        defn = ComponentDefinition(
            name="test-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        mock_registry.register(defn)
        mock_registry.mark_status("test-provider", ComponentStatus.HEALTHY)

        # Register callable
        async def primary_callable(prompt):
            return f"Primary response: {prompt}"

        router_with_registry.register_provider_callable(
            "inference", "test-provider", primary_callable
        )

        result = await router_with_registry.call_with_fallback("inference", prompt="Hello")
        assert result == "Primary response: Hello"

    @pytest.mark.asyncio
    async def test_call_with_fallback_uses_fallback_on_failure(self, mock_registry, router_with_registry):
        """call_with_fallback uses fallback when primary fails."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        defn = ComponentDefinition(
            name="test-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        mock_registry.register(defn)
        mock_registry.mark_status("test-provider", ComponentStatus.HEALTHY)

        # Register failing callable
        async def failing_callable(prompt):
            raise RuntimeError("Primary failed")

        async def fallback_callable(prompt):
            return f"Fallback response: {prompt}"

        router_with_registry.register_provider_callable(
            "inference", "test-provider", failing_callable
        )
        router_with_registry.register_fallback_callable("inference", fallback_callable)

        result = await router_with_registry.call_with_fallback("inference", prompt="Hello")
        assert result == "Fallback response: Hello"

        # Verify circuit breaker recorded failure
        breaker = router_with_registry._get_or_create_breaker("test-provider")
        assert breaker.failure_count == 1

    @pytest.mark.asyncio
    async def test_call_with_fallback_raises_when_no_fallback(self, mock_registry, router_with_registry):
        """call_with_fallback raises when no provider or fallback available."""
        with pytest.raises(RuntimeError, match="No provider or fallback available"):
            await router_with_registry.call_with_fallback("nonexistent", prompt="Hello")

    def test_get_fallback_chain(self, mock_registry, router_with_registry):
        """get_fallback_chain returns ordered list of providers."""
        from backend.core.component_registry import (
            ComponentDefinition, Criticality, ProcessType, ComponentStatus
        )
        defn = ComponentDefinition(
            name="primary-provider",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"],
            fallback_for_capabilities={"inference": "backup-provider"}
        )
        mock_registry.register(defn)
        mock_registry.mark_status("primary-provider", ComponentStatus.HEALTHY)

        # Register fallback callable
        async def fallback():
            return "fallback"
        router_with_registry.register_fallback_callable("inference", fallback)

        chain = router_with_registry.get_fallback_chain("inference")
        assert "primary-provider" in chain
        assert "fallback" in chain

    def test_reset_circuit_breaker(self, router_with_registry):
        """reset_circuit_breaker resets breaker state."""
        from backend.core.capability_router import CircuitState

        breaker = router_with_registry._get_or_create_breaker("test-provider")
        breaker.state = CircuitState.OPEN
        breaker.failure_count = 10

        router_with_registry.reset_circuit_breaker("test-provider")

        breaker = router_with_registry._get_or_create_breaker("test-provider")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_get_circuit_breaker_status(self, router_with_registry):
        """get_circuit_breaker_status returns all breaker states."""
        from backend.core.capability_router import CircuitState

        breaker1 = router_with_registry._get_or_create_breaker("provider-a")
        breaker1.state = CircuitState.CLOSED

        breaker2 = router_with_registry._get_or_create_breaker("provider-b")
        breaker2.state = CircuitState.OPEN
        breaker2.failure_count = 5
        breaker2.last_failure = datetime.now()

        status = router_with_registry.get_circuit_breaker_status()

        assert "provider-a" in status
        assert "provider-b" in status
        assert status["provider-a"]["state"] == "closed"
        assert status["provider-b"]["state"] == "open"
        assert status["provider-b"]["failure_count"] == 5

    def test_register_provider_callable(self, router_with_registry):
        """register_provider_callable stores callable correctly."""
        async def test_callable():
            return "result"

        router_with_registry.register_provider_callable(
            "inference", "test-provider", test_callable
        )

        key = "inference:test-provider"
        assert key in router_with_registry._provider_callables
        assert router_with_registry._provider_callables[key] is test_callable

    def test_register_fallback_callable(self, router_with_registry):
        """register_fallback_callable stores callable correctly."""
        async def fallback_callable():
            return "fallback"

        router_with_registry.register_fallback_callable("inference", fallback_callable)

        assert "inference" in router_with_registry._fallback_callables
        assert router_with_registry._fallback_callables["inference"] is fallback_callable


class TestCapabilityRouterWithFallbackConfiguration:
    """Tests for CapabilityRouter with fallback_for_capabilities configuration."""

    @pytest.fixture
    def registry_with_fallback(self):
        """Create registry with primary and fallback providers."""
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, Criticality,
            ProcessType, ComponentStatus
        )
        registry = ComponentRegistry()
        registry._reset_for_testing()

        # Register fallback provider first (so primary registered last becomes the provider)
        fallback = ComponentDefinition(
            name="claude-api",
            criticality=Criticality.REQUIRED,
            process_type=ProcessType.EXTERNAL_SERVICE,
            provides_capabilities=["cloud-llm"]  # Don't provide inference to avoid conflict
        )
        registry.register(fallback)

        # Register primary provider with fallback configuration LAST
        # (last registered becomes the get_provider result)
        primary = ComponentDefinition(
            name="jarvis-prime",
            criticality=Criticality.DEGRADED_OK,
            process_type=ProcessType.SUBPROCESS,
            provides_capabilities=["inference", "local-llm"],
            fallback_for_capabilities={"inference": "claude-api"}
        )
        registry.register(primary)

        return registry

    @pytest.mark.asyncio
    async def test_fallback_to_configured_provider(self, registry_with_fallback):
        """Router falls back to configured fallback provider."""
        from backend.core.capability_router import CapabilityRouter
        from backend.core.component_registry import ComponentStatus

        router = CapabilityRouter(registry_with_fallback)

        # Primary is failed, fallback is healthy
        registry_with_fallback.mark_status("jarvis-prime", ComponentStatus.FAILED, "Error")
        registry_with_fallback.mark_status("claude-api", ComponentStatus.HEALTHY)

        decision = await router.get_routing_decision("inference")
        assert decision.is_fallback is True
        assert decision.provider == "claude-api"

    def test_fallback_chain_includes_configured_fallback(self, registry_with_fallback):
        """Fallback chain includes configured fallback provider."""
        from backend.core.capability_router import CapabilityRouter
        from backend.core.component_registry import ComponentStatus

        router = CapabilityRouter(registry_with_fallback)
        registry_with_fallback.mark_status("jarvis-prime", ComponentStatus.HEALTHY)

        chain = router.get_fallback_chain("inference")
        assert "jarvis-prime" in chain
        assert "claude-api" in chain


class TestFactoryFunction:
    """Tests for factory function."""

    def test_get_capability_router_creates_router(self):
        """get_capability_router creates CapabilityRouter instance."""
        from backend.core.capability_router import get_capability_router, CapabilityRouter
        from backend.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._reset_for_testing()

        router = get_capability_router(registry)
        assert isinstance(router, CapabilityRouter)
        assert router.registry is registry


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with routing."""

    @pytest.fixture
    def router_with_providers(self):
        """Create router with multiple providers."""
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, Criticality,
            ProcessType, ComponentStatus
        )
        from backend.core.capability_router import CapabilityRouter

        registry = ComponentRegistry()
        registry._reset_for_testing()

        # Register providers
        provider1 = ComponentDefinition(
            name="provider-1",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference"]
        )
        registry.register(provider1)
        registry.mark_status("provider-1", ComponentStatus.HEALTHY)

        return CapabilityRouter(registry)

    @pytest.mark.asyncio
    async def test_repeated_failures_open_circuit(self, router_with_providers):
        """Repeated failures open the circuit breaker."""
        from backend.core.capability_router import CircuitState

        # Register failing callable
        call_count = 0
        async def failing_callable(prompt):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        async def fallback_callable(prompt):
            return "fallback"

        router_with_providers.register_provider_callable(
            "inference", "provider-1", failing_callable
        )
        router_with_providers.register_fallback_callable("inference", fallback_callable)

        # Set low threshold for testing
        breaker = router_with_providers._get_or_create_breaker("provider-1")
        breaker.failure_threshold = 3

        # Make calls until circuit opens
        for _ in range(5):
            result = await router_with_providers.call_with_fallback("inference", prompt="test")
            assert result == "fallback"

        # Circuit should be open now
        assert breaker.state == CircuitState.OPEN

        # Verify primary was called only until circuit opened
        assert call_count == 3  # Called until threshold reached

    @pytest.mark.asyncio
    async def test_circuit_recovery(self, router_with_providers):
        """Circuit recovers after timeout and successful calls."""
        from backend.core.capability_router import CircuitState
        import time

        # Start with open circuit
        breaker = router_with_providers._get_or_create_breaker("provider-1")
        breaker.state = CircuitState.OPEN
        breaker.last_state_change = datetime.now()
        breaker.timeout_seconds = 0.1  # Short timeout
        breaker.success_threshold = 2

        # Register working callable
        async def working_callable(prompt):
            return f"Success: {prompt}"

        router_with_providers.register_provider_callable(
            "inference", "provider-1", working_callable
        )

        # Wait for timeout
        time.sleep(0.15)

        # First call should succeed and start recovery
        result = await router_with_providers.call_with_fallback("inference", prompt="test1")
        assert result == "Success: test1"
        assert breaker.state == CircuitState.HALF_OPEN

        # Second success should close circuit
        result = await router_with_providers.call_with_fallback("inference", prompt="test2")
        assert result == "Success: test2"
        assert breaker.state == CircuitState.CLOSED
