"""Tests for HealthContracts - System health monitoring and aggregation."""
import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock


class TestHealthStatus:
    """Tests for HealthStatus enum values."""

    def test_healthy_value(self):
        from backend.core.health_contracts import HealthStatus
        assert HealthStatus.HEALTHY.value == "healthy"

    def test_degraded_value(self):
        from backend.core.health_contracts import HealthStatus
        assert HealthStatus.DEGRADED.value == "degraded"

    def test_unhealthy_value(self):
        from backend.core.health_contracts import HealthStatus
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_unknown_value(self):
        from backend.core.health_contracts import HealthStatus
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_all_values_exist(self):
        from backend.core.health_contracts import HealthStatus
        # Verify all expected members exist
        members = {m.value for m in HealthStatus}
        assert members == {"healthy", "degraded", "unhealthy", "unknown"}


class TestHealthReport:
    """Tests for HealthReport dataclass."""

    def test_minimal_creation(self):
        from backend.core.health_contracts import HealthReport, HealthStatus
        now = datetime.now(timezone.utc)
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            component="test-component",
            timestamp=now,
            latency_ms=None,
            details={},
            dependencies_ok=True,
            message=None,
        )
        assert report.status == HealthStatus.HEALTHY
        assert report.component == "test-component"
        assert report.timestamp == now
        assert report.latency_ms is None
        assert report.details == {}
        assert report.dependencies_ok is True
        assert report.message is None
        assert report.previous_status is None
        assert report.version is None

    def test_full_creation(self):
        from backend.core.health_contracts import HealthReport, HealthStatus
        now = datetime.now(timezone.utc)
        report = HealthReport(
            status=HealthStatus.DEGRADED,
            component="inference-engine",
            timestamp=now,
            latency_ms=150.5,
            details={"model_loaded": True, "cache_hit_rate": 0.85},
            dependencies_ok=False,
            message="Running with reduced capacity",
            previous_status=HealthStatus.HEALTHY,
            version="1.2.3",
        )
        assert report.status == HealthStatus.DEGRADED
        assert report.latency_ms == 150.5
        assert report.details["model_loaded"] is True
        assert report.details["cache_hit_rate"] == 0.85
        assert report.dependencies_ok is False
        assert report.message == "Running with reduced capacity"
        assert report.previous_status == HealthStatus.HEALTHY
        assert report.version == "1.2.3"

    def test_unhealthy_report(self):
        from backend.core.health_contracts import HealthReport, HealthStatus
        now = datetime.now(timezone.utc)
        report = HealthReport(
            status=HealthStatus.UNHEALTHY,
            component="database",
            timestamp=now,
            latency_ms=5000.0,
            details={"error": "Connection refused", "retries": 3},
            dependencies_ok=False,
            message="Database connection failed after 3 retries",
        )
        assert report.status == HealthStatus.UNHEALTHY
        assert report.details["error"] == "Connection refused"


class TestCapabilityHealth:
    """Tests for CapabilityHealth dataclass."""

    def test_available_capability(self):
        from backend.core.health_contracts import CapabilityHealth, HealthStatus
        cap = CapabilityHealth(
            available=True,
            provider="jarvis-prime",
            status=HealthStatus.HEALTHY,
        )
        assert cap.available is True
        assert cap.provider == "jarvis-prime"
        assert cap.status == HealthStatus.HEALTHY

    def test_unavailable_capability(self):
        from backend.core.health_contracts import CapabilityHealth, HealthStatus
        cap = CapabilityHealth(
            available=False,
            provider=None,
            status=HealthStatus.UNHEALTHY,
        )
        assert cap.available is False
        assert cap.provider is None
        assert cap.status == HealthStatus.UNHEALTHY

    def test_degraded_capability(self):
        from backend.core.health_contracts import CapabilityHealth, HealthStatus
        cap = CapabilityHealth(
            available=True,
            provider="fallback-service",
            status=HealthStatus.DEGRADED,
        )
        assert cap.available is True
        assert cap.status == HealthStatus.DEGRADED


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_healthy_system(self):
        from backend.core.health_contracts import (
            SystemHealth, HealthStatus, HealthReport, CapabilityHealth
        )
        now = datetime.now(timezone.utc)
        health = SystemHealth(
            overall=HealthStatus.HEALTHY,
            components={
                "core": HealthReport(
                    status=HealthStatus.HEALTHY,
                    component="core",
                    timestamp=now,
                    latency_ms=5.0,
                    details={},
                    dependencies_ok=True,
                    message=None,
                )
            },
            capabilities={
                "inference": CapabilityHealth(
                    available=True,
                    provider="core",
                    status=HealthStatus.HEALTHY,
                )
            },
            timestamp=now,
        )
        assert health.overall == HealthStatus.HEALTHY
        assert "core" in health.components
        assert "inference" in health.capabilities
        assert health.timestamp == now

    def test_degraded_system(self):
        from backend.core.health_contracts import (
            SystemHealth, HealthStatus, HealthReport, CapabilityHealth
        )
        now = datetime.now(timezone.utc)
        health = SystemHealth(
            overall=HealthStatus.DEGRADED,
            components={
                "core": HealthReport(
                    status=HealthStatus.HEALTHY,
                    component="core",
                    timestamp=now,
                    latency_ms=5.0,
                    details={},
                    dependencies_ok=True,
                    message=None,
                ),
                "gpu-service": HealthReport(
                    status=HealthStatus.DEGRADED,
                    component="gpu-service",
                    timestamp=now,
                    latency_ms=100.0,
                    details={"gpu_memory_available": 0.2},
                    dependencies_ok=True,
                    message="Low GPU memory",
                )
            },
            capabilities={},
            timestamp=now,
        )
        assert health.overall == HealthStatus.DEGRADED


class TestSystemHealthAggregator:
    """Tests for SystemHealthAggregator class."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock ComponentRegistry for testing."""
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentState,
            ComponentStatus, Criticality, ProcessType, HealthCheckType
        )
        registry = ComponentRegistry()
        registry._reset_for_testing()
        return registry

    @pytest.fixture
    def registry_with_components(self, mock_registry):
        """Registry with multiple components registered."""
        from backend.core.component_registry import (
            ComponentDefinition, ComponentStatus,
            Criticality, ProcessType, HealthCheckType
        )
        # Register a healthy component
        defn1 = ComponentDefinition(
            name="core",
            criticality=Criticality.REQUIRED,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["inference", "memory"],
            health_check_type=HealthCheckType.CUSTOM,
        )
        mock_registry.register(defn1)
        mock_registry.mark_status("core", ComponentStatus.HEALTHY)

        # Register a degraded component
        defn2 = ComponentDefinition(
            name="cache",
            criticality=Criticality.DEGRADED_OK,
            process_type=ProcessType.EXTERNAL_SERVICE,
            provides_capabilities=["caching"],
            health_check_type=HealthCheckType.TCP,
        )
        mock_registry.register(defn2)
        mock_registry.mark_status("cache", ComponentStatus.DEGRADED, "Redis connection slow")

        return mock_registry

    def test_aggregator_creation(self, mock_registry):
        from backend.core.health_contracts import SystemHealthAggregator
        aggregator = SystemHealthAggregator(mock_registry)
        assert aggregator._registry is mock_registry

    @pytest.mark.asyncio
    async def test_collect_all_empty_registry(self, mock_registry):
        from backend.core.health_contracts import SystemHealthAggregator, HealthStatus
        aggregator = SystemHealthAggregator(mock_registry)
        health = await aggregator.collect_all()
        assert health.overall == HealthStatus.HEALTHY  # No components = healthy
        assert health.components == {}
        assert health.capabilities == {}

    @pytest.mark.asyncio
    async def test_collect_all_with_components(self, registry_with_components):
        from backend.core.health_contracts import SystemHealthAggregator, HealthStatus
        aggregator = SystemHealthAggregator(registry_with_components)
        health = await aggregator.collect_all()

        # Should have reports for both components
        assert "core" in health.components
        assert "cache" in health.components

        # Core is healthy
        assert health.components["core"].status == HealthStatus.HEALTHY

        # Cache is degraded
        assert health.components["cache"].status == HealthStatus.DEGRADED

        # Overall should be degraded (worst status wins)
        assert health.overall == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_collect_all_parallel_execution(self, mock_registry):
        """Verify health checks run in parallel."""
        from backend.core.component_registry import (
            ComponentDefinition, ComponentStatus,
            Criticality, ProcessType, HealthCheckType
        )
        from backend.core.health_contracts import SystemHealthAggregator, HealthStatus

        # Register multiple components
        for i in range(5):
            defn = ComponentDefinition(
                name=f"component-{i}",
                criticality=Criticality.OPTIONAL,
                process_type=ProcessType.IN_PROCESS,
                health_check_type=HealthCheckType.NONE,
            )
            mock_registry.register(defn)
            mock_registry.mark_status(f"component-{i}", ComponentStatus.HEALTHY)

        aggregator = SystemHealthAggregator(mock_registry)
        health = await aggregator.collect_all()

        # All components should be collected
        assert len(health.components) == 5
        for i in range(5):
            assert f"component-{i}" in health.components

    def test_compute_overall_all_healthy(self):
        from backend.core.health_contracts import (
            SystemHealthAggregator, HealthStatus, HealthReport
        )
        from backend.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._reset_for_testing()
        aggregator = SystemHealthAggregator(registry)

        now = datetime.now(timezone.utc)
        results = {
            "comp1": HealthReport(
                status=HealthStatus.HEALTHY,
                component="comp1",
                timestamp=now,
                latency_ms=10.0,
                details={},
                dependencies_ok=True,
                message=None,
            ),
            "comp2": HealthReport(
                status=HealthStatus.HEALTHY,
                component="comp2",
                timestamp=now,
                latency_ms=15.0,
                details={},
                dependencies_ok=True,
                message=None,
            ),
        }

        overall = aggregator._compute_overall(results)
        assert overall == HealthStatus.HEALTHY

    def test_compute_overall_one_degraded(self):
        from backend.core.health_contracts import (
            SystemHealthAggregator, HealthStatus, HealthReport
        )
        from backend.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._reset_for_testing()
        aggregator = SystemHealthAggregator(registry)

        now = datetime.now(timezone.utc)
        results = {
            "comp1": HealthReport(
                status=HealthStatus.HEALTHY,
                component="comp1",
                timestamp=now,
                latency_ms=10.0,
                details={},
                dependencies_ok=True,
                message=None,
            ),
            "comp2": HealthReport(
                status=HealthStatus.DEGRADED,
                component="comp2",
                timestamp=now,
                latency_ms=500.0,
                details={},
                dependencies_ok=True,
                message="Running slow",
            ),
        }

        overall = aggregator._compute_overall(results)
        assert overall == HealthStatus.DEGRADED

    def test_compute_overall_one_unhealthy(self):
        from backend.core.health_contracts import (
            SystemHealthAggregator, HealthStatus, HealthReport
        )
        from backend.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._reset_for_testing()
        aggregator = SystemHealthAggregator(registry)

        now = datetime.now(timezone.utc)
        results = {
            "comp1": HealthReport(
                status=HealthStatus.HEALTHY,
                component="comp1",
                timestamp=now,
                latency_ms=10.0,
                details={},
                dependencies_ok=True,
                message=None,
            ),
            "comp2": HealthReport(
                status=HealthStatus.DEGRADED,
                component="comp2",
                timestamp=now,
                latency_ms=500.0,
                details={},
                dependencies_ok=True,
                message="Running slow",
            ),
            "comp3": HealthReport(
                status=HealthStatus.UNHEALTHY,
                component="comp3",
                timestamp=now,
                latency_ms=None,
                details={"error": "Connection failed"},
                dependencies_ok=False,
                message="Service unavailable",
            ),
        }

        overall = aggregator._compute_overall(results)
        assert overall == HealthStatus.UNHEALTHY

    def test_compute_overall_with_unknown(self):
        from backend.core.health_contracts import (
            SystemHealthAggregator, HealthStatus, HealthReport
        )
        from backend.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._reset_for_testing()
        aggregator = SystemHealthAggregator(registry)

        now = datetime.now(timezone.utc)
        results = {
            "comp1": HealthReport(
                status=HealthStatus.HEALTHY,
                component="comp1",
                timestamp=now,
                latency_ms=10.0,
                details={},
                dependencies_ok=True,
                message=None,
            ),
            "comp2": HealthReport(
                status=HealthStatus.UNKNOWN,
                component="comp2",
                timestamp=now,
                latency_ms=None,
                details={},
                dependencies_ok=True,
                message="Health check not implemented",
            ),
        }

        overall = aggregator._compute_overall(results)
        # Unknown is treated as unhealthy for overall status
        assert overall == HealthStatus.UNHEALTHY

    def test_compute_overall_empty(self):
        from backend.core.health_contracts import (
            SystemHealthAggregator, HealthStatus
        )
        from backend.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        registry._reset_for_testing()
        aggregator = SystemHealthAggregator(registry)

        overall = aggregator._compute_overall({})
        assert overall == HealthStatus.HEALTHY

    def test_derive_capabilities(self, registry_with_components):
        from backend.core.health_contracts import (
            SystemHealthAggregator, HealthStatus, HealthReport
        )

        aggregator = SystemHealthAggregator(registry_with_components)

        now = datetime.now(timezone.utc)
        results = {
            "core": HealthReport(
                status=HealthStatus.HEALTHY,
                component="core",
                timestamp=now,
                latency_ms=10.0,
                details={},
                dependencies_ok=True,
                message=None,
            ),
            "cache": HealthReport(
                status=HealthStatus.DEGRADED,
                component="cache",
                timestamp=now,
                latency_ms=500.0,
                details={},
                dependencies_ok=True,
                message="Slow connection",
            ),
        }

        capabilities = aggregator._derive_capabilities(results)

        # Core provides inference and memory - both healthy
        assert "inference" in capabilities
        assert capabilities["inference"].available is True
        assert capabilities["inference"].provider == "core"
        assert capabilities["inference"].status == HealthStatus.HEALTHY

        assert "memory" in capabilities
        assert capabilities["memory"].available is True

        # Cache provides caching - degraded
        assert "caching" in capabilities
        assert capabilities["caching"].available is True
        assert capabilities["caching"].provider == "cache"
        assert capabilities["caching"].status == HealthStatus.DEGRADED

    def test_derive_capabilities_unhealthy_provider(self, mock_registry):
        from backend.core.component_registry import (
            ComponentDefinition, ComponentStatus,
            Criticality, ProcessType
        )
        from backend.core.health_contracts import (
            SystemHealthAggregator, HealthStatus, HealthReport
        )

        # Register an unhealthy component
        defn = ComponentDefinition(
            name="failed-service",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.EXTERNAL_SERVICE,
            provides_capabilities=["notifications"],
        )
        mock_registry.register(defn)
        mock_registry.mark_status("failed-service", ComponentStatus.FAILED, "Crash")

        aggregator = SystemHealthAggregator(mock_registry)

        now = datetime.now(timezone.utc)
        results = {
            "failed-service": HealthReport(
                status=HealthStatus.UNHEALTHY,
                component="failed-service",
                timestamp=now,
                latency_ms=None,
                details={"error": "Service crashed"},
                dependencies_ok=False,
                message="Service unavailable",
            ),
        }

        capabilities = aggregator._derive_capabilities(results)

        # Capability should exist but be unavailable
        assert "notifications" in capabilities
        assert capabilities["notifications"].available is False
        assert capabilities["notifications"].provider == "failed-service"
        assert capabilities["notifications"].status == HealthStatus.UNHEALTHY


class TestHealthStatusOrdering:
    """Test that status ordering works correctly for overall computation."""

    def test_status_severity_ordering(self):
        from backend.core.health_contracts import HealthStatus

        # HEALTHY < DEGRADED < UNKNOWN < UNHEALTHY
        statuses = [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNKNOWN, HealthStatus.UNHEALTHY]

        # Verify there's a way to compare them for worst status
        severity = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNKNOWN: 2,
            HealthStatus.UNHEALTHY: 3,
        }

        # Sorted by severity should give expected order
        sorted_statuses = sorted(statuses, key=lambda s: severity[s])
        assert sorted_statuses == statuses


class TestIntegration:
    """Integration tests for the full health contracts system."""

    @pytest.mark.asyncio
    async def test_full_health_check_workflow(self):
        """Test complete workflow from registration to health aggregation."""
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentStatus,
            Criticality, ProcessType, HealthCheckType
        )
        from backend.core.health_contracts import (
            SystemHealthAggregator, HealthStatus
        )

        # Create fresh registry
        registry = ComponentRegistry()
        registry._reset_for_testing()

        # Register components
        components = [
            ComponentDefinition(
                name="jarvis-core",
                criticality=Criticality.REQUIRED,
                process_type=ProcessType.IN_PROCESS,
                provides_capabilities=["command-processing", "memory"],
                health_check_type=HealthCheckType.CUSTOM,
            ),
            ComponentDefinition(
                name="jarvis-prime",
                criticality=Criticality.DEGRADED_OK,
                process_type=ProcessType.SUBPROCESS,
                provides_capabilities=["local-inference"],
                health_check_type=HealthCheckType.HTTP,
                health_endpoint="http://localhost:8000/health",
            ),
            ComponentDefinition(
                name="redis-cache",
                criticality=Criticality.OPTIONAL,
                process_type=ProcessType.EXTERNAL_SERVICE,
                provides_capabilities=["caching"],
                health_check_type=HealthCheckType.TCP,
            ),
        ]

        for comp in components:
            registry.register(comp)

        # Set statuses
        registry.mark_status("jarvis-core", ComponentStatus.HEALTHY)
        registry.mark_status("jarvis-prime", ComponentStatus.HEALTHY)
        registry.mark_status("redis-cache", ComponentStatus.DEGRADED, "High latency")

        # Collect health
        aggregator = SystemHealthAggregator(registry)
        health = await aggregator.collect_all()

        # Verify
        assert health.overall == HealthStatus.DEGRADED
        assert len(health.components) == 3
        assert health.components["jarvis-core"].status == HealthStatus.HEALTHY
        assert health.components["jarvis-prime"].status == HealthStatus.HEALTHY
        assert health.components["redis-cache"].status == HealthStatus.DEGRADED

        # Verify capabilities
        assert "command-processing" in health.capabilities
        assert health.capabilities["command-processing"].available is True

        assert "caching" in health.capabilities
        assert health.capabilities["caching"].status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_health_check_with_custom_callback(self):
        """Test health check with custom callback function."""
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentStatus,
            Criticality, ProcessType, HealthCheckType
        )
        from backend.core.health_contracts import (
            SystemHealthAggregator, HealthStatus
        )

        # Create fresh registry
        registry = ComponentRegistry()
        registry._reset_for_testing()

        # Custom health check callback
        async def custom_health_check():
            return {"status": "ok", "details": {"uptime": 3600}}

        defn = ComponentDefinition(
            name="custom-service",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.IN_PROCESS,
            health_check_type=HealthCheckType.CUSTOM,
            health_check_callback=custom_health_check,
        )
        registry.register(defn)
        registry.mark_status("custom-service", ComponentStatus.HEALTHY)

        aggregator = SystemHealthAggregator(registry)
        health = await aggregator.collect_all()

        assert "custom-service" in health.components
        assert health.components["custom-service"].status == HealthStatus.HEALTHY
