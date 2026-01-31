# backend/core/health_contracts.py
"""
HealthContracts - System health monitoring and aggregation.

This module provides:
- HealthStatus: Enum for component health states
- HealthReport: Detailed health report for a single component
- CapabilityHealth: Health status for a capability and its provider
- SystemHealth: Aggregated system-wide health status
- SystemHealthAggregator: Collects health from all components in parallel

Usage:
    from backend.core.health_contracts import (
        HealthStatus, HealthReport, SystemHealth, SystemHealthAggregator
    )
    from backend.core.component_registry import get_component_registry

    # Create aggregator with registry
    registry = get_component_registry()
    aggregator = SystemHealthAggregator(registry)

    # Collect system health
    health = await aggregator.collect_all()

    if health.overall == HealthStatus.UNHEALTHY:
        logger.error("System is unhealthy")

    # Check specific capability
    if health.capabilities.get("inference", {}).available:
        logger.info("Inference capability available")
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.component_registry import ComponentRegistry

logger = logging.getLogger("jarvis.health_contracts")


class HealthStatus(Enum):
    """Health status levels for components and system."""
    HEALTHY = "healthy"      # Component is fully operational
    DEGRADED = "degraded"    # Component is operational with reduced capability
    UNHEALTHY = "unhealthy"  # Component is not operational
    UNKNOWN = "unknown"      # Health status cannot be determined


# Severity ordering for status comparison (higher = worse)
_STATUS_SEVERITY: Dict[HealthStatus, int] = {
    HealthStatus.HEALTHY: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.UNKNOWN: 2,
    HealthStatus.UNHEALTHY: 3,
}


@dataclass
class HealthReport:
    """
    Detailed health report for a single component.

    Attributes:
        status: Current health status of the component
        component: Name of the component this report is for
        timestamp: When this report was generated
        latency_ms: Health check latency in milliseconds (None if check failed)
        details: Additional health check details (metrics, counters, etc.)
        dependencies_ok: Whether all dependencies are healthy
        message: Human-readable status message
        previous_status: Previous health status (for transition tracking)
        version: Component version (if available)
    """
    status: HealthStatus
    component: str
    timestamp: datetime
    latency_ms: Optional[float]
    details: Dict[str, Any]
    dependencies_ok: bool
    message: Optional[str]
    previous_status: Optional[HealthStatus] = None
    version: Optional[str] = None


@dataclass
class CapabilityHealth:
    """
    Health status for a capability and its provider.

    Attributes:
        available: Whether the capability is currently available
        provider: Name of the component providing this capability
        status: Health status of the capability provider
    """
    available: bool
    provider: Optional[str]
    status: HealthStatus


@dataclass
class SystemHealth:
    """
    Aggregated system-wide health status.

    Attributes:
        overall: Overall system health (worst status among components)
        components: Health reports for each component
        capabilities: Health status for each capability
        timestamp: When this system health was collected
    """
    overall: HealthStatus
    components: Dict[str, HealthReport]
    capabilities: Dict[str, CapabilityHealth]
    timestamp: datetime


class SystemHealthAggregator:
    """
    Collects health from all registered components in parallel.

    This class aggregates health information from all components
    registered in the ComponentRegistry, computing overall system
    health and capability availability.

    Usage:
        registry = get_component_registry()
        aggregator = SystemHealthAggregator(registry)
        health = await aggregator.collect_all()
    """

    def __init__(self, registry: 'ComponentRegistry'):
        """
        Initialize the aggregator with a component registry.

        Args:
            registry: The ComponentRegistry to collect health from
        """
        self._registry = registry

    async def collect_all(self) -> SystemHealth:
        """
        Collect health from all registered components in parallel.

        Returns:
            SystemHealth with aggregated health information
        """
        timestamp = datetime.now(timezone.utc)

        # Get all component states
        states = self._registry.all_states()

        if not states:
            # No components registered
            return SystemHealth(
                overall=HealthStatus.HEALTHY,
                components={},
                capabilities={},
                timestamp=timestamp,
            )

        # Collect health from all components in parallel
        tasks = [
            self._collect_component_health(state)
            for state in states
        ]

        reports = await asyncio.gather(*tasks, return_exceptions=True)

        # Build results dict, handling any exceptions
        results: Dict[str, HealthReport] = {}
        for state, report in zip(states, reports):
            if isinstance(report, Exception):
                # Component health check failed
                logger.warning(
                    f"Health check failed for {state.definition.name}: {report}"
                )
                results[state.definition.name] = HealthReport(
                    status=HealthStatus.UNKNOWN,
                    component=state.definition.name,
                    timestamp=timestamp,
                    latency_ms=None,
                    details={"error": str(report)},
                    dependencies_ok=False,
                    message=f"Health check failed: {report}",
                )
            else:
                results[state.definition.name] = report

        # Compute overall status and capabilities
        overall = self._compute_overall(results)
        capabilities = self._derive_capabilities(results)

        return SystemHealth(
            overall=overall,
            components=results,
            capabilities=capabilities,
            timestamp=timestamp,
        )

    async def _collect_component_health(self, state) -> HealthReport:
        """
        Collect health for a single component.

        Args:
            state: ComponentState to check

        Returns:
            HealthReport for the component
        """
        from backend.core.component_registry import ComponentStatus, HealthCheckType

        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc)
        component_name = state.definition.name

        # Map ComponentStatus to HealthStatus
        status_mapping = {
            ComponentStatus.PENDING: HealthStatus.UNKNOWN,
            ComponentStatus.STARTING: HealthStatus.UNKNOWN,
            ComponentStatus.HEALTHY: HealthStatus.HEALTHY,
            ComponentStatus.DEGRADED: HealthStatus.DEGRADED,
            ComponentStatus.FAILED: HealthStatus.UNHEALTHY,
            ComponentStatus.DISABLED: HealthStatus.UNHEALTHY,
        }

        health_status = status_mapping.get(state.status, HealthStatus.UNKNOWN)
        details: Dict[str, Any] = {}
        message: Optional[str] = state.failure_reason

        # If component has custom health check callback, run it
        if (
            state.definition.health_check_type == HealthCheckType.CUSTOM
            and state.definition.health_check_callback is not None
        ):
            try:
                callback = state.definition.health_check_callback
                if asyncio.iscoroutinefunction(callback):
                    check_result = await callback()
                else:
                    check_result = callback()

                if isinstance(check_result, dict):
                    details.update(check_result)
            except Exception as e:
                logger.warning(f"Custom health check failed for {component_name}: {e}")
                details["health_check_error"] = str(e)

        # Check dependencies
        dependencies_ok = self._check_dependencies_health(state)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        return HealthReport(
            status=health_status,
            component=component_name,
            timestamp=timestamp,
            latency_ms=latency_ms,
            details=details,
            dependencies_ok=dependencies_ok,
            message=message,
        )

    def _check_dependencies_health(self, state) -> bool:
        """
        Check if all dependencies of a component are healthy.

        Args:
            state: ComponentState to check dependencies for

        Returns:
            True if all dependencies are healthy or degraded, False otherwise
        """
        from backend.core.component_registry import ComponentStatus, Dependency

        for dep in state.definition.dependencies:
            # Handle both string and Dependency types
            if isinstance(dep, Dependency):
                dep_name = dep.component
                is_soft = dep.soft
            else:
                dep_name = dep
                is_soft = False

            # Skip soft dependencies
            if is_soft:
                continue

            # Check if dependency exists and is healthy
            if not self._registry.has(dep_name):
                return False

            dep_state = self._registry.get_state(dep_name)
            if dep_state.status not in (
                ComponentStatus.HEALTHY,
                ComponentStatus.DEGRADED
            ):
                return False

        return True

    def _compute_overall(self, results: Dict[str, HealthReport]) -> HealthStatus:
        """
        Compute overall system health from component reports.

        The overall status is the worst (highest severity) status
        among all components. UNKNOWN is treated as UNHEALTHY for
        overall status since we cannot verify system health.

        Args:
            results: Dict mapping component name to HealthReport

        Returns:
            The worst HealthStatus among all components.
            Returns UNHEALTHY if any component has UNKNOWN status.
        """
        if not results:
            return HealthStatus.HEALTHY

        worst_status = HealthStatus.HEALTHY
        worst_severity = _STATUS_SEVERITY[worst_status]

        for report in results.values():
            severity = _STATUS_SEVERITY.get(report.status, 3)  # Unknown defaults to unhealthy
            if severity > worst_severity:
                worst_severity = severity
                worst_status = report.status

        # UNKNOWN is treated as UNHEALTHY for overall status
        # since we cannot verify system health when any component is unknown
        if worst_status == HealthStatus.UNKNOWN:
            return HealthStatus.UNHEALTHY

        return worst_status

    def _derive_capabilities(
        self,
        results: Dict[str, HealthReport]
    ) -> Dict[str, CapabilityHealth]:
        """
        Derive capability health from component health reports.

        Maps capabilities to their providers and determines availability
        based on provider health status.

        Args:
            results: Dict mapping component name to HealthReport

        Returns:
            Dict mapping capability name to CapabilityHealth
        """
        capabilities: Dict[str, CapabilityHealth] = {}

        # Get all component definitions
        for state in self._registry.all_states():
            component_name = state.definition.name
            report = results.get(component_name)

            if report is None:
                continue

            # Map each provided capability to this component's health
            for capability in state.definition.provides_capabilities:
                # Capability is available if provider is healthy or degraded
                available = report.status in (
                    HealthStatus.HEALTHY,
                    HealthStatus.DEGRADED
                )

                capabilities[capability] = CapabilityHealth(
                    available=available,
                    provider=component_name,
                    status=report.status,
                )

        return capabilities


def get_health_aggregator(registry: Optional['ComponentRegistry'] = None) -> SystemHealthAggregator:
    """
    Factory function to create a SystemHealthAggregator.

    If no registry is provided, uses the global ComponentRegistry instance.

    Args:
        registry: Optional ComponentRegistry to use

    Returns:
        SystemHealthAggregator instance
    """
    if registry is None:
        from backend.core.component_registry import get_component_registry
        registry = get_component_registry()

    return SystemHealthAggregator(registry)
