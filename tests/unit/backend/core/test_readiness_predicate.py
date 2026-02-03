"""
Tests for readiness predicate.

This module tests the unified readiness predicate that determines
when JARVIS can be marked FULLY_READY.

Key logic:
- FULLY_READY iff (all critical components healthy) AND (optional components can be healthy, skipped, or errored)
- Critical components MUST be healthy to pass
- Optional components do NOT block readiness
"""

import pytest
from backend.core.readiness_predicate import (
    ReadinessPredicate,
    ReadinessResult,
    HEALTHY_STATUSES,
    ACCEPTABLE_OPTIONAL_STATUSES,
)


class TestReadinessPredicate:
    """Test readiness predicate logic."""

    def test_all_critical_healthy_optional_healthy(self):
        """FULLY_READY when all critical healthy and optional healthy."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "healthy",
            "reactor_core": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True
        assert result.blocking_components == []

    def test_all_critical_healthy_optional_skipped(self):
        """FULLY_READY when all critical healthy and optional skipped."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "skipped",
            "reactor_core": "skipped",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True
        assert result.blocking_components == []

    def test_critical_unhealthy_blocks_ready(self):
        """NOT FULLY_READY when critical component unhealthy."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "error",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "healthy",
            "reactor_core": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is False
        assert "backend" in result.blocking_components

    def test_critical_starting_blocks_ready(self):
        """NOT FULLY_READY when critical component still starting."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "starting",
            "loading_server": "healthy",
            "preflight": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is False
        assert "backend" in result.blocking_components

    def test_optional_error_does_not_block(self):
        """FULLY_READY even when optional component has error."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "error",
            "reactor_core": "error",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True
        assert "jarvis_prime" in result.degraded_components
        assert "reactor_core" in result.degraded_components

    def test_missing_critical_blocks(self):
        """NOT FULLY_READY when critical component missing from states."""
        predicate = ReadinessPredicate()
        component_states = {
            "loading_server": "healthy",
            "preflight": "healthy",
            # backend missing
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is False
        assert "backend" in result.blocking_components

    def test_result_has_readiness_message(self):
        """Result includes human-readable message."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "skipped",
        }
        result = predicate.evaluate(component_states)
        assert result.message is not None
        assert len(result.message) > 0


class TestReadinessResultDataclass:
    """Test ReadinessResult dataclass fields and behavior."""

    def test_result_contains_all_required_fields(self):
        """ReadinessResult has all required fields."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
        }
        result = predicate.evaluate(component_states)

        # Verify all required fields exist
        assert hasattr(result, "is_fully_ready")
        assert hasattr(result, "message")
        assert hasattr(result, "blocking_components")
        assert hasattr(result, "degraded_components")
        assert hasattr(result, "skipped_components")
        assert hasattr(result, "healthy_components")
        assert hasattr(result, "component_states")

    def test_skipped_components_tracked(self):
        """Skipped optional components are in skipped_components list."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "skipped",
            "reactor_core": "unavailable",
        }
        result = predicate.evaluate(component_states)
        assert "jarvis_prime" in result.skipped_components
        assert "reactor_core" in result.skipped_components

    def test_healthy_components_tracked(self):
        """Healthy components are in healthy_components list."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert "backend" in result.healthy_components
        assert "loading_server" in result.healthy_components
        assert "preflight" in result.healthy_components

    def test_component_states_preserved(self):
        """Original component states are preserved in result."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "starting",
            "preflight": "error",
        }
        result = predicate.evaluate(component_states)
        assert result.component_states == component_states


class TestIsComponentReady:
    """Test the is_component_ready helper method."""

    def test_healthy_status_is_ready(self):
        """Component with healthy status is ready."""
        predicate = ReadinessPredicate()
        assert predicate.is_component_ready("backend", "healthy") is True

    def test_complete_status_is_ready(self):
        """Component with complete status is ready."""
        predicate = ReadinessPredicate()
        assert predicate.is_component_ready("backend", "complete") is True

    def test_ready_status_is_ready(self):
        """Component with ready status is ready."""
        predicate = ReadinessPredicate()
        assert predicate.is_component_ready("backend", "ready") is True

    def test_running_status_is_ready(self):
        """Component with running status is ready."""
        predicate = ReadinessPredicate()
        assert predicate.is_component_ready("backend", "running") is True

    def test_error_status_not_ready(self):
        """Component with error status is not ready."""
        predicate = ReadinessPredicate()
        assert predicate.is_component_ready("backend", "error") is False

    def test_starting_status_not_ready(self):
        """Component with starting status is not ready."""
        predicate = ReadinessPredicate()
        assert predicate.is_component_ready("backend", "starting") is False

    def test_skipped_status_not_ready_for_critical(self):
        """Critical component with skipped status is not ready."""
        predicate = ReadinessPredicate()
        # backend is a critical component
        assert predicate.is_component_ready("backend", "skipped") is False


class TestHealthyStatuses:
    """Test HEALTHY_STATUSES constant."""

    def test_healthy_statuses_contains_healthy(self):
        """HEALTHY_STATUSES contains 'healthy'."""
        assert "healthy" in HEALTHY_STATUSES

    def test_healthy_statuses_contains_complete(self):
        """HEALTHY_STATUSES contains 'complete'."""
        assert "complete" in HEALTHY_STATUSES

    def test_healthy_statuses_contains_ready(self):
        """HEALTHY_STATUSES contains 'ready'."""
        assert "ready" in HEALTHY_STATUSES

    def test_healthy_statuses_contains_running(self):
        """HEALTHY_STATUSES contains 'running'."""
        assert "running" in HEALTHY_STATUSES


class TestAcceptableOptionalStatuses:
    """Test ACCEPTABLE_OPTIONAL_STATUSES constant."""

    def test_acceptable_optional_includes_healthy_statuses(self):
        """ACCEPTABLE_OPTIONAL_STATUSES includes all healthy statuses."""
        for status in HEALTHY_STATUSES:
            assert status in ACCEPTABLE_OPTIONAL_STATUSES

    def test_acceptable_optional_includes_skipped(self):
        """ACCEPTABLE_OPTIONAL_STATUSES includes 'skipped'."""
        assert "skipped" in ACCEPTABLE_OPTIONAL_STATUSES

    def test_acceptable_optional_includes_unavailable(self):
        """ACCEPTABLE_OPTIONAL_STATUSES includes 'unavailable'."""
        assert "unavailable" in ACCEPTABLE_OPTIONAL_STATUSES


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_component_states(self):
        """Empty component states means critical components are missing."""
        predicate = ReadinessPredicate()
        result = predicate.evaluate({})
        assert result.is_fully_ready is False
        # All critical components should be in blocking_components
        assert len(result.blocking_components) > 0

    def test_unknown_component_ignored(self):
        """Unknown components are not counted as blocking."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "unknown_component": "error",  # Unknown component with error
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True
        assert "unknown_component" not in result.blocking_components

    def test_case_insensitive_status(self):
        """Status comparison should handle case variations."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "HEALTHY",
            "loading_server": "Healthy",
            "preflight": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True

    def test_multiple_critical_blocking(self):
        """Multiple critical components can block simultaneously."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "error",
            "loading_server": "starting",
            "preflight": "degraded",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is False
        assert "backend" in result.blocking_components
        assert "loading_server" in result.blocking_components
        assert "preflight" in result.blocking_components

    def test_custom_config_critical_components(self):
        """ReadinessPredicate respects custom config for critical components."""
        from backend.core.readiness_config import ReadinessConfig

        custom_config = ReadinessConfig(
            critical_components=frozenset({"custom_critical"}),
            optional_components=frozenset({"custom_optional"}),
        )
        predicate = ReadinessPredicate(config=custom_config)
        component_states = {
            "custom_critical": "healthy",
            "custom_optional": "error",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True

    def test_degraded_message_when_optional_errors(self):
        """Message indicates degraded state when optional components have errors."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "error",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True
        # Message should mention degraded state
        assert "degraded" in result.message.lower() or len(result.degraded_components) > 0


class TestMessageContent:
    """Test the content and quality of readiness messages."""

    def test_ready_message_positive(self):
        """Message is positive when fully ready."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True
        # Message should indicate readiness
        assert "ready" in result.message.lower() or "healthy" in result.message.lower()

    def test_not_ready_message_lists_blockers(self):
        """Message lists blocking components when not ready."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "error",
            "loading_server": "healthy",
            "preflight": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is False
        # Message should mention the blocking component
        assert "backend" in result.message.lower() or len(result.blocking_components) > 0
