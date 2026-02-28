"""Tests for readiness configuration module."""
import os
import pytest
from unittest.mock import patch


class TestReadinessConfig:
    """Test readiness configuration."""

    def test_critical_components_defined(self):
        """Critical components must be defined."""
        from backend.core.readiness_config import get_readiness_config
        config = get_readiness_config()
        assert "backend" in config.critical_components
        assert "loading_server" in config.critical_components

    def test_optional_components_defined(self):
        """Optional components must be defined."""
        from backend.core.readiness_config import get_readiness_config
        config = get_readiness_config()
        assert "jarvis_prime" in config.optional_components
        assert "reactor_core" in config.optional_components

    def test_component_criticality_lookup(self):
        """Can look up criticality by component name."""
        from backend.core.readiness_config import (
            get_readiness_config,
            ComponentCriticality,
        )
        config = get_readiness_config()
        assert config.get_criticality("backend") == ComponentCriticality.CRITICAL
        assert config.get_criticality("jarvis_prime") == ComponentCriticality.OPTIONAL
        assert config.get_criticality("unknown") == ComponentCriticality.UNKNOWN

    def test_status_display_mapping(self):
        """Status display mapping is correct."""
        from backend.core.readiness_config import get_readiness_config
        config = get_readiness_config()
        assert config.status_to_display("healthy") == "HEAL"
        assert config.status_to_display("starting") == "STAR"
        assert config.status_to_display("pending") == "PEND"
        assert config.status_to_display("stopped") == "STOP"
        assert config.status_to_display("skipped") == "SKIP"
        assert config.status_to_display("unavailable") == "UNAV"

    def test_skipped_not_equals_stopped(self):
        """Skipped status is distinct from stopped."""
        from backend.core.readiness_config import get_readiness_config
        config = get_readiness_config()
        skipped_display = config.status_to_display("skipped")
        stopped_display = config.status_to_display("stopped")
        assert skipped_display != stopped_display
        assert skipped_display == "SKIP"
        assert stopped_display == "STOP"


class TestComponentCriticalityEnum:
    """Test ComponentCriticality enum."""

    def test_enum_values(self):
        """ComponentCriticality enum has expected values."""
        from backend.core.readiness_config import ComponentCriticality
        assert ComponentCriticality.CRITICAL.value == "critical"
        assert ComponentCriticality.OPTIONAL.value == "optional"
        assert ComponentCriticality.UNKNOWN.value == "unknown"


class TestComponentStatusEnum:
    """Test ComponentStatus enum."""

    def test_enum_values(self):
        """ComponentStatus enum has expected values."""
        from backend.core.readiness_config import ComponentStatus
        assert ComponentStatus.PENDING.value == "pending"
        assert ComponentStatus.STARTING.value == "starting"
        assert ComponentStatus.HEALTHY.value == "healthy"
        assert ComponentStatus.DEGRADED.value == "degraded"
        assert ComponentStatus.ERROR.value == "error"
        assert ComponentStatus.STOPPED.value == "stopped"
        assert ComponentStatus.SKIPPED.value == "skipped"
        assert ComponentStatus.UNAVAILABLE.value == "unavailable"


class TestStatusDisplayMap:
    """Test STATUS_DISPLAY_MAP constant."""

    def test_all_statuses_have_display_codes(self):
        """All ComponentStatus values have display mappings."""
        from backend.core.readiness_config import (
            ComponentStatus,
            STATUS_DISPLAY_MAP,
        )
        for status in ComponentStatus:
            assert status.value in STATUS_DISPLAY_MAP, (
                f"Missing display mapping for status: {status.value}"
            )

    def test_display_codes_are_4_chars(self):
        """All display codes are exactly 4 characters."""
        from backend.core.readiness_config import STATUS_DISPLAY_MAP
        for status, code in STATUS_DISPLAY_MAP.items():
            assert len(code) == 4, (
                f"Display code for '{status}' is not 4 chars: '{code}'"
            )

    def test_display_codes_are_unique(self):
        """All display codes are unique."""
        from backend.core.readiness_config import STATUS_DISPLAY_MAP
        codes = list(STATUS_DISPLAY_MAP.values())
        assert len(codes) == len(set(codes)), "Duplicate display codes found"


class TestDashboardStatusMap:
    """Test DASHBOARD_STATUS_MAP constant."""

    def test_skipped_maps_to_skipped_not_stopped(self):
        """CRITICAL: 'skipped' must map to 'skipped', NOT 'stopped'."""
        from backend.core.readiness_config import DASHBOARD_STATUS_MAP
        assert DASHBOARD_STATUS_MAP["skipped"] == "skipped"
        assert DASHBOARD_STATUS_MAP["stopped"] == "stopped"
        assert DASHBOARD_STATUS_MAP["skipped"] != DASHBOARD_STATUS_MAP["stopped"]

    def test_all_statuses_have_dashboard_mappings(self):
        """All ComponentStatus values have dashboard mappings."""
        from backend.core.readiness_config import (
            ComponentStatus,
            DASHBOARD_STATUS_MAP,
        )
        for status in ComponentStatus:
            assert status.value in DASHBOARD_STATUS_MAP, (
                f"Missing dashboard mapping for status: {status.value}"
            )


class TestReadinessConfigDataclass:
    """Test ReadinessConfig dataclass."""

    def test_default_values(self):
        """ReadinessConfig has expected default values."""
        from backend.core.readiness_config import ReadinessConfig
        config = ReadinessConfig()
        assert isinstance(config.critical_components, frozenset)
        assert isinstance(config.optional_components, frozenset)
        assert config.verification_timeout > 0
        assert config.unhealthy_threshold_failures > 0
        assert config.unhealthy_threshold_seconds > 0
        assert config.revocation_cooldown_seconds >= 0

    def test_preflight_in_critical_components(self):
        """Preflight is a critical component."""
        from backend.core.readiness_config import ReadinessConfig
        config = ReadinessConfig()
        assert "preflight" in config.critical_components

    def test_get_criticality_method(self):
        """get_criticality returns correct criticality."""
        from backend.core.readiness_config import (
            ReadinessConfig,
            ComponentCriticality,
        )
        config = ReadinessConfig()
        # Critical
        assert config.get_criticality("backend") == ComponentCriticality.CRITICAL
        assert config.get_criticality("loading_server") == ComponentCriticality.CRITICAL
        # Optional
        assert config.get_criticality("jarvis_prime") == ComponentCriticality.OPTIONAL
        assert config.get_criticality("reactor_core") == ComponentCriticality.OPTIONAL
        # Unknown
        assert config.get_criticality("some_random_component") == ComponentCriticality.UNKNOWN

    def test_status_to_display_static_method(self):
        """status_to_display returns correct codes."""
        from backend.core.readiness_config import ReadinessConfig
        assert ReadinessConfig.status_to_display("healthy") == "HEAL"
        assert ReadinessConfig.status_to_display("skipped") == "SKIP"
        assert ReadinessConfig.status_to_display("stopped") == "STOP"
        # Unknown status returns "????"
        assert ReadinessConfig.status_to_display("unknown_status") == "????"

    def test_status_to_dashboard_static_method(self):
        """status_to_dashboard returns correct dashboard statuses."""
        from backend.core.readiness_config import ReadinessConfig
        assert ReadinessConfig.status_to_dashboard("healthy") == "healthy"
        assert ReadinessConfig.status_to_dashboard("skipped") == "skipped"
        assert ReadinessConfig.status_to_dashboard("stopped") == "stopped"
        # Unknown status returns "unknown"
        assert ReadinessConfig.status_to_dashboard("unknown_status") == "unknown"


class TestSingletonPattern:
    """Test get_readiness_config singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """get_readiness_config returns the same instance."""
        from backend.core.readiness_config import get_readiness_config, _reset_config
        # Reset to ensure clean state
        _reset_config()
        config1 = get_readiness_config()
        config2 = get_readiness_config()
        assert config1 is config2

    def test_reset_config_clears_singleton(self):
        """_reset_config clears the singleton for testing."""
        from backend.core.readiness_config import get_readiness_config, _reset_config
        config1 = get_readiness_config()
        _reset_config()
        config2 = get_readiness_config()
        assert config1 is not config2


class TestEnvironmentVariableConfiguration:
    """Test configuration from environment variables."""

    def test_verification_timeout_from_env(self):
        """verification_timeout can be configured via env var."""
        from backend.core.readiness_config import ReadinessConfig, _reset_config
        _reset_config()
        with patch.dict(os.environ, {"Ironcliw_VERIFICATION_TIMEOUT": "120"}):
            config = ReadinessConfig()
            assert config.verification_timeout == 120.0

    def test_unhealthy_threshold_failures_from_env(self):
        """unhealthy_threshold_failures can be configured via env var."""
        from backend.core.readiness_config import ReadinessConfig, _reset_config
        _reset_config()
        with patch.dict(os.environ, {"Ironcliw_UNHEALTHY_THRESHOLD_FAILURES": "5"}):
            config = ReadinessConfig()
            assert config.unhealthy_threshold_failures == 5

    def test_unhealthy_threshold_seconds_from_env(self):
        """unhealthy_threshold_seconds can be configured via env var."""
        from backend.core.readiness_config import ReadinessConfig, _reset_config
        _reset_config()
        with patch.dict(os.environ, {"Ironcliw_UNHEALTHY_THRESHOLD_SECONDS": "60"}):
            config = ReadinessConfig()
            assert config.unhealthy_threshold_seconds == 60.0

    def test_revocation_cooldown_from_env(self):
        """revocation_cooldown_seconds can be configured via env var."""
        from backend.core.readiness_config import ReadinessConfig, _reset_config
        _reset_config()
        with patch.dict(os.environ, {"Ironcliw_REVOCATION_COOLDOWN_SECONDS": "10"}):
            config = ReadinessConfig()
            assert config.revocation_cooldown_seconds == 10.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_component_name_returns_unknown(self):
        """Empty component name returns UNKNOWN criticality."""
        from backend.core.readiness_config import (
            get_readiness_config,
            ComponentCriticality,
        )
        config = get_readiness_config()
        assert config.get_criticality("") == ComponentCriticality.UNKNOWN

    def test_case_insensitive_criticality_lookup(self):
        """Criticality lookup should be case-insensitive."""
        from backend.core.readiness_config import (
            get_readiness_config,
            ComponentCriticality,
        )
        config = get_readiness_config()
        # Should match regardless of case
        assert config.get_criticality("Backend") == ComponentCriticality.CRITICAL
        assert config.get_criticality("BACKEND") == ComponentCriticality.CRITICAL
        assert config.get_criticality("backend") == ComponentCriticality.CRITICAL

    def test_status_to_display_case_insensitive(self):
        """status_to_display should be case-insensitive."""
        from backend.core.readiness_config import ReadinessConfig
        assert ReadinessConfig.status_to_display("HEALTHY") == "HEAL"
        assert ReadinessConfig.status_to_display("Healthy") == "HEAL"
        assert ReadinessConfig.status_to_display("healthy") == "HEAL"

    def test_status_to_dashboard_case_insensitive(self):
        """status_to_dashboard should be case-insensitive."""
        from backend.core.readiness_config import ReadinessConfig
        assert ReadinessConfig.status_to_dashboard("SKIPPED") == "skipped"
        assert ReadinessConfig.status_to_dashboard("Skipped") == "skipped"
        assert ReadinessConfig.status_to_dashboard("skipped") == "skipped"
