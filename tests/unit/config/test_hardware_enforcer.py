"""
Tests for backend.config.hardware_enforcer module.

These tests verify that:
1. get_system_ram_gb() returns the correct system RAM in GB
2. enforce_hollow_client() enforces Hollow Client mode based on RAM threshold
3. Module-level enforcement runs on import
4. Idempotency is maintained when JARVIS_HOLLOW_CLIENT is already set
5. The RAM threshold is configurable via environment variable

Following TDD principles - tests written first, then implementation.
"""

import logging
import os
from unittest.mock import patch, MagicMock

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def clean_env():
    """
    Fixture to ensure clean environment for each test.

    Removes JARVIS_HOLLOW_CLIENT and JARVIS_HOLLOW_RAM_THRESHOLD_GB env vars
    and resets the hardware_enforcer module state.
    """
    # Store original values
    original_env = {}
    vars_to_clean = ["JARVIS_HOLLOW_CLIENT", "JARVIS_HOLLOW_RAM_THRESHOLD_GB"]
    for key in vars_to_clean:
        if key in os.environ:
            original_env[key] = os.environ.pop(key)

    yield

    # Restore original env vars
    for key in vars_to_clean:
        if key in original_env:
            os.environ[key] = original_env[key]
        elif key in os.environ:
            del os.environ[key]


@pytest.fixture
def caplog_info(caplog):
    """Fixture to capture info-level logs."""
    caplog.set_level(logging.INFO)
    return caplog


@pytest.fixture
def caplog_debug(caplog):
    """Fixture to capture debug-level logs."""
    caplog.set_level(logging.DEBUG)
    return caplog


# =============================================================================
# Test: get_system_ram_gb()
# =============================================================================


class TestGetSystemRamGb:
    """Tests for get_system_ram_gb function."""

    def test_get_system_ram_gb_returns_float(self, clean_env) -> None:
        """Verify get_system_ram_gb returns a positive float."""
        from backend.config.hardware_enforcer import get_system_ram_gb

        result = get_system_ram_gb()

        assert isinstance(result, float)
        assert result > 0, "System RAM should be positive"

    def test_get_system_ram_gb_uses_psutil(self, clean_env) -> None:
        """Verify get_system_ram_gb uses psutil.virtual_memory().total."""
        # Mock psutil to return a known value
        mock_vmem = MagicMock()
        mock_vmem.total = 16 * (1024**3)  # 16 GB in bytes

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import get_system_ram_gb

            result = get_system_ram_gb()

            mock_psutil.virtual_memory.assert_called_once()
            assert result == 16.0

    def test_get_system_ram_gb_calculates_correctly(self, clean_env) -> None:
        """Verify get_system_ram_gb calculates GB correctly from bytes."""
        # Test with 64 GB
        mock_vmem = MagicMock()
        mock_vmem.total = 64 * (1024**3)  # 64 GB in bytes

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import get_system_ram_gb

            result = get_system_ram_gb()

            assert result == 64.0


# =============================================================================
# Test: enforce_hollow_client() - Enforcement When RAM Below Threshold
# =============================================================================


class TestEnforceHollowClientBelowThreshold:
    """Tests for enforce_hollow_client when RAM is below threshold."""

    def test_enforce_hollow_client_when_ram_below_threshold(
        self, clean_env, caplog_info
    ) -> None:
        """Test: 16GB RAM with 32GB threshold -> True, env var set."""
        mock_vmem = MagicMock()
        mock_vmem.total = 16 * (1024**3)  # 16 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            result = enforce_hollow_client(source="test")

            assert result is True
            assert os.environ.get("JARVIS_HOLLOW_CLIENT") == "true"
            assert "[HardwareEnforcer] Hollow Client enforced" in caplog_info.text
            assert "source: test" in caplog_info.text
            assert "16.0GB < 32.0GB" in caplog_info.text

    def test_enforce_sets_env_var_to_true_string(self, clean_env) -> None:
        """Test that enforce_hollow_client sets JARVIS_HOLLOW_CLIENT to 'true' string."""
        mock_vmem = MagicMock()
        mock_vmem.total = 8 * (1024**3)  # 8 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            enforce_hollow_client(source="test_string")

            # Must be the string "true", not boolean True
            assert os.environ.get("JARVIS_HOLLOW_CLIENT") == "true"
            assert os.environ.get("JARVIS_HOLLOW_CLIENT") != True  # noqa: E712


# =============================================================================
# Test: enforce_hollow_client() - No Enforcement When RAM Above Threshold
# =============================================================================


class TestEnforceHollowClientAboveThreshold:
    """Tests for enforce_hollow_client when RAM is above threshold."""

    def test_no_enforcement_when_ram_above_threshold(
        self, clean_env, caplog_debug
    ) -> None:
        """Test: 64GB RAM with 32GB threshold -> False, env var NOT set."""
        mock_vmem = MagicMock()
        mock_vmem.total = 64 * (1024**3)  # 64 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            result = enforce_hollow_client(source="test")

            assert result is False
            assert "JARVIS_HOLLOW_CLIENT" not in os.environ
            assert "Full mode available" in caplog_debug.text
            assert "64.0GB >= 32.0GB" in caplog_debug.text

    def test_no_enforcement_when_ram_equals_threshold(
        self, clean_env, caplog_debug
    ) -> None:
        """Test: 32GB RAM with 32GB threshold -> False, env var NOT set."""
        mock_vmem = MagicMock()
        mock_vmem.total = 32 * (1024**3)  # 32 GB - exactly at threshold

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            result = enforce_hollow_client(source="test")

            assert result is False
            assert "JARVIS_HOLLOW_CLIENT" not in os.environ


# =============================================================================
# Test: enforce_hollow_client() - Configurable Threshold
# =============================================================================


class TestConfigurableThreshold:
    """Tests for configurable RAM threshold via environment variable."""

    def test_threshold_is_configurable(self, clean_env, caplog_info) -> None:
        """Test: JARVIS_HOLLOW_RAM_THRESHOLD_GB=64, 48GB RAM -> True (enforced)."""
        mock_vmem = MagicMock()
        mock_vmem.total = 48 * (1024**3)  # 48 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem
            os.environ["JARVIS_HOLLOW_RAM_THRESHOLD_GB"] = "64"

            from backend.config.hardware_enforcer import enforce_hollow_client

            result = enforce_hollow_client(source="test")

            assert result is True
            assert os.environ.get("JARVIS_HOLLOW_CLIENT") == "true"
            assert "48.0GB < 64.0GB" in caplog_info.text

    def test_threshold_higher_allows_machine_that_would_fail_default(
        self, clean_env
    ) -> None:
        """Test: Higher threshold (64GB) catches machines with 48GB RAM."""
        mock_vmem = MagicMock()
        mock_vmem.total = 48 * (1024**3)  # 48 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            # First test with default threshold (32GB) - should NOT enforce
            os.environ.pop("JARVIS_HOLLOW_RAM_THRESHOLD_GB", None)
            os.environ.pop("JARVIS_HOLLOW_CLIENT", None)

            result_default = enforce_hollow_client(source="default_threshold")
            assert result_default is False  # 48GB >= 32GB

            # Now test with 64GB threshold - should enforce
            os.environ["JARVIS_HOLLOW_RAM_THRESHOLD_GB"] = "64"
            os.environ.pop("JARVIS_HOLLOW_CLIENT", None)

            result_custom = enforce_hollow_client(source="custom_threshold")
            assert result_custom is True  # 48GB < 64GB

    def test_invalid_threshold_uses_default(self, clean_env) -> None:
        """Test: Invalid threshold value falls back to 32GB default."""
        mock_vmem = MagicMock()
        mock_vmem.total = 16 * (1024**3)  # 16 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem
            os.environ["JARVIS_HOLLOW_RAM_THRESHOLD_GB"] = "not_a_number"

            from backend.config.hardware_enforcer import enforce_hollow_client

            # Should still work with default threshold (32GB)
            result = enforce_hollow_client(source="test")
            assert result is True  # 16GB < 32GB (default)


# =============================================================================
# Test: Idempotency
# =============================================================================


class TestIdempotency:
    """Tests for idempotent behavior when JARVIS_HOLLOW_CLIENT is already set."""

    def test_idempotent_when_already_set(self, clean_env, caplog_info) -> None:
        """Test: JARVIS_HOLLOW_CLIENT already 'true' -> returns True immediately."""
        os.environ["JARVIS_HOLLOW_CLIENT"] = "true"

        mock_vmem = MagicMock()
        mock_vmem.total = 64 * (1024**3)  # 64 GB - would NOT trigger enforcement

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            # Clear log to check for new messages
            caplog_info.clear()

            result = enforce_hollow_client(source="test")

            assert result is True
            # Should NOT re-log or re-set
            assert "[HardwareEnforcer] Hollow Client enforced" not in caplog_info.text

    def test_idempotent_does_not_call_psutil(self, clean_env) -> None:
        """Test: When already set, psutil should not be called."""
        os.environ["JARVIS_HOLLOW_CLIENT"] = "true"

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            from backend.config.hardware_enforcer import enforce_hollow_client

            enforce_hollow_client(source="test")

            # psutil should NOT be called since we short-circuit
            mock_psutil.virtual_memory.assert_not_called()


# =============================================================================
# Test: Source Parameter
# =============================================================================


class TestSourceParameter:
    """Tests for the source parameter in enforce_hollow_client."""

    def test_default_source_is_unknown(self, clean_env, caplog_info) -> None:
        """Test: Default source parameter is 'unknown'."""
        mock_vmem = MagicMock()
        mock_vmem.total = 16 * (1024**3)  # 16 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            enforce_hollow_client()  # No source specified

            assert "source: unknown" in caplog_info.text

    def test_custom_source_appears_in_log(self, clean_env, caplog_info) -> None:
        """Test: Custom source parameter appears in log message."""
        mock_vmem = MagicMock()
        mock_vmem.total = 16 * (1024**3)  # 16 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            enforce_hollow_client(source="startup_lock_context")

            assert "source: startup_lock_context" in caplog_info.text


# =============================================================================
# Test: Module-Level Enforcement
# =============================================================================


class TestModuleLevelEnforcement:
    """Tests for module-level enforcement on import."""

    def test_module_import_calls_enforce(self, clean_env) -> None:
        """Test: Importing the module calls enforce_hollow_client(source='module_import')."""
        import sys

        mock_vmem = MagicMock()
        mock_vmem.total = 16 * (1024**3)  # 16 GB - should trigger enforcement

        # Remove the module from cache to force reimport
        modules_to_remove = [
            key for key in sys.modules if "hardware_enforcer" in key
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with patch("psutil.virtual_memory", return_value=mock_vmem):
            # Import the module - should trigger enforcement
            import backend.config.hardware_enforcer  # noqa: F401

            # Verify enforcement happened
            assert os.environ.get("JARVIS_HOLLOW_CLIENT") == "true"

    def test_module_import_with_sufficient_ram_does_not_set_env(
        self, clean_env
    ) -> None:
        """Test: Module import with sufficient RAM does not set JARVIS_HOLLOW_CLIENT."""
        import sys

        mock_vmem = MagicMock()
        mock_vmem.total = 64 * (1024**3)  # 64 GB - should NOT trigger enforcement

        # Remove the module from cache to force reimport
        modules_to_remove = [
            key for key in sys.modules if "hardware_enforcer" in key
        ]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with patch("psutil.virtual_memory", return_value=mock_vmem):
            # Import the module - should NOT trigger enforcement
            import backend.config.hardware_enforcer  # noqa: F401

            # Verify enforcement did NOT happen
            assert "JARVIS_HOLLOW_CLIENT" not in os.environ


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_low_ram(self, clean_env) -> None:
        """Test: Very low RAM (4GB) correctly triggers enforcement."""
        mock_vmem = MagicMock()
        mock_vmem.total = 4 * (1024**3)  # 4 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            result = enforce_hollow_client(source="test")

            assert result is True
            assert os.environ.get("JARVIS_HOLLOW_CLIENT") == "true"

    def test_very_high_ram(self, clean_env) -> None:
        """Test: Very high RAM (256GB) correctly skips enforcement."""
        mock_vmem = MagicMock()
        mock_vmem.total = 256 * (1024**3)  # 256 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import enforce_hollow_client

            result = enforce_hollow_client(source="test")

            assert result is False
            assert "JARVIS_HOLLOW_CLIENT" not in os.environ

    def test_fractional_ram(self, clean_env) -> None:
        """Test: Fractional RAM values are handled correctly."""
        mock_vmem = MagicMock()
        # 15.7 GB in bytes (not an exact power)
        mock_vmem.total = int(15.7 * (1024**3))

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem

            from backend.config.hardware_enforcer import get_system_ram_gb

            result = get_system_ram_gb()

            assert 15.6 < result < 15.8  # Approximately 15.7

    def test_zero_threshold_enforces_for_all(self, clean_env) -> None:
        """Test: Zero threshold (if allowed) would enforce for all machines."""
        mock_vmem = MagicMock()
        mock_vmem.total = 256 * (1024**3)  # 256 GB

        with patch("backend.config.hardware_enforcer.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vmem
            # Note: Implementation should handle or reject zero threshold
            os.environ["JARVIS_HOLLOW_RAM_THRESHOLD_GB"] = "0"

            from backend.config.hardware_enforcer import enforce_hollow_client

            # Zero threshold: all RAM values are >= 0, so should NOT enforce
            # unless implementation treats 0 as special
            result = enforce_hollow_client(source="test")

            # With 0 threshold, 256GB >= 0, so should NOT enforce
            assert result is False
