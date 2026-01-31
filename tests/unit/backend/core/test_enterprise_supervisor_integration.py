# tests/unit/backend/core/test_enterprise_supervisor_integration.py
"""
Tests for EnterpriseSupervisorIntegration.

Tests cover:
- is_enterprise_mode_available() function
- EnterpriseStartupConfig defaults
- EnterpriseIntegration initialization
- startup() with defaults
- startup() acquires lock
- startup() displays summary
- shutdown() releases resources
- health_check() returns status
- enterprise_startup() convenience function
- get_enterprise_registry() returns registry
"""
from __future__ import annotations

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import tempfile
import shutil

from backend.core.component_registry import (
    ComponentDefinition,
    ComponentRegistry,
    ComponentStatus,
    Criticality,
    ProcessType,
)
from backend.core.enterprise_startup_orchestrator import (
    EnterpriseStartupOrchestrator,
    StartupResult,
    ComponentResult,
    StartupEvent,
)
from backend.core.startup_lock import StartupLock
from backend.core.startup_context import StartupContext


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_state_dir():
    """Create a temporary state directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_startup_lock(temp_state_dir):
    """Create a mock startup lock."""
    lock = StartupLock(state_dir=temp_state_dir)
    return lock


@pytest.fixture
def registry():
    """Create a fresh ComponentRegistry for each test."""
    reg = ComponentRegistry()
    reg._reset_for_testing()
    return reg


@pytest.fixture
def simple_components():
    """Simple component definitions for testing."""
    return [
        ComponentDefinition(
            name="core",
            criticality=Criticality.REQUIRED,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["core"],
            dependencies=[],
            startup_timeout=5.0,
        ),
        ComponentDefinition(
            name="database",
            criticality=Criticality.DEGRADED_OK,
            process_type=ProcessType.EXTERNAL_SERVICE,
            provides_capabilities=["persistence"],
            dependencies=["core"],
            startup_timeout=5.0,
        ),
    ]


@pytest.fixture
def registry_with_components(registry, simple_components):
    """Registry populated with simple components."""
    for comp in simple_components:
        registry.register(comp)
    return registry


# =============================================================================
# Test is_enterprise_mode_available
# =============================================================================

class TestIsEnterpriseModeAvailable:
    """Tests for is_enterprise_mode_available function."""

    def test_returns_true_when_modules_available(self):
        """Test that function returns True when all modules are available."""
        from backend.core.enterprise_supervisor_integration import (
            is_enterprise_mode_available
        )
        # Since all modules are installed in test environment, should return True
        assert is_enterprise_mode_available() is True

    def test_is_bool(self):
        """Test that function returns a boolean."""
        from backend.core.enterprise_supervisor_integration import (
            is_enterprise_mode_available
        )
        result = is_enterprise_mode_available()
        assert isinstance(result, bool)


# =============================================================================
# Test EnterpriseStartupConfig
# =============================================================================

class TestEnterpriseStartupConfig:
    """Tests for EnterpriseStartupConfig dataclass."""

    def test_default_values(self):
        """Test default values for config."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseStartupConfig
        )
        config = EnterpriseStartupConfig()

        assert config.register_defaults is True
        assert config.display_summary is True
        assert config.use_startup_lock is True
        assert config.enable_subprocess_manager is True
        assert config.enable_capability_router is True
        assert config.component_starters == {}
        assert config.event_handlers == {}

    def test_custom_values(self):
        """Test setting custom config values."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseStartupConfig
        )

        async def custom_starter():
            return True

        config = EnterpriseStartupConfig(
            register_defaults=False,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
            component_starters={"test": custom_starter},
        )

        assert config.register_defaults is False
        assert config.display_summary is False
        assert config.use_startup_lock is False
        assert config.enable_subprocess_manager is False
        assert config.enable_capability_router is False
        assert "test" in config.component_starters


# =============================================================================
# Test EnterpriseIntegration Initialization
# =============================================================================

class TestEnterpriseIntegrationInit:
    """Tests for EnterpriseIntegration initialization."""

    def test_basic_init(self):
        """Test basic initialization without config."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration
        )
        integration = EnterpriseIntegration()

        assert integration.config is not None
        assert integration._lock is None
        assert integration._orchestrator is None
        assert integration._subprocess_manager is None
        assert integration._capability_router is None
        assert integration._startup_result is None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )
        config = EnterpriseStartupConfig(
            register_defaults=False,
            use_startup_lock=False,
        )
        integration = EnterpriseIntegration(config)

        assert integration.config is config
        assert integration.config.register_defaults is False
        assert integration.config.use_startup_lock is False


# =============================================================================
# Test startup() method
# =============================================================================

class TestEnterpriseIntegrationStartup:
    """Tests for EnterpriseIntegration.startup() method."""

    @pytest.mark.asyncio
    async def test_startup_with_defaults(self, temp_state_dir):
        """Test startup with default configuration."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,  # Suppress output during test
            use_startup_lock=False,  # Don't use lock in test
            enable_subprocess_manager=False,  # Skip subprocess manager
            enable_capability_router=False,  # Skip capability router
        )

        integration = EnterpriseIntegration(config)
        result = await integration.startup()

        assert result is not None
        assert isinstance(result.success, bool)
        assert result.total_time >= 0

        await integration.shutdown()

    @pytest.mark.asyncio
    async def test_startup_acquires_lock(self, temp_state_dir):
        """Test that startup acquires the startup lock."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        config = EnterpriseStartupConfig(
            register_defaults=False,
            display_summary=False,
            use_startup_lock=True,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        # Patch get_startup_lock to use our temp dir
        with patch('backend.core.enterprise_supervisor_integration.get_startup_lock') as mock_get_lock:
            mock_lock = MagicMock()
            mock_lock.acquire.return_value = True
            mock_get_lock.return_value = mock_lock

            integration = EnterpriseIntegration(config)
            result = await integration.startup()

            # Lock should have been acquired
            mock_get_lock.assert_called_once()
            mock_lock.acquire.assert_called_once()

            await integration.shutdown()

    @pytest.mark.asyncio
    async def test_startup_fails_if_lock_not_acquired(self, temp_state_dir):
        """Test that startup fails if lock cannot be acquired."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        config = EnterpriseStartupConfig(
            register_defaults=False,
            display_summary=False,
            use_startup_lock=True,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        # Patch get_startup_lock to return a lock that fails to acquire
        with patch('backend.core.enterprise_supervisor_integration.get_startup_lock') as mock_get_lock:
            mock_lock = MagicMock()
            mock_lock.acquire.return_value = False
            mock_get_lock.return_value = mock_lock

            integration = EnterpriseIntegration(config)
            result = await integration.startup()

            assert result.success is False
            assert result.overall_status == "FAILED"

    @pytest.mark.asyncio
    async def test_startup_registers_custom_starters(self, temp_state_dir):
        """Test that startup registers custom component starters."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        starter_called = [False]

        async def custom_starter():
            starter_called[0] = True
            return True

        config = EnterpriseStartupConfig(
            register_defaults=False,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
            component_starters={"test-component": custom_starter},
        )

        integration = EnterpriseIntegration(config)

        # Starter won't be called because component isn't registered
        # but we can verify it was passed to the orchestrator
        result = await integration.startup()

        assert integration._orchestrator is not None
        assert "test-component" in integration._orchestrator._component_starters

        await integration.shutdown()


# =============================================================================
# Test shutdown() method
# =============================================================================

class TestEnterpriseIntegrationShutdown:
    """Tests for EnterpriseIntegration.shutdown() method."""

    @pytest.mark.asyncio
    async def test_shutdown_releases_lock(self, temp_state_dir):
        """Test that shutdown releases the startup lock."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        config = EnterpriseStartupConfig(
            register_defaults=False,
            display_summary=False,
            use_startup_lock=True,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        with patch('backend.core.enterprise_supervisor_integration.get_startup_lock') as mock_get_lock:
            mock_lock = MagicMock()
            mock_lock.acquire.return_value = True
            mock_get_lock.return_value = mock_lock

            integration = EnterpriseIntegration(config)
            await integration.startup()
            await integration.shutdown()

            # Lock should have been released
            mock_lock.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_startup_is_safe(self):
        """Test that shutdown is safe even without prior startup."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
        )

        integration = EnterpriseIntegration()
        # Should not raise
        await integration.shutdown()


# =============================================================================
# Test health_check() method
# =============================================================================

class TestEnterpriseIntegrationHealthCheck:
    """Tests for EnterpriseIntegration.health_check() method."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self, temp_state_dir):
        """Test that health_check returns status dict."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        integration = EnterpriseIntegration(config)
        await integration.startup()

        health = await integration.health_check()

        assert health is not None
        assert "status" in health
        assert "components" in health
        assert "capabilities" in health

        await integration.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_before_startup(self):
        """Test health_check before startup returns not_initialized."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
        )

        integration = EnterpriseIntegration()
        health = await integration.health_check()

        assert health == {"status": "not_initialized"}


# =============================================================================
# Test registry property
# =============================================================================

class TestEnterpriseIntegrationRegistry:
    """Tests for EnterpriseIntegration.registry property."""

    @pytest.mark.asyncio
    async def test_registry_returns_component_registry(self, temp_state_dir):
        """Test that registry property returns ComponentRegistry after startup."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )
        from backend.core.component_registry import ComponentRegistry

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        integration = EnterpriseIntegration(config)
        await integration.startup()

        assert integration.registry is not None
        assert isinstance(integration.registry, ComponentRegistry)

        await integration.shutdown()

    def test_registry_returns_none_before_startup(self):
        """Test that registry returns None before startup."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
        )

        integration = EnterpriseIntegration()
        assert integration.registry is None


# =============================================================================
# Test convenience functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_enterprise_startup(self, temp_state_dir):
        """Test enterprise_startup convenience function."""
        from backend.core.enterprise_supervisor_integration import (
            enterprise_startup,
            enterprise_shutdown,
            EnterpriseStartupConfig,
        )

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        result = await enterprise_startup(config)

        assert result is not None
        assert isinstance(result.success, bool)

        await enterprise_shutdown()

    @pytest.mark.asyncio
    async def test_get_enterprise_registry(self, temp_state_dir):
        """Test get_enterprise_registry convenience function."""
        from backend.core.enterprise_supervisor_integration import (
            enterprise_startup,
            enterprise_shutdown,
            get_enterprise_registry,
            EnterpriseStartupConfig,
        )
        from backend.core.component_registry import ComponentRegistry

        # Before startup, should return None
        assert get_enterprise_registry() is None

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        await enterprise_startup(config)

        # After startup, should return registry
        registry = get_enterprise_registry()
        assert registry is not None
        assert isinstance(registry, ComponentRegistry)

        await enterprise_shutdown()

    @pytest.mark.asyncio
    async def test_get_enterprise_integration(self, temp_state_dir):
        """Test get_enterprise_integration convenience function."""
        from backend.core.enterprise_supervisor_integration import (
            enterprise_startup,
            enterprise_shutdown,
            get_enterprise_integration,
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        # Before startup, should return None
        assert get_enterprise_integration() is None

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        await enterprise_startup(config)

        # After startup, should return integration
        integration = get_enterprise_integration()
        assert integration is not None
        assert isinstance(integration, EnterpriseIntegration)

        await enterprise_shutdown()

    @pytest.mark.asyncio
    async def test_get_enterprise_router(self, temp_state_dir):
        """Test get_enterprise_router convenience function."""
        from backend.core.enterprise_supervisor_integration import (
            enterprise_startup,
            enterprise_shutdown,
            get_enterprise_router,
            EnterpriseStartupConfig,
        )
        from backend.core.capability_router import CapabilityRouter

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=True,  # Enable router
        )

        await enterprise_startup(config)

        router = get_enterprise_router()
        assert router is not None
        assert isinstance(router, CapabilityRouter)

        await enterprise_shutdown()


# =============================================================================
# Test capability router integration
# =============================================================================

class TestCapabilityRouterIntegration:
    """Tests for capability router integration."""

    @pytest.mark.asyncio
    async def test_capability_router_created(self, temp_state_dir):
        """Test that capability router is created when enabled."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )
        from backend.core.capability_router import CapabilityRouter

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=True,
        )

        integration = EnterpriseIntegration(config)
        await integration.startup()

        assert integration.capability_router is not None
        assert isinstance(integration.capability_router, CapabilityRouter)

        await integration.shutdown()

    @pytest.mark.asyncio
    async def test_capability_router_not_created_when_disabled(self, temp_state_dir):
        """Test that capability router is not created when disabled."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        integration = EnterpriseIntegration(config)
        await integration.startup()

        assert integration.capability_router is None

        await integration.shutdown()


# =============================================================================
# Test subprocess manager integration
# =============================================================================

class TestSubprocessManagerIntegration:
    """Tests for subprocess manager integration."""

    @pytest.mark.asyncio
    async def test_subprocess_manager_created(self, temp_state_dir):
        """Test that subprocess manager is created when enabled."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )
        from backend.core.subprocess_manager import SubprocessManager

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=True,
            enable_capability_router=False,
        )

        integration = EnterpriseIntegration(config)
        await integration.startup()

        assert integration.subprocess_manager is not None
        assert isinstance(integration.subprocess_manager, SubprocessManager)

        await integration.shutdown()

    @pytest.mark.asyncio
    async def test_subprocess_manager_not_created_when_disabled(self, temp_state_dir):
        """Test that subprocess manager is not created when disabled."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        integration = EnterpriseIntegration(config)
        await integration.startup()

        assert integration.subprocess_manager is None

        await integration.shutdown()


# =============================================================================
# Test event handler registration
# =============================================================================

class TestEventHandlerRegistration:
    """Tests for event handler registration."""

    @pytest.mark.asyncio
    async def test_event_handlers_registered(self, temp_state_dir):
        """Test that event handlers are registered with orchestrator."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        events_received = []

        def handler(event, data):
            events_received.append(event.value)

        config = EnterpriseStartupConfig(
            register_defaults=True,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
            event_handlers={"STARTUP_BEGUN": [handler]},
        )

        integration = EnterpriseIntegration(config)
        await integration.startup()

        # Handler should have been called
        assert "startup_begun" in events_received

        await integration.shutdown()


# =============================================================================
# Test error handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in integration."""

    @pytest.mark.asyncio
    async def test_handles_startup_exception_gracefully(self):
        """Test that startup exceptions are handled gracefully."""
        from backend.core.enterprise_supervisor_integration import (
            EnterpriseIntegration,
            EnterpriseStartupConfig,
        )

        config = EnterpriseStartupConfig(
            register_defaults=False,
            display_summary=False,
            use_startup_lock=False,
            enable_subprocess_manager=False,
            enable_capability_router=False,
        )

        integration = EnterpriseIntegration(config)

        # Mock create_enterprise_orchestrator to raise
        with patch('backend.core.enterprise_supervisor_integration.create_enterprise_orchestrator') as mock_create:
            mock_create.side_effect = RuntimeError("Orchestrator creation failed")

            result = await integration.startup()

            assert result.success is False
            assert result.overall_status == "FAILED"


# =============================================================================
# Test module exports
# =============================================================================

class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports_defined(self):
        """Test that __all__ exports are defined."""
        from backend.core import enterprise_supervisor_integration

        expected_exports = [
            'is_enterprise_mode_available',
            'enterprise_startup',
            'enterprise_shutdown',
            'get_enterprise_integration',
            'get_enterprise_registry',
            'get_enterprise_router',
            'EnterpriseStartupConfig',
            'EnterpriseIntegration',
        ]

        for name in expected_exports:
            assert hasattr(enterprise_supervisor_integration, name), f"Missing export: {name}"
