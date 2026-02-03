"""
Tests for startup resilience utilities.

Tests cover:
- Health check factory functions
- Health probe factory functions
- Background recovery factory functions
- StartupResilience coordinator
- Integration scenarios
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.resilience.startup import (
    StartupResilienceConfig,
    StartupResilience,
    create_docker_health_check,
    create_ollama_health_check,
    create_invincible_node_health_check,
    create_docker_health_probe,
    create_ollama_health_probe,
    create_invincible_node_health_probe,
    create_docker_recovery,
    create_ollama_recovery,
    create_invincible_node_recovery,
    create_local_llm_capability_upgrade,
)
from backend.core.resilience.types import CapabilityState, RecoveryState


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def config():
    """Create test configuration with shorter timeouts."""
    return StartupResilienceConfig(
        docker_check_timeout=1.0,
        docker_cache_ttl=5.0,
        docker_unhealthy_threshold=2,
        docker_recovery_base_delay=0.1,
        docker_recovery_max_delay=1.0,
        docker_recovery_max_attempts=3,
        ollama_check_timeout=1.0,
        ollama_cache_ttl=5.0,
        ollama_unhealthy_threshold=2,
        ollama_recovery_base_delay=0.1,
        ollama_recovery_max_delay=1.0,
        ollama_recovery_max_attempts=3,
        invincible_node_check_timeout=1.0,
        invincible_node_cache_ttl=5.0,
        invincible_node_unhealthy_threshold=2,
        invincible_node_recovery_base_delay=0.1,
        invincible_node_recovery_max_delay=1.0,
        invincible_node_recovery_max_attempts=3,
        local_llm_upgrade_interval=1.0,
    )


@pytest.fixture
def mock_vm_manager():
    """Create a mock GCP VM manager."""
    manager = AsyncMock()
    manager.is_static_vm_mode = True
    manager.check_static_vm_health = AsyncMock(return_value={"healthy": True})
    manager.ensure_static_vm_ready = AsyncMock(return_value=(True, "10.0.0.1", "RUNNING"))
    return manager


# =============================================================================
# HEALTH CHECK FACTORY TESTS
# =============================================================================

class TestDockerHealthCheck:
    """Tests for Docker health check factory."""

    @pytest.mark.asyncio
    async def test_healthy_docker(self):
        """Test that healthy Docker returns True."""
        check = create_docker_health_check(timeout=5.0)

        # Mock subprocess to return success
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = mock_proc

            result = await check()
            assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy_docker(self):
        """Test that unhealthy Docker returns False."""
        check = create_docker_health_check(timeout=5.0)

        # Mock subprocess to return failure
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.wait = AsyncMock(return_value=1)
            mock_exec.return_value = mock_proc

            result = await check()
            assert result is False

    @pytest.mark.asyncio
    async def test_docker_not_installed(self):
        """Test that missing Docker returns False."""
        check = create_docker_health_check(timeout=5.0)

        # Mock FileNotFoundError
        with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError):
            result = await check()
            assert result is False

    @pytest.mark.asyncio
    async def test_docker_timeout(self):
        """Test that Docker timeout returns False."""
        check = create_docker_health_check(timeout=0.01)

        # Mock subprocess that never completes
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.wait = AsyncMock(side_effect=asyncio.TimeoutError)
            mock_proc.kill = MagicMock()
            mock_exec.return_value = mock_proc

            result = await check()
            assert result is False


class TestOllamaHealthCheck:
    """Tests for Ollama health check factory."""

    @pytest.mark.asyncio
    async def test_healthy_ollama_with_aiohttp(self):
        """Test healthy Ollama with aiohttp returns True."""
        check = create_ollama_health_check(timeout=5.0)

        # Mock aiohttp
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            result = await check()
            assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy_ollama(self):
        """Test unhealthy Ollama returns False."""
        check = create_ollama_health_check(timeout=5.0)

        # Mock aiohttp with error
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session_class.side_effect = Exception("Connection refused")

            result = await check()
            assert result is False


class TestInvincibleNodeHealthCheck:
    """Tests for Invincible Node health check factory."""

    @pytest.mark.asyncio
    async def test_healthy_node(self, mock_vm_manager):
        """Test healthy VM returns True."""
        async def get_manager():
            return mock_vm_manager

        check = create_invincible_node_health_check(
            get_vm_manager=get_manager,
            timeout=5.0,
        )

        result = await check()
        assert result is True

    @pytest.mark.asyncio
    async def test_unhealthy_node(self, mock_vm_manager):
        """Test unhealthy VM returns False."""
        mock_vm_manager.check_static_vm_health = AsyncMock(return_value={"healthy": False})

        async def get_manager():
            return mock_vm_manager

        check = create_invincible_node_health_check(
            get_vm_manager=get_manager,
            timeout=5.0,
        )

        result = await check()
        assert result is False

    @pytest.mark.asyncio
    async def test_not_static_vm_mode(self, mock_vm_manager):
        """Test non-static VM mode returns False."""
        mock_vm_manager.is_static_vm_mode = False

        async def get_manager():
            return mock_vm_manager

        check = create_invincible_node_health_check(
            get_vm_manager=get_manager,
            timeout=5.0,
        )

        result = await check()
        assert result is False


# =============================================================================
# HEALTH PROBE FACTORY TESTS
# =============================================================================

class TestDockerHealthProbe:
    """Tests for Docker health probe factory."""

    @pytest.mark.asyncio
    async def test_probe_creation(self, config):
        """Test that probe is created with correct configuration."""
        probe = create_docker_health_probe(config=config)

        assert probe.cache_ttl == config.docker_cache_ttl
        assert probe.unhealthy_threshold == config.docker_unhealthy_threshold

    @pytest.mark.asyncio
    async def test_probe_with_callbacks(self, config):
        """Test probe with callbacks."""
        unhealthy_called = False
        healthy_called = False

        async def on_unhealthy():
            nonlocal unhealthy_called
            unhealthy_called = True

        async def on_healthy():
            nonlocal healthy_called
            healthy_called = True

        probe = create_docker_health_probe(
            config=config,
            on_unhealthy=on_unhealthy,
            on_healthy=on_healthy,
        )

        assert probe.on_unhealthy is not None
        assert probe.on_healthy is not None


class TestOllamaHealthProbe:
    """Tests for Ollama health probe factory."""

    @pytest.mark.asyncio
    async def test_probe_creation(self, config):
        """Test that probe is created with correct configuration."""
        probe = create_ollama_health_probe(config=config)

        assert probe.cache_ttl == config.ollama_cache_ttl
        assert probe.unhealthy_threshold == config.ollama_unhealthy_threshold


class TestInvincibleNodeHealthProbe:
    """Tests for Invincible Node health probe factory."""

    @pytest.mark.asyncio
    async def test_probe_creation(self, config, mock_vm_manager):
        """Test that probe is created with correct configuration."""
        async def get_manager():
            return mock_vm_manager

        probe = create_invincible_node_health_probe(
            get_vm_manager=get_manager,
            config=config,
        )

        assert probe.cache_ttl == config.invincible_node_cache_ttl
        assert probe.unhealthy_threshold == config.invincible_node_unhealthy_threshold


# =============================================================================
# BACKGROUND RECOVERY FACTORY TESTS
# =============================================================================

class TestDockerRecovery:
    """Tests for Docker recovery factory."""

    @pytest.mark.asyncio
    async def test_recovery_creation(self, config):
        """Test that recovery is created with correct configuration."""
        recovery = create_docker_recovery(config=config)

        assert recovery.config.base_delay == config.docker_recovery_base_delay
        assert recovery.config.max_delay == config.docker_recovery_max_delay
        assert recovery.config.max_attempts == config.docker_recovery_max_attempts

    @pytest.mark.asyncio
    async def test_recovery_with_callbacks(self, config):
        """Test recovery with callbacks."""
        success_called = False
        paused_called = False

        async def on_success():
            nonlocal success_called
            success_called = True

        async def on_paused():
            nonlocal paused_called
            paused_called = True

        recovery = create_docker_recovery(
            config=config,
            on_success=on_success,
            on_paused=on_paused,
        )

        assert recovery.on_success is not None
        assert recovery.on_paused is not None


class TestOllamaRecovery:
    """Tests for Ollama recovery factory."""

    @pytest.mark.asyncio
    async def test_recovery_creation(self, config):
        """Test that recovery is created with correct configuration."""
        recovery = create_ollama_recovery(config=config)

        assert recovery.config.base_delay == config.ollama_recovery_base_delay
        assert recovery.config.max_delay == config.ollama_recovery_max_delay
        assert recovery.config.max_attempts == config.ollama_recovery_max_attempts


class TestInvincibleNodeRecovery:
    """Tests for Invincible Node recovery factory."""

    @pytest.mark.asyncio
    async def test_recovery_creation(self, config, mock_vm_manager):
        """Test that recovery is created with correct configuration."""
        async def get_manager():
            return mock_vm_manager

        recovery = create_invincible_node_recovery(
            get_vm_manager=get_manager,
            config=config,
        )

        assert recovery.config.base_delay == config.invincible_node_recovery_base_delay
        assert recovery.config.max_delay == config.invincible_node_recovery_max_delay
        assert recovery.config.max_attempts == config.invincible_node_recovery_max_attempts


# =============================================================================
# CAPABILITY UPGRADE FACTORY TESTS
# =============================================================================

class TestLocalLLMCapabilityUpgrade:
    """Tests for local LLM capability upgrade factory."""

    @pytest.mark.asyncio
    async def test_upgrade_creation(self):
        """Test that capability upgrade is created correctly."""
        check_called = False
        activate_called = False
        deactivate_called = False

        async def check_cloud():
            nonlocal check_called
            check_called = True
            return True

        async def activate():
            nonlocal activate_called
            activate_called = True

        async def deactivate():
            nonlocal deactivate_called
            deactivate_called = True

        upgrade = create_local_llm_capability_upgrade(
            check_cloud_available=check_cloud,
            activate_cloud=activate,
            deactivate_cloud=deactivate,
        )

        assert upgrade.name == "local_llm_mode"
        assert upgrade.state == CapabilityState.DEGRADED

        # Test upgrade
        result = await upgrade.try_upgrade()
        assert result is True
        assert check_called
        assert activate_called
        assert upgrade.state == CapabilityState.FULL

        # Test downgrade
        await upgrade.downgrade()
        assert deactivate_called
        assert upgrade.state == CapabilityState.DEGRADED


# =============================================================================
# STARTUP RESILIENCE COORDINATOR TESTS
# =============================================================================

class TestStartupResilience:
    """Tests for StartupResilience coordinator."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, mock_logger, config):
        """Test starting and stopping the coordinator."""
        resilience = StartupResilience(logger=mock_logger, config=config)

        await resilience.start()
        assert resilience._started is True
        assert resilience._docker_probe is not None
        assert resilience._ollama_probe is not None

        await resilience.stop()
        assert resilience._started is False

    @pytest.mark.asyncio
    async def test_check_docker_healthy(self, mock_logger, config):
        """Test checking Docker when healthy."""
        resilience = StartupResilience(logger=mock_logger, config=config)
        await resilience.start()

        # Mock Docker as healthy
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock(return_value=0)
            mock_exec.return_value = mock_proc

            result = await resilience.check_docker(force=True)
            assert result is True

        await resilience.stop()

    @pytest.mark.asyncio
    async def test_check_docker_unhealthy_starts_recovery(self, mock_logger, config):
        """Test that unhealthy Docker starts background recovery."""
        resilience = StartupResilience(logger=mock_logger, config=config)
        await resilience.start()

        # Mock Docker as unhealthy
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.wait = AsyncMock(return_value=1)
            mock_exec.return_value = mock_proc

            # First check - should trigger recovery
            result = await resilience.check_docker(force=True)
            assert result is False

            # Give a moment for recovery to start
            await asyncio.sleep(0.1)

            # Recovery should be started
            assert resilience._docker_recovery is not None
            assert resilience._docker_recovery.state in (RecoveryState.RECOVERING, RecoveryState.IDLE)

        await resilience.stop()

    @pytest.mark.asyncio
    async def test_configure_invincible_node(self, mock_logger, config, mock_vm_manager):
        """Test configuring Invincible Node."""
        resilience = StartupResilience(logger=mock_logger, config=config)

        async def get_manager():
            return mock_vm_manager

        resilience.configure_invincible_node(get_vm_manager=get_manager)

        assert resilience._invincible_node_probe is not None
        assert resilience._invincible_node_recovery is not None

        await resilience.start()
        await resilience.stop()

    @pytest.mark.asyncio
    async def test_configure_llm_upgrade(self, mock_logger, config):
        """Test configuring LLM upgrade."""
        resilience = StartupResilience(logger=mock_logger, config=config)

        async def check_cloud():
            return True

        async def activate():
            pass

        async def deactivate():
            pass

        resilience.configure_llm_upgrade(
            check_cloud_available=check_cloud,
            activate_cloud=activate,
            deactivate_cloud=deactivate,
        )

        assert resilience._llm_upgrade is not None
        assert resilience._llm_upgrade.state == CapabilityState.DEGRADED

        await resilience.start()

        # Try upgrade
        result = await resilience.try_llm_upgrade()
        assert result is True
        assert resilience._llm_upgrade.state == CapabilityState.FULL

        await resilience.stop()

    @pytest.mark.asyncio
    async def test_get_status(self, mock_logger, config):
        """Test getting status of all components."""
        resilience = StartupResilience(logger=mock_logger, config=config)
        await resilience.start()

        status = resilience.get_status()

        assert status["started"] is True
        assert "docker" in status
        assert "ollama" in status
        assert "invincible_node" in status
        assert "llm_mode" in status

        await resilience.stop()

    @pytest.mark.asyncio
    async def test_notify_conditions_changed(self, mock_logger, config):
        """Test notifying conditions changed."""
        resilience = StartupResilience(logger=mock_logger, config=config)
        await resilience.start()

        # This should not raise
        resilience.notify_conditions_changed()

        await resilience.stop()

    @pytest.mark.asyncio
    async def test_idempotent_start_stop(self, mock_logger, config):
        """Test that start/stop are idempotent."""
        resilience = StartupResilience(logger=mock_logger, config=config)

        # Multiple starts should be safe
        await resilience.start()
        await resilience.start()
        assert resilience._started is True

        # Multiple stops should be safe
        await resilience.stop()
        await resilience.stop()
        assert resilience._started is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for startup resilience."""

    @pytest.mark.asyncio
    async def test_full_startup_flow(self, mock_logger, config, mock_vm_manager):
        """Test a complete startup flow with all components."""
        resilience = StartupResilience(logger=mock_logger, config=config)

        # Configure Invincible Node
        async def get_manager():
            return mock_vm_manager

        resilience.configure_invincible_node(get_vm_manager=get_manager)

        # Configure LLM upgrade
        async def check_cloud():
            return True

        async def activate():
            pass

        async def deactivate():
            pass

        resilience.configure_llm_upgrade(
            check_cloud_available=check_cloud,
            activate_cloud=activate,
            deactivate_cloud=deactivate,
        )

        # Start
        await resilience.start()

        # Get initial status
        status = resilience.get_status()
        assert status["started"] is True
        assert status["llm_mode"]["configured"] is True

        # Check Invincible Node
        result = await resilience.check_invincible_node()
        assert result is True

        # Upgrade LLM
        result = await resilience.try_llm_upgrade()
        assert result is True

        # Final status
        status = resilience.get_status()
        assert status["llm_mode"]["is_full_mode"] is True

        # Stop
        await resilience.stop()

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, mock_logger, config):
        """Test graceful degradation when services are unavailable."""
        resilience = StartupResilience(logger=mock_logger, config=config)
        await resilience.start()

        # Mock all services as unavailable
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.wait = AsyncMock(return_value=1)
            mock_exec.return_value = mock_proc

            # Check Docker - should be False but not raise
            docker_ok = await resilience.check_docker(force=True)
            assert docker_ok is False

        # The system should continue functioning
        status = resilience.get_status()
        assert status["started"] is True

        await resilience.stop()


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestStartupResilienceConfig:
    """Tests for StartupResilienceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StartupResilienceConfig()

        assert config.docker_check_timeout == 5.0
        assert config.docker_cache_ttl == 30.0
        assert config.docker_unhealthy_threshold == 3
        assert config.docker_recovery_enabled is True

        assert config.ollama_check_timeout == 10.0
        assert config.ollama_recovery_max_attempts == 20

        assert config.invincible_node_check_timeout == 30.0
        assert config.invincible_node_recovery_max_attempts == 5

        assert config.local_llm_upgrade_interval == 60.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StartupResilienceConfig(
            docker_check_timeout=10.0,
            docker_recovery_enabled=False,
            ollama_cache_ttl=60.0,
        )

        assert config.docker_check_timeout == 10.0
        assert config.docker_recovery_enabled is False
        assert config.ollama_cache_ttl == 60.0
