# tests/unit/supervisor/test_cross_repo_startup_orchestrator.py
"""
Tests for ProcessOrchestrator.startup_lock_context() method.

TDD approach for Pillar 1: Lock-Guarded Single-Owner Startup.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestStartupLockContext:
    """Tests for ProcessOrchestrator.startup_lock_context() async context manager."""

    @pytest.mark.asyncio
    async def test_startup_lock_context_acquires_lock(self):
        """Lock is acquired in __aenter__ and released in __aexit__."""
        from backend.supervisor.cross_repo_startup_orchestrator import ProcessOrchestrator

        orchestrator = ProcessOrchestrator()

        # Mock the internal lock methods
        with patch.object(orchestrator, '_acquire_startup_lock', new_callable=AsyncMock) as mock_acquire:
            with patch.object(orchestrator, '_release_startup_lock', new_callable=AsyncMock) as mock_release:
                with patch.object(orchestrator, '_enforce_hardware_environment', new_callable=AsyncMock):
                    with patch.object(orchestrator, '_initialize_cross_repo_state', new_callable=AsyncMock):
                        mock_acquire.return_value = True

                        async with orchestrator.startup_lock_context(spawn_processes=False) as ctx:
                            mock_acquire.assert_called_once()
                            assert ctx is orchestrator

                        mock_release.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_lock_context_spawn_processes_false(self):
        """When spawn_processes=False, orchestrator does not spawn processes."""
        from backend.supervisor.cross_repo_startup_orchestrator import ProcessOrchestrator

        orchestrator = ProcessOrchestrator()

        with patch.object(orchestrator, '_acquire_startup_lock', new_callable=AsyncMock, return_value=True):
            with patch.object(orchestrator, '_release_startup_lock', new_callable=AsyncMock):
                with patch.object(orchestrator, '_enforce_hardware_environment', new_callable=AsyncMock):
                    with patch.object(orchestrator, '_initialize_cross_repo_state', new_callable=AsyncMock) as mock_init:
                        async with orchestrator.startup_lock_context(spawn_processes=False):
                            # Verify spawn_processes flag is stored
                            assert orchestrator._spawn_processes is False

                        # verify _initialize_cross_repo_state was called with spawn_processes=False
                        mock_init.assert_called_once_with(spawn_processes=False)

    @pytest.mark.asyncio
    async def test_startup_lock_context_spawn_processes_true(self):
        """When spawn_processes=True (default), orchestrator allows spawning."""
        from backend.supervisor.cross_repo_startup_orchestrator import ProcessOrchestrator

        orchestrator = ProcessOrchestrator()

        with patch.object(orchestrator, '_acquire_startup_lock', new_callable=AsyncMock, return_value=True):
            with patch.object(orchestrator, '_release_startup_lock', new_callable=AsyncMock):
                with patch.object(orchestrator, '_enforce_hardware_environment', new_callable=AsyncMock):
                    with patch.object(orchestrator, '_initialize_cross_repo_state', new_callable=AsyncMock) as mock_init:
                        async with orchestrator.startup_lock_context(spawn_processes=True):
                            assert orchestrator._spawn_processes is True

                        mock_init.assert_called_once_with(spawn_processes=True)

    @pytest.mark.asyncio
    async def test_startup_lock_context_failure_raises_error(self):
        """Lock acquisition failure raises StartupLockError."""
        from backend.supervisor.cross_repo_startup_orchestrator import (
            ProcessOrchestrator,
            StartupLockError,
        )

        orchestrator = ProcessOrchestrator()

        with patch.object(orchestrator, '_acquire_startup_lock', new_callable=AsyncMock, return_value=False):
            with pytest.raises(StartupLockError, match="Failed to acquire"):
                async with orchestrator.startup_lock_context(spawn_processes=False):
                    pass

    @pytest.mark.asyncio
    async def test_startup_lock_released_on_exception(self):
        """Lock is released even when an exception occurs in the context body."""
        from backend.supervisor.cross_repo_startup_orchestrator import ProcessOrchestrator

        orchestrator = ProcessOrchestrator()

        with patch.object(orchestrator, '_acquire_startup_lock', new_callable=AsyncMock, return_value=True):
            with patch.object(orchestrator, '_release_startup_lock', new_callable=AsyncMock) as mock_release:
                with patch.object(orchestrator, '_enforce_hardware_environment', new_callable=AsyncMock):
                    with patch.object(orchestrator, '_initialize_cross_repo_state', new_callable=AsyncMock):
                        with pytest.raises(ValueError, match="test error"):
                            async with orchestrator.startup_lock_context(spawn_processes=False):
                                raise ValueError("test error")

                        # Lock should still be released
                        mock_release.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_lock_context_calls_hardware_enforcement(self):
        """startup_lock_context calls _enforce_hardware_environment."""
        from backend.supervisor.cross_repo_startup_orchestrator import ProcessOrchestrator

        orchestrator = ProcessOrchestrator()

        with patch.object(orchestrator, '_acquire_startup_lock', new_callable=AsyncMock, return_value=True):
            with patch.object(orchestrator, '_release_startup_lock', new_callable=AsyncMock):
                with patch.object(orchestrator, '_enforce_hardware_environment', new_callable=AsyncMock) as mock_hw:
                    with patch.object(orchestrator, '_initialize_cross_repo_state', new_callable=AsyncMock):
                        async with orchestrator.startup_lock_context(spawn_processes=False):
                            pass

                        mock_hw.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_lock_context_starts_gcp_prewarm_when_enabled(self):
        """startup_lock_context starts GCP prewarm when _gcp_prewarm_enabled is True."""
        from backend.supervisor.cross_repo_startup_orchestrator import ProcessOrchestrator

        orchestrator = ProcessOrchestrator()
        orchestrator._gcp_prewarm_enabled = True

        with patch.object(orchestrator, '_acquire_startup_lock', new_callable=AsyncMock, return_value=True):
            with patch.object(orchestrator, '_release_startup_lock', new_callable=AsyncMock):
                with patch.object(orchestrator, '_enforce_hardware_environment', new_callable=AsyncMock):
                    with patch.object(orchestrator, '_initialize_cross_repo_state', new_callable=AsyncMock):
                        with patch.object(orchestrator, '_start_gcp_prewarm', new_callable=AsyncMock) as mock_gcp:
                            async with orchestrator.startup_lock_context(spawn_processes=False):
                                pass

                            mock_gcp.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_lock_context_skips_gcp_prewarm_when_disabled(self):
        """startup_lock_context skips GCP prewarm when _gcp_prewarm_enabled is False."""
        from backend.supervisor.cross_repo_startup_orchestrator import ProcessOrchestrator

        orchestrator = ProcessOrchestrator()
        orchestrator._gcp_prewarm_enabled = False

        with patch.object(orchestrator, '_acquire_startup_lock', new_callable=AsyncMock, return_value=True):
            with patch.object(orchestrator, '_release_startup_lock', new_callable=AsyncMock):
                with patch.object(orchestrator, '_enforce_hardware_environment', new_callable=AsyncMock):
                    with patch.object(orchestrator, '_initialize_cross_repo_state', new_callable=AsyncMock):
                        with patch.object(orchestrator, '_start_gcp_prewarm', new_callable=AsyncMock) as mock_gcp:
                            async with orchestrator.startup_lock_context(spawn_processes=False):
                                pass

                            mock_gcp.assert_not_called()


class TestStartupLockError:
    """Tests for the StartupLockError exception class."""

    def test_startup_lock_error_exists(self):
        """StartupLockError is importable from the module."""
        from backend.supervisor.cross_repo_startup_orchestrator import StartupLockError
        assert issubclass(StartupLockError, Exception)

    def test_startup_lock_error_message(self):
        """StartupLockError can be raised with a message."""
        from backend.supervisor.cross_repo_startup_orchestrator import StartupLockError

        with pytest.raises(StartupLockError, match="test message"):
            raise StartupLockError("test message")


class TestSpawnProcessesFlag:
    """Tests for the _spawn_processes flag behavior."""

    @pytest.mark.asyncio
    async def test_spawn_processes_flag_persists(self):
        """The _spawn_processes flag persists on the orchestrator instance."""
        from backend.supervisor.cross_repo_startup_orchestrator import ProcessOrchestrator

        orchestrator = ProcessOrchestrator()

        with patch.object(orchestrator, '_acquire_startup_lock', new_callable=AsyncMock, return_value=True):
            with patch.object(orchestrator, '_release_startup_lock', new_callable=AsyncMock):
                with patch.object(orchestrator, '_enforce_hardware_environment', new_callable=AsyncMock):
                    with patch.object(orchestrator, '_initialize_cross_repo_state', new_callable=AsyncMock):
                        async with orchestrator.startup_lock_context(spawn_processes=False):
                            # Inside context, flag should be False
                            assert orchestrator._spawn_processes is False

                        # After context, flag should still be False (persists)
                        assert orchestrator._spawn_processes is False
