#!/usr/bin/env python3
"""
Unit Tests for Ironcliw Supervisor Module
========================================

Tests for the Self-Updating Lifecycle Manager components.

Run with:
    python -m pytest tests/unit/backend/test_supervisor.py -v
"""

import asyncio
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "backend"))


class TestSupervisorConfig:
    """Tests for SupervisorConfig."""
    
    def test_default_config(self):
        """Test loading default configuration."""
        from core.supervisor.supervisor_config import SupervisorConfig, SupervisorMode
        
        config = SupervisorConfig()
        
        assert config.enabled is True
        assert config.mode == SupervisorMode.AUTO
        assert config.update.check.enabled is True
        assert config.idle.threshold_seconds == 7200
        assert config.health.boot_stability_window == 60
        assert config.rollback.max_versions == 5
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        from core.supervisor.supervisor_config import load_config
        
        config_content = """
supervisor:
  enabled: true
  mode: manual
  
update:
  check:
    enabled: false
    interval_seconds: 600
    
idle:
  enabled: false
  threshold_seconds: 3600
"""
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            f.flush()
            
            config = load_config(Path(f.name))
            
            assert config.mode.value == "manual"
            assert config.update.check.enabled is False
            assert config.update.check.interval_seconds == 600
            assert config.idle.enabled is False
            assert config.idle.threshold_seconds == 3600
    
    def test_env_override(self):
        """Test environment variable overrides."""
        from core.supervisor.supervisor_config import _env_override
        
        import os
        os.environ["Ironcliw_SUPERVISOR_TEST_VAR"] = "hello"
        
        result = _env_override("test_var", "default")
        assert result == "hello"
        
        del os.environ["Ironcliw_SUPERVISOR_TEST_VAR"]
        
        result = _env_override("test_var", "default")
        assert result == "default"
    
    def test_bool_env_override(self):
        """Test boolean environment variable parsing."""
        from core.supervisor.supervisor_config import _env_override
        
        import os
        
        for true_val in ["true", "1", "yes", "on"]:
            os.environ["Ironcliw_SUPERVISOR_BOOL_TEST"] = true_val
            assert _env_override("bool_test", False, bool) is True
        
        for false_val in ["false", "0", "no", "off"]:
            os.environ["Ironcliw_SUPERVISOR_BOOL_TEST"] = false_val
            assert _env_override("bool_test", True, bool) is False
        
        if "Ironcliw_SUPERVISOR_BOOL_TEST" in os.environ:
            del os.environ["Ironcliw_SUPERVISOR_BOOL_TEST"]


class TestExitCode:
    """Tests for ExitCode enum."""
    
    def test_exit_codes(self):
        """Test exit code values."""
        from core.supervisor.jarvis_supervisor import ExitCode
        
        assert ExitCode.CLEAN_SHUTDOWN == 0
        assert ExitCode.ERROR_CRASH == 1
        assert ExitCode.UPDATE_REQUEST == 100
        assert ExitCode.ROLLBACK_REQUEST == 101
        assert ExitCode.RESTART_REQUEST == 102


class TestSupervisorState:
    """Tests for SupervisorState enum."""
    
    def test_states(self):
        """Test supervisor state values."""
        from core.supervisor.jarvis_supervisor import SupervisorState
        
        assert SupervisorState.INITIALIZING.value == "initializing"
        assert SupervisorState.RUNNING.value == "running"
        assert SupervisorState.UPDATING.value == "updating"
        assert SupervisorState.ROLLING_BACK.value == "rolling_back"


class TestProcessInfo:
    """Tests for ProcessInfo dataclass."""
    
    def test_is_stable(self):
        """Test stability calculation."""
        from core.supervisor.jarvis_supervisor import ProcessInfo
        from datetime import timedelta
        
        # Not stable - no start time
        info = ProcessInfo()
        assert info.is_stable(60) is False
        
        # Not stable - too recent
        info.start_time = datetime.now()
        assert info.is_stable(60) is False
        
        # Stable - started long ago
        info.start_time = datetime.now() - timedelta(seconds=120)
        assert info.is_stable(60) is True


class TestUpdatePhase:
    """Tests for UpdatePhase enum."""
    
    def test_phases(self):
        """Test update phase values."""
        from core.supervisor.update_engine import UpdatePhase
        
        assert UpdatePhase.IDLE.value == "idle"
        assert UpdatePhase.FETCHING.value == "fetching"
        assert UpdatePhase.INSTALLING.value == "installing"
        assert UpdatePhase.COMPLETE.value == "complete"
        assert UpdatePhase.FAILED.value == "failed"


class TestVersionSnapshot:
    """Tests for VersionSnapshot dataclass."""
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        from core.supervisor.rollback_manager import VersionSnapshot
        
        snapshot = VersionSnapshot(
            id=1,
            git_commit="abc123",
            git_branch="main",
            is_stable=True,
        )
        
        data = snapshot.to_dict()
        
        assert data["id"] == 1
        assert data["git_commit"] == "abc123"
        assert data["git_branch"] == "main"
        assert data["is_stable"] is True
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        from core.supervisor.rollback_manager import VersionSnapshot
        
        data = {
            "id": 2,
            "git_commit": "def456",
            "git_branch": "develop",
            "timestamp": "2025-01-01T12:00:00",
            "is_stable": False,
        }
        
        snapshot = VersionSnapshot.from_dict(data)
        
        assert snapshot.id == 2
        assert snapshot.git_commit == "def456"
        assert snapshot.git_branch == "develop"
        assert snapshot.is_stable is False


class TestHealthStatus:
    """Tests for HealthStatus enum."""
    
    def test_statuses(self):
        """Test health status values."""
        from core.supervisor.health_monitor import HealthStatus
        
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestChangeCategory:
    """Tests for ChangeCategory enum."""
    
    def test_categories(self):
        """Test change category values."""
        from core.supervisor.changelog_analyzer import ChangeCategory
        
        assert ChangeCategory.SECURITY.value == "security"
        assert ChangeCategory.FEATURE.value == "feature"
        assert ChangeCategory.FIX.value == "fix"
        assert ChangeCategory.PERFORMANCE.value == "performance"


class TestChangelogAnalyzer:
    """Tests for ChangelogAnalyzer."""
    
    def test_classify_category(self):
        """Test commit category classification."""
        from core.supervisor.changelog_analyzer import ChangelogAnalyzer, ChangeCategory
        
        analyzer = ChangelogAnalyzer()
        
        assert analyzer._classify_category("feat: add new feature") == ChangeCategory.FEATURE
        assert analyzer._classify_category("fix: bug in login") == ChangeCategory.FIX
        assert analyzer._classify_category("docs: update readme") == ChangeCategory.DOCS
        assert analyzer._classify_category("perf: optimize query") == ChangeCategory.PERFORMANCE
        assert analyzer._classify_category("random message") == ChangeCategory.UNKNOWN
    
    def test_assess_impact(self):
        """Test impact assessment."""
        from core.supervisor.changelog_analyzer import ChangelogAnalyzer
        
        analyzer = ChangelogAnalyzer()
        
        assert analyzer._assess_impact("BREAKING: major change") == "high"
        assert analyzer._assess_impact("Critical security fix") == "high"
        assert analyzer._assess_impact("Add new button") == "medium"
        assert analyzer._assess_impact("Fix typo in docs") == "low"
    
    def test_extract_summary(self):
        """Test summary extraction."""
        from core.supervisor.changelog_analyzer import ChangelogAnalyzer
        
        analyzer = ChangelogAnalyzer()
        
        assert analyzer._extract_summary("feat(voice): add wake word") == "Add wake word"
        assert analyzer._extract_summary("fix: login bug") == "Login bug"
        assert analyzer._extract_summary("Simple message") == "Simple message"


class TestActivityLevel:
    """Tests for ActivityLevel enum."""
    
    def test_levels(self):
        """Test activity level values."""
        from core.supervisor.idle_detector import ActivityLevel
        
        assert ActivityLevel.ACTIVE.value == "active"
        assert ActivityLevel.IDLE.value == "idle"
        assert ActivityLevel.DEEP_IDLE.value == "deep_idle"


class TestIdleState:
    """Tests for IdleState dataclass."""
    
    def test_defaults(self):
        """Test default values."""
        from core.supervisor.idle_detector import IdleState, ActivityLevel
        
        state = IdleState()
        
        assert state.level == ActivityLevel.UNKNOWN
        assert state.idle_seconds == 0.0
        assert state.is_idle is False
        assert state.checked_at is not None


class TestCommitInfo:
    """Tests for CommitInfo dataclass."""
    
    def test_creation(self):
        """Test CommitInfo creation."""
        from core.supervisor.update_detector import CommitInfo
        
        info = CommitInfo(
            sha="abc123def456",
            message="Test commit",
            author="Developer",
            date=datetime.now(),
        )
        
        assert info.sha == "abc123def456"
        assert info.message == "Test commit"
        assert info.author == "Developer"


class TestUpdateInfo:
    """Tests for UpdateInfo dataclass."""
    
    def test_defaults(self):
        """Test default values."""
        from core.supervisor.update_detector import UpdateInfo
        
        info = UpdateInfo()
        
        assert info.available is False
        assert info.commits_behind == 0
        assert len(info.commits) == 0


# Async tests

@pytest.mark.asyncio
async def test_rollback_manager_init():
    """Test RollbackManager initialization."""
    from core.supervisor.rollback_manager import RollbackManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MagicMock()
        config.rollback.history_db = "test_history.db"
        config.rollback.include_pip_freeze = False
        config.rollback.max_versions = 3
        
        manager = RollbackManager(config, repo_path=Path(tmpdir))
        await manager.initialize()
        
        # Check database was created
        db_path = Path(tmpdir) / "test_history.db"
        assert db_path.exists()
        
        await manager.close()


@pytest.mark.asyncio
async def test_health_monitor_check():
    """Test HealthMonitor health check."""
    from core.supervisor.health_monitor import HealthMonitor, HealthStatus
    
    config = MagicMock()
    config.health.check_timeout_seconds = 5
    config.health.boot_stability_window = 60
    
    monitor = HealthMonitor(config)
    monitor.record_boot_start()
    
    # Check health (will fail because no server running)
    health = await monitor.check_health()
    
    assert health is not None
    assert health.uptime_seconds > 0
    
    await monitor.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
