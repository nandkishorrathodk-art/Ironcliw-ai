"""Tests for StartupSummary - Human-readable startup completion reporting."""
import pytest
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


class TestStartupCompletionCriteria:
    """Tests for StartupCompletionCriteria determining when startup is done."""

    def test_initial_state_is_in_progress(self):
        """New criteria should report in_progress."""
        from backend.core.startup_summary import StartupCompletionCriteria

        criteria = StartupCompletionCriteria(
            start_time=time.time(),
            global_timeout=180.0
        )
        is_complete, reason = criteria.is_complete()
        assert is_complete is False
        assert reason == "in_progress"

    def test_required_failure_completes_startup(self):
        """Required component failure should complete startup with failure reason."""
        from backend.core.startup_summary import StartupCompletionCriteria

        criteria = StartupCompletionCriteria(
            start_time=time.time(),
            global_timeout=180.0
        )
        criteria.required_failure = True
        is_complete, reason = criteria.is_complete()
        assert is_complete is True
        assert reason == "required_component_failed"

    def test_all_components_resolved_completes_startup(self):
        """All components resolved should complete startup successfully."""
        from backend.core.startup_summary import StartupCompletionCriteria

        criteria = StartupCompletionCriteria(
            start_time=time.time(),
            global_timeout=180.0
        )
        criteria.all_components_resolved = True
        is_complete, reason = criteria.is_complete()
        assert is_complete is True
        assert reason == "all_resolved"

    def test_global_timeout_completes_startup(self):
        """Exceeding global timeout should complete startup with timeout reason."""
        from backend.core.startup_summary import StartupCompletionCriteria

        # Set start time in the past so we're past timeout
        criteria = StartupCompletionCriteria(
            start_time=time.time() - 200.0,  # 200 seconds ago
            global_timeout=180.0  # 180 second timeout
        )
        is_complete, reason = criteria.is_complete()
        assert is_complete is True
        assert reason == "global_timeout"

    def test_required_failure_takes_priority_over_resolved(self):
        """Required failure should take priority over all_resolved."""
        from backend.core.startup_summary import StartupCompletionCriteria

        criteria = StartupCompletionCriteria(
            start_time=time.time(),
            global_timeout=180.0
        )
        criteria.required_failure = True
        criteria.all_components_resolved = True
        is_complete, reason = criteria.is_complete()
        assert is_complete is True
        assert reason == "required_component_failed"

    def test_all_resolved_takes_priority_over_timeout(self):
        """All resolved should take priority over timeout."""
        from backend.core.startup_summary import StartupCompletionCriteria

        criteria = StartupCompletionCriteria(
            start_time=time.time() - 200.0,  # Past timeout
            global_timeout=180.0
        )
        criteria.all_components_resolved = True
        is_complete, reason = criteria.is_complete()
        assert is_complete is True
        assert reason == "all_resolved"

    def test_custom_global_timeout(self):
        """Should respect custom global timeout value."""
        from backend.core.startup_summary import StartupCompletionCriteria

        # 10 seconds ago, 5 second timeout -> should timeout
        criteria = StartupCompletionCriteria(
            start_time=time.time() - 10.0,
            global_timeout=5.0
        )
        is_complete, reason = criteria.is_complete()
        assert is_complete is True
        assert reason == "global_timeout"


class TestComponentSummary:
    """Tests for ComponentSummary dataclass."""

    def test_component_summary_creation(self):
        """ComponentSummary should store all required fields."""
        from backend.core.startup_summary import ComponentSummary
        from backend.core.component_registry import ComponentStatus, Criticality

        summary = ComponentSummary(
            name="jarvis-core",
            status=ComponentStatus.HEALTHY,
            criticality=Criticality.REQUIRED,
            startup_time=1.234,
            message=None
        )
        assert summary.name == "jarvis-core"
        assert summary.status == ComponentStatus.HEALTHY
        assert summary.criticality == Criticality.REQUIRED
        assert summary.startup_time == 1.234
        assert summary.message is None

    def test_component_summary_with_message(self):
        """ComponentSummary should store failure message."""
        from backend.core.startup_summary import ComponentSummary
        from backend.core.component_registry import ComponentStatus, Criticality

        summary = ComponentSummary(
            name="reactor-core",
            status=ComponentStatus.FAILED,
            criticality=Criticality.OPTIONAL,
            startup_time=12.345,
            message="Connection refused"
        )
        assert summary.name == "reactor-core"
        assert summary.status == ComponentStatus.FAILED
        assert summary.message == "Connection refused"


class TestStartupSummary:
    """Tests for StartupSummary formatting and output."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock ComponentRegistry with test components."""
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentState,
            ComponentStatus, Criticality, ProcessType
        )

        registry = ComponentRegistry()

        # Register some test components
        core_def = ComponentDefinition(
            name="jarvis-core",
            criticality=Criticality.REQUIRED,
            process_type=ProcessType.IN_PROCESS,
            provides_capabilities=["core"]
        )
        core_state = registry.register(core_def)
        core_state.mark_starting()
        core_state.mark_healthy()

        redis_def = ComponentDefinition(
            name="redis",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.EXTERNAL_SERVICE,
            provides_capabilities=["cache"]
        )
        redis_state = registry.register(redis_def)
        redis_state.mark_starting()
        redis_state.mark_healthy()

        prime_def = ComponentDefinition(
            name="jarvis-prime",
            criticality=Criticality.DEGRADED_OK,
            process_type=ProcessType.SUBPROCESS,
            provides_capabilities=["inference"]
        )
        prime_state = registry.register(prime_def)
        prime_state.mark_starting()

        reactor_def = ComponentDefinition(
            name="reactor-core",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.EXTERNAL_SERVICE,
            provides_capabilities=["training"]
        )
        reactor_state = registry.register(reactor_def)
        reactor_state.mark_starting()
        reactor_state.mark_failed("Connection refused")

        trinity_def = ComponentDefinition(
            name="trinity",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.SUBPROCESS,
            disable_env_var="TRINITY_ENABLED"
        )
        trinity_state = registry.register(trinity_def)
        trinity_state.mark_disabled("TRINITY_ENABLED=false")

        return registry

    def test_status_icons_mapping(self):
        """STATUS_ICONS should have icons for all ComponentStatus values."""
        from backend.core.startup_summary import StartupSummary
        from backend.core.component_registry import ComponentStatus

        # Verify all statuses have icons
        for status in ComponentStatus:
            assert status in StartupSummary.STATUS_ICONS

        # Verify specific icons
        assert StartupSummary.STATUS_ICONS[ComponentStatus.HEALTHY] == "✓"
        assert StartupSummary.STATUS_ICONS[ComponentStatus.FAILED] == "✗"
        assert StartupSummary.STATUS_ICONS[ComponentStatus.DISABLED] == "○"
        assert StartupSummary.STATUS_ICONS[ComponentStatus.PENDING] == "·"

    def test_format_summary_header(self, mock_registry):
        """format_summary should include version header."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)
        output = summary.format_summary()

        assert "Ironcliw Startup Summary" in output
        assert "v148.0" in output
        assert "━" in output  # Decorative lines

    def test_format_summary_component_lines(self, mock_registry):
        """format_summary should show each component with status icon and criticality."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)
        output = summary.format_summary()

        # Check for component names
        assert "jarvis-core" in output
        assert "redis" in output
        assert "jarvis-prime" in output
        assert "reactor-core" in output
        assert "trinity" in output

        # Check for status icons
        assert "✓" in output  # HEALTHY
        assert "✗" in output  # FAILED
        assert "○" in output  # DISABLED

    def test_format_summary_criticality_labels(self, mock_registry):
        """format_summary should show criticality labels."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)
        output = summary.format_summary()

        assert "[required]" in output
        assert "[optional]" in output
        assert "[degraded_ok]" in output

    def test_format_summary_failure_messages(self, mock_registry):
        """format_summary should show failure messages."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)
        output = summary.format_summary()

        assert "Connection refused" in output

    def test_format_summary_capabilities_section(self, mock_registry):
        """format_summary should include capabilities section."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)
        output = summary.format_summary()

        assert "Capabilities" in output
        # Healthy capability from healthy component
        assert "core" in output
        # Failed capability from failed component
        assert "training" in output

    def test_format_summary_total_time(self, mock_registry):
        """format_summary should show total startup time."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)
        output = summary.format_summary()

        assert "Total startup time:" in output or "startup time" in output.lower()

    def test_format_summary_system_status(self, mock_registry):
        """format_summary should show overall system status."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)
        output = summary.format_summary()

        assert "System status:" in output or "status:" in output

    def test_to_dict_serializable(self, mock_registry):
        """to_dict should return JSON-serializable dict."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)
        data = summary.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        # Should contain expected keys
        assert "version" in data
        assert "start_time" in data
        assert "end_time" in data
        assert "components" in data
        assert "capabilities" in data
        assert "overall_status" in data

    def test_to_dict_component_details(self, mock_registry):
        """to_dict should include component details."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)
        data = summary.to_dict()

        components = data["components"]
        assert "jarvis-core" in components
        assert "reactor-core" in components

        # Check component structure
        core_data = components["jarvis-core"]
        assert "status" in core_data
        assert "criticality" in core_data

    def test_compute_overall_status_healthy(self):
        """compute_overall_status should return HEALTHY when all components healthy."""
        from backend.core.startup_summary import StartupSummary
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentStatus,
            Criticality, ProcessType
        )

        registry = ComponentRegistry()

        core_def = ComponentDefinition(
            name="core",
            criticality=Criticality.REQUIRED,
            process_type=ProcessType.IN_PROCESS,
        )
        core_state = registry.register(core_def)
        core_state.mark_healthy()

        redis_def = ComponentDefinition(
            name="redis",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.EXTERNAL_SERVICE,
        )
        redis_state = registry.register(redis_def)
        redis_state.mark_healthy()

        summary = StartupSummary(registry=registry)
        status = summary.compute_overall_status()
        assert status == "HEALTHY"

    def test_compute_overall_status_degraded(self):
        """compute_overall_status should return DEGRADED when optional component failed."""
        from backend.core.startup_summary import StartupSummary
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentStatus,
            Criticality, ProcessType
        )

        registry = ComponentRegistry()

        core_def = ComponentDefinition(
            name="core",
            criticality=Criticality.REQUIRED,
            process_type=ProcessType.IN_PROCESS,
        )
        core_state = registry.register(core_def)
        core_state.mark_healthy()

        redis_def = ComponentDefinition(
            name="redis",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.EXTERNAL_SERVICE,
        )
        redis_state = registry.register(redis_def)
        redis_state.mark_failed("Connection refused")

        summary = StartupSummary(registry=registry)
        status = summary.compute_overall_status()
        assert status == "DEGRADED"

    def test_compute_overall_status_failed(self):
        """compute_overall_status should return FAILED when required component failed."""
        from backend.core.startup_summary import StartupSummary
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentStatus,
            Criticality, ProcessType
        )

        registry = ComponentRegistry()

        core_def = ComponentDefinition(
            name="core",
            criticality=Criticality.REQUIRED,
            process_type=ProcessType.IN_PROCESS,
        )
        core_state = registry.register(core_def)
        core_state.mark_failed("Startup failed")

        summary = StartupSummary(registry=registry)
        status = summary.compute_overall_status()
        assert status == "FAILED"

    def test_compute_overall_status_with_starting(self):
        """compute_overall_status should indicate starting components."""
        from backend.core.startup_summary import StartupSummary
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentStatus,
            Criticality, ProcessType
        )

        registry = ComponentRegistry()

        core_def = ComponentDefinition(
            name="core",
            criticality=Criticality.REQUIRED,
            process_type=ProcessType.IN_PROCESS,
        )
        core_state = registry.register(core_def)
        core_state.mark_healthy()

        prime_def = ComponentDefinition(
            name="prime",
            criticality=Criticality.DEGRADED_OK,
            process_type=ProcessType.SUBPROCESS,
        )
        prime_state = registry.register(prime_def)
        prime_state.mark_starting()

        summary = StartupSummary(registry=registry)
        status = summary.compute_overall_status()
        # Should indicate degraded because a component is still starting
        assert status in ("DEGRADED", "STARTING")

    def test_compute_overall_status_disabled_ignored(self):
        """compute_overall_status should ignore disabled components."""
        from backend.core.startup_summary import StartupSummary
        from backend.core.component_registry import (
            ComponentRegistry, ComponentDefinition, ComponentStatus,
            Criticality, ProcessType
        )

        registry = ComponentRegistry()

        core_def = ComponentDefinition(
            name="core",
            criticality=Criticality.REQUIRED,
            process_type=ProcessType.IN_PROCESS,
        )
        core_state = registry.register(core_def)
        core_state.mark_healthy()

        trinity_def = ComponentDefinition(
            name="trinity",
            criticality=Criticality.OPTIONAL,
            process_type=ProcessType.SUBPROCESS,
        )
        trinity_state = registry.register(trinity_def)
        trinity_state.mark_disabled("Disabled by env")

        summary = StartupSummary(registry=registry)
        status = summary.compute_overall_status()
        assert status == "HEALTHY"

    def test_save_to_file_creates_file(self, mock_registry, tmp_path):
        """save_to_file should create JSON file with correct content."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)

        test_path = tmp_path / "startup_summary.json"
        summary.save_to_file(path=test_path)

        assert test_path.exists()

        # Verify content is valid JSON
        with open(test_path) as f:
            data = json.load(f)

        assert data["version"] == "148.0"
        assert "components" in data
        assert "jarvis-core" in data["components"]

    def test_save_to_file_default_path(self, mock_registry):
        """save_to_file should use default ~/.jarvis/state/ path when not specified."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)

        # Mock the path expansion
        with patch.object(Path, 'mkdir') as mock_mkdir:
            with patch('builtins.open', MagicMock()) as mock_open:
                # Should not raise
                summary.save_to_file()
                # Default path should be used
                mock_open.assert_called_once()
                call_path = mock_open.call_args[0][0]
                assert "startup_summary.json" in str(call_path)

    def test_save_to_file_creates_parent_dirs(self, mock_registry, tmp_path):
        """save_to_file should create parent directories if needed."""
        from backend.core.startup_summary import StartupSummary

        summary = StartupSummary(registry=mock_registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)

        nested_path = tmp_path / "deep" / "nested" / "startup_summary.json"
        summary.save_to_file(path=nested_path)

        assert nested_path.exists()


class TestStartupSummaryEmptyRegistry:
    """Tests for StartupSummary with empty registry."""

    def test_format_summary_empty_registry(self):
        """format_summary should handle empty registry gracefully."""
        from backend.core.startup_summary import StartupSummary
        from backend.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        summary = StartupSummary(registry=registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)

        # Should not raise
        output = summary.format_summary()
        assert "Ironcliw Startup Summary" in output
        assert "v148.0" in output

    def test_compute_overall_status_empty_registry(self):
        """compute_overall_status should return HEALTHY for empty registry."""
        from backend.core.startup_summary import StartupSummary
        from backend.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        summary = StartupSummary(registry=registry)
        status = summary.compute_overall_status()
        assert status == "HEALTHY"

    def test_to_dict_empty_registry(self):
        """to_dict should handle empty registry."""
        from backend.core.startup_summary import StartupSummary
        from backend.core.component_registry import ComponentRegistry

        registry = ComponentRegistry()
        summary = StartupSummary(registry=registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)

        data = summary.to_dict()
        assert data["components"] == {}
        assert data["capabilities"] == {}
        assert data["overall_status"] == "HEALTHY"
