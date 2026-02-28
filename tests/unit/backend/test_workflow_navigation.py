from datetime import datetime
import subprocess
import sys
import types

import pytest

# Keep workflow executor imports lightweight in test environments
if "pyautogui" not in sys.modules:
    sys.modules["pyautogui"] = types.SimpleNamespace(
        moveRel=lambda *args, **kwargs: None,
        moveTo=lambda *args, **kwargs: None,
        click=lambda *args, **kwargs: None,
        hotkey=lambda *args, **kwargs: None,
        typewrite=lambda *args, **kwargs: None,
        size=lambda: (1920, 1080),
    )

from backend.api.action_executors import NavigationExecutor
from backend.api.workflow_engine import ExecutionContext
from backend.api.workflow_parser import ActionType, WorkflowAction, WorkflowParser


def test_parser_extracts_navigation_action_for_repo_target():
    parser = WorkflowParser()
    workflow = parser.parse_command("open safari and go to jarvis repo")

    assert len(workflow.actions) == 2
    assert workflow.actions[0].action_type == ActionType.OPEN_APP
    assert workflow.actions[1].action_type == ActionType.NAVIGATE
    assert workflow.actions[1].target == "jarvis repo"


@pytest.mark.asyncio
async def test_navigation_executor_opens_repo_remote_url(monkeypatch, tmp_path):
    repo_path = tmp_path / "Ironcliw-AI-Agent"
    repo_path.mkdir()

    monkeypatch.setattr(
        NavigationExecutor,
        "_discover_repositories",
        lambda self: {"jarvis": repo_path},
    )
    monkeypatch.setattr(
        NavigationExecutor,
        "_get_git_remote_url",
        lambda self, _repo: "https://github.com/drussell23/Ironcliw-AI-Agent",
    )

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    executor = NavigationExecutor()
    context = ExecutionContext(
        workflow_id="wf-nav-test",
        start_time=datetime.now(),
        user_id="test-user",
    )
    action = WorkflowAction(
        action_type=ActionType.NAVIGATE,
        target="jarvis repo",
        parameters={"destination": "jarvis repo"},
    )

    result = await executor.execute(action, context)

    assert result["status"] == "success"
    assert result["destination"] == "https://github.com/drussell23/Ironcliw-AI-Agent"
    assert context.get_variable("last_navigation_url") == "https://github.com/drussell23/Ironcliw-AI-Agent"
    assert any(cmd[:3] == ["open", "-a", "Safari"] for cmd in calls)
