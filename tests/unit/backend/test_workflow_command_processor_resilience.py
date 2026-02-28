import importlib
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest


def _load_workflow_command_processor_module(monkeypatch):
    fake_voice_module = types.ModuleType("backend.api.jarvis_voice_api")

    @dataclass
    class IroncliwCommand:
        text: str
        deadline: float = None

    fake_voice_module.IroncliwCommand = IroncliwCommand
    monkeypatch.setitem(sys.modules, "backend.api.jarvis_voice_api", fake_voice_module)

    fake_engine_module = types.ModuleType("backend.api.workflow_engine")

    class WorkflowExecutionEngine:
        async def execute_workflow(self, workflow, user_id, websocket=None):
            raise RuntimeError("test engine stub should be replaced per-test")

    fake_engine_module.WorkflowExecutionEngine = WorkflowExecutionEngine
    monkeypatch.setitem(sys.modules, "backend.api.workflow_engine", fake_engine_module)

    sys.modules.pop("backend.api.workflow_command_processor", None)
    module = importlib.import_module("backend.api.workflow_command_processor")
    return importlib.reload(module)


@pytest.mark.asyncio
async def test_process_workflow_command_uses_basic_fallback_when_response_generation_fails(monkeypatch):
    module = _load_workflow_command_processor_module(monkeypatch)
    processor = module.WorkflowCommandProcessor(use_intelligent_selection=False)

    action = SimpleNamespace(
        action_type=SimpleNamespace(value="open_app"),
        target="Safari",
        description="open Safari",
    )
    workflow = SimpleNamespace(
        original_command="open safari and go to jarvis repo",
        actions=[action],
        complexity="simple",
        estimated_duration=2,
    )

    completed_status = SimpleNamespace(value="completed")
    workflow_result = SimpleNamespace(
        workflow_id="wf-test-1",
        status=completed_status,
        success_rate=1.0,
        total_duration=0.1,
        action_results=[SimpleNamespace(status=completed_status, error=None)],
    )

    async def fake_execute_workflow(_workflow, _user_id, _websocket=None):
        return workflow_result

    async def failing_dynamic_response(*_args, **_kwargs):
        raise RuntimeError("dynamic response unavailable")

    processor.parser = SimpleNamespace(parse_command=lambda _cmd: workflow)
    processor.engine = SimpleNamespace(execute_workflow=fake_execute_workflow)
    processor._generate_response_with_claude = failing_dynamic_response
    processor._generate_basic_response = lambda *_args, **_kwargs: "fallback response"

    response = await processor.process_workflow_command(module.IroncliwCommand(text=workflow.original_command))

    assert response["command_type"] == "workflow"
    assert response["response"] == "fallback response"
    assert response["success"] is True
    assert response["workflow_result"]["actions_completed"] == 1


@pytest.mark.asyncio
async def test_generate_response_with_claude_returns_basic_when_prime_router_disabled(monkeypatch):
    module = _load_workflow_command_processor_module(monkeypatch)
    processor = module.WorkflowCommandProcessor(use_intelligent_selection=False)
    processor._use_prime_router = False

    action = SimpleNamespace(
        action_type=SimpleNamespace(value="open_app"),
        target="Safari",
        description="open Safari",
    )
    completed_status = SimpleNamespace(value="completed")
    workflow = SimpleNamespace(
        original_command="open safari",
        actions=[action],
    )
    result = SimpleNamespace(
        action_results=[SimpleNamespace(status=completed_status, error=None)],
        total_duration=0.1,
    )

    response = await processor._generate_response_with_claude(workflow, result)

    assert isinstance(response, str)
    assert response
