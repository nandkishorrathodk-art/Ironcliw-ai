import json
import sys
import types

from backend.api.workflow_engine import ActionExecutorRegistry
from backend.api.workflow_parser import ActionType


def test_executor_registry_supports_api_namespace_fallback(tmp_path, monkeypatch):
    fake_module = types.ModuleType("backend.api.fake_exec_module")

    async def fake_executor(action, context):
        return {"status": "ok"}

    fake_module.fake_executor = fake_executor
    monkeypatch.setitem(sys.modules, "backend.api.fake_exec_module", fake_module)

    config_path = tmp_path / "action_executors.json"
    config_path.write_text(
        json.dumps(
            {
                "check": {
                    "module": "api.fake_exec_module",
                    "function": "fake_executor",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ActionExecutorRegistry,
        "_register_default_executors",
        lambda self: None,
    )

    registry = ActionExecutorRegistry(config_path=str(config_path))

    assert registry.get_executor(ActionType.CHECK) is fake_executor
