import subprocess
import time
from pathlib import Path

import pytest

from loading_server import ServerConfig, SupervisorRecoveryController


class _DummyProcess:
    def __init__(self, pid: int = 1234, running: bool = True):
        self.pid = pid
        self._running = running

    def poll(self):
        return None if self._running else 0


@pytest.mark.asyncio
async def test_recovery_short_circuits_when_backend_and_kernel_are_healthy(monkeypatch):
    controller = SupervisorRecoveryController(ServerConfig())

    async def _healthy():
        return True

    monkeypatch.setattr(controller, "_check_backend_health", _healthy)
    monkeypatch.setattr(controller, "_check_kernel_health", _healthy)

    result = await controller.request_recovery(reason="unit_test")

    assert result["accepted"] is False
    assert result["status"] == "already_healthy"


@pytest.mark.asyncio
async def test_recovery_returns_in_progress_when_process_already_running(monkeypatch):
    controller = SupervisorRecoveryController(ServerConfig())
    controller._recovery_process = _DummyProcess(pid=9876, running=True)
    controller._last_attempt_id = "attempt123"

    async def _unhealthy():
        return False

    monkeypatch.setattr(controller, "_check_backend_health", _unhealthy)
    monkeypatch.setattr(controller, "_check_kernel_health", _unhealthy)

    result = await controller.request_recovery(reason="unit_test")

    assert result["accepted"] is True
    assert result["status"] == "in_progress"
    assert result["recovery_pid"] == 9876
    assert result["attempt_id"] == "attempt123"


@pytest.mark.asyncio
async def test_recovery_respects_cooldown(monkeypatch):
    controller = SupervisorRecoveryController(ServerConfig())
    controller._cooldown_seconds = 60.0
    controller._last_attempt_at = time.monotonic()
    controller._last_attempt_id = "cooldown-id"

    async def _unhealthy():
        return False

    monkeypatch.setattr(controller, "_check_backend_health", _unhealthy)
    monkeypatch.setattr(controller, "_check_kernel_health", _unhealthy)

    result = await controller.request_recovery(reason="unit_test")

    assert result["accepted"] is False
    assert result["status"] == "cooldown"
    assert result["attempt_id"] == "cooldown-id"
    assert result["retry_after_seconds"] > 0


@pytest.mark.asyncio
async def test_recovery_spawns_unified_supervisor_restart(monkeypatch, tmp_path: Path):
    controller = SupervisorRecoveryController(ServerConfig())
    controller._project_root = tmp_path
    controller._supervisor_script = tmp_path / "unified_supervisor.py"
    controller._supervisor_script.write_text("print('stub supervisor')\n", encoding="utf-8")
    controller._python_executable = "/usr/bin/python3"
    monkeypatch.setattr(controller, "_get_recovery_log_path", lambda: tmp_path / "recovery.log")

    async def _unhealthy():
        return False

    monkeypatch.setattr(controller, "_check_backend_health", _unhealthy)
    monkeypatch.setattr(controller, "_check_kernel_health", _unhealthy)

    captured = {}

    class _PopenStub:
        def __init__(self, cmd, cwd, stdout, stderr, start_new_session, env):
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            captured["start_new_session"] = start_new_session
            captured["env"] = env
            self.pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", _PopenStub)

    result = await controller.request_recovery(reason="unit_test", action="restart")

    assert result["accepted"] is True
    assert result["status"] == "restart_initiated"
    assert result["recovery_pid"] == 4242
    assert captured["cmd"][0] == "/usr/bin/python3"
    assert str(controller._supervisor_script) in captured["cmd"]
    assert "--restart" in captured["cmd"]
    assert captured["cwd"] == str(tmp_path)
    assert captured["start_new_session"] is True
