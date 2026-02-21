"""Unit tests for CloudSQL proxy monitor timeout/liveness behavior."""

import contextlib
import asyncio
from pathlib import Path

import pytest

from backend.intelligence.cloud_sql_proxy_manager import CloudSQLProxyManager


def _make_manager(monkeypatch: pytest.MonkeyPatch) -> CloudSQLProxyManager:
    """Create a manager instance without requiring real local GCP config/binary."""
    monkeypatch.setattr(
        CloudSQLProxyManager,
        "_discover_config_path",
        lambda self: Path("/tmp/database_config.json"),
    )
    monkeypatch.setattr(
        CloudSQLProxyManager,
        "_load_config",
        lambda self: {
            "project_id": "test-project",
            "cloud_sql": {
                "connection_name": "test-project:us-central1:test-instance",
                "port": 5432,
            },
        },
    )
    monkeypatch.setattr(
        CloudSQLProxyManager,
        "_discover_proxy_binary",
        lambda self: "/usr/bin/true",
    )
    return CloudSQLProxyManager()


@pytest.mark.asyncio
async def test_monitor_timeout_with_alive_fallback_does_not_restart(monkeypatch):
    """If primary check times out but fallback says alive, monitor must not restart."""
    manager = _make_manager(monkeypatch)
    monkeypatch.setenv("TIMEOUT_PROXY_HEALTH_CHECK", "0.01")
    monkeypatch.setenv("TIMEOUT_PROXY_HEALTH_FALLBACK", "0.01")
    monkeypatch.setenv("CLOUDSQL_MONITOR_TIMEOUT_TOLERANCE", "1")

    async def _slow_is_running_async(_strict=False):
        await asyncio.sleep(1.0)
        return True

    async def _alive_fallback(timeout: float = 1.5):
        return True

    restart_calls = 0

    async def _start(*args, **kwargs):
        nonlocal restart_calls
        restart_calls += 1
        return True

    monkeypatch.setattr(manager, "is_running_async", _slow_is_running_async)
    monkeypatch.setattr(manager, "_quick_liveness_probe_async", _alive_fallback)
    monkeypatch.setattr(manager, "start", _start)

    task = asyncio.create_task(
        manager.monitor(check_interval=0, max_recovery_attempts=2, db_level_check=False)
    )
    await asyncio.sleep(0.06)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert restart_calls == 0


@pytest.mark.asyncio
async def test_monitor_timeout_with_down_fallback_triggers_restart(monkeypatch):
    """If primary check times out and fallback confirms down, monitor should restart."""
    manager = _make_manager(monkeypatch)
    monkeypatch.setenv("TIMEOUT_PROXY_HEALTH_CHECK", "0.01")
    monkeypatch.setenv("TIMEOUT_PROXY_HEALTH_FALLBACK", "0.01")

    async def _slow_is_running_async(_strict=False):
        await asyncio.sleep(1.0)
        return True

    async def _down_fallback(timeout: float = 1.5):
        return False

    restart_calls = 0

    async def _start(*args, **kwargs):
        nonlocal restart_calls
        restart_calls += 1
        return False

    monkeypatch.setattr(manager, "is_running_async", _slow_is_running_async)
    monkeypatch.setattr(manager, "_quick_liveness_probe_async", _down_fallback)
    monkeypatch.setattr(manager, "start", _start)

    task = asyncio.create_task(
        manager.monitor(check_interval=0, max_recovery_attempts=2, db_level_check=False)
    )
    await asyncio.sleep(0.08)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert restart_calls >= 1


@pytest.mark.asyncio
async def test_monitor_prevents_duplicate_active_loops(monkeypatch):
    """A second monitor() call should exit immediately while one loop is active."""
    manager = _make_manager(monkeypatch)

    async def _alive_is_running_async(_strict=False):
        return True

    async def _alive_fallback(timeout: float = 1.5):
        return True

    monkeypatch.setattr(manager, "is_running_async", _alive_is_running_async)
    monkeypatch.setattr(manager, "_quick_liveness_probe_async", _alive_fallback)

    t1 = asyncio.create_task(
        manager.monitor(check_interval=0.05, max_recovery_attempts=2, db_level_check=False)
    )
    await asyncio.sleep(0.01)

    t2 = asyncio.create_task(
        manager.monitor(check_interval=0.05, max_recovery_attempts=2, db_level_check=False)
    )
    await asyncio.sleep(0.03)

    assert t2.done()
    assert t2.exception() is None

    t1.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await t1
