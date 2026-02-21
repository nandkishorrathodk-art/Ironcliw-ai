from __future__ import annotations

import asyncio
import time

import pytest


class _FakeEventBus:
    async def emit(self, _event):
        return None


def _minimal_config():
    from backend.macos_helper.system_event_monitor import MonitorConfig

    return MonitorConfig(
        enable_app_monitoring=False,
        enable_window_monitoring=False,
        enable_space_monitoring=False,
        enable_system_state_monitoring=False,
        enable_idle_monitoring=False,
        enable_event_batching=False,
        startup_step_timeout_seconds=0.2,
        startup_warmup_timeout_seconds=0.25,
        subprocess_timeout_seconds=0.2,
    )


async def test_start_is_non_blocking_while_warmup_runs(monkeypatch):
    from backend.macos_helper.system_event_monitor import SystemEventMonitor

    async def _fake_get_event_bus():
        return _FakeEventBus()

    monkeypatch.setattr(
        "backend.macos_helper.event_bus.get_macos_event_bus",
        _fake_get_event_bus,
        raising=False,
    )

    monitor = SystemEventMonitor(config=_minimal_config())

    async def _slow_capture():
        await asyncio.sleep(0.5)

    monitor._check_yabai = lambda: asyncio.sleep(0)  # type: ignore[method-assign]
    monitor._capture_initial_state = _slow_capture  # type: ignore[method-assign]

    started = time.monotonic()
    await monitor.start()
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    status = monitor.get_status()
    assert status["running"] is True
    assert status["startup_state"] == "running"

    if monitor._startup_task is not None:
        await monitor._startup_task

    status = monitor.get_status()
    assert status["startup_state"] == "ready"
    assert status["startup_error"] is None

    await monitor.stop()


async def test_start_failure_does_not_mark_running(monkeypatch):
    from backend.macos_helper.system_event_monitor import SystemEventMonitor

    async def _failing_get_event_bus():
        raise RuntimeError("event bus unavailable")

    monkeypatch.setattr(
        "backend.macos_helper.event_bus.get_macos_event_bus",
        _failing_get_event_bus,
        raising=False,
    )

    monitor = SystemEventMonitor(config=_minimal_config())
    await monitor.start()

    status = monitor.get_status()
    assert status["running"] is False
    assert status["startup_state"] == "failed"
    assert "event bus init failed" in (status["startup_error"] or "")


async def test_warmup_timeout_keeps_monitor_running(monkeypatch):
    from backend.macos_helper.system_event_monitor import SystemEventMonitor

    async def _fake_get_event_bus():
        return _FakeEventBus()

    monkeypatch.setattr(
        "backend.macos_helper.event_bus.get_macos_event_bus",
        _fake_get_event_bus,
        raising=False,
    )

    monitor = SystemEventMonitor(config=_minimal_config())

    async def _slow_check():
        await asyncio.sleep(0.5)

    monitor._check_yabai = _slow_check  # type: ignore[method-assign]
    monitor._capture_initial_state = lambda: asyncio.sleep(0)  # type: ignore[method-assign]

    await monitor.start()
    assert monitor.get_status()["running"] is True

    if monitor._startup_task is not None:
        await monitor._startup_task

    status = monitor.get_status()
    assert status["running"] is True
    assert status["startup_state"] == "degraded"
    assert "warmup timed out" in (status["startup_error"] or "")

    await monitor.stop()


async def test_get_system_event_monitor_raises_if_auto_start_fails(monkeypatch):
    from backend.macos_helper import system_event_monitor as sem

    async def _failing_get_event_bus():
        raise RuntimeError("bus down")

    monkeypatch.setattr(
        "backend.macos_helper.event_bus.get_macos_event_bus",
        _failing_get_event_bus,
        raising=False,
    )

    sem._system_event_monitor = None
    with pytest.raises(RuntimeError, match="SystemEventMonitor failed to start"):
        await sem.get_system_event_monitor(auto_start=True)
    sem._system_event_monitor = None
