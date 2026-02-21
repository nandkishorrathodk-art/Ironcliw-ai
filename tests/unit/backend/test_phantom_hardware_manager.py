"""Unit tests for PhantomHardwareManager concurrency and compatibility paths."""

import asyncio

import pytest

from backend.system import phantom_hardware_manager as phm


def _fresh_manager() -> phm.PhantomHardwareManager:
    """Return an isolated singleton instance for each test."""
    phm.PhantomHardwareManager._instance = None
    phm._phantom_manager_instance = None
    return phm.get_phantom_manager()


@pytest.mark.asyncio
async def test_ensure_ghost_display_is_single_flight(monkeypatch):
    """Concurrent ensure calls should share one in-flight operation."""
    manager = _fresh_manager()
    calls = 0
    gate = asyncio.Event()

    async def _fake_impl(self, wait_for_registration=True, max_wait_seconds=15.0):
        nonlocal calls
        calls += 1
        await gate.wait()
        return True, None

    monkeypatch.setattr(
        phm.PhantomHardwareManager,
        "_ensure_ghost_display_exists_impl",
        _fake_impl,
    )

    t1 = asyncio.create_task(manager.ensure_ghost_display_exists_async())
    t2 = asyncio.create_task(manager.ensure_ghost_display_exists_async())
    for _ in range(100):
        if manager._ensure_inflight is not None:
            break
        await asyncio.sleep(0.01)
    for _ in range(100):
        if calls > 0:
            break
        await asyncio.sleep(0.01)

    assert calls == 1
    gate.set()

    assert await t1 == (True, None)
    assert await t2 == (True, None)
    assert manager._ensure_inflight is None


@pytest.mark.asyncio
async def test_ensure_ghost_display_retries_after_failure(monkeypatch):
    """A failed ensure should clear inflight state so later calls can retry."""
    manager = _fresh_manager()
    calls = 0

    async def _failing_impl(self, wait_for_registration=True, max_wait_seconds=15.0):
        nonlocal calls
        calls += 1
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(
        phm.PhantomHardwareManager,
        "_ensure_ghost_display_exists_impl",
        _failing_impl,
    )

    with pytest.raises(RuntimeError, match="simulated failure"):
        await manager.ensure_ghost_display_exists_async()

    assert manager._ensure_inflight is None

    with pytest.raises(RuntimeError, match="simulated failure"):
        await manager.ensure_ghost_display_exists_async()

    assert calls == 2


@pytest.mark.asyncio
async def test_get_display_status_async_alias(monkeypatch):
    """Compatibility alias should forward to get_status_async."""
    manager = _fresh_manager()
    expected = phm.PhantomHardwareStatus(
        cli_available=True,
        ghost_display_active=True,
    )

    async def _fake_status():
        return expected

    monkeypatch.setattr(manager, "get_status_async", _fake_status)

    result = await manager.get_display_status_async()
    assert result is expected
