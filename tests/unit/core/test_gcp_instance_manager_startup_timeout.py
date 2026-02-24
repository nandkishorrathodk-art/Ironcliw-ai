import asyncio

import pytest

from unified_supervisor import GCPInstanceManager


@pytest.mark.asyncio
async def test_gcp_client_init_timeout_is_bounded(monkeypatch):
    monkeypatch.setenv("GCP_ENABLED", "true")
    monkeypatch.setenv("GCP_PROJECT_ID", "jarvis-test")

    async def _slow_to_thread(_fn, *args, **kwargs):
        await asyncio.sleep(0.2)
        return None

    monkeypatch.setattr("unified_supervisor.asyncio.to_thread", _slow_to_thread)

    manager = GCPInstanceManager()
    manager.client_init_timeout = 0.05

    with pytest.raises(TimeoutError):
        await manager._initialize_clients()

    assert manager._compute_client is None
    assert manager._run_client is None


@pytest.mark.asyncio
async def test_gcp_initialize_gracefully_degrades_on_client_timeout(monkeypatch):
    monkeypatch.setenv("GCP_ENABLED", "true")
    monkeypatch.setenv("GCP_PROJECT_ID", "jarvis-test")

    async def _slow_to_thread(_fn, *args, **kwargs):
        await asyncio.sleep(0.2)
        return None

    monkeypatch.setattr("unified_supervisor.asyncio.to_thread", _slow_to_thread)

    manager = GCPInstanceManager()
    manager.client_init_timeout = 0.05

    result = await manager.initialize()

    assert result is True
    assert manager._initialized is True
    assert manager._error is not None
    assert "timed out" in manager._error.lower()
