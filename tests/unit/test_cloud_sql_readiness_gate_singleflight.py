import asyncio
import types

import pytest

import backend.intelligence.cloud_sql_connection_manager as gate_module
from backend.intelligence.cloud_sql_connection_manager import (
    ProxyReadinessGate,
    ReadinessResult,
    ReadinessState,
)


@pytest.fixture(autouse=True)
def reset_proxy_gate_singletons():
    gate_module.ProxyReadinessGate._instance = None
    gate_module._readiness_gate = None
    yield
    gate_module.ProxyReadinessGate._instance = None
    gate_module._readiness_gate = None


@pytest.mark.asyncio
async def test_ensure_proxy_ready_uses_single_flight(monkeypatch):
    gate = ProxyReadinessGate()
    call_counter = {"count": 0}

    async def fake_internal(
        self,
        timeout=None,
        auto_start=True,
        max_start_attempts=3,
        notify_cross_repo=True,
    ):
        call_counter["count"] += 1
        await asyncio.sleep(0.05)
        return ReadinessResult(
            state=ReadinessState.READY,
            timed_out=False,
            message="ready",
        )

    monkeypatch.setattr(
        gate,
        "_ensure_proxy_ready_internal",
        types.MethodType(fake_internal, gate),
    )

    results = await asyncio.gather(
        *[gate.ensure_proxy_ready(timeout=1.0) for _ in range(5)]
    )

    assert call_counter["count"] == 1
    assert all(result.state == ReadinessState.READY for result in results)
    assert gate._ensure_proxy_ready_future is None


@pytest.mark.asyncio
async def test_ensure_proxy_ready_joiner_timeout_does_not_cancel_leader(monkeypatch):
    gate = ProxyReadinessGate()
    internal_started = asyncio.Event()

    async def fake_internal(
        self,
        timeout=None,
        auto_start=True,
        max_start_attempts=3,
        notify_cross_repo=True,
    ):
        internal_started.set()
        await asyncio.sleep(0.2)
        return ReadinessResult(
            state=ReadinessState.READY,
            timed_out=False,
            message="ready",
        )

    monkeypatch.setattr(
        gate,
        "_ensure_proxy_ready_internal",
        types.MethodType(fake_internal, gate),
    )

    leader = asyncio.create_task(gate.ensure_proxy_ready(timeout=1.0))
    await asyncio.wait_for(internal_started.wait(), timeout=0.2)

    joiner = await gate.ensure_proxy_ready(timeout=0.02)
    assert joiner.timed_out is True
    assert joiner.failure_reason == "timeout"

    leader_result = await leader
    assert leader_result.state == ReadinessState.READY


@pytest.mark.asyncio
async def test_ensure_proxy_ready_returns_shutdown_result_when_cancelled(monkeypatch):
    gate = ProxyReadinessGate()
    block_event = asyncio.Event()

    async def fake_internal(
        self,
        timeout=None,
        auto_start=True,
        max_start_attempts=3,
        notify_cross_repo=True,
    ):
        await block_event.wait()
        return ReadinessResult(
            state=ReadinessState.READY,
            timed_out=False,
            message="ready",
        )

    monkeypatch.setattr(
        gate,
        "_ensure_proxy_ready_internal",
        types.MethodType(fake_internal, gate),
    )

    ensure_task = asyncio.create_task(gate.ensure_proxy_ready(timeout=5.0))
    await asyncio.sleep(0.02)

    await gate.shutdown()
    result = await ensure_task

    assert result.state == ReadinessState.UNAVAILABLE
    assert result.failure_reason == "shutdown"
    assert gate._ensure_proxy_ready_future is None
