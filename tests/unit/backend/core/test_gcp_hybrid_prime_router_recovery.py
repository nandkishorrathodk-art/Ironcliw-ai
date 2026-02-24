import asyncio
import time

import pytest

from backend.core.gcp_hybrid_prime_router import (
    EMERGENCY_UNLOAD_RAM_PERCENT,
    GCPHybridPrimeRouter,
)


class _FakeModelServing:
    def __init__(self, load_ok=True, smoke_ok=True):
        self._load_ok = load_ok
        self._smoke_ok = smoke_ok
        self.force_open_calls = 0
        self.reset_calls = 0
        self._state = "open"

    def force_open_local_circuit_breaker(self, reason=""):
        self.force_open_calls += 1
        self._state = "open"

    def get_local_circuit_state(self):
        return self._state

    async def load_model(self):
        return self._load_ok

    async def smoke_test_local_model(self, timeout_seconds=20.0):
        return self._smoke_ok

    def reset_local_circuit_breaker(self):
        self.reset_calls += 1
        self._state = "closed"


@pytest.mark.asyncio
async def test_post_crisis_recovery_reloads_and_closes_circuit(monkeypatch):
    router = GCPHybridPrimeRouter()
    router._model_needs_recovery = True
    router._recovery_stable_since = time.time() - 65
    now = time.time()
    router._recovery_percent_history.clear()
    router._recovery_percent_history.extend([(now - 30, 60.0), (now, 60.0)])

    fake_serving = _FakeModelServing(load_ok=True, smoke_ok=True)

    async def _fake_get_model_serving():
        return fake_serving

    async def _fake_signal(*_args, **_kwargs):
        return None

    async def _fake_ram_info():
        return {"used_percent": 60.0, "used_mb": 1024}

    import backend.intelligence.unified_model_serving as ums

    monkeypatch.setattr(ums, "get_model_serving", _fake_get_model_serving)
    monkeypatch.setattr(router, "_signal_memory_pressure_to_repos", _fake_signal)
    monkeypatch.setattr(router, "_get_ram_info_with_mb", _fake_ram_info)

    await router._check_model_recovery(current_used_percent=60.0)
    assert router._recovery_task is not None
    await router._recovery_task

    assert fake_serving.force_open_calls == 1
    assert fake_serving.reset_calls == 1
    assert router._model_needs_recovery is False
    assert router._local_circuit_state == "closed"


@pytest.mark.asyncio
async def test_recovery_failure_enters_backoff_without_tight_loop(monkeypatch):
    router = GCPHybridPrimeRouter()
    router._model_needs_recovery = True
    router._recovery_stable_since = time.time() - 65
    now = time.time()
    router._recovery_percent_history.clear()
    router._recovery_percent_history.extend([(now - 30, 62.0), (now, 62.0)])

    fake_serving = _FakeModelServing(load_ok=False, smoke_ok=False)

    async def _fake_get_model_serving():
        return fake_serving

    async def _fake_signal(*_args, **_kwargs):
        return None

    import backend.intelligence.unified_model_serving as ums

    monkeypatch.setattr(ums, "get_model_serving", _fake_get_model_serving)
    monkeypatch.setattr(router, "_signal_memory_pressure_to_repos", _fake_signal)

    await router._check_model_recovery(current_used_percent=62.0)
    assert router._recovery_task is not None
    await router._recovery_task

    assert router._recovery_attempts == 1
    assert router._recovery_cooldown_until > time.time()
    cooldown_deadline = router._recovery_cooldown_until

    # Immediate re-check should respect cooldown and not schedule another attempt.
    await router._check_model_recovery(current_used_percent=62.0)
    assert router._recovery_attempts == 1
    assert router._recovery_cooldown_until == cooldown_deadline


@pytest.mark.asyncio
async def test_recovery_aborts_when_memory_returns_critical(monkeypatch):
    router = GCPHybridPrimeRouter()
    router._model_needs_recovery = True
    router._recovery_stable_since = time.time() - 40

    router._recovery_task = asyncio.create_task(asyncio.sleep(10.0))

    await router._check_model_recovery(current_used_percent=EMERGENCY_UNLOAD_RAM_PERCENT + 1.0)
    await asyncio.sleep(0)

    assert router._recovery_stable_since == 0.0
    assert router._recovery_cooldown_until > time.time()
    assert router._recovery_task.cancelled() or router._recovery_task.done()
