from __future__ import annotations

import asyncio

import pytest

from backend.main import _await_first_tts_success


@pytest.mark.asyncio
async def test_waits_for_later_success_instead_of_fast_failure():
    async def _fast_none():
        await asyncio.sleep(0.01)
        return None

    async def _slow_success():
        await asyncio.sleep(0.03)
        return "audio-response"

    tasks = {
        "fast_none": asyncio.create_task(_fast_none()),
        "slow_success": asyncio.create_task(_slow_success()),
    }

    result = await _await_first_tts_success(tasks, timeout_seconds=1.0)
    assert result == "audio-response"


@pytest.mark.asyncio
async def test_returns_none_when_all_strategies_fail_or_timeout():
    async def _fast_none():
        await asyncio.sleep(0.01)
        return None

    async def _raises():
        await asyncio.sleep(0.01)
        raise RuntimeError("tts failure")

    tasks = {
        "fast_none": asyncio.create_task(_fast_none()),
        "raises": asyncio.create_task(_raises()),
    }

    result = await _await_first_tts_success(tasks, timeout_seconds=0.2)
    assert result is None
