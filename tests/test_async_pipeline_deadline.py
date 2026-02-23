"""Tests for deadline-aware async pipeline behavior."""

import time
from unittest.mock import patch

import pytest

from backend.core.async_pipeline import AdvancedAsyncPipeline


class _FakeProcessor:
    def __init__(self):
        self.deadline_seen = None

    async def process_command(self, *args, **kwargs):
        self.deadline_seen = kwargs.get("deadline")
        return {"success": True, "response": "ok"}


@pytest.mark.asyncio
async def test_pipeline_propagates_deadline_to_unified_processor():
    pipeline = AdvancedAsyncPipeline(
        config={
            "follow_up_enabled": False,
            "agi_os_enabled": False,
            "vbi_health_monitor_enabled": False,
        }
    )
    fake = _FakeProcessor()
    deadline = time.monotonic() + 10.0

    with patch(
        "api.unified_command_processor.get_unified_processor",
        return_value=fake,
    ):
        result = await pipeline.process_async(
            "what's happening across my workspace",
            metadata={
                "screen_just_unlocked": True,  # skip proactive unlock path
                "deadline_monotonic": deadline,
            },
        )

    assert result["success"] is True
    assert fake.deadline_seen is not None
    assert abs(fake.deadline_seen - deadline) < 0.05
