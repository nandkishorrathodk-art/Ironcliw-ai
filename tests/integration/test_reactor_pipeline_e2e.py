"""
End-to-end integration test for the Reactor Core training pipeline.

Requires: Reactor-Core running on port 8090
Marks: @pytest.mark.integration, @pytest.mark.e2e

These tests verify the full pipeline loop works when Reactor-Core is live:
1. Experiences can be streamed to Reactor-Core
2. Health endpoint reports training readiness
3. ReactorCoreClient endpoint paths match server routes (no 404s)
"""

from __future__ import annotations

import aiohttp
import pytest
from datetime import datetime


REACTOR_URL = "http://localhost:8090"


@pytest.mark.integration
@pytest.mark.e2e
class TestReactorPipelineE2E:
    """End-to-end tests that verify the full pipeline loop."""

    @pytest.mark.asyncio
    async def test_experience_accepted_by_reactor(self):
        """Verify a manually sent experience is accepted by Reactor-Core."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "experience": {
                    "event_type": "INTERACTION",
                    "user_input": "What is 2+2?",
                    "assistant_output": "4",
                    "source": "JARVIS_BODY",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.95,
                    "task_type": "math_simple",
                },
                "timestamp": datetime.now().isoformat(),
                "source": "jarvis_agent",
            }
            async with session.post(
                f"{REACTOR_URL}/api/v1/experiences/stream",
                json=payload,
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["accepted"] is True
                assert data["count"] >= 1

    @pytest.mark.asyncio
    async def test_reactor_health_reports_training_ready(self):
        """Verify Reactor-Core health includes training readiness."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REACTOR_URL}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "training_ready" in data
                assert "phase" in data

    @pytest.mark.asyncio
    async def test_scout_topic_endpoint_accepts_topics(self):
        """Verify Scout topic enqueue endpoint exists and accepts valid payloads."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "topic": "Python asyncio cancellation best practices",
                "category": "general",
                "priority": 5,
                "urls": [],
                "added_by": "jarvis_contract_test",
            }
            async with session.post(f"{REACTOR_URL}/api/v1/scout/topics", json=payload) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert "added" in data
                assert "topic_id" in data

    @pytest.mark.asyncio
    async def test_client_paths_match_server(self):
        """Verify ReactorCoreClient endpoint paths resolve (no 404s)."""
        from backend.clients.reactor_core_client import (
            ReactorCoreClient,
            ReactorCoreConfig,
        )

        config = ReactorCoreConfig(api_url=REACTOR_URL)
        client = ReactorCoreClient(config)
        await client.initialize()
        try:
            # Health should work
            healthy = await client.health_check()
            assert healthy is True

            # Experience count should not 404
            count = await client.get_experience_count()
            assert isinstance(count, int)
        finally:
            await client.close()
