# tests/unit/backend/core/test_voice_coalescer.py
"""Tests for voice message coalescing."""

import asyncio
import pytest

from backend.core.voice_orchestrator import (
    VoiceCoalescer,
    VoiceMessage,
    VoicePriority,
    CATEGORY_PRIORITY,
)


@pytest.mark.asyncio
async def test_coalescer_batches_messages():
    """Test that coalescer batches messages within window."""
    results = []

    async def on_flush(summary: str):
        results.append(summary)

    coalescer = VoiceCoalescer(on_flush=on_flush, window_ms=100, idle_ms=50)
    await coalescer.start()

    # Add messages
    await coalescer.add(VoiceMessage("Init 1", VoicePriority.NORMAL, "init", "test"))
    await coalescer.add(VoiceMessage("Init 2", VoicePriority.NORMAL, "init", "test"))
    await coalescer.add(VoiceMessage("Init 3", VoicePriority.NORMAL, "init", "test"))

    # Wait for flush
    await asyncio.sleep(0.2)
    await coalescer.stop()

    assert len(results) == 1
    assert "3" in results[0]  # Should mention count


@pytest.mark.asyncio
async def test_coalescer_flushes_on_idle():
    """Test that coalescer flushes early when idle."""
    results = []
    flush_times = []

    async def on_flush(summary: str):
        results.append(summary)
        flush_times.append(asyncio.get_event_loop().time())

    coalescer = VoiceCoalescer(on_flush=on_flush, window_ms=1000, idle_ms=50)
    await coalescer.start()

    start = asyncio.get_event_loop().time()
    await coalescer.add(VoiceMessage("Single", VoicePriority.NORMAL, "init", "test"))

    # Wait for idle flush (should be < 1000ms window)
    await asyncio.sleep(0.2)
    await coalescer.stop()

    assert len(results) == 1
    assert flush_times[0] - start < 0.5  # Flushed early due to idle


@pytest.mark.asyncio
async def test_supersession_drops_lower_priority():
    """Test that shutdown supersedes ready messages."""
    results = []

    async def on_flush(summary: str):
        results.append(summary)

    coalescer = VoiceCoalescer(on_flush=on_flush, window_ms=100, idle_ms=50)
    await coalescer.start()

    # Add ready, then shutdown
    await coalescer.add(VoiceMessage("System ready", VoicePriority.NORMAL, "ready", "test"))
    await coalescer.add(VoiceMessage("Shutting down", VoicePriority.HIGH, "shutdown", "test"))

    await asyncio.sleep(0.2)
    await coalescer.stop()

    assert len(results) == 1
    assert "shutdown" in results[0].lower() or "Shutting" in results[0]
