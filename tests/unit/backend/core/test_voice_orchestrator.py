# tests/unit/backend/core/test_voice_orchestrator.py
"""Tests for VoiceOrchestrator - IPC server and collector."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from backend.core.voice_orchestrator import VoiceOrchestrator, VoiceMessage


@pytest.fixture
def temp_socket_dir():
    """Create temporary directory for socket."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_orchestrator_starts_and_stops(temp_socket_dir, monkeypatch):
    """Test basic lifecycle."""
    socket_path = temp_socket_dir / "voice.sock"
    monkeypatch.setenv("VOICE_SOCKET_PATH", str(socket_path))

    orchestrator = VoiceOrchestrator()

    await orchestrator.start()
    assert socket_path.exists()
    assert orchestrator._running

    await orchestrator.stop()
    assert not orchestrator._running


@pytest.mark.asyncio
async def test_orchestrator_receives_message(temp_socket_dir, monkeypatch):
    """Test that orchestrator receives messages via IPC."""
    socket_path = temp_socket_dir / "voice.sock"
    monkeypatch.setenv("VOICE_SOCKET_PATH", str(socket_path))

    orchestrator = VoiceOrchestrator()
    received_messages = []

    # Mock the collector to capture messages
    original_collect = orchestrator._collect_message
    async def capture_collect(msg):
        received_messages.append(msg)
        await original_collect(msg)
    orchestrator._collect_message = capture_collect

    await orchestrator.start()

    # Connect and send a message
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    msg = json.dumps({
        "text": "Test message",
        "priority": "NORMAL",
        "category": "test",
        "source": "test_client",
    }) + "\n"
    writer.write(msg.encode())
    await writer.drain()
    writer.close()
    await writer.wait_closed()

    # Wait for message to be processed
    await asyncio.sleep(0.1)

    await orchestrator.stop()

    assert len(received_messages) == 1
    assert received_messages[0].text == "Test message"


@pytest.mark.asyncio
async def test_in_process_announce(temp_socket_dir, monkeypatch):
    """Test that in-process announce works without socket."""
    socket_path = temp_socket_dir / "voice.sock"
    monkeypatch.setenv("VOICE_SOCKET_PATH", str(socket_path))

    orchestrator = VoiceOrchestrator()
    received = []

    async def capture(msg):
        received.append(msg)
    orchestrator._collect_message = capture

    await orchestrator.start()

    # In-process announce (no socket hop)
    await orchestrator.announce("Kernel message", "HIGH", "kernel")

    await asyncio.sleep(0.05)
    await orchestrator.stop()

    assert len(received) == 1
    assert received[0].text == "Kernel message"
    assert received[0].source == "kernel"


@pytest.mark.asyncio
async def test_connection_limit(temp_socket_dir, monkeypatch):
    """Test that connection limit is enforced."""
    socket_path = temp_socket_dir / "voice.sock"
    monkeypatch.setenv("VOICE_SOCKET_PATH", str(socket_path))
    monkeypatch.setenv("VOICE_MAX_CONNECTIONS", "2")

    orchestrator = VoiceOrchestrator()
    await orchestrator.start()

    # Open 2 connections (at limit)
    conn1_r, conn1_w = await asyncio.open_unix_connection(str(socket_path))
    conn2_r, conn2_w = await asyncio.open_unix_connection(str(socket_path))

    # Third should get rejected
    conn3_r, conn3_w = await asyncio.open_unix_connection(str(socket_path))
    response = await asyncio.wait_for(conn3_r.readline(), timeout=1.0)
    data = json.loads(response)
    assert data.get("error") == "busy"

    # Cleanup
    for w in [conn1_w, conn2_w, conn3_w]:
        w.close()
        try:
            await w.wait_closed()
        except Exception:
            pass

    await orchestrator.stop()
