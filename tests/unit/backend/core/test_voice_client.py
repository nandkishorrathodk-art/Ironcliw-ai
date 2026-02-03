# tests/unit/backend/core/test_voice_client.py
"""Tests for VoiceClient - Cross-repo voice announcement client."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.core.voice_client import VoiceClient, VoicePriority


@pytest.fixture
def temp_socket_dir():
    """Create temporary directory for socket."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_client_queues_when_disconnected(temp_socket_dir, monkeypatch):
    """Test that client queues messages locally when disconnected."""
    socket_path = temp_socket_dir / "voice.sock"
    monkeypatch.setenv("VOICE_SOCKET_PATH", str(socket_path))

    client = VoiceClient(source="test")

    # Don't start reconnect loop - stay disconnected
    assert not client._connected

    # Send some messages
    await client.announce("Message 1", VoicePriority.NORMAL, "init")
    await client.announce("Message 2", VoicePriority.NORMAL, "init")

    # Should be queued locally
    assert len(client._local_queue) == 2


@pytest.mark.asyncio
async def test_client_drops_oldest_when_full(temp_socket_dir, monkeypatch):
    """Test that client drops oldest message when queue is full."""
    socket_path = temp_socket_dir / "voice.sock"
    monkeypatch.setenv("VOICE_SOCKET_PATH", str(socket_path))
    monkeypatch.setenv("VOICE_CLIENT_QUEUE_MAX", "3")

    client = VoiceClient(source="test")

    # Fill queue beyond capacity
    await client.announce("Message 1", VoicePriority.NORMAL, "init")
    await client.announce("Message 2", VoicePriority.NORMAL, "init")
    await client.announce("Message 3", VoicePriority.NORMAL, "init")
    await client.announce("Message 4", VoicePriority.NORMAL, "init")  # Drops Message 1

    # Queue should have 3 (most recent)
    assert len(client._local_queue) == 3
    assert client._local_queue[0].text == "Message 2"
    assert client._dropped_count == 1


@pytest.mark.asyncio
async def test_message_format():
    """Test that message is formatted correctly as JSON-lines."""
    client = VoiceClient(source="jarvis")

    msg = client._format_message("Hello", VoicePriority.HIGH, "greeting")
    data = json.loads(msg.rstrip("\n"))

    assert data["text"] == "Hello"
    assert data["priority"] == "HIGH"
    assert data["category"] == "greeting"
    assert data["source"] == "jarvis"
    assert "timestamp" in data
