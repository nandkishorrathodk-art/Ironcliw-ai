# backend/core/voice_client.py
"""
VoiceClient - Cross-repo voice announcement client.

Sends announcements to the VoiceOrchestrator via Unix domain socket.
Queues locally when disconnected and drains on reconnect.

Usage:
    client = VoiceClient(source="jarvis-prime")
    await client.start()  # Start reconnect loop

    await client.announce("System ready", VoicePriority.NORMAL, "init")

    await client.stop()  # Cleanup
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

logger = logging.getLogger(__name__)


class VoicePriority(Enum):
    """Priority levels for voice announcements."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


@dataclass
class QueuedMessage:
    """Message queued locally when disconnected."""
    text: str
    priority: VoicePriority
    category: str
    timestamp: float


class VoiceClient:
    """
    Cross-repo voice client. Sends to VoiceOrchestrator via Unix socket.

    Features:
    - Bounded local queue with drop-oldest policy
    - Reconnect loop with exponential backoff
    - Coalesces large queues on reconnect
    """

    def __init__(self, source: str):
        """
        Initialize client.

        Args:
            source: Identifier for this client (e.g., "jarvis", "prime", "reactor")
        """
        self._source = source
        self._socket_path = self._get_socket_path()
        self._connected = False
        self._writer: Optional[asyncio.StreamWriter] = None
        self._reader: Optional[asyncio.StreamReader] = None

        # Bounded local queue with drop-oldest policy
        max_size = int(os.environ.get("VOICE_CLIENT_QUEUE_MAX", "20"))
        self._local_queue: Deque[QueuedMessage] = deque(maxlen=max_size)
        self._dropped_count = 0

        # Reconnect task
        self._reconnect_task: Optional[asyncio.Task] = None
        self._shutdown = False

    def _get_socket_path(self) -> str:
        """Get socket path with expansion."""
        raw = os.environ.get("VOICE_SOCKET_PATH", "~/.jarvis/voice.sock")
        return os.path.expanduser(os.path.expandvars(raw))

    async def start(self) -> None:
        """Start the reconnect loop."""
        self._shutdown = False
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def stop(self) -> None:
        """Stop client gracefully."""
        self._shutdown = True
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        if self._writer:
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except Exception:
                pass

    async def announce(
        self,
        text: str,
        priority: VoicePriority = VoicePriority.NORMAL,
        category: str = "general",
    ) -> bool:
        """
        Announce a message.

        Args:
            text: Message to announce
            priority: Priority level
            category: Category for coalescing (e.g., "init", "health", "error")

        Returns:
            True if sent immediately, False if queued locally
        """
        if self._connected:
            success = await self._send(text, priority, category)
            if success:
                return True
            # Send failed - disconnect and queue
            self._connected = False

        self._queue_locally(text, priority, category)
        return False

    def _queue_locally(self, text: str, priority: VoicePriority, category: str) -> None:
        """Queue message locally. Drops oldest if full."""
        if len(self._local_queue) == self._local_queue.maxlen:
            self._dropped_count += 1
        self._local_queue.append(
            QueuedMessage(text=text, priority=priority, category=category, timestamp=time.time())
        )

    def _format_message(self, text: str, priority: VoicePriority, category: str) -> str:
        """Format message as JSON-line."""
        data = {
            "text": text,
            "priority": priority.value,
            "category": category,
            "source": self._source,
            "timestamp": time.time(),
        }
        return json.dumps(data) + "\n"

    async def _send(self, text: str, priority: VoicePriority, category: str) -> bool:
        """Send to socket. Returns False on failure."""
        if not self._writer:
            return False

        try:
            line = self._format_message(text, priority, category)
            self._writer.write(line.encode())
            await self._writer.drain()
            return True
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            logger.debug(f"[VoiceClient] Send failed: {e}")
            return False

    async def _reconnect_loop(self) -> None:
        """Background task: reconnect with exponential backoff."""
        backoff = 1.0
        max_backoff = float(os.environ.get("VOICE_CLIENT_MAX_BACKOFF_S", "30"))

        while not self._shutdown:
            if not self._connected:
                try:
                    self._reader, self._writer = await asyncio.open_unix_connection(
                        self._socket_path
                    )
                    self._connected = True
                    backoff = 1.0
                    logger.info(f"[VoiceClient] Connected to {self._socket_path}")
                    await self._drain_on_reconnect()
                except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
                    logger.debug(f"[VoiceClient] Connect failed: {e}")
                    self._connected = False
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
            else:
                await asyncio.sleep(1.0)

    async def _drain_on_reconnect(self) -> None:
        """Drain local queue on reconnect. Coalesces if large."""
        queue_size = len(self._local_queue)

        if queue_size == 0:
            return

        if queue_size > 5:
            # Coalesce: send summary instead of flooding
            summary = f"{self._source} reconnected with {queue_size} pending messages"
            await self._send(summary, VoicePriority.LOW, "reconnect")
            self._local_queue.clear()
            logger.info(f"[VoiceClient] Coalesced {queue_size} pending messages")
        else:
            # Drain individually with rate limit
            rate_limit_ms = int(os.environ.get("VOICE_CLIENT_DRAIN_RATE_MS", "100"))
            while self._local_queue:
                msg = self._local_queue.popleft()
                success = await self._send(msg.text, msg.priority, msg.category)
                if not success:
                    self._local_queue.appendleft(msg)
                    self._connected = False
                    break
                await asyncio.sleep(rate_limit_ms / 1000)


# Convenience function for simple usage
_default_client: Optional[VoiceClient] = None


async def announce(
    text: str,
    priority: VoicePriority = VoicePriority.NORMAL,
    category: str = "general",
    source: str = "jarvis",
) -> bool:
    """
    Announce via the default client.

    Creates and starts a client on first call.
    """
    global _default_client
    if _default_client is None:
        _default_client = VoiceClient(source=source)
        await _default_client.start()
    return await _default_client.announce(text, priority, category)
