# backend/core/voice_orchestrator.py
"""
VoiceOrchestrator - Unified voice coordination for JARVIS ecosystem.

Single playback authority that:
- Receives announcements via Unix domain socket (IPC)
- Collects and coalesces announcements
- Serializes playback (one voice at a time)
- Provides metrics for observability

Architecture:
    IPC Receiver -> Bounded Collector -> Coalescer -> Serialized Speaker
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration - Defaults (read fresh from env at instantiation time)
# =============================================================================

DEFAULT_VOICE_SOCKET_PATH = "~/.jarvis/voice.sock"
DEFAULT_VOICE_SOCKET_MODE = "0600"
DEFAULT_VOICE_MAX_CONNECTIONS = "20"
DEFAULT_VOICE_READ_TIMEOUT_MS = "5000"
DEFAULT_VOICE_MAX_MESSAGE_LENGTH = "1000"
DEFAULT_VOICE_QUEUE_MAX_SIZE = "50"


def _get_config_int(key: str, default: str) -> int:
    """Get integer config from environment."""
    return int(os.environ.get(key, default))


def _get_config_float(key: str, default: str) -> float:
    """Get float config from environment."""
    return float(os.environ.get(key, default))


class VoicePriority(Enum):
    """Priority levels for voice messages."""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25


@dataclass
class VoiceMessage:
    """A voice announcement message."""
    text: str
    priority: VoicePriority
    category: str
    source: str
    timestamp: float = field(default_factory=time.time)
    max_length: int = field(default=1000, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], max_length: int = 1000) -> "VoiceMessage":
        """Create from dictionary (IPC message)."""
        priority_str = data.get("priority", "NORMAL")
        try:
            priority = VoicePriority[priority_str]
        except KeyError:
            priority = VoicePriority.NORMAL

        return cls(
            text=data.get("text", "")[:max_length],
            priority=priority,
            category=data.get("category", "general"),
            source=data.get("source", "unknown"),
            timestamp=data.get("timestamp", time.time()),
            max_length=max_length,
        )


@dataclass
class VoiceMetrics:
    """Metrics for observability."""
    queue_depth: int = 0
    coalesced_count: int = 0
    spoken_count: int = 0
    dropped_count: int = 0
    interrupt_count: int = 0
    last_spoken_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue_depth": self.queue_depth,
            "coalesced_count": self.coalesced_count,
            "spoken_count": self.spoken_count,
            "dropped_count": self.dropped_count,
            "interrupt_count": self.interrupt_count,
            "last_spoken_at": self.last_spoken_at,
        }


class VoiceOrchestrator:
    """
    Unified voice orchestrator - single playback authority.

    Receives announcements via IPC socket and in-process calls,
    coalesces them intelligently, and plays them one at a time.
    """

    def __init__(self):
        """Initialize orchestrator."""
        # Read config fresh from environment at instantiation time
        self._socket_path = self._get_socket_path()
        self._socket_mode = int(os.environ.get("VOICE_SOCKET_MODE", DEFAULT_VOICE_SOCKET_MODE), 8)
        self._max_connections = _get_config_int("VOICE_MAX_CONNECTIONS", DEFAULT_VOICE_MAX_CONNECTIONS)
        self._read_timeout_s = _get_config_float("VOICE_READ_TIMEOUT_MS", DEFAULT_VOICE_READ_TIMEOUT_MS) / 1000
        self._max_message_length = _get_config_int("VOICE_MAX_MESSAGE_LENGTH", DEFAULT_VOICE_MAX_MESSAGE_LENGTH)
        self._queue_max_size = _get_config_int("VOICE_QUEUE_MAX_SIZE", DEFAULT_VOICE_QUEUE_MAX_SIZE)

        self._server: Optional[asyncio.Server] = None
        self._running = False
        self._shutting_down = False

        # Connection limiting
        self._active_connections = 0
        self._connection_lock = asyncio.Lock()

        # Message queue
        self._queue: asyncio.Queue[VoiceMessage] = asyncio.Queue(maxsize=self._queue_max_size)

        # Metrics
        self._metrics = VoiceMetrics()

        # Tasks
        self._consumer_task: Optional[asyncio.Task] = None

        # TTS callback (set by kernel)
        self._tts_callback: Optional[Callable[[str], asyncio.Future]] = None

    def _get_socket_path(self) -> Path:
        """Get socket path with expansion."""
        raw = os.environ.get("VOICE_SOCKET_PATH", DEFAULT_VOICE_SOCKET_PATH)
        return Path(os.path.expanduser(os.path.expandvars(raw)))

    async def start(self) -> None:
        """Start the orchestrator."""
        if self._running:
            return

        # Ensure parent directory exists
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket
        try:
            self._socket_path.unlink()
            logger.info(f"[Voice] Removed stale socket: {self._socket_path}")
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.error(f"[Voice] Failed to unlink socket: {e}")
            raise

        # Start IPC server
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self._socket_path),
        )

        # Set permissions
        try:
            os.chmod(self._socket_path, self._socket_mode)
        except OSError as e:
            logger.warning(f"[Voice] chmod failed: {e}")

        # Start consumer task
        self._consumer_task = asyncio.create_task(self._consumer_loop())

        self._running = True
        logger.info(f"[Voice] Orchestrator started: {self._socket_path}")

    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        self._shutting_down = True
        self._running = False

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Stop consumer
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        # Remove socket
        try:
            self._socket_path.unlink()
        except (FileNotFoundError, OSError):
            pass

        logger.info("[Voice] Orchestrator stopped")

    async def announce(
        self,
        text: str,
        priority: str = "NORMAL",
        category: str = "kernel",
    ) -> None:
        """
        In-process announce (no socket hop).

        For kernel messages that originate within the same process.
        """
        try:
            prio = VoicePriority[priority]
        except KeyError:
            prio = VoicePriority.NORMAL

        msg = VoiceMessage(
            text=text[:self._max_message_length],
            priority=prio,
            category=category,
            source="kernel",
        )
        await self._collect_message(msg)

    async def _collect_message(self, msg: VoiceMessage) -> None:
        """Add message to queue (drops oldest if full)."""
        try:
            self._queue.put_nowait(msg)
            self._metrics.queue_depth = self._queue.qsize()
        except asyncio.QueueFull:
            # Drop oldest (get and discard)
            try:
                self._queue.get_nowait()
                self._metrics.dropped_count += 1
            except asyncio.QueueEmpty:
                pass
            # Try again
            try:
                self._queue.put_nowait(msg)
            except asyncio.QueueFull:
                self._metrics.dropped_count += 1

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection."""
        # Check connection limit
        async with self._connection_lock:
            if self._active_connections >= self._max_connections:
                # Reject
                writer.write(b'{"error":"busy","retry_after_ms":1000}\n')
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return
            self._active_connections += 1

        try:
            await self._process_client(reader, writer)
        finally:
            async with self._connection_lock:
                self._active_connections -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _process_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Process messages from a client."""
        while not self._shutting_down:
            try:
                line = await asyncio.wait_for(
                    reader.readline(),
                    timeout=self._read_timeout_s,
                )

                if not line:
                    break  # Client disconnected

                # Parse JSON
                try:
                    data = json.loads(line.decode().strip())
                    msg = VoiceMessage.from_dict(data, max_length=self._max_message_length)
                    await self._collect_message(msg)
                except json.JSONDecodeError as e:
                    logger.warning(f"[Voice] Invalid JSON from client: {e}")
                    continue

            except asyncio.TimeoutError:
                continue  # Keep connection alive
            except (ConnectionResetError, BrokenPipeError):
                break

    async def _consumer_loop(self) -> None:
        """Consume messages from queue and speak them."""
        while not self._shutting_down:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                self._metrics.queue_depth = self._queue.qsize()

                # Speak the message
                if self._tts_callback:
                    try:
                        await self._tts_callback(msg.text)
                        self._metrics.spoken_count += 1
                        self._metrics.last_spoken_at = time.time()
                    except Exception as e:
                        logger.error(f"[Voice] TTS error: {e}")
                else:
                    # No TTS - just log
                    logger.info(f"[Voice] Would speak: {msg.text}")
                    self._metrics.spoken_count += 1

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def set_tts_callback(self, callback: Callable[[str], asyncio.Future]) -> None:
        """Set the TTS callback for actual speech synthesis."""
        self._tts_callback = callback

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self._metrics.to_dict()
