"""
v77.3: IDE WebSocket Handler
============================

Real-time bidirectional communication between IDE extension and JARVIS.

Features:
- Async WebSocket server
- Message routing and dispatching
- Connection pooling
- Heartbeat/keepalive
- Reconnection handling
- Rate limiting per connection
- Message compression

Protocol:
    IDE Extension <--WebSocket--> JARVIS Backend

    Messages are JSON with structure:
    {
        "type": "event_type",
        "id": "unique_message_id",
        "data": { ... }
    }

Author: JARVIS v77.3
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import time
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class WebSocketConfig:
    """WebSocket configuration."""

    PORT: int = int(os.getenv("IDE_WS_PORT", "8015"))
    HOST: str = os.getenv("IDE_WS_HOST", "0.0.0.0")
    MAX_CONNECTIONS: int = int(os.getenv("IDE_WS_MAX_CONNECTIONS", "10"))
    HEARTBEAT_INTERVAL: float = float(os.getenv("IDE_WS_HEARTBEAT", "30"))
    MESSAGE_SIZE_LIMIT: int = int(os.getenv("IDE_WS_MAX_SIZE", "1048576"))  # 1MB
    RATE_LIMIT_MESSAGES: int = int(os.getenv("IDE_WS_RATE_LIMIT", "100"))
    RATE_LIMIT_WINDOW: float = float(os.getenv("IDE_WS_RATE_WINDOW", "10"))
    COMPRESSION_THRESHOLD: int = int(os.getenv("IDE_WS_COMPRESS_THRESHOLD", "1024"))


# =============================================================================
# Message Types
# =============================================================================

class MessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    FILE_OPENED = "file_opened"
    FILE_CLOSED = "file_closed"
    FILE_CHANGED = "file_changed"
    FILE_SAVED = "file_saved"
    CURSOR_MOVED = "cursor_moved"
    SELECTION_CHANGED = "selection_changed"
    DIAGNOSTICS_UPDATED = "diagnostics_updated"
    REQUEST_SUGGESTION = "request_suggestion"
    SUGGESTION_FEEDBACK = "suggestion_feedback"
    COMMAND = "command"
    PING = "ping"

    # Server -> Client
    SUGGESTION = "suggestion"
    EVOLUTION_STATUS = "evolution_status"
    NOTIFICATION = "notification"
    ERROR = "error"
    PONG = "pong"
    CONTEXT_UPDATE = "context_update"


@dataclass
class WebSocketMessage:
    """A WebSocket message."""
    type: MessageType
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    compressed: bool = False

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "id": self.id,
            "data": self.data,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, data: str) -> "WebSocketMessage":
        obj = json.loads(data)
        return cls(
            type=MessageType(obj["type"]),
            data=obj.get("data", {}),
            id=obj.get("id", str(uuid.uuid4())[:8]),
            timestamp=obj.get("timestamp", time.time()),
        )


# =============================================================================
# Rate Limiter
# =============================================================================

class ConnectionRateLimiter:
    """Per-connection rate limiter."""

    def __init__(self, max_messages: int, window_seconds: float):
        self._max_messages = max_messages
        self._window = window_seconds
        self._timestamps: List[float] = []

    def check(self) -> bool:
        """Check if message is allowed. Returns True if allowed."""
        now = time.time()

        # Remove old timestamps
        self._timestamps = [
            t for t in self._timestamps
            if now - t < self._window
        ]

        # Check limit
        if len(self._timestamps) >= self._max_messages:
            return False

        self._timestamps.append(now)
        return True


# =============================================================================
# Connection Wrapper
# =============================================================================

@dataclass
class IDEConnection:
    """Wrapper for a WebSocket connection."""
    websocket: Any  # aiohttp.web.WebSocketResponse or similar
    session_id: str
    client_info: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    rate_limiter: ConnectionRateLimiter = field(default_factory=lambda: ConnectionRateLimiter(
        max_messages=WebSocketConfig.RATE_LIMIT_MESSAGES,
        window_seconds=WebSocketConfig.RATE_LIMIT_WINDOW,
    ))
    subscriptions: Set[str] = field(default_factory=set)

    async def send(self, message: WebSocketMessage) -> bool:
        """Send a message to this connection."""
        try:
            data = message.to_json()

            # Compress if large
            if len(data) > WebSocketConfig.COMPRESSION_THRESHOLD:
                compressed = gzip.compress(data.encode())
                if len(compressed) < len(data):
                    # Send as binary with compression flag
                    await self.websocket.send_bytes(b'\x01' + compressed)
                    return True

            await self.websocket.send_str(data)
            return True

        except Exception as e:
            logger.warning(f"[WebSocket] Send failed: {e}")
            return False

    async def close(self) -> None:
        """Close the connection."""
        try:
            await self.websocket.close()
        except Exception:
            pass


# =============================================================================
# Message Router
# =============================================================================

class MessageRouter:
    """Routes messages to appropriate handlers."""

    def __init__(self):
        self._handlers: Dict[MessageType, List[Callable]] = {}
        self._middleware: List[Callable] = []

    def register(
        self,
        message_type: MessageType,
        handler: Callable[[IDEConnection, WebSocketMessage], Coroutine]
    ) -> None:
        """Register a message handler."""
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)

    def add_middleware(
        self,
        middleware: Callable[[IDEConnection, WebSocketMessage], Coroutine[Any, Any, bool]]
    ) -> None:
        """Add middleware that runs before handlers. Return False to stop processing."""
        self._middleware.append(middleware)

    async def route(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> bool:
        """Route a message to its handlers."""
        # Run middleware
        for mw in self._middleware:
            try:
                result = await mw(connection, message)
                if result is False:
                    return False
            except Exception as e:
                logger.error(f"[Router] Middleware error: {e}")
                return False

        # Find handlers
        handlers = self._handlers.get(message.type, [])
        if not handlers:
            logger.debug(f"[Router] No handler for {message.type}")
            return True

        # Run handlers
        results = await asyncio.gather(
            *(h(connection, message) for h in handlers),
            return_exceptions=True
        )

        # Log errors
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"[Router] Handler error: {r}")

        return True


# =============================================================================
# WebSocket Handler
# =============================================================================

class IDEWebSocketHandler:
    """
    Main WebSocket handler for IDE connections.

    Manages:
    - Connection lifecycle
    - Message routing
    - Heartbeat monitoring
    - Broadcasting
    """

    def __init__(self):
        self._connections: Dict[str, IDEConnection] = {}
        self._router = MessageRouter()
        self._lock = asyncio.Lock()
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        # IDE Bridge reference
        self._ide_bridge: Optional[Any] = None

        # Setup default handlers
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        """Setup default message handlers."""

        # Ping/Pong
        self._router.register(MessageType.PING, self._handle_ping)

        # File events
        self._router.register(MessageType.FILE_OPENED, self._handle_file_opened)
        self._router.register(MessageType.FILE_CLOSED, self._handle_file_closed)
        self._router.register(MessageType.FILE_CHANGED, self._handle_file_changed)
        self._router.register(MessageType.FILE_SAVED, self._handle_file_saved)

        # Cursor events
        self._router.register(MessageType.CURSOR_MOVED, self._handle_cursor_moved)

        # Diagnostics
        self._router.register(MessageType.DIAGNOSTICS_UPDATED, self._handle_diagnostics)

        # Suggestions
        self._router.register(MessageType.REQUEST_SUGGESTION, self._handle_suggestion_request)
        self._router.register(MessageType.SUGGESTION_FEEDBACK, self._handle_suggestion_feedback)

        # Rate limiting middleware
        self._router.add_middleware(self._rate_limit_middleware)

    async def _rate_limit_middleware(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> bool:
        """Rate limiting middleware."""
        if not connection.rate_limiter.check():
            await connection.send(WebSocketMessage(
                type=MessageType.ERROR,
                data={"error": "rate_limited", "message": "Too many messages"},
            ))
            return False
        return True

    async def initialize(self) -> bool:
        """Initialize the handler."""
        try:
            # Get IDE bridge
            from .bridge import get_ide_bridge
            self._ide_bridge = await get_ide_bridge()

            self._running = True

            # Start heartbeat monitor
            self._heartbeat_task = asyncio.create_task(self._heartbeat_monitor())

            logger.info("[WebSocketHandler] Initialized")
            return True

        except Exception as e:
            logger.error(f"[WebSocketHandler] Init failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Close all connections
        async with self._lock:
            for conn in list(self._connections.values()):
                await conn.close()
            self._connections.clear()

        logger.info("[WebSocketHandler] Shutdown complete")

    async def handle_connection(self, websocket: Any, request: Any = None) -> None:
        """Handle a new WebSocket connection."""
        session_id = str(uuid.uuid4())

        # Check connection limit
        async with self._lock:
            if len(self._connections) >= WebSocketConfig.MAX_CONNECTIONS:
                await websocket.close(code=1013, message=b"Max connections reached")
                return

        # Create connection wrapper
        connection = IDEConnection(
            websocket=websocket,
            session_id=session_id,
            client_info={
                "remote": str(getattr(request, "remote", "unknown")) if request else "unknown",
            },
        )

        # Register connection
        async with self._lock:
            self._connections[session_id] = connection

        # Notify IDE bridge
        if self._ide_bridge:
            await self._ide_bridge.create_session(session_id, connection.client_info)

        logger.info(f"[WebSocket] New connection: {session_id}")

        try:
            # Send welcome message
            await connection.send(WebSocketMessage(
                type=MessageType.NOTIFICATION,
                data={
                    "message": "Connected to JARVIS Coding Council",
                    "session_id": session_id,
                    "capabilities": ["suggestions", "evolution", "trinity_sync"],
                },
            ))

            # Message loop
            async for msg in websocket:
                if msg.type == 1:  # TEXT
                    await self._handle_message(connection, msg.data)
                elif msg.type == 2:  # BINARY
                    # Decompress if needed
                    data = msg.data
                    if data[0:1] == b'\x01':
                        data = gzip.decompress(data[1:]).decode()
                    else:
                        data = data.decode()
                    await self._handle_message(connection, data)
                elif msg.type == 8:  # CLOSE
                    break

        except Exception as e:
            logger.error(f"[WebSocket] Connection error: {e}")

        finally:
            # Cleanup
            async with self._lock:
                if session_id in self._connections:
                    del self._connections[session_id]

            if self._ide_bridge:
                await self._ide_bridge.end_session(session_id)

            logger.info(f"[WebSocket] Connection closed: {session_id}")

    async def _handle_message(self, connection: IDEConnection, data: str) -> None:
        """Handle an incoming message."""
        try:
            message = WebSocketMessage.from_json(data)
            connection.last_activity = time.time()

            await self._router.route(connection, message)

        except json.JSONDecodeError as e:
            logger.warning(f"[WebSocket] Invalid JSON: {e}")
            await connection.send(WebSocketMessage(
                type=MessageType.ERROR,
                data={"error": "invalid_json", "message": str(e)},
            ))

        except Exception as e:
            logger.error(f"[WebSocket] Message handling error: {e}")

    async def _heartbeat_monitor(self) -> None:
        """Monitor connections and send heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(WebSocketConfig.HEARTBEAT_INTERVAL)

                async with self._lock:
                    now = time.time()
                    stale = []

                    for session_id, conn in self._connections.items():
                        # Check for stale connections
                        if now - conn.last_activity > WebSocketConfig.HEARTBEAT_INTERVAL * 3:
                            stale.append(session_id)
                            continue

                        # Send heartbeat
                        await conn.send(WebSocketMessage(type=MessageType.PONG))

                    # Remove stale connections
                    for session_id in stale:
                        conn = self._connections.pop(session_id)
                        await conn.close()
                        logger.info(f"[WebSocket] Removed stale connection: {session_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WebSocket] Heartbeat error: {e}")

    # -------------------------------------------------------------------------
    # Message Handlers
    # -------------------------------------------------------------------------

    async def _handle_ping(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> None:
        """Handle ping message."""
        await connection.send(WebSocketMessage(
            type=MessageType.PONG,
            data={"original_id": message.id},
        ))

    async def _handle_file_opened(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> None:
        """Handle file opened event."""
        if not self._ide_bridge:
            return

        data = message.data
        from .bridge import FileContext, CursorPosition

        file_context = FileContext(
            uri=data.get("uri", ""),
            path=data.get("path", ""),
            content=data.get("content", ""),
            language_id=data.get("languageId", ""),
            version=data.get("version", 1),
            is_active=data.get("isActive", False),
        )

        await self._ide_bridge.update_file(file_context)

        if file_context.is_active:
            await self._ide_bridge.set_active_file(file_context.uri)

    async def _handle_file_closed(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> None:
        """Handle file closed event."""
        if not self._ide_bridge:
            return

        uri = message.data.get("uri")
        if uri:
            await self._ide_bridge.remove_file(uri)

    async def _handle_file_changed(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> None:
        """Handle file content changed event."""
        if not self._ide_bridge:
            return

        data = message.data
        uri = data.get("uri")

        if uri:
            context = await self._ide_bridge.get_context()
            if uri in context.open_files:
                file_ctx = context.open_files[uri]
                file_ctx.content = data.get("content", file_ctx.content)
                file_ctx.version = data.get("version", file_ctx.version + 1)
                file_ctx.is_dirty = True
                await self._ide_bridge.update_file(file_ctx)

    async def _handle_file_saved(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> None:
        """Handle file saved event."""
        if not self._ide_bridge:
            return

        data = message.data
        uri = data.get("uri")

        if uri:
            context = await self._ide_bridge.get_context()
            if uri in context.open_files:
                file_ctx = context.open_files[uri]
                file_ctx.is_dirty = False
                await self._ide_bridge.update_file(file_ctx)

    async def _handle_cursor_moved(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> None:
        """Handle cursor moved event."""
        if not self._ide_bridge:
            return

        data = message.data
        uri = data.get("uri")
        position = data.get("position")

        if uri and position:
            from .bridge import CursorPosition
            cursor = CursorPosition(
                line=position.get("line", 0),
                character=position.get("character", 0),
            )
            await self._ide_bridge.update_cursor(uri, cursor)
            await self._ide_bridge.set_active_file(uri)

    async def _handle_diagnostics(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> None:
        """Handle diagnostics updated event."""
        if not self._ide_bridge:
            return

        data = message.data
        uri = data.get("uri")
        diagnostics_data = data.get("diagnostics", [])

        if uri:
            from .bridge import Diagnostic, DiagnosticSeverity, TextRange, CursorPosition

            diagnostics = []
            for d in diagnostics_data:
                range_data = d.get("range", {})
                start = range_data.get("start", {})
                end = range_data.get("end", {})

                diagnostics.append(Diagnostic(
                    range=TextRange(
                        start=CursorPosition(
                            line=start.get("line", 0),
                            character=start.get("character", 0),
                        ),
                        end=CursorPosition(
                            line=end.get("line", 0),
                            character=end.get("character", 0),
                        ),
                    ),
                    message=d.get("message", ""),
                    severity=DiagnosticSeverity(d.get("severity", 1)),
                    source=d.get("source", ""),
                    code=d.get("code"),
                ))

            await self._ide_bridge.update_diagnostics(uri, diagnostics)

    async def _handle_suggestion_request(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> None:
        """Handle suggestion request."""
        data = message.data
        uri = data.get("uri")
        line = data.get("line", 0)
        character = data.get("character", 0)

        if not uri:
            return

        try:
            from .suggestions import get_suggestion_engine, TriggerKind

            engine = await get_suggestion_engine()

            # Get file content from context
            context = await self._ide_bridge.get_context() if self._ide_bridge else None
            file_content = ""
            language_id = "python"
            errors = []

            if context and uri in context.open_files:
                fc = context.open_files[uri]
                file_content = fc.content
                language_id = fc.language_id
                errors = [{"line": d.range.start.line, "message": d.message} for d in fc.diagnostics]

            # Get suggestions
            result = await engine.get_suggestions(
                file_path=uri,
                file_content=file_content,
                line=line,
                character=character,
                language_id=language_id,
                errors=errors,
                trigger_kind=TriggerKind.INVOKED if data.get("triggered") else TriggerKind.AUTOMATIC,
            )

            # Send suggestions
            await connection.send(WebSocketMessage(
                type=MessageType.SUGGESTION,
                data={
                    "request_id": message.id,
                    "suggestions": [
                        {
                            "text": s.text,
                            "type": s.type.value,
                            "confidence": s.confidence,
                            "documentation": s.documentation,
                        }
                        for s in result.suggestions
                    ],
                    "latency_ms": result.latency_ms,
                    "cached": result.cached,
                },
            ))

        except Exception as e:
            logger.error(f"[WebSocket] Suggestion error: {e}")
            await connection.send(WebSocketMessage(
                type=MessageType.ERROR,
                data={"error": "suggestion_failed", "message": str(e)},
            ))

    async def _handle_suggestion_feedback(
        self,
        connection: IDEConnection,
        message: WebSocketMessage,
    ) -> None:
        """Handle suggestion feedback (accept/reject)."""
        data = message.data

        try:
            from .suggestions import get_suggestion_engine

            engine = await get_suggestion_engine()
            await engine.record_feedback(
                suggestion_text=data.get("text", ""),
                accepted=data.get("accepted", False),
                file_path=data.get("uri", ""),
                line=data.get("line", 0),
                character=data.get("character", 0),
            )

        except Exception as e:
            logger.error(f"[WebSocket] Feedback error: {e}")

    # -------------------------------------------------------------------------
    # Broadcasting
    # -------------------------------------------------------------------------

    async def broadcast(
        self,
        message: WebSocketMessage,
        filter_sessions: Optional[Set[str]] = None,
    ) -> int:
        """Broadcast message to all (or filtered) connections."""
        sent = 0

        async with self._lock:
            for session_id, conn in self._connections.items():
                if filter_sessions and session_id not in filter_sessions:
                    continue

                if await conn.send(message):
                    sent += 1

        return sent

    async def send_to_session(
        self,
        session_id: str,
        message: WebSocketMessage,
    ) -> bool:
        """Send message to a specific session."""
        async with self._lock:
            if session_id in self._connections:
                return await self._connections[session_id].send(message)
        return False

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# =============================================================================
# Factory
# =============================================================================

_websocket_handler: Optional[IDEWebSocketHandler] = None


async def get_websocket_handler() -> IDEWebSocketHandler:
    """Get or create WebSocket handler instance."""
    global _websocket_handler

    if _websocket_handler is None:
        _websocket_handler = IDEWebSocketHandler()
        await _websocket_handler.initialize()

    return _websocket_handler
