"""
v77.3: LSP-Compatible Server
============================

Language Server Protocol compatible server for IDE integration.

This provides a standard LSP interface that any LSP-compatible IDE
can connect to, enabling:
- Inline completions
- Code actions (quick fixes)
- Hover information
- Diagnostics
- Code lens

LSP Spec: https://microsoft.github.io/language-server-protocol/

Author: JARVIS v77.3
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class LSPConfig:
    """LSP server configuration."""

    PORT: int = int(os.getenv("LSP_PORT", "8017"))
    HOST: str = os.getenv("LSP_HOST", "127.0.0.1")
    MODE: str = os.getenv("LSP_MODE", "tcp")  # tcp, stdio
    LOG_LEVEL: str = os.getenv("LSP_LOG_LEVEL", "INFO")


# =============================================================================
# LSP Constants
# =============================================================================

class ErrorCode(IntEnum):
    """LSP error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_NOT_INITIALIZED = -32002
    REQUEST_CANCELLED = -32800


class CompletionTriggerKind(IntEnum):
    """How completion was triggered."""
    INVOKED = 1
    TRIGGER_CHARACTER = 2
    TRIGGER_FOR_INCOMPLETE = 3


class CompletionItemKind(IntEnum):
    """Kind of completion item."""
    TEXT = 1
    METHOD = 2
    FUNCTION = 3
    CONSTRUCTOR = 4
    FIELD = 5
    VARIABLE = 6
    CLASS = 7
    INTERFACE = 8
    MODULE = 9
    PROPERTY = 10
    UNIT = 11
    VALUE = 12
    ENUM = 13
    KEYWORD = 14
    SNIPPET = 15
    COLOR = 16
    FILE = 17
    REFERENCE = 18
    FOLDER = 19
    ENUM_MEMBER = 20
    CONSTANT = 21
    STRUCT = 22
    EVENT = 23
    OPERATOR = 24
    TYPE_PARAMETER = 25


class DiagnosticSeverity(IntEnum):
    """Diagnostic severity."""
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


class TextDocumentSyncKind(IntEnum):
    """Text document sync kind."""
    NONE = 0
    FULL = 1
    INCREMENTAL = 2


# =============================================================================
# LSP Data Structures
# =============================================================================

@dataclass
class Position:
    """Position in a document."""
    line: int
    character: int

    def to_dict(self) -> Dict[str, int]:
        return {"line": self.line, "character": self.character}

    @classmethod
    def from_dict(cls, data: Dict) -> "Position":
        return cls(line=data["line"], character=data["character"])


@dataclass
class Range:
    """Range in a document."""
    start: Position
    end: Position

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict) -> "Range":
        return cls(
            start=Position.from_dict(data["start"]),
            end=Position.from_dict(data["end"]),
        )


@dataclass
class TextDocumentIdentifier:
    """Text document identifier."""
    uri: str

    @classmethod
    def from_dict(cls, data: Dict) -> "TextDocumentIdentifier":
        return cls(uri=data["uri"])


@dataclass
class TextDocumentItem:
    """Text document item."""
    uri: str
    language_id: str
    version: int
    text: str

    @classmethod
    def from_dict(cls, data: Dict) -> "TextDocumentItem":
        return cls(
            uri=data["uri"],
            language_id=data.get("languageId", ""),
            version=data.get("version", 1),
            text=data.get("text", ""),
        )


@dataclass
class CompletionItem:
    """A completion item."""
    label: str
    kind: CompletionItemKind = CompletionItemKind.TEXT
    detail: Optional[str] = None
    documentation: Optional[str] = None
    insert_text: Optional[str] = None
    insert_text_format: int = 1  # 1 = PlainText, 2 = Snippet
    text_edit: Optional[Dict] = None
    additional_text_edits: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "label": self.label,
            "kind": self.kind.value,
        }
        if self.detail:
            result["detail"] = self.detail
        if self.documentation:
            result["documentation"] = self.documentation
        if self.insert_text:
            result["insertText"] = self.insert_text
        if self.insert_text_format != 1:
            result["insertTextFormat"] = self.insert_text_format
        if self.text_edit:
            result["textEdit"] = self.text_edit
        if self.additional_text_edits:
            result["additionalTextEdits"] = self.additional_text_edits
        return result


@dataclass
class Diagnostic:
    """A diagnostic."""
    range: Range
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    source: str = "jarvis"
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "range": self.range.to_dict(),
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
        }
        if self.code:
            result["code"] = self.code
        return result


# =============================================================================
# JSON-RPC Handler
# =============================================================================

class JSONRPCHandler:
    """Handles JSON-RPC message parsing and formatting."""

    @staticmethod
    def parse_message(data: bytes) -> Optional[Dict]:
        """Parse a JSON-RPC message from LSP format."""
        try:
            # Find content-length header
            text = data.decode("utf-8")
            header_end = text.find("\r\n\r\n")

            if header_end == -1:
                return None

            header = text[:header_end]
            content = text[header_end + 4:]

            # Parse content-length
            for line in header.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    content_length = int(line.split(":")[1].strip())
                    break
            else:
                return None

            return json.loads(content[:content_length])

        except Exception as e:
            logger.error(f"[LSP] Parse error: {e}")
            return None

    @staticmethod
    def format_message(data: Dict) -> bytes:
        """Format a JSON-RPC message for LSP."""
        content = json.dumps(data)
        content_bytes = content.encode("utf-8")
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n"
        return header.encode("utf-8") + content_bytes

    @staticmethod
    def create_response(id: Union[int, str], result: Any) -> Dict:
        """Create a JSON-RPC response."""
        return {
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }

    @staticmethod
    def create_error(id: Union[int, str, None], code: int, message: str) -> Dict:
        """Create a JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    @staticmethod
    def create_notification(method: str, params: Any) -> Dict:
        """Create a JSON-RPC notification."""
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }


# =============================================================================
# LSP Server
# =============================================================================

class LSPServer:
    """
    Language Server Protocol server.

    Provides LSP interface for any compatible IDE.
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._initialized = False
        self._running = False
        self._documents: Dict[str, TextDocumentItem] = {}
        self._client_capabilities: Dict = {}

        # IDE bridge reference
        self._ide_bridge: Optional[Any] = None

        # Setup handlers
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup LSP method handlers."""
        self._handlers = {
            "initialize": self._handle_initialize,
            "initialized": self._handle_initialized,
            "shutdown": self._handle_shutdown,
            "exit": self._handle_exit,
            "textDocument/didOpen": self._handle_did_open,
            "textDocument/didClose": self._handle_did_close,
            "textDocument/didChange": self._handle_did_change,
            "textDocument/didSave": self._handle_did_save,
            "textDocument/completion": self._handle_completion,
            "textDocument/hover": self._handle_hover,
            "textDocument/codeAction": self._handle_code_action,
        }

    async def initialize(self) -> bool:
        """Initialize the LSP server."""
        try:
            from .bridge import get_ide_bridge
            self._ide_bridge = await get_ide_bridge()

            self._running = True
            logger.info("[LSPServer] Initialized")
            return True

        except Exception as e:
            logger.error(f"[LSPServer] Init failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the server."""
        self._running = False
        self._initialized = False
        logger.info("[LSPServer] Shutdown")

    async def start_tcp(self, host: str = None, port: int = None) -> None:
        """Start TCP server."""
        host = host or LSPConfig.HOST
        port = port or LSPConfig.PORT

        server = await asyncio.start_server(
            self._handle_connection,
            host=host,
            port=port,
        )

        logger.info(f"[LSPServer] Listening on {host}:{port}")

        async with server:
            await server.serve_forever()

    async def start_stdio(self) -> None:
        """Start stdio mode for editor integration."""
        logger.info("[LSPServer] Starting stdio mode")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)

        loop = asyncio.get_running_loop()
        await loop.connect_read_pipe(
            lambda: protocol, sys.stdin
        )

        writer_transport, writer_protocol = await loop.connect_write_pipe(
            lambda: asyncio.streams.FlowControlMixin(), sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, loop)

        await self._handle_stream(reader, writer)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a TCP connection."""
        logger.info("[LSPServer] New connection")
        await self._handle_stream(reader, writer)
        logger.info("[LSPServer] Connection closed")

    async def _handle_stream(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle message stream."""
        buffer = b""

        while self._running:
            try:
                # Read data
                data = await asyncio.wait_for(reader.read(4096), timeout=60.0)
                if not data:
                    break

                buffer += data

                # Try to parse messages
                while buffer:
                    # Find content-length
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end == -1:
                        break

                    header = buffer[:header_end].decode("utf-8")
                    content_length = None

                    for line in header.split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            content_length = int(line.split(":")[1].strip())
                            break

                    if content_length is None:
                        break

                    # Check if we have full message
                    message_end = header_end + 4 + content_length
                    if len(buffer) < message_end:
                        break

                    # Extract message
                    content = buffer[header_end + 4:message_end]
                    buffer = buffer[message_end:]

                    # Parse and handle
                    try:
                        message = json.loads(content)
                        response = await self._handle_message(message)

                        if response:
                            writer.write(JSONRPCHandler.format_message(response))
                            await writer.drain()

                    except json.JSONDecodeError as e:
                        logger.error(f"[LSPServer] JSON error: {e}")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[LSPServer] Stream error: {e}")
                break

    async def _handle_message(self, message: Dict) -> Optional[Dict]:
        """Handle a JSON-RPC message."""
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")

        # Check if notification (no id)
        is_notification = msg_id is None

        # Find handler
        handler = self._handlers.get(method)

        if handler is None:
            if not is_notification:
                return JSONRPCHandler.create_error(
                    msg_id,
                    ErrorCode.METHOD_NOT_FOUND,
                    f"Method not found: {method}",
                )
            return None

        try:
            result = await handler(params)

            if is_notification:
                return None

            return JSONRPCHandler.create_response(msg_id, result)

        except Exception as e:
            logger.error(f"[LSPServer] Handler error for {method}: {e}")
            if not is_notification:
                return JSONRPCHandler.create_error(
                    msg_id,
                    ErrorCode.INTERNAL_ERROR,
                    str(e),
                )
            return None

    # -------------------------------------------------------------------------
    # LSP Handlers
    # -------------------------------------------------------------------------

    async def _handle_initialize(self, params: Dict) -> Dict:
        """Handle initialize request."""
        self._client_capabilities = params.get("capabilities", {})

        self._initialized = True

        return {
            "capabilities": {
                "textDocumentSync": {
                    "openClose": True,
                    "change": TextDocumentSyncKind.FULL,
                    "save": {"includeText": True},
                },
                "completionProvider": {
                    "triggerCharacters": [".", "(", "[", "{", " "],
                    "resolveProvider": False,
                },
                "hoverProvider": True,
                "codeActionProvider": True,
            },
            "serverInfo": {
                "name": "JARVIS Coding Council",
                "version": "77.3",
            },
        }

    async def _handle_initialized(self, params: Dict) -> None:
        """Handle initialized notification."""
        logger.info("[LSPServer] Client initialized")

    async def _handle_shutdown(self, params: Dict) -> None:
        """Handle shutdown request."""
        self._initialized = False
        return None

    async def _handle_exit(self, params: Dict) -> None:
        """Handle exit notification."""
        self._running = False

    async def _handle_did_open(self, params: Dict) -> None:
        """Handle textDocument/didOpen."""
        doc = params.get("textDocument", {})
        item = TextDocumentItem.from_dict(doc)
        self._documents[item.uri] = item

        # Notify IDE bridge
        if self._ide_bridge:
            from .bridge import FileContext
            file_ctx = FileContext(
                uri=item.uri,
                path=item.uri.replace("file://", ""),
                content=item.text,
                language_id=item.language_id,
                version=item.version,
                is_active=True,
            )
            await self._ide_bridge.update_file(file_ctx)
            await self._ide_bridge.set_active_file(item.uri)

    async def _handle_did_close(self, params: Dict) -> None:
        """Handle textDocument/didClose."""
        doc = params.get("textDocument", {})
        uri = doc.get("uri")

        if uri in self._documents:
            del self._documents[uri]

        if self._ide_bridge:
            await self._ide_bridge.remove_file(uri)

    async def _handle_did_change(self, params: Dict) -> None:
        """Handle textDocument/didChange."""
        doc = params.get("textDocument", {})
        uri = doc.get("uri")
        changes = params.get("contentChanges", [])

        if uri in self._documents and changes:
            # For full sync, just use first change
            self._documents[uri].text = changes[0].get("text", "")
            self._documents[uri].version = doc.get("version", self._documents[uri].version + 1)

            if self._ide_bridge:
                from .bridge import FileContext
                item = self._documents[uri]
                file_ctx = FileContext(
                    uri=item.uri,
                    path=item.uri.replace("file://", ""),
                    content=item.text,
                    language_id=item.language_id,
                    version=item.version,
                    is_dirty=True,
                )
                await self._ide_bridge.update_file(file_ctx)

    async def _handle_did_save(self, params: Dict) -> None:
        """Handle textDocument/didSave."""
        doc = params.get("textDocument", {})
        uri = doc.get("uri")
        text = params.get("text")

        if uri in self._documents and text:
            self._documents[uri].text = text

        if self._ide_bridge:
            from .bridge import FileContext
            if uri in self._documents:
                item = self._documents[uri]
                file_ctx = FileContext(
                    uri=item.uri,
                    path=item.uri.replace("file://", ""),
                    content=item.text,
                    language_id=item.language_id,
                    version=item.version,
                    is_dirty=False,
                )
                await self._ide_bridge.update_file(file_ctx)

    async def _handle_completion(self, params: Dict) -> Dict:
        """Handle textDocument/completion."""
        doc = params.get("textDocument", {})
        uri = doc.get("uri")
        position = params.get("position", {})

        if uri not in self._documents:
            return {"isIncomplete": False, "items": []}

        try:
            from .suggestions import get_suggestion_engine, TriggerKind

            engine = await get_suggestion_engine()
            item = self._documents[uri]

            result = await engine.get_suggestions(
                file_path=uri,
                file_content=item.text,
                line=position.get("line", 0),
                character=position.get("character", 0),
                language_id=item.language_id,
                trigger_kind=TriggerKind.INVOKED,
            )

            # Convert to LSP completion items
            items = []
            for s in result.suggestions:
                items.append(CompletionItem(
                    label=s.text[:50] + ("..." if len(s.text) > 50 else ""),
                    kind=CompletionItemKind.SNIPPET,
                    detail=f"JARVIS ({s.confidence:.0%})",
                    insert_text=s.text,
                    documentation=s.documentation,
                ).to_dict())

            return {"isIncomplete": False, "items": items}

        except Exception as e:
            logger.error(f"[LSPServer] Completion error: {e}")
            return {"isIncomplete": False, "items": []}

    async def _handle_hover(self, params: Dict) -> Optional[Dict]:
        """Handle textDocument/hover."""
        # Basic hover - could be extended with type info
        return None

    async def _handle_code_action(self, params: Dict) -> List[Dict]:
        """Handle textDocument/codeAction."""
        doc = params.get("textDocument", {})
        uri = doc.get("uri")
        range_data = params.get("range", {})
        context = params.get("context", {})

        diagnostics = context.get("diagnostics", [])

        if not diagnostics:
            return []

        actions = []

        # For each diagnostic, offer a "Fix with JARVIS" action
        for diag in diagnostics:
            actions.append({
                "title": f"Fix with JARVIS: {diag.get('message', 'Unknown')[:50]}",
                "kind": "quickfix",
                "diagnostics": [diag],
                "command": {
                    "title": "Fix with JARVIS",
                    "command": "jarvis.fix",
                    "arguments": [uri, diag],
                },
            })

        return actions


# =============================================================================
# Factory
# =============================================================================

_lsp_server: Optional[LSPServer] = None


async def get_lsp_server() -> LSPServer:
    """Get or create LSP server instance."""
    global _lsp_server

    if _lsp_server is None:
        _lsp_server = LSPServer()
        await _lsp_server.initialize()

    return _lsp_server
