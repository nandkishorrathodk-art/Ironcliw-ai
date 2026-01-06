"""
v77.3: IDE Integration Module for Coding Council
=================================================

Real-time IDE integration providing Claude Code-like capabilities:
- Inline suggestions
- Real-time context awareness
- LSP-compatible protocol
- WebSocket streaming
- Error-aware editing
- Trinity cross-repo sync

This module bridges VS Code/Cursor to the JARVIS Coding Council,
enabling real-time code assistance while maintaining autonomous
self-evolution capabilities.

Author: JARVIS v77.3
Version: 1.0.0
"""

from __future__ import annotations

__all__ = [
    # Bridge components
    "IDEBridge",
    "IDEContext",
    "get_ide_bridge",
    "initialize_ide_bridge",
    # Suggestion engine
    "InlineSuggestionEngine",
    # Server components
    "LSPServer",
    "IDEWebSocketHandler",
    # Trinity sync
    "TrinityIDESynchronizer",
    "get_trinity_synchronizer",
    "initialize_trinity_sync",
    "shutdown_trinity_sync",
    "SyncEvent",
    "SyncEventType",
    "RepoType",
]


def __getattr__(name: str):
    """Lazy import IDE components."""
    # Bridge
    if name == "IDEBridge":
        from .bridge import IDEBridge
        return IDEBridge
    elif name == "IDEContext":
        from .bridge import IDEContext
        return IDEContext
    elif name == "get_ide_bridge":
        from .bridge import get_ide_bridge
        return get_ide_bridge
    elif name == "initialize_ide_bridge":
        from .bridge import initialize_ide_bridge
        return initialize_ide_bridge

    # Suggestions
    elif name == "InlineSuggestionEngine":
        from .suggestions import InlineSuggestionEngine
        return InlineSuggestionEngine

    # Servers
    elif name == "LSPServer":
        from .lsp_server import LSPServer
        return LSPServer
    elif name == "IDEWebSocketHandler":
        from .websocket_handler import IDEWebSocketHandler
        return IDEWebSocketHandler

    # Trinity sync
    elif name == "TrinityIDESynchronizer":
        from .trinity_sync import TrinityIDESynchronizer
        return TrinityIDESynchronizer
    elif name == "get_trinity_synchronizer":
        from .trinity_sync import get_trinity_synchronizer
        return get_trinity_synchronizer
    elif name == "initialize_trinity_sync":
        from .trinity_sync import initialize_trinity_sync
        return initialize_trinity_sync
    elif name == "shutdown_trinity_sync":
        from .trinity_sync import shutdown_trinity_sync
        return shutdown_trinity_sync
    elif name == "SyncEvent":
        from .trinity_sync import SyncEvent
        return SyncEvent
    elif name == "SyncEventType":
        from .trinity_sync import SyncEventType
        return SyncEventType
    elif name == "RepoType":
        from .trinity_sync import RepoType
        return RepoType

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
