"""
v77.3: Coding Council Startup Integration (Full IDE + Anthropic)
================================================================

Integrates the Unified Coding Council with JARVIS startup sequence,
including the complete IDE bridge and Trinity cross-repo sync.

v77.3 Features:
- Anthropic Claude API integration (primary engine)
- No external tool dependencies required
- Aider-style editing via Claude
- MetaGPT-style multi-agent planning via Claude
- Cross-repo Trinity synchronization
- Real-time IDE integration (LSP + WebSocket)
- Inline suggestions engine
- Distributed suggestion caching

This module provides:
- Single-command startup integration
- run_supervisor.py hook
- Lifespan event handlers
- Health check endpoints
- Anthropic engine auto-initialization
- IDE bridge auto-initialization
- Trinity sync auto-initialization

Usage in run_supervisor.py:
    from backend.core.coding_council.startup import (
        initialize_coding_council_startup,
        shutdown_coding_council_startup,
        get_coding_council_health,
    )

    # During JARVIS startup phase
    await initialize_coding_council_startup()

    # During JARVIS shutdown
    await shutdown_coding_council_startup()

Author: JARVIS v77.3
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import UnifiedCodingCouncil
    from .adapters.anthropic_engine import AnthropicUnifiedEngine

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CODING_COUNCIL_ENABLED = os.getenv("CODING_COUNCIL_ENABLED", "true").lower() == "true"
CODING_COUNCIL_STARTUP_TIMEOUT = float(os.getenv("CODING_COUNCIL_STARTUP_TIMEOUT", "30.0"))
CODING_COUNCIL_VOICE_ANNOUNCE = os.getenv("CODING_COUNCIL_VOICE_ANNOUNCE", "true").lower() == "true"


# =============================================================================
# Global State
# =============================================================================

_council: Optional["UnifiedCodingCouncil"] = None
_anthropic_engine: Optional[Any] = None
_ide_bridge: Optional[Any] = None
_trinity_sync: Optional[Any] = None
_lsp_server: Optional[Any] = None
_startup_time: Optional[float] = None
_initialized = False

# IDE Configuration
IDE_BRIDGE_ENABLED = os.getenv("IDE_BRIDGE_ENABLED", "true").lower() == "true"
LSP_SERVER_PORT = int(os.getenv("LSP_SERVER_PORT", "9257"))
IDE_WEBSOCKET_PORT = int(os.getenv("IDE_WEBSOCKET_PORT", "9258"))


# =============================================================================
# Startup Functions
# =============================================================================

async def initialize_coding_council_startup(
    narrator=None,
    logger_instance=None,
) -> bool:
    """
    Initialize Coding Council during JARVIS startup.

    This should be called from run_supervisor.py during the
    SUPERVISOR_INIT or JARVIS_START phase.

    Args:
        narrator: Optional narrator for voice announcements
        logger_instance: Optional logger instance

    Returns:
        True if initialization succeeded
    """
    global _council, _startup_time, _initialized

    if not CODING_COUNCIL_ENABLED:
        logger.info("[CodingCouncilStartup] Disabled via CODING_COUNCIL_ENABLED=false")
        return False

    if _initialized:
        logger.debug("[CodingCouncilStartup] Already initialized")
        return True

    log = logger_instance or logger
    start_time = time.time()

    try:
        log.info("=" * 60)
        log.info("v77.0 UNIFIED CODING COUNCIL: Initializing")
        log.info("=" * 60)

        # Voice announcement if narrator available
        if narrator and CODING_COUNCIL_VOICE_ANNOUNCE:
            try:
                await narrator.speak(
                    "Initializing Unified Coding Council for self-evolution.",
                    wait=False
                )
            except Exception:
                pass  # Voice is optional

        # Initialize with timeout
        try:
            _council = await asyncio.wait_for(
                _initialize_council_full(),
                timeout=CODING_COUNCIL_STARTUP_TIMEOUT
            )
        except asyncio.TimeoutError:
            log.warning(
                f"[CodingCouncilStartup] Initialization timed out after "
                f"{CODING_COUNCIL_STARTUP_TIMEOUT}s, continuing in background"
            )
            # Start in background instead
            asyncio.create_task(_initialize_council_full())
            return False

        _startup_time = time.time()
        _initialized = True

        # Log success
        duration = time.time() - start_time
        log.info("=" * 60)
        log.info(f"v77.0 UNIFIED CODING COUNCIL: Online ({duration:.2f}s)")
        log.info(f"  Frameworks: Aider, MetaGPT, RepoMaster, OpenHands, Continue")
        log.info(f"  Cross-Repo: {os.getenv('CODING_COUNCIL_CROSS_REPO', 'true')}")
        log.info("=" * 60)

        # Voice success
        if narrator and CODING_COUNCIL_VOICE_ANNOUNCE:
            try:
                await narrator.speak(
                    "Coding Council online. Self-evolution capabilities active.",
                    wait=False
                )
            except Exception:
                pass

        return True

    except Exception as e:
        log.error(f"[CodingCouncilStartup] Initialization failed: {e}")
        return False


async def _initialize_council_full() -> "UnifiedCodingCouncil":
    """Internal: Full council initialization with Trinity, Anthropic engine, and IDE bridge."""
    global _anthropic_engine, _ide_bridge, _trinity_sync, _lsp_server

    from . import initialize_coding_council_full

    # Initialize council
    council = await initialize_coding_council_full()

    # Initialize Anthropic engine (primary engine for Aider/MetaGPT style operations)
    try:
        from .adapters.anthropic_engine import get_anthropic_engine
        _anthropic_engine = await get_anthropic_engine()
        if _anthropic_engine:
            logger.info("[CodingCouncilStartup] Anthropic engine initialized (Claude API)")
    except Exception as e:
        logger.warning(f"[CodingCouncilStartup] Anthropic engine not available: {e}")
        _anthropic_engine = None

    # Initialize IDE Bridge and Trinity Sync
    if IDE_BRIDGE_ENABLED:
        await _initialize_ide_components()

    return council


async def _initialize_ide_components() -> None:
    """Initialize IDE bridge, Trinity sync, and LSP server."""
    global _ide_bridge, _trinity_sync, _lsp_server

    logger.info("[CodingCouncilStartup] Initializing IDE components...")

    # Initialize Trinity cross-repo synchronizer
    try:
        from .ide.trinity_sync import initialize_trinity_sync
        _trinity_sync = await initialize_trinity_sync()
        logger.info("[CodingCouncilStartup] Trinity cross-repo sync initialized")
    except Exception as e:
        logger.warning(f"[CodingCouncilStartup] Trinity sync not available: {e}")
        _trinity_sync = None

    # Initialize IDE Bridge
    try:
        from .ide.bridge import initialize_ide_bridge
        _ide_bridge = await initialize_ide_bridge()
        logger.info("[CodingCouncilStartup] IDE Bridge initialized")
    except Exception as e:
        logger.warning(f"[CodingCouncilStartup] IDE Bridge not available: {e}")
        _ide_bridge = None

    # Initialize LSP Server (runs in background)
    try:
        from .ide.lsp_server import LSPServer
        _lsp_server = LSPServer()
        # Start LSP server in background task
        asyncio.create_task(_start_lsp_server())
        logger.info(f"[CodingCouncilStartup] LSP Server starting on port {LSP_SERVER_PORT}")
    except Exception as e:
        logger.warning(f"[CodingCouncilStartup] LSP Server not available: {e}")
        _lsp_server = None


async def _start_lsp_server() -> None:
    """Start LSP server in background."""
    global _lsp_server

    if _lsp_server is None:
        return

    try:
        await _lsp_server.start_tcp(host="127.0.0.1", port=LSP_SERVER_PORT)
    except Exception as e:
        logger.error(f"[CodingCouncilStartup] LSP Server failed: {e}")


async def shutdown_coding_council_startup() -> None:
    """
    Shutdown Coding Council during JARVIS shutdown.

    This should be called from run_supervisor.py during shutdown.
    """
    global _council, _anthropic_engine, _ide_bridge, _trinity_sync, _lsp_server, _initialized

    if not _initialized:
        return

    logger.info("[CodingCouncilStartup] Shutting down Coding Council...")

    # Shutdown LSP Server
    try:
        if _lsp_server:
            await _lsp_server.shutdown()
            logger.info("[CodingCouncilStartup] LSP Server closed")
    except Exception as e:
        logger.warning(f"[CodingCouncilStartup] LSP Server shutdown warning: {e}")

    # Shutdown IDE Bridge
    try:
        if _ide_bridge:
            await _ide_bridge.shutdown()
            logger.info("[CodingCouncilStartup] IDE Bridge closed")
    except Exception as e:
        logger.warning(f"[CodingCouncilStartup] IDE Bridge shutdown warning: {e}")

    # Shutdown Trinity Sync
    try:
        if _trinity_sync:
            from .ide.trinity_sync import shutdown_trinity_sync
            await shutdown_trinity_sync()
            logger.info("[CodingCouncilStartup] Trinity sync closed")
    except Exception as e:
        logger.warning(f"[CodingCouncilStartup] Trinity sync shutdown warning: {e}")

    # Shutdown Anthropic engine
    try:
        if _anthropic_engine:
            await _anthropic_engine.close()
            logger.info("[CodingCouncilStartup] Anthropic engine closed")
    except Exception as e:
        logger.warning(f"[CodingCouncilStartup] Anthropic engine shutdown warning: {e}")

    # Shutdown council
    try:
        from . import shutdown_coding_council
        await shutdown_coding_council()
    except Exception as e:
        logger.warning(f"[CodingCouncilStartup] Shutdown warning: {e}")

    _council = None
    _anthropic_engine = None
    _ide_bridge = None
    _trinity_sync = None
    _lsp_server = None
    _initialized = False
    logger.info("[CodingCouncilStartup] Shutdown complete")


# =============================================================================
# Health Check
# =============================================================================

async def get_coding_council_health() -> Dict[str, Any]:
    """
    Get Coding Council health status for health endpoints.

    Returns:
        Health status dict
    """
    global _council, _startup_time, _initialized, _ide_bridge, _trinity_sync, _lsp_server

    if not CODING_COUNCIL_ENABLED:
        return {
            "enabled": False,
            "status": "disabled",
        }

    if not _initialized:
        return {
            "enabled": True,
            "status": "not_initialized",
        }

    try:
        status = _council.get_status() if _council else {}

        # Get IDE component statuses
        ide_status = {}
        if IDE_BRIDGE_ENABLED:
            ide_status["ide_bridge"] = {
                "available": _ide_bridge is not None,
                "status": "running" if _ide_bridge else "not_initialized",
            }
            ide_status["trinity_sync"] = {
                "available": _trinity_sync is not None,
                "status": _trinity_sync.get_status() if _trinity_sync else "not_initialized",
            }
            ide_status["lsp_server"] = {
                "available": _lsp_server is not None,
                "port": LSP_SERVER_PORT if _lsp_server else None,
                "status": "running" if _lsp_server else "not_initialized",
            }
            ide_status["websocket_port"] = IDE_WEBSOCKET_PORT

        return {
            "enabled": True,
            "status": "healthy",
            "uptime_seconds": time.time() - _startup_time if _startup_time else 0,
            "active_tasks": status.get("active_tasks", 0),
            "circuit_breakers": status.get("circuit_breakers", {}),
            "frameworks": status.get("frameworks_available", []),
            "ide_integration": ide_status,
        }
    except Exception as e:
        return {
            "enabled": True,
            "status": "error",
            "error": str(e),
        }


# =============================================================================
# API Registration for FastAPI
# =============================================================================

def register_coding_council_routes(app):
    """
    Register Coding Council routes with FastAPI app.

    Usage:
        from backend.core.coding_council.startup import register_coding_council_routes
        register_coding_council_routes(app)
    """
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse

    router = APIRouter(prefix="/coding-council", tags=["Coding Council"])

    @router.get("/health")
    async def health():
        """Get Coding Council health status."""
        return await get_coding_council_health()

    @router.get("/status")
    async def status():
        """Get detailed Coding Council status."""
        if not _initialized or not _council:
            raise HTTPException(status_code=503, detail="Coding Council not initialized")
        return _council.get_status()

    @router.post("/evolve")
    async def evolve(request: dict):
        """
        Trigger code evolution.

        Request body:
        {
            "description": "What to change",
            "target_files": ["file1.py", "file2.py"],  // optional
            "require_approval": true,  // optional, default true
            "require_sandbox": false,  // optional
            "require_planning": false  // optional
        }
        """
        if not _initialized or not _council:
            raise HTTPException(status_code=503, detail="Coding Council not initialized")

        description = request.get("description")
        if not description:
            raise HTTPException(status_code=400, detail="description required")

        result = await _council.evolve(
            description=description,
            target_files=request.get("target_files"),
            require_approval=request.get("require_approval", True),
            require_sandbox=request.get("require_sandbox", False),
            require_planning=request.get("require_planning", False),
        )

        return {
            "success": result.success,
            "task_id": result.task_id,
            "changes_made": result.changes_made,
            "files_modified": result.files_modified,
            "error": result.error,
        }

    @router.get("/frameworks")
    async def frameworks():
        """Get available framework status."""
        if not _initialized or not _council:
            raise HTTPException(status_code=503, detail="Coding Council not initialized")

        framework_status = {}

        # Check Anthropic engine first (primary engine)
        if _anthropic_engine:
            try:
                available = await _anthropic_engine.is_available()
                framework_status["anthropic_engine"] = {
                    "available": available,
                    "type": "primary",
                    "features": ["aider_style", "metagpt_style", "git_aware", "no_external_deps"],
                    "tokens_used": _anthropic_engine.tokens_used,
                }
            except Exception as e:
                framework_status["anthropic_engine"] = {"available": False, "error": str(e)}
        else:
            framework_status["anthropic_engine"] = {"available": False, "error": "Not initialized"}

        # Check traditional adapters (fallback)
        for name in ["aider", "repomaster", "metagpt", "openhands", "continue"]:
            adapter = getattr(_council, f"_{name}", None)
            if adapter:
                try:
                    available = await adapter.is_available()
                    framework_status[name] = {"available": available, "type": "adapter"}
                except Exception as e:
                    framework_status[name] = {"available": False, "error": str(e)}
            else:
                framework_status[name] = {"available": False, "error": "Not loaded"}

        return {"frameworks": framework_status}

    # IDE Integration Routes
    @router.get("/ide/status")
    async def ide_status():
        """Get IDE integration status."""
        if not IDE_BRIDGE_ENABLED:
            return {"enabled": False}

        return {
            "enabled": True,
            "ide_bridge": {
                "available": _ide_bridge is not None,
            },
            "trinity_sync": {
                "available": _trinity_sync is not None,
                "status": _trinity_sync.get_status() if _trinity_sync else None,
            },
            "lsp_server": {
                "available": _lsp_server is not None,
                "port": LSP_SERVER_PORT,
            },
            "websocket_port": IDE_WEBSOCKET_PORT,
        }

    @router.get("/ide/context")
    async def ide_context():
        """Get current IDE context."""
        if not _ide_bridge:
            raise HTTPException(status_code=503, detail="IDE Bridge not initialized")

        context = await _ide_bridge.get_context()
        return {
            "active_files": len(context.active_files) if context else 0,
            "recent_files": list(context.recent_files)[:10] if context else [],
            "diagnostics_count": len(context.diagnostics) if context else 0,
        }

    @router.post("/ide/suggest")
    async def ide_suggest(request: dict):
        """
        Get inline suggestions.

        Request body:
        {
            "file_path": "/path/to/file.py",
            "content": "file content",
            "line": 10,
            "character": 5,
            "language_id": "python"
        }
        """
        if not _ide_bridge:
            raise HTTPException(status_code=503, detail="IDE Bridge not initialized")

        file_path = request.get("file_path")
        content = request.get("content")
        line = request.get("line", 0)
        character = request.get("character", 0)

        if not file_path or content is None:
            raise HTTPException(status_code=400, detail="file_path and content required")

        suggestion = await _ide_bridge.get_inline_suggestion(
            uri=f"file://{file_path}",
            line=line,
            character=character,
            trigger_kind="invoked",
        )

        return {"suggestion": suggestion}

    @router.get("/ide/trinity/repos")
    async def trinity_repos():
        """Get Trinity repository status."""
        if not _trinity_sync:
            raise HTTPException(status_code=503, detail="Trinity Sync not initialized")

        contexts = await _trinity_sync.get_all_contexts()
        return {
            repo.value: ctx.to_dict()
            for repo, ctx in contexts.items()
        }

    @router.post("/ide/trinity/publish")
    async def trinity_publish(request: dict):
        """
        Publish a file change event to Trinity.

        Request body:
        {
            "file_path": "/path/to/file.py",
            "change_type": "modified",  // created, modified, deleted
            "content": "optional file content"
        }
        """
        if not _trinity_sync:
            raise HTTPException(status_code=503, detail="Trinity Sync not initialized")

        file_path = request.get("file_path")
        change_type = request.get("change_type", "modified")
        content = request.get("content")

        if not file_path:
            raise HTTPException(status_code=400, detail="file_path required")

        # Detect repo type
        from .ide.trinity_sync import detect_repo_type
        repo = detect_repo_type(file_path)

        if not repo:
            raise HTTPException(status_code=400, detail="File not in any Trinity repo")

        success = await _trinity_sync.publish_file_change(
            repo=repo,
            file_path=file_path,
            change_type=change_type,
            content=content,
        )

        return {"success": success, "repo": repo.value}

    # Register routes
    app.include_router(router)
    logger.info("[CodingCouncilStartup] API routes registered at /coding-council")


# =============================================================================
# Hook for run_supervisor.py Integration
# =============================================================================

async def coding_council_startup_hook(
    bootstrapper=None,
    phase: str = "supervisor_init",
) -> bool:
    """
    Hook for run_supervisor.py to call during startup.

    Args:
        bootstrapper: SupervisorBootstrapper instance
        phase: Current startup phase

    Returns:
        True if hook completed successfully
    """
    if not CODING_COUNCIL_ENABLED:
        return True

    # Only initialize during supervisor_init phase
    if phase != "supervisor_init":
        return True

    narrator = None
    logger_instance = None

    if bootstrapper:
        narrator = getattr(bootstrapper, 'narrator', None)
        logger_instance = getattr(bootstrapper, 'logger', None)

    return await initialize_coding_council_startup(
        narrator=narrator,
        logger_instance=logger_instance,
    )


async def coding_council_shutdown_hook(
    bootstrapper=None,
) -> None:
    """
    Hook for run_supervisor.py to call during shutdown.

    Args:
        bootstrapper: SupervisorBootstrapper instance
    """
    await shutdown_coding_council_startup()


# =============================================================================
# Accessor Functions
# =============================================================================

def get_council() -> Optional["UnifiedCodingCouncil"]:
    """Get the global Coding Council instance."""
    return _council


def is_initialized() -> bool:
    """Check if Coding Council is initialized."""
    return _initialized


def get_ide_bridge() -> Optional[Any]:
    """Get the global IDE Bridge instance."""
    return _ide_bridge


def get_trinity_sync() -> Optional[Any]:
    """Get the global Trinity Sync instance."""
    return _trinity_sync


def get_lsp_server() -> Optional[Any]:
    """Get the global LSP Server instance."""
    return _lsp_server


def get_anthropic_engine() -> Optional[Any]:
    """Get the global Anthropic Engine instance."""
    return _anthropic_engine
