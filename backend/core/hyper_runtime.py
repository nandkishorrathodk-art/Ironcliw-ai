"""
JARVIS Hyper-Runtime v9.0 - Rust-First Async Engine
=====================================================

Intelligent runtime selection that maximizes async performance by choosing
the fastest available engine in order of preference:

    Level 3 (Fastest):  Granian (Rust/Tokio)     - 3-5x faster than uvicorn
    Level 2 (Fast):     uvloop (C/libuv)         - 2-4x faster than asyncio
    Level 1 (Standard): asyncio                  - Python standard library

The Rust-based Granian server replaces the entire Python async stack with:
- Tokio: Rust's premier async runtime (faster than libuv)
- Hyper: Rust's HTTP library (faster than Python HTTP parsing)
- Native threading: Rust threads are lighter than Python threads

This is the absolute performance ceiling before rewriting logic in Rust/Mojo.

Usage:
    from backend.core.hyper_runtime import (
        get_runtime_engine,
        start_hyper_server,
        get_runtime_stats,
        RuntimeLevel,
    )

    # Check what's available
    engine = get_runtime_engine()
    print(f"Using: {engine.level.name} - {engine.name}")

    # Start the hyper server (auto-selects best runtime)
    start_hyper_server("backend.main:app", host="0.0.0.0", port=8010)

Version: 9.0.0 - Rust-First Edition
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("jarvis.hyper_runtime")


# =============================================================================
# Runtime Level Hierarchy
# =============================================================================

class RuntimeLevel(IntEnum):
    """
    Performance levels for async runtime engines.

    Higher level = faster performance:
    - STANDARD (1): Python asyncio - baseline
    - FAST (2): uvloop (C/libuv) - 2-4x faster
    - HYPER (3): Granian (Rust/Tokio) - 3-5x faster
    """
    STANDARD = 1  # asyncio (Python)
    FAST = 2      # uvloop (C/libuv)
    HYPER = 3     # Granian (Rust/Tokio)


@dataclass
class RuntimeEngine:
    """Information about an async runtime engine."""
    name: str
    level: RuntimeLevel
    available: bool
    version: Optional[str] = None
    backend: Optional[str] = None
    description: str = ""
    activated: bool = False


def _get_intelligent_worker_count() -> int:
    """
    Dynamically determine optimal worker count based on context.

    Strategy:
    - JARVIS_SERVER_WORKERS env var: Explicit override (highest priority)
    - JARVIS_FAST_STARTUP=true: 1 worker (fastest boot, single-user mode)
    - JARVIS_SINGLE_USER=true: 1-2 workers (personal assistant mode)
    - Development mode (JARVIS_DEV_MODE): 1 worker (fast iteration)
    - Production mode: CPU cores (max throughput)

    Returns:
        Optimal worker count as integer
    """
    # Check for explicit override first
    explicit_workers = os.getenv("JARVIS_SERVER_WORKERS")
    if explicit_workers:
        return int(explicit_workers)

    # Fast startup mode: Single worker for instant boot
    fast_startup = os.getenv("JARVIS_FAST_STARTUP", "").lower() in ("1", "true", "yes")
    if fast_startup:
        return 1

    # Single user mode: Personal assistant doesn't need multiple workers
    single_user = os.getenv("JARVIS_SINGLE_USER", "true").lower() in ("1", "true", "yes")
    if single_user:
        return 1

    # Development mode: Single worker for fast code iteration
    dev_mode = os.getenv("JARVIS_DEV_MODE", "").lower() in ("1", "true", "yes")
    if dev_mode:
        return 1

    # Supervised mode (run_supervisor.py): Single worker for controlled startup
    supervised = os.getenv("JARVIS_SUPERVISED", "").lower() in ("1", "true", "yes")
    if supervised:
        return 1

    # Production/server mode: Use CPU cores (capped at 4 for personal systems)
    cpu_count = multiprocessing.cpu_count()
    # Cap at 4 workers to avoid resource contention on personal machines
    return min(4, max(1, cpu_count))


@dataclass
class ServerConfig:
    """
    Dynamic server configuration - no hardcoding.

    All values can be overridden via environment variables:
        JARVIS_SERVER_HOST: Bind address (default: 0.0.0.0)
        JARVIS_SERVER_PORT: Port number (default: 8010)
        JARVIS_SERVER_WORKERS: Worker count (default: intelligent detection)
        JARVIS_SERVER_THREADS: Threads per worker (default: 2)
        JARVIS_SERVER_INTERFACE: ASGI/WSGI (default: asgi)
        JARVIS_SERVER_HTTP_VERSION: HTTP/1 or HTTP/2 (default: auto)
        JARVIS_SERVER_WEBSOCKETS: Enable WebSockets (default: true)
        JARVIS_SERVER_BACKLOG: Connection backlog (default: 1024)
        JARVIS_SERVER_TIMEOUT_KEEP_ALIVE: Keep-alive timeout (default: 5)
        JARVIS_RUNTIME_LEVEL: Force runtime level (1=asyncio, 2=uvloop, 3=granian)

    Worker count is intelligently determined:
        - JARVIS_FAST_STARTUP=true: 1 worker (instant boot)
        - JARVIS_SINGLE_USER=true: 1 worker (default for personal use)
        - Development mode: 1 worker (fast iteration)
        - Production: min(4, CPU cores)
    """
    host: str = field(default_factory=lambda: os.getenv("JARVIS_SERVER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("JARVIS_SERVER_PORT", "8010")))
    workers: int = field(default_factory=_get_intelligent_worker_count)
    threads: int = field(default_factory=lambda: int(os.getenv("JARVIS_SERVER_THREADS", "2")))
    interface: str = field(default_factory=lambda: os.getenv("JARVIS_SERVER_INTERFACE", "asgi"))
    http_version: str = field(default_factory=lambda: os.getenv("JARVIS_SERVER_HTTP_VERSION", "auto"))
    websockets: bool = field(default_factory=lambda: os.getenv("JARVIS_SERVER_WEBSOCKETS", "true").lower() == "true")
    backlog: int = field(default_factory=lambda: int(os.getenv("JARVIS_SERVER_BACKLOG", "1024")))
    timeout_keep_alive: int = field(default_factory=lambda: int(os.getenv("JARVIS_SERVER_TIMEOUT_KEEP_ALIVE", "5")))
    log_level: str = field(default_factory=lambda: os.getenv("JARVIS_SERVER_LOG_LEVEL", "warning"))
    reload: bool = field(default_factory=lambda: os.getenv("JARVIS_DEV_MODE", "false").lower() == "true")


# =============================================================================
# Runtime Detection & Selection
# =============================================================================

_runtime_cache: Optional[RuntimeEngine] = None
_activation_attempted: bool = False


def _detect_granian() -> RuntimeEngine:
    """Detect if Granian (Rust/Tokio) is available with all required features."""
    try:
        import granian
        # Also verify the constants we need are available (ThreadModes was added later)
        from granian.constants import Interfaces, Loops, HTTPModes, ThreadModes  # noqa: F401
        version = getattr(granian, "__version__", "unknown")
        return RuntimeEngine(
            name="Granian",
            level=RuntimeLevel.HYPER,
            available=True,
            version=version,
            backend="Rust/Tokio",
            description="Rust-based ASGI server with Tokio async runtime",
        )
    except ImportError as e:
        # Distinguish between granian not installed vs missing features
        if "granian" in str(e).lower() and "constants" not in str(e).lower():
            desc = "Not installed - run: pip install granian"
        else:
            desc = f"Incompatible version (missing required features): {e}"
        return RuntimeEngine(
            name="Granian",
            level=RuntimeLevel.HYPER,
            available=False,
            description=desc,
        )


def _detect_uvloop() -> RuntimeEngine:
    """Detect if uvloop (C/libuv) is available."""
    if sys.platform == "win32":
        return RuntimeEngine(
            name="uvloop",
            level=RuntimeLevel.FAST,
            available=False,
            description="Not available on Windows",
        )

    try:
        import uvloop
        version = getattr(uvloop, "__version__", "unknown")
        return RuntimeEngine(
            name="uvloop",
            level=RuntimeLevel.FAST,
            available=True,
            version=version,
            backend="C/libuv",
            description="C-based event loop using libuv",
        )
    except ImportError:
        return RuntimeEngine(
            name="uvloop",
            level=RuntimeLevel.FAST,
            available=False,
            description="Not installed - run: pip install uvloop",
        )


def _detect_asyncio() -> RuntimeEngine:
    """Asyncio is always available (Python standard library)."""
    return RuntimeEngine(
        name="asyncio",
        level=RuntimeLevel.STANDARD,
        available=True,
        version=f"Python {sys.version_info.major}.{sys.version_info.minor}",
        backend="Python",
        description="Python standard library async runtime",
        activated=True,  # Always the fallback
    )


def detect_all_runtimes() -> Dict[RuntimeLevel, RuntimeEngine]:
    """Detect all available runtime engines."""
    return {
        RuntimeLevel.HYPER: _detect_granian(),
        RuntimeLevel.FAST: _detect_uvloop(),
        RuntimeLevel.STANDARD: _detect_asyncio(),
    }


def get_runtime_engine(force_level: Optional[int] = None) -> RuntimeEngine:
    """
    Get the best available runtime engine.

    Selection priority (highest first):
    1. Granian (Rust/Tokio) - Level 3
    2. uvloop (C/libuv) - Level 2
    3. asyncio (Python) - Level 1

    Args:
        force_level: Force a specific runtime level (1, 2, or 3)

    Returns:
        RuntimeEngine with the best available engine
    """
    global _runtime_cache

    # Check environment variable for forced level
    env_level = os.getenv("JARVIS_RUNTIME_LEVEL")
    if env_level and force_level is None:
        try:
            force_level = int(env_level)
        except ValueError:
            pass

    # Return cached if available and not forcing
    if _runtime_cache is not None and force_level is None:
        return _runtime_cache

    runtimes = detect_all_runtimes()

    # If forcing a level, try that first
    if force_level is not None:
        try:
            level = RuntimeLevel(force_level)
            if runtimes[level].available:
                _runtime_cache = runtimes[level]
                return _runtime_cache
            else:
                logger.warning(
                    f"Forced runtime level {level.name} not available, "
                    f"falling back to auto-selection"
                )
        except ValueError:
            logger.warning(f"Invalid runtime level {force_level}, using auto-selection")

    # Auto-select best available (highest level first)
    for level in sorted(RuntimeLevel, reverse=True):
        if runtimes[level].available:
            _runtime_cache = runtimes[level]
            return _runtime_cache

    # Fallback to asyncio (should always be available)
    _runtime_cache = runtimes[RuntimeLevel.STANDARD]
    return _runtime_cache


def activate_runtime() -> RuntimeEngine:
    """
    Activate the best available runtime engine.

    For uvloop: Sets the event loop policy
    For Granian: No activation needed (handles its own loop)
    For asyncio: No activation needed (default)

    Returns:
        The activated RuntimeEngine
    """
    global _activation_attempted

    if _activation_attempted:
        return get_runtime_engine()

    _activation_attempted = True
    engine = get_runtime_engine()

    if engine.level == RuntimeLevel.FAST and engine.available:
        # Activate uvloop as the event loop policy
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            engine.activated = True
            logger.info(f"🚀 [HYPER-RUNTIME] uvloop activated (C/libuv engine)")
        except Exception as e:
            logger.warning(f"Failed to activate uvloop: {e}")

    elif engine.level == RuntimeLevel.HYPER and engine.available:
        # Granian handles its own loop, just mark as ready
        engine.activated = True
        logger.info(f"⚡ [HYPER-RUNTIME] Granian ready (Rust/Tokio engine)")

    else:
        # Standard asyncio
        engine.activated = True
        logger.debug("[HYPER-RUNTIME] Using standard asyncio")

    return engine


# =============================================================================
# Hyper Server Launcher
# =============================================================================

def start_hyper_server(
    app,  # Union[str, ASGIApplication] - accepts both app path string or app object
    host: Optional[str] = None,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    config: Optional[ServerConfig] = None,
    **kwargs,
) -> None:
    """
    Start the ASGI server with the best available runtime.

    Automatically selects:
    - Granian (Rust/Tokio) if available - 3-5x faster
    - uvicorn + uvloop if Granian unavailable - 2-4x faster
    - uvicorn + asyncio as fallback

    Args:
        app: Application - either path string (e.g., "backend.main:app") or app object.
             OPTIMIZATION: Pass app object directly in single-worker mode to avoid
             module reimport and double-initialization.
        host: Bind address (default from config)
        port: Port number (default from config)
        workers: Worker count (default from config)
        config: Full ServerConfig object (overrides individual args)
        **kwargs: Additional server-specific options
    """
    if config is None:
        config = ServerConfig()

    # Override config with explicit args
    if host is not None:
        config.host = host
    if port is not None:
        config.port = port
    if workers is not None:
        config.workers = workers

    engine = get_runtime_engine()

    if engine.level == RuntimeLevel.HYPER and engine.available:
        _start_granian(app, config, **kwargs)
    else:
        _start_uvicorn(app, config, engine, **kwargs)


def _start_granian(app: str, config: ServerConfig, **kwargs) -> None:
    """Start Granian (Rust/Tokio) server."""
    from granian import Granian
    from granian.constants import Interfaces, Loops, HTTPModes, ThreadModes

    # Map config to Granian options
    interface = Interfaces.ASGI if config.interface == "asgi" else Interfaces.WSGI

    # HTTP mode selection
    if config.http_version == "2":
        http_mode = HTTPModes.HTTP2
    elif config.http_version == "1":
        http_mode = HTTPModes.HTTP1
    else:
        http_mode = HTTPModes.AUTO

    logger.info(
        f"⚡ [HYPER-RUNTIME] Starting Granian Rust Server\n"
        f"   Engine: Rust/Tokio (Level 3 - HYPER)\n"
        f"   Address: {config.host}:{config.port}\n"
        f"   Workers: {config.workers}\n"
        f"   Interface: {config.interface.upper()}\n"
        f"   HTTP: {http_mode.value if hasattr(http_mode, 'value') else http_mode}"
    )

    granian = Granian(
        app,
        address=config.host,
        port=config.port,
        interface=interface,
        workers=config.workers,
        threads=config.threads,
        threading_mode=ThreadModes.runtime,
        loop=Loops.auto,
        http=http_mode,
        websockets=config.websockets,
        backlog=config.backlog,
        log_level=config.log_level,
        reload=config.reload,
        **kwargs,
    )

    granian.serve()


def _start_uvicorn(app: str, config: ServerConfig, engine: RuntimeEngine, **kwargs) -> None:
    """Start uvicorn with the best available event loop."""
    import uvicorn

    # Activate uvloop if available
    if engine.level == RuntimeLevel.FAST and engine.available:
        try:
            import uvloop
            uvloop.install()
            logger.info(
                f"🚀 [HYPER-RUNTIME] Starting uvicorn with uvloop\n"
                f"   Engine: C/libuv (Level 2 - FAST)\n"
                f"   Address: {config.host}:{config.port}\n"
                f"   Workers: {config.workers}"
            )
        except Exception:
            logger.info(
                f"🐍 [HYPER-RUNTIME] Starting uvicorn with asyncio\n"
                f"   Engine: Python (Level 1 - STANDARD)\n"
                f"   Address: {config.host}:{config.port}"
            )
    else:
        logger.info(
            f"🐍 [HYPER-RUNTIME] Starting uvicorn with asyncio\n"
            f"   Engine: Python (Level 1 - STANDARD)\n"
            f"   Address: {config.host}:{config.port}"
        )

    # Use uvloop if available, otherwise default
    loop = "uvloop" if engine.level >= RuntimeLevel.FAST and engine.available else "auto"

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers if config.workers > 1 else None,
        loop=loop,
        log_level=config.log_level,
        reload=config.reload,
        timeout_keep_alive=config.timeout_keep_alive,
        **kwargs,
    )


# =============================================================================
# Async Runner with Best Runtime
# =============================================================================

def run_async(coro, *, debug: bool = False):
    """
    Run a coroutine with the best available event loop.

    This is a drop-in replacement for asyncio.run() that automatically
    uses uvloop if available.

    Args:
        coro: Coroutine to run
        debug: Enable asyncio debug mode

    Returns:
        The coroutine's return value
    """
    engine = get_runtime_engine()

    # Activate uvloop if available
    if engine.level == RuntimeLevel.FAST and engine.available:
        try:
            import uvloop
            return uvloop.run(coro, debug=debug)
        except Exception:
            pass

    # Fallback to asyncio
    return asyncio.run(coro, debug=debug)


# =============================================================================
# Stats & Diagnostics
# =============================================================================

def get_runtime_stats() -> Dict[str, Any]:
    """Get comprehensive runtime statistics."""
    runtimes = detect_all_runtimes()
    current = get_runtime_engine()

    return {
        "current_engine": {
            "name": current.name,
            "level": current.level.name,
            "level_value": current.level.value,
            "version": current.version,
            "backend": current.backend,
            "activated": current.activated,
        },
        "available_engines": {
            level.name: {
                "name": engine.name,
                "available": engine.available,
                "version": engine.version,
                "description": engine.description,
            }
            for level, engine in runtimes.items()
        },
        "performance_hierarchy": [
            "Level 3 (HYPER): Granian - Rust/Tokio - 3-5x faster",
            "Level 2 (FAST): uvloop - C/libuv - 2-4x faster",
            "Level 1 (STANDARD): asyncio - Python - baseline",
        ],
    }


def print_runtime_report() -> None:
    """Print a formatted runtime status report."""
    stats = get_runtime_stats()
    current = stats["current_engine"]

    print("\n" + "=" * 60)
    print("  JARVIS Hyper-Runtime Status v9.0")
    print("=" * 60)

    # Current engine
    level_icons = {
        "HYPER": "⚡",
        "FAST": "🚀",
        "STANDARD": "🐍",
    }
    icon = level_icons.get(current["level"], "❓")

    print(f"\n  Active Engine: {icon} {current['name']}")
    print(f"  Level: {current['level']} ({current['level_value']}/3)")
    print(f"  Backend: {current['backend']}")
    print(f"  Version: {current['version']}")

    # Available engines
    print("\n  Available Engines:")
    for level_name, engine in stats["available_engines"].items():
        status = "✅" if engine["available"] else "❌"
        print(f"    {status} {level_name}: {engine['name']}")
        if not engine["available"]:
            print(f"       → {engine['description']}")

    # Performance hierarchy
    print("\n  Performance Hierarchy:")
    for line in stats["performance_hierarchy"]:
        print(f"    {line}")

    print("\n" + "=" * 60 + "\n")


# =============================================================================
# Module Initialization
# =============================================================================

# Auto-activate on import for non-Granian use cases
# (Granian manages its own loop, so we skip activation if it will be used)
def _auto_init():
    """Initialize runtime on module import."""
    engine = get_runtime_engine()

    # Only activate uvloop for non-Granian cases
    # Granian handles its own Tokio runtime
    if engine.level == RuntimeLevel.FAST:
        activate_runtime()
    elif engine.level == RuntimeLevel.HYPER:
        logger.debug("[HYPER-RUNTIME] Granian detected - deferring to Rust runtime")
    else:
        logger.debug("[HYPER-RUNTIME] Using standard asyncio")


# Initialize on import
_auto_init()
