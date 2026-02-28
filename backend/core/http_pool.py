"""
HTTP Connection Pool v2.0 - Shared Session Management for Ironcliw
================================================================

Provides a singleton HTTP connection pool that maintains one aiohttp.ClientSession
per base URL (scheme + host + port). Eliminates connection churn from 32+ ephemeral
session creations throughout the codebase.

Features:
    - One aiohttp.ClientSession per base URL (scheme + netloc)
    - TTL-based idle session cleanup (background async task)
    - Max sessions limit with LRU eviction
    - Per-session connection limits (total + per-host)
    - Thread-safe session dict via threading.Lock
    - Active user tracking per session (ref-counting)
    - Full stats reporting + cache registry integration
    - Graceful shutdown (close all sessions)
    - All thresholds configurable via environment variables

Backward Compatibility:
    The v1.0 API (get_session, close_all_sessions, get_pool_stats, PoolConfig,
    HTTPConnectionPool) is preserved as aliases.

Environment Variables:
    Ironcliw_HTTP_POOL_MAX_SESSIONS       Max sessions to maintain (default: 20)
    Ironcliw_HTTP_POOL_TTL_MINUTES        Idle TTL before cleanup (default: 30)
    Ironcliw_HTTP_POOL_CONN_LIMIT         Total connections per session (default: 100)
    Ironcliw_HTTP_POOL_CONN_PER_HOST      Connections per host (default: 10)
    Ironcliw_HTTP_POOL_TIMEOUT            Total request timeout seconds (default: 30)
    Ironcliw_HTTP_POOL_CONNECT_TIMEOUT    Connection timeout seconds (default: 10)
    Ironcliw_HTTP_POOL_CLEANUP_INTERVAL   Cleanup check interval seconds (default: 300)

Usage:
    from backend.core.http_pool import managed_session, get_http_pool

    # Context manager (preferred):
    async with managed_session("https://api.example.com") as session:
        async with session.get("/endpoint") as resp:
            data = await resp.json()

    # Direct session access:
    pool = get_http_pool()
    session = await pool.get_session("https://api.example.com")

    # Shutdown:
    await close_http_pool()

Author: Ironcliw Development Team
Version: 2.0.0 (February 2026)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# aiohttp import guard
# ---------------------------------------------------------------------------

try:
    import aiohttp

    _AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    _AIOHTTP_AVAILABLE = False
    logger.warning(
        "[HTTPPool] aiohttp not available — HTTP pool will operate in stub mode"
    )


# ---------------------------------------------------------------------------
# Helpers — safe env var parsing
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable with a safe fallback."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
        if value < 0:
            logger.warning(
                "[HTTPPool] Negative value for %s=%s, using default %d",
                name, raw, default,
            )
            return default
        return value
    except (ValueError, TypeError):
        logger.warning(
            "[HTTPPool] Invalid value for %s=%s, using default %d",
            name, raw, default,
        )
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from an environment variable with a safe fallback."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
        if value < 0:
            logger.warning(
                "[HTTPPool] Negative value for %s=%s, using default %s",
                name, raw, default,
            )
            return default
        return value
    except (ValueError, TypeError):
        logger.warning(
            "[HTTPPool] Invalid value for %s=%s, using default %s",
            name, raw, default,
        )
        return default


def _extract_base_url(url: str) -> str:
    """
    Extract the base URL (scheme + netloc) from a full URL.

    Examples:
        "https://api.example.com/v1/chat" -> "https://api.example.com"
        "http://localhost:8080/health"     -> "http://localhost:8080"
        "http://localhost:8080"            -> "http://localhost:8080"
    """
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    if not netloc:
        # Handle bare host:port strings like "localhost:8080"
        netloc = parsed.path.split("/")[0]
    if not netloc:
        raise ValueError(f"Cannot extract base URL from: {url!r}")
    return f"{scheme}://{netloc}"


# ---------------------------------------------------------------------------
# Data classes — per-session and aggregate stats
# ---------------------------------------------------------------------------

@dataclass
class SessionStats:
    """Per-session statistics (read-only snapshot)."""

    base_url: str
    active_users: int = 0
    total_requests: int = 0
    created_at: float = 0.0
    last_used_at: float = 0.0
    is_closed: bool = False

    @property
    def idle_seconds(self) -> float:
        if self.last_used_at <= 0:
            return 0.0
        return time.monotonic() - self.last_used_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "active_users": self.active_users,
            "total_requests": self.total_requests,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "idle_seconds": round(self.idle_seconds, 1),
            "is_closed": self.is_closed,
        }


@dataclass
class HTTPPoolStats:
    """Aggregate pool statistics (read-only snapshot)."""

    total_sessions: int = 0
    active_sessions: int = 0  # sessions with active_users > 0
    idle_sessions: int = 0
    total_requests: int = 0
    sessions_created: int = 0
    sessions_closed: int = 0
    sessions_evicted_ttl: int = 0
    sessions_evicted_lru: int = 0
    sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sessions": self.total_sessions,
            "active_sessions": self.active_sessions,
            "idle_sessions": self.idle_sessions,
            "total_requests": self.total_requests,
            "sessions_created": self.sessions_created,
            "sessions_closed": self.sessions_closed,
            "sessions_evicted_ttl": self.sessions_evicted_ttl,
            "sessions_evicted_lru": self.sessions_evicted_lru,
            "sessions": self.sessions,
        }


# ---------------------------------------------------------------------------
# _SessionEntry — internal wrapper around an aiohttp.ClientSession
# ---------------------------------------------------------------------------

class _SessionEntry:
    """Internal bookkeeping for a single managed session."""

    __slots__ = (
        "base_url",
        "session",
        "connector",
        "active_users",
        "total_requests",
        "created_at",
        "last_used_at",
        "_closed",
    )

    def __init__(
        self,
        base_url: str,
        session: Any,  # aiohttp.ClientSession
        connector: Any,  # aiohttp.TCPConnector
    ) -> None:
        self.base_url = base_url
        self.session = session
        self.connector = connector
        self.active_users: int = 0
        self.total_requests: int = 0
        now = time.monotonic()
        self.created_at: float = now
        self.last_used_at: float = now
        self._closed: bool = False

    @property
    def is_idle(self) -> bool:
        return self.active_users <= 0 and not self._closed

    @property
    def idle_seconds(self) -> float:
        if self.last_used_at <= 0:
            return 0.0
        return time.monotonic() - self.last_used_at

    def touch(self) -> None:
        """Record a usage timestamp."""
        self.last_used_at = time.monotonic()

    def acquire(self) -> None:
        """Record a new active user."""
        self.active_users += 1
        self.total_requests += 1
        self.touch()

    def release(self) -> None:
        """Record that an active user is done."""
        self.active_users = max(0, self.active_users - 1)
        self.touch()

    async def close(self) -> None:
        """Close the underlying session and connector."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.session is not None and not self.session.closed:
                await self.session.close()
        except Exception as exc:
            logger.debug(
                "[HTTPPool] Error closing session for %s: %s",
                self.base_url, exc,
            )
        # Allow connector cleanup time (aiohttp recommendation)
        try:
            await asyncio.sleep(0.25)
        except Exception:
            pass

    def get_stats(self) -> SessionStats:
        return SessionStats(
            base_url=self.base_url,
            active_users=self.active_users,
            total_requests=self.total_requests,
            created_at=self.created_at,
            last_used_at=self.last_used_at,
            is_closed=self._closed,
        )


# ---------------------------------------------------------------------------
# HTTPPool — singleton connection pool
# ---------------------------------------------------------------------------

class HTTPPool:
    """
    Singleton HTTP connection pool.

    Maintains one aiohttp.ClientSession per base URL (scheme + netloc).
    Sessions are lazily created, TTL-evicted when idle, and capped at
    a configurable maximum count.

    Thread-safe: the internal session dict is guarded by a threading.Lock
    (needed because sessions can be requested from multiple threads via
    run_coroutine_threadsafe or similar patterns).
    """

    _instance: Optional["HTTPPool"] = None
    _init_lock = threading.Lock()

    # -- Singleton ----------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "HTTPPool":
        """Get or create the singleton HTTPPool."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing only)."""
        with cls._init_lock:
            cls._instance = None

    # -- Init ---------------------------------------------------------------

    def __init__(self) -> None:
        # Configuration — all from env vars with sane defaults
        self._max_sessions: int = _env_int(
            "Ironcliw_HTTP_POOL_MAX_SESSIONS", 20,
        )
        self._ttl_seconds: float = (
            _env_float("Ironcliw_HTTP_POOL_TTL_MINUTES", 30) * 60.0
        )
        self._conn_limit: int = _env_int(
            "Ironcliw_HTTP_POOL_CONN_LIMIT", 100,
        )
        self._conn_per_host: int = _env_int(
            "Ironcliw_HTTP_POOL_CONN_PER_HOST", 10,
        )
        self._total_timeout: float = _env_float(
            "Ironcliw_HTTP_POOL_TIMEOUT", 30,
        )
        self._connect_timeout: float = _env_float(
            "Ironcliw_HTTP_POOL_CONNECT_TIMEOUT", 10,
        )
        self._cleanup_interval: float = _env_float(
            "Ironcliw_HTTP_POOL_CLEANUP_INTERVAL", 300,
        )

        # Internal state
        self._sessions: Dict[str, _SessionEntry] = {}
        self._lock = threading.Lock()

        # Lifetime counters
        self._sessions_created: int = 0
        self._sessions_closed: int = 0
        self._sessions_evicted_ttl: int = 0
        self._sessions_evicted_lru: int = 0

        # Cleanup task handle
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown: bool = False

        # Cache registry integration
        self._register_with_cache_registry()

        logger.info(
            "[HTTPPool] Initialized (max_sessions=%d, ttl=%ds, "
            "conn_limit=%d, conn_per_host=%d, timeout=%ss, "
            "connect_timeout=%ss)",
            self._max_sessions,
            int(self._ttl_seconds),
            self._conn_limit,
            self._conn_per_host,
            self._total_timeout,
            self._connect_timeout,
        )

    # -- Cache registry integration -----------------------------------------

    def _register_with_cache_registry(self) -> None:
        """Register this pool with the global cache registry if available."""
        try:
            from backend.utils.cache_registry import get_cache_registry

            get_cache_registry().register("http_pool", self)
            logger.debug("[HTTPPool] Registered with cache registry")
        except Exception:
            pass  # Cache registry not available — that's fine

    # -- Session lifecycle --------------------------------------------------

    def _create_session_sync(self, base_url: str) -> _SessionEntry:
        """
        Create a new aiohttp.ClientSession for the given base URL.

        MUST be called from an async context (inside a running event loop).
        The caller MUST already hold ``self._lock``.
        """
        if not _AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is not installed. Install it with: pip install aiohttp"
            )

        connector = aiohttp.TCPConnector(
            limit=self._conn_limit,
            limit_per_host=self._conn_per_host,
            ttl_dns_cache=300,  # 5-minute DNS cache
            enable_cleanup_closed=True,
            force_close=False,
        )

        timeout_obj = aiohttp.ClientTimeout(
            total=self._total_timeout,
            connect=self._connect_timeout,
        )

        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_obj,
            headers={"User-Agent": "Ironcliw/1.0"},
            trust_env=True,  # Respect HTTP_PROXY / HTTPS_PROXY env vars
        )

        entry = _SessionEntry(
            base_url=base_url,
            session=session,
            connector=connector,
        )

        self._sessions_created += 1
        logger.debug(
            "[HTTPPool] Created session for %s (total: %d)",
            base_url,
            len(self._sessions) + 1,
        )
        return entry

    async def _evict_lru_session_locked(self) -> None:
        """
        Evict the least-recently-used idle session to make room.

        The caller MUST already hold ``self._lock``.
        """
        # Prefer idle sessions for eviction
        candidates = [
            entry
            for entry in self._sessions.values()
            if entry.is_idle and not entry._closed
        ]
        if not candidates:
            # Fall back to any non-closed session
            candidates = [
                entry
                for entry in self._sessions.values()
                if not entry._closed
            ]
        if not candidates:
            return

        candidates.sort(key=lambda e: e.last_used_at)
        victim = candidates[0]

        logger.info(
            "[HTTPPool] Evicting LRU session for %s "
            "(idle %.0fs, %d active users)",
            victim.base_url,
            victim.idle_seconds,
            victim.active_users,
        )

        self._sessions.pop(victim.base_url, None)
        self._sessions_evicted_lru += 1
        self._sessions_closed += 1

        # Close the session (best-effort, we're holding the lock but close
        # only does a non-blocking mark + schedules connector cleanup)
        try:
            await victim.close()
        except Exception as exc:
            logger.debug(
                "[HTTPPool] Error during LRU eviction close: %s", exc,
            )

    async def get_session(
        self,
        url: str,
        *,
        config: Optional[Any] = None,
    ) -> Any:  # returns aiohttp.ClientSession
        """
        Get or create an aiohttp.ClientSession for the given URL.

        The base URL (scheme + netloc) is extracted and used as the
        cache key.  If the pool is at max capacity, the least-recently-used
        idle session is evicted first.

        The optional *config* parameter is accepted for backward compatibility
        with the v1 ``PoolConfig`` API but is ignored in v2.

        Args:
            url: Any URL.  Only scheme + host + port are used as key.
            config: (Ignored) Backward-compat placeholder.

        Returns:
            An aiohttp.ClientSession configured for the base URL.

        Raises:
            RuntimeError: If aiohttp is not installed.
            ValueError: If the URL cannot be parsed.
        """
        if not _AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is not installed. Install it with: pip install aiohttp"
            )

        base_url = _extract_base_url(url)

        with self._lock:
            entry = self._sessions.get(base_url)

            # Reuse existing live session
            if entry is not None and not entry._closed:
                if entry.session is not None and not entry.session.closed:
                    entry.touch()
                    return entry.session
                else:
                    # Session was closed externally — clean up entry
                    del self._sessions[base_url]
                    self._sessions_closed += 1

            # Capacity check — evict if necessary
            if len(self._sessions) >= self._max_sessions:
                await self._evict_lru_session_locked()

            # Create new session (sync creation, runs inside event loop)
            entry = self._create_session_sync(base_url)
            self._sessions[base_url] = entry
            return entry.session

    async def close_session(self, url: str) -> bool:
        """
        Close the session for a specific base URL.

        Args:
            url: The URL whose base-URL session should be closed.

        Returns:
            True if a session was found and closed, False otherwise.
        """
        base_url = _extract_base_url(url)

        with self._lock:
            entry = self._sessions.pop(base_url, None)

        if entry is None:
            return False

        self._sessions_closed += 1
        await entry.close()
        logger.debug("[HTTPPool] Closed session for %s", base_url)
        return True

    async def close_all(self) -> int:
        """
        Close all managed sessions.

        Returns:
            Number of sessions closed.
        """
        with self._lock:
            entries = list(self._sessions.values())
            self._sessions.clear()

        closed_count = 0
        for entry in entries:
            try:
                await entry.close()
                closed_count += 1
                self._sessions_closed += 1
            except Exception as exc:
                logger.debug(
                    "[HTTPPool] Error closing session for %s: %s",
                    entry.base_url, exc,
                )

        logger.info("[HTTPPool] Closed %d sessions", closed_count)
        return closed_count

    # -- Cleanup loop -------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Background task that periodically evicts idle sessions past TTL."""
        logger.debug(
            "[HTTPPool] Cleanup loop started (interval=%.0fs, ttl=%.0fs)",
            self._cleanup_interval,
            self._ttl_seconds,
        )
        while not self._shutdown:
            try:
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break

            if self._shutdown:
                break

            await self._run_cleanup()

        logger.debug("[HTTPPool] Cleanup loop stopped")

    async def _run_cleanup(self) -> None:
        """Single pass of TTL-based session cleanup."""
        now = time.monotonic()
        to_close: list = []

        with self._lock:
            expired_keys: list = []
            for base_url, entry in self._sessions.items():
                if entry._closed:
                    expired_keys.append(base_url)
                    continue
                if (
                    entry.is_idle
                    and (now - entry.last_used_at) > self._ttl_seconds
                ):
                    expired_keys.append(base_url)
                    to_close.append(entry)

            for key in expired_keys:
                self._sessions.pop(key, None)

        if to_close:
            for entry in to_close:
                try:
                    await entry.close()
                    self._sessions_evicted_ttl += 1
                    self._sessions_closed += 1
                except Exception as exc:
                    logger.debug(
                        "[HTTPPool] Error closing expired session for %s: %s",
                        entry.base_url, exc,
                    )

            logger.info(
                "[HTTPPool] TTL cleanup: evicted %d idle session(s), "
                "%d remaining",
                len(to_close),
                len(self._sessions),
            )

    def start_cleanup_loop(self) -> None:
        """
        Start the background cleanup task.

        Safe to call multiple times — only one loop will run.
        Must be called from within an async context (running event loop).
        """
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return  # Already running
        self._shutdown = False
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(
                self._cleanup_loop(),
                name="http-pool-cleanup",
            )
        except RuntimeError:
            logger.warning(
                "[HTTPPool] No running event loop — cleanup loop not started"
            )

    def stop_cleanup_loop(self) -> None:
        """Cancel the background cleanup task."""
        self._shutdown = True
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            self._cleanup_task = None
            logger.debug("[HTTPPool] Cleanup loop cancelled")

    # -- Stats / cache registry interface -----------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return pool statistics.

        Compatible with the CacheRegistry ``CacheLike`` protocol
        (has ``get_stats() -> dict``).
        """
        with self._lock:
            entries = list(self._sessions.values())

        active_count = sum(
            1 for e in entries if e.active_users > 0 and not e._closed
        )
        idle_count = sum(
            1 for e in entries if e.is_idle and not e._closed
        )
        total_requests = sum(e.total_requests for e in entries)

        per_session: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            per_session[entry.base_url] = entry.get_stats().to_dict()

        return {
            # CacheRegistry-compatible fields
            "entries": len(entries),
            "size_mb": 0.0,  # Sessions don't have meaningful size
            # Pool-specific fields
            "total_sessions": len(entries),
            "active_sessions": active_count,
            "idle_sessions": idle_count,
            "total_requests": total_requests,
            "active_users": sum(e.active_users for e in entries),
            "sessions_created": self._sessions_created,
            "sessions_closed": self._sessions_closed,
            "sessions_evicted_ttl": self._sessions_evicted_ttl,
            "sessions_evicted_lru": self._sessions_evicted_lru,
            "max_sessions": self._max_sessions,
            "ttl_seconds": self._ttl_seconds,
            "cleanup_running": (
                self._cleanup_task is not None
                and not self._cleanup_task.done()
            ),
            "sessions": per_session,
        }

    def get_pool_stats(self) -> HTTPPoolStats:
        """Return typed pool statistics as an ``HTTPPoolStats`` dataclass."""
        raw = self.get_stats()
        return HTTPPoolStats(
            total_sessions=raw["total_sessions"],
            active_sessions=raw["active_sessions"],
            idle_sessions=raw["idle_sessions"],
            total_requests=raw["total_requests"],
            sessions_created=raw["sessions_created"],
            sessions_closed=raw["sessions_closed"],
            sessions_evicted_ttl=raw["sessions_evicted_ttl"],
            sessions_evicted_lru=raw["sessions_evicted_lru"],
            sessions=raw["sessions"],
        )

    # -- Managed session context manager ------------------------------------

    @asynccontextmanager
    async def managed(self, url: str) -> AsyncIterator[Any]:
        """
        Async context manager that yields a shared session.

        The session is NOT closed on exit — it remains in the pool.
        Active user count is tracked for idle/eviction decisions.

        Usage:
            pool = get_http_pool()
            async with pool.managed("https://api.example.com") as session:
                async with session.get("/endpoint") as resp:
                    data = await resp.json()
        """
        base_url = _extract_base_url(url)
        session = await self.get_session(url)

        # Track active user
        with self._lock:
            entry = self._sessions.get(base_url)
            if entry is not None:
                entry.acquire()

        try:
            yield session
        finally:
            with self._lock:
                entry = self._sessions.get(base_url)
                if entry is not None:
                    entry.release()

    # -- Shutdown -----------------------------------------------------------

    async def shutdown(self) -> None:
        """Full shutdown: stop cleanup loop and close all sessions."""
        logger.info("[HTTPPool] Shutting down...")
        self.stop_cleanup_loop()
        count = await self.close_all()
        logger.info("[HTTPPool] Shutdown complete (%d sessions closed)", count)

    # -- Dunder helpers -----------------------------------------------------

    def __del__(self) -> None:
        """Best-effort warning on garbage collection with unclosed sessions."""
        remaining = len(self._sessions) if hasattr(self, "_sessions") else 0
        if remaining:
            logger.debug(
                "[HTTPPool] __del__ with %d unclosed sessions — "
                "call shutdown() explicitly for clean teardown",
                remaining,
            )

    def __repr__(self) -> str:
        return (
            f"HTTPPool(sessions={len(self._sessions)}, "
            f"max={self._max_sessions}, "
            f"created={self._sessions_created}, "
            f"closed={self._sessions_closed})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience functions (v2 API)
# ---------------------------------------------------------------------------

def get_http_pool() -> HTTPPool:
    """Get the singleton HTTPPool instance."""
    return HTTPPool.get_instance()


@asynccontextmanager
async def managed_session(url: str) -> AsyncIterator[Any]:
    """
    Module-level convenience: get a managed session from the pool.

    The session is shared and NOT closed on context exit.
    Active user count is tracked for idle detection.

    Usage:
        from backend.core.http_pool import managed_session

        async with managed_session("https://api.example.com") as session:
            async with session.get("/endpoint") as resp:
                data = await resp.json()
    """
    pool = get_http_pool()
    async with pool.managed(url) as session:
        yield session


async def close_http_pool() -> None:
    """
    Module-level convenience: shut down the singleton pool.

    Call this during application shutdown to cleanly close all sessions.
    """
    pool = get_http_pool()
    await pool.shutdown()


def get_http_pool_stats() -> Dict[str, Any]:
    """Module-level convenience: get pool stats (sync call)."""
    pool = get_http_pool()
    return pool.get_stats()


# ---------------------------------------------------------------------------
# Backward compatibility aliases (v1 API)
# ---------------------------------------------------------------------------
# The original v1 module exposed these names.  Preserve them so that
# existing ``from backend.core.http_pool import get_session`` continues
# to work without changes.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PoolConfig:
    """Backward-compat: v1 per-session configuration (ignored by v2 pool)."""

    max_connections: int = _env_int("Ironcliw_HTTP_POOL_CONN_LIMIT", 100)
    max_connections_per_host: int = _env_int("Ironcliw_HTTP_POOL_CONN_PER_HOST", 10)
    ttl_seconds: float = _env_float("Ironcliw_HTTP_POOL_TTL_MINUTES", 30) * 60.0
    connect_timeout: float = _env_float("Ironcliw_HTTP_POOL_CONNECT_TIMEOUT", 10)
    total_timeout: float = _env_float("Ironcliw_HTTP_POOL_TIMEOUT", 30)
    keepalive_timeout: float = 30.0
    enable_cleanup_task: bool = True
    cleanup_interval: float = 300.0


# Alias the class name used in v1
HTTPConnectionPool = HTTPPool


async def get_session(
    base_url: str,
    *,
    config: Optional[PoolConfig] = None,
) -> Any:
    """Backward-compat: get or create a pooled session for *base_url*."""
    return await get_http_pool().get_session(base_url, config=config)


async def close_all_sessions() -> None:
    """Backward-compat: close all pooled sessions (call at shutdown)."""
    await close_http_pool()


def get_pool_stats() -> Dict[str, Any]:
    """Backward-compat: return current pool statistics."""
    return get_http_pool_stats()


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # v2 core class
    "HTTPPool",
    # v2 data classes
    "HTTPPoolStats",
    "SessionStats",
    # v2 module-level functions
    "get_http_pool",
    "managed_session",
    "close_http_pool",
    "get_http_pool_stats",
    # Backward compatibility (v1)
    "PoolConfig",
    "HTTPConnectionPool",
    "get_session",
    "close_all_sessions",
    "get_pool_stats",
]
