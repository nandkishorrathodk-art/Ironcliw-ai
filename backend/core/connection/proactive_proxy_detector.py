"""
Proactive Proxy Detector with Sub-100ms Fast-Fail
==================================================

Detects Cloud SQL Proxy availability BEFORE connection attempts.
Uses multiple detection strategies for fastest possible result.

Detection strategies (in order of speed):
1. Cached status (0ms) - use cached result if fresh
2. Non-blocking socket check (10-50ms) - is port listening?
3. Process check (5-20ms) - is cloud_sql_proxy running?

This is PROACTIVE detection - we detect BEFORE attempting connections,
avoiding the 2+ second TCP timeout when proxy is down.

Author: Ironcliw System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ProxyStatus(Enum):
    """Proxy availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class ProxyDetectorConfig:
    """
    Configuration for proactive proxy detection.

    All values can be overridden via environment variables.
    """
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 5432
    socket_timeout_ms: int = 50  # 50ms max for socket check
    cache_ttl_seconds: float = 2.0  # Cache validity
    process_name: str = "cloud_sql_proxy"

    def __post_init__(self):
        """Load from environment if available."""
        self.proxy_host = os.getenv('CLOUD_SQL_PROXY_HOST', self.proxy_host)
        self.proxy_port = int(os.getenv('CLOUD_SQL_PROXY_PORT', str(self.proxy_port)))
        self.socket_timeout_ms = int(os.getenv(
            'CLOUD_SQL_SOCKET_TIMEOUT_MS', str(self.socket_timeout_ms)
        ))
        self.cache_ttl_seconds = float(os.getenv(
            'CLOUD_SQL_CACHE_TTL', str(self.cache_ttl_seconds)
        ))
        self.process_name = os.getenv(
            'CLOUD_SQL_PROCESS_NAME', self.process_name
        )


class ProactiveProxyDetector:
    """
    Proactive proxy detector with sub-100ms detection.

    Uses a tiered detection strategy:
    1. Check cache first (0ms)
    2. Non-blocking socket check (10-50ms)
    3. Process check as fallback (5-20ms)

    Never waits for TCP connection timeout (2+ seconds).

    Usage:
        detector = ProactiveProxyDetector()

        # Before making a connection
        status, msg = await detector.detect()
        if status == ProxyStatus.UNAVAILABLE:
            # Fast-fail, don't try connection
            return use_fallback()

        # Proxy available, proceed with connection
        conn = await create_connection()
    """

    __slots__ = (
        '_config', '_cached_status', '_cache_time', '_process_check_available',
    )

    def __init__(self, config: Optional[ProxyDetectorConfig] = None):
        """
        Initialize detector.

        Args:
            config: Configuration (uses defaults + env vars if None)
        """
        self._config = config or ProxyDetectorConfig()
        self._cached_status: Optional[ProxyStatus] = None
        self._cache_time: Optional[datetime] = None
        self._process_check_available = True

        logger.debug(
            f"ProactiveProxyDetector initialized: "
            f"{self._config.proxy_host}:{self._config.proxy_port}"
        )

    @property
    def config(self) -> ProxyDetectorConfig:
        """Get configuration."""
        return self._config

    async def detect(self, force: bool = False) -> Tuple[ProxyStatus, str]:
        """
        Detect proxy availability with sub-100ms guarantee.

        Args:
            force: Bypass cache and check fresh

        Returns:
            Tuple of (ProxyStatus, info_message)
        """
        # Strategy 1: Check cache (0ms)
        if not force and self._is_cache_valid() and self._cached_status is not None:
            return self._cached_status, "Cached status"

        # Strategy 2: Non-blocking socket check (10-50ms)
        status, msg = await self._check_socket_fast()

        if status == ProxyStatus.AVAILABLE:
            self._update_cache(status)
            return status, msg

        # Strategy 3: Process check as confirmation (5-20ms)
        if self._process_check_available and status == ProxyStatus.UNAVAILABLE:
            process_running = await self._check_process()
            if not process_running:
                status = ProxyStatus.UNAVAILABLE
                msg = "Proxy process not running"

        self._update_cache(status)
        return status, msg

    def _is_cache_valid(self) -> bool:
        """Check if cached status is still valid."""
        if self._cached_status is None or self._cache_time is None:
            return False
        elapsed = (datetime.now() - self._cache_time).total_seconds()
        return elapsed < self._config.cache_ttl_seconds

    def _update_cache(self, status: ProxyStatus) -> None:
        """Update cached status."""
        self._cached_status = status
        self._cache_time = datetime.now()

    async def _check_socket_fast(self) -> Tuple[ProxyStatus, str]:
        """
        Non-blocking socket check with fast timeout.

        Uses socket.setblocking(False) for immediate return on connection refused.
        This avoids the 2+ second TCP timeout.
        """
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setblocking(False)

            # Non-blocking connect - returns immediately with error code
            result = sock.connect_ex((self._config.proxy_host, self._config.proxy_port))

            if result == 0:
                # Immediately connected - proxy is listening
                return ProxyStatus.AVAILABLE, "Port accepting connections"

            elif result in (111, 61):  # ECONNREFUSED (Linux=111, macOS=61)
                # Connection refused - port not listening
                return ProxyStatus.UNAVAILABLE, "Connection refused (proxy not running)"

            elif result in (115, 36, 10035):  # EINPROGRESS (Linux=115, macOS=36, Windows=10035)
                # Connection in progress - wait briefly with timeout
                try:
                    # Use selector for cross-platform compatibility
                    loop = asyncio.get_running_loop()
                    await asyncio.wait_for(
                        loop.sock_connect(sock, (
                            self._config.proxy_host, self._config.proxy_port
                        )),
                        timeout=self._config.socket_timeout_ms / 1000
                    )
                    return ProxyStatus.AVAILABLE, "Port accepting connections"
                except asyncio.TimeoutError:
                    return ProxyStatus.UNAVAILABLE, "Connection timeout (proxy not responding)"
                except OSError as e:
                    if "Connection refused" in str(e) or getattr(e, 'errno', 0) in (61, 111):
                        return ProxyStatus.UNAVAILABLE, "Connection refused"
                    return ProxyStatus.UNKNOWN, f"Connection error: {e}"

            else:
                return ProxyStatus.UNKNOWN, f"Socket error code: {result}"

        except OSError as e:
            error_str = str(e)
            if "Connection refused" in error_str or getattr(e, 'errno', 0) in (61, 111):
                return ProxyStatus.UNAVAILABLE, "Connection refused"
            return ProxyStatus.UNKNOWN, f"Socket error: {e}"

        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    async def _check_process(self) -> bool:
        """
        Check if proxy process is running (fast subprocess check).

        Uses pgrep on Unix systems for fast process lookup.
        Falls back gracefully if pgrep not available.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                'pgrep', '-f', self._config.process_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            try:
                await asyncio.wait_for(proc.wait(), timeout=0.02)  # 20ms max
            except asyncio.TimeoutError:
                # Process check timed out - kill and assume unknown
                try:
                    proc.kill()
                except Exception:
                    pass
                return True  # Assume running if can't check

            return proc.returncode == 0

        except FileNotFoundError:
            # pgrep not available (e.g., Windows)
            self._process_check_available = False
            logger.debug("pgrep not available, skipping process check")
            return True  # Assume running if can't check

        except Exception as e:
            logger.debug(f"Process check failed: {e}")
            return True  # Assume running on error

    def invalidate_cache(self) -> None:
        """
        Invalidate cached status.

        Call this when you know the proxy state might have changed
        (e.g., after starting/stopping proxy).
        """
        self._cached_status = None
        self._cache_time = None
        logger.debug("Proxy cache invalidated")

    def __repr__(self) -> str:
        """String representation for debugging."""
        cache_status = "valid" if self._is_cache_valid() else "invalid"
        return (
            f"ProactiveProxyDetector("
            f"{self._config.proxy_host}:{self._config.proxy_port}, "
            f"cache={cache_status})"
        )
