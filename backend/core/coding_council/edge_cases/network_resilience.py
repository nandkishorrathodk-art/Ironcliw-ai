"""
v77.0: Network Resilience - Gap #33
====================================

Network resilience and reconnection:
- Connection pooling
- Automatic reconnection
- Backoff strategies
- Connection health monitoring
- Failover support

Author: JARVIS v77.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConnectionState(Enum):
    """State of a connection."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class ReconnectionStrategy(Enum):
    """Reconnection strategy."""
    IMMEDIATE = "immediate"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


@dataclass
class ConnectionConfig:
    """Configuration for a connection."""
    name: str
    host: str
    port: int
    timeout: float = 30.0
    max_retries: int = 10
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    strategy: ReconnectionStrategy = ReconnectionStrategy.EXPONENTIAL
    health_check_interval: float = 30.0


@dataclass
class ConnectionStats:
    """Statistics for a connection."""
    total_connects: int = 0
    total_disconnects: int = 0
    total_errors: int = 0
    current_retry: int = 0
    last_connect_time: Optional[float] = None
    last_disconnect_time: Optional[float] = None
    last_error: Optional[str] = None
    uptime_seconds: float = 0.0


class Connection(ABC, Generic[T]):
    """
    Base class for resilient connections.

    Subclass this for specific connection types (WebSocket, Redis, etc.)
    """

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.state = ConnectionState.DISCONNECTED
        self.stats = ConnectionStats()
        self._connection: Optional[T] = None
        self._lock = asyncio.Lock()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._state_callbacks: List[Callable[[ConnectionState, ConnectionState], Coroutine]] = []

    @abstractmethod
    async def _connect(self) -> T:
        """Establish the connection. Override in subclass."""
        pass

    @abstractmethod
    async def _disconnect(self, connection: T) -> None:
        """Close the connection. Override in subclass."""
        pass

    @abstractmethod
    async def _health_check(self, connection: T) -> bool:
        """Check connection health. Override in subclass."""
        pass

    async def connect(self) -> bool:
        """Connect with retry logic."""
        async with self._lock:
            if self.state == ConnectionState.CONNECTED:
                return True

            await self._set_state(ConnectionState.CONNECTING)
            retry = 0

            while retry <= self.config.max_retries:
                try:
                    self._connection = await asyncio.wait_for(
                        self._connect(),
                        timeout=self.config.timeout,
                    )

                    await self._set_state(ConnectionState.CONNECTED)
                    self.stats.total_connects += 1
                    self.stats.last_connect_time = time.time()
                    self.stats.current_retry = 0

                    # Start health check task
                    self._start_health_check()

                    logger.info(f"[{self.config.name}] Connected to {self.config.host}:{self.config.port}")
                    return True

                except Exception as e:
                    retry += 1
                    self.stats.total_errors += 1
                    self.stats.last_error = str(e)
                    self.stats.current_retry = retry

                    if retry > self.config.max_retries:
                        await self._set_state(ConnectionState.FAILED)
                        logger.error(f"[{self.config.name}] Failed to connect after {retry} attempts: {e}")
                        return False

                    delay = self._get_retry_delay(retry)
                    logger.warning(f"[{self.config.name}] Connection failed (attempt {retry}), retrying in {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)

        return False

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        async with self._lock:
            self._stop_health_check()

            if self._connection:
                try:
                    await self._disconnect(self._connection)
                except Exception as e:
                    logger.error(f"[{self.config.name}] Error during disconnect: {e}")
                finally:
                    self._connection = None

            self.stats.total_disconnects += 1
            self.stats.last_disconnect_time = time.time()
            await self._set_state(ConnectionState.DISCONNECTED)

            logger.info(f"[{self.config.name}] Disconnected")

    async def reconnect(self) -> bool:
        """Force reconnection."""
        await self._set_state(ConnectionState.RECONNECTING)
        await self.disconnect()
        return await self.connect()

    def get_connection(self) -> Optional[T]:
        """Get the underlying connection."""
        return self._connection

    def is_connected(self) -> bool:
        """Check if connected."""
        return self.state == ConnectionState.CONNECTED

    def on_state_change(self, callback: Callable[[ConnectionState, ConnectionState], Coroutine]) -> None:
        """Register state change callback."""
        self._state_callbacks.append(callback)

    async def _set_state(self, new_state: ConnectionState) -> None:
        """Set state and notify callbacks."""
        old_state = self.state
        self.state = new_state

        if old_state != new_state:
            for callback in self._state_callbacks:
                try:
                    await callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"[{self.config.name}] State callback error: {e}")

    def _get_retry_delay(self, retry: int) -> float:
        """Calculate retry delay based on strategy."""
        base = self.config.reconnect_delay
        max_delay = self.config.max_reconnect_delay

        if self.config.strategy == ReconnectionStrategy.IMMEDIATE:
            delay = 0
        elif self.config.strategy == ReconnectionStrategy.LINEAR:
            delay = base * retry
        elif self.config.strategy == ReconnectionStrategy.EXPONENTIAL:
            delay = base * (2 ** (retry - 1))
        elif self.config.strategy == ReconnectionStrategy.FIBONACCI:
            a, b = 1, 1
            for _ in range(retry):
                a, b = b, a + b
            delay = base * a
        else:
            delay = base

        return min(delay, max_delay)

    def _start_health_check(self) -> None:
        """Start health check task."""
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_check_loop())

    def _stop_health_check(self) -> None:
        """Stop health check task."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if self._connection:
                    healthy = await self._health_check(self._connection)

                    if not healthy:
                        logger.warning(f"[{self.config.name}] Health check failed, reconnecting")
                        await self.reconnect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.config.name}] Health check error: {e}")
                await self.reconnect()

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "host": f"{self.config.host}:{self.config.port}",
            "total_connects": self.stats.total_connects,
            "total_disconnects": self.stats.total_disconnects,
            "total_errors": self.stats.total_errors,
            "current_retry": self.stats.current_retry,
            "last_error": self.stats.last_error,
        }


class ConnectionPool(Generic[T]):
    """
    Pool of connections with load balancing.

    Features:
    - Multiple connections
    - Round-robin or least-connections routing
    - Automatic failover
    - Health monitoring
    """

    def __init__(
        self,
        name: str,
        connection_factory: Callable[[ConnectionConfig], Connection[T]],
        configs: List[ConnectionConfig],
        min_connections: int = 1,
    ):
        self.name = name
        self.connection_factory = connection_factory
        self.configs = configs
        self.min_connections = min_connections

        self._connections: List[Connection[T]] = []
        self._current_index = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the connection pool."""
        for config in self.configs[:self.min_connections]:
            conn = self.connection_factory(config)
            await conn.connect()
            self._connections.append(conn)

        logger.info(f"[ConnectionPool:{self.name}] Started with {len(self._connections)} connections")

    async def stop(self) -> None:
        """Stop all connections."""
        for conn in self._connections:
            await conn.disconnect()
        self._connections.clear()
        logger.info(f"[ConnectionPool:{self.name}] Stopped")

    async def get_connection(self) -> Optional[T]:
        """Get an available connection (round-robin)."""
        async with self._lock:
            if not self._connections:
                return None

            # Try to find a connected connection
            for _ in range(len(self._connections)):
                conn = self._connections[self._current_index]
                self._current_index = (self._current_index + 1) % len(self._connections)

                if conn.is_connected():
                    return conn.get_connection()

            # No healthy connections, try to reconnect
            for conn in self._connections:
                if await conn.reconnect():
                    return conn.get_connection()

            return None

    async def execute(
        self,
        operation: Callable[[T], Coroutine],
        retries: int = 3,
    ) -> Any:
        """
        Execute an operation on any available connection.

        Retries on different connections on failure.
        """
        last_error = None

        for _ in range(retries):
            connection = await self.get_connection()

            if connection is None:
                last_error = Exception("No available connections")
                continue

            try:
                return await operation(connection)
            except Exception as e:
                last_error = e
                logger.warning(f"[ConnectionPool:{self.name}] Operation failed: {e}")

        raise last_error or Exception("Operation failed")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        connected = sum(1 for c in self._connections if c.is_connected())

        return {
            "name": self.name,
            "total_connections": len(self._connections),
            "connected": connected,
            "disconnected": len(self._connections) - connected,
            "connections": [c.get_stats() for c in self._connections],
        }


class NetworkResilience:
    """
    Central network resilience manager.

    Manages multiple connection pools and provides
    unified health monitoring.
    """

    def __init__(self):
        self._pools: Dict[str, ConnectionPool] = {}
        self._connections: Dict[str, Connection] = {}

    def register_pool(self, pool: ConnectionPool) -> None:
        """Register a connection pool."""
        self._pools[pool.name] = pool

    def register_connection(self, name: str, connection: Connection) -> None:
        """Register a single connection."""
        self._connections[name] = connection

    async def start_all(self) -> None:
        """Start all pools and connections."""
        for pool in self._pools.values():
            await pool.start()

        for conn in self._connections.values():
            await conn.connect()

    async def stop_all(self) -> None:
        """Stop all pools and connections."""
        for pool in self._pools.values():
            await pool.stop()

        for conn in self._connections.values():
            await conn.disconnect()

    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """Get a pool by name."""
        return self._pools.get(name)

    def get_connection(self, name: str) -> Optional[Connection]:
        """Get a connection by name."""
        return self._connections.get(name)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get stats for all connections."""
        return {
            "pools": {name: pool.get_stats() for name, pool in self._pools.items()},
            "connections": {name: conn.get_stats() for name, conn in self._connections.items()},
        }


# Global network resilience manager
_resilience: Optional[NetworkResilience] = None


def get_network_resilience() -> NetworkResilience:
    """Get global network resilience manager."""
    global _resilience
    if _resilience is None:
        _resilience = NetworkResilience()
    return _resilience
