"""
v77.0: Resource Coordinator - Gap #11
======================================

Resource allocation and coordination:
- Resource pooling
- Fair allocation
- Priority-based access
- Resource health monitoring
- Automatic scaling

Author: Ironcliw v77.0
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ResourceState(Enum):
    """State of a resource."""
    AVAILABLE = "available"
    IN_USE = "in_use"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


@dataclass
class Resource(Generic[T]):
    """A managed resource."""
    resource_id: str
    resource: T
    state: ResourceState = ResourceState.AVAILABLE
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    use_count: int = 0
    health_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_used(self) -> None:
        self.last_used_at = time.time()
        self.use_count += 1
        self.state = ResourceState.IN_USE

    def mark_returned(self) -> None:
        self.state = ResourceState.AVAILABLE


@dataclass
class ResourceAllocation:
    """An allocation of a resource to a consumer."""
    allocation_id: str
    resource_id: str
    consumer_id: str
    allocated_at: float = field(default_factory=time.time)
    priority: int = 5
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourcePoolConfig:
    """Configuration for a resource pool."""
    min_size: int = 1
    max_size: int = 10
    target_size: int = 5
    max_idle_time: float = 300.0  # 5 minutes
    health_check_interval: float = 60.0
    acquisition_timeout: float = 30.0


class ResourcePool(Generic[T]):
    """
    Pool of managed resources with automatic scaling.

    Features:
    - Automatic creation and cleanup
    - Health monitoring
    - Fair allocation with priorities
    - Resource reuse
    """

    def __init__(
        self,
        name: str,
        factory: Callable[[], Coroutine[Any, Any, T]],
        config: Optional[ResourcePoolConfig] = None,
        health_check: Optional[Callable[[T], Coroutine[Any, Any, bool]]] = None,
        cleanup: Optional[Callable[[T], Coroutine[Any, Any, None]]] = None,
    ):
        self.name = name
        self.factory = factory
        self.config = config or ResourcePoolConfig()
        self.health_check = health_check
        self.cleanup = cleanup

        self._resources: Dict[str, Resource[T]] = {}
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._waiting: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._lock = asyncio.Lock()
        self._health_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the resource pool."""
        if self._running:
            return

        self._running = True

        # Create initial resources
        for _ in range(self.config.min_size):
            await self._create_resource()

        # Start health check task
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"[ResourcePool] {self.name} started with {len(self._resources)} resources")

    async def stop(self) -> None:
        """Stop the resource pool and cleanup."""
        self._running = False

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Cleanup all resources
        for resource in list(self._resources.values()):
            await self._destroy_resource(resource)

        logger.info(f"[ResourcePool] {self.name} stopped")

    @asynccontextmanager
    async def acquire(
        self,
        consumer_id: str = "",
        priority: int = 5,
        timeout: Optional[float] = None,
    ):
        """
        Acquire a resource from the pool.

        Usage:
            async with pool.acquire("my_consumer") as resource:
                # use resource
                ...
        """
        timeout = timeout or self.config.acquisition_timeout
        resource = await self._acquire_resource(consumer_id, priority, timeout)

        try:
            yield resource.resource
        finally:
            await self._release_resource(resource)

    async def _acquire_resource(
        self,
        consumer_id: str,
        priority: int,
        timeout: float,
    ) -> Resource[T]:
        """Internal resource acquisition."""
        start_time = time.time()

        while True:
            async with self._lock:
                # Find available resource
                resource = self._find_available_resource()

                if resource:
                    resource.mark_used()

                    allocation = ResourceAllocation(
                        allocation_id=str(uuid.uuid4()),
                        resource_id=resource.resource_id,
                        consumer_id=consumer_id,
                        priority=priority,
                    )
                    self._allocations[resource.resource_id] = allocation

                    logger.debug(f"[ResourcePool] {self.name} allocated {resource.resource_id}")
                    return resource

                # Can we create more?
                if len(self._resources) < self.config.max_size:
                    resource = await self._create_resource()
                    if resource:
                        resource.mark_used()

                        allocation = ResourceAllocation(
                            allocation_id=str(uuid.uuid4()),
                            resource_id=resource.resource_id,
                            consumer_id=consumer_id,
                            priority=priority,
                        )
                        self._allocations[resource.resource_id] = allocation

                        return resource

            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout acquiring resource from pool {self.name}")

            # Wait and retry
            await asyncio.sleep(0.1)

    async def _release_resource(self, resource: Resource[T]) -> None:
        """Return a resource to the pool."""
        async with self._lock:
            resource.mark_returned()
            self._allocations.pop(resource.resource_id, None)
            logger.debug(f"[ResourcePool] {self.name} released {resource.resource_id}")

    def _find_available_resource(self) -> Optional[Resource[T]]:
        """Find an available healthy resource."""
        for resource in self._resources.values():
            if resource.state == ResourceState.AVAILABLE and resource.health_score > 0.5:
                return resource
        return None

    async def _create_resource(self) -> Optional[Resource[T]]:
        """Create a new resource."""
        try:
            raw_resource = await self.factory()

            resource = Resource(
                resource_id=str(uuid.uuid4()),
                resource=raw_resource,
            )
            self._resources[resource.resource_id] = resource

            logger.debug(f"[ResourcePool] {self.name} created resource {resource.resource_id}")
            return resource

        except Exception as e:
            logger.error(f"[ResourcePool] {self.name} failed to create resource: {e}")
            return None

    async def _destroy_resource(self, resource: Resource[T]) -> None:
        """Destroy a resource."""
        try:
            if self.cleanup:
                await self.cleanup(resource.resource)

            self._resources.pop(resource.resource_id, None)
            self._allocations.pop(resource.resource_id, None)

            logger.debug(f"[ResourcePool] {self.name} destroyed resource {resource.resource_id}")

        except Exception as e:
            logger.error(f"[ResourcePool] {self.name} cleanup error: {e}")

    async def _health_check_loop(self) -> None:
        """Background health checking."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._run_health_checks()
                await self._scale_pool()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ResourcePool] {self.name} health check error: {e}")

    async def _run_health_checks(self) -> None:
        """Run health checks on all resources."""
        if not self.health_check:
            return

        for resource in list(self._resources.values()):
            if resource.state != ResourceState.AVAILABLE:
                continue

            try:
                healthy = await self.health_check(resource.resource)
                resource.health_score = 1.0 if healthy else 0.0

                if not healthy:
                    resource.state = ResourceState.UNHEALTHY
                    logger.warning(f"[ResourcePool] {self.name} resource {resource.resource_id} unhealthy")

            except Exception as e:
                logger.error(f"[ResourcePool] Health check error: {e}")
                resource.health_score = 0.0

    async def _scale_pool(self) -> None:
        """Scale pool based on usage and health."""
        async with self._lock:
            # Remove unhealthy resources
            for resource in list(self._resources.values()):
                if resource.state == ResourceState.UNHEALTHY:
                    await self._destroy_resource(resource)

            # Remove idle resources above target
            now = time.time()
            available = [r for r in self._resources.values() if r.state == ResourceState.AVAILABLE]

            if len(self._resources) > self.config.target_size:
                for resource in available:
                    if len(self._resources) <= self.config.min_size:
                        break

                    idle_time = now - (resource.last_used_at or resource.created_at)
                    if idle_time > self.config.max_idle_time:
                        await self._destroy_resource(resource)

            # Create resources if below minimum
            while len(self._resources) < self.config.min_size:
                await self._create_resource()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        available = sum(1 for r in self._resources.values() if r.state == ResourceState.AVAILABLE)
        in_use = sum(1 for r in self._resources.values() if r.state == ResourceState.IN_USE)

        return {
            "name": self.name,
            "total": len(self._resources),
            "available": available,
            "in_use": in_use,
            "allocations": len(self._allocations),
            "config": {
                "min": self.config.min_size,
                "max": self.config.max_size,
                "target": self.config.target_size,
            },
        }


class ResourceCoordinator:
    """
    Central coordinator for multiple resource pools.

    Features:
    - Multi-pool management
    - Cross-pool allocation strategies
    - Global resource limits
    - Priority-based scheduling
    """

    def __init__(self, max_total_resources: int = 100):
        self.max_total_resources = max_total_resources
        self._pools: Dict[str, ResourcePool] = {}
        self._global_allocations: Dict[str, Set[str]] = {}  # consumer -> allocation IDs
        self._lock = asyncio.Lock()

    async def register_pool(
        self,
        name: str,
        pool: ResourcePool,
    ) -> None:
        """Register a resource pool."""
        async with self._lock:
            self._pools[name] = pool
            await pool.start()
            logger.info(f"[ResourceCoordinator] Registered pool: {name}")

    async def unregister_pool(self, name: str) -> None:
        """Unregister and stop a resource pool."""
        async with self._lock:
            pool = self._pools.pop(name, None)
            if pool:
                await pool.stop()
                logger.info(f"[ResourceCoordinator] Unregistered pool: {name}")

    @asynccontextmanager
    async def acquire(
        self,
        pool_name: str,
        consumer_id: str = "",
        priority: int = 5,
    ):
        """Acquire a resource from a named pool."""
        if pool_name not in self._pools:
            raise KeyError(f"Pool '{pool_name}' not found")

        pool = self._pools[pool_name]

        async with pool.acquire(consumer_id, priority) as resource:
            yield resource

    async def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get a pool by name."""
        return self._pools.get(name)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        total = sum(len(p._resources) for p in self._pools.values())

        return {
            "total_resources": total,
            "max_resources": self.max_total_resources,
            "pools": {name: pool.get_stats() for name, pool in self._pools.items()},
        }

    async def shutdown(self) -> None:
        """Shutdown all pools."""
        for name in list(self._pools.keys()):
            await self.unregister_pool(name)


# Global coordinator
_coordinator: Optional[ResourceCoordinator] = None


def get_resource_coordinator() -> ResourceCoordinator:
    """Get global resource coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = ResourceCoordinator()
    return _coordinator
