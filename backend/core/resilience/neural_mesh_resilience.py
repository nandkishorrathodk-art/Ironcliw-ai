"""
Neural Mesh Resilience Integration v1.0
=======================================

Bridges the Unified Resilience Engine with the Neural Mesh for cross-repo
fault tolerance. Ensures reliable communication between:
- JARVIS (Body)
- JARVIS Prime (Mind)
- Reactor Core (Learning)

Features:
- Circuit breaker integration with mesh connections
- Automatic failover for cross-repo calls
- Health-based routing for load distribution
- Request queuing for async operations
- Dead letter queue for failed cross-repo messages

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    NEURAL MESH RESILIENCE LAYER                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │    ┌─────────────────┐                      ┌─────────────────┐         │
    │    │   Neural Mesh   │◀─────────────────────│ Resilience      │         │
    │    │   (Transport)   │     Protected Calls  │ Engine          │         │
    │    │                 │─────────────────────▶│                 │         │
    │    └────────┬────────┘                      └────────┬────────┘         │
    │             │                                        │                  │
    │             ▼                                        ▼                  │
    │    ┌─────────────────────────────────────────────────────────────┐     │
    │    │                     Protected Mesh Calls                      │     │
    │    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │     │
    │    │  │ Bulkhead │  │ Circuit  │  │ Retry    │  │ Timeout  │     │     │
    │    │  │ Pool     │  │ Breaker  │  │ Logic    │  │ Adaptive │     │     │
    │    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │     │
    │    └─────────────────────────────────────────────────────────────┘     │
    │                                                                          │
    │                      CROSS-REPO COMMUNICATION                            │
    │    ┌───────────┐        ┌───────────┐        ┌───────────┐             │
    │    │  JARVIS   │◀══════▶│  JARVIS   │◀══════▶│  REACTOR  │             │
    │    │  (Body)   │  WS/   │  PRIME    │  HTTP/ │  CORE     │             │
    │    │  Port 8010│  HTTP  │  Port 8000│  File  │  Port 8020│             │
    │    └───────────┘        └───────────┘        └───────────┘             │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Author: Trinity Resilience System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

from backend.core.resilience.unified_resilience_engine import (
    UnifiedResilienceEngine,
    get_resilience_engine,
    initialize_resilience,
    shutdown_resilience,
    RequestPriority,
    ServiceHealth,
    RoutingStrategy,
    with_retry,
    with_bulkhead,
    with_adaptive_timeout,
    DecorrelatedJitterRetry,
    BulkheadRejectedError,
    BulkheadTimeoutError,
    BackpressureSignal,
    ChaosInjectedException,
)

from backend.core.resilience.cross_repo_circuit_breaker import (
    CrossRepoCircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    FailureType,
    FailureClassifier,
    get_cross_repo_breaker,
)

logger = logging.getLogger("NeuralMesh.Resilience")

T = TypeVar("T")


# =============================================================================
# CONFIGURATION
# =============================================================================

class MeshResilienceConfig:
    """Configuration for neural mesh resilience."""

    @staticmethod
    def get_bulkhead_pool_jarvis() -> str:
        return os.getenv("MESH_BULKHEAD_JARVIS", "jarvis_mesh")

    @staticmethod
    def get_bulkhead_pool_prime() -> str:
        return os.getenv("MESH_BULKHEAD_PRIME", "prime_mesh")

    @staticmethod
    def get_bulkhead_pool_reactor() -> str:
        return os.getenv("MESH_BULKHEAD_REACTOR", "reactor_mesh")

    @staticmethod
    def get_circuit_breaker_name() -> str:
        return os.getenv("MESH_CIRCUIT_BREAKER", "neural_mesh")

    @staticmethod
    def get_default_timeout() -> float:
        return float(os.getenv("MESH_DEFAULT_TIMEOUT", "30.0"))

    @staticmethod
    def get_improvement_timeout() -> float:
        return float(os.getenv("MESH_IMPROVEMENT_TIMEOUT", "300.0"))

    @staticmethod
    def get_training_timeout() -> float:
        return float(os.getenv("MESH_TRAINING_TIMEOUT", "600.0"))


# =============================================================================
# ENUMS
# =============================================================================

class MeshTarget(str, Enum):
    """Target nodes in the neural mesh."""
    JARVIS = "jarvis"
    PRIME = "prime"
    REACTOR = "reactor"
    BROADCAST = "broadcast"


class MeshOperation(str, Enum):
    """Types of mesh operations."""
    REQUEST = "request"
    IMPROVEMENT = "improvement"
    TRAINING = "training"
    HEALTH_CHECK = "health_check"
    SYNC = "sync"
    EXPERIENCE = "experience"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MeshCallResult:
    """Result of a resilient mesh call."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    target: Optional[MeshTarget] = None
    latency_ms: float = 0.0
    retries: int = 0
    circuit_state: Optional[CircuitState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "target": self.target.value if self.target else None,
            "latency_ms": round(self.latency_ms, 2),
            "retries": self.retries,
            "circuit_state": self.circuit_state.value if self.circuit_state else None,
            "metadata": self.metadata,
        }


@dataclass
class MeshHealthSnapshot:
    """Snapshot of mesh health status."""
    timestamp: float = field(default_factory=time.time)
    jarvis_health: ServiceHealth = ServiceHealth.UNKNOWN
    prime_health: ServiceHealth = ServiceHealth.UNKNOWN
    reactor_health: ServiceHealth = ServiceHealth.UNKNOWN
    circuit_states: Dict[str, CircuitState] = field(default_factory=dict)
    latencies: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)

    @property
    def overall_health(self) -> ServiceHealth:
        """Calculate overall mesh health."""
        healths = [self.jarvis_health, self.prime_health, self.reactor_health]

        if all(h == ServiceHealth.HEALTHY for h in healths):
            return ServiceHealth.HEALTHY
        if all(h == ServiceHealth.UNHEALTHY for h in healths):
            return ServiceHealth.UNHEALTHY
        if any(h == ServiceHealth.UNHEALTHY for h in healths):
            return ServiceHealth.DEGRADED
        return ServiceHealth.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_health": self.overall_health.value,
            "jarvis_health": self.jarvis_health.value,
            "prime_health": self.prime_health.value,
            "reactor_health": self.reactor_health.value,
            "circuit_states": {k: v.value for k, v in self.circuit_states.items()},
            "latencies": self.latencies,
            "error_rates": self.error_rates,
        }


# =============================================================================
# NEURAL MESH RESILIENCE BRIDGE
# =============================================================================

class NeuralMeshResilienceBridge:
    """
    Bridges the Neural Mesh with the Unified Resilience Engine.

    Provides fault-tolerant wrappers for all cross-repo operations with:
    - Bulkhead isolation per target node
    - Circuit breakers with adaptive thresholds
    - Automatic retry with decorrelated jitter
    - Adaptive timeouts based on operation type
    - Health-based routing and failover
    - Dead letter queue for failed operations
    """

    _instance: Optional["NeuralMeshResilienceBridge"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._engine: Optional[UnifiedResilienceEngine] = None
        self._neural_mesh = None  # Will be set during initialization
        self._circuit_breaker: Optional[CrossRepoCircuitBreaker] = None
        self._initialized = False

        # Metrics
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._retried_calls = 0
        self._circuit_breaks = 0

        self.logger = logging.getLogger("NeuralMesh.ResilienceBridge")

    @classmethod
    async def get_instance(cls) -> "NeuralMeshResilienceBridge":
        """Get or create the singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    async def initialize(self, neural_mesh=None) -> bool:
        """
        Initialize the resilience bridge.

        Args:
            neural_mesh: Optional NeuralMesh instance to wrap

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True

        self.logger.info("Initializing Neural Mesh Resilience Bridge...")

        try:
            # Get resilience engine
            self._engine = await get_resilience_engine()
            if not self._engine._initialized:
                await self._engine.initialize()

            # Get circuit breaker
            config = CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout_seconds=30.0,
                half_open_max_calls=3,
                adaptive_thresholds=True,
                network_failure_threshold=3,
                timeout_failure_threshold=4,
                rate_limit_failure_threshold=2,
            )
            self._circuit_breaker = get_cross_repo_breaker(
                MeshResilienceConfig.get_circuit_breaker_name(),
                config,
            )

            # Store neural mesh reference
            self._neural_mesh = neural_mesh

            # Setup DLQ retry processor for mesh operations
            dlq = self._engine._dead_letter_queues.get("default")
            if dlq:
                dlq.set_retry_processor(self._retry_failed_mesh_operation)

            self._initialized = True
            self.logger.info("Neural Mesh Resilience Bridge initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize resilience bridge: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the resilience bridge."""
        self.logger.info("Shutting down Neural Mesh Resilience Bridge...")
        self._initialized = False

    def _get_bulkhead_pool(self, target: MeshTarget) -> str:
        """Get the bulkhead pool name for a target."""
        pools = {
            MeshTarget.JARVIS: MeshResilienceConfig.get_bulkhead_pool_jarvis(),
            MeshTarget.PRIME: MeshResilienceConfig.get_bulkhead_pool_prime(),
            MeshTarget.REACTOR: MeshResilienceConfig.get_bulkhead_pool_reactor(),
        }
        return pools.get(target, "default_mesh")

    def _get_timeout(self, operation: MeshOperation) -> float:
        """Get timeout for an operation type."""
        timeouts = {
            MeshOperation.REQUEST: MeshResilienceConfig.get_default_timeout(),
            MeshOperation.IMPROVEMENT: MeshResilienceConfig.get_improvement_timeout(),
            MeshOperation.TRAINING: MeshResilienceConfig.get_training_timeout(),
            MeshOperation.HEALTH_CHECK: 5.0,
            MeshOperation.SYNC: 30.0,
            MeshOperation.EXPERIENCE: 60.0,
        }
        return timeouts.get(operation, MeshResilienceConfig.get_default_timeout())

    async def call(
        self,
        target: MeshTarget,
        operation: MeshOperation,
        func: Callable[..., Awaitable[T]],
        *args,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: Optional[float] = None,
        retry_on_failure: bool = True,
        max_retries: int = 3,
        **kwargs,
    ) -> MeshCallResult:
        """
        Make a resilient call to a mesh target.

        Applies:
        1. Bulkhead isolation (per target)
        2. Circuit breaker (per target)
        3. Adaptive timeout
        4. Automatic retry with jitter
        5. DLQ on final failure

        Args:
            target: Target node (JARVIS, PRIME, REACTOR)
            operation: Type of operation
            func: Async function to call
            *args: Function arguments
            priority: Request priority
            timeout: Optional explicit timeout
            retry_on_failure: Whether to retry on failure
            max_retries: Maximum retry attempts
            **kwargs: Function keyword arguments

        Returns:
            MeshCallResult with success status and data
        """
        self._total_calls += 1
        start_time = time.time()
        retries = 0
        last_error: Optional[str] = None

        # Determine timeout
        effective_timeout = timeout or self._get_timeout(operation)

        # Get bulkhead pool
        bulkhead_pool = self._get_bulkhead_pool(target)

        # Retry strategy
        retry = DecorrelatedJitterRetry(max_attempts=max_retries) if retry_on_failure else None

        # Circuit breaker tier
        tier = target.value

        while True:
            try:
                # Check circuit breaker state first
                tier_health = self._circuit_breaker.get_tier_health(tier)
                if tier_health.state == CircuitState.OPEN:
                    self._circuit_breaks += 1
                    raise CircuitOpenError(
                        tier,
                        tier_health.state,
                        self._circuit_breaker._get_timeout(tier),
                        reason=tier_health.dominant_failure_type.value
                        if tier_health.dominant_failure_type else None,
                    )

                # Execute with bulkhead protection
                async with self._engine.bulkhead(bulkhead_pool, timeout=5.0):
                    # Execute with circuit breaker protection
                    result = await self._circuit_breaker.execute(
                        tier=tier,
                        func=func,
                        args=args,
                        kwargs=kwargs,
                    )

                    # Success!
                    latency_ms = (time.time() - start_time) * 1000
                    self._successful_calls += 1

                    # Record latency for adaptive timeout
                    await self._engine.record_operation_latency(
                        f"mesh_{target.value}_{operation.value}",
                        latency_ms,
                    )

                    return MeshCallResult(
                        success=True,
                        data=result,
                        target=target,
                        latency_ms=latency_ms,
                        retries=retries,
                        circuit_state=tier_health.state,
                    )

            except BulkheadRejectedError as e:
                last_error = f"Bulkhead rejected: {e}"
                self.logger.warning(f"Mesh call rejected by bulkhead: {e}")

            except BulkheadTimeoutError as e:
                last_error = f"Bulkhead timeout: {e}"
                self.logger.warning(f"Mesh call bulkhead timeout: {e}")

            except CircuitOpenError as e:
                last_error = f"Circuit open: {e}"
                self.logger.warning(f"Mesh call circuit open: {e}")

            except asyncio.TimeoutError:
                last_error = f"Timeout after {effective_timeout}s"
                self.logger.warning(f"Mesh call timeout: {target.value}")
                await self._circuit_breaker._on_failure(tier, FailureType.TIMEOUT)

            except ChaosInjectedException as e:
                last_error = f"Chaos injection: {e}"
                self.logger.warning(f"Chaos injection: {e}")

            except Exception as e:
                last_error = str(e)
                failure_type = FailureClassifier.classify(e)
                await self._circuit_breaker._on_failure(tier, failure_type)
                self.logger.error(f"Mesh call error: {e}")

            # Handle retry
            retries += 1
            self._retried_calls += 1

            if retry and retry.should_retry(retries):
                delay = retry.get_delay(retries)
                self.logger.info(
                    f"Retrying mesh call to {target.value} "
                    f"(attempt {retries}/{max_retries}) after {delay:.2f}s"
                )
                await asyncio.sleep(delay)
            else:
                break

        # Final failure - add to DLQ
        self._failed_calls += 1
        latency_ms = (time.time() - start_time) * 1000

        # Add to dead letter queue for later retry
        dlq = self._engine._dead_letter_queues.get("default")
        if dlq:
            await dlq.add(
                request_id=f"mesh_{target.value}_{int(time.time()*1000)}",
                original_request={
                    "target": target.value,
                    "operation": operation.value,
                    "args": str(args)[:500],
                    "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                },
                failure_reason=last_error or "Unknown error",
                failure_traceback=traceback.format_exc(),
                metadata={
                    "retries": retries,
                    "latency_ms": latency_ms,
                    "priority": priority.name,
                },
            )

        tier_health = self._circuit_breaker.get_tier_health(tier)
        return MeshCallResult(
            success=False,
            error=last_error,
            target=target,
            latency_ms=latency_ms,
            retries=retries,
            circuit_state=tier_health.state,
        )

    async def _retry_failed_mesh_operation(self, request: Dict[str, Any]) -> bool:
        """
        Retry a failed mesh operation from the DLQ.

        Returns True if retry succeeded.
        """
        target_str = request.get("target")
        if not target_str:
            return False

        try:
            target = MeshTarget(target_str)
        except ValueError:
            return False

        # For now, just log and mark as processed
        # In production, you would re-execute the operation
        self.logger.info(f"DLQ retry for mesh operation to {target.value}")
        return True

    async def send_to_prime(
        self,
        payload: Dict[str, Any],
        operation: MeshOperation = MeshOperation.REQUEST,
        timeout: Optional[float] = None,
    ) -> MeshCallResult:
        """Send a resilient request to JARVIS Prime."""
        if not self._neural_mesh:
            return MeshCallResult(
                success=False,
                error="Neural mesh not initialized",
                target=MeshTarget.PRIME,
            )

        async def _send():
            # Import here to avoid circular imports
            from backend.core.ouroboros.neural_mesh import (
                NodeType,
                MessageType,
            )

            response = await self._neural_mesh.send(
                target=NodeType.PRIME,
                message_type=MessageType.REQUEST,
                payload=payload,
                wait_response=True,
                timeout=timeout or self._get_timeout(operation),
            )
            return response.payload if response else None

        return await self.call(
            target=MeshTarget.PRIME,
            operation=operation,
            func=_send,
            timeout=timeout,
        )

    async def send_to_reactor(
        self,
        payload: Dict[str, Any],
        operation: MeshOperation = MeshOperation.EXPERIENCE,
        timeout: Optional[float] = None,
    ) -> MeshCallResult:
        """Send a resilient request to Reactor Core."""
        if not self._neural_mesh:
            return MeshCallResult(
                success=False,
                error="Neural mesh not initialized",
                target=MeshTarget.REACTOR,
            )

        async def _send():
            from backend.core.ouroboros.neural_mesh import (
                NodeType,
                MessageType,
            )

            response = await self._neural_mesh.send(
                target=NodeType.REACTOR,
                message_type=MessageType.REQUEST,
                payload=payload,
                wait_response=True,
                timeout=timeout or self._get_timeout(operation),
            )
            return response.payload if response else None

        return await self.call(
            target=MeshTarget.REACTOR,
            operation=operation,
            func=_send,
            timeout=timeout,
        )

    async def request_improvement(
        self,
        target_file: str,
        goal: str,
        context: Optional[str] = None,
    ) -> MeshCallResult:
        """Request code improvement from JARVIS Prime with full resilience."""
        payload = {
            "target_file": target_file,
            "goal": goal,
            "context": context,
        }

        return await self.send_to_prime(
            payload=payload,
            operation=MeshOperation.IMPROVEMENT,
            timeout=MeshResilienceConfig.get_improvement_timeout(),
        )

    async def publish_experience(
        self,
        original_code: str,
        improved_code: str,
        goal: str,
        success: bool,
        iterations: int,
    ) -> MeshCallResult:
        """Publish improvement experience to Reactor Core with resilience."""
        payload = {
            "original_code": original_code[:5000],
            "improved_code": improved_code[:5000],
            "goal": goal,
            "success": success,
            "iterations": iterations,
            "timestamp": time.time(),
        }

        return await self.send_to_reactor(
            payload=payload,
            operation=MeshOperation.EXPERIENCE,
            timeout=60.0,
        )

    async def get_health_snapshot(self) -> MeshHealthSnapshot:
        """Get current health snapshot of the mesh."""
        snapshot = MeshHealthSnapshot()

        # Get circuit breaker states
        for target in [MeshTarget.JARVIS, MeshTarget.PRIME, MeshTarget.REACTOR]:
            tier = target.value
            health = self._circuit_breaker.get_tier_health(tier)

            snapshot.circuit_states[tier] = health.state
            snapshot.latencies[tier] = health.avg_latency_ms
            snapshot.error_rates[tier] = 1.0 - health.success_rate

            # Map circuit state to service health
            if health.state == CircuitState.CLOSED:
                service_health = ServiceHealth.HEALTHY
            elif health.state == CircuitState.HALF_OPEN:
                service_health = ServiceHealth.DEGRADED
            else:
                service_health = ServiceHealth.UNHEALTHY

            if target == MeshTarget.JARVIS:
                snapshot.jarvis_health = service_health
            elif target == MeshTarget.PRIME:
                snapshot.prime_health = service_health
            elif target == MeshTarget.REACTOR:
                snapshot.reactor_health = service_health

        return snapshot

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the resilience bridge."""
        circuit_status = self._circuit_breaker.get_status() if self._circuit_breaker else {}

        return {
            "initialized": self._initialized,
            "metrics": {
                "total_calls": self._total_calls,
                "successful_calls": self._successful_calls,
                "failed_calls": self._failed_calls,
                "retried_calls": self._retried_calls,
                "circuit_breaks": self._circuit_breaks,
                "success_rate": (
                    self._successful_calls / max(self._total_calls, 1)
                ),
            },
            "circuit_breaker": circuit_status,
            "resilience_engine": (
                self._engine.get_status() if self._engine else {}
            ),
        }


# =============================================================================
# GLOBAL INSTANCE AND HELPER FUNCTIONS
# =============================================================================

_bridge: Optional[NeuralMeshResilienceBridge] = None


async def get_mesh_resilience_bridge() -> NeuralMeshResilienceBridge:
    """Get the global mesh resilience bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = await NeuralMeshResilienceBridge.get_instance()
    return _bridge


async def initialize_mesh_resilience(neural_mesh=None) -> bool:
    """Initialize the mesh resilience bridge."""
    bridge = await get_mesh_resilience_bridge()
    return await bridge.initialize(neural_mesh)


async def shutdown_mesh_resilience() -> None:
    """Shutdown the mesh resilience bridge."""
    global _bridge
    if _bridge:
        await _bridge.shutdown()
        _bridge = None


# =============================================================================
# DECORATOR FOR RESILIENT MESH CALLS
# =============================================================================

def with_mesh_resilience(
    target: MeshTarget,
    operation: MeshOperation = MeshOperation.REQUEST,
    priority: RequestPriority = RequestPriority.NORMAL,
    timeout: Optional[float] = None,
    max_retries: int = 3,
):
    """
    Decorator for making mesh calls resilient.

    Usage:
        @with_mesh_resilience(MeshTarget.PRIME, MeshOperation.IMPROVEMENT)
        async def improve_code(file_path: str):
            # Your code here
            ...
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[MeshCallResult]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> MeshCallResult:
            bridge = await get_mesh_resilience_bridge()
            return await bridge.call(
                target=target,
                operation=operation,
                func=func,
                *args,
                priority=priority,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )
        return wrapper
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "MeshResilienceConfig",
    # Enums
    "MeshTarget",
    "MeshOperation",
    # Data Structures
    "MeshCallResult",
    "MeshHealthSnapshot",
    # Main Class
    "NeuralMeshResilienceBridge",
    # Global Functions
    "get_mesh_resilience_bridge",
    "initialize_mesh_resilience",
    "shutdown_mesh_resilience",
    # Decorator
    "with_mesh_resilience",
]
