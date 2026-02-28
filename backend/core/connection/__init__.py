"""
Ironcliw Connection Management Package
=====================================

Enterprise-grade connection management with:
- Generic finite state machine (FSM) base class
- Atomic circuit breaker (CAS pattern)
- Proactive proxy detection (sub-100ms)
- Event-loop-aware async primitives
- Cross-repo coordination support
- Factory functions for common FSM patterns

Author: Ironcliw System
Version: 2.0.0
"""

from .state_machine import (
    # Base FSM class
    FiniteStateMachine,
    # Legacy circuit breaker state machine
    AtomicStateMachine,
    # State enums
    CircuitState,
    ConnectionState,
    # Data classes
    StateTransition,
    # Exceptions
    StateTransitionError,
    # Factory functions
    create_lifecycle_fsm,
    create_connection_fsm,
    # Callback types
    TransitionCallback,
    StateCallback,
)

from .circuit_breaker import (
    AtomicCircuitBreaker,
    CircuitBreakerConfig,
)

from .proactive_proxy_detector import (
    ProactiveProxyDetector,
    ProxyDetectorConfig,
    ProxyStatus,
)

from .async_primitives import (
    EventLoopAwareLock,
)

__all__ = [
    # Finite State Machine (base)
    'FiniteStateMachine',

    # State Machine (legacy / circuit breaker)
    'AtomicStateMachine',
    'CircuitState',
    'StateTransition',
    'StateTransitionError',

    # Connection State
    'ConnectionState',

    # Factory Functions
    'create_lifecycle_fsm',
    'create_connection_fsm',

    # Callback Types
    'TransitionCallback',
    'StateCallback',

    # Circuit Breaker
    'AtomicCircuitBreaker',
    'CircuitBreakerConfig',

    # Proxy Detection
    'ProactiveProxyDetector',
    'ProxyDetectorConfig',
    'ProxyStatus',

    # Primitives
    'EventLoopAwareLock',
]
