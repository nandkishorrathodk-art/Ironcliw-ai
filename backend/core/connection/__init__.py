"""
JARVIS Connection Management Package
=====================================

Enterprise-grade connection management with:
- Atomic circuit breaker (CAS pattern)
- Proactive proxy detection (sub-100ms)
- Event-loop-aware async primitives
- Cross-repo coordination support

Author: JARVIS System
Version: 1.0.0
"""

from .state_machine import (
    AtomicStateMachine,
    CircuitState,
    StateTransition,
    StateTransitionError,
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
    # State Machine
    'AtomicStateMachine',
    'CircuitState',
    'StateTransition',
    'StateTransitionError',

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
