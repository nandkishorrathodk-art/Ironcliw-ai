"""
Connection Services Registration
=================================

Registers enterprise connection management services with the DI container.

Services registered:
- ProxyDetectorConfig: Configuration for proxy detection
- ProactiveProxyDetector: Sub-100ms proxy availability detection
- CircuitBreakerConfig: Configuration for circuit breaker
- AtomicCircuitBreaker: Race-condition-safe circuit breaker

These services provide:
- Proactive proxy detection (sub-100ms) vs reactive (2s timeout)
- Atomic circuit breaker with CAS pattern (prevents thundering herd)
- Cross-repo connection coordination support

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("jarvis.di.connection")


def register_connection_services(
    container: Any,
    emitter: Optional[Any] = None,
) -> Dict[str, bool]:
    """
    Register connection management services with the container.

    This enables enterprise-grade connection management with:
    - Sub-100ms proxy detection (proactive, not reactive)
    - Atomic circuit breaker (CAS pattern, no thundering herd)
    - Event emission for observability

    Args:
        container: The ServiceContainer instance
        emitter: Optional EventEmitter for circuit breaker events

    Returns:
        Dict mapping service name to registration success

    Environment Variables:
        JARVIS_PROACTIVE_PROXY_ENABLED: Enable proxy detection (default: true)
        JARVIS_CIRCUIT_BREAKER_ENABLED: Enable circuit breaker (default: true)
        CLOUD_SQL_PROXY_HOST: Proxy host (default: 127.0.0.1)
        CLOUD_SQL_PROXY_PORT: Proxy port (default: 5432)
        CIRCUIT_FAILURE_THRESHOLD: Failures before opening (default: 5)
        CIRCUIT_RECOVERY_TIMEOUT: Recovery timeout in seconds (default: 30)
        CIRCUIT_HALF_OPEN_MAX_REQUESTS: Max test requests (default: 1)
    """
    from backend.core.di.protocols import Scope, ServiceCriticality

    registered: Dict[str, bool] = {}

    # =========================================================================
    # PROACTIVE PROXY DETECTOR
    # =========================================================================
    if os.getenv("JARVIS_PROACTIVE_PROXY_ENABLED", "true").lower() == "true":
        try:
            from backend.core.connection import (
                ProactiveProxyDetector,
                ProxyDetectorConfig,
            )

            # Register proxy detector config (no dependencies)
            container.register(
                ProxyDetectorConfig,
                scope=Scope.SINGLETON,
                criticality=ServiceCriticality.OPTIONAL,
                factory=lambda: ProxyDetectorConfig(),
            )

            # Register proxy detector (depends on config)
            container.register(
                ProactiveProxyDetector,
                scope=Scope.SINGLETON,
                criticality=ServiceCriticality.OPTIONAL,
                factory=lambda: ProactiveProxyDetector(ProxyDetectorConfig()),
            )

            registered["ProxyDetectorConfig"] = True
            registered["ProactiveProxyDetector"] = True
            logger.info(
                "Registered ProactiveProxyDetector "
                f"({ProxyDetectorConfig().proxy_host}:{ProxyDetectorConfig().proxy_port})"
            )

        except ImportError as e:
            logger.debug(f"ProactiveProxyDetector not available: {e}")
            registered["ProactiveProxyDetector"] = False

    # =========================================================================
    # ATOMIC CIRCUIT BREAKER
    # =========================================================================
    if os.getenv("JARVIS_CIRCUIT_BREAKER_ENABLED", "true").lower() == "true":
        try:
            from backend.core.connection import (
                AtomicCircuitBreaker,
                CircuitBreakerConfig,
            )

            # Register circuit breaker config (no dependencies)
            container.register(
                CircuitBreakerConfig,
                scope=Scope.SINGLETON,
                criticality=ServiceCriticality.OPTIONAL,
                factory=lambda: CircuitBreakerConfig(),
            )

            # Register circuit breaker (depends on config)
            # Uses emitter if provided for cross-service event coordination
            def create_circuit_breaker() -> AtomicCircuitBreaker:
                config = CircuitBreakerConfig()
                return AtomicCircuitBreaker(config=config, emitter=emitter)

            container.register(
                AtomicCircuitBreaker,
                scope=Scope.SINGLETON,
                criticality=ServiceCriticality.OPTIONAL,
                factory=create_circuit_breaker,
            )

            registered["CircuitBreakerConfig"] = True
            registered["AtomicCircuitBreaker"] = True

            config = CircuitBreakerConfig()
            logger.info(
                f"Registered AtomicCircuitBreaker "
                f"(threshold={config.failure_threshold}, "
                f"recovery={config.recovery_timeout_seconds}s, "
                f"half_open_max={config.half_open_max_requests})"
            )

        except ImportError as e:
            logger.debug(f"AtomicCircuitBreaker not available: {e}")
            registered["AtomicCircuitBreaker"] = False

    return registered


async def get_connection_health(container: Any) -> Dict[str, Any]:
    """
    Get health status of connection services.

    Args:
        container: The ServiceContainer instance

    Returns:
        Dict with health information for each connection service
    """
    health: Dict[str, Any] = {}

    # Check proxy detector
    try:
        from backend.core.connection import ProactiveProxyDetector, ProxyStatus

        detector = await container.resolve(ProactiveProxyDetector)
        if detector:
            status, msg = await detector.detect()
            health["proxy_detector"] = {
                "available": True,
                "status": status.value,
                "message": msg,
                "is_proxy_available": status == ProxyStatus.AVAILABLE,
            }
        else:
            health["proxy_detector"] = {"available": False, "reason": "not registered"}
    except Exception as e:
        health["proxy_detector"] = {"available": False, "error": str(e)}

    # Check circuit breaker
    try:
        from backend.core.connection import AtomicCircuitBreaker

        breaker = await container.resolve(AtomicCircuitBreaker)
        if breaker:
            state_info = breaker.get_state_info()
            health["circuit_breaker"] = {
                "available": True,
                **state_info,
            }
        else:
            health["circuit_breaker"] = {"available": False, "reason": "not registered"}
    except Exception as e:
        health["circuit_breaker"] = {"available": False, "error": str(e)}

    return health


def get_connection_services_status(registered: Dict[str, bool]) -> str:
    """
    Format connection services status for display.

    Args:
        registered: Dict from register_connection_services

    Returns:
        Formatted status string
    """
    lines = ["Connection Services:"]

    for service_name, success in sorted(registered.items()):
        if success:
            lines.append(f"  ✓ {service_name}: Registered")
        else:
            lines.append(f"  ⚠️ {service_name}: Not available")

    return "\n".join(lines)
