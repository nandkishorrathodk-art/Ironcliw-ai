"""
Trinity Orchestration Configuration v108.0 - Unified Configuration Hub
========================================================================

Single source of truth for ALL timeout, threshold, and configuration values
across the Trinity orchestration system.

This module solves the critical root cause: MISMATCHED CONFIGURATION VALUES
spread across multiple files causing cascading failures.

Key Fixes:
1. Unified timeout coordination (startup, health check, heartbeat)
2. Environment-based configurability (no hardcoding)
3. Startup-aware thresholds (grace periods respected)
4. Dynamic timeout adaptation based on component type
5. Consistent values across cross_repo_startup_orchestrator, trinity_health_monitor,
   heartbeat_validator, and brain_orchestrator

Architecture:
    ┌──────────────────────────────────────────────────────────────────────┐
    │           Trinity Orchestration Config v108.0                        │
    ├──────────────────────────────────────────────────────────────────────┤
    │  SINGLE SOURCE OF TRUTH - All timeouts/thresholds configured here    │
    │                                                                      │
    │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐                 │
    │  │ Startup     │ │ Health      │ │ Heartbeat       │                 │
    │  │ Orchestrator│ │ Monitor     │ │ Validator       │                 │
    │  └──────┬──────┘ └──────┬──────┘ └───────┬─────────┘                 │
    │         │               │                 │                          │
    │         └───────────────┴─────────────────┘                          │
    │                         │                                            │
    │              Uses TrinityOrchestrationConfig                         │
    │              (environment-based, no hardcoding)                      │
    └──────────────────────────────────────────────────────────────────────┘

Author: JARVIS Development Team
Version: 108.0.0 (January 2026)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _env_float(key: str, default: float) -> float:
    """Get float from environment with default."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_int(key: str, default: int) -> int:
    """Get int from environment with default."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Get bool from environment with default."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


class ComponentType(str, Enum):
    """Component types with different timeout characteristics."""
    JARVIS_BODY = "jarvis_body"      # Fast startup (10-30s)
    JARVIS_PRIME = "jarvis_prime"    # Slow startup (ML models, 60-300s)
    REACTOR_CORE = "reactor_core"    # Medium startup (30-60s)
    CODING_COUNCIL = "coding_council"  # Fast startup (10-20s)


@dataclass
class ComponentTimeoutProfile:
    """
    Timeout profile for a specific component type.

    This ensures consistency: heartbeat thresholds are always >= startup timeout.
    """
    startup_timeout: float        # Max time to wait for healthy status
    health_check_timeout: float   # Individual HTTP request timeout
    heartbeat_stale: float        # Time before marking heartbeat as stale
    heartbeat_dead: float         # Time before marking component as dead
    startup_grace_period: float   # Extended tolerance during startup
    retry_attempts: int           # Number of health check retries
    retry_delay: float            # Delay between retries

    @property
    def effective_dead_threshold(self) -> float:
        """
        Effective dead threshold accounting for startup grace.

        Critical fix: Dead threshold must ALWAYS be >= startup_timeout
        to prevent components being marked dead while still initializing.
        """
        return max(self.heartbeat_dead, self.startup_timeout * 1.5)


# =============================================================================
# Default Timeout Profiles by Component Type
# =============================================================================

DEFAULT_PROFILES: Dict[ComponentType, ComponentTimeoutProfile] = {
    ComponentType.JARVIS_BODY: ComponentTimeoutProfile(
        startup_timeout=_env_float("JARVIS_BODY_STARTUP_TIMEOUT", 60.0),
        health_check_timeout=_env_float("JARVIS_BODY_HEALTH_TIMEOUT", 10.0),
        heartbeat_stale=_env_float("JARVIS_BODY_HEARTBEAT_STALE", 45.0),
        heartbeat_dead=_env_float("JARVIS_BODY_HEARTBEAT_DEAD", 90.0),
        startup_grace_period=_env_float("JARVIS_BODY_STARTUP_GRACE", 120.0),
        retry_attempts=_env_int("JARVIS_BODY_RETRY_ATTEMPTS", 3),
        retry_delay=_env_float("JARVIS_BODY_RETRY_DELAY", 2.0),
    ),
    ComponentType.JARVIS_PRIME: ComponentTimeoutProfile(
        # J-Prime loads ML models, needs much longer timeouts
        startup_timeout=_env_float("JARVIS_PRIME_STARTUP_TIMEOUT", 300.0),
        health_check_timeout=_env_float("JARVIS_PRIME_HEALTH_TIMEOUT", 15.0),
        heartbeat_stale=_env_float("JARVIS_PRIME_HEARTBEAT_STALE", 120.0),
        heartbeat_dead=_env_float("JARVIS_PRIME_HEARTBEAT_DEAD", 450.0),  # 1.5x startup
        startup_grace_period=_env_float("JARVIS_PRIME_STARTUP_GRACE", 360.0),
        retry_attempts=_env_int("JARVIS_PRIME_RETRY_ATTEMPTS", 5),
        retry_delay=_env_float("JARVIS_PRIME_RETRY_DELAY", 5.0),
    ),
    ComponentType.REACTOR_CORE: ComponentTimeoutProfile(
        startup_timeout=_env_float("REACTOR_CORE_STARTUP_TIMEOUT", 120.0),
        health_check_timeout=_env_float("REACTOR_CORE_HEALTH_TIMEOUT", 10.0),
        heartbeat_stale=_env_float("REACTOR_CORE_HEARTBEAT_STALE", 60.0),
        heartbeat_dead=_env_float("REACTOR_CORE_HEARTBEAT_DEAD", 180.0),
        startup_grace_period=_env_float("REACTOR_CORE_STARTUP_GRACE", 180.0),
        retry_attempts=_env_int("REACTOR_CORE_RETRY_ATTEMPTS", 3),
        retry_delay=_env_float("REACTOR_CORE_RETRY_DELAY", 3.0),
    ),
    ComponentType.CODING_COUNCIL: ComponentTimeoutProfile(
        startup_timeout=_env_float("CODING_COUNCIL_STARTUP_TIMEOUT", 30.0),
        health_check_timeout=_env_float("CODING_COUNCIL_HEALTH_TIMEOUT", 5.0),
        heartbeat_stale=_env_float("CODING_COUNCIL_HEARTBEAT_STALE", 30.0),
        heartbeat_dead=_env_float("CODING_COUNCIL_HEARTBEAT_DEAD", 60.0),
        startup_grace_period=_env_float("CODING_COUNCIL_STARTUP_GRACE", 60.0),
        retry_attempts=_env_int("CODING_COUNCIL_RETRY_ATTEMPTS", 2),
        retry_delay=_env_float("CODING_COUNCIL_RETRY_DELAY", 1.0),
    ),
}


@dataclass
class TrinityOrchestrationConfig:
    """
    Unified configuration for the entire Trinity orchestration system.

    This class is the SINGLE SOURCE OF TRUTH for all orchestration configuration.
    All values are configurable via environment variables.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    trinity_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "trinity"
    )
    heartbeat_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "trinity" / "heartbeats"
    )
    components_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "trinity" / "components"
    )

    # =========================================================================
    # Port Configuration (SINGLE SOURCE OF TRUTH)
    # =========================================================================
    jarvis_body_port: int = field(
        default_factory=lambda: _env_int("JARVIS_BODY_PORT", 8010)
    )
    jarvis_prime_port: int = field(
        default_factory=lambda: _env_int("JARVIS_PRIME_PORT", 8000)
    )
    reactor_core_port: int = field(
        default_factory=lambda: _env_int("REACTOR_CORE_PORT", 8090)
    )

    # Legacy ports to clean up
    legacy_jarvis_prime_ports: tuple = field(
        default_factory=lambda: (8001, 8002, 8003)
    )
    legacy_reactor_core_ports: tuple = field(
        default_factory=lambda: (8003, 8004, 8005)
    )

    # =========================================================================
    # Component Timeout Profiles
    # =========================================================================
    profiles: Dict[ComponentType, ComponentTimeoutProfile] = field(
        default_factory=lambda: DEFAULT_PROFILES.copy()
    )

    # =========================================================================
    # Global Health Check Settings
    # =========================================================================
    # HTTP session settings
    http_connection_limit: int = field(
        default_factory=lambda: _env_int("TRINITY_HTTP_CONNECTION_LIMIT", 100)
    )
    http_connection_limit_per_host: int = field(
        default_factory=lambda: _env_int("TRINITY_HTTP_CONNECTION_PER_HOST", 30)
    )
    http_keepalive_timeout: float = field(
        default_factory=lambda: _env_float("TRINITY_HTTP_KEEPALIVE_TIMEOUT", 60.0)
    )

    # Health check intervals
    health_check_interval: float = field(
        default_factory=lambda: _env_float("TRINITY_HEALTH_CHECK_INTERVAL", 10.0)
    )

    # Consecutive failures before marking unhealthy
    consecutive_failures_threshold: int = field(
        default_factory=lambda: _env_int("TRINITY_CONSECUTIVE_FAILURES_THRESHOLD", 3)
    )

    # =========================================================================
    # Heartbeat Settings (Coordinated with Startup Timeouts)
    # =========================================================================
    heartbeat_publish_interval: float = field(
        default_factory=lambda: _env_float("TRINITY_HEARTBEAT_PUBLISH_INTERVAL", 5.0)
    )
    heartbeat_monitor_interval: float = field(
        default_factory=lambda: _env_float("TRINITY_HEARTBEAT_MONITOR_INTERVAL", 10.0)
    )

    # =========================================================================
    # Process Management
    # =========================================================================
    auto_healing_enabled: bool = field(
        default_factory=lambda: _env_bool("TRINITY_AUTO_HEALING_ENABLED", True)
    )
    max_restart_attempts: int = field(
        default_factory=lambda: _env_int("TRINITY_MAX_RESTART_ATTEMPTS", 5)
    )
    restart_backoff_base: float = field(
        default_factory=lambda: _env_float("TRINITY_RESTART_BACKOFF_BASE", 2.0)
    )
    restart_backoff_max: float = field(
        default_factory=lambda: _env_float("TRINITY_RESTART_BACKOFF_MAX", 60.0)
    )

    # Process output streaming
    stream_process_output: bool = field(
        default_factory=lambda: _env_bool("TRINITY_STREAM_OUTPUT", True)
    )
    capture_stderr_always: bool = field(
        default_factory=lambda: _env_bool("TRINITY_CAPTURE_STDERR", True)
    )

    # Port validation
    port_validation_timeout: float = field(
        default_factory=lambda: _env_float("TRINITY_PORT_VALIDATION_TIMEOUT", 5.0)
    )
    port_cleanup_wait: float = field(
        default_factory=lambda: _env_float("TRINITY_PORT_CLEANUP_WAIT", 2.0)
    )

    # =========================================================================
    # Circuit Breaker Settings
    # =========================================================================
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("TRINITY_CIRCUIT_BREAKER_FAILURES", 5)
    )
    circuit_breaker_success_threshold: int = field(
        default_factory=lambda: _env_int("TRINITY_CIRCUIT_BREAKER_SUCCESSES", 2)
    )
    circuit_breaker_timeout: float = field(
        default_factory=lambda: _env_float("TRINITY_CIRCUIT_BREAKER_TIMEOUT", 30.0)
    )

    # =========================================================================
    # Startup Coordination
    # =========================================================================
    parallel_startup_enabled: bool = field(
        default_factory=lambda: _env_bool("TRINITY_PARALLEL_STARTUP", True)
    )
    startup_coordination_timeout: float = field(
        default_factory=lambda: _env_float("TRINITY_STARTUP_COORDINATION_TIMEOUT", 600.0)
    )

    def __post_init__(self):
        """Ensure directories exist and validate configuration."""
        # Create directories
        self.trinity_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_dir.mkdir(parents=True, exist_ok=True)
        self.components_dir.mkdir(parents=True, exist_ok=True)

        # Validate and log configuration
        self._validate_config()
        logger.info("[TrinityOrchestrationConfig] Initialized with unified configuration")

    def _validate_config(self) -> None:
        """
        Validate configuration consistency.

        Critical validation: heartbeat_dead >= startup_timeout for all components
        """
        for comp_type, profile in self.profiles.items():
            if profile.heartbeat_dead < profile.startup_timeout:
                logger.warning(
                    f"[Config] {comp_type.value}: heartbeat_dead ({profile.heartbeat_dead}s) "
                    f"< startup_timeout ({profile.startup_timeout}s). "
                    f"Using effective threshold: {profile.effective_dead_threshold}s"
                )

            if profile.heartbeat_stale < profile.health_check_timeout * 2:
                logger.warning(
                    f"[Config] {comp_type.value}: heartbeat_stale ({profile.heartbeat_stale}s) "
                    f"is less than 2x health_check_timeout ({profile.health_check_timeout}s). "
                    f"This may cause false stale detection."
                )

    def get_profile(self, component_type: ComponentType) -> ComponentTimeoutProfile:
        """Get timeout profile for a component type."""
        return self.profiles.get(component_type, self.profiles[ComponentType.JARVIS_BODY])

    def get_profile_by_name(self, name: str) -> ComponentTimeoutProfile:
        """Get timeout profile by component name string."""
        name_map = {
            "jarvis": ComponentType.JARVIS_BODY,
            "jarvis_body": ComponentType.JARVIS_BODY,
            "jarvis-body": ComponentType.JARVIS_BODY,
            "jarvis_prime": ComponentType.JARVIS_PRIME,
            "jarvis-prime": ComponentType.JARVIS_PRIME,
            "j-prime": ComponentType.JARVIS_PRIME,
            "jprime": ComponentType.JARVIS_PRIME,
            "reactor_core": ComponentType.REACTOR_CORE,
            "reactor-core": ComponentType.REACTOR_CORE,
            "reactor": ComponentType.REACTOR_CORE,
            "coding_council": ComponentType.CODING_COUNCIL,
            "coding-council": ComponentType.CODING_COUNCIL,
        }
        comp_type = name_map.get(name.lower(), ComponentType.JARVIS_BODY)
        return self.get_profile(comp_type)

    def is_in_startup_grace_period(
        self,
        component_type: ComponentType,
        startup_time: float,
        current_time: Optional[float] = None
    ) -> bool:
        """
        Check if a component is still in its startup grace period.

        During grace period, components should NOT be marked as dead/stale
        even if heartbeats are missing.
        """
        import time
        current = current_time or time.time()
        profile = self.get_profile(component_type)
        elapsed = current - startup_time
        return elapsed < profile.startup_grace_period

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary for logging/debugging."""
        return {
            "ports": {
                "jarvis_body": self.jarvis_body_port,
                "jarvis_prime": self.jarvis_prime_port,
                "reactor_core": self.reactor_core_port,
            },
            "profiles": {
                comp.value: {
                    "startup_timeout": profile.startup_timeout,
                    "health_check_timeout": profile.health_check_timeout,
                    "heartbeat_stale": profile.heartbeat_stale,
                    "heartbeat_dead": profile.heartbeat_dead,
                    "effective_dead_threshold": profile.effective_dead_threshold,
                    "startup_grace_period": profile.startup_grace_period,
                }
                for comp, profile in self.profiles.items()
            },
            "health_check": {
                "interval": self.health_check_interval,
                "consecutive_failures_threshold": self.consecutive_failures_threshold,
            },
            "heartbeat": {
                "publish_interval": self.heartbeat_publish_interval,
                "monitor_interval": self.heartbeat_monitor_interval,
            },
            "process_management": {
                "auto_healing_enabled": self.auto_healing_enabled,
                "max_restart_attempts": self.max_restart_attempts,
                "stream_output": self.stream_process_output,
            },
        }


# =============================================================================
# Global Singleton Instance
# =============================================================================

_global_config: Optional[TrinityOrchestrationConfig] = None


def get_orchestration_config() -> TrinityOrchestrationConfig:
    """
    Get the global orchestration configuration singleton.

    Thread-safe singleton pattern.
    """
    global _global_config
    if _global_config is None:
        _global_config = TrinityOrchestrationConfig()
    return _global_config


def reset_orchestration_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _global_config
    _global_config = None


# =============================================================================
# v113.0: Adaptive Timeout Functions
# =============================================================================

# Minimum healthy providers before allowing EMERGENCY degradation
MIN_HEALTHY_PROVIDERS_BEFORE_EMERGENCY = _env_int("JARVIS_MIN_HEALTHY_PROVIDERS", 1)

# Consecutive failures before triggering emergency mode
EMERGENCY_DEGRADATION_THRESHOLD = _env_int("JARVIS_EMERGENCY_THRESHOLD", 3)

# Startup phase duration (global system startup, not per-component)
GLOBAL_STARTUP_PHASE_DURATION = _env_float("JARVIS_GLOBAL_STARTUP_DURATION", 180.0)


def get_adaptive_timeout(
    component_name: str,
    is_startup: bool = False,
    operation: str = "health_check",
) -> float:
    """
    Get adaptive timeout based on component and current phase.
    
    v113.0: Dynamic timeout calculation that accounts for:
    - Component type (J-Prime needs more time than jarvis-body)
    - Startup phase (more lenient during startup)
    - Operation type (heartbeat vs health check)
    
    Args:
        component_name: Name of the component (e.g., "reactor-core", "jarvis-prime")
        is_startup: Whether system is still in startup phase
        operation: Type of operation ("health_check", "heartbeat", "startup")
        
    Returns:
        Timeout in seconds
    """
    config = get_orchestration_config()
    profile = config.get_profile_by_name(component_name)
    
    # Base timeout by operation type
    if operation == "startup":
        base_timeout = profile.startup_timeout
    elif operation == "heartbeat":
        base_timeout = profile.heartbeat_stale
    else:  # health_check
        base_timeout = profile.health_check_timeout
    
    # Multiplier during startup phase
    if is_startup:
        multiplier = 2.0  # Double timeout during startup
    else:
        multiplier = 1.0
    
    return base_timeout * multiplier


def is_global_startup_phase(system_start_time: float) -> bool:
    """
    Check if the system is still in global startup phase.
    
    During global startup, emergency degradation should be suppressed
    to allow all components time to initialize.
    
    Args:
        system_start_time: Unix timestamp when system started
        
    Returns:
        True if still in startup phase
    """
    import time
    elapsed = time.time() - system_start_time
    return elapsed < GLOBAL_STARTUP_PHASE_DURATION


def should_trigger_emergency_degradation(
    healthy_provider_count: int,
    consecutive_all_unhealthy: int,
    system_start_time: float,
) -> bool:
    """
    Determine if emergency degradation should be triggered.
    
    v113.0: Implements graceful degradation with:
    - Grace period during startup
    - Consecutive failure requirement
    - Minimum healthy provider threshold
    
    Args:
        healthy_provider_count: Number of currently healthy providers
        consecutive_all_unhealthy: How many consecutive checks had 0 healthy
        system_start_time: When the system started
        
    Returns:
        True if emergency degradation should be triggered
    """
    # Never trigger during startup phase
    if is_global_startup_phase(system_start_time):
        logger.debug("[v113.0] Suppressing emergency degradation: still in startup phase")
        return False
    
    # Must have no healthy providers
    if healthy_provider_count > 0:
        return False
    
    # Must have consecutive failures exceeding threshold
    if consecutive_all_unhealthy < EMERGENCY_DEGRADATION_THRESHOLD:
        logger.warning(
            f"[v113.0] All providers unhealthy "
            f"({consecutive_all_unhealthy}/{EMERGENCY_DEGRADATION_THRESHOLD}) - waiting"
        )
        return False
    
    # Now trigger emergency
    logger.error(
        f"[v113.0] EMERGENCY degradation triggered: "
        f"0 healthy providers for {EMERGENCY_DEGRADATION_THRESHOLD} consecutive checks"
    )
    return True


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ComponentType",
    "ComponentTimeoutProfile",
    "TrinityOrchestrationConfig",
    "get_orchestration_config",
    "reset_orchestration_config",
    "get_adaptive_timeout",
    "is_global_startup_phase",
    "should_trigger_emergency_degradation",
    "MIN_HEALTHY_PROVIDERS_BEFORE_EMERGENCY",
    "EMERGENCY_DEGRADATION_THRESHOLD",
    "GLOBAL_STARTUP_PHASE_DURATION",
]

