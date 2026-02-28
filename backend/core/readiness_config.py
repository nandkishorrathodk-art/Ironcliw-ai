"""
Ironcliw Unified Readiness Configuration Module
==============================================

Central configuration for component criticality and status display semantics.

This module provides:
1. ComponentCriticality enum - CRITICAL, OPTIONAL, UNKNOWN
2. ComponentStatus enum - All possible component lifecycle states
3. STATUS_DISPLAY_MAP - Maps status strings to 4-char display codes
4. DASHBOARD_STATUS_MAP - Maps internal status to dashboard status
5. ReadinessConfig dataclass - Central configuration with env var support
6. get_readiness_config() - Singleton access to configuration

CRITICAL FIX: "skipped" displays as "SKIP" (NOT "STOP")
This fixes the bug where skipped components appeared the same as stopped components.

Usage:
    from backend.core.readiness_config import (
        ReadinessConfig,
        ComponentCriticality,
        get_readiness_config,
    )

    config = get_readiness_config()

    # Check component criticality
    if config.get_criticality("backend") == ComponentCriticality.CRITICAL:
        # Handle critical component
        pass

    # Get display code for status
    display = config.status_to_display("skipped")  # Returns "SKIP"

    # Get dashboard status
    dashboard = config.status_to_dashboard("skipped")  # Returns "skipped"

Author: Ironcliw Trinity - Readiness Integrity Fixes
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Optional


# =============================================================================
# Enums
# =============================================================================

class ComponentCriticality(Enum):
    """
    Classification of component importance for system readiness.

    CRITICAL: Component must be healthy for system to be considered ready.
              Failures block readiness certification.

    OPTIONAL: Component enhances functionality but system can operate without.
              Failures do not block readiness certification.

    UNKNOWN:  Component is not recognized. Default handling applies.
    """
    CRITICAL = "critical"
    OPTIONAL = "optional"
    UNKNOWN = "unknown"


class ComponentStatus(Enum):
    """
    Possible lifecycle states for a component.

    State transitions:
        PENDING -> STARTING -> HEALTHY (success)
                            -> DEGRADED (partial success)
                            -> ERROR (unrecoverable)
                            -> STOPPED (intentionally stopped)
                            -> SKIPPED (intentionally not started)
                            -> UNAVAILABLE (temporarily unavailable)

    CRITICAL: SKIPPED and STOPPED are DISTINCT states:
    - STOPPED: Component was running but was intentionally stopped
    - SKIPPED: Component was intentionally never started (e.g., optional in dev mode)
    """
    PENDING = "pending"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


# =============================================================================
# Status Display Mappings
# =============================================================================

# Maps status strings to 4-character display codes for dashboard/UI
# CRITICAL: "skipped" -> "SKIP" (NOT "STOP")
STATUS_DISPLAY_MAP: Dict[str, str] = {
    "pending": "PEND",
    "starting": "STAR",
    "healthy": "HEAL",
    "degraded": "DEGR",
    "error": "EROR",
    "stopped": "STOP",
    "skipped": "SKIP",  # CRITICAL: Must be distinct from "stopped"
    "unavailable": "UNAV",
}

# Maps internal status to dashboard-friendly status strings
# CRITICAL: "skipped" -> "skipped" (NOT "stopped")
DASHBOARD_STATUS_MAP: Dict[str, str] = {
    "pending": "pending",
    "starting": "starting",
    "healthy": "healthy",
    "degraded": "degraded",
    "error": "error",
    "stopped": "stopped",
    "skipped": "skipped",  # CRITICAL: Must be distinct from "stopped"
    "unavailable": "unavailable",
}


# =============================================================================
# Configuration Defaults
# =============================================================================

# Default critical components - must be healthy for system readiness
DEFAULT_CRITICAL_COMPONENTS: FrozenSet[str] = frozenset({
    "backend",
    "loading_server",
    "preflight",
})

# Default optional components - enhance functionality but not required
DEFAULT_OPTIONAL_COMPONENTS: FrozenSet[str] = frozenset({
    "jarvis_prime",
    "reactor_core",
    "enterprise",
    "agi_os",
    "gcp_vm",
})

# Default timeout values
DEFAULT_VERIFICATION_TIMEOUT = 60.0  # seconds
DEFAULT_UNHEALTHY_THRESHOLD_FAILURES = 3  # consecutive failures
DEFAULT_UNHEALTHY_THRESHOLD_SECONDS = 30.0  # seconds
DEFAULT_REVOCATION_COOLDOWN_SECONDS = 5.0  # seconds


# =============================================================================
# ReadinessConfig Dataclass
# =============================================================================

@dataclass(frozen=False)
class ReadinessConfig:
    """
    Central configuration for readiness behavior.

    Provides single source of truth for:
    - Component criticality classification
    - Status display mappings
    - Timeout and threshold values

    Configuration can be overridden via environment variables:
    - Ironcliw_VERIFICATION_TIMEOUT: Verification timeout in seconds
    - Ironcliw_UNHEALTHY_THRESHOLD_FAILURES: Consecutive failures before unhealthy
    - Ironcliw_UNHEALTHY_THRESHOLD_SECONDS: Seconds before unhealthy
    - Ironcliw_REVOCATION_COOLDOWN_SECONDS: Cooldown between revocations
    """

    # Component classification
    critical_components: FrozenSet[str] = field(
        default_factory=lambda: DEFAULT_CRITICAL_COMPONENTS
    )
    optional_components: FrozenSet[str] = field(
        default_factory=lambda: DEFAULT_OPTIONAL_COMPONENTS
    )

    # Timeout and threshold values (populated from env vars in __post_init__)
    verification_timeout: float = field(default=DEFAULT_VERIFICATION_TIMEOUT)
    unhealthy_threshold_failures: int = field(default=DEFAULT_UNHEALTHY_THRESHOLD_FAILURES)
    unhealthy_threshold_seconds: float = field(default=DEFAULT_UNHEALTHY_THRESHOLD_SECONDS)
    revocation_cooldown_seconds: float = field(default=DEFAULT_REVOCATION_COOLDOWN_SECONDS)

    def __post_init__(self) -> None:
        """Load configuration from environment variables."""
        # Override with environment variables if set
        if env_timeout := os.environ.get("Ironcliw_VERIFICATION_TIMEOUT"):
            object.__setattr__(self, "verification_timeout", float(env_timeout))

        if env_failures := os.environ.get("Ironcliw_UNHEALTHY_THRESHOLD_FAILURES"):
            object.__setattr__(self, "unhealthy_threshold_failures", int(env_failures))

        if env_seconds := os.environ.get("Ironcliw_UNHEALTHY_THRESHOLD_SECONDS"):
            object.__setattr__(self, "unhealthy_threshold_seconds", float(env_seconds))

        if env_cooldown := os.environ.get("Ironcliw_REVOCATION_COOLDOWN_SECONDS"):
            object.__setattr__(self, "revocation_cooldown_seconds", float(env_cooldown))

    def get_criticality(self, component_name: str) -> ComponentCriticality:
        """
        Get the criticality classification for a component.

        Args:
            component_name: Name of the component (case-insensitive)

        Returns:
            ComponentCriticality enum value
        """
        name_lower = component_name.lower()

        # Check if empty
        if not name_lower:
            return ComponentCriticality.UNKNOWN

        # Check critical first
        if name_lower in {c.lower() for c in self.critical_components}:
            return ComponentCriticality.CRITICAL

        # Check optional
        if name_lower in {c.lower() for c in self.optional_components}:
            return ComponentCriticality.OPTIONAL

        return ComponentCriticality.UNKNOWN

    @staticmethod
    def status_to_display(status: str) -> str:
        """
        Convert status string to 4-character display code.

        Args:
            status: Status string (case-insensitive)

        Returns:
            4-character display code, or "????" for unknown status

        Note: CRITICAL - "skipped" returns "SKIP", not "STOP"
        """
        status_lower = status.lower()
        return STATUS_DISPLAY_MAP.get(status_lower, "????")

    @staticmethod
    def status_to_dashboard(status: str) -> str:
        """
        Convert status string to dashboard-friendly status.

        Args:
            status: Status string (case-insensitive)

        Returns:
            Dashboard status string, or "unknown" for unknown status

        Note: CRITICAL - "skipped" returns "skipped", not "stopped"
        """
        status_lower = status.lower()
        return DASHBOARD_STATUS_MAP.get(status_lower, "unknown")


# =============================================================================
# Singleton Access
# =============================================================================

_config_instance: Optional[ReadinessConfig] = None


def get_readiness_config() -> ReadinessConfig:
    """
    Get the singleton ReadinessConfig instance.

    Returns:
        ReadinessConfig instance (singleton)

    Example:
        config = get_readiness_config()
        if config.get_criticality("backend") == ComponentCriticality.CRITICAL:
            # Handle critical component
            pass
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ReadinessConfig()
    return _config_instance


def _reset_config() -> None:
    """
    Reset the singleton instance (for testing purposes).

    This allows tests to verify that the singleton pattern works correctly
    and to reset state between tests.
    """
    global _config_instance
    _config_instance = None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "ComponentCriticality",
    "ComponentStatus",
    # Constants
    "STATUS_DISPLAY_MAP",
    "DASHBOARD_STATUS_MAP",
    "DEFAULT_CRITICAL_COMPONENTS",
    "DEFAULT_OPTIONAL_COMPONENTS",
    # Dataclass
    "ReadinessConfig",
    # Functions
    "get_readiness_config",
    "_reset_config",
]
