"""
Configuration package for the JARVIS backend application.
==========================================================

This package provides configuration management for the backend application,
including settings for database connections, API configurations, environment
variables, and other application-wide parameters.

The configuration system supports multiple environments (development, testing,
production) and provides a centralized way to manage application settings.

Key Modules:
- startup_timeouts: Centralized timeout configuration for startup/shutdown operations
- hardware_enforcer: Automatic Hollow Client mode enforcement based on system RAM

Usage:
    # Import startup timeouts
    from backend.config import StartupTimeouts, get_timeouts

    # Access the singleton
    timeouts = get_timeouts()
    timeout_val = timeouts.backend_health_timeout

    # Or create a fresh instance
    my_timeouts = StartupTimeouts()

    # Hardware enforcement (auto-runs on import)
    from backend.config import enforce_hollow_client, get_system_ram_gb

    # Check system RAM
    ram_gb = get_system_ram_gb()

    # Manually trigger enforcement (usually not needed - runs on import)
    enforce_hollow_client(source="my_context")
"""

# Startup timeout configuration - centralized timeouts for all operations
from backend.config.startup_timeouts import (
    StartupTimeouts,
    get_timeouts,
    reset_timeouts,
)

# Hardware enforcer for Hollow Client mode - NOTE: importing this module
# triggers automatic RAM detection and enforcement on import
from backend.config.hardware_enforcer import (
    enforce_hollow_client,
    get_system_ram_gb,
)


__all__ = [
    # Startup timeouts
    "StartupTimeouts",
    "get_timeouts",
    "reset_timeouts",
    # Hardware enforcer
    "enforce_hollow_client",
    "get_system_ram_gb",
]
