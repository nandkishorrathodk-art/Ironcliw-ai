# backend/core/enterprise_supervisor_integration.py
"""
Enterprise Supervisor Integration - Unified entry point for enterprise hardening.

This module provides a clean facade for the enterprise hardening stack,
making it easy to integrate with the existing run_supervisor.py.

Usage in run_supervisor.py:
    from backend.core.enterprise_supervisor_integration import (
        enterprise_startup,
        is_enterprise_mode_available,
    )

    if is_enterprise_mode_available():
        result = await enterprise_startup()
        if not result.success:
            # Handle failure
            pass

Key Design Principles:
1. Minimal integration: run_supervisor.py just needs to call a few functions
2. Backward compatible: If imports fail, gracefully degrade
3. Observable: Emit startup summary at the end
4. Robust: Handle errors gracefully

Integration Components:
- StartupLock: Prevents concurrent supervisor runs
- ComponentRegistry: Single source of truth
- StartupDAG: Dependency-ordered startup
- RecoveryEngine: Intelligent error handling
- StartupSummary: Progress display
- SubprocessManager: Cross-repo process lifecycle
- CapabilityRouter: Dynamic capability routing
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Awaitable

logger = logging.getLogger("jarvis.enterprise_integration")

# Lazy imports to avoid circular dependencies and allow graceful degradation
_ENTERPRISE_AVAILABLE = False
try:
    from backend.core.component_registry import (
        ComponentRegistry,
        ComponentDefinition,
        ComponentStatus,
        Criticality,
        ProcessType,
        get_component_registry,
    )
    from backend.core.startup_lock import StartupLock, get_startup_lock
    from backend.core.startup_context import StartupContext, CrashHistory
    from backend.core.startup_dag import StartupDAG
    from backend.core.recovery_engine import RecoveryEngine, ErrorClassifier
    from backend.core.startup_summary import StartupSummary
    from backend.core.health_contracts import SystemHealthAggregator, HealthStatus
    from backend.core.default_components import register_default_components
    from backend.core.enterprise_startup_orchestrator import (
        EnterpriseStartupOrchestrator,
        StartupResult,
        StartupEvent,
        ComponentResult,
        create_enterprise_orchestrator,
    )
    from backend.core.subprocess_manager import SubprocessManager, get_subprocess_manager
    from backend.core.capability_router import CapabilityRouter, get_capability_router
    _ENTERPRISE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enterprise hardening modules not available: {e}")
    # Define stub classes for type hints when imports fail
    StartupResult = None
    ComponentRegistry = None
    CapabilityRouter = None
    SubprocessManager = None


def is_enterprise_mode_available() -> bool:
    """
    Check if enterprise hardening modules are available.

    This function returns True if all required enterprise hardening
    modules are successfully imported. It can be used to conditionally
    enable enterprise features.

    Returns:
        True if enterprise mode is available, False otherwise.
    """
    return _ENTERPRISE_AVAILABLE


@dataclass
class EnterpriseStartupConfig:
    """
    Configuration for enterprise startup.

    This dataclass holds all configuration options for the enterprise
    startup process, allowing customization of which features are enabled
    and how the startup behaves.

    Attributes:
        register_defaults: If True, register default Ironcliw components.
        display_summary: If True, display startup summary to console.
        use_startup_lock: If True, acquire startup lock to prevent concurrent runs.
        enable_subprocess_manager: If True, enable subprocess manager for cross-repo processes.
        enable_capability_router: If True, enable capability router for dynamic routing.
        component_starters: Dict mapping component names to async starter functions.
        event_handlers: Dict mapping event names to list of handler functions.
    """
    register_defaults: bool = True
    display_summary: bool = True
    use_startup_lock: bool = True
    enable_subprocess_manager: bool = True
    enable_capability_router: bool = True
    component_starters: Dict[str, Callable[[], Awaitable[bool]]] = field(default_factory=dict)
    event_handlers: Dict[str, List[Callable]] = field(default_factory=dict)


class EnterpriseIntegration:
    """
    Unified facade for the enterprise hardening stack.

    This class integrates all enterprise hardening components into a
    single, easy-to-use interface. It manages the lifecycle of:

    - StartupLock: Prevents concurrent supervisor runs
    - ComponentRegistry: Single source of truth for component definitions
    - StartupDAG: Dependency-ordered startup execution
    - RecoveryEngine: Intelligent error handling and recovery
    - StartupSummary: Progress display and reporting
    - SubprocessManager: Cross-repo process lifecycle management
    - CapabilityRouter: Dynamic capability routing with fallback

    Usage:
        integration = EnterpriseIntegration()
        result = await integration.startup()
        if not result.success:
            # Handle failure
            pass
        # ... run your application ...
        await integration.shutdown()
    """

    def __init__(self, config: Optional[EnterpriseStartupConfig] = None):
        """
        Initialize the EnterpriseIntegration.

        Args:
            config: Optional configuration. If None, uses default configuration.
        """
        self.config = config or EnterpriseStartupConfig()
        self._lock: Optional['StartupLock'] = None
        self._orchestrator: Optional['EnterpriseStartupOrchestrator'] = None
        self._subprocess_manager: Optional['SubprocessManager'] = None
        self._capability_router: Optional['CapabilityRouter'] = None
        self._startup_result: Optional['StartupResult'] = None

    async def startup(self) -> 'StartupResult':
        """
        Execute enterprise startup sequence.

        The startup sequence performs the following steps:
        1. Acquire startup lock (if enabled)
        2. Load startup context (crash history)
        3. Register default components (if enabled)
        4. Create orchestrator with all integrations
        5. Register custom component starters
        6. Register event handlers
        7. Execute startup
        8. Display summary (if enabled)
        9. Return result

        Returns:
            StartupResult with success status, timing, and component results.
        """
        if not _ENTERPRISE_AVAILABLE:
            logger.error("Enterprise hardening modules not available")
            # Return a minimal failure result
            from dataclasses import dataclass as dc
            @dc
            class MinimalResult:
                success: bool = False
                total_time: float = 0.0
                components: dict = field(default_factory=dict)
                healthy_count: int = 0
                failed_count: int = 1
                disabled_count: int = 0
                overall_status: str = "FAILED"
            return MinimalResult()

        start_time = datetime.now()

        # Acquire startup lock
        if self.config.use_startup_lock:
            self._lock = get_startup_lock()
            if not self._lock.acquire():
                logger.error("Failed to acquire startup lock - another supervisor is running")
                return StartupResult(
                    success=False,
                    total_time=0.0,
                    components={},
                    healthy_count=0,
                    failed_count=1,
                    disabled_count=0,
                    overall_status="FAILED"
                )

        try:
            # Load context
            context = StartupContext.load()

            if context.needs_conservative_startup:
                logger.warning("Conservative startup mode - recent crashes detected")

            # Create orchestrator
            self._orchestrator = create_enterprise_orchestrator(
                register_defaults=self.config.register_defaults,
                startup_context=context,
            )

            # Register custom component starters
            for name, starter in self.config.component_starters.items():
                self._orchestrator.register_component_starter(name, starter)

            # Register event handlers
            for event_name, handlers in self.config.event_handlers.items():
                try:
                    event = StartupEvent[event_name.upper()]
                    for handler in handlers:
                        self._orchestrator.on_event(event, handler)
                except (KeyError, ValueError):
                    logger.warning(f"Unknown event: {event_name}")

            # Set up subprocess manager if enabled
            if self.config.enable_subprocess_manager:
                self._subprocess_manager = get_subprocess_manager(
                    self._orchestrator.registry,
                    self._orchestrator.recovery_engine,
                )

                # Register subprocess starters for cross-repo components
                self._register_subprocess_starters()

            # Set up capability router if enabled
            if self.config.enable_capability_router:
                self._capability_router = get_capability_router(
                    self._orchestrator.registry
                )

            # Execute startup
            self._startup_result = await self._orchestrator.orchestrate_startup()

            # Display summary
            if self.config.display_summary:
                summary = StartupSummary(self._orchestrator.registry)
                summary.start_time = start_time
                summary.end_time = datetime.now()
                print(summary.format_summary())

                # Save to state file
                try:
                    summary.save_to_file()
                except Exception as e:
                    logger.warning(f"Failed to save startup summary: {e}")

            return self._startup_result

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Enterprise startup failed: {e}", exc_info=True)
            return StartupResult(
                success=False,
                total_time=(datetime.now() - start_time).total_seconds(),
                components={},
                healthy_count=0,
                failed_count=1,
                disabled_count=0,
                overall_status="FAILED"
            )

    async def shutdown(self) -> None:
        """
        Execute graceful shutdown.

        Shuts down all managed resources in the following order:
        1. Subprocess manager (all managed subprocesses)
        2. Orchestrator (all components)
        3. Startup lock (release)

        This method is safe to call even if startup was not called or failed.
        """
        if self._subprocess_manager:
            try:
                await self._subprocess_manager.shutdown_all()
            except Exception as e:
                logger.warning(f"Error shutting down subprocess manager: {e}")

        if self._orchestrator:
            try:
                await self._orchestrator.shutdown_all()
            except Exception as e:
                logger.warning(f"Error shutting down orchestrator: {e}")

        if self._lock:
            try:
                self._lock.release()
            except Exception as e:
                logger.warning(f"Error releasing startup lock: {e}")

    def _register_subprocess_starters(self) -> None:
        """
        Register subprocess starters for cross-repo components.

        For each component with ProcessType.SUBPROCESS, registers an async
        starter function that uses the SubprocessManager to start the process.
        """
        if not self._orchestrator or not self._subprocess_manager:
            return

        for defn in self._orchestrator.registry.all_definitions():
            if defn.process_type == ProcessType.SUBPROCESS:
                # Create a closure to capture the definition
                async def start_subprocess(d=defn):
                    handle = await self._subprocess_manager.start(d)
                    return handle.is_alive

                self._orchestrator.register_component_starter(
                    defn.name,
                    start_subprocess
                )

    @property
    def registry(self) -> Optional['ComponentRegistry']:
        """
        Get the component registry.

        Returns:
            ComponentRegistry if startup has been called, None otherwise.
        """
        return self._orchestrator.registry if self._orchestrator else None

    @property
    def capability_router(self) -> Optional['CapabilityRouter']:
        """
        Get the capability router.

        Returns:
            CapabilityRouter if enabled and startup has been called, None otherwise.
        """
        return self._capability_router

    @property
    def subprocess_manager(self) -> Optional['SubprocessManager']:
        """
        Get the subprocess manager.

        Returns:
            SubprocessManager if enabled and startup has been called, None otherwise.
        """
        return self._subprocess_manager

    async def health_check(self) -> Dict[str, Any]:
        """
        Get system health status.

        Performs a health check on all registered components and returns
        aggregated health information.

        Returns:
            Dict containing:
            - status: Overall system status ("healthy", "degraded", "unhealthy", or "not_initialized")
            - components: Dict mapping component names to their status
            - capabilities: Dict mapping capability names to availability info
        """
        if not self._orchestrator:
            return {"status": "not_initialized"}

        try:
            health = await self._orchestrator.health_check_all()
            return {
                "status": health.overall.value,
                "components": {
                    name: report.status.value
                    for name, report in health.components.items()
                },
                "capabilities": {
                    name: cap.available
                    for name, cap in health.capabilities.items()
                },
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


# Module-level singleton
_integration: Optional[EnterpriseIntegration] = None


async def enterprise_startup(
    config: Optional[EnterpriseStartupConfig] = None
) -> 'StartupResult':
    """
    Execute enterprise startup.

    This is the main entry point for integrating with run_supervisor.py.
    Creates a global EnterpriseIntegration instance and executes startup.

    Usage:
        from backend.core.enterprise_supervisor_integration import enterprise_startup

        result = await enterprise_startup()
        if not result.success:
            print("Startup failed")
            sys.exit(1)

    Args:
        config: Optional EnterpriseStartupConfig for customization.

    Returns:
        StartupResult with success status, timing, and component results.
    """
    global _integration
    _integration = EnterpriseIntegration(config)
    return await _integration.startup()


async def enterprise_shutdown() -> None:
    """
    Execute enterprise shutdown.

    Shuts down the global EnterpriseIntegration instance and clears
    the singleton reference.
    """
    global _integration
    if _integration:
        await _integration.shutdown()
        _integration = None


def get_enterprise_integration() -> Optional[EnterpriseIntegration]:
    """
    Get the current enterprise integration instance.

    Returns:
        The global EnterpriseIntegration instance, or None if not initialized.
    """
    return _integration


def get_enterprise_registry() -> Optional['ComponentRegistry']:
    """
    Get the component registry from the current integration.

    Returns:
        ComponentRegistry if enterprise startup has been called, None otherwise.
    """
    return _integration.registry if _integration else None


def get_enterprise_router() -> Optional['CapabilityRouter']:
    """
    Get the capability router from the current integration.

    Returns:
        CapabilityRouter if enabled and startup has been called, None otherwise.
    """
    return _integration.capability_router if _integration else None


# Export convenience functions for common operations
__all__ = [
    'is_enterprise_mode_available',
    'enterprise_startup',
    'enterprise_shutdown',
    'get_enterprise_integration',
    'get_enterprise_registry',
    'get_enterprise_router',
    'EnterpriseStartupConfig',
    'EnterpriseIntegration',
]
