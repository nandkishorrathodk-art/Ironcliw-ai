"""
Startup Resilience Utilities - Ironcliw-Specific Startup Integration
===================================================================

This module provides pre-configured resilience primitives for Ironcliw startup,
making it easy to integrate graceful degradation, background recovery, and
health probes into the startup sequence.

Features:
- Pre-configured health probes for Docker, Ollama, and Invincible Node
- Factory functions for creating Ironcliw-specific resilience components
- StartupResilience coordinator for managing all startup-related resilience
- Integration helpers for broadcasting progress and handling failures

Design Principles:
- NEVER block startup on recoverable failures
- Graceful degradation with background recovery
- Minimal changes to unified_supervisor.py - use these utilities

Example usage:
    from backend.core.resilience.startup import (
        StartupResilience,
        create_docker_health_probe,
        create_invincible_node_recovery,
    )

    # In JarvisSystemKernel
    resilience = StartupResilience(logger=self.logger)
    await resilience.start()

    # Check Docker health (non-blocking)
    docker_healthy = await resilience.check_docker()
    if not docker_healthy:
        # Background recovery already started
        pass

    # On shutdown
    await resilience.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Optional,
    Protocol,
)

from backend.core.resilience.health import HealthProbe
from backend.core.resilience.recovery import BackgroundRecovery, RecoveryConfig
from backend.core.resilience.capability import CapabilityUpgrade
from backend.core.resilience.types import CapabilityState, RecoveryState


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StartupResilienceConfig:
    """
    Configuration for startup resilience behavior.

    Attributes:
        docker_check_timeout: Timeout for Docker health checks in seconds.
        docker_cache_ttl: How long to cache Docker health status.
        docker_unhealthy_threshold: Consecutive failures before unhealthy.
        docker_recovery_enabled: Whether to auto-recover Docker.
        ollama_check_timeout: Timeout for Ollama health checks.
        ollama_cache_ttl: How long to cache Ollama health status.
        ollama_recovery_base_delay: Initial delay for Ollama recovery retries.
        invincible_node_check_timeout: Timeout for cloud VM health checks.
        invincible_node_recovery_max_attempts: Max recovery attempts for cloud VM.
        local_llm_upgrade_interval: Interval for checking if full LLM mode available.
    """
    # Docker settings
    docker_check_timeout: float = 5.0
    docker_cache_ttl: float = 30.0
    docker_unhealthy_threshold: int = 3
    docker_recovery_enabled: bool = True
    docker_recovery_base_delay: float = 10.0
    docker_recovery_max_delay: float = 120.0
    docker_recovery_max_attempts: int = 10

    # Ollama settings
    ollama_check_timeout: float = 10.0
    ollama_cache_ttl: float = 30.0
    ollama_unhealthy_threshold: int = 3
    ollama_recovery_enabled: bool = True
    ollama_recovery_base_delay: float = 5.0
    ollama_recovery_max_delay: float = 60.0
    ollama_recovery_max_attempts: int = 20

    # Invincible Node (GCP VM) settings
    invincible_node_check_timeout: float = 30.0
    invincible_node_cache_ttl: float = 60.0
    invincible_node_unhealthy_threshold: int = 2
    invincible_node_recovery_enabled: bool = True
    invincible_node_recovery_base_delay: float = 30.0
    invincible_node_recovery_max_delay: float = 300.0
    invincible_node_recovery_max_attempts: int = 5

    # Local LLM capability upgrade settings
    local_llm_upgrade_interval: float = 60.0


# =============================================================================
# LOGGER PROTOCOL
# =============================================================================

class StartupLogger(Protocol):
    """Protocol for loggers used by startup resilience."""

    def info(self, msg: str) -> None:
        """Log info message."""
        ...

    def warning(self, msg: str) -> None:
        """Log warning message."""
        ...

    def debug(self, msg: str) -> None:
        """Log debug message."""
        ...


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_docker_health_check(
    timeout: float = 5.0,
) -> Callable[[], Awaitable[bool]]:
    """
    Create an async health check function for Docker daemon.

    Uses 'docker info' command which is lightweight and verifies
    the daemon is fully responsive, not just running.

    Args:
        timeout: Subprocess timeout in seconds.

    Returns:
        Async function that returns True if Docker is healthy.
    """
    async def check_docker() -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                'docker', 'info',
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
                return proc.returncode == 0
            except asyncio.TimeoutError:
                proc.kill()
                return False
        except FileNotFoundError:
            # Docker not installed
            return False
        except Exception:
            return False

    return check_docker


def create_ollama_health_check(
    host: str = "localhost",
    port: int = 11434,
    timeout: float = 10.0,
) -> Callable[[], Awaitable[bool]]:
    """
    Create an async health check function for Ollama server.

    Uses the /api/tags endpoint which is lightweight and
    verifies the server is fully responsive.

    Args:
        host: Ollama server host.
        port: Ollama server port.
        timeout: HTTP request timeout in seconds.

    Returns:
        Async function that returns True if Ollama is healthy.
    """
    async def check_ollama() -> bool:
        try:
            # Use aiohttp if available, otherwise fall back to urllib
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    url = f"http://{host}:{port}/api/tags"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                        return resp.status == 200
            except ImportError:
                # Fallback to urllib in thread
                import urllib.request
                url = f"http://{host}:{port}/api/tags"

                def sync_check() -> bool:
                    try:
                        req = urllib.request.Request(url, method='GET')
                        with urllib.request.urlopen(req, timeout=timeout) as resp:
                            return resp.status == 200
                    except Exception:
                        return False

                return await asyncio.to_thread(sync_check)
        except Exception:
            return False

    return check_ollama


def create_invincible_node_health_check(
    get_vm_manager: Callable[[], Awaitable[Any]],
    port: int = 8000,
    timeout: float = 30.0,
) -> Callable[[], Awaitable[bool]]:
    """
    Create an async health check function for Invincible Node (GCP VM).

    Uses the GCP VM manager to check if the static IP spot VM is
    healthy and ready for inference.

    Args:
        get_vm_manager: Async function to get the GCP VM manager.
        port: The port to check health on.
        timeout: Health check timeout in seconds.

    Returns:
        Async function that returns True if the VM is healthy.
    """
    async def check_invincible_node() -> bool:
        try:
            manager = await get_vm_manager()
            if not manager.is_static_vm_mode:
                return False

            # Check if VM is running and healthy
            health = await asyncio.wait_for(
                manager.check_static_vm_health(port=port),
                timeout=timeout,
            )
            return health.get("healthy", False)
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    return check_invincible_node


def create_docker_health_probe(
    config: Optional[StartupResilienceConfig] = None,
    on_unhealthy: Optional[Callable[[], Awaitable[None]]] = None,
    on_healthy: Optional[Callable[[], Awaitable[None]]] = None,
) -> HealthProbe:
    """
    Create a pre-configured health probe for Docker daemon.

    Args:
        config: Startup resilience configuration.
        on_unhealthy: Callback when Docker becomes unhealthy.
        on_healthy: Callback when Docker recovers.

    Returns:
        Configured HealthProbe instance.
    """
    config = config or StartupResilienceConfig()

    return HealthProbe(
        check_fn=create_docker_health_check(timeout=config.docker_check_timeout),
        cache_ttl=config.docker_cache_ttl,
        timeout=config.docker_check_timeout + 1.0,  # Slightly longer than subprocess timeout
        unhealthy_threshold=config.docker_unhealthy_threshold,
        on_unhealthy=on_unhealthy,
        on_healthy=on_healthy,
    )


def create_ollama_health_probe(
    config: Optional[StartupResilienceConfig] = None,
    host: str = "localhost",
    port: int = 11434,
    on_unhealthy: Optional[Callable[[], Awaitable[None]]] = None,
    on_healthy: Optional[Callable[[], Awaitable[None]]] = None,
) -> HealthProbe:
    """
    Create a pre-configured health probe for Ollama server.

    Args:
        config: Startup resilience configuration.
        host: Ollama server host.
        port: Ollama server port.
        on_unhealthy: Callback when Ollama becomes unhealthy.
        on_healthy: Callback when Ollama recovers.

    Returns:
        Configured HealthProbe instance.
    """
    config = config or StartupResilienceConfig()

    return HealthProbe(
        check_fn=create_ollama_health_check(host=host, port=port, timeout=config.ollama_check_timeout),
        cache_ttl=config.ollama_cache_ttl,
        timeout=config.ollama_check_timeout + 1.0,
        unhealthy_threshold=config.ollama_unhealthy_threshold,
        on_unhealthy=on_unhealthy,
        on_healthy=on_healthy,
    )


def create_invincible_node_health_probe(
    get_vm_manager: Callable[[], Awaitable[Any]],
    config: Optional[StartupResilienceConfig] = None,
    port: int = 8000,
    on_unhealthy: Optional[Callable[[], Awaitable[None]]] = None,
    on_healthy: Optional[Callable[[], Awaitable[None]]] = None,
) -> HealthProbe:
    """
    Create a pre-configured health probe for Invincible Node (GCP VM).

    Args:
        get_vm_manager: Async function to get the GCP VM manager.
        config: Startup resilience configuration.
        port: The port to check health on.
        on_unhealthy: Callback when VM becomes unhealthy.
        on_healthy: Callback when VM recovers.

    Returns:
        Configured HealthProbe instance.
    """
    config = config or StartupResilienceConfig()

    return HealthProbe(
        check_fn=create_invincible_node_health_check(
            get_vm_manager=get_vm_manager,
            port=port,
            timeout=config.invincible_node_check_timeout,
        ),
        cache_ttl=config.invincible_node_cache_ttl,
        timeout=config.invincible_node_check_timeout + 5.0,
        unhealthy_threshold=config.invincible_node_unhealthy_threshold,
        on_unhealthy=on_unhealthy,
        on_healthy=on_healthy,
    )


def create_docker_recovery(
    config: Optional[StartupResilienceConfig] = None,
    on_success: Optional[Callable[[], Awaitable[None]]] = None,
    on_paused: Optional[Callable[[], Awaitable[None]]] = None,
) -> BackgroundRecovery:
    """
    Create a background recovery for Docker daemon.

    Attempts to start the Docker daemon (on macOS) or wait for it
    to become available (on other platforms).

    Args:
        config: Startup resilience configuration.
        on_success: Callback when Docker is recovered.
        on_paused: Callback when recovery is paused due to max attempts.

    Returns:
        Configured BackgroundRecovery instance.
    """
    config = config or StartupResilienceConfig()

    async def recover_docker() -> bool:
        """Attempt to recover Docker daemon."""
        import platform
        system = platform.system().lower()

        try:
            if system == "darwin":
                # Try to open Docker Desktop on macOS
                proc = await asyncio.create_subprocess_exec(
                    'open', '-a', 'Docker',
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()

                # Wait a bit for Docker to start
                await asyncio.sleep(5.0)

            # Check if Docker is now healthy
            check = create_docker_health_check(timeout=config.docker_check_timeout)
            return await check()

        except Exception:
            return False

    return BackgroundRecovery(
        recover_fn=recover_docker,
        config=RecoveryConfig(
            base_delay=config.docker_recovery_base_delay,
            max_delay=config.docker_recovery_max_delay,
            max_attempts=config.docker_recovery_max_attempts,
            timeout=config.docker_check_timeout + 10.0,
        ),
        on_success=on_success,
        on_paused=on_paused,
    )


def create_ollama_recovery(
    config: Optional[StartupResilienceConfig] = None,
    host: str = "localhost",
    port: int = 11434,
    on_success: Optional[Callable[[], Awaitable[None]]] = None,
    on_paused: Optional[Callable[[], Awaitable[None]]] = None,
) -> BackgroundRecovery:
    """
    Create a background recovery for Ollama server.

    Attempts to start Ollama serve or wait for it to become available.

    Args:
        config: Startup resilience configuration.
        host: Ollama server host.
        port: Ollama server port.
        on_success: Callback when Ollama is recovered.
        on_paused: Callback when recovery is paused due to max attempts.

    Returns:
        Configured BackgroundRecovery instance.
    """
    config = config or StartupResilienceConfig()

    async def recover_ollama() -> bool:
        """Attempt to recover Ollama server."""
        try:
            # Try to start ollama serve in the background
            # This is a no-op if Ollama is already running
            proc = await asyncio.create_subprocess_exec(
                'ollama', 'serve',
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,  # Detach from parent
            )

            # Don't wait for the process, just check if Ollama is healthy
            await asyncio.sleep(2.0)

            check = create_ollama_health_check(host=host, port=port, timeout=config.ollama_check_timeout)
            return await check()

        except FileNotFoundError:
            # Ollama not installed
            return False
        except Exception:
            return False

    return BackgroundRecovery(
        recover_fn=recover_ollama,
        config=RecoveryConfig(
            base_delay=config.ollama_recovery_base_delay,
            max_delay=config.ollama_recovery_max_delay,
            max_attempts=config.ollama_recovery_max_attempts,
            timeout=config.ollama_check_timeout + 5.0,
        ),
        on_success=on_success,
        on_paused=on_paused,
    )


def create_invincible_node_recovery(
    get_vm_manager: Callable[[], Awaitable[Any]],
    config: Optional[StartupResilienceConfig] = None,
    port: int = 8000,
    on_success: Optional[Callable[[], Awaitable[None]]] = None,
    on_paused: Optional[Callable[[], Awaitable[None]]] = None,
) -> BackgroundRecovery:
    """
    Create a background recovery for Invincible Node (GCP VM).

    Attempts to wake up the cloud VM and wait for it to become healthy.

    Args:
        get_vm_manager: Async function to get the GCP VM manager.
        config: Startup resilience configuration.
        port: The port to check health on.
        on_success: Callback when VM is recovered.
        on_paused: Callback when recovery is paused due to max attempts.

    Returns:
        Configured BackgroundRecovery instance.
    """
    config = config or StartupResilienceConfig()

    async def recover_invincible_node() -> bool:
        """Attempt to wake up the Invincible Node."""
        try:
            manager = await get_vm_manager()
            if not manager.is_static_vm_mode:
                return False

            # Try to ensure VM is ready
            success, ip, status = await asyncio.wait_for(
                manager.ensure_static_vm_ready(port=port),
                timeout=config.invincible_node_check_timeout,
            )
            return success

        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    return BackgroundRecovery(
        recover_fn=recover_invincible_node,
        config=RecoveryConfig(
            base_delay=config.invincible_node_recovery_base_delay,
            max_delay=config.invincible_node_recovery_max_delay,
            max_attempts=config.invincible_node_recovery_max_attempts,
            timeout=config.invincible_node_check_timeout + 30.0,
        ),
        on_success=on_success,
        on_paused=on_paused,
    )


def create_local_llm_capability_upgrade(
    check_cloud_available: Callable[[], Awaitable[bool]],
    activate_cloud: Callable[[], Awaitable[None]],
    deactivate_cloud: Callable[[], Awaitable[None]],
    on_upgrade: Optional[Callable[[], Awaitable[None]]] = None,
    on_downgrade: Optional[Callable[[], Awaitable[None]]] = None,
) -> CapabilityUpgrade:
    """
    Create a capability upgrade for switching between local and cloud LLM.

    This enables hot-swapping between:
    - DEGRADED: Local/cached LLM (Ollama)
    - FULL: Cloud LLM (GCP VM / Invincible Node)

    Args:
        check_cloud_available: Async function to check if cloud LLM is available.
        activate_cloud: Async function to switch to cloud LLM.
        deactivate_cloud: Async function to switch back to local LLM.
        on_upgrade: Callback when upgrading to cloud.
        on_downgrade: Callback when downgrading to local.

    Returns:
        Configured CapabilityUpgrade instance.
    """
    return CapabilityUpgrade(
        name="local_llm_mode",
        check_available=check_cloud_available,
        activate=activate_cloud,
        deactivate=deactivate_cloud,
        on_upgrade=on_upgrade,
        on_downgrade=on_downgrade,
    )


# =============================================================================
# STARTUP RESILIENCE COORDINATOR
# =============================================================================

@dataclass
class StartupResilience:
    """
    Coordinator for all startup-related resilience components.

    Manages health probes, background recovery, and capability upgrades
    for Docker, Ollama, and Invincible Node services.

    This class provides a single interface for:
    - Starting/stopping all resilience components
    - Checking service health (non-blocking)
    - Getting status of all services
    - Notifying when conditions change (e.g., network restored)

    Example:
        # Initialize
        resilience = StartupResilience(logger=self.logger)
        await resilience.start()

        # Check services (non-blocking)
        docker_ok = await resilience.check_docker()
        ollama_ok = await resilience.check_ollama()

        # Get overall status
        status = resilience.get_status()

        # When network comes back
        resilience.notify_conditions_changed()

        # Cleanup
        await resilience.stop()
    """

    logger: Any  # StartupLogger or compatible
    config: StartupResilienceConfig = field(default_factory=StartupResilienceConfig)

    # Health probes (lazy initialized)
    _docker_probe: Optional[HealthProbe] = field(default=None, init=False, repr=False)
    _ollama_probe: Optional[HealthProbe] = field(default=None, init=False, repr=False)
    _invincible_node_probe: Optional[HealthProbe] = field(default=None, init=False, repr=False)

    # Background recovery (lazy initialized)
    _docker_recovery: Optional[BackgroundRecovery] = field(default=None, init=False, repr=False)
    _ollama_recovery: Optional[BackgroundRecovery] = field(default=None, init=False, repr=False)
    _invincible_node_recovery: Optional[BackgroundRecovery] = field(default=None, init=False, repr=False)

    # Capability upgrade (lazy initialized)
    _llm_upgrade: Optional[CapabilityUpgrade] = field(default=None, init=False, repr=False)

    # State
    _started: bool = field(default=False, init=False, repr=False)
    _get_vm_manager: Optional[Callable[[], Awaitable[Any]]] = field(default=None, init=False, repr=False)
    _recovery_pause_deadlines: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def configure_invincible_node(
        self,
        get_vm_manager: Callable[[], Awaitable[Any]],
        port: int = 8000,
    ) -> None:
        """
        Configure Invincible Node (GCP VM) resilience.

        Must be called before start() if Invincible Node support is desired.

        Args:
            get_vm_manager: Async function to get the GCP VM manager.
            port: The port for health checks.
        """
        self._get_vm_manager = get_vm_manager

        # Create probe and recovery for Invincible Node
        self._invincible_node_probe = create_invincible_node_health_probe(
            get_vm_manager=get_vm_manager,
            config=self.config,
            port=port,
            on_unhealthy=self._on_invincible_node_unhealthy,
            on_healthy=self._on_invincible_node_healthy,
        )

        if self.config.invincible_node_recovery_enabled:
            self._invincible_node_recovery = create_invincible_node_recovery(
                get_vm_manager=get_vm_manager,
                config=self.config,
                port=port,
                on_success=self._on_invincible_node_recovered,
                on_paused=self._on_invincible_node_recovery_paused,
            )

    def configure_llm_upgrade(
        self,
        check_cloud_available: Callable[[], Awaitable[bool]],
        activate_cloud: Callable[[], Awaitable[None]],
        deactivate_cloud: Callable[[], Awaitable[None]],
    ) -> None:
        """
        Configure LLM capability upgrade.

        Enables hot-swapping between local (Ollama) and cloud (GCP) LLM.

        Args:
            check_cloud_available: Check if cloud LLM is available.
            activate_cloud: Switch to cloud LLM.
            deactivate_cloud: Switch back to local LLM.
        """
        self._llm_upgrade = create_local_llm_capability_upgrade(
            check_cloud_available=check_cloud_available,
            activate_cloud=activate_cloud,
            deactivate_cloud=deactivate_cloud,
            on_upgrade=self._on_llm_upgraded,
            on_downgrade=self._on_llm_downgraded,
        )

    async def start(self) -> None:
        """
        Start all resilience components.

        Initializes health probes and starts background monitoring.
        Does NOT block on service availability.
        """
        if self._started:
            return

        self._started = True
        self.logger.info("[Resilience] Starting startup resilience components...")

        # Initialize Docker probe and recovery
        self._docker_probe = create_docker_health_probe(
            config=self.config,
            on_unhealthy=self._on_docker_unhealthy,
            on_healthy=self._on_docker_healthy,
        )

        if self.config.docker_recovery_enabled:
            self._docker_recovery = create_docker_recovery(
                config=self.config,
                on_success=self._on_docker_recovered,
                on_paused=self._on_docker_recovery_paused,
            )

        # Initialize Ollama probe and recovery
        self._ollama_probe = create_ollama_health_probe(
            config=self.config,
            on_unhealthy=self._on_ollama_unhealthy,
            on_healthy=self._on_ollama_healthy,
        )

        if self.config.ollama_recovery_enabled:
            self._ollama_recovery = create_ollama_recovery(
                config=self.config,
                on_success=self._on_ollama_recovered,
                on_paused=self._on_ollama_recovery_paused,
            )

        # Start LLM upgrade monitoring if configured
        if self._llm_upgrade:
            await self._llm_upgrade.start_monitoring(
                interval=self.config.local_llm_upgrade_interval
            )

        self.logger.info("[Resilience] Startup resilience components ready")

    async def stop(self) -> None:
        """
        Stop all resilience components cleanly.
        """
        if not self._started:
            return

        self.logger.info("[Resilience] Stopping startup resilience components...")

        # Stop LLM upgrade monitoring
        if self._llm_upgrade:
            await self._llm_upgrade.stop_monitoring()

        # Stop background recoveries
        if self._docker_recovery:
            await self._docker_recovery.stop()
        if self._ollama_recovery:
            await self._ollama_recovery.stop()
        if self._invincible_node_recovery:
            await self._invincible_node_recovery.stop()

        self._recovery_pause_deadlines.clear()
        self._started = False
        self.logger.info("[Resilience] Startup resilience components stopped")

    def _set_recovery_pause_deadline(
        self,
        service: str,
        recovery: Optional[BackgroundRecovery],
    ) -> None:
        """
        Set the minimum delay before attempting resume() after PAUSED.

        Uses the recovery's own backoff configuration instead of hardcoded
        values so cooldown behavior stays aligned with service policy.
        """
        if recovery is None:
            return
        cooldown = max(recovery.config.base_delay, recovery.config.max_delay)
        self._recovery_pause_deadlines[service] = time.monotonic() + cooldown

    def _clear_recovery_pause_deadline(self, service: str) -> None:
        """Clear paused-recovery cooldown gate for a service."""
        self._recovery_pause_deadlines.pop(service, None)

    async def _activate_recovery_if_needed(
        self,
        service: str,
        is_healthy: bool,
        recovery: Optional[BackgroundRecovery],
    ) -> None:
        """
        Keep recovery state machine aligned with current health.

        Handles all non-running states explicitly:
        - IDLE: start recovery
        - SUCCEEDED: restart recovery loop if service regressed
        - PAUSED: resume after cooldown window
        """
        if recovery is None:
            return

        if is_healthy:
            self._clear_recovery_pause_deadline(service)
            return

        if recovery.state in (RecoveryState.IDLE, RecoveryState.SUCCEEDED):
            action = "starting" if recovery.state == RecoveryState.IDLE else "restarting"
            self.logger.info(
                f"[Resilience] {service.title()} unhealthy, {action} background recovery..."
            )
            self._clear_recovery_pause_deadline(service)
            await recovery.start()
            return

        if recovery.state == RecoveryState.PAUSED:
            now = time.monotonic()
            resume_at = self._recovery_pause_deadlines.get(service, 0.0)
            if now < resume_at:
                return
            self.logger.warning(
                f"[Resilience] {service.title()} still unhealthy after pause cooldown, resuming recovery..."
            )
            self._clear_recovery_pause_deadline(service)
            await recovery.resume()

    async def check_docker(self, force: bool = False) -> bool:
        """
        Check Docker health (non-blocking).

        If Docker is unhealthy and recovery is enabled, starts background
        recovery automatically.

        Args:
            force: Bypass cache and force fresh check.

        Returns:
            True if Docker is healthy, False otherwise.
        """
        if self._docker_probe is None:
            return False

        is_healthy = await self._docker_probe.check(force=force)
        await self._activate_recovery_if_needed(
            service="docker",
            is_healthy=is_healthy,
            recovery=self._docker_recovery,
        )

        return is_healthy

    async def check_ollama(self, force: bool = False) -> bool:
        """
        Check Ollama health (non-blocking).

        If Ollama is unhealthy and recovery is enabled, starts background
        recovery automatically.

        Args:
            force: Bypass cache and force fresh check.

        Returns:
            True if Ollama is healthy, False otherwise.
        """
        if self._ollama_probe is None:
            return False

        is_healthy = await self._ollama_probe.check(force=force)
        await self._activate_recovery_if_needed(
            service="ollama",
            is_healthy=is_healthy,
            recovery=self._ollama_recovery,
        )

        return is_healthy

    async def check_invincible_node(self, force: bool = False) -> bool:
        """
        Check Invincible Node health (non-blocking).

        If the node is unhealthy and recovery is enabled, starts background
        recovery automatically.

        Args:
            force: Bypass cache and force fresh check.

        Returns:
            True if Invincible Node is healthy, False otherwise.
        """
        if self._invincible_node_probe is None:
            return False

        is_healthy = await self._invincible_node_probe.check(force=force)
        await self._activate_recovery_if_needed(
            service="invincible node",
            is_healthy=is_healthy,
            recovery=self._invincible_node_recovery,
        )

        return is_healthy

    async def try_llm_upgrade(self) -> bool:
        """
        Attempt to upgrade from local to cloud LLM.

        Returns:
            True if upgrade succeeded (or already in full mode).
        """
        if self._llm_upgrade is None:
            return False

        return await self._llm_upgrade.try_upgrade()

    def notify_conditions_changed(self) -> None:
        """
        Notify that conditions have changed (e.g., network restored).

        Speeds up all background recoveries by waking them early
        and reducing their next delay.
        """
        # Allow immediate resume attempts on next health check cycle.
        self._recovery_pause_deadlines.clear()
        if self._docker_recovery:
            self._docker_recovery.notify_conditions_changed()
        if self._ollama_recovery:
            self._ollama_recovery.notify_conditions_changed()
        if self._invincible_node_recovery:
            self._invincible_node_recovery.notify_conditions_changed()

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of all resilience components.

        Returns:
            Dict with status of each service and overall health.
        """
        docker_status = {
            "healthy": self._docker_probe.is_unhealthy if self._docker_probe else None,
            "consecutive_failures": self._docker_probe.consecutive_failures if self._docker_probe else 0,
            "recovery_state": self._docker_recovery.state.name if self._docker_recovery else "N/A",
            "recovery_attempts": self._docker_recovery.attempt_count if self._docker_recovery else 0,
        }
        if self._docker_probe:
            docker_status["healthy"] = not self._docker_probe.is_unhealthy

        ollama_status = {
            "healthy": self._ollama_probe.is_unhealthy if self._ollama_probe else None,
            "consecutive_failures": self._ollama_probe.consecutive_failures if self._ollama_probe else 0,
            "recovery_state": self._ollama_recovery.state.name if self._ollama_recovery else "N/A",
            "recovery_attempts": self._ollama_recovery.attempt_count if self._ollama_recovery else 0,
        }
        if self._ollama_probe:
            ollama_status["healthy"] = not self._ollama_probe.is_unhealthy

        invincible_node_status = {
            "configured": self._invincible_node_probe is not None,
            "healthy": None,
            "consecutive_failures": 0,
            "recovery_state": "N/A",
            "recovery_attempts": 0,
        }
        if self._invincible_node_probe:
            invincible_node_status["healthy"] = not self._invincible_node_probe.is_unhealthy
            invincible_node_status["consecutive_failures"] = self._invincible_node_probe.consecutive_failures
        if self._invincible_node_recovery:
            invincible_node_status["recovery_state"] = self._invincible_node_recovery.state.name
            invincible_node_status["recovery_attempts"] = self._invincible_node_recovery.attempt_count

        llm_status = {
            "configured": self._llm_upgrade is not None,
            "state": self._llm_upgrade.state.name if self._llm_upgrade else "N/A",
            "is_full_mode": self._llm_upgrade.is_full if self._llm_upgrade else False,
        }

        return {
            "started": self._started,
            "docker": docker_status,
            "ollama": ollama_status,
            "invincible_node": invincible_node_status,
            "llm_mode": llm_status,
        }

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    async def _on_docker_unhealthy(self) -> None:
        """Called when Docker becomes unhealthy."""
        self.logger.warning("[Resilience] Docker daemon is unhealthy")

    async def _on_docker_healthy(self) -> None:
        """Called when Docker recovers to healthy."""
        self.logger.info("[Resilience] Docker daemon is now healthy")

    async def _on_docker_recovered(self) -> None:
        """Called when Docker is recovered via background recovery."""
        self.logger.info("[Resilience] Docker daemon recovered successfully")
        self._clear_recovery_pause_deadline("docker")
        # Reset the health probe so next check uses fresh state
        if self._docker_probe:
            self._docker_probe.reset()

    async def _on_docker_recovery_paused(self) -> None:
        """Called when Docker recovery is paused due to max attempts."""
        self.logger.warning("[Resilience] Docker recovery paused after max attempts")
        self._set_recovery_pause_deadline("docker", self._docker_recovery)

    async def _on_ollama_unhealthy(self) -> None:
        """Called when Ollama becomes unhealthy."""
        self.logger.warning("[Resilience] Ollama server is unhealthy")

    async def _on_ollama_healthy(self) -> None:
        """Called when Ollama recovers to healthy."""
        self.logger.info("[Resilience] Ollama server is now healthy")

    async def _on_ollama_recovered(self) -> None:
        """Called when Ollama is recovered via background recovery."""
        self.logger.info("[Resilience] Ollama server recovered successfully")
        self._clear_recovery_pause_deadline("ollama")
        if self._ollama_probe:
            self._ollama_probe.reset()

    async def _on_ollama_recovery_paused(self) -> None:
        """Called when Ollama recovery is paused due to max attempts."""
        self.logger.warning("[Resilience] Ollama recovery paused after max attempts")
        self._set_recovery_pause_deadline("ollama", self._ollama_recovery)

    async def _on_invincible_node_unhealthy(self) -> None:
        """
        Called when Invincible Node becomes unhealthy.
        v219.0: Clear hollow client env vars so clients fall back to local/alternate paths.
        """
        self.logger.warning("[Resilience] Invincible Node (GCP VM) is unhealthy")
        
        # v219.0: Clear the hollow client flag so clients know to fall back
        # This is the shared state between kernel and clients
        os.environ.pop("Ironcliw_HOLLOW_CLIENT_ACTIVE", None)
        self.logger.info("[Resilience] v219.0 Cleared Ironcliw_HOLLOW_CLIENT_ACTIVE (unhealthy)")

    async def _on_invincible_node_healthy(self) -> None:
        """
        Called when Invincible Node recovers to healthy.
        v219.0: Re-propagate hollow client env vars if IP is known.
        """
        self.logger.info("[Resilience] Invincible Node (GCP VM) is now healthy")
        
        # v219.0: Re-activate hollow client if we have the IP
        invincible_ip = os.environ.get("Ironcliw_INVINCIBLE_NODE_IP", "")
        invincible_port = os.environ.get("Ironcliw_INVINCIBLE_NODE_PORT", "8000")
        if invincible_ip:
            os.environ["Ironcliw_HOLLOW_CLIENT_ACTIVE"] = "true"
            prime_url = f"http://{invincible_ip}:{invincible_port}"
            os.environ["Ironcliw_PRIME_URL"] = prime_url
            os.environ["GCP_PRIME_ENDPOINT"] = prime_url
            self.logger.info(f"[Resilience] v219.0 Re-activated hollow client: {prime_url}")

    async def _on_invincible_node_recovered(self) -> None:
        """
        Called when Invincible Node is recovered via background recovery.
        v219.0: Re-propagate hollow client env vars after recovery.
        """
        self.logger.info("[Resilience] Invincible Node (GCP VM) recovered successfully")
        self._clear_recovery_pause_deadline("invincible node")
        if self._invincible_node_probe:
            self._invincible_node_probe.reset()
        
        # v219.0: Re-activate hollow client after recovery
        invincible_ip = os.environ.get("Ironcliw_INVINCIBLE_NODE_IP", "")
        invincible_port = os.environ.get("Ironcliw_INVINCIBLE_NODE_PORT", "8000")
        if invincible_ip:
            os.environ["Ironcliw_HOLLOW_CLIENT_ACTIVE"] = "true"
            prime_url = f"http://{invincible_ip}:{invincible_port}"
            os.environ["Ironcliw_PRIME_URL"] = prime_url
            os.environ["GCP_PRIME_ENDPOINT"] = prime_url
            self.logger.info(f"[Resilience] v219.0 Hollow client re-activated after recovery: {prime_url}")

    async def _on_invincible_node_recovery_paused(self) -> None:
        """
        Called when Invincible Node recovery is paused due to max attempts.
        v219.0: Clear hollow client since node is not recoverable.
        """
        self.logger.warning("[Resilience] Invincible Node recovery paused after max attempts")
        self._set_recovery_pause_deadline("invincible node", self._invincible_node_recovery)
        
        # v219.0: Clear hollow client flag since recovery failed
        os.environ.pop("Ironcliw_HOLLOW_CLIENT_ACTIVE", None)
        self.logger.warning("[Resilience] v219.0 Cleared Ironcliw_HOLLOW_CLIENT_ACTIVE (recovery failed)")

    async def _on_llm_upgraded(self) -> None:
        """Called when LLM mode is upgraded to cloud."""
        self.logger.info("[Resilience] LLM mode upgraded to cloud (full capability)")

    async def _on_llm_downgraded(self) -> None:
        """Called when LLM mode is downgraded to local."""
        self.logger.warning("[Resilience] LLM mode downgraded to local (degraded capability)")


__all__ = [
    # Configuration
    "StartupResilienceConfig",
    # Factory functions
    "create_docker_health_check",
    "create_ollama_health_check",
    "create_invincible_node_health_check",
    "create_docker_health_probe",
    "create_ollama_health_probe",
    "create_invincible_node_health_probe",
    "create_docker_recovery",
    "create_ollama_recovery",
    "create_invincible_node_recovery",
    "create_local_llm_capability_upgrade",
    # Main coordinator
    "StartupResilience",
]
