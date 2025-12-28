"""
Docker Daemon Manager - Production-Grade (v10.6)

Robust async Docker daemon management with intelligent startup,
parallel health checks, and comprehensive diagnostics.

Features:
- Async/await throughout
- Parallel health checks (daemon + API + containers)
- Intelligent retry with exponential backoff
- Platform-specific optimizations (macOS/Linux/Windows)
- Real-time progress reporting
- Comprehensive error handling
- No hardcoding - fully configurable via environment variables
"""

import asyncio
import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import psutil

logger = logging.getLogger(__name__)


class DaemonStatus(Enum):
    """Docker daemon status states"""
    UNKNOWN = "unknown"
    NOT_INSTALLED = "not_installed"
    INSTALLED_NOT_RUNNING = "installed_not_running"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class DockerConfig:
    """Dynamic Docker configuration - NO HARDCODING"""

    # Startup settings
    max_startup_wait_seconds: int = field(
        default_factory=lambda: int(os.getenv('DOCKER_MAX_STARTUP_WAIT', '120'))
    )
    poll_interval_seconds: float = field(
        default_factory=lambda: float(os.getenv('DOCKER_POLL_INTERVAL', '2.0'))
    )
    max_retry_attempts: int = field(
        default_factory=lambda: int(os.getenv('DOCKER_MAX_RETRIES', '3'))
    )

    # Health check settings
    enable_parallel_health_checks: bool = field(
        default_factory=lambda: os.getenv('DOCKER_PARALLEL_HEALTH', 'true').lower() == 'true'
    )
    health_check_timeout: float = field(
        default_factory=lambda: float(os.getenv('DOCKER_HEALTH_TIMEOUT', '5.0'))
    )

    # Application paths (platform-specific defaults)
    docker_app_path_macos: str = field(
        default_factory=lambda: os.getenv('DOCKER_APP_MACOS', '/Applications/Docker.app')
    )
    docker_app_path_windows: str = field(
        default_factory=lambda: os.getenv('DOCKER_APP_WINDOWS', 'Docker Desktop')
    )

    # Retry settings
    retry_backoff_base: float = field(
        default_factory=lambda: float(os.getenv('DOCKER_RETRY_BACKOFF', '1.5'))
    )
    retry_backoff_max: float = field(
        default_factory=lambda: float(os.getenv('DOCKER_RETRY_BACKOFF_MAX', '10.0'))
    )

    # Diagnostics
    enable_verbose_logging: bool = field(
        default_factory=lambda: os.getenv('DOCKER_VERBOSE', 'false').lower() == 'true'
    )


@dataclass
class DaemonHealth:
    """Docker daemon health metrics"""
    status: DaemonStatus
    daemon_responsive: bool = False
    api_accessible: bool = False
    containers_queryable: bool = False
    socket_exists: bool = False
    process_running: bool = False
    startup_time_ms: int = 0
    error_message: Optional[str] = None
    last_check_timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'daemon_responsive': self.daemon_responsive,
            'api_accessible': self.api_accessible,
            'containers_queryable': self.containers_queryable,
            'socket_exists': self.socket_exists,
            'process_running': self.process_running,
            'startup_time_ms': self.startup_time_ms,
            'error': self.error_message,
            'last_check': self.last_check_timestamp,
        }

    def is_healthy(self) -> bool:
        """Check if daemon is fully healthy"""
        return (
            self.status == DaemonStatus.RUNNING and
            self.daemon_responsive and
            self.api_accessible
        )


class DockerDaemonManager:
    """
    Production-grade Docker daemon manager

    Handles Docker Desktop/daemon lifecycle with:
    - Async startup and monitoring
    - Intelligent health checks
    - Platform-specific optimizations
    - Comprehensive error handling
    """

    def __init__(self, config: Optional[DockerConfig] = None,
                 progress_callback: Optional[Callable] = None):
        self.config = config or DockerConfig()
        self.progress_callback = progress_callback
        self.platform = platform.system().lower()

        # State
        self.health = DaemonHealth(status=DaemonStatus.UNKNOWN)
        self._startup_task: Optional[asyncio.Task] = None

        logger.info(f"Docker Daemon Manager initialized (platform: {self.platform})")

    async def check_installation(self) -> bool:
        """
        Check if Docker is installed

        Returns:
            True if Docker command is available
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                'docker', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            if proc.returncode == 0:
                version = stdout.decode().strip()
                logger.info(f"✓ Docker installed: {version}")
                return True
            else:
                logger.warning("✗ Docker command failed")
                return False

        except FileNotFoundError:
            logger.warning("✗ Docker not found in PATH")
            return False

        except asyncio.TimeoutError:
            logger.warning("✗ Docker version check timeout")
            return False

        except Exception as e:
            logger.error(f"Error checking Docker installation: {e}")
            return False

    async def check_daemon_health(self) -> DaemonHealth:
        """
        Comprehensive daemon health check with parallel checks

        Checks multiple aspects in parallel for speed:
        1. Docker socket exists
        2. Docker process running
        3. Docker daemon responsive (docker info)
        4. Docker API accessible (docker ps)

        Returns:
            DaemonHealth with comprehensive status
        """
        start_time = time.time()
        health = DaemonHealth(status=DaemonStatus.UNKNOWN)

        if self.config.enable_parallel_health_checks:
            # Run all checks in parallel for speed
            checks = await asyncio.gather(
                self._check_socket_exists(),
                self._check_process_running(),
                self._check_daemon_responsive(),
                self._check_api_accessible(),
                return_exceptions=True
            )

            health.socket_exists = checks[0] if not isinstance(checks[0], Exception) else False
            health.process_running = checks[1] if not isinstance(checks[1], Exception) else False
            health.daemon_responsive = checks[2] if not isinstance(checks[2], Exception) else False
            health.api_accessible = checks[3] if not isinstance(checks[3], Exception) else False

        else:
            # Sequential checks (fallback)
            health.socket_exists = await self._check_socket_exists()
            health.process_running = await self._check_process_running()
            health.daemon_responsive = await self._check_daemon_responsive()
            health.api_accessible = await self._check_api_accessible()

        # Determine overall status
        if health.daemon_responsive and health.api_accessible:
            health.status = DaemonStatus.RUNNING
        elif health.socket_exists or health.process_running:
            health.status = DaemonStatus.STARTING
        else:
            health.status = DaemonStatus.INSTALLED_NOT_RUNNING

        health.last_check_timestamp = time.time()
        elapsed_ms = int((time.time() - start_time) * 1000)

        if self.config.enable_verbose_logging:
            logger.debug(f"Health check completed in {elapsed_ms}ms: {health.to_dict()}")

        self.health = health
        return health

    async def _check_socket_exists(self) -> bool:
        """Check if Docker socket exists"""
        try:
            socket_paths = [
                Path('/var/run/docker.sock'),  # Linux/macOS (daemon)
                Path.home() / '.docker' / 'run' / 'docker.sock',  # macOS (Desktop)
                Path('\\\\.\\pipe\\docker_engine'),  # Windows
            ]

            for socket_path in socket_paths:
                if socket_path.exists():
                    return True

            return False

        except Exception as e:
            logger.debug(f"Error checking socket: {e}")
            return False

    async def _check_process_running(self) -> bool:
        """Check if Docker process is running"""
        try:
            if self.platform == 'darwin':  # macOS
                # Check for Docker Desktop or dockerd
                proc = await asyncio.create_subprocess_exec(
                    'pgrep', '-x', 'Docker Desktop',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await asyncio.wait_for(proc.communicate(), timeout=2.0)
                return proc.returncode == 0

            elif self.platform == 'linux':
                # Check for dockerd
                proc = await asyncio.create_subprocess_exec(
                    'pgrep', '-x', 'dockerd',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await asyncio.wait_for(proc.communicate(), timeout=2.0)
                return proc.returncode == 0

            elif self.platform == 'windows':
                # Check for Docker Desktop
                proc = await asyncio.create_subprocess_exec(
                    'tasklist', '/FI', 'IMAGENAME eq Docker Desktop.exe',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
                return b'Docker Desktop.exe' in stdout

            return False

        except Exception as e:
            logger.debug(f"Error checking process: {e}")
            return False

    async def _check_daemon_responsive(self) -> bool:
        """Check if daemon responds to 'docker info'"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'docker', 'info',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.health_check_timeout
            )

            return proc.returncode == 0

        except asyncio.TimeoutError:
            return False

        except Exception as e:
            logger.debug(f"Error checking daemon: {e}")
            return False

    async def _check_api_accessible(self) -> bool:
        """Check if Docker API is accessible via 'docker ps'"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'docker', 'ps', '--format', '{{.ID}}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.health_check_timeout
            )

            return proc.returncode == 0

        except asyncio.TimeoutError:
            return False

        except Exception as e:
            logger.debug(f"Error checking API: {e}")
            return False

    async def start_daemon(self) -> bool:
        """
        Start Docker daemon/Desktop with intelligent retry

        Returns:
            True if daemon started successfully
        """
        if not await self.check_installation():
            self.health.status = DaemonStatus.NOT_INSTALLED
            self.health.error_message = "Docker not installed"
            return False

        # Check if already running
        health = await self.check_daemon_health()
        if health.is_healthy():
            logger.info("✓ Docker daemon already running")
            return True

        logger.info("🐳 Starting Docker daemon...")
        self._report_progress("Starting Docker daemon...")

        # Try to start with retries
        for attempt in range(1, self.config.max_retry_attempts + 1):
            logger.info(f"→ Start attempt {attempt}/{self.config.max_retry_attempts}")
            self._report_progress(f"Start attempt {attempt}/{self.config.max_retry_attempts}")

            # Launch Docker
            if await self._launch_docker_app():
                # Wait for daemon to become ready
                logger.info(f"→ Waiting for daemon (up to {self.config.max_startup_wait_seconds}s)...")
                self._report_progress(f"Waiting for daemon...")

                if await self._wait_for_daemon_ready():
                    logger.info("✓ Docker daemon started successfully!")
                    return True

                logger.warning(f"✗ Daemon did not become ready (attempt {attempt})")

            # Exponential backoff between retries
            if attempt < self.config.max_retry_attempts:
                backoff = min(
                    self.config.retry_backoff_base ** attempt,
                    self.config.retry_backoff_max
                )
                logger.info(f"⏱️  Waiting {backoff:.1f}s before retry...")
                await asyncio.sleep(backoff)

        logger.error(f"✗ Failed to start Docker daemon after {self.config.max_retry_attempts} attempts")
        self.health.error_message = "Failed to start after multiple attempts"
        return False

    async def _launch_docker_app(self) -> bool:
        """
        Launch Docker Desktop application

        Returns:
            True if launch command succeeded
        """
        try:
            if self.platform == 'darwin':  # macOS
                app_path = self.config.docker_app_path_macos

                if not Path(app_path).exists():
                    logger.error(f"✗ Docker.app not found at {app_path}")
                    return False

                proc = await asyncio.create_subprocess_exec(
                    'open', '-a', app_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                await asyncio.wait_for(proc.communicate(), timeout=10.0)
                return proc.returncode == 0

            elif self.platform == 'linux':
                # Try systemd first
                proc = await asyncio.create_subprocess_exec(
                    'sudo', 'systemctl', 'start', 'docker',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                await asyncio.wait_for(proc.communicate(), timeout=10.0)
                return proc.returncode == 0

            elif self.platform == 'windows':
                proc = await asyncio.create_subprocess_exec(
                    'cmd', '/c', 'start', '', self.config.docker_app_path_windows,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                await asyncio.wait_for(proc.communicate(), timeout=10.0)
                return proc.returncode == 0

            return False

        except Exception as e:
            logger.error(f"Error launching Docker: {e}")
            return False

    async def _wait_for_daemon_ready(self) -> bool:
        """
        Wait for daemon to become fully ready

        Uses intelligent polling with health checks

        Returns:
            True if daemon became ready within timeout
        """
        start_time = time.time()
        check_count = 0

        while (time.time() - start_time) < self.config.max_startup_wait_seconds:
            check_count += 1

            # Check daemon health
            health = await self.check_daemon_health()

            if health.is_healthy():
                elapsed = time.time() - start_time
                self.health.startup_time_ms = int(elapsed * 1000)
                logger.info(f"✓ Daemon ready in {elapsed:.1f}s")
                return True

            # Progress reporting
            if check_count % 5 == 0:
                elapsed = time.time() - start_time
                self._report_progress(f"Still waiting ({elapsed:.0f}s)...")
                logger.info(f"  ...waiting ({elapsed:.0f}s elapsed)")

            # Adaptive polling
            await asyncio.sleep(self.config.poll_interval_seconds)

        logger.warning(f"✗ Timeout waiting for daemon ({self.config.max_startup_wait_seconds}s)")
        return False

    def _report_progress(self, message: str):
        """Report progress via callback"""
        if self.progress_callback:
            try:
                self.progress_callback(message)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

    async def stop_daemon(self) -> bool:
        """
        Stop Docker daemon/Desktop gracefully

        Returns:
            True if stopped successfully
        """
        logger.info("Stopping Docker daemon...")

        try:
            if self.platform == 'darwin':
                # Quit Docker Desktop on macOS
                proc = await asyncio.create_subprocess_exec(
                    'osascript', '-e', 'quit app "Docker"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                await asyncio.wait_for(proc.communicate(), timeout=10.0)
                return proc.returncode == 0

            elif self.platform == 'linux':
                # Stop via systemd
                proc = await asyncio.create_subprocess_exec(
                    'sudo', 'systemctl', 'stop', 'docker',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                await asyncio.wait_for(proc.communicate(), timeout=10.0)
                return proc.returncode == 0

            return False

        except Exception as e:
            logger.error(f"Error stopping Docker: {e}")
            return False

    def get_health(self) -> DaemonHealth:
        """Get current daemon health"""
        return self.health

    async def ensure_daemon_running(
        self,
        auto_start: bool = True,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ) -> dict:
        """
        Ensure Docker daemon is running, with optional auto-start.

        This is the main entry point for Docker daemon management,
        providing backward compatibility with start_system.py.

        Args:
            auto_start: Whether to automatically start Docker if not running
            timeout: Maximum time to wait for daemon startup (overrides config)
            max_retries: Maximum number of start attempts (overrides config)

        Returns:
            dict: Status information including:
                - installed: bool
                - daemon_running: bool
                - version: str or None
                - started_automatically: bool
                - startup_time_ms: int or None
                - error: str or None
                - platform: str
        """
        # Override config if parameters provided
        if timeout is not None:
            self.config.max_startup_wait_seconds = int(timeout)
        if max_retries is not None:
            self.config.max_retry_attempts = max_retries

        status = {
            "installed": False,
            "daemon_running": False,
            "version": None,
            "started_automatically": False,
            "startup_time_ms": None,
            "error": None,
            "platform": self.platform
        }

        logger.info("🐳 Docker Daemon Status Check")
        self._report_progress("Checking Docker installation...")

        # Step 1: Check if Docker is installed
        if not await self.check_installation():
            status["error"] = "Docker not installed"
            logger.warning("✗ Docker not installed")
            logger.info("  Install Docker Desktop: https://www.docker.com/products/docker-desktop")
            return status

        status["installed"] = True
        logger.info("✓ Docker installed")

        # Get version
        try:
            proc = await asyncio.create_subprocess_exec(
                'docker', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            if proc.returncode == 0:
                status["version"] = stdout.decode().strip()
        except Exception:
            pass

        # Step 2: Check if daemon is already running
        health = await self.check_daemon_health()
        if health.is_healthy():
            status["daemon_running"] = True
            logger.info("✓ Docker daemon is running")
            return status

        # Step 3: Daemon not running - attempt auto-start if enabled
        if not auto_start:
            status["error"] = "Docker daemon not running"
            logger.warning("✗ Docker daemon not running (auto-start disabled)")
            return status

        logger.info("⚠️  Docker daemon not running - attempting auto-start")
        self._report_progress("Docker daemon not running - starting...")

        # Use the enhanced start_daemon() method with built-in retry
        if await self.start_daemon():
            status["daemon_running"] = True
            status["started_automatically"] = True
            status["startup_time_ms"] = self.health.startup_time_ms
            logger.info(f"✓ Docker daemon started ({self.health.startup_time_ms}ms)")
            return status
        else:
            status["error"] = self.health.error_message or "Failed to start Docker daemon"
            logger.error(f"✗ Failed to start Docker daemon: {status['error']}")
            return status

    def get_status(self) -> dict:
        """
        Get current daemon status without performing checks.

        Returns:
            dict: Status dictionary with current health state
        """
        health = self.health
        return {
            "installed": health.status != DaemonStatus.NOT_INSTALLED,
            "daemon_running": health.is_healthy(),
            "version": None,  # Would need to cache from ensure_daemon_running
            "started_automatically": False,  # Would need to track
            "startup_time_ms": health.startup_time_ms,
            "error": health.error_message,
            "platform": self.platform
        }

    def get_status_emoji(self) -> str:
        """
        Get a formatted status string with emoji.

        Returns:
            str: Colored status string with emoji
        """
        health = self.health

        if health.is_healthy():
            return f"✓ Docker: Running"
        elif health.status == DaemonStatus.NOT_INSTALLED:
            return f"✗ Docker: Not installed"
        else:
            error = health.error_message or "Not running"
            return f"✗ Docker: {error}"


# Factory function
async def create_docker_manager(
    config: Optional[DockerConfig] = None,
    progress_callback: Optional[Callable] = None
) -> DockerDaemonManager:
    """
    Create and initialize Docker daemon manager

    Args:
        config: Optional configuration (uses environment if not provided)
        progress_callback: Optional callback for progress updates

    Returns:
        Initialized DockerDaemonManager
    """
    manager = DockerDaemonManager(config, progress_callback)

    # Initial health check
    await manager.check_daemon_health()

    return manager
