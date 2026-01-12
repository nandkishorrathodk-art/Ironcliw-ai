#!/usr/bin/env python3
"""
JARVIS Body Entry Point - Trinity-Integrated Computer Use Agent
================================================================

v92.0 - Unified Entry Point for Cross-Repo Orchestration

This script starts the JARVIS Body (Computer Use Agent) as part of the
Trinity ecosystem. It's designed to be called by the unified supervisor
in JARVIS-Prime via:

    python3 run_supervisor.py --unified

FEATURES:
    - Trinity Protocol integration for cross-repo communication
    - Health endpoint for supervisor monitoring
    - Graceful shutdown handling
    - Auto-registration with service mesh
    - WebSocket connection to JARVIS-Prime

TRINITY ARCHITECTURE:
    JARVIS-Prime (Mind)  <-->  JARVIS (Body)  <-->  Reactor-Core (Nerves)
         Port 8000                Port 8080              Port 8090

USAGE:
    # Direct execution (standalone)
    python3 run_jarvis.py --port 8080

    # Via unified supervisor (recommended)
    cd ../jarvis-prime && python3 run_supervisor.py --unified

ENVIRONMENT VARIABLES:
    JARVIS_PORT: Port for HTTP server (default: 8080)
    JARVIS_PRIME_URL: URL of JARVIS-Prime (default: http://localhost:8000)
    TRINITY_ENABLED: Enable Trinity Protocol (default: true)
    LOG_LEVEL: Logging level (default: INFO)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("jarvis.body")


# =============================================================================
# CONFIGURATION
# =============================================================================

class JARVISBodyConfig:
    """Configuration for JARVIS Body service."""

    def __init__(self):
        self.port = int(os.getenv("JARVIS_PORT", "8080"))
        self.host = os.getenv("JARVIS_HOST", "0.0.0.0")
        self.jarvis_prime_url = os.getenv("JARVIS_PRIME_URL", "http://localhost:8000")
        self.trinity_enabled = os.getenv("TRINITY_ENABLED", "true").lower() == "true"
        self.service_name = "jarvis"
        self.version = "v92.0"

        # Directories
        self.state_dir = Path.home() / ".jarvis" / "body_state"
        self.cross_repo_dir = Path.home() / ".jarvis" / "cross_repo"
        self.trinity_dir = Path.home() / ".jarvis" / "trinity"

        # Create directories
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.cross_repo_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HEALTH SERVER
# =============================================================================

async def create_health_server(config: JARVISBodyConfig, state: Dict[str, Any]):
    """Create HTTP server with health endpoint."""
    try:
        from aiohttp import web
        AIOHTTP_AVAILABLE = True
    except ImportError:
        logger.warning("aiohttp not available - using basic HTTP server")
        AIOHTTP_AVAILABLE = False

    if AIOHTTP_AVAILABLE:
        app = web.Application()

        async def health_handler(request):
            return web.json_response({
                "status": "healthy" if state.get("running") else "starting",
                "service": config.service_name,
                "version": config.version,
                "uptime_seconds": time.time() - state.get("start_time", time.time()),
                "trinity_connected": state.get("trinity_connected", False),
                "prime_connected": state.get("prime_connected", False),
                "timestamp": datetime.now().isoformat(),
            })

        async def status_handler(request):
            return web.json_response({
                "status": "healthy" if state.get("running") else "starting",
                "state": state,
            })

        async def metrics_handler(request):
            return web.json_response({
                "actions_executed": state.get("actions_executed", 0),
                "actions_failed": state.get("actions_failed", 0),
                "avg_action_latency_ms": state.get("avg_action_latency_ms", 0),
            })

        app.router.add_get("/health", health_handler)
        app.router.add_get("/status", status_handler)
        app.router.add_get("/metrics", metrics_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, config.host, config.port)
        await site.start()

        logger.info(f"Health server started on http://{config.host}:{config.port}")
        return runner
    else:
        # Fallback to basic socket server
        return None


# =============================================================================
# TRINITY INTEGRATION
# =============================================================================

class TrinityClient:
    """Client for Trinity Protocol communication with JARVIS-Prime."""

    def __init__(self, config: JARVISBodyConfig):
        self._config = config
        self._connected = False
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to Trinity service mesh."""
        try:
            # Register with service mesh
            registry_path = self._config.trinity_dir / "service_registry.json"

            service_info = {
                "name": self._config.service_name,
                "host": "localhost",
                "port": self._config.port,
                "capabilities": ["computer_use", "action_execution", "screen_capture"],
                "health_endpoint": "/health",
                "version": self._config.version,
                "registered_at": datetime.now().isoformat(),
            }

            # Load existing registry or create new
            registry = {"services": {}}
            if registry_path.exists():
                try:
                    with open(registry_path, "r") as f:
                        registry = json.load(f)
                except Exception:
                    pass

            registry["services"][self._config.service_name] = service_info

            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)

            logger.info("Registered with Trinity service mesh")
            self._connected = True

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Trinity: {e}")
            return False

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        heartbeat_path = self._config.trinity_dir / "heartbeats" / f"{self._config.service_name}.json"
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                heartbeat = {
                    "service": self._config.service_name,
                    "timestamp": time.time(),
                    "timestamp_iso": datetime.now().isoformat(),
                    "status": "healthy",
                    "port": self._config.port,
                }

                with open(heartbeat_path, "w") as f:
                    json.dump(heartbeat, f)

                await asyncio.sleep(5)  # Heartbeat every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def disconnect(self):
        """Disconnect from Trinity."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        self._connected = False
        logger.info("Disconnected from Trinity")


# =============================================================================
# JARVIS BODY SERVICE
# =============================================================================

class JARVISBodyService:
    """Main JARVIS Body service."""

    def __init__(self, config: JARVISBodyConfig):
        self._config = config
        self._state: Dict[str, Any] = {
            "running": False,
            "start_time": time.time(),
            "trinity_connected": False,
            "prime_connected": False,
            "actions_executed": 0,
            "actions_failed": 0,
        }
        self._trinity_client: Optional[TrinityClient] = None
        self._health_runner = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the JARVIS Body service."""
        logger.info(f"Starting JARVIS Body service {self._config.version}")

        # Start health server
        self._health_runner = await create_health_server(self._config, self._state)

        # Connect to Trinity if enabled
        if self._config.trinity_enabled:
            self._trinity_client = TrinityClient(self._config)
            connected = await self._trinity_client.connect()
            self._state["trinity_connected"] = connected

        self._state["running"] = True
        logger.info(f"JARVIS Body service started on port {self._config.port}")

        # Write state for cross-repo coordination
        await self._write_state()

    async def _write_state(self):
        """Write state for cross-repo coordination."""
        state_path = self._config.cross_repo_dir / "computer_use_state.json"
        with open(state_path, "w") as f:
            json.dump({
                **self._state,
                "port": self._config.port,
                "version": self._config.version,
                "updated_at": datetime.now().isoformat(),
            }, f, indent=2)

    async def run(self):
        """Run the service until shutdown."""
        logger.info("JARVIS Body service running. Press Ctrl+C to stop.")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def stop(self):
        """Stop the JARVIS Body service."""
        logger.info("Stopping JARVIS Body service...")

        self._state["running"] = False

        # Disconnect from Trinity
        if self._trinity_client:
            await self._trinity_client.disconnect()

        # Stop health server
        if self._health_runner:
            await self._health_runner.cleanup()

        logger.info("JARVIS Body service stopped")

    def request_shutdown(self):
        """Request service shutdown."""
        self._shutdown_event.set()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main(args: argparse.Namespace):
    """Main entry point."""
    # Create configuration
    config = JARVISBodyConfig()

    # Override from args
    if args.port:
        config.port = args.port
    if args.prime_url:
        config.jarvis_prime_url = args.prime_url

    # Create and start service
    service = JARVISBodyService(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        service.request_shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await service.start()
        await service.run()
    finally:
        await service.stop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="JARVIS Body - Computer Use Agent Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port for HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--prime-url",
        type=str,
        default=None,
        help="URL of JARVIS-Prime (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Shutdown by keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
