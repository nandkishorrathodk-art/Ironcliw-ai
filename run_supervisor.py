#!/usr/bin/env python3
"""
JARVIS Supervisor Entry Point
==============================

This script runs the JARVIS Lifecycle Supervisor, which sits above the main
JARVIS application and manages its lifecycle including updates, restarts,
and rollbacks.

Usage:
    # Run supervisor (recommended way to start JARVIS)
    python run_supervisor.py

    # With custom config
    JARVIS_SUPERVISOR_CONFIG=/path/to/config.yaml python run_supervisor.py

Author: JARVIS System
Version: 1.0.0
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


def setup_logging() -> None:
    """Configure logging for the supervisor."""
    log_level = os.environ.get("JARVIS_SUPERVISOR_LOG_LEVEL", "INFO")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def print_banner() -> None:
    """Print an engaging startup banner."""
    print()
    print("\033[36m" + "=" * 65 + "\033[0m")
    print("\033[36m" + " " * 15 + "⚡ JARVIS LIFECYCLE SUPERVISOR ⚡" + " " * 15 + "\033[0m")
    print("\033[36m" + "=" * 65 + "\033[0m")
    print()
    print("  \033[33m🤖 Self-Updating • Self-Healing • Autonomous\033[0m")
    print()
    print("  \033[90mThe Living OS - Manages updates, restarts, and rollbacks")
    print("  while keeping JARVIS online and responsive.\033[0m")
    print()
    print("\033[36m" + "-" * 65 + "\033[0m")
    print()


async def main() -> None:
    """Main entry point for the supervisor."""
    from core.supervisor import JARVISSupervisor
    
    print_banner()
    
    supervisor = JARVISSupervisor()
    
    # Print configuration summary
    config = supervisor.config
    print(f"  \033[32m●\033[0m Mode:          \033[1m{config.mode.value.upper()}\033[0m")
    print(f"  \033[32m●\033[0m Update Check:  {'Enabled (' + str(config.update.check.interval_seconds) + 's)' if config.update.check.enabled else 'Disabled'}")
    print(f"  \033[32m●\033[0m Idle Updates:  {'Enabled (' + str(config.idle.threshold_seconds // 3600) + 'h threshold)' if config.idle.enabled else 'Disabled'}")
    print(f"  \033[32m●\033[0m Auto-Rollback: {'Enabled' if config.rollback.auto_on_boot_failure else 'Disabled'}")
    print(f"  \033[32m●\033[0m Max Retries:   {config.health.max_crash_retries}")
    print()
    print("\033[36m" + "-" * 65 + "\033[0m")
    print()
    print("  \033[90mStarting JARVIS Core...\033[0m")
    print()
    
    try:
        await supervisor.run()
    except KeyboardInterrupt:
        print("\n\033[33m👋 Supervisor interrupted by user\033[0m")
    except Exception as e:
        logging.error(f"Supervisor error: {e}")
        raise


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
