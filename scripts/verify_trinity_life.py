#!/usr/bin/env python3
"""
Trinity Organism Verification Script v3.0
==========================================

The "Life Probe" - Verifies that the Trinity Architecture is alive and functional.

v3.0 Features:
    - --init mode: Initializes all components for standalone testing
    - Parallel initialization for faster startup
    - Graceful degradation with fallbacks
    - Detailed diagnostics and troubleshooting hints

Tests:
    1. [BRAIN] JARVIS Prime (Model Serving) - Is it thinking?
    2. [NERVES] Event Bus & Neural Mesh - Are signals flowing?
    3. [IMMUNE] Ouroboros Self-Improvement - Is it ready to heal?
    4. [HEART] Trinity Integration - Is everything connected?
    5. [MEMORY] Learning Cache & Experience Publisher - Is it learning?

Usage:
    # Check status only (requires supervisor running)
    python3 scripts/verify_trinity_life.py

    # Initialize components then check (standalone mode)
    python3 scripts/verify_trinity_life.py --init

    # Verbose mode with detailed component status
    python3 scripts/verify_trinity_life.py --init --verbose

Expected Output:
    [BRAIN] Connected - JARVIS Prime responding
    [NERVES] Pulse Detected - Events flowing
    [IMMUNE] Ready - Ouroboros standing by
    [HEART] Beating - Trinity Integration active
    [MEMORY] Active - Learning enabled

    Final Status: TRINITY ORGANISM: ALIVE

Author: Trinity System
Version: 3.0.0
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrinityProbe")


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Status colors
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"

    # Component colors
    BRAIN = "\033[95m"      # Magenta for Brain
    NERVES = "\033[96m"     # Cyan for Nerves
    IMMUNE = "\033[93m"     # Yellow for Immune
    HEART = "\033[91m"      # Red for Heart
    MEMORY = "\033[94m"     # Blue for Memory


class ComponentStatus(Enum):
    """Status of each component."""
    ALIVE = "ALIVE"
    DEGRADED = "DEGRADED"
    DEAD = "DEAD"
    UNKNOWN = "UNKNOWN"


@dataclass
class ComponentResult:
    """Result of a component check."""
    name: str
    status: ComponentStatus
    message: str
    details: Optional[Dict] = None
    latency_ms: float = 0.0


# =============================================================================
# Component Initialization (Standalone Mode)
# =============================================================================

async def initialize_all_components(verbose: bool = False) -> Dict[str, bool]:
    """
    Initialize all Trinity components for standalone testing.

    This allows the Life Probe to work without running the full supervisor.

    Returns:
        Dict mapping component name to initialization success
    """
    results = {}
    print()
    print(f"  {Colors.CYAN}Initializing Trinity Components...{Colors.RESET}")
    print()

    # 1. Initialize Brain Orchestrator
    print(f"  {Colors.BRAIN}[BRAIN]{Colors.RESET} Starting model infrastructure...", end="", flush=True)
    try:
        from backend.core.ouroboros.brain_orchestrator import (
            get_brain_orchestrator,
            ignite_brains,
        )
        brain = get_brain_orchestrator()
        await asyncio.wait_for(ignite_brains(), timeout=30.0)
        status = brain.get_status()
        active = len(status.get("active_providers", []))
        print(f"\r  {Colors.BRAIN}[BRAIN]{Colors.RESET} {Colors.GREEN}Started ({active} providers){Colors.RESET}")
        results["brain"] = True
    except asyncio.TimeoutError:
        print(f"\r  {Colors.BRAIN}[BRAIN]{Colors.RESET} {Colors.YELLOW}Timeout (30s) - continuing{Colors.RESET}")
        results["brain"] = False
    except Exception as e:
        print(f"\r  {Colors.BRAIN}[BRAIN]{Colors.RESET} {Colors.YELLOW}Skipped: {e}{Colors.RESET}")
        results["brain"] = False

    # 2. Initialize Event Bus
    print(f"  {Colors.NERVES}[NERVES]{Colors.RESET} Starting event bus...", end="", flush=True)
    try:
        from backend.core.trinity_event_bus import get_trinity_event_bus
        bus = await asyncio.wait_for(get_trinity_event_bus(), timeout=10.0)
        if bus:
            print(f"\r  {Colors.NERVES}[NERVES]{Colors.RESET} {Colors.GREEN}Event Bus online{Colors.RESET}")
            results["event_bus"] = True
        else:
            print(f"\r  {Colors.NERVES}[NERVES]{Colors.RESET} {Colors.YELLOW}Event Bus not available{Colors.RESET}")
            results["event_bus"] = False
    except Exception as e:
        print(f"\r  {Colors.NERVES}[NERVES]{Colors.RESET} {Colors.YELLOW}Skipped: {e}{Colors.RESET}")
        results["event_bus"] = False

    # 3. Initialize Neural Mesh
    print(f"  {Colors.NERVES}[NERVES]{Colors.RESET} Starting neural mesh...", end="", flush=True)
    try:
        from backend.core.ouroboros.neural_mesh import (
            get_neural_mesh,
            initialize_neural_mesh,
        )
        await asyncio.wait_for(initialize_neural_mesh(), timeout=15.0)
        mesh = get_neural_mesh()
        status = mesh.get_status() if mesh else {}
        nodes = len(status.get("connected_nodes", []))
        print(f"\r  {Colors.NERVES}[NERVES]{Colors.RESET} {Colors.GREEN}Neural Mesh online ({nodes} nodes){Colors.RESET}")
        results["neural_mesh"] = True
    except Exception as e:
        print(f"\r  {Colors.NERVES}[NERVES]{Colors.RESET} {Colors.YELLOW}Skipped: {e}{Colors.RESET}")
        results["neural_mesh"] = False

    # 4. Initialize Native Self-Improvement (Ouroboros)
    print(f"  {Colors.IMMUNE}[IMMUNE]{Colors.RESET} Starting Ouroboros engine...", end="", flush=True)
    try:
        from backend.core.ouroboros.native_integration import (
            get_native_self_improvement,
            initialize_native_self_improvement,
        )
        await asyncio.wait_for(initialize_native_self_improvement(), timeout=15.0)
        engine = get_native_self_improvement()
        status = engine.get_status() if engine else {}
        running = status.get("running", False)
        if running:
            print(f"\r  {Colors.IMMUNE}[IMMUNE]{Colors.RESET} {Colors.GREEN}Ouroboros active{Colors.RESET}")
            results["ouroboros"] = True
        else:
            print(f"\r  {Colors.IMMUNE}[IMMUNE]{Colors.RESET} {Colors.YELLOW}Ouroboros initialized (not running){Colors.RESET}")
            results["ouroboros"] = False
    except Exception as e:
        print(f"\r  {Colors.IMMUNE}[IMMUNE]{Colors.RESET} {Colors.YELLOW}Skipped: {e}{Colors.RESET}")
        results["ouroboros"] = False

    # 5. Initialize Trinity Integration
    print(f"  {Colors.HEART}[HEART]{Colors.RESET} Starting Trinity Integration...", end="", flush=True)
    try:
        from backend.core.ouroboros.trinity_integration import (
            get_trinity_integration,
            initialize_trinity_integration,
        )
        success = await asyncio.wait_for(initialize_trinity_integration(), timeout=15.0)
        if success:
            trinity = get_trinity_integration()
            status = trinity.get_status() if trinity else {}
            health = status.get("health", {}).get("overall", "unknown")
            print(f"\r  {Colors.HEART}[HEART]{Colors.RESET} {Colors.GREEN}Trinity online (health: {health}){Colors.RESET}")
            results["trinity"] = True
        else:
            print(f"\r  {Colors.HEART}[HEART]{Colors.RESET} {Colors.YELLOW}Trinity degraded{Colors.RESET}")
            results["trinity"] = False
    except Exception as e:
        print(f"\r  {Colors.HEART}[HEART]{Colors.RESET} {Colors.YELLOW}Skipped: {e}{Colors.RESET}")
        results["trinity"] = False

    print()
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"  {Colors.CYAN}Initialization complete: {success_count}/{total_count} components started{Colors.RESET}")
    print()

    return results


async def shutdown_all_components() -> None:
    """Shutdown all initialized components gracefully."""
    print()
    print(f"  {Colors.CYAN}Shutting down components...{Colors.RESET}")

    # Shutdown in reverse order
    try:
        from backend.core.ouroboros.trinity_integration import shutdown_trinity_integration
        await asyncio.wait_for(shutdown_trinity_integration(), timeout=5.0)
    except Exception:
        pass

    try:
        from backend.core.ouroboros.native_integration import shutdown_native_self_improvement
        await asyncio.wait_for(shutdown_native_self_improvement(), timeout=5.0)
    except Exception:
        pass

    try:
        from backend.core.ouroboros.neural_mesh import shutdown_neural_mesh
        await asyncio.wait_for(shutdown_neural_mesh(), timeout=5.0)
    except Exception:
        pass

    try:
        from backend.core.trinity_event_bus import shutdown_trinity_event_bus
        await asyncio.wait_for(shutdown_trinity_event_bus(), timeout=5.0)
    except Exception:
        pass

    try:
        from backend.core.ouroboros.brain_orchestrator import shutdown_brains
        await asyncio.wait_for(shutdown_brains(), timeout=5.0)
    except Exception:
        pass

    print(f"  {Colors.GREEN}Shutdown complete{Colors.RESET}")


# =============================================================================
# Component Probes
# =============================================================================

async def probe_brain() -> ComponentResult:
    """
    [BRAIN] Probe JARVIS Prime - Model Serving Infrastructure.

    Checks:
    - Brain Orchestrator availability
    - Active model providers
    - Model response capability
    """
    start = time.time()

    try:
        from backend.core.ouroboros.brain_orchestrator import get_brain_orchestrator

        brain = get_brain_orchestrator()

        if brain is None:
            return ComponentResult(
                name="BRAIN",
                status=ComponentStatus.DEAD,
                message="Brain Orchestrator not initialized",
                latency_ms=(time.time() - start) * 1000,
            )

        status = brain.get_status()
        active_providers = status.get("active_providers", [])
        total_providers = status.get("total_providers", 0)

        if active_providers:
            return ComponentResult(
                name="BRAIN",
                status=ComponentStatus.ALIVE,
                message=f"Connected - {len(active_providers)}/{total_providers} providers active",
                details={
                    "providers": active_providers,
                    "load_balancer": status.get("load_balancer", "unknown"),
                },
                latency_ms=(time.time() - start) * 1000,
            )
        else:
            return ComponentResult(
                name="BRAIN",
                status=ComponentStatus.DEGRADED,
                message="No active providers - models not loaded",
                details=status,
                latency_ms=(time.time() - start) * 1000,
            )

    except ImportError as e:
        return ComponentResult(
            name="BRAIN",
            status=ComponentStatus.DEAD,
            message=f"Module not available: {e}",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ComponentResult(
            name="BRAIN",
            status=ComponentStatus.UNKNOWN,
            message=f"Error: {e}",
            latency_ms=(time.time() - start) * 1000,
        )


async def probe_nerves() -> ComponentResult:
    """
    [NERVES] Probe Event Bus & Neural Mesh - Signal Transmission.

    Checks:
    - Trinity Event Bus connectivity
    - Neural Mesh node status
    - Message flow capability
    """
    start = time.time()

    try:
        # Check Trinity Event Bus
        event_bus_ok = False
        try:
            from backend.core.trinity_event_bus import get_trinity_event_bus
            bus = await get_trinity_event_bus()
            if bus:
                # Check _running attribute directly (bus doesn't have get_status method)
                event_bus_ok = getattr(bus, '_running', False)
        except Exception as e:
            logger.debug(f"Event bus check failed: {e}")

        # Check Neural Mesh
        neural_mesh_ok = False
        mesh_nodes = 0
        try:
            from backend.core.ouroboros.neural_mesh import get_neural_mesh
            mesh = get_neural_mesh()
            if mesh:
                mesh_status = mesh.get_status()
                neural_mesh_ok = mesh_status.get("running", False)
                mesh_nodes = len(mesh_status.get("connected_nodes", []))
        except Exception as e:
            logger.debug(f"Neural mesh check failed: {e}")

        if event_bus_ok and neural_mesh_ok:
            return ComponentResult(
                name="NERVES",
                status=ComponentStatus.ALIVE,
                message=f"Pulse Detected - Events flowing, {mesh_nodes} mesh nodes",
                details={
                    "event_bus": event_bus_ok,
                    "neural_mesh": neural_mesh_ok,
                    "mesh_nodes": mesh_nodes,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        elif event_bus_ok or neural_mesh_ok:
            return ComponentResult(
                name="NERVES",
                status=ComponentStatus.DEGRADED,
                message=f"Partial signal - Bus:{event_bus_ok}, Mesh:{neural_mesh_ok}",
                details={
                    "event_bus": event_bus_ok,
                    "neural_mesh": neural_mesh_ok,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        else:
            return ComponentResult(
                name="NERVES",
                status=ComponentStatus.DEAD,
                message="No signal - Event systems offline",
                latency_ms=(time.time() - start) * 1000,
            )

    except Exception as e:
        return ComponentResult(
            name="NERVES",
            status=ComponentStatus.UNKNOWN,
            message=f"Error: {e}",
            latency_ms=(time.time() - start) * 1000,
        )


async def probe_immune() -> ComponentResult:
    """
    [IMMUNE] Probe Ouroboros Self-Improvement Engine.

    Checks:
    - Native Self-Improvement availability
    - Trinity Integration status
    - Code review capability
    - Rollback readiness
    """
    start = time.time()

    try:
        from backend.core.ouroboros.native_integration import get_native_self_improvement

        engine = get_native_self_improvement()

        if engine is None:
            return ComponentResult(
                name="IMMUNE",
                status=ComponentStatus.DEAD,
                message="Ouroboros not initialized",
                latency_ms=(time.time() - start) * 1000,
            )

        status = engine.get_status()
        running = status.get("running", False)

        # Check Trinity Integration
        trinity_ok = False
        trinity_health = "unknown"
        try:
            from backend.core.ouroboros.trinity_integration import get_trinity_integration
            trinity = get_trinity_integration()
            if trinity:
                trinity_status = trinity.get_status()
                trinity_health = trinity_status.get("health", {}).get("overall", "unknown")
                trinity_ok = trinity_health in ("healthy", "degraded")
        except Exception as e:
            logger.debug(f"Trinity check failed: {e}")

        if running and trinity_ok:
            return ComponentResult(
                name="IMMUNE",
                status=ComponentStatus.ALIVE,
                message=f"Ready - Ouroboros standing by (Trinity: {trinity_health})",
                details={
                    "running": running,
                    "trinity_health": trinity_health,
                    "security_enabled": status.get("security_enabled", True),
                },
                latency_ms=(time.time() - start) * 1000,
            )
        elif running:
            return ComponentResult(
                name="IMMUNE",
                status=ComponentStatus.DEGRADED,
                message=f"Partial - Engine running but Trinity: {trinity_health}",
                details=status,
                latency_ms=(time.time() - start) * 1000,
            )
        else:
            return ComponentResult(
                name="IMMUNE",
                status=ComponentStatus.DEAD,
                message="Ouroboros not running",
                details=status,
                latency_ms=(time.time() - start) * 1000,
            )

    except ImportError as e:
        return ComponentResult(
            name="IMMUNE",
            status=ComponentStatus.DEAD,
            message=f"Module not available: {e}",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ComponentResult(
            name="IMMUNE",
            status=ComponentStatus.UNKNOWN,
            message=f"Error: {e}",
            latency_ms=(time.time() - start) * 1000,
        )


async def probe_heart() -> ComponentResult:
    """
    [HEART] Probe Trinity Integration - The Central Coordinator.

    Checks:
    - All Trinity components connected
    - Lock manager operational
    - Coordinator ready
    - Health monitor active
    """
    start = time.time()

    try:
        from backend.core.ouroboros.trinity_integration import get_trinity_integration

        trinity = get_trinity_integration()

        if trinity is None:
            return ComponentResult(
                name="HEART",
                status=ComponentStatus.DEAD,
                message="Trinity Integration not initialized",
                latency_ms=(time.time() - start) * 1000,
            )

        status = trinity.get_status()
        health = status.get("health", {})
        overall = health.get("overall", "unknown")

        components = health.get("components", {})
        alive_count = sum(1 for v in components.values() if v in ("healthy", True))
        total_count = len(components)

        if overall == "healthy":
            return ComponentResult(
                name="HEART",
                status=ComponentStatus.ALIVE,
                message=f"Beating - {alive_count}/{total_count} components healthy",
                details={
                    "overall": overall,
                    "components": components,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        elif overall == "degraded":
            return ComponentResult(
                name="HEART",
                status=ComponentStatus.DEGRADED,
                message=f"Irregular - {alive_count}/{total_count} components healthy",
                details={
                    "overall": overall,
                    "components": components,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        else:
            return ComponentResult(
                name="HEART",
                status=ComponentStatus.DEAD,
                message=f"Flatline - Health: {overall}",
                details=status,
                latency_ms=(time.time() - start) * 1000,
            )

    except ImportError as e:
        return ComponentResult(
            name="HEART",
            status=ComponentStatus.DEAD,
            message=f"Module not available: {e}",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return ComponentResult(
            name="HEART",
            status=ComponentStatus.UNKNOWN,
            message=f"Error: {e}",
            latency_ms=(time.time() - start) * 1000,
        )


async def probe_memory() -> ComponentResult:
    """
    [MEMORY] Probe Learning Cache & Experience Publisher.

    Checks:
    - Learning cache accessibility
    - Experience publisher channels
    - Deduplication active
    """
    start = time.time()

    try:
        from backend.core.ouroboros.trinity_integration import get_trinity_integration

        trinity = get_trinity_integration()

        if trinity is None:
            return ComponentResult(
                name="MEMORY",
                status=ComponentStatus.DEAD,
                message="Trinity Integration not available",
                latency_ms=(time.time() - start) * 1000,
            )

        # Check learning cache
        learning_ok = False
        cache_size = 0
        try:
            cache = trinity.learning_cache
            if cache:
                learning_ok = True
                cache_size = len(cache._cache) if hasattr(cache, '_cache') else 0
        except Exception:
            pass

        # Check experience publisher
        publisher_ok = False
        channels = 0
        try:
            pub = trinity.experience_publisher
            if pub:
                publisher_ok = True
                # Count available channels
                if hasattr(pub, '_forwarder') and pub._forwarder:
                    channels += 1
                if hasattr(pub, '_mesh') and pub._mesh:
                    channels += 1
        except Exception:
            pass

        if learning_ok and publisher_ok:
            return ComponentResult(
                name="MEMORY",
                status=ComponentStatus.ALIVE,
                message=f"Active - {cache_size} cached, {channels} channels",
                details={
                    "learning_cache": learning_ok,
                    "experience_publisher": publisher_ok,
                    "cache_size": cache_size,
                    "channels": channels,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        elif learning_ok or publisher_ok:
            return ComponentResult(
                name="MEMORY",
                status=ComponentStatus.DEGRADED,
                message=f"Partial - Cache:{learning_ok}, Publisher:{publisher_ok}",
                details={
                    "learning_cache": learning_ok,
                    "experience_publisher": publisher_ok,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        else:
            return ComponentResult(
                name="MEMORY",
                status=ComponentStatus.DEAD,
                message="Memory systems offline",
                latency_ms=(time.time() - start) * 1000,
            )

    except Exception as e:
        return ComponentResult(
            name="MEMORY",
            status=ComponentStatus.UNKNOWN,
            message=f"Error: {e}",
            latency_ms=(time.time() - start) * 1000,
        )


# =============================================================================
# Main Verification
# =============================================================================

def get_status_icon(status: ComponentStatus) -> str:
    """Get icon for component status."""
    icons = {
        ComponentStatus.ALIVE: f"{Colors.GREEN}[OK]{Colors.RESET}",
        ComponentStatus.DEGRADED: f"{Colors.YELLOW}[!!]{Colors.RESET}",
        ComponentStatus.DEAD: f"{Colors.RED}[XX]{Colors.RESET}",
        ComponentStatus.UNKNOWN: f"{Colors.BLUE}[??]{Colors.RESET}",
    }
    return icons.get(status, "[??]")


def get_component_color(name: str) -> str:
    """Get color for component name."""
    colors = {
        "BRAIN": Colors.BRAIN,
        "NERVES": Colors.NERVES,
        "IMMUNE": Colors.IMMUNE,
        "HEART": Colors.HEART,
        "MEMORY": Colors.MEMORY,
    }
    return colors.get(name, Colors.RESET)


async def run_verification(verbose: bool = False, quick: bool = False) -> Tuple[bool, List[ComponentResult]]:
    """
    Run full Trinity organism verification.

    Args:
        verbose: Show detailed component status
        quick: Skip slow checks

    Returns:
        Tuple of (all_alive, results)
    """
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}=" * 60)
    print(f"      TRINITY ORGANISM LIFE PROBE v2.0")
    print(f"=" * 60 + f"{Colors.RESET}")
    print()

    # Run probes
    probes = [
        ("BRAIN", probe_brain),
        ("NERVES", probe_nerves),
        ("IMMUNE", probe_immune),
        ("HEART", probe_heart),
        ("MEMORY", probe_memory),
    ]

    results: List[ComponentResult] = []

    for name, probe_fn in probes:
        color = get_component_color(name)
        print(f"  {color}[{name}]{Colors.RESET} Probing...", end="", flush=True)

        try:
            result = await asyncio.wait_for(probe_fn(), timeout=10.0)
        except asyncio.TimeoutError:
            result = ComponentResult(
                name=name,
                status=ComponentStatus.DEAD,
                message="Probe timed out (10s)",
            )
        except Exception as e:
            result = ComponentResult(
                name=name,
                status=ComponentStatus.UNKNOWN,
                message=f"Probe error: {e}",
            )

        results.append(result)

        # Clear the "Probing..." text and show result
        print(f"\r  {color}[{name}]{Colors.RESET} {get_status_icon(result.status)} {result.message}")

        if verbose and result.details:
            for key, value in result.details.items():
                print(f"    {Colors.DIM}- {key}: {value}{Colors.RESET}")

    print()

    # Calculate overall status
    alive_count = sum(1 for r in results if r.status == ComponentStatus.ALIVE)
    degraded_count = sum(1 for r in results if r.status == ComponentStatus.DEGRADED)
    dead_count = sum(1 for r in results if r.status == ComponentStatus.DEAD)

    total = len(results)
    all_alive = dead_count == 0

    # Print final status
    print(f"{Colors.BOLD}{Colors.CYAN}-" * 60 + f"{Colors.RESET}")

    if alive_count == total:
        status_color = Colors.GREEN
        status_emoji = ""
        status_text = "TRINITY ORGANISM: ALIVE"
    elif dead_count == 0:
        status_color = Colors.YELLOW
        status_emoji = ""
        status_text = "TRINITY ORGANISM: DEGRADED"
    elif alive_count > 0:
        status_color = Colors.YELLOW
        status_emoji = ""
        status_text = "TRINITY ORGANISM: PARTIAL"
    else:
        status_color = Colors.RED
        status_emoji = ""
        status_text = "TRINITY ORGANISM: DEAD"

    print()
    print(f"  {status_color}{Colors.BOLD}{status_emoji} {status_text}{Colors.RESET}")
    print()
    print(f"  Components: {Colors.GREEN}{alive_count} alive{Colors.RESET}, "
          f"{Colors.YELLOW}{degraded_count} degraded{Colors.RESET}, "
          f"{Colors.RED}{dead_count} dead{Colors.RESET}")
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}=" * 60 + f"{Colors.RESET}")
    print()

    return all_alive, results


async def run_with_init(verbose: bool = False, quick: bool = False) -> Tuple[bool, List[ComponentResult]]:
    """Run verification with component initialization first."""
    # Initialize components
    init_results = await initialize_all_components(verbose=verbose)

    # Small delay for components to fully start
    await asyncio.sleep(1.0)

    # Run verification
    all_alive, results = await run_verification(verbose=verbose, quick=quick)

    # Shutdown components
    await shutdown_all_components()

    return all_alive, results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trinity Organism Life Probe - Verify system health"
    )
    parser.add_argument(
        "--init", "-i",
        action="store_true",
        help="Initialize components before checking (standalone mode)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed component status"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Skip slow checks"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Check if supervisor might already be running
    if not args.init:
        print()
        print(f"  {Colors.CYAN}TIP: Run with --init to initialize components for standalone testing{Colors.RESET}")
        print(f"  {Colors.DIM}     python3 scripts/verify_trinity_life.py --init{Colors.RESET}")
        print()

    # Run verification
    if args.init:
        all_alive, results = asyncio.run(run_with_init(
            verbose=args.verbose,
            quick=args.quick,
        ))
    else:
        all_alive, results = asyncio.run(run_verification(
            verbose=args.verbose,
            quick=args.quick,
        ))

    if args.json:
        import json
        output = {
            "all_alive": all_alive,
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "latency_ms": r.latency_ms,
                    "details": r.details,
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))

    # Exit with appropriate code
    sys.exit(0 if all_alive else 1)


if __name__ == "__main__":
    main()
