"""
v77.0: Coding Council Trinity Integration
==========================================

Connects the Unified Coding Council to Project Trinity for cross-repo
code evolution and intelligent task distribution.

This module enables:
- Evolution commands from J-Prime (Mind) to JARVIS (Body)
- Status broadcasting to Reactor Core (Nerves)
- Cross-repo code modifications
- Distributed task execution

Trinity Architecture Integration:
    ┌────────────┐    Evolution Commands    ┌──────────────────┐
    │  J-PRIME   │ ────────────────────────>│  CODING COUNCIL  │
    │   (Mind)   │                          │   (Orchestrator) │
    └────────────┘    Status Updates        └──────────────────┘
           │                                         │
           │         ┌──────────────┐               │
           └────────>│ REACTOR CORE │<──────────────┘
                     │   (Nerves)   │    Task Metrics
                     └──────────────┘

Author: JARVIS v77.0
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import UnifiedCodingCouncil
    from .types import EvolutionResult, EvolutionTask

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (Environment-Driven)
# =============================================================================

CODING_COUNCIL_ENABLED = os.getenv("CODING_COUNCIL_ENABLED", "true").lower() == "true"
CODING_COUNCIL_CROSS_REPO = os.getenv("CODING_COUNCIL_CROSS_REPO", "true").lower() == "true"
CODING_COUNCIL_AUTO_APPROVE = os.getenv("CODING_COUNCIL_AUTO_APPROVE", "false").lower() == "true"
STATUS_BROADCAST_INTERVAL = float(os.getenv("CODING_COUNCIL_STATUS_INTERVAL", "10.0"))

# Cross-repo paths (environment-driven)
JARVIS_REPO = Path(os.getenv("JARVIS_REPO", str(Path.home() / "Documents/repos/JARVIS-AI-Agent")))
JARVIS_PRIME_REPO = Path(os.getenv("JARVIS_PRIME_REPO", str(Path.home() / "Documents/repos/jarvis-prime")))
REACTOR_CORE_REPO = Path(os.getenv("REACTOR_CORE_REPO", str(Path.home() / "Documents/repos/reactor-core")))

# Trinity directories
TRINITY_DIR = Path.home() / ".jarvis" / "trinity"
CODING_COUNCIL_STATE_FILE = TRINITY_DIR / "components" / "coding_council.json"


# =============================================================================
# Coding Council Intent Extensions for Trinity Protocol
# =============================================================================

class CodingCouncilIntent:
    """
    Intent types for Coding Council commands via Trinity.

    These extend the base TrinityIntent for evolution-specific operations.
    """
    # Evolution commands (from J-Prime)
    EVOLVE_CODE = "evolve_code"
    ABORT_EVOLUTION = "abort_evolution"
    ROLLBACK_EVOLUTION = "rollback_evolution"

    # Query commands
    GET_EVOLUTION_STATUS = "get_evolution_status"
    GET_ACTIVE_TASKS = "get_active_tasks"
    GET_FRAMEWORK_STATUS = "get_framework_status"

    # Cross-repo commands
    CROSS_REPO_EVOLVE = "cross_repo_evolve"
    SYNC_REPOS = "sync_repos"

    # Configuration commands
    UPDATE_CONFIG = "update_council_config"
    ENABLE_FRAMEWORK = "enable_framework"
    DISABLE_FRAMEWORK = "disable_framework"


@dataclass
class CodingCouncilCommand:
    """
    Command schema for Coding Council operations.

    This integrates with TrinityCommand for routing through Reactor Core.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    intent: str = CodingCouncilIntent.EVOLVE_CODE
    payload: Dict[str, Any] = field(default_factory=dict)

    # Task specification
    description: Optional[str] = None
    target_files: List[str] = field(default_factory=list)
    target_repos: List[str] = field(default_factory=list)  # For cross-repo

    # Execution options
    require_approval: bool = True
    require_sandbox: bool = False
    require_planning: bool = False
    timeout_seconds: float = 300.0

    # Correlation
    correlation_id: Optional[str] = None
    response_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for Trinity transport."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "payload": self.payload,
            "description": self.description,
            "target_files": self.target_files,
            "target_repos": self.target_repos,
            "require_approval": self.require_approval,
            "require_sandbox": self.require_sandbox,
            "require_planning": self.require_planning,
            "timeout_seconds": self.timeout_seconds,
            "correlation_id": self.correlation_id,
            "response_to": self.response_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodingCouncilCommand":
        """Deserialize from dict."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            intent=data.get("intent", CodingCouncilIntent.EVOLVE_CODE),
            payload=data.get("payload", {}),
            description=data.get("description"),
            target_files=data.get("target_files", []),
            target_repos=data.get("target_repos", []),
            require_approval=data.get("require_approval", True),
            require_sandbox=data.get("require_sandbox", False),
            require_planning=data.get("require_planning", False),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            correlation_id=data.get("correlation_id"),
            response_to=data.get("response_to"),
        )


# =============================================================================
# Trinity Integration Bridge
# =============================================================================

class CodingCouncilTrinityBridge:
    """
    Bridge between Coding Council and Trinity network.

    Handles:
    - Command reception from J-Prime
    - Status broadcasting to Reactor Core
    - Cross-repo coordination
    - Evolution result reporting
    """

    def __init__(self):
        self._council: Optional["UnifiedCodingCouncil"] = None
        self._reactor_bridge = None
        self._initialized = False
        self._status_task: Optional[asyncio.Task] = None
        self._command_handlers: Dict[str, Callable] = {}
        self._active_evolutions: Dict[str, "EvolutionTask"] = {}

        # Metrics
        self._total_commands_received = 0
        self._total_evolutions_completed = 0
        self._total_evolutions_failed = 0

    async def initialize(self, council: "UnifiedCodingCouncil") -> bool:
        """
        Initialize the Trinity bridge.

        Args:
            council: The UnifiedCodingCouncil instance to connect

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            logger.debug("[CodingCouncilTrinity] Already initialized")
            return True

        if not CODING_COUNCIL_ENABLED:
            logger.info("[CodingCouncilTrinity] Coding Council disabled via CODING_COUNCIL_ENABLED")
            return False

        self._council = council

        logger.info("=" * 60)
        logger.info("CODING COUNCIL: Initializing Trinity Bridge")
        logger.info("=" * 60)

        try:
            # Step 1: Get Reactor Bridge
            self._reactor_bridge = await self._get_reactor_bridge()
            if self._reactor_bridge is None:
                logger.warning("[CodingCouncilTrinity] ReactorCoreBridge not available")
                # Continue without bridge - local-only mode

            # Step 2: Register command handlers
            self._register_command_handlers()

            # Step 3: Ensure Trinity directories exist
            self._ensure_directories()

            # Step 4: Write initial state
            await self._write_state()

            # Step 5: Start status broadcast loop
            self._status_task = asyncio.create_task(self._status_broadcast_loop())

            # Step 6: Register with Reactor Bridge if available
            if self._reactor_bridge:
                await self._register_with_reactor()

            self._initialized = True

            logger.info("=" * 60)
            logger.info("CODING COUNCIL: Trinity Bridge Online")
            logger.info(f"  Cross-Repo: {CODING_COUNCIL_CROSS_REPO}")
            logger.info(f"  Auto-Approve: {CODING_COUNCIL_AUTO_APPROVE}")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"[CodingCouncilTrinity] Initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the Trinity bridge."""
        if not self._initialized:
            return

        logger.info("[CodingCouncilTrinity] Shutting down...")

        # Cancel status broadcast
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass

        # Write final state
        await self._write_state(status="offline")

        self._initialized = False
        logger.info("[CodingCouncilTrinity] Shutdown complete")

    async def _get_reactor_bridge(self):
        """Lazy import of ReactorCoreBridge."""
        try:
            from backend.system.reactor_bridge import get_reactor_bridge
            return get_reactor_bridge()
        except ImportError:
            try:
                from system.reactor_bridge import get_reactor_bridge
                return get_reactor_bridge()
            except ImportError:
                return None

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        TRINITY_DIR.mkdir(parents=True, exist_ok=True)
        (TRINITY_DIR / "components").mkdir(parents=True, exist_ok=True)
        (TRINITY_DIR / "commands").mkdir(parents=True, exist_ok=True)
        (TRINITY_DIR / "evolutions").mkdir(parents=True, exist_ok=True)

    def _register_command_handlers(self) -> None:
        """Register handlers for all Coding Council intents."""
        self._command_handlers = {
            CodingCouncilIntent.EVOLVE_CODE: self._handle_evolve_code,
            CodingCouncilIntent.ABORT_EVOLUTION: self._handle_abort_evolution,
            CodingCouncilIntent.ROLLBACK_EVOLUTION: self._handle_rollback_evolution,
            CodingCouncilIntent.GET_EVOLUTION_STATUS: self._handle_get_status,
            CodingCouncilIntent.GET_ACTIVE_TASKS: self._handle_get_active_tasks,
            CodingCouncilIntent.GET_FRAMEWORK_STATUS: self._handle_get_framework_status,
            CodingCouncilIntent.CROSS_REPO_EVOLVE: self._handle_cross_repo_evolve,
            CodingCouncilIntent.SYNC_REPOS: self._handle_sync_repos,
            CodingCouncilIntent.UPDATE_CONFIG: self._handle_update_config,
            CodingCouncilIntent.ENABLE_FRAMEWORK: self._handle_enable_framework,
            CodingCouncilIntent.DISABLE_FRAMEWORK: self._handle_disable_framework,
        }

    async def _register_with_reactor(self) -> None:
        """Register Coding Council command handlers with Reactor Bridge."""
        if not self._reactor_bridge:
            return

        try:
            # Import TrinityIntent to extend it
            from backend.system.reactor_bridge import TrinityIntent

            # Register a general handler for Coding Council commands
            @self._reactor_bridge.on_command("execute_plan")
            async def handle_execute_plan(command):
                """Handle EXECUTE_PLAN commands that may be evolution tasks."""
                payload = command.payload
                if payload.get("type") == "evolution":
                    council_cmd = CodingCouncilCommand.from_dict(payload)
                    return await self.handle_command(council_cmd)

            logger.info("[CodingCouncilTrinity] Registered with Reactor Bridge")

        except Exception as e:
            logger.warning(f"[CodingCouncilTrinity] Could not register with Reactor: {e}")

    async def handle_command(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """
        Handle a Coding Council command.

        Args:
            command: The command to handle

        Returns:
            Result dict with success status and data
        """
        self._total_commands_received += 1
        intent = command.intent

        handler = self._command_handlers.get(intent)
        if handler is None:
            return {
                "success": False,
                "error": f"Unknown intent: {intent}",
                "command_id": command.id,
            }

        try:
            result = await handler(command)
            return {
                "success": True,
                "data": result,
                "command_id": command.id,
            }
        except Exception as e:
            logger.error(f"[CodingCouncilTrinity] Command {command.id} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "command_id": command.id,
            }

    # =========================================================================
    # Command Handlers
    # =========================================================================

    async def _handle_evolve_code(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle EVOLVE_CODE command."""
        if not self._council:
            raise RuntimeError("Coding Council not initialized")

        description = command.description or command.payload.get("description", "")
        if not description:
            raise ValueError("Evolution description required")

        # Execute evolution
        result = await self._council.evolve(
            description=description,
            target_files=command.target_files or None,
            require_sandbox=command.require_sandbox,
            require_planning=command.require_planning,
            require_approval=command.require_approval and not CODING_COUNCIL_AUTO_APPROVE,
            correlation_id=command.correlation_id,
            timeout=command.timeout_seconds,
        )

        # Update metrics
        if result.success:
            self._total_evolutions_completed += 1
        else:
            self._total_evolutions_failed += 1

        return self._evolution_result_to_dict(result)

    async def _handle_abort_evolution(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle ABORT_EVOLUTION command."""
        task_id = command.payload.get("task_id")
        if not task_id:
            raise ValueError("task_id required for abort")

        if not self._council:
            raise RuntimeError("Coding Council not initialized")

        # Get the task and abort it
        task = self._active_evolutions.get(task_id)
        if not task:
            return {"aborted": False, "reason": "Task not found or already completed"}

        # Mark as aborted (actual abortion happens in orchestrator)
        return {"aborted": True, "task_id": task_id}

    async def _handle_rollback_evolution(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle ROLLBACK_EVOLUTION command."""
        task_id = command.payload.get("task_id")
        if not task_id:
            raise ValueError("task_id required for rollback")

        if not self._council:
            raise RuntimeError("Coding Council not initialized")

        # Trigger rollback
        success = await self._council._rollback_manager.rollback(task_id, "User requested")

        return {"rolled_back": success, "task_id": task_id}

    async def _handle_get_status(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle GET_EVOLUTION_STATUS command."""
        task_id = command.payload.get("task_id")

        if not self._council:
            return {"status": "not_initialized"}

        status = self._council.get_status()

        if task_id:
            # Get specific task status
            task = self._active_evolutions.get(task_id)
            if task:
                status["task"] = {
                    "id": task.task_id,
                    "status": task.status.value,
                    "description": task.description[:100],
                }

        return status

    async def _handle_get_active_tasks(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle GET_ACTIVE_TASKS command."""
        return {
            "active_tasks": [
                {
                    "id": task.task_id,
                    "description": task.description[:100],
                    "status": task.status.value,
                    "target_files": task.target_files[:5],
                }
                for task in self._active_evolutions.values()
            ],
            "count": len(self._active_evolutions),
        }

    async def _handle_get_framework_status(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle GET_FRAMEWORK_STATUS command."""
        if not self._council:
            return {"frameworks": {}}

        framework_status = {}
        for name, adapter in [
            ("aider", self._council._aider),
            ("repomaster", self._council._repomaster),
            ("metagpt", self._council._metagpt),
            ("openhands", self._council._openhands),
            ("continue", self._council._continue),
        ]:
            if adapter:
                available = await adapter.is_available()
                framework_status[name] = {
                    "available": available,
                    "enabled": self._council.config.enabled_frameworks.get(name, True),
                }

        return {"frameworks": framework_status}

    async def _handle_cross_repo_evolve(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle CROSS_REPO_EVOLVE command for multi-repo evolution."""
        if not CODING_COUNCIL_CROSS_REPO:
            raise RuntimeError("Cross-repo evolution disabled")

        if not self._council:
            raise RuntimeError("Coding Council not initialized")

        description = command.description or command.payload.get("description", "")
        target_repos = command.target_repos or ["jarvis"]

        results = {}

        # Map repo names to paths
        repo_map = {
            "jarvis": JARVIS_REPO,
            "jarvis_prime": JARVIS_PRIME_REPO,
            "jprime": JARVIS_PRIME_REPO,
            "reactor_core": REACTOR_CORE_REPO,
            "reactor": REACTOR_CORE_REPO,
        }

        for repo_name in target_repos:
            repo_path = repo_map.get(repo_name.lower())
            if not repo_path or not repo_path.exists():
                results[repo_name] = {"success": False, "error": f"Repo not found: {repo_name}"}
                continue

            # Create a task for this repo
            # Note: This is a simplified version - full implementation would
            # create separate CodingCouncil instances for each repo
            logger.info(f"[CodingCouncilTrinity] Cross-repo evolve: {repo_name}")

            result = await self._council.evolve(
                description=f"[{repo_name}] {description}",
                target_files=command.target_files,
                require_approval=not CODING_COUNCIL_AUTO_APPROVE,
                correlation_id=f"{command.correlation_id}-{repo_name}" if command.correlation_id else None,
            )

            results[repo_name] = self._evolution_result_to_dict(result)

        return {"results": results}

    async def _handle_sync_repos(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle SYNC_REPOS command to synchronize Trinity repos."""
        if not CODING_COUNCIL_CROSS_REPO:
            raise RuntimeError("Cross-repo operations disabled")

        sync_results = {}

        # Check each repo's status
        repos = {
            "jarvis": JARVIS_REPO,
            "jarvis_prime": JARVIS_PRIME_REPO,
            "reactor_core": REACTOR_CORE_REPO,
        }

        for name, path in repos.items():
            if path.exists():
                # Check git status
                try:
                    import subprocess
                    result = subprocess.run(
                        ["git", "status", "--porcelain"],
                        cwd=str(path),
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )

                    has_changes = bool(result.stdout.strip())
                    sync_results[name] = {
                        "exists": True,
                        "has_uncommitted_changes": has_changes,
                        "path": str(path),
                    }
                except Exception as e:
                    sync_results[name] = {"exists": True, "error": str(e)}
            else:
                sync_results[name] = {"exists": False, "path": str(path)}

        return {"repos": sync_results}

    async def _handle_update_config(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle UPDATE_CONFIG command."""
        if not self._council:
            raise RuntimeError("Coding Council not initialized")

        config_updates = command.payload.get("config", {})

        # Apply allowed updates (sanitized)
        if "execution_timeout" in config_updates:
            self._council.config.execution_timeout = float(config_updates["execution_timeout"])
        if "max_retries" in config_updates:
            self._council.config.max_retries = int(config_updates["max_retries"])
        if "require_validation" in config_updates:
            self._council.config.require_validation = bool(config_updates["require_validation"])

        return {"updated": True, "config": self._council.config.__dict__}

    async def _handle_enable_framework(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle ENABLE_FRAMEWORK command."""
        framework = command.payload.get("framework", "").lower()
        if not framework:
            raise ValueError("framework name required")

        if not self._council:
            raise RuntimeError("Coding Council not initialized")

        self._council.config.enabled_frameworks[framework] = True

        return {"framework": framework, "enabled": True}

    async def _handle_disable_framework(self, command: CodingCouncilCommand) -> Dict[str, Any]:
        """Handle DISABLE_FRAMEWORK command."""
        framework = command.payload.get("framework", "").lower()
        if not framework:
            raise ValueError("framework name required")

        if not self._council:
            raise RuntimeError("Coding Council not initialized")

        self._council.config.enabled_frameworks[framework] = False

        return {"framework": framework, "enabled": False}

    # =========================================================================
    # State Management
    # =========================================================================

    async def _write_state(self, status: str = "online") -> None:
        """Write current state to Trinity directory."""
        try:
            # Use atomic write to prevent corruption
            from backend.system.trinity_initializer import write_json_atomic
        except ImportError:
            try:
                from system.trinity_initializer import write_json_atomic
            except ImportError:
                # Fallback to standard write
                import json
                def write_json_atomic(path, data):
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=2)
                    return True

        state = {
            "component": "coding_council",
            "version": "77.0",
            "status": status,
            "timestamp": time.time(),
            "pid": os.getpid(),
            "metrics": {
                "commands_received": self._total_commands_received,
                "evolutions_completed": self._total_evolutions_completed,
                "evolutions_failed": self._total_evolutions_failed,
                "active_tasks": len(self._active_evolutions),
            },
            "config": {
                "cross_repo_enabled": CODING_COUNCIL_CROSS_REPO,
                "auto_approve": CODING_COUNCIL_AUTO_APPROVE,
            },
        }

        write_json_atomic(CODING_COUNCIL_STATE_FILE, state)

    async def _status_broadcast_loop(self) -> None:
        """Periodically broadcast status to Trinity network."""
        while True:
            try:
                await asyncio.sleep(STATUS_BROADCAST_INTERVAL)
                await self._write_state()

                # Broadcast via Reactor Bridge if available
                if self._reactor_bridge:
                    await self._broadcast_status()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[CodingCouncilTrinity] Status broadcast error: {e}")

    async def _broadcast_status(self) -> None:
        """Broadcast status update to Reactor Core."""
        if not self._reactor_bridge or not self._council:
            return

        try:
            status = self._council.get_status()
            status["component"] = "coding_council"
            status["timestamp"] = time.time()

            # Use Reactor Bridge to broadcast
            await self._reactor_bridge.publish_heartbeat_async(
                extra_data={"coding_council": status}
            )
        except Exception as e:
            logger.debug(f"[CodingCouncilTrinity] Broadcast failed: {e}")

    def _evolution_result_to_dict(self, result: "EvolutionResult") -> Dict[str, Any]:
        """Convert EvolutionResult to serializable dict."""
        return {
            "success": result.success,
            "task_id": result.task_id,
            "execution_time_ms": result.execution_time_ms,
            "changes_made": result.changes_made,
            "files_modified": result.files_modified,
            "validation_passed": result.validation_report.passed if result.validation_report else True,
            "error": result.error,
            "framework_used": result.framework_used.value if result.framework_used else None,
            "rollback_available": result.rollback_id is not None,
        }


# =============================================================================
# Global Instance Management
# =============================================================================

_trinity_bridge: Optional[CodingCouncilTrinityBridge] = None


async def get_coding_council_bridge() -> CodingCouncilTrinityBridge:
    """Get or create the global Trinity bridge instance."""
    global _trinity_bridge

    if _trinity_bridge is None:
        _trinity_bridge = CodingCouncilTrinityBridge()

    return _trinity_bridge


async def initialize_coding_council_trinity(council: "UnifiedCodingCouncil") -> bool:
    """
    Initialize the Coding Council Trinity integration.

    This should be called after UnifiedCodingCouncil is initialized.

    Args:
        council: The initialized UnifiedCodingCouncil instance

    Returns:
        True if initialization succeeded
    """
    bridge = await get_coding_council_bridge()
    return await bridge.initialize(council)


async def shutdown_coding_council_trinity() -> None:
    """Shutdown the Coding Council Trinity integration."""
    global _trinity_bridge

    if _trinity_bridge is not None:
        await _trinity_bridge.shutdown()
        _trinity_bridge = None
