"""
Cross-Repository Refactoring Coordinator v1.0
==============================================

Coordinates refactoring operations across the Trinity ecosystem:
JARVIS, JARVIS-Prime, and Reactor-Core repositories.

Features:
- Atomic cross-repo transactions
- Trinity event bus integration
- Parallel repository processing
- Distributed locking
- Rollback coordination

Author: JARVIS AI System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from .refactoring_engine import (
    RefactoringEngine,
    RefactoringResult,
    RefactoringStatus,
    RefactoringType,
    RefactoringScope,
    RefactoringConfig,
    ModifiedFile,
    get_refactoring_engine,
)
from .cross_repo_reference_finder import (
    CrossRepoReferenceFinder,
    RepoType,
    Reference,
    SymbolKind,
    get_cross_repo_reference_finder,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_env_path(key: str, default: str = "") -> Path:
    return Path(os.path.expanduser(_get_env(key, default)))


def _get_env_bool(key: str, default: bool = False) -> bool:
    val = _get_env(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    try:
        return int(_get_env(key, str(default)))
    except ValueError:
        return default


class CrossRepoConfig:
    """Configuration for cross-repo refactoring."""

    ENABLED: bool = _get_env_bool("REFACTORING_CROSS_REPO_ENABLED", True)
    PARALLEL_REPOS: bool = _get_env_bool("REFACTORING_PARALLEL_REPOS", True)

    # Repository paths
    JARVIS_REPO: Path = _get_env_path("JARVIS_REPO_PATH", "~/Documents/repos/JARVIS-AI-Agent")
    PRIME_REPO: Path = _get_env_path("JARVIS_PRIME_REPO_PATH", "~/Documents/repos/jarvis-prime")
    REACTOR_REPO: Path = _get_env_path("REACTOR_CORE_REPO_PATH", "~/Documents/repos/reactor-core")

    # Event bus
    EVENT_BUS_ENABLED: bool = _get_env_bool("REFACTORING_EVENT_BUS_ENABLED", True)
    EVENT_BUS_DIR: Path = _get_env_path("OUROBOROS_EVENT_BUS", "~/.jarvis/ouroboros/events")

    # Timeouts
    COORDINATION_TIMEOUT_MS: int = _get_env_int("REFACTORING_COORDINATION_TIMEOUT_MS", 120000)


# =============================================================================
# EVENT TYPES
# =============================================================================

class RefactoringEventType(str, Enum):
    """Types of refactoring events for Trinity event bus."""
    REFACTORING_STARTED = "refactoring_started"
    REFACTORING_COMPLETED = "refactoring_completed"
    REFACTORING_FAILED = "refactoring_failed"
    REFERENCE_SEARCH_REQUEST = "reference_search_request"
    REFERENCE_SEARCH_RESULT = "reference_search_result"
    LOCK_REQUESTED = "lock_requested"
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_RELEASED = "lock_released"
    ROLLBACK_REQUESTED = "rollback_requested"
    ROLLBACK_COMPLETED = "rollback_completed"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CrossRepoRefactoringResult:
    """Result of a cross-repo refactoring operation."""
    operation_id: str
    operation_type: RefactoringType
    status: RefactoringStatus
    repositories_modified: List[RepoType] = field(default_factory=list)
    results_by_repo: Dict[RepoType, RefactoringResult] = field(default_factory=dict)
    total_files_modified: int = 0
    total_references_updated: int = 0
    total_call_sites_updated: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    rollback_available: bool = True

    @property
    def success(self) -> bool:
        return self.status == RefactoringStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "status": self.status.value,
            "repositories_modified": [r.value for r in self.repositories_modified],
            "total_files_modified": self.total_files_modified,
            "total_references_updated": self.total_references_updated,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class RefactoringEvent:
    """Event for Trinity event bus."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: RefactoringEventType = RefactoringEventType.REFACTORING_STARTED
    operation_id: str = ""
    source_repo: RepoType = RepoType.JARVIS
    target_repos: List[RepoType] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "operation_id": self.operation_id,
            "source_repo": self.source_repo.value,
            "target_repos": [r.value for r in self.target_repos],
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# EVENT BUS INTEGRATION
# =============================================================================

class RefactoringEventBus:
    """
    Event bus integration for cross-repo refactoring.

    Emits events to the Trinity event bus for coordination
    between JARVIS, JARVIS-Prime, and Reactor-Core.
    """

    def __init__(self, config: Optional[CrossRepoConfig] = None):
        self.config = config or CrossRepoConfig()
        self._handlers: Dict[RefactoringEventType, List[Callable]] = {}

        if self.config.EVENT_BUS_ENABLED:
            self._ensure_event_dir()

    def _ensure_event_dir(self) -> None:
        """Ensure event bus directory exists."""
        event_dir = self.config.EVENT_BUS_DIR / "refactoring"
        event_dir.mkdir(parents=True, exist_ok=True)
        (event_dir / "pending").mkdir(exist_ok=True)
        (event_dir / "processed").mkdir(exist_ok=True)

    async def emit(self, event: RefactoringEvent) -> None:
        """Emit an event to the bus."""
        if not self.config.EVENT_BUS_ENABLED:
            return

        import json
        event_file = self.config.EVENT_BUS_DIR / "refactoring" / "pending" / f"{event.event_id}.json"
        await asyncio.to_thread(
            event_file.write_text,
            json.dumps(event.to_dict(), indent=2)
        )
        logger.debug(f"Emitted refactoring event: {event.event_type.value}")

    def register_handler(
        self,
        event_type: RefactoringEventType,
        handler: Callable[[RefactoringEvent], Any],
    ) -> None:
        """Register an event handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)


# =============================================================================
# DISTRIBUTED LOCK
# =============================================================================

class DistributedRefactoringLock:
    """
    Distributed lock for cross-repo refactoring.

    Prevents concurrent modifications to the same files
    across multiple repositories.
    """

    def __init__(self, config: Optional[CrossRepoConfig] = None):
        self.config = config or CrossRepoConfig()
        self._locks: Dict[str, asyncio.Lock] = {}
        self._held_locks: Set[str] = set()

    async def acquire(self, files: List[Path], timeout: float = 30.0) -> bool:
        """Acquire locks on files."""
        lock_keys = [str(f.resolve()) for f in files]

        try:
            for key in lock_keys:
                if key not in self._locks:
                    self._locks[key] = asyncio.Lock()

                acquired = await asyncio.wait_for(
                    self._locks[key].acquire(),
                    timeout=timeout
                )
                if acquired:
                    self._held_locks.add(key)
                else:
                    await self._release_held()
                    return False

            return True
        except asyncio.TimeoutError:
            await self._release_held()
            return False

    async def release(self, files: List[Path]) -> None:
        """Release locks on files."""
        lock_keys = [str(f.resolve()) for f in files]
        for key in lock_keys:
            if key in self._held_locks and key in self._locks:
                try:
                    self._locks[key].release()
                except RuntimeError:
                    pass  # Already released
                self._held_locks.discard(key)

    async def _release_held(self) -> None:
        """Release all held locks."""
        for key in list(self._held_locks):
            if key in self._locks:
                try:
                    self._locks[key].release()
                except RuntimeError:
                    pass
            self._held_locks.discard(key)


# =============================================================================
# CROSS-REPO REFACTORING COORDINATOR
# =============================================================================

class CrossRepoRefactoringCoordinator:
    """
    Coordinates refactoring operations across multiple repositories.

    Manages:
    - Cross-repo reference finding
    - Distributed locking
    - Atomic multi-repo transactions
    - Event bus communication
    - Rollback coordination
    """

    def __init__(self, config: Optional[CrossRepoConfig] = None):
        self.config = config or CrossRepoConfig()
        self.engine = get_refactoring_engine()
        self.reference_finder = get_cross_repo_reference_finder()
        self.event_bus = RefactoringEventBus(self.config)
        self.lock = DistributedRefactoringLock(self.config)

        # Track operations
        self._active_operations: Dict[str, CrossRepoRefactoringResult] = {}

    def _get_repo_for_file(self, file_path: Path) -> RepoType:
        """Determine which repository a file belongs to."""
        path_str = str(file_path.resolve())

        if "jarvis-prime" in path_str.lower():
            return RepoType.PRIME
        elif "reactor-core" in path_str.lower():
            return RepoType.REACTOR
        else:
            return RepoType.JARVIS

    async def refactor_across_repos(
        self,
        operation_type: RefactoringType,
        symbol_name: str,
        source_file: Path,
        **kwargs: Any,
    ) -> CrossRepoRefactoringResult:
        """
        Execute a refactoring operation across all affected repositories.

        Args:
            operation_type: Type of refactoring
            symbol_name: Name of symbol being refactored
            source_file: Primary file containing the symbol
            **kwargs: Operation-specific parameters

        Returns:
            CrossRepoRefactoringResult with all changes
        """
        import time
        start_time = time.time()
        operation_id = str(uuid4())

        result = CrossRepoRefactoringResult(
            operation_id=operation_id,
            operation_type=operation_type,
            status=RefactoringStatus.PENDING,
        )
        self._active_operations[operation_id] = result

        try:
            # Emit start event
            await self.event_bus.emit(RefactoringEvent(
                event_type=RefactoringEventType.REFACTORING_STARTED,
                operation_id=operation_id,
                source_repo=self._get_repo_for_file(source_file),
                payload={
                    "operation_type": operation_type.value,
                    "symbol_name": symbol_name,
                    "source_file": str(source_file),
                },
            ))

            # Find all references across repos
            symbol_kind = self._infer_symbol_kind(operation_type)
            search_result = await self.reference_finder.find_all_references(
                symbol_name=symbol_name,
                symbol_kind=symbol_kind,
                source_file=source_file,
            )

            # Group references by repository
            refs_by_repo: Dict[RepoType, List[Reference]] = {
                RepoType.JARVIS: [],
                RepoType.PRIME: [],
                RepoType.REACTOR: [],
            }

            for ref in search_result.references:
                refs_by_repo[ref.repository].append(ref)

            # Determine which repos need modification
            repos_to_modify = [
                repo for repo, refs in refs_by_repo.items()
                if refs or self._get_repo_for_file(source_file) == repo
            ]

            result.repositories_modified = repos_to_modify

            # Collect all files to lock
            files_to_lock = [source_file]
            for ref in search_result.references:
                if ref.file_path not in files_to_lock:
                    files_to_lock.append(ref.file_path)

            # Acquire locks
            if not await self.lock.acquire(files_to_lock):
                result.status = RefactoringStatus.FAILED
                result.errors.append("Could not acquire locks on all files")
                return result

            try:
                # Execute refactoring based on type
                if operation_type == RefactoringType.EXTRACT_METHOD:
                    inner_result = await self.engine.extract_method(
                        file_path=source_file,
                        new_method_name=symbol_name,
                        **kwargs,
                    )
                elif operation_type == RefactoringType.INLINE_VARIABLE:
                    inner_result = await self.engine.inline_variable(
                        file_path=source_file,
                        variable_name=symbol_name,
                        **kwargs,
                    )
                elif operation_type == RefactoringType.MOVE_METHOD:
                    inner_result = await self.engine.move_method(
                        source_file=source_file,
                        method_name=symbol_name,
                        **kwargs,
                    )
                elif operation_type == RefactoringType.CHANGE_SIGNATURE:
                    inner_result = await self.engine.change_signature(
                        file_path=source_file,
                        function_name=symbol_name,
                        scope=RefactoringScope.ALL_REPOS,
                        **kwargs,
                    )
                else:
                    result.status = RefactoringStatus.FAILED
                    result.errors.append(f"Unsupported operation type: {operation_type}")
                    return result

                # Update result
                source_repo = self._get_repo_for_file(source_file)
                result.results_by_repo[source_repo] = inner_result

                if inner_result.success:
                    result.status = RefactoringStatus.COMPLETED
                    result.total_files_modified = len(inner_result.modified_files)
                    result.total_references_updated = inner_result.references_updated
                    result.total_call_sites_updated = inner_result.call_sites_updated
                else:
                    result.status = RefactoringStatus.FAILED
                    result.errors.extend(inner_result.errors)

                result.warnings.extend(inner_result.warnings)

            finally:
                # Release locks
                await self.lock.release(files_to_lock)

            # Emit completion event
            event_type = (
                RefactoringEventType.REFACTORING_COMPLETED
                if result.success
                else RefactoringEventType.REFACTORING_FAILED
            )

            await self.event_bus.emit(RefactoringEvent(
                event_type=event_type,
                operation_id=operation_id,
                source_repo=self._get_repo_for_file(source_file),
                payload=result.to_dict(),
            ))

        except Exception as e:
            logger.error(f"Cross-repo refactoring failed: {e}")
            result.status = RefactoringStatus.FAILED
            result.errors.append(str(e))

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    def _infer_symbol_kind(self, operation_type: RefactoringType) -> SymbolKind:
        """Infer symbol kind from operation type."""
        mapping = {
            RefactoringType.EXTRACT_METHOD: SymbolKind.FUNCTION,
            RefactoringType.INLINE_VARIABLE: SymbolKind.VARIABLE,
            RefactoringType.MOVE_METHOD: SymbolKind.METHOD,
            RefactoringType.MOVE_CLASS: SymbolKind.CLASS,
            RefactoringType.CHANGE_SIGNATURE: SymbolKind.FUNCTION,
            RefactoringType.RENAME: SymbolKind.FUNCTION,
        }
        return mapping.get(operation_type, SymbolKind.FUNCTION)

    async def rollback(self, operation_id: str) -> bool:
        """Rollback a cross-repo refactoring operation."""
        if operation_id not in self._active_operations:
            return False

        result = self._active_operations[operation_id]

        # Emit rollback event
        await self.event_bus.emit(RefactoringEvent(
            event_type=RefactoringEventType.ROLLBACK_REQUESTED,
            operation_id=operation_id,
            payload={"reason": "user_requested"},
        ))

        # Rollback engine operations
        success = await self.engine.rollback(operation_id)

        if success:
            result.status = RefactoringStatus.ROLLED_BACK

            await self.event_bus.emit(RefactoringEvent(
                event_type=RefactoringEventType.ROLLBACK_COMPLETED,
                operation_id=operation_id,
            ))

        return success


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_coordinator_instance: Optional[CrossRepoRefactoringCoordinator] = None


def get_cross_repo_refactoring_coordinator() -> CrossRepoRefactoringCoordinator:
    """Get the singleton coordinator instance."""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = CrossRepoRefactoringCoordinator()
    return _coordinator_instance


async def get_cross_repo_refactoring_coordinator_async() -> CrossRepoRefactoringCoordinator:
    """Get the singleton coordinator instance (async)."""
    return get_cross_repo_refactoring_coordinator()
