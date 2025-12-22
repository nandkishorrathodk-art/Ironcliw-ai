#!/usr/bin/env python3
"""
Bulk Operation Guardian - Automated Backup Before Risky Operations
===================================================================

Production-grade system that automatically creates backups before
bulk/risky operations like:
- AI-assisted code modifications
- Mass file updates
- Refactoring operations
- Migrations

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Bulk Operation Guardian v1.0                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Operation      │  │ Snapshot       │  │ Rollback       │                 │
│  │ Monitor        │  │ Manager        │  │ Engine         │                 │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                 │
│          └───────────────────┼───────────────────┘                          │
│                              ▼                                               │
│          ┌─────────────────────────────────────────┐                        │
│          │    Operation Context Manager            │                        │
│          │  • Automatic snapshot on entry          │                        │
│          │  • Integrity validation on exit         │                        │
│          │  • Automatic rollback on failure        │                        │
│          │  • Audit logging                        │                        │
│          └─────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    async with BulkOperationGuardian.protect("refactoring") as guardian:
        # Your risky operation here
        await modify_many_files(files)
        
    # If anything goes wrong, files are automatically restored

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
)
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


# =============================================================================
# DATA CLASSES
# =============================================================================

class OperationType(Enum):
    """Types of bulk operations."""
    AI_MODIFICATION = "ai_modification"
    REFACTORING = "refactoring"
    MIGRATION = "migration"
    BULK_UPDATE = "bulk_update"
    IMPORT_OPTIMIZATION = "import_optimization"
    FORMAT_UPDATE = "format_update"
    DEPENDENCY_UPDATE = "dependency_update"
    CUSTOM = "custom"


class OperationStatus(Enum):
    """Status of an operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class FileSnapshot:
    """Snapshot of a single file."""
    original_path: str
    snapshot_path: str
    content_hash: str
    file_size: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_path": self.original_path,
            "snapshot_path": self.snapshot_path,
            "content_hash": self.content_hash,
            "file_size": self.file_size,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OperationSnapshot:
    """Snapshot of an entire operation."""
    operation_id: str
    operation_type: OperationType
    description: str
    file_snapshots: List[FileSnapshot] = field(default_factory=list)
    status: OperationStatus = OperationStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "description": self.description,
            "file_count": len(self.file_snapshots),
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


# =============================================================================
# SNAPSHOT MANAGER
# =============================================================================

class SnapshotManager:
    """
    Manages file snapshots for backup and rollback.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or os.environ.get(
            "JARVIS_SNAPSHOT_DIR",
            os.path.expanduser("~/.jarvis/operation_snapshots")
        ))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_snapshots = _env_int("JARVIS_MAX_SNAPSHOTS", 20)
        self.retention_hours = _env_int("JARVIS_SNAPSHOT_RETENTION_HOURS", 72)
        self._lock = threading.Lock()
    
    def create_snapshot(self, file_path: str, operation_id: str) -> Optional[FileSnapshot]:
        """Create a snapshot of a single file."""
        source = Path(file_path)
        
        if not source.exists():
            logger.warning(f"Cannot snapshot non-existent file: {file_path}")
            return None
        
        with self._lock:
            try:
                # Read content and compute hash
                content = source.read_bytes()
                content_hash = hashlib.sha256(content).hexdigest()
                
                # Create snapshot directory for this operation
                snapshot_dir = self.base_dir / operation_id
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                
                # Create snapshot filename
                safe_name = source.name.replace('/', '_').replace('\\', '_')
                file_hash = hashlib.md5(str(source.absolute()).encode()).hexdigest()[:8]
                snapshot_name = f"{file_hash}_{safe_name}"
                snapshot_path = snapshot_dir / snapshot_name
                
                # Copy file
                shutil.copy2(source, snapshot_path)
                
                snapshot = FileSnapshot(
                    original_path=str(source.absolute()),
                    snapshot_path=str(snapshot_path),
                    content_hash=content_hash,
                    file_size=len(content),
                )
                
                logger.debug(f"📸 Snapshot created: {file_path}")
                return snapshot
                
            except Exception as e:
                logger.error(f"❌ Failed to create snapshot for {file_path}: {e}")
                return None
    
    def create_operation_snapshot(
        self,
        files: List[str],
        operation_type: OperationType,
        description: str
    ) -> OperationSnapshot:
        """Create snapshots for all files in an operation."""
        operation_id = f"{operation_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        operation = OperationSnapshot(
            operation_id=operation_id,
            operation_type=operation_type,
            description=description,
        )
        
        for file_path in files:
            snapshot = self.create_snapshot(file_path, operation_id)
            if snapshot:
                operation.file_snapshots.append(snapshot)
        
        # Save operation metadata
        metadata_path = self.base_dir / operation_id / "operation.json"
        with open(metadata_path, 'w') as f:
            json.dump(operation.to_dict(), f, indent=2)
        
        logger.info(f"📦 Operation snapshot created: {operation_id} ({len(operation.file_snapshots)} files)")
        
        # Cleanup old snapshots
        self._cleanup_old_snapshots()
        
        return operation
    
    def restore_file(self, snapshot: FileSnapshot) -> bool:
        """Restore a single file from snapshot."""
        try:
            if not Path(snapshot.snapshot_path).exists():
                logger.error(f"Snapshot not found: {snapshot.snapshot_path}")
                return False
            
            shutil.copy2(snapshot.snapshot_path, snapshot.original_path)
            logger.info(f"♻️ Restored: {snapshot.original_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restore {snapshot.original_path}: {e}")
            return False
    
    def restore_operation(self, operation: OperationSnapshot) -> Tuple[int, int]:
        """Restore all files from an operation snapshot."""
        success = 0
        failed = 0
        
        for snapshot in operation.file_snapshots:
            if self.restore_file(snapshot):
                success += 1
            else:
                failed += 1
        
        logger.info(f"♻️ Operation rollback: {success} restored, {failed} failed")
        return success, failed
    
    def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond retention period."""
        try:
            cutoff = datetime.now() - timedelta(hours=self.retention_hours)
            
            # Get all operation directories
            operations = sorted(self.base_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            
            # Remove old ones
            for op_dir in operations:
                if not op_dir.is_dir():
                    continue
                
                mtime = datetime.fromtimestamp(op_dir.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(op_dir)
                    logger.debug(f"Cleaned up old snapshot: {op_dir.name}")
            
            # Keep only max_snapshots most recent
            remaining = sorted(
                [d for d in self.base_dir.iterdir() if d.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            for op_dir in remaining[self.max_snapshots:]:
                shutil.rmtree(op_dir)
                logger.debug(f"Cleaned up excess snapshot: {op_dir.name}")
                
        except Exception as e:
            logger.debug(f"Snapshot cleanup error: {e}")
    
    def get_operation_snapshots(self) -> List[OperationSnapshot]:
        """Get list of available operation snapshots."""
        snapshots = []
        
        for op_dir in self.base_dir.iterdir():
            if not op_dir.is_dir():
                continue
            
            metadata_path = op_dir / "operation.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    
                    snapshots.append(OperationSnapshot(
                        operation_id=data["operation_id"],
                        operation_type=OperationType(data["operation_type"]),
                        description=data["description"],
                        status=OperationStatus(data.get("status", "completed")),
                        started_at=datetime.fromisoformat(data["started_at"]),
                    ))
                except Exception:
                    pass
        
        return sorted(snapshots, key=lambda s: s.started_at, reverse=True)


# =============================================================================
# INTEGRITY VALIDATOR
# =============================================================================

class IntegrityValidator:
    """
    Validates file integrity after operations.
    Uses the FileIntegrityGuardian if available.
    """
    
    def __init__(self):
        self._guardian = None
    
    def _get_guardian(self):
        """Lazy load the guardian to avoid circular imports."""
        if self._guardian is None:
            try:
                from backend.core.file_integrity_guardian import get_file_integrity_guardian
                self._guardian = get_file_integrity_guardian()
            except ImportError:
                pass
        return self._guardian
    
    def validate_files(self, file_paths: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate a list of files for integrity.
        
        Returns:
            Tuple of (all_valid, list_of_error_messages)
        """
        errors = []
        guardian = self._get_guardian()
        
        if guardian:
            for path in file_paths:
                report = guardian.check_file(path)
                if report.status.value != "healthy":
                    errors.append(f"{path}: {report.status.value}")
                    if report.syntax_error:
                        errors.append(f"  → {report.syntax_error}")
        else:
            # Fallback to basic syntax check
            import ast
            for path in file_paths:
                try:
                    content = Path(path).read_text(encoding='utf-8')
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(f"{path}: Syntax error at line {e.lineno}: {e.msg}")
                except Exception as e:
                    errors.append(f"{path}: {str(e)}")
        
        return len(errors) == 0, errors


# =============================================================================
# BULK OPERATION GUARDIAN (Main Class)
# =============================================================================

class BulkOperationGuardian:
    """
    Context manager for protecting bulk operations with automatic backup and rollback.
    
    Usage:
        async with BulkOperationGuardian.protect(
            files=["file1.py", "file2.py"],
            operation_type=OperationType.REFACTORING,
            description="Renaming functions"
        ) as guardian:
            # Your risky operations here
            await modify_files(files)
            
        # If exception occurs, files are automatically restored
    """
    
    _snapshot_manager: Optional[SnapshotManager] = None
    _lock = threading.Lock()
    
    @classmethod
    def _get_snapshot_manager(cls) -> SnapshotManager:
        if cls._snapshot_manager is None:
            with cls._lock:
                if cls._snapshot_manager is None:
                    cls._snapshot_manager = SnapshotManager()
        return cls._snapshot_manager
    
    def __init__(
        self,
        files: List[str],
        operation_type: OperationType = OperationType.CUSTOM,
        description: str = "",
        auto_rollback: bool = True,
        validate_after: bool = True,
    ):
        self.files = files
        self.operation_type = operation_type
        self.description = description
        self.auto_rollback = auto_rollback
        self.validate_after = validate_after
        
        self.snapshot: Optional[OperationSnapshot] = None
        self.validator = IntegrityValidator()
        self._snapshot_manager = self._get_snapshot_manager()
        
        self._start_time: Optional[float] = None
        self._completed = False
        self._rolled_back = False
    
    async def __aenter__(self) -> "BulkOperationGuardian":
        """Create snapshot on entry."""
        self._start_time = time.time()
        
        # Create snapshot
        loop = asyncio.get_event_loop()
        self.snapshot = await loop.run_in_executor(
            None,
            lambda: self._snapshot_manager.create_operation_snapshot(
                self.files,
                self.operation_type,
                self.description
            )
        )
        
        if self.snapshot:
            self.snapshot.status = OperationStatus.IN_PROGRESS
        
        logger.info(f"🔒 Started protected operation: {self.description or self.operation_type.value}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Validate and optionally rollback on exit."""
        duration = time.time() - self._start_time if self._start_time else 0
        
        if exc_type is not None:
            # Exception occurred - rollback if enabled
            logger.error(f"❌ Operation failed: {exc_val}")
            
            if self.auto_rollback and self.snapshot:
                logger.info("♻️ Initiating automatic rollback...")
                
                loop = asyncio.get_event_loop()
                success, failed = await loop.run_in_executor(
                    None,
                    lambda: self._snapshot_manager.restore_operation(self.snapshot)
                )
                
                if failed == 0:
                    logger.info(f"✅ Rollback complete: {success} files restored")
                    self._rolled_back = True
                else:
                    logger.error(f"⚠️ Partial rollback: {success} restored, {failed} failed")
                
                if self.snapshot:
                    self.snapshot.status = OperationStatus.ROLLED_BACK
                    self.snapshot.error_message = str(exc_val)
            
            # Don't suppress the exception
            return False
        
        # Operation completed - validate if enabled
        if self.validate_after and self.files:
            logger.info("🔍 Validating file integrity...")
            
            all_valid, errors = self.validator.validate_files(self.files)
            
            if not all_valid:
                logger.error("❌ Integrity validation failed:")
                for error in errors[:10]:  # Limit output
                    logger.error(f"  {error}")
                
                if self.auto_rollback and self.snapshot:
                    logger.info("♻️ Rolling back due to validation failure...")
                    
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self._snapshot_manager.restore_operation(self.snapshot)
                    )
                    
                    self._rolled_back = True
                    
                    if self.snapshot:
                        self.snapshot.status = OperationStatus.ROLLED_BACK
                        self.snapshot.error_message = "Validation failed: " + "; ".join(errors[:3])
                    
                    # Raise an exception to signal failure
                    raise RuntimeError(f"File integrity validation failed: {errors[0]}")
        
        # Mark as completed
        self._completed = True
        if self.snapshot:
            self.snapshot.status = OperationStatus.COMPLETED
            self.snapshot.completed_at = datetime.now()
        
        logger.info(f"✅ Operation completed in {duration:.2f}s: {self.description or self.operation_type.value}")
        return False
    
    def __enter__(self) -> "BulkOperationGuardian":
        """Synchronous context manager entry."""
        self._start_time = time.time()
        
        self.snapshot = self._snapshot_manager.create_operation_snapshot(
            self.files,
            self.operation_type,
            self.description
        )
        
        if self.snapshot:
            self.snapshot.status = OperationStatus.IN_PROGRESS
        
        logger.info(f"🔒 Started protected operation: {self.description or self.operation_type.value}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Synchronous context manager exit."""
        duration = time.time() - self._start_time if self._start_time else 0
        
        if exc_type is not None:
            logger.error(f"❌ Operation failed: {exc_val}")
            
            if self.auto_rollback and self.snapshot:
                logger.info("♻️ Initiating automatic rollback...")
                success, failed = self._snapshot_manager.restore_operation(self.snapshot)
                
                if failed == 0:
                    logger.info(f"✅ Rollback complete: {success} files restored")
                    self._rolled_back = True
                else:
                    logger.error(f"⚠️ Partial rollback: {success} restored, {failed} failed")
                
                if self.snapshot:
                    self.snapshot.status = OperationStatus.ROLLED_BACK
                    self.snapshot.error_message = str(exc_val)
            
            return False
        
        if self.validate_after and self.files:
            all_valid, errors = self.validator.validate_files(self.files)
            
            if not all_valid:
                logger.error("❌ Integrity validation failed")
                
                if self.auto_rollback and self.snapshot:
                    self._snapshot_manager.restore_operation(self.snapshot)
                    self._rolled_back = True
                    
                    if self.snapshot:
                        self.snapshot.status = OperationStatus.ROLLED_BACK
                    
                    raise RuntimeError(f"File integrity validation failed: {errors[0]}")
        
        self._completed = True
        if self.snapshot:
            self.snapshot.status = OperationStatus.COMPLETED
            self.snapshot.completed_at = datetime.now()
        
        logger.info(f"✅ Operation completed in {duration:.2f}s")
        return False
    
    @classmethod
    def protect(
        cls,
        files: List[str],
        operation_type: OperationType = OperationType.CUSTOM,
        description: str = "",
        auto_rollback: bool = True,
        validate_after: bool = True,
    ) -> "BulkOperationGuardian":
        """
        Create a guardian for protecting a bulk operation.
        
        Args:
            files: List of file paths to protect
            operation_type: Type of operation
            description: Human-readable description
            auto_rollback: Whether to automatically rollback on failure
            validate_after: Whether to validate files after operation
            
        Returns:
            BulkOperationGuardian context manager
        """
        return cls(
            files=files,
            operation_type=operation_type,
            description=description,
            auto_rollback=auto_rollback,
            validate_after=validate_after,
        )
    
    def rollback(self) -> bool:
        """Manually trigger a rollback."""
        if self._rolled_back:
            logger.warning("Already rolled back")
            return True
        
        if not self.snapshot:
            logger.error("No snapshot available for rollback")
            return False
        
        success, failed = self._snapshot_manager.restore_operation(self.snapshot)
        self._rolled_back = failed == 0
        
        if self._rolled_back:
            self.snapshot.status = OperationStatus.ROLLED_BACK
        
        return self._rolled_back
    
    @property
    def was_rolled_back(self) -> bool:
        return self._rolled_back
    
    @property
    def is_completed(self) -> bool:
        return self._completed


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

def get_snapshot_manager() -> SnapshotManager:
    """Get the snapshot manager instance."""
    return BulkOperationGuardian._get_snapshot_manager()


def list_snapshots() -> List[OperationSnapshot]:
    """List available operation snapshots."""
    return get_snapshot_manager().get_operation_snapshots()


def restore_snapshot(operation_id: str) -> bool:
    """Restore files from a specific snapshot."""
    manager = get_snapshot_manager()
    snapshot_dir = manager.base_dir / operation_id
    
    if not snapshot_dir.exists():
        logger.error(f"Snapshot not found: {operation_id}")
        return False
    
    metadata_path = snapshot_dir / "operation.json"
    if not metadata_path.exists():
        logger.error(f"Snapshot metadata not found: {operation_id}")
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Load file snapshots
        file_snapshots = []
        for file_path in snapshot_dir.glob("*"):
            if file_path.name == "operation.json":
                continue
            
            # Find original path from metadata or filename
            file_snapshots.append(FileSnapshot(
                original_path="",  # Would need to store this
                snapshot_path=str(file_path),
                content_hash="",
                file_size=file_path.stat().st_size,
            ))
        
        # This is a simplified restore - full implementation would need
        # the original paths stored in metadata
        logger.warning("Full restore requires original path metadata")
        return False
        
    except Exception as e:
        logger.error(f"Failed to restore snapshot: {e}")
        return False


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bulk Operation Guardian - Automatic backup and rollback"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available snapshots")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from snapshot")
    restore_parser.add_argument("operation_id", help="Operation ID to restore")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    if args.command == "list":
        snapshots = list_snapshots()
        
        if not snapshots:
            print("No snapshots available")
            return
        
        print(f"\n{'='*60}")
        print("Available Operation Snapshots")
        print(f"{'='*60}\n")
        
        for s in snapshots:
            print(f"  ID: {s.operation_id}")
            print(f"  Type: {s.operation_type.value}")
            print(f"  Description: {s.description or 'N/A'}")
            print(f"  Status: {s.status.value}")
            print(f"  Time: {s.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        print(f"Total: {len(snapshots)} snapshots")
    
    elif args.command == "restore":
        success = restore_snapshot(args.operation_id)
        if success:
            print(f"✅ Restored from {args.operation_id}")
        else:
            print(f"❌ Failed to restore from {args.operation_id}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

