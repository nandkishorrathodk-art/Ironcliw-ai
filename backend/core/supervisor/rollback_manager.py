#!/usr/bin/env python3
"""
JARVIS Rollback Manager
========================

Version history tracking and rollback logic for the Self-Updating Lifecycle Manager.
Maintains snapshots of working versions and provides atomic rollback capability.

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .supervisor_config import SupervisorConfig, get_supervisor_config

logger = logging.getLogger(__name__)


@dataclass
class VersionSnapshot:
    """Snapshot of a working JARVIS version."""
    id: int = 0
    git_commit: str = ""
    git_branch: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    pip_freeze: Optional[str] = None  # Output of pip freeze
    is_stable: bool = False  # Confirmed working
    boot_count: int = 0
    crash_count: int = 0
    notes: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "timestamp": self.timestamp.isoformat(),
            "pip_freeze": self.pip_freeze,
            "is_stable": self.is_stable,
            "boot_count": self.boot_count,
            "crash_count": self.crash_count,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VersionSnapshot:
        """Create from dictionary."""
        return cls(
            id=data.get("id", 0),
            git_commit=data.get("git_commit", ""),
            git_branch=data.get("git_branch", ""),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            pip_freeze=data.get("pip_freeze"),
            is_stable=data.get("is_stable", False),
            boot_count=data.get("boot_count", 0),
            crash_count=data.get("crash_count", 0),
            notes=data.get("notes", ""),
        )


class RollbackManager:
    """
    Version history and rollback management.
    
    Features:
    - SQLite-based version history
    - Git reflog integration
    - pip freeze snapshots
    - Automatic rollback on boot failure
    - Configurable rollback depth
    
    Example:
        >>> manager = RollbackManager(config)
        >>> await manager.initialize()
        >>> await manager.create_snapshot()
        >>> # ... after failed update ...
        >>> await manager.rollback()
    """
    
    def __init__(
        self,
        config: Optional[SupervisorConfig] = None,
        repo_path: Optional[Path] = None,
    ):
        """
        Initialize the rollback manager.
        
        Args:
            config: Supervisor configuration
            repo_path: Path to git repository
        """
        self.config = config or get_supervisor_config()
        self.repo_path = repo_path or self._detect_repo_path()
        
        # Database path
        db_path = self.repo_path / self.config.rollback.history_db
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False
        
        logger.info(f"🔧 Rollback manager initialized (db: {self.db_path})")
    
    def _detect_repo_path(self) -> Path:
        """Detect the git repository path."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists():
                return parent
        return Path.cwd()
    
    async def initialize(self) -> None:
        """Initialize the database."""
        if self._initialized:
            return
        
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        
        # Create tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS version_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                git_commit TEXT NOT NULL,
                git_branch TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                pip_freeze TEXT,
                is_stable INTEGER DEFAULT 0,
                boot_count INTEGER DEFAULT 0,
                crash_count INTEGER DEFAULT 0,
                notes TEXT DEFAULT ''
            )
        """)
        
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_commit ON version_history(git_commit)
        """)
        
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stable ON version_history(is_stable)
        """)
        
        self._conn.commit()
        self._initialized = True
        
        logger.info("✅ Rollback database initialized")
    
    async def _run_command(self, command: str, timeout: int = 30) -> tuple[bool, str]:
        """Run a shell command asynchronously."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            if process.returncode == 0:
                return True, stdout.decode().strip()
            else:
                return False, stderr.decode().strip()
                
        except asyncio.TimeoutError:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
    
    async def get_current_commit(self) -> str:
        """Get the current git commit hash."""
        success, output = await self._run_command("git rev-parse HEAD")
        return output[:40] if success else ""
    
    async def get_current_branch(self) -> str:
        """Get the current git branch."""
        success, output = await self._run_command("git rev-parse --abbrev-ref HEAD")
        return output if success else "unknown"
    
    async def get_pip_freeze(self) -> Optional[str]:
        """Get pip freeze output if enabled."""
        if not self.config.rollback.include_pip_freeze:
            return None
        
        success, output = await self._run_command("pip freeze", timeout=60)
        return output if success else None
    
    async def create_snapshot(self, notes: str = "") -> Optional[VersionSnapshot]:
        """
        Create a snapshot of the current version.
        
        Args:
            notes: Optional notes about this snapshot
            
        Returns:
            The created snapshot, or None on failure
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            commit = await self.get_current_commit()
            branch = await self.get_current_branch()
            pip_freeze = await self.get_pip_freeze()
            
            snapshot = VersionSnapshot(
                git_commit=commit,
                git_branch=branch,
                timestamp=datetime.now(),
                pip_freeze=pip_freeze,
                notes=notes,
            )
            
            cursor = self._conn.execute(
                """
                INSERT INTO version_history (
                    git_commit, git_branch, timestamp, pip_freeze, notes
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    snapshot.git_commit,
                    snapshot.git_branch,
                    snapshot.timestamp.isoformat(),
                    snapshot.pip_freeze,
                    snapshot.notes,
                ),
            )
            
            self._conn.commit()
            snapshot.id = cursor.lastrowid
            
            # Cleanup old snapshots
            await self._cleanup_old_snapshots()
            
            logger.info(f"📸 Snapshot created: {commit[:12]} ({branch})")
            return snapshot
            
        except Exception as e:
            logger.error(f"❌ Failed to create snapshot: {e}")
            return None
    
    async def _cleanup_old_snapshots(self) -> None:
        """Remove old snapshots beyond max_versions."""
        max_versions = self.config.rollback.max_versions
        
        # Keep stable versions separate
        self._conn.execute(
            """
            DELETE FROM version_history
            WHERE id NOT IN (
                SELECT id FROM version_history
                WHERE is_stable = 1
                ORDER BY timestamp DESC
                LIMIT ?
            )
            AND id NOT IN (
                SELECT id FROM version_history
                WHERE is_stable = 0
                ORDER BY timestamp DESC
                LIMIT ?
            )
            """,
            (max_versions, max_versions),
        )
        
        self._conn.commit()
    
    async def mark_stable(self, commit: Optional[str] = None) -> bool:
        """Mark a version as stable (confirmed working)."""
        if not self._initialized:
            await self.initialize()
        
        if commit is None:
            commit = await self.get_current_commit()
        
        try:
            self._conn.execute(
                "UPDATE version_history SET is_stable = 1 WHERE git_commit = ?",
                (commit,),
            )
            self._conn.commit()
            
            logger.info(f"✅ Marked {commit[:12]} as stable")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to mark stable: {e}")
            return False
    
    async def record_boot(self, commit: Optional[str] = None) -> None:
        """Record a successful boot for a version."""
        if not self._initialized:
            await self.initialize()
        
        if commit is None:
            commit = await self.get_current_commit()
        
        try:
            self._conn.execute(
                "UPDATE version_history SET boot_count = boot_count + 1 WHERE git_commit = ?",
                (commit,),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record boot: {e}")
    
    async def record_crash(self, commit: Optional[str] = None) -> None:
        """Record a crash for a version."""
        if not self._initialized:
            await self.initialize()
        
        if commit is None:
            commit = await self.get_current_commit()
        
        try:
            self._conn.execute(
                "UPDATE version_history SET crash_count = crash_count + 1 WHERE git_commit = ?",
                (commit,),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record crash: {e}")
    
    async def get_last_stable(self) -> Optional[VersionSnapshot]:
        """Get the most recent stable version."""
        if not self._initialized:
            await self.initialize()
        
        cursor = self._conn.execute(
            """
            SELECT * FROM version_history
            WHERE is_stable = 1
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        
        row = cursor.fetchone()
        if row:
            return VersionSnapshot(
                id=row["id"],
                git_commit=row["git_commit"],
                git_branch=row["git_branch"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                pip_freeze=row["pip_freeze"],
                is_stable=bool(row["is_stable"]),
                boot_count=row["boot_count"],
                crash_count=row["crash_count"],
                notes=row["notes"],
            )
        
        return None
    
    async def get_previous_version(self) -> Optional[VersionSnapshot]:
        """Get the previous version (for rollback)."""
        if not self._initialized:
            await self.initialize()
        
        current_commit = await self.get_current_commit()
        
        # First try to get the last stable version
        stable = await self.get_last_stable()
        if stable and stable.git_commit != current_commit:
            return stable
        
        # Otherwise get the previous snapshot
        cursor = self._conn.execute(
            """
            SELECT * FROM version_history
            WHERE git_commit != ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (current_commit,),
        )
        
        row = cursor.fetchone()
        if row:
            return VersionSnapshot(
                id=row["id"],
                git_commit=row["git_commit"],
                git_branch=row["git_branch"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                pip_freeze=row["pip_freeze"],
                is_stable=bool(row["is_stable"]),
                boot_count=row["boot_count"],
                crash_count=row["crash_count"],
                notes=row["notes"],
            )
        
        return None
    
    async def rollback(self, target: Optional[VersionSnapshot] = None) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            target: Target version (default: previous version)
            
        Returns:
            True if rollback succeeded
        """
        if not self.config.rollback.enabled:
            logger.warning("⚠️ Rollback is disabled in config")
            return False
        
        if not self._initialized:
            await self.initialize()
        
        # Get target version
        if target is None:
            target = await self.get_previous_version()
        
        if target is None:
            logger.error("❌ No previous version available for rollback")
            return False
        
        logger.info(f"🔄 Rolling back to {target.git_commit[:12]} ({target.git_branch})")
        
        try:
            # Git reset to target commit
            success, output = await self._run_command(
                f"git reset --hard {target.git_commit}",
                timeout=60,
            )
            
            if not success:
                logger.error(f"❌ Git reset failed: {output}")
                return False
            
            # Restore pip dependencies if available
            if target.pip_freeze and self.config.rollback.include_pip_freeze:
                logger.info("📦 Restoring pip dependencies...")
                
                # Write pip freeze to temp file
                req_file = self.repo_path / ".rollback_requirements.txt"
                req_file.write_text(target.pip_freeze)
                
                success, output = await self._run_command(
                    f"pip install -r {req_file} --quiet",
                    timeout=300,
                )
                
                req_file.unlink(missing_ok=True)
                
                if not success:
                    logger.warning(f"⚠️ pip install failed: {output}")
                    # Continue anyway, git rollback is more important
            
            logger.info(f"✅ Rollback complete to {target.git_commit[:12]}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return False
    
    async def rollback_using_reflog(self, steps: int = 1) -> bool:
        """
        Rollback using git reflog (useful if no snapshots).
        
        Args:
            steps: Number of steps to go back in reflog
            
        Returns:
            True if rollback succeeded
        """
        logger.info(f"🔄 Rolling back {steps} step(s) via reflog")
        
        try:
            success, output = await self._run_command(
                f"git reset --hard HEAD@{{{steps}}}",
                timeout=60,
            )
            
            if success:
                logger.info(f"✅ Reflog rollback complete")
                return True
            else:
                logger.error(f"❌ Reflog rollback failed: {output}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Reflog rollback error: {e}")
            return False
    
    async def get_history(self, limit: int = 10) -> list[VersionSnapshot]:
        """Get version history."""
        if not self._initialized:
            await self.initialize()
        
        cursor = self._conn.execute(
            """
            SELECT * FROM version_history
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        
        snapshots = []
        for row in cursor.fetchall():
            snapshots.append(VersionSnapshot(
                id=row["id"],
                git_commit=row["git_commit"],
                git_branch=row["git_branch"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                pip_freeze=row["pip_freeze"],
                is_stable=bool(row["is_stable"]),
                boot_count=row["boot_count"],
                crash_count=row["crash_count"],
                notes=row["notes"],
            ))
        
        return snapshots
    
    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False
