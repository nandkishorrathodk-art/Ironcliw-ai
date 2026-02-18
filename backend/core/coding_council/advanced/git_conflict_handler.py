"""
v77.2: Git Conflict Resolution Handler - Gap #62
==================================================

Handles git merge conflicts during distributed transactions.

Problem:
    During 2PC, git merge can fail with conflicts:
    - Both repos modified same file differently
    - Semantic conflicts (function renamed in one, called in another)
    - Content conflicts within files

Solution:
    - Detect conflicts before they happen (dry-run merge)
    - Parse conflict markers to understand conflicts
    - Provide resolution strategies:
        1. Ours (keep local changes)
        2. Theirs (accept incoming changes)
        3. Union (merge both non-overlapping changes)
        4. Manual (flag for human review)
    - Auto-resolve simple conflicts
    - Track unresolved for human review

Features:
    - Pre-merge conflict detection
    - Conflict parsing and classification
    - Automatic resolution for safe conflicts
    - Conflict reporting and tracking
    - Recovery from failed merges

Author: JARVIS v77.2
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of git conflicts."""

    CONTENT = "content"  # Both modified same lines
    ADD_ADD = "add_add"  # Both added same file
    MODIFY_DELETE = "modify_delete"  # One modified, other deleted
    RENAME_RENAME = "rename_rename"  # Both renamed same file differently
    RENAME_DELETE = "rename_delete"  # One renamed, other deleted


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""

    OURS = "ours"  # Keep local changes
    THEIRS = "theirs"  # Accept incoming changes
    UNION = "union"  # Merge non-overlapping changes
    MANUAL = "manual"  # Requires human review
    ABORT = "abort"  # Abort the merge


@dataclass
class ConflictHunk:
    """A single conflict hunk within a file."""

    file_path: str
    line_start: int
    line_end: int
    ours_content: List[str]
    theirs_content: List[str]
    ancestor_content: Optional[List[str]] = None  # Common ancestor (3-way merge)
    conflict_type: ConflictType = ConflictType.CONTENT
    suggested_resolution: ResolutionStrategy = ResolutionStrategy.MANUAL

    @property
    def is_simple(self) -> bool:
        """Check if conflict is simple (can be auto-resolved)."""
        # Simple cases:
        # 1. One side is empty (addition vs no change)
        # 2. Both sides made identical changes
        # 3. Changes don't overlap semantically
        if not self.ours_content or not self.theirs_content:
            return True
        if self.ours_content == self.theirs_content:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "ours_lines": len(self.ours_content),
            "theirs_lines": len(self.theirs_content),
            "conflict_type": self.conflict_type.value,
            "suggested_resolution": self.suggested_resolution.value,
            "is_simple": self.is_simple,
        }


@dataclass
class FileConflict:
    """Conflict information for a single file."""

    file_path: str
    conflict_type: ConflictType
    hunks: List[ConflictHunk] = field(default_factory=list)
    can_auto_resolve: bool = False
    suggested_resolution: ResolutionStrategy = ResolutionStrategy.MANUAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "conflict_type": self.conflict_type.value,
            "hunk_count": len(self.hunks),
            "can_auto_resolve": self.can_auto_resolve,
            "suggested_resolution": self.suggested_resolution.value,
            "hunks": [h.to_dict() for h in self.hunks],
        }


@dataclass
class MergeConflictResult:
    """Result of conflict detection/resolution."""

    has_conflicts: bool
    conflicts: List[FileConflict] = field(default_factory=list)
    auto_resolved: List[str] = field(default_factory=list)  # Files auto-resolved
    needs_manual: List[str] = field(default_factory=list)  # Files needing review
    error: Optional[str] = None

    @property
    def total_hunks(self) -> int:
        return sum(len(c.hunks) for c in self.conflicts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_conflicts": self.has_conflicts,
            "total_files": len(self.conflicts),
            "total_hunks": self.total_hunks,
            "auto_resolved_count": len(self.auto_resolved),
            "needs_manual_count": len(self.needs_manual),
            "error": self.error,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }


class ConflictParser:
    """
    Parses git conflict markers from files.

    Understands both 2-way and 3-way merge markers.
    """

    # Conflict marker patterns
    MARKER_OURS = re.compile(r"^<<<<<<< (.+)$")
    MARKER_BASE = re.compile(r"^\|\|\|\|\|\|\| (.+)$")  # 3-way merge base
    MARKER_SEPARATOR = re.compile(r"^=======$")
    MARKER_THEIRS = re.compile(r"^>>>>>>> (.+)$")

    @classmethod
    def parse_file(cls, file_path: Path) -> List[ConflictHunk]:
        """
        Parse conflict hunks from a file.

        Args:
            file_path: Path to file with conflict markers

        Returns:
            List of ConflictHunk objects
        """
        if not file_path.exists():
            return []

        try:
            content = file_path.read_text()
        except Exception as e:
            logger.error(f"[ConflictParser] Could not read {file_path}: {e}")
            return []

        hunks = []
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            # Look for conflict start
            match = cls.MARKER_OURS.match(lines[i])
            if match:
                hunk = cls._parse_hunk(lines, i, str(file_path))
                if hunk:
                    hunks.append(hunk)
                    i = hunk.line_end + 1
                    continue
            i += 1

        return hunks

    @classmethod
    def _parse_hunk(
        cls,
        lines: List[str],
        start: int,
        file_path: str,
    ) -> Optional[ConflictHunk]:
        """Parse a single conflict hunk."""
        ours_content = []
        base_content = []
        theirs_content = []

        in_ours = True
        in_base = False
        in_theirs = False
        end_line = start

        for i in range(start + 1, len(lines)):
            line = lines[i]

            # Check for markers
            if cls.MARKER_BASE.match(line):
                in_ours = False
                in_base = True
                in_theirs = False
                continue
            elif cls.MARKER_SEPARATOR.match(line):
                in_ours = False
                in_base = False
                in_theirs = True
                continue
            elif cls.MARKER_THEIRS.match(line):
                end_line = i
                break

            # Collect content
            if in_ours:
                ours_content.append(line)
            elif in_base:
                base_content.append(line)
            elif in_theirs:
                theirs_content.append(line)

        return ConflictHunk(
            file_path=file_path,
            line_start=start,
            line_end=end_line,
            ours_content=ours_content,
            theirs_content=theirs_content,
            ancestor_content=base_content if base_content else None,
        )


class ConflictDetector:
    """
    Detects merge conflicts before they happen.

    Uses git merge --no-commit --no-ff to simulate merge.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    async def detect(
        self,
        source_branch: str,
        target_branch: str = "main",
    ) -> MergeConflictResult:
        """
        Detect conflicts between branches without committing.

        Args:
            source_branch: Branch to merge
            target_branch: Branch to merge into

        Returns:
            MergeConflictResult with conflict details
        """
        # Store current branch
        original_branch = await self._get_current_branch()

        try:
            # Checkout target branch
            await self._run_git(["checkout", target_branch])

            # Try merge without committing
            _, stderr, returncode = await self._run_git(
                ["merge", "--no-commit", "--no-ff", source_branch],
                check=False,
            )

            if returncode == 0:
                # No conflicts
                await self._run_git(["merge", "--abort"], check=False)
                return MergeConflictResult(has_conflicts=False)

            # Parse conflicts
            conflicts = await self._parse_conflicts()

            # Abort the merge
            await self._run_git(["merge", "--abort"], check=False)

            return MergeConflictResult(
                has_conflicts=bool(conflicts),
                conflicts=conflicts,
                needs_manual=[c.file_path for c in conflicts if not c.can_auto_resolve],
            )

        except Exception as e:
            logger.error(f"[ConflictDetector] Error detecting conflicts: {e}")
            # Try to abort any partial merge
            await self._run_git(["merge", "--abort"], check=False)
            return MergeConflictResult(
                has_conflicts=True,
                error=str(e),
            )

        finally:
            # Restore original branch
            if original_branch:
                await self._run_git(["checkout", original_branch], check=False)

    async def _parse_conflicts(self) -> List[FileConflict]:
        """Parse all conflicted files."""
        # Get list of conflicted files
        stdout, _, _ = await self._run_git(["diff", "--name-only", "--diff-filter=U"])
        conflicted_files = [f.strip() for f in stdout.split("\n") if f.strip()]

        conflicts = []
        for file_name in conflicted_files:
            file_path = self.repo_path / file_name
            hunks = ConflictParser.parse_file(file_path)

            conflict = FileConflict(
                file_path=file_name,
                conflict_type=ConflictType.CONTENT,
                hunks=hunks,
                can_auto_resolve=all(h.is_simple for h in hunks),
            )

            # Suggest resolution
            if conflict.can_auto_resolve:
                conflict.suggested_resolution = ResolutionStrategy.UNION
            elif len(hunks) == 1 and not hunks[0].ours_content:
                conflict.suggested_resolution = ResolutionStrategy.THEIRS
            elif len(hunks) == 1 and not hunks[0].theirs_content:
                conflict.suggested_resolution = ResolutionStrategy.OURS

            conflicts.append(conflict)

        return conflicts

    async def _get_current_branch(self) -> Optional[str]:
        """Get current branch name."""
        try:
            stdout, _, _ = await self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
            return stdout.strip()
        except Exception:
            return None

    async def _run_git(
        self,
        args: List[str],
        check: bool = True,
    ) -> Tuple[str, str, int]:
        """Run git command."""
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if check and proc.returncode != 0:
            raise RuntimeError(f"Git command failed: {stderr.decode()}")

        return stdout.decode(), stderr.decode(), proc.returncode or 0


class ConflictResolver:
    """
    Resolves git merge conflicts.

    Supports automatic resolution for simple conflicts
    and manual resolution assistance.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    async def resolve_file(
        self,
        file_path: str,
        strategy: ResolutionStrategy,
    ) -> bool:
        """
        Resolve conflicts in a single file.

        Args:
            file_path: Relative path to conflicted file
            strategy: Resolution strategy to use

        Returns:
            True if resolved successfully
        """
        full_path = self.repo_path / file_path

        if strategy == ResolutionStrategy.OURS:
            return await self._resolve_ours(file_path)
        elif strategy == ResolutionStrategy.THEIRS:
            return await self._resolve_theirs(file_path)
        elif strategy == ResolutionStrategy.UNION:
            return await self._resolve_union(full_path)
        elif strategy == ResolutionStrategy.ABORT:
            await self._run_git(["checkout", "--", file_path])
            return False
        else:
            # MANUAL - just return False to indicate needs manual resolution
            return False

    async def resolve_all(
        self,
        conflicts: List[FileConflict],
        default_strategy: ResolutionStrategy = ResolutionStrategy.MANUAL,
    ) -> Tuple[List[str], List[str]]:
        """
        Resolve all conflicts.

        Args:
            conflicts: List of file conflicts
            default_strategy: Default strategy for non-auto-resolvable

        Returns:
            Tuple of (resolved files, unresolved files)
        """
        resolved = []
        unresolved = []

        for conflict in conflicts:
            strategy = conflict.suggested_resolution
            if strategy == ResolutionStrategy.MANUAL:
                strategy = default_strategy

            success = await self.resolve_file(conflict.file_path, strategy)

            if success:
                # Stage the resolved file
                await self._run_git(["add", conflict.file_path])
                resolved.append(conflict.file_path)
            else:
                unresolved.append(conflict.file_path)

        return resolved, unresolved

    async def _resolve_ours(self, file_path: str) -> bool:
        """Resolve using our version."""
        try:
            await self._run_git(["checkout", "--ours", file_path])
            return True
        except Exception as e:
            logger.error(f"[ConflictResolver] Failed to resolve with ours: {e}")
            return False

    async def _resolve_theirs(self, file_path: str) -> bool:
        """Resolve using their version."""
        try:
            await self._run_git(["checkout", "--theirs", file_path])
            return True
        except Exception as e:
            logger.error(f"[ConflictResolver] Failed to resolve with theirs: {e}")
            return False

    async def _resolve_union(self, file_path: Path) -> bool:
        """
        Resolve by merging both changes (union).

        This removes conflict markers and keeps all unique lines.
        """
        if not file_path.exists():
            return False

        try:
            content = file_path.read_text()
            hunks = ConflictParser.parse_file(file_path)

            if not hunks:
                return True  # No conflicts to resolve

            lines = content.split("\n")
            result_lines = []

            current_pos = 0
            for hunk in hunks:
                # Add lines before hunk
                result_lines.extend(lines[current_pos:hunk.line_start])

                # Merge hunk content
                merged = self._merge_union(hunk.ours_content, hunk.theirs_content)
                result_lines.extend(merged)

                current_pos = hunk.line_end + 1

            # Add remaining lines
            result_lines.extend(lines[current_pos:])

            # Write resolved content
            file_path.write_text("\n".join(result_lines))
            return True

        except Exception as e:
            logger.error(f"[ConflictResolver] Union merge failed: {e}")
            return False

    def _merge_union(
        self,
        ours: List[str],
        theirs: List[str],
    ) -> List[str]:
        """
        Merge two sets of lines, keeping all unique.

        Maintains order: ours first, then theirs (excluding duplicates).
        """
        seen = set()
        result = []

        for line in ours:
            if line not in seen:
                seen.add(line)
                result.append(line)

        for line in theirs:
            if line not in seen:
                seen.add(line)
                result.append(line)

        return result

    async def _run_git(
        self,
        args: List[str],
    ) -> Tuple[str, str, int]:
        """Run git command."""
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Git command failed: {stderr.decode()}")

        return stdout.decode(), stderr.decode(), proc.returncode or 0


class GitConflictHandler:
    """
    High-level handler for git conflicts in distributed transactions.

    Coordinates detection, resolution, and recovery.

    Usage:
        handler = GitConflictHandler(repo_path)

        # Before merge
        result = await handler.pre_merge_check("feature-branch")
        if result.has_conflicts:
            # Handle conflicts

        # After merge with conflicts
        resolved, unresolved = await handler.auto_resolve()

        # Get conflict report
        report = handler.get_conflict_report()
    """

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.detector = ConflictDetector(repo_path)
        self.resolver = ConflictResolver(repo_path)

        self._last_result: Optional[MergeConflictResult] = None
        self._resolution_history: List[Dict[str, Any]] = []

    async def pre_merge_check(
        self,
        source_branch: str,
        target_branch: str = "main",
    ) -> MergeConflictResult:
        """
        Check for conflicts before merging.

        Args:
            source_branch: Branch to merge
            target_branch: Branch to merge into

        Returns:
            MergeConflictResult with predictions
        """
        result = await self.detector.detect(source_branch, target_branch)
        self._last_result = result

        if result.has_conflicts:
            logger.warning(
                f"[GitConflictHandler] Detected {len(result.conflicts)} file(s) "
                f"with conflicts between {source_branch} and {target_branch}"
            )
        else:
            logger.info(
                f"[GitConflictHandler] No conflicts detected between "
                f"{source_branch} and {target_branch}"
            )

        return result

    async def detect_current_conflicts(self) -> MergeConflictResult:
        """
        Detect conflicts in current merge state.

        Use this when already in a merge with conflicts.
        """
        # Get list of conflicted files
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--name-only", "--diff-filter=U",
            cwd=str(self.repo_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        conflicted_files = [f.strip() for f in stdout.decode().split("\n") if f.strip()]

        if not conflicted_files:
            return MergeConflictResult(has_conflicts=False)

        conflicts = []
        for file_name in conflicted_files:
            file_path = self.repo_path / file_name
            hunks = ConflictParser.parse_file(file_path)

            conflicts.append(FileConflict(
                file_path=file_name,
                conflict_type=ConflictType.CONTENT,
                hunks=hunks,
                can_auto_resolve=all(h.is_simple for h in hunks),
            ))

        result = MergeConflictResult(
            has_conflicts=True,
            conflicts=conflicts,
            needs_manual=[c.file_path for c in conflicts if not c.can_auto_resolve],
        )

        self._last_result = result
        return result

    async def auto_resolve(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.UNION,
    ) -> Tuple[List[str], List[str]]:
        """
        Automatically resolve conflicts where possible.

        Args:
            default_strategy: Strategy for non-simple conflicts

        Returns:
            Tuple of (resolved files, unresolved files)
        """
        if not self._last_result or not self._last_result.has_conflicts:
            result = await self.detect_current_conflicts()
        else:
            result = self._last_result

        if not result.conflicts:
            return [], []

        resolved, unresolved = await self.resolver.resolve_all(
            result.conflicts,
            default_strategy,
        )

        # Update result
        result.auto_resolved = resolved
        result.needs_manual = unresolved

        # Record history
        self._resolution_history.append({
            "timestamp": asyncio.get_running_loop().time(),
            "resolved": resolved,
            "unresolved": unresolved,
            "strategy": default_strategy.value,
        })

        if resolved:
            logger.info(
                f"[GitConflictHandler] Auto-resolved {len(resolved)} file(s)"
            )
        if unresolved:
            logger.warning(
                f"[GitConflictHandler] {len(unresolved)} file(s) need manual resolution"
            )

        return resolved, unresolved

    async def abort_merge(self) -> bool:
        """
        Abort the current merge.

        Returns:
            True if aborted successfully
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "merge", "--abort",
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"[GitConflictHandler] Failed to abort merge: {e}")
            return False

    def get_conflict_report(self) -> Dict[str, Any]:
        """Get detailed conflict report."""
        if not self._last_result:
            return {"has_conflicts": False}

        return {
            **self._last_result.to_dict(),
            "resolution_history": self._resolution_history,
        }
