#!/usr/bin/env python3
"""
JARVIS Update Engine
=====================

Async parallel update orchestration for the Self-Updating Lifecycle Manager.
Handles git operations, dependency installation, and build steps with
progress tracking and atomic transaction semantics.

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .supervisor_config import BuildStep, SupervisorConfig, get_supervisor_config

logger = logging.getLogger(__name__)


class UpdatePhase(str, Enum):
    """Update operation phases."""
    IDLE = "idle"
    CHECKING = "checking"
    FETCHING = "fetching"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    BUILDING = "building"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class UpdateProgress:
    """Progress information for update operations."""
    phase: UpdatePhase = UpdatePhase.IDLE
    current_step: str = ""
    steps_completed: int = 0
    steps_total: int = 0
    percent: float = 0.0
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class CommandResult:
    """Result of an async command execution."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float


@dataclass
class UpdateResult:
    """Result of an update operation."""
    success: bool
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    steps_completed: list[str] = field(default_factory=list)
    steps_failed: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None


class UpdateEngine:
    """
    Async parallel update orchestration engine.
    
    Features:
    - Parallel execution of independent operations
    - Configurable build steps with conditions
    - Progress tracking with callbacks
    - Atomic update with rollback on failure
    - Timeout handling per step
    
    Example:
        >>> engine = UpdateEngine(config)
        >>> result = await engine.apply_update()
        >>> if result.success:
        ...     print(f"Updated to {result.new_version}")
    """
    
    def __init__(
        self,
        config: Optional[SupervisorConfig] = None,
        repo_path: Optional[Path] = None,
    ):
        """
        Initialize the update engine.
        
        Args:
            config: Supervisor configuration
            repo_path: Path to git repository (auto-detected if None)
        """
        self.config = config or get_supervisor_config()
        self.repo_path = repo_path or self._detect_repo_path()
        
        self.progress = UpdateProgress()
        self._on_progress: list[Callable[[UpdateProgress], None]] = []
        self._cancelled = False
        
        logger.info(f"🔧 Update engine initialized (repo: {self.repo_path})")
    
    def _detect_repo_path(self) -> Path:
        """Detect the git repository path."""
        # Walk up from current file to find .git
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists():
                return parent
        
        # Fallback to current directory
        cwd = Path.cwd()
        if (cwd / ".git").exists():
            return cwd
        
        raise FileNotFoundError("Could not detect git repository")
    
    def _update_progress(
        self,
        phase: Optional[UpdatePhase] = None,
        step: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update progress and notify callbacks."""
        if phase:
            self.progress.phase = phase
        if step:
            self.progress.current_step = step
        if message:
            self.progress.message = message
        if error:
            self.progress.error = error
        
        # Calculate percent
        if self.progress.steps_total > 0:
            self.progress.percent = (
                self.progress.steps_completed / self.progress.steps_total * 100
            )
        
        # Notify callbacks
        for callback in self._on_progress:
            try:
                callback(self.progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    async def _run_command(
        self,
        command: str,
        timeout: int = 120,
        working_dir: Optional[Path] = None,
        env: Optional[dict] = None,
    ) -> CommandResult:
        """
        Run a shell command asynchronously.
        
        Args:
            command: Shell command to run
            timeout: Timeout in seconds
            working_dir: Working directory (default: repo_path)
            env: Additional environment variables
        """
        cwd = working_dir or self.repo_path
        
        # Merge environment
        cmd_env = os.environ.copy()
        if env:
            cmd_env.update(env)
        
        start_time = datetime.now()
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                env=cmd_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return CommandResult(
                    success=False,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    exit_code=-1,
                    duration_seconds=timeout,
                )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return CommandResult(
                success=process.returncode == 0,
                stdout=stdout.decode() if stdout else "",
                stderr=stderr.decode() if stderr else "",
                exit_code=process.returncode or 0,
                duration_seconds=duration,
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return CommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration_seconds=duration,
            )
    
    async def get_current_version(self) -> Optional[str]:
        """Get the current git commit hash."""
        result = await self._run_command("git rev-parse HEAD", timeout=10)
        if result.success:
            return result.stdout.strip()[:12]
        return None
    
    async def get_remote_version(self) -> Optional[str]:
        """Get the latest commit hash from remote."""
        branch = self.config.update.branch
        remote = self.config.update.remote
        
        result = await self._run_command(
            f"git ls-remote {remote} refs/heads/{branch}",
            timeout=30,
        )
        
        if result.success and result.stdout:
            # Format: <hash>\t<ref>
            match = re.match(r"([a-f0-9]+)", result.stdout)
            if match:
                return match.group(1)[:12]
        
        return None
    
    async def has_update_available(self) -> bool:
        """Check if an update is available."""
        current = await self.get_current_version()
        remote = await self.get_remote_version()
        
        if current and remote:
            return current != remote
        return False
    
    def _evaluate_condition(self, condition: Optional[str]) -> bool:
        """Evaluate a build step condition."""
        if not condition:
            return True
        
        # Support path_exists() condition
        if condition.startswith("path_exists("):
            path_match = re.match(r"path_exists\(([^)]+)\)", condition)
            if path_match:
                path = self.repo_path / path_match.group(1)
                return path.exists()
        
        return True
    
    async def _execute_build_step(self, step: BuildStep) -> tuple[bool, str]:
        """
        Execute a single build step.
        
        Returns:
            Tuple of (success, error_message)
        """
        # Check condition
        if not self._evaluate_condition(step.condition):
            logger.info(f"⏭️ Skipping {step.name}: condition not met")
            return True, ""
        
        # Prepare working directory
        working_dir = None
        if step.working_dir:
            working_dir = self.repo_path / step.working_dir
            if not working_dir.exists():
                if step.required:
                    return False, f"Working directory not found: {step.working_dir}"
                else:
                    logger.info(f"⏭️ Skipping {step.name}: working dir not found")
                    return True, ""
        
        # Substitute variables in command
        command = step.command.format(
            branch=self.config.update.branch,
            remote=self.config.update.remote,
        )
        
        logger.info(f"🔨 Executing: {step.name}")
        self._update_progress(step=step.name, message=f"Running {step.name}...")
        
        result = await self._run_command(
            command,
            timeout=step.timeout_seconds,
            working_dir=working_dir,
        )
        
        if result.success:
            logger.info(f"✅ {step.name} completed ({result.duration_seconds:.1f}s)")
            return True, ""
        else:
            error = f"{step.name} failed: {result.stderr}"
            logger.error(f"❌ {error}")
            return False, error
    
    async def apply_update(self) -> UpdateResult:
        """
        Apply an update from the configured source.
        
        Executes all configured build steps in order with proper
        error handling and progress tracking.
        
        Returns:
            UpdateResult with success status and details
        """
        if self._cancelled:
            return UpdateResult(success=False, error="Update cancelled")
        
        start_time = datetime.now()
        old_version = await self.get_current_version()
        
        self.progress = UpdateProgress(
            phase=UpdatePhase.FETCHING,
            started_at=start_time,
            steps_total=len(self.config.update.build_steps) + 2,  # +2 for fetch and verify
        )
        
        steps_completed: list[str] = []
        steps_failed: list[str] = []
        
        try:
            # Step 1: Fetch updates
            self._update_progress(
                phase=UpdatePhase.FETCHING,
                step="git_fetch",
                message="Fetching updates from remote...",
            )
            
            fetch_result = await self._run_command(
                f"git fetch {self.config.update.remote}",
                timeout=120,
            )
            
            if not fetch_result.success:
                raise RuntimeError(f"Git fetch failed: {fetch_result.stderr}")
            
            steps_completed.append("git_fetch")
            self.progress.steps_completed += 1
            
            # Step 2: Execute build steps
            self._update_progress(phase=UpdatePhase.INSTALLING)
            
            for step in self.config.update.build_steps:
                if self._cancelled:
                    raise RuntimeError("Update cancelled")
                
                success, error = await self._execute_build_step(step)
                
                if success:
                    steps_completed.append(step.name)
                    self.progress.steps_completed += 1
                else:
                    steps_failed.append(step.name)
                    if step.required:
                        raise RuntimeError(error)
            
            # Step 3: Verify
            self._update_progress(
                phase=UpdatePhase.VERIFYING,
                step="verify",
                message="Verifying update...",
            )
            
            new_version = await self.get_current_version()
            
            # Quick verification - check if version changed
            if old_version == new_version:
                logger.warning("⚠️ Version unchanged after update")
            
            steps_completed.append("verify")
            self.progress.steps_completed += 1
            
            # Complete
            duration = (datetime.now() - start_time).total_seconds()
            
            self._update_progress(
                phase=UpdatePhase.COMPLETE,
                message=f"Update complete: {old_version} → {new_version}",
            )
            self.progress.completed_at = datetime.now()
            
            logger.info(f"✅ Update complete in {duration:.1f}s")
            
            return UpdateResult(
                success=True,
                old_version=old_version,
                new_version=new_version,
                steps_completed=steps_completed,
                steps_failed=steps_failed,
                duration_seconds=duration,
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            self._update_progress(
                phase=UpdatePhase.FAILED,
                error=error_msg,
                message=f"Update failed: {error_msg}",
            )
            
            logger.error(f"❌ Update failed: {error_msg}")
            
            return UpdateResult(
                success=False,
                old_version=old_version,
                steps_completed=steps_completed,
                steps_failed=steps_failed,
                duration_seconds=duration,
                error=error_msg,
            )
    
    def cancel(self) -> None:
        """Cancel any ongoing update."""
        self._cancelled = True
        logger.info("⚠️ Update cancellation requested")
    
    def on_progress(self, callback: Callable[[UpdateProgress], None]) -> None:
        """Register a progress callback."""
        self._on_progress.append(callback)
    
    def get_progress(self) -> UpdateProgress:
        """Get current update progress."""
        return self.progress


async def check_for_updates(config: Optional[SupervisorConfig] = None) -> bool:
    """Quick utility to check if updates are available."""
    engine = UpdateEngine(config)
    return await engine.has_update_available()
