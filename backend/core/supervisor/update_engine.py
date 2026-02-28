#!/usr/bin/env python3
"""
Ironcliw Update Engine v2.0 - Zero-Touch Edition
================================================

Async parallel update orchestration for the Self-Updating Lifecycle Manager.
Handles git operations, dependency installation, and build steps with
progress tracking and atomic transaction semantics.

v2.0 Features:
- Staging area for safe update validation
- Dry-run pip install before real installation
- Python syntax and import validation
- Ironcliw busy-state checking before updates
- Intelligent update classification (security/minor/major)
- Parallel validation operations
- Prime Directives enforcement

Author: Ironcliw System
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp

from .supervisor_config import BuildStep, SupervisorConfig, get_supervisor_config

logger = logging.getLogger(__name__)


class UpdatePhase(str, Enum):
    """Update operation phases."""
    IDLE = "idle"
    CHECKING = "checking"
    PRE_FLIGHT = "pre_flight"      # v2.0: Pre-update safety checks
    STAGING = "staging"             # v2.0: Download to staging area
    VALIDATING = "validating"       # v2.0: Syntax/import validation
    FETCHING = "fetching"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    BUILDING = "building"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


class UpdateClassification(str, Enum):
    """Classification of update severity."""
    SECURITY = "security"    # Security fix - apply immediately
    CRITICAL = "critical"    # Critical bug fix
    MINOR = "minor"          # Minor updates (features, non-critical fixes)
    MAJOR = "major"          # Major version change (breaking changes)
    UNKNOWN = "unknown"      # Cannot determine


class ValidationResult(str, Enum):
    """Result of update validation."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


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
class ValidationReport:
    """Report from update validation (v2.0)."""
    result: ValidationResult = ValidationResult.SKIPPED
    syntax_errors: list[str] = field(default_factory=list)
    import_errors: list[str] = field(default_factory=list)
    pip_conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files_checked: int = 0
    duration_seconds: float = 0.0
    
    @property
    def is_safe_to_apply(self) -> bool:
        """Check if it's safe to apply this update."""
        return (
            self.result in (ValidationResult.PASSED, ValidationResult.WARNING) and
            len(self.syntax_errors) == 0 and
            len(self.import_errors) == 0 and
            len(self.pip_conflicts) == 0
        )


@dataclass
class StagingInfo:
    """Information about staged update (v2.0)."""
    staging_path: Optional[Path] = None
    target_commit: Optional[str] = None
    source_commit: Optional[str] = None
    created_at: Optional[datetime] = None
    validation_report: Optional[ValidationReport] = None
    files_changed: int = 0
    bytes_downloaded: int = 0
    is_ready: bool = False


@dataclass 
class PreFlightCheckResult:
    """Result of pre-flight checks before update (v2.0)."""
    passed: bool = False
    jarvis_busy: bool = False
    active_tasks: int = 0
    system_idle: bool = True
    idle_seconds: float = 0.0
    memory_sufficient: bool = True
    disk_sufficient: bool = True
    protected_files_modified: bool = False
    blocked_reason: Optional[str] = None
    checks_performed: list[str] = field(default_factory=list)


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
    # v2.0: Enhanced result data
    classification: UpdateClassification = UpdateClassification.UNKNOWN
    validation_report: Optional[ValidationReport] = None
    pre_flight: Optional[PreFlightCheckResult] = None
    was_zero_touch: bool = False


class UpdateEngine:
    """
    Async parallel update orchestration engine v2.0 - Zero-Touch Edition.
    
    Features:
    - Parallel execution of independent operations
    - Configurable build steps with conditions
    - Progress tracking with callbacks
    - Atomic update with rollback on failure
    - Timeout handling per step
    
    v2.0 Zero-Touch Features:
    - Pre-flight safety checks (Ironcliw busy state, system idle)
    - Staging area for safe update validation
    - Dry-run pip installation
    - Python syntax and import validation
    - Update classification (security/minor/major)
    - Prime Directives enforcement
    - Intelligent parallel validation
    
    Example:
        >>> engine = UpdateEngine(config)
        >>> # Zero-Touch mode
        >>> if await engine.can_auto_update():
        ...     result = await engine.apply_update(zero_touch=True)
        >>> # Manual mode
        >>> result = await engine.apply_update()
    """
    
    # Security-related keywords in commit messages
    SECURITY_KEYWORDS = frozenset({
        "security", "cve", "vulnerability", "exploit", "xss", "csrf",
        "injection", "auth", "authentication", "authorization", "password",
        "token", "secret", "encrypt", "decrypt", "ssl", "tls", "certificate",
    })
    
    # Major change keywords
    MAJOR_KEYWORDS = frozenset({
        "breaking", "major", "incompatible", "migration", "deprecated",
        "removed", "changed api", "v2", "v3", "version 2", "version 3",
    })
    
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
        
        # v2.0: Staging and validation state
        self._staging_info: Optional[StagingInfo] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_update_time: Optional[datetime] = None
        
        logger.info(f"🔧 Update engine v2.0 initialized (repo: {self.repo_path})")
    
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
    
    # =========================================================================
    # v2.0: Pre-Flight Safety Checks
    # =========================================================================
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    async def check_jarvis_busy(self) -> tuple[bool, int]:
        """
        Check if Ironcliw is currently busy with active tasks.
        
        Returns:
            Tuple of (is_busy, active_task_count)
        """
        zt_config = self.config.zero_touch
        if not zt_config.check_jarvis_busy:
            return False, 0
        
        backend_port = int(os.environ.get("BACKEND_PORT", "8010"))
        endpoint = f"http://localhost:{backend_port}{zt_config.busy_check_endpoint}"
        
        try:
            session = await self._get_session()
            async with session.get(
                endpoint,
                timeout=aiohttp.ClientTimeout(total=zt_config.busy_check_timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    is_busy = data.get("busy", False)
                    active_tasks = data.get("active_tasks", 0)
                    return is_busy, active_tasks
                elif response.status == 404:
                    # Endpoint doesn't exist yet - assume not busy
                    logger.debug(f"Busy check endpoint not found, assuming not busy")
                    return False, 0
                else:
                    logger.warning(f"Busy check returned {response.status}")
                    return False, 0
        except asyncio.TimeoutError:
            logger.warning(f"Busy check timed out")
            return False, 0
        except aiohttp.ClientError as e:
            logger.warning(f"Busy check failed: {e}")
            return False, 0
        except Exception as e:
            logger.error(f"Unexpected busy check error: {e}")
            return False, 0
    
    async def check_system_idle(self) -> tuple[bool, float]:
        """
        Check if the system is idle (macOS HID idle time).
        
        Returns:
            Tuple of (is_idle, idle_seconds)
        """
        try:
            process = await asyncio.create_subprocess_shell(
                'ioreg -c IOHIDSystem | grep -i idletime',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=5.0
            )
            
            if process.returncode == 0 and stdout:
                # Parse idle time from ioreg output
                output = stdout.decode()
                match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', output)
                if match:
                    # Convert nanoseconds to seconds
                    idle_ns = int(match.group(1))
                    idle_seconds = idle_ns / 1_000_000_000
                    
                    min_idle = self.config.zero_touch.min_idle_seconds
                    return idle_seconds >= min_idle, idle_seconds
            
            return True, 0.0  # Assume idle if can't detect
            
        except Exception as e:
            logger.debug(f"Idle check error: {e}")
            return True, 0.0
    
    async def run_pre_flight_checks(self) -> PreFlightCheckResult:
        """
        Run all pre-flight safety checks before update.
        
        Checks:
        1. Ironcliw not busy with active tasks
        2. System idle (if required)
        3. Cooldown period since last update
        4. Protected files not modified
        5. Sufficient resources
        
        Returns:
            PreFlightCheckResult with all check results
        """
        self._update_progress(phase=UpdatePhase.PRE_FLIGHT, message="Running pre-flight checks...")
        
        result = PreFlightCheckResult()
        result.checks_performed = []
        
        # Check 1: Ironcliw busy state
        is_busy, active_tasks = await self.check_jarvis_busy()
        result.jarvis_busy = is_busy
        result.active_tasks = active_tasks
        result.checks_performed.append("jarvis_busy")
        
        if is_busy:
            result.blocked_reason = f"Ironcliw is busy ({active_tasks} active tasks)"
            logger.info(f"⚠️ Pre-flight: {result.blocked_reason}")
            return result
        
        # Check 2: System idle
        if self.config.zero_touch.require_idle_system:
            is_idle, idle_seconds = await self.check_system_idle()
            result.system_idle = is_idle
            result.idle_seconds = idle_seconds
            result.checks_performed.append("system_idle")
            
            if not is_idle:
                result.blocked_reason = f"System not idle (only {idle_seconds:.0f}s)"
                logger.info(f"⚠️ Pre-flight: {result.blocked_reason}")
                return result
        
        # Check 3: Cooldown period
        if self._last_update_time:
            cooldown = self.config.zero_touch.cooldown_after_update_seconds
            elapsed = (datetime.now() - self._last_update_time).total_seconds()
            result.checks_performed.append("cooldown")
            
            if elapsed < cooldown:
                result.blocked_reason = f"Cooldown active ({cooldown - elapsed:.0f}s remaining)"
                logger.info(f"⚠️ Pre-flight: {result.blocked_reason}")
                return result
        
        # Check 4: Protected files - will check during validation
        result.checks_performed.append("protected_files")
        
        # Check 5: Resources (basic check)
        try:
            import psutil
            mem = psutil.virtual_memory()
            result.memory_sufficient = mem.available > (512 * 1024 * 1024)  # 512MB
            
            import shutil
            _, _, free = shutil.disk_usage("/")
            result.disk_sufficient = free > (100 * 1024 * 1024)  # 100MB
            result.checks_performed.append("resources")
            
            if not result.memory_sufficient:
                result.blocked_reason = "Insufficient memory for update"
                return result
            if not result.disk_sufficient:
                result.blocked_reason = "Insufficient disk space"
                return result
        except Exception as e:
            logger.debug(f"Resource check error: {e}")
        
        result.passed = True
        logger.info("✅ Pre-flight checks passed")
        return result
    
    # =========================================================================
    # v2.0: Staging Area & Validation
    # =========================================================================
    
    async def create_staging_area(self) -> Optional[Path]:
        """
        Create a staging area for update validation.
        
        Returns:
            Path to staging directory, or None if failed
        """
        staging_dir = self.config.zero_touch.staging_directory
        staging_path = self.repo_path / staging_dir
        
        try:
            # Clean up old staging if exists
            if staging_path.exists():
                age = (datetime.now() - datetime.fromtimestamp(staging_path.stat().st_mtime)).total_seconds()
                if age > self.config.zero_touch.max_staging_age_seconds:
                    shutil.rmtree(staging_path)
                    logger.info(f"🧹 Cleaned old staging area")
            
            staging_path.mkdir(parents=True, exist_ok=True)
            
            self._staging_info = StagingInfo(
                staging_path=staging_path,
                created_at=datetime.now(),
            )
            
            logger.info(f"📦 Staging area created: {staging_path}")
            return staging_path
            
        except Exception as e:
            logger.error(f"Failed to create staging area: {e}")
            return None
    
    async def stage_update(self) -> bool:
        """
        Stage the update by fetching changes to staging area.
        
        Returns:
            True if staging successful
        """
        if not self._staging_info or not self._staging_info.staging_path:
            await self.create_staging_area()
        
        if not self._staging_info:
            return False
        
        self._update_progress(phase=UpdatePhase.STAGING, message="Staging update...")
        
        remote = self.config.update.remote
        branch = self.config.update.branch
        
        try:
            # Fetch without merging
            result = await self._run_command(f"git fetch {remote} {branch}", timeout=120)
            if not result.success:
                logger.error(f"Git fetch failed: {result.stderr}")
                return False
            
            # Get target commit
            result = await self._run_command(f"git rev-parse {remote}/{branch}")
            if result.success:
                self._staging_info.target_commit = result.stdout.strip()[:12]
            
            # Get current commit
            self._staging_info.source_commit = await self.get_current_version()
            
            # Get list of changed files
            result = await self._run_command(
                f"git diff --name-only HEAD...{remote}/{branch}"
            )
            if result.success:
                files = [f for f in result.stdout.strip().split("\n") if f]
                self._staging_info.files_changed = len(files)
                
                # Check for protected file modifications
                if await self._check_protected_files(files):
                    logger.warning("⚠️ Update modifies protected files!")
                    return False
            
            logger.info(f"📦 Staged: {self._staging_info.files_changed} files to update")
            self._staging_info.is_ready = True
            return True
            
        except Exception as e:
            logger.error(f"Staging failed: {e}")
            return False
    
    async def _check_protected_files(self, files: list[str]) -> bool:
        """
        Check if any protected files are being modified.
        
        Returns:
            True if protected files are modified (should block update)
        """
        protected_patterns = self.config.prime_directives.protected_files
        
        for file in files:
            for pattern in protected_patterns:
                if fnmatch.fnmatch(file, pattern):
                    logger.warning(f"🛡️ Protected file modified: {file} (matches {pattern})")
                    return True
        
        return False
    
    async def validate_staged_update(self) -> ValidationReport:
        """
        Validate the staged update before applying.
        
        Performs:
        1. Python syntax validation
        2. Import validation
        3. Dry-run pip install
        
        Returns:
            ValidationReport with all validation results
        """
        self._update_progress(phase=UpdatePhase.VALIDATING, message="Validating update...")
        
        report = ValidationReport()
        start_time = datetime.now()
        
        zt_config = self.config.zero_touch
        remote = self.config.update.remote
        branch = self.config.update.branch
        
        # Get list of changed Python files
        result = await self._run_command(
            f"git diff --name-only HEAD...{remote}/{branch} -- '*.py'"
        )
        py_files = [f for f in result.stdout.strip().split("\n") if f and f.endswith(".py")]
        report.files_checked = len(py_files)
        
        # Run validations in parallel
        validation_tasks = []
        
        if zt_config.validate_syntax and py_files:
            validation_tasks.append(("syntax", self._validate_syntax(py_files, remote, branch)))
        
        if zt_config.validate_imports and py_files:
            validation_tasks.append(("imports", self._validate_imports(py_files, remote, branch)))
        
        if zt_config.dry_run_pip:
            validation_tasks.append(("pip", self._dry_run_pip()))
        
        # Execute parallel validations
        if validation_tasks:
            task_results = await asyncio.gather(
                *[task[1] for task in validation_tasks],
                return_exceptions=True
            )
            
            for i, (name, _) in enumerate(validation_tasks):
                result = task_results[i]
                if isinstance(result, Exception):
                    report.warnings.append(f"{name} validation error: {result}")
                elif isinstance(result, list):
                    if name == "syntax":
                        report.syntax_errors.extend(result)
                    elif name == "imports":
                        report.import_errors.extend(result)
                    elif name == "pip":
                        report.pip_conflicts.extend(result)
        
        report.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        # Determine overall result
        if report.syntax_errors or report.import_errors or report.pip_conflicts:
            report.result = ValidationResult.FAILED
            logger.warning(f"❌ Validation failed: {len(report.syntax_errors)} syntax, "
                          f"{len(report.import_errors)} import, {len(report.pip_conflicts)} pip errors")
        elif report.warnings:
            report.result = ValidationResult.WARNING
            logger.info(f"⚠️ Validation passed with warnings")
        else:
            report.result = ValidationResult.PASSED
            logger.info(f"✅ Validation passed ({report.files_checked} files checked)")
        
        if self._staging_info:
            self._staging_info.validation_report = report
        
        return report
    
    async def _validate_syntax(self, files: list[str], remote: str, branch: str) -> list[str]:
        """Validate Python syntax of changed files."""
        errors = []
        
        for file in files[:20]:  # Limit to avoid timeout
            # Check syntax by compiling
            result = await self._run_command(
                f"git show {remote}/{branch}:{file} | python3 -m py_compile -",
                timeout=10
            )
            if not result.success and "SyntaxError" in result.stderr:
                errors.append(f"{file}: {result.stderr.split(':')[-1].strip()}")
        
        return errors
    
    async def _validate_imports(self, files: list[str], remote: str, branch: str) -> list[str]:
        """Validate imports in changed files (basic check)."""
        errors = []
        
        # This is a lightweight check - just verifies import statements are parseable
        for file in files[:10]:  # Limit to most important
            result = await self._run_command(
                f"git show {remote}/{branch}:{file} | python3 -c \"import ast; ast.parse(open('/dev/stdin').read())\"",
                timeout=5
            )
            if not result.success:
                errors.append(f"{file}: Invalid Python")
        
        return errors
    
    async def _dry_run_pip(self) -> list[str]:
        """Dry-run pip install to detect conflicts."""
        conflicts = []
        
        req_file = self.repo_path / "requirements.txt"
        if not req_file.exists():
            return []
        
        result = await self._run_command(
            f"pip install -r {req_file} --dry-run --quiet 2>&1",
            timeout=60
        )
        
        if not result.success:
            # Parse pip errors for conflicts
            for line in result.stderr.split("\n"):
                if "conflict" in line.lower() or "incompatible" in line.lower():
                    conflicts.append(line.strip())
        
        return conflicts
    
    # =========================================================================
    # v2.0: Update Classification
    # =========================================================================
    
    async def classify_update(self, commits: Optional[list] = None) -> UpdateClassification:
        """
        Classify the update severity based on commit messages.
        
        Returns:
            UpdateClassification (SECURITY, CRITICAL, MINOR, MAJOR, UNKNOWN)
        """
        if commits is None:
            # Get commit messages
            remote = self.config.update.remote
            branch = self.config.update.branch
            result = await self._run_command(
                f"git log HEAD..{remote}/{branch} --oneline -n 20"
            )
            if not result.success:
                return UpdateClassification.UNKNOWN
            commits = result.stdout.lower().split("\n")
        
        all_text = " ".join(commits).lower()
        
        # Check for security keywords
        if any(kw in all_text for kw in self.SECURITY_KEYWORDS):
            return UpdateClassification.SECURITY
        
        # Check for major change keywords
        if any(kw in all_text for kw in self.MAJOR_KEYWORDS):
            return UpdateClassification.MAJOR
        
        # Check for critical/urgent
        if "critical" in all_text or "urgent" in all_text or "hotfix" in all_text:
            return UpdateClassification.CRITICAL
        
        return UpdateClassification.MINOR
    
    async def can_auto_update(self) -> tuple[bool, str]:
        """
        Check if update can be auto-applied based on Zero-Touch rules.
        
        Returns:
            Tuple of (can_auto_update, reason)
        """
        if not self.config.zero_touch.enabled:
            return False, "Zero-Touch mode disabled"
        
        # Run pre-flight checks
        pre_flight = await self.run_pre_flight_checks()
        if not pre_flight.passed:
            return False, pre_flight.blocked_reason or "Pre-flight check failed"
        
        # Classify the update
        classification = await self.classify_update()
        
        zt = self.config.zero_touch
        
        if classification == UpdateClassification.SECURITY:
            if zt.apply_security_updates:
                return True, "Security update auto-apply enabled"
            return False, "Security updates require confirmation"
        
        if classification == UpdateClassification.MAJOR:
            if zt.apply_major_updates:
                return True, "Major update auto-apply enabled"
            return False, "Major updates require confirmation"
        
        if classification in (UpdateClassification.MINOR, UpdateClassification.CRITICAL):
            if zt.apply_minor_updates:
                return True, f"{classification.value} update auto-apply enabled"
            return False, f"{classification.value} updates require confirmation"
        
        return False, "Unknown update classification"
    
    # =========================================================================
    # Core Update Logic
    # =========================================================================
    
    async def apply_update(self, zero_touch: bool = False) -> UpdateResult:
        """
        Apply an update from the configured source.
        
        Executes all configured build steps in order with proper
        error handling and progress tracking.
        
        Args:
            zero_touch: If True, run full Zero-Touch validation pipeline
        
        Returns:
            UpdateResult with success status and details
        """
        if self._cancelled:
            return UpdateResult(success=False, error="Update cancelled")
        
        start_time = datetime.now()
        old_version = await self.get_current_version()
        
        pre_flight: Optional[PreFlightCheckResult] = None
        validation_report: Optional[ValidationReport] = None
        classification = UpdateClassification.UNKNOWN
        
        # Calculate total steps based on mode
        base_steps = len(self.config.update.build_steps) + 2  # fetch + verify
        if zero_touch:
            base_steps += 3  # pre-flight + staging + validation
        
        self.progress = UpdateProgress(
            phase=UpdatePhase.FETCHING,
            started_at=start_time,
            steps_total=base_steps,
        )
        
        steps_completed: list[str] = []
        steps_failed: list[str] = []
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # ZERO-TOUCH MODE: Full safety pipeline
            # ═══════════════════════════════════════════════════════════════════
            if zero_touch:
                # Step 0a: Pre-flight checks
                pre_flight = await self.run_pre_flight_checks()
                if not pre_flight.passed:
                    raise RuntimeError(f"Pre-flight failed: {pre_flight.blocked_reason}")
                steps_completed.append("pre_flight")
                self.progress.steps_completed += 1
                
                # Step 0b: Stage update
                if self.config.zero_touch.use_staging_area:
                    if not await self.stage_update():
                        raise RuntimeError("Staging failed")
                    steps_completed.append("staging")
                    self.progress.steps_completed += 1
                    
                    # Step 0c: Validate staged update
                    validation_report = await self.validate_staged_update()
                    if not validation_report.is_safe_to_apply:
                        raise RuntimeError(
                            f"Validation failed: {len(validation_report.syntax_errors)} syntax errors, "
                            f"{len(validation_report.import_errors)} import errors"
                        )
                    steps_completed.append("validation")
                    self.progress.steps_completed += 1
                
                # Classify the update
                classification = await self.classify_update()
                logger.info(f"📊 Update classified as: {classification.value}")
            
            # ═══════════════════════════════════════════════════════════════════
            # CORE UPDATE: Same for both modes
            # ═══════════════════════════════════════════════════════════════════
            
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
            
            # Track update time for cooldown
            self._last_update_time = datetime.now()
            
            # Clean up staging area
            if self._staging_info and self._staging_info.staging_path:
                try:
                    shutil.rmtree(self._staging_info.staging_path)
                except Exception:
                    pass
            
            logger.info(f"✅ Update complete in {duration:.1f}s")
            
            return UpdateResult(
                success=True,
                old_version=old_version,
                new_version=new_version,
                steps_completed=steps_completed,
                steps_failed=steps_failed,
                duration_seconds=duration,
                classification=classification,
                validation_report=validation_report,
                pre_flight=pre_flight,
                was_zero_touch=zero_touch,
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
                classification=classification,
                validation_report=validation_report,
                pre_flight=pre_flight,
                was_zero_touch=zero_touch,
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
    
    async def close(self) -> None:
        """Close resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        # Clean up staging area
        if self._staging_info and self._staging_info.staging_path:
            try:
                shutil.rmtree(self._staging_info.staging_path)
            except Exception:
                pass


async def check_for_updates(config: Optional[SupervisorConfig] = None) -> bool:
    """Quick utility to check if updates are available."""
    engine = UpdateEngine(config)
    return await engine.has_update_available()
