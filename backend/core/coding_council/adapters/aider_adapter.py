"""
v77.0: Aider Framework Adapter
==============================

Adapter for Aider - the fast, git-integrated AI coding assistant.

Aider is ideal for:
- Single file edits
- Quick bug fixes
- Refactoring
- Adding features to existing code
- Git-aware changes

Installation:
    pip install aider-chat

Usage:
    adapter = AiderAdapter(config)
    result = await adapter.execute(task, analysis, plan)

Safety Features:
- Subprocess isolation
- Timeout handling
- Output parsing
- Error recovery

Author: JARVIS v77.0
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import (
        AnalysisResult,
        CodingCouncilConfig,
        EvolutionTask,
        FrameworkResult,
        FrameworkType,
        PlanResult,
    )

logger = logging.getLogger(__name__)


class AiderAdapter:
    """
    Adapter for Aider AI coding assistant.

    Aider provides:
    - Direct file editing
    - Git integration (auto-commits)
    - Multiple LLM backends
    - Code context awareness

    Addresses:
    - Gap #10: Framework Timeout
    - Gap #15: API Key Management
    - Gap #36: Subprocess Zombie
    """

    def __init__(self, config: "CodingCouncilConfig"):
        self.config = config
        self.repo_root = config.repo_root
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._available: Optional[bool] = None

        # API key sources (environment-driven)
        self._api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
        }

        # Aider configuration
        self._model = os.getenv("AIDER_MODEL", "claude-3-5-sonnet-20241022")
        self._auto_commits = os.getenv("AIDER_AUTO_COMMITS", "false").lower() == "true"
        self._show_diffs = os.getenv("AIDER_SHOW_DIFFS", "true").lower() == "true"

    async def is_available(self) -> bool:
        """Check if Aider is installed and configured."""
        if self._available is not None:
            return self._available

        try:
            # Check if aider is installed
            result = await asyncio.create_subprocess_exec(
                "aider", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()

            if result.returncode == 0:
                version = stdout.decode().strip()
                logger.info(f"[AiderAdapter] Found Aider: {version}")
                self._available = True
            else:
                self._available = False

        except FileNotFoundError:
            logger.warning("[AiderAdapter] Aider not installed")
            self._available = False
        except Exception as e:
            logger.warning(f"[AiderAdapter] Availability check failed: {e}")
            self._available = False

        return self._available

    async def execute(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"] = None,
        plan: Optional["PlanResult"] = None
    ) -> "FrameworkResult":
        """
        Execute code changes using Aider.

        Args:
            task: The evolution task to execute
            analysis: Optional codebase analysis from RepoMaster
            plan: Optional plan from MetaGPT

        Returns:
            FrameworkResult with execution details
        """
        from ..types import FrameworkResult, FrameworkType

        # Check availability
        if not await self.is_available():
            return FrameworkResult(
                framework=FrameworkType.AIDER,
                success=False,
                error="Aider not available"
            )

        # Build the message for Aider
        message = self._build_message(task, analysis, plan)

        # Build command
        cmd = self._build_command(task, message)

        logger.info(f"[AiderAdapter] Executing task: {task.task_id}")
        logger.debug(f"[AiderAdapter] Command: {' '.join(cmd)}")

        try:
            # Run Aider with timeout
            result = await self._run_aider(cmd, task)

            if result["success"]:
                # Parse changes from output
                changes = self._parse_changes(result["output"])
                modified_files = self._detect_modified_files(task.target_files)

                return FrameworkResult(
                    framework=FrameworkType.AIDER,
                    success=True,
                    changes_made=changes,
                    files_modified=modified_files,
                    output=result["output"],
                )
            else:
                return FrameworkResult(
                    framework=FrameworkType.AIDER,
                    success=False,
                    error=result.get("error", "Unknown error"),
                    output=result.get("output", ""),
                )

        except asyncio.TimeoutError:
            return FrameworkResult(
                framework=FrameworkType.AIDER,
                success=False,
                error=f"Timeout after {self.config.execution_timeout}s"
            )
        except Exception as e:
            logger.error(f"[AiderAdapter] Execution failed: {e}")
            return FrameworkResult(
                framework=FrameworkType.AIDER,
                success=False,
                error=str(e)
            )

    def _build_message(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"],
        plan: Optional["PlanResult"]
    ) -> str:
        """Build the message to send to Aider."""
        parts = []

        # Task description
        parts.append(f"Task: {task.description}")

        # Add analysis context if available
        if analysis:
            parts.append("\nCodebase Analysis:")
            for insight in analysis.insights[:5]:  # Limit to 5 insights
                parts.append(f"- {insight}")

            if analysis.risk_score > 0.7:
                parts.append(f"\n⚠️ High risk modification (score: {analysis.risk_score:.2f})")
                parts.append("Be careful with changes to these files.")

        # Add plan context if available
        if plan and plan.steps:
            parts.append("\nExecution Plan:")
            for i, step in enumerate(plan.steps[:5], 1):  # Limit to 5 steps
                step_desc = step.get("description", str(step))
                parts.append(f"{i}. {step_desc}")

        # Add safety reminders
        parts.append("\n\nIMPORTANT:")
        parts.append("- Make minimal, focused changes")
        parts.append("- Don't break existing functionality")
        parts.append("- Preserve existing code style")
        parts.append("- Add comments for complex logic")

        return "\n".join(parts)

    def _build_command(
        self,
        task: "EvolutionTask",
        message: str
    ) -> List[str]:
        """Build the Aider command."""
        cmd = ["aider"]

        # Auto-yes for non-interactive mode
        cmd.append("--yes")

        # Model selection
        if self._model:
            cmd.extend(["--model", self._model])

        # Auto-commits setting
        if not self._auto_commits:
            cmd.append("--no-auto-commits")

        # Show diffs
        if self._show_diffs:
            cmd.append("--show-diffs")

        # Message
        cmd.extend(["--message", message])

        # Target files
        if task.target_files:
            for filepath in task.target_files:
                full_path = self.repo_root / filepath
                if full_path.exists():
                    cmd.append(str(full_path))
                else:
                    # For new files, just pass the path
                    cmd.append(str(full_path))

        return cmd

    async def _run_aider(
        self,
        cmd: List[str],
        task: "EvolutionTask"
    ) -> Dict[str, Any]:
        """Run Aider subprocess with proper handling."""
        env = os.environ.copy()

        # Set API keys if available
        if self._api_keys.get("anthropic"):
            env["ANTHROPIC_API_KEY"] = self._api_keys["anthropic"]
        if self._api_keys.get("openai"):
            env["OPENAI_API_KEY"] = self._api_keys["openai"]

        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.execution_timeout
                )
            except asyncio.TimeoutError:
                # Kill the process on timeout
                process.kill()
                try:
                    await process.wait()
                except Exception:
                    pass
                raise

            output = stdout.decode() if stdout else ""
            error_output = stderr.decode() if stderr else ""

            if process.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "stderr": error_output,
                }
            else:
                return {
                    "success": False,
                    "error": error_output or "Aider exited with non-zero status",
                    "output": output,
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
            }

    def _parse_changes(self, output: str) -> List[str]:
        """Parse changes from Aider output."""
        changes = []

        # Look for diff blocks
        diff_pattern = re.compile(r"^[\+\-].*", re.MULTILINE)
        diffs = diff_pattern.findall(output)

        if diffs:
            # Summarize changes
            additions = sum(1 for d in diffs if d.startswith("+"))
            deletions = sum(1 for d in diffs if d.startswith("-"))
            changes.append(f"Added {additions} lines, removed {deletions} lines")

        # Look for "Applied edit to" messages
        edit_pattern = re.compile(r"Applied edit to (.+)", re.IGNORECASE)
        edits = edit_pattern.findall(output)
        for filepath in edits:
            changes.append(f"Modified: {filepath}")

        # Look for commit messages if auto-commits enabled
        commit_pattern = re.compile(r"Commit [a-f0-9]+: (.+)", re.IGNORECASE)
        commits = commit_pattern.findall(output)
        for commit_msg in commits:
            changes.append(f"Committed: {commit_msg}")

        return changes

    def _detect_modified_files(self, target_files: List[str]) -> List[str]:
        """Detect which files were actually modified."""
        modified = []

        try:
            # Use git status to find modified files
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line and len(line) > 3:
                        # Git status format: "XY path"
                        filepath = line[3:].strip()
                        if filepath in target_files or any(
                            filepath.endswith(tf) for tf in target_files
                        ):
                            modified.append(filepath)

        except Exception as e:
            logger.warning(f"[AiderAdapter] Could not detect modified files: {e}")
            # Fall back to target files
            modified = target_files.copy()

        return modified


class AiderCLI:
    """
    CLI wrapper for direct Aider interaction.

    For cases where we need more control than the subprocess approach.
    """

    @staticmethod
    async def run_interactive(
        repo_root: Path,
        files: List[str],
        model: str = "claude-3-5-sonnet-20241022"
    ) -> None:
        """
        Run Aider in interactive mode.

        This is for manual intervention when automated execution fails.
        """
        cmd = [
            "aider",
            "--model", model,
            *[str(repo_root / f) for f in files]
        ]

        # Run in foreground (interactive)
        subprocess.run(cmd, cwd=str(repo_root))
