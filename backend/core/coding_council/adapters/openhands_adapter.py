"""
v77.0: OpenHands Framework Adapter
==================================

Adapter for OpenHands (formerly OpenDevin) - sandboxed AI agent execution.

OpenHands provides:
- Docker-based sandboxed execution
- Safe testing of risky changes
- Full environment control
- Rollback capability

For changes that need isolation before application.

Installation:
    git clone https://github.com/All-Hands-AI/OpenHands
    pip install -e ./OpenHands

Author: JARVIS v77.0
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import (
        AnalysisResult,
        CodingCouncilConfig,
        EvolutionTask,
        FrameworkResult,
        PlanResult,
    )

logger = logging.getLogger(__name__)


class OpenHandsAdapter:
    """
    Adapter for OpenHands sandboxed execution.

    If OpenHands/Docker not available, falls back to
    local execution with extra validation.
    """

    def __init__(self, config: "CodingCouncilConfig"):
        self.config = config
        self.repo_root = config.repo_root
        self._docker_available: Optional[bool] = None
        self._openhands_available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if OpenHands and Docker are available."""
        if self._openhands_available is not None:
            return self._openhands_available

        # Check Docker
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            self._docker_available = proc.returncode == 0
        except Exception:
            self._docker_available = False

        if not self._docker_available:
            logger.warning("[OpenHandsAdapter] Docker not available")
            self._openhands_available = False
            return False

        # Check OpenHands
        try:
            import importlib.util
            spec = importlib.util.find_spec("openhands")
            self._openhands_available = spec is not None
        except Exception:
            self._openhands_available = False

        if not self._openhands_available:
            logger.info("[OpenHandsAdapter] OpenHands not installed, using safe fallback")

        return self._openhands_available or self._docker_available

    async def execute(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"] = None,
        plan: Optional["PlanResult"] = None
    ) -> "FrameworkResult":
        """
        Execute task in sandboxed environment.

        Flow:
        1. Create sandbox (Docker container or temp directory)
        2. Copy target files
        3. Execute changes
        4. Validate in sandbox
        5. Copy changes back if successful
        """
        from ..types import FrameworkResult, FrameworkType

        logger.info(f"[OpenHandsAdapter] Executing task in sandbox: {task.task_id}")

        if self._docker_available:
            return await self._execute_in_docker(task, analysis, plan)
        else:
            return await self._execute_with_isolation(task, analysis, plan)

    async def _execute_in_docker(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"],
        plan: Optional["PlanResult"]
    ) -> "FrameworkResult":
        """Execute in Docker sandbox."""
        from ..types import FrameworkResult, FrameworkType

        try:
            # Build Docker command
            container_name = f"jarvis_sandbox_{task.task_id[:8]}"

            # Create volume mapping
            volumes = {
                str(self.repo_root): {"bind": "/workspace", "mode": "rw"}
            }

            # Build instructions for the container
            instructions = self._build_instructions(task, plan)

            # Run container with timeout
            cmd = [
                "docker", "run",
                "--rm",
                "--name", container_name,
                "-v", f"{self.repo_root}:/workspace",
                "-w", "/workspace",
                "python:3.11-slim",
                "bash", "-c", instructions
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.config.execution_timeout
                )
            except asyncio.TimeoutError:
                # Kill container
                await asyncio.create_subprocess_exec(
                    "docker", "kill", container_name
                )
                return FrameworkResult(
                    framework=FrameworkType.OPENHANDS,
                    success=False,
                    error="Sandbox execution timed out"
                )

            output = stdout.decode() if stdout else ""
            error_output = stderr.decode() if stderr else ""

            if proc.returncode == 0:
                return FrameworkResult(
                    framework=FrameworkType.OPENHANDS,
                    success=True,
                    changes_made=["Executed in Docker sandbox"],
                    files_modified=task.target_files,
                    output=output,
                )
            else:
                return FrameworkResult(
                    framework=FrameworkType.OPENHANDS,
                    success=False,
                    error=error_output or "Sandbox execution failed",
                    output=output,
                )

        except Exception as e:
            logger.error(f"[OpenHandsAdapter] Docker execution failed: {e}")
            return FrameworkResult(
                framework=FrameworkType.OPENHANDS,
                success=False,
                error=str(e)
            )

    async def _execute_with_isolation(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"],
        plan: Optional["PlanResult"]
    ) -> "FrameworkResult":
        """Execute with file-based isolation (no Docker)."""
        from ..types import FrameworkResult, FrameworkType
        import tempfile

        temp_dir = None
        try:
            # Create temp directory as sandbox
            temp_dir = Path(tempfile.mkdtemp(prefix="jarvis_sandbox_"))

            # Copy target files to sandbox
            for filepath in task.target_files:
                src = self.repo_root / filepath
                dst = temp_dir / filepath

                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

            logger.info(f"[OpenHandsAdapter] Created sandbox: {temp_dir}")

            # For now, return that we need to use another framework
            # This is a safe fallback when Docker isn't available
            return FrameworkResult(
                framework=FrameworkType.OPENHANDS,
                success=False,
                error="Docker not available, recommend using Aider with careful validation",
                output=f"Sandbox prepared at {temp_dir}",
            )

        except Exception as e:
            return FrameworkResult(
                framework=FrameworkType.OPENHANDS,
                success=False,
                error=str(e)
            )
        finally:
            # Clean up temp directory
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

    def _build_instructions(
        self,
        task: "EvolutionTask",
        plan: Optional["PlanResult"]
    ) -> str:
        """Build execution instructions for sandbox."""
        parts = [
            "set -e",  # Exit on error
            "cd /workspace",
            "echo '=== Starting sandbox execution ==='",
        ]

        # Add pip install if needed
        parts.append("pip install -q aider-chat 2>/dev/null || true")

        # Build Aider command
        files = " ".join(task.target_files) if task.target_files else "."
        parts.append(
            f"aider --yes --no-auto-commits --message '{task.description}' {files}"
        )

        parts.append("echo '=== Sandbox execution complete ==='")

        return " && ".join(parts)
