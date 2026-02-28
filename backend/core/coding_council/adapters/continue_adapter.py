"""
v77.0: Continue.dev Framework Adapter
=====================================

Adapter for Continue.dev - IDE-integrated AI coding assistant.

Continue.dev provides:
- Real-time IDE context
- Code completion
- Inline edits
- Chat interface

For IDE-assisted development with VS Code integration.

Note: Continue.dev runs as a VS Code extension.
This adapter communicates via its local API.

Author: Ironcliw v77.0
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
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


class ContinueAdapter:
    """
    Adapter for Continue.dev IDE integration.

    Continue runs as a VS Code extension and exposes
    a local API for programmatic access.
    """

    # Continue.dev local API port
    DEFAULT_PORT = 65432

    def __init__(self, config: "CodingCouncilConfig"):
        self.config = config
        self.repo_root = config.repo_root
        self.port = int(os.getenv("CONTINUE_PORT", str(self.DEFAULT_PORT)))
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if Continue.dev server is running."""
        if self._available is not None:
            return self._available

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{self.port}/health",
                    timeout=5.0
                )
                self._available = response.status_code == 200

        except Exception:
            self._available = False

        if not self._available:
            logger.info("[ContinueAdapter] Continue.dev not running")

        return self._available

    async def execute(
        self,
        task: "EvolutionTask",
        analysis: Optional["AnalysisResult"] = None,
        plan: Optional["PlanResult"] = None
    ) -> "FrameworkResult":
        """
        Execute via Continue.dev API.

        Note: Continue.dev is primarily for IDE assistance.
        For autonomous execution, other frameworks are preferred.
        """
        from ..types import FrameworkResult, FrameworkType

        if not await self.is_available():
            return FrameworkResult(
                framework=FrameworkType.CONTINUE,
                success=False,
                error="Continue.dev not available (VS Code extension not running)"
            )

        try:
            import httpx

            # Build request
            request_data = {
                "instruction": task.description,
                "files": task.target_files,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:{self.port}/api/edit",
                    json=request_data,
                    timeout=self.config.execution_timeout
                )

                if response.status_code == 200:
                    result = response.json()
                    return FrameworkResult(
                        framework=FrameworkType.CONTINUE,
                        success=result.get("success", False),
                        changes_made=result.get("changes", []),
                        files_modified=result.get("files", task.target_files),
                        output=result.get("output", ""),
                    )
                else:
                    return FrameworkResult(
                        framework=FrameworkType.CONTINUE,
                        success=False,
                        error=f"API error: {response.status_code}"
                    )

        except Exception as e:
            return FrameworkResult(
                framework=FrameworkType.CONTINUE,
                success=False,
                error=str(e)
            )

    async def get_context(
        self,
        files: List[str]
    ) -> Dict[str, Any]:
        """
        Get IDE context for files.

        Returns context like:
        - Current cursor position
        - Open files
        - Recent edits
        - Diagnostics
        """
        if not await self.is_available():
            return {}

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:{self.port}/api/context",
                    params={"files": ",".join(files)},
                    timeout=10.0
                )

                if response.status_code == 200:
                    return response.json()

        except Exception as e:
            logger.debug(f"[ContinueAdapter] Context fetch failed: {e}")

        return {}
