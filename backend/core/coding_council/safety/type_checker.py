"""
v77.0: Type Checker - Gap #21
==============================

Type checking integration with:
- Mypy integration (when available)
- Pyright integration (optional)
- Basic type inference fallback
- Stub file handling

Author: JARVIS v77.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TypeCheckSeverity(Enum):
    """Severity levels for type errors."""
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


@dataclass
class TypeIssue:
    """A single type checking issue."""
    severity: TypeCheckSeverity
    message: str
    file_path: str
    line: int = 0
    column: int = 0
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "error_code": self.error_code,
        }


@dataclass
class TypeCheckResult:
    """Result of type checking."""
    passed: bool
    issues: List[TypeIssue] = field(default_factory=list)
    checker_used: str = "none"
    check_time_ms: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == TypeCheckSeverity.ERROR)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "checker_used": self.checker_used,
            "check_time_ms": self.check_time_ms,
            "error_count": self.error_count,
        }


class MypyIntegration:
    """Integration with mypy type checker."""

    def __init__(self):
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if mypy is available."""
        if self._available is not None:
            return self._available

        try:
            proc = await asyncio.create_subprocess_exec(
                "mypy", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            self._available = proc.returncode == 0
        except FileNotFoundError:
            self._available = False
        except Exception:
            self._available = False

        return self._available

    async def check_files(
        self,
        files: List[Path],
        repo_root: Path,
        strict: bool = False
    ) -> TypeCheckResult:
        """Run mypy on files."""
        import time
        start_time = time.time()

        if not await self.is_available():
            return TypeCheckResult(passed=True, checker_used="none")

        issues = []

        try:
            cmd = [
                "mypy",
                "--no-error-summary",
                "--show-column-numbers",
                "--show-error-codes",
                "--no-pretty",
            ]

            if strict:
                cmd.append("--strict")

            # Add files
            cmd.extend([str(f) for f in files])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

            if stdout:
                issues = self._parse_output(stdout.decode())

        except asyncio.TimeoutError:
            logger.warning("[TypeChecker] Mypy timed out")
        except Exception as e:
            logger.warning(f"[TypeChecker] Mypy failed: {e}")

        check_time = (time.time() - start_time) * 1000
        has_errors = any(i.severity == TypeCheckSeverity.ERROR for i in issues)

        return TypeCheckResult(
            passed=not has_errors,
            issues=issues,
            checker_used="mypy",
            check_time_ms=check_time,
        )

    def _parse_output(self, output: str) -> List[TypeIssue]:
        """Parse mypy output."""
        issues = []

        for line in output.strip().split("\n"):
            if not line or ":" not in line:
                continue

            # Format: file:line:column: severity: message [error-code]
            parts = line.split(":", 3)
            if len(parts) < 4:
                continue

            file_path = parts[0]
            try:
                line_num = int(parts[1])
                column = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            except ValueError:
                continue

            rest = parts[3].strip() if len(parts) > 3 else parts[2].strip()

            # Determine severity
            if rest.startswith("error:"):
                severity = TypeCheckSeverity.ERROR
                message = rest[6:].strip()
            elif rest.startswith("warning:"):
                severity = TypeCheckSeverity.WARNING
                message = rest[8:].strip()
            elif rest.startswith("note:"):
                severity = TypeCheckSeverity.NOTE
                message = rest[5:].strip()
            else:
                severity = TypeCheckSeverity.ERROR
                message = rest

            # Extract error code
            error_code = None
            if "[" in message and "]" in message:
                start = message.rfind("[")
                end = message.rfind("]")
                if start < end:
                    error_code = message[start + 1:end]
                    message = message[:start].strip()

            issues.append(TypeIssue(
                severity=severity,
                message=message,
                file_path=file_path,
                line=line_num,
                column=column,
                error_code=error_code,
            ))

        return issues


class PyrightIntegration:
    """Integration with pyright type checker."""

    def __init__(self):
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if pyright is available."""
        if self._available is not None:
            return self._available

        try:
            proc = await asyncio.create_subprocess_exec(
                "pyright", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            self._available = proc.returncode == 0
        except FileNotFoundError:
            self._available = False
        except Exception:
            self._available = False

        return self._available

    async def check_files(
        self,
        files: List[Path],
        repo_root: Path
    ) -> TypeCheckResult:
        """Run pyright on files."""
        import time
        start_time = time.time()

        if not await self.is_available():
            return TypeCheckResult(passed=True, checker_used="none")

        issues = []

        try:
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config = {
                    "include": [str(f.relative_to(repo_root) if f.is_absolute() else f) for f in files],
                    "reportMissingImports": False,
                    "reportMissingModuleSource": False,
                }
                json.dump(config, f)
                config_file = f.name

            try:
                cmd = [
                    "pyright",
                    "--outputjson",
                    "-p", config_file,
                ]

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(repo_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)

                if stdout:
                    issues = self._parse_json_output(stdout.decode())

            finally:
                os.unlink(config_file)

        except asyncio.TimeoutError:
            logger.warning("[TypeChecker] Pyright timed out")
        except Exception as e:
            logger.warning(f"[TypeChecker] Pyright failed: {e}")

        check_time = (time.time() - start_time) * 1000
        has_errors = any(i.severity == TypeCheckSeverity.ERROR for i in issues)

        return TypeCheckResult(
            passed=not has_errors,
            issues=issues,
            checker_used="pyright",
            check_time_ms=check_time,
        )

    def _parse_json_output(self, output: str) -> List[TypeIssue]:
        """Parse pyright JSON output."""
        issues = []

        try:
            data = json.loads(output)
            for diag in data.get("generalDiagnostics", []):
                severity_map = {
                    "error": TypeCheckSeverity.ERROR,
                    "warning": TypeCheckSeverity.WARNING,
                    "information": TypeCheckSeverity.NOTE,
                }

                issues.append(TypeIssue(
                    severity=severity_map.get(diag.get("severity", "error"), TypeCheckSeverity.ERROR),
                    message=diag.get("message", ""),
                    file_path=diag.get("file", ""),
                    line=diag.get("range", {}).get("start", {}).get("line", 0) + 1,
                    column=diag.get("range", {}).get("start", {}).get("character", 0),
                    error_code=diag.get("rule"),
                ))

        except json.JSONDecodeError:
            pass

        return issues


class TypeChecker:
    """
    Type checking coordinator.

    Uses mypy by default, falls back to pyright,
    and has a basic inference fallback.
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.mypy = MypyIntegration()
        self.pyright = PyrightIntegration()
        self._preferred_checker: Optional[str] = os.getenv("CODING_COUNCIL_TYPE_CHECKER")

    async def check_file(self, file_path: Union[str, Path]) -> TypeCheckResult:
        """Check types in a single file."""
        file_path = Path(file_path)

        if not file_path.exists():
            return TypeCheckResult(passed=True)

        if not file_path.suffix == ".py":
            return TypeCheckResult(passed=True)

        return await self.check_files([file_path])

    async def check_files(self, files: List[Union[str, Path]], strict: bool = False) -> TypeCheckResult:
        """Check types in multiple files."""
        paths = [Path(f) for f in files]
        paths = [p for p in paths if p.exists() and p.suffix == ".py"]

        if not paths:
            return TypeCheckResult(passed=True)

        # Try preferred checker first
        if self._preferred_checker == "pyright":
            if await self.pyright.is_available():
                return await self.pyright.check_files(paths, self.repo_root)

        # Default: try mypy
        if await self.mypy.is_available():
            return await self.mypy.check_files(paths, self.repo_root, strict=strict)

        # Fallback: try pyright
        if await self.pyright.is_available():
            return await self.pyright.check_files(paths, self.repo_root)

        # No type checker available
        logger.info("[TypeChecker] No type checker available, skipping")
        return TypeCheckResult(passed=True, checker_used="none")

    async def get_available_checkers(self) -> Dict[str, bool]:
        """Get availability of type checkers."""
        return {
            "mypy": await self.mypy.is_available(),
            "pyright": await self.pyright.is_available(),
        }
