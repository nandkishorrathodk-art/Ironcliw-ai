"""
v77.3: Comprehensive Diagnostics & Pre-Flight Checks
=====================================================

Advanced diagnostic system for the Unified Coding Council that:
- Validates all critical connections before startup
- Tests Trinity cross-repo connectivity
- Verifies API keys and credentials
- Checks port availability
- Provides detailed failure analysis
- Implements auto-recovery for common issues

This module can be run standalone:
    python -m backend.core.coding_council.diagnostics

Or imported for programmatic checks:
    from backend.core.coding_council.diagnostics import run_preflight_checks

Author: JARVIS v77.3
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Configuration
# =============================================================================

TRINITY_REPOS = {
    "jarvis": Path(os.getenv("JARVIS_REPO", "/Users/djrussell23/Documents/repos/JARVIS-AI-Agent")),
    "j_prime": Path(os.getenv("J_PRIME_REPO", "/Users/djrussell23/Documents/repos/jarvis-prime")),
    "reactor_core": Path(os.getenv("REACTOR_CORE_REPO", "/Users/djrussell23/Documents/repos/reactor-core")),
}

REQUIRED_PORTS = {
    "jarvis_api": int(os.getenv("JARVIS_PORT", "8010")),
    "lsp_server": int(os.getenv("LSP_SERVER_PORT", "9257")),
    "websocket": int(os.getenv("IDE_WEBSOCKET_PORT", "9258")),
}

REQUIRED_ENV_VARS = {
    "ANTHROPIC_API_KEY": {
        "required": True,
        "format": r"^sk-ant-|^sk-",
        "description": "Claude API key for Anthropic",
        "fix": "export ANTHROPIC_API_KEY=sk-ant-api03-...",
    },
}

OPTIONAL_ENV_VARS = {
    "CODING_COUNCIL_ENABLED": {
        "default": "true",
        "description": "Enable/disable Coding Council",
    },
    "IDE_BRIDGE_ENABLED": {
        "default": "true",
        "description": "Enable/disable IDE integration",
    },
}


# =============================================================================
# Enums and Data Classes
# =============================================================================

class CheckStatus(Enum):
    """Status of a diagnostic check."""
    PASS = auto()
    WARN = auto()
    FAIL = auto()
    SKIP = auto()


class CheckCategory(Enum):
    """Category of diagnostic checks."""
    ENVIRONMENT = "environment"
    CONNECTIVITY = "connectivity"
    DEPENDENCIES = "dependencies"
    TRINITY = "trinity"
    PORTS = "ports"
    PERMISSIONS = "permissions"
    RUNTIME = "runtime"


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    category: CheckCategory
    status: CheckStatus
    message: str
    details: Optional[str] = None
    fix_command: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status.name,
            "message": self.message,
            "details": self.details,
            "fix_command": self.fix_command,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    checks: List[CheckResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    system_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.SKIP)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def is_healthy(self) -> bool:
        return self.failed == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "passed": self.passed,
                "warnings": self.warnings,
                "failed": self.failed,
                "skipped": self.skipped,
                "total": self.total,
                "healthy": self.is_healthy,
                "duration_ms": round(self.duration_ms, 2),
            },
            "checks": [c.to_dict() for c in self.checks],
            "system_info": self.system_info,
        }


# =============================================================================
# Port Checker
# =============================================================================

class PortChecker:
    """Check port availability and conflicts."""

    @staticmethod
    def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
        """Check if a port is available for binding."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            sock.close()
            return True
        except OSError:
            return False

    @staticmethod
    def get_port_user(port: int) -> Optional[str]:
        """Get the process using a port (macOS/Linux)."""
        try:
            result = subprocess.run(
                ["lsof", "-i", f":{port}", "-t"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                pid = result.stdout.strip().split("\n")[0]
                # Get process name
                proc_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "comm="],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc_result.returncode == 0:
                    return f"PID {pid}: {proc_result.stdout.strip()}"
                return f"PID {pid}"
            return None
        except Exception:
            return None

    @staticmethod
    async def check_port(
        port: int,
        name: str,
        host: str = "127.0.0.1",
    ) -> CheckResult:
        """Check a port and return a diagnostic result."""
        start = time.time()

        if PortChecker.is_port_available(port, host):
            return CheckResult(
                name=f"Port {port} ({name})",
                category=CheckCategory.PORTS,
                status=CheckStatus.PASS,
                message=f"Port {port} is available",
                duration_ms=(time.time() - start) * 1000,
            )
        else:
            user = PortChecker.get_port_user(port)
            return CheckResult(
                name=f"Port {port} ({name})",
                category=CheckCategory.PORTS,
                status=CheckStatus.FAIL,
                message=f"Port {port} is in use",
                details=f"Used by: {user}" if user else "Unknown process",
                fix_command=f"lsof -ti:{port} | xargs kill -9" if user else None,
                duration_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# Environment Checker
# =============================================================================

class EnvironmentChecker:
    """Check environment variables and configuration."""

    @staticmethod
    async def check_env_var(
        name: str,
        config: Dict[str, Any],
    ) -> CheckResult:
        """Check an environment variable."""
        import re

        start = time.time()
        value = os.getenv(name, "")
        required = config.get("required", False)
        format_pattern = config.get("format")
        description = config.get("description", name)
        fix_command = config.get("fix")

        if not value:
            if required:
                return CheckResult(
                    name=f"Env: {name}",
                    category=CheckCategory.ENVIRONMENT,
                    status=CheckStatus.FAIL,
                    message=f"{name} not set (required)",
                    details=description,
                    fix_command=fix_command,
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                default = config.get("default", "")
                return CheckResult(
                    name=f"Env: {name}",
                    category=CheckCategory.ENVIRONMENT,
                    status=CheckStatus.WARN,
                    message=f"{name} not set, using default: {default}",
                    details=description,
                    duration_ms=(time.time() - start) * 1000,
                )

        # Check format if specified
        if format_pattern and not re.match(format_pattern, value):
            return CheckResult(
                name=f"Env: {name}",
                category=CheckCategory.ENVIRONMENT,
                status=CheckStatus.WARN,
                message=f"{name} format looks unusual",
                details=f"Expected pattern: {format_pattern}",
                duration_ms=(time.time() - start) * 1000,
            )

        # Mask sensitive values
        masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"

        return CheckResult(
            name=f"Env: {name}",
            category=CheckCategory.ENVIRONMENT,
            status=CheckStatus.PASS,
            message=f"{name} is set ({masked})",
            duration_ms=(time.time() - start) * 1000,
        )


# =============================================================================
# Trinity Checker
# =============================================================================

class TrinityChecker:
    """Check Trinity cross-repo connectivity."""

    @staticmethod
    async def check_repo(
        name: str,
        path: Path,
    ) -> CheckResult:
        """Check a Trinity repository."""
        start = time.time()

        if not path.exists():
            return CheckResult(
                name=f"Trinity: {name}",
                category=CheckCategory.TRINITY,
                status=CheckStatus.WARN,
                message=f"Repository not found: {path}",
                details="Some cross-repo features may not work",
                duration_ms=(time.time() - start) * 1000,
            )

        # Check if it's a git repo
        git_dir = path / ".git"
        if not git_dir.exists():
            return CheckResult(
                name=f"Trinity: {name}",
                category=CheckCategory.TRINITY,
                status=CheckStatus.WARN,
                message=f"Not a git repository: {path}",
                details="Git operations will not work",
                fix_command=f"cd {path} && git init",
                duration_ms=(time.time() - start) * 1000,
            )

        # Check for key files based on repo type
        key_files = {
            "jarvis": ["backend/main.py", "run_supervisor.py"],
            "j_prime": ["jarvis_prime/__init__.py"],
            "reactor_core": ["reactor_core/__init__.py"],
        }

        missing_files = []
        for kf in key_files.get(name, []):
            if not (path / kf).exists():
                missing_files.append(kf)

        if missing_files:
            return CheckResult(
                name=f"Trinity: {name}",
                category=CheckCategory.TRINITY,
                status=CheckStatus.WARN,
                message=f"Missing expected files in {name}",
                details=f"Missing: {', '.join(missing_files)}",
                duration_ms=(time.time() - start) * 1000,
            )

        return CheckResult(
            name=f"Trinity: {name}",
            category=CheckCategory.TRINITY,
            status=CheckStatus.PASS,
            message=f"Repository OK: {path}",
            duration_ms=(time.time() - start) * 1000,
        )

    @staticmethod
    async def check_heartbeat_dir() -> CheckResult:
        """Check Trinity heartbeat directory."""
        start = time.time()
        heartbeat_dir = Path(os.path.expanduser("~/.jarvis/trinity"))

        if not heartbeat_dir.exists():
            try:
                heartbeat_dir.mkdir(parents=True, exist_ok=True)
                return CheckResult(
                    name="Trinity: Heartbeat Directory",
                    category=CheckCategory.TRINITY,
                    status=CheckStatus.PASS,
                    message=f"Created heartbeat directory: {heartbeat_dir}",
                    duration_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                return CheckResult(
                    name="Trinity: Heartbeat Directory",
                    category=CheckCategory.TRINITY,
                    status=CheckStatus.FAIL,
                    message=f"Cannot create heartbeat directory",
                    details=str(e),
                    fix_command=f"mkdir -p {heartbeat_dir}",
                    duration_ms=(time.time() - start) * 1000,
                )

        # Check write permissions
        test_file = heartbeat_dir / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            return CheckResult(
                name="Trinity: Heartbeat Directory",
                category=CheckCategory.TRINITY,
                status=CheckStatus.PASS,
                message=f"Heartbeat directory OK: {heartbeat_dir}",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return CheckResult(
                name="Trinity: Heartbeat Directory",
                category=CheckCategory.TRINITY,
                status=CheckStatus.FAIL,
                message=f"Heartbeat directory not writable",
                details=str(e),
                fix_command=f"chmod 755 {heartbeat_dir}",
                duration_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# Module Import Checker
# =============================================================================

class ModuleChecker:
    """Check Python module availability."""

    @staticmethod
    async def check_module(
        module_path: str,
        description: str,
    ) -> CheckResult:
        """Check if a Python module can be imported."""
        start = time.time()

        try:
            # Use importlib for dynamic import
            import importlib

            # Try direct import first (for module paths like "backend.core.coding_council.startup")
            module = importlib.import_module(module_path)

            return CheckResult(
                name=f"Module: {description}",
                category=CheckCategory.DEPENDENCIES,
                status=CheckStatus.PASS,
                message=f"Successfully imported {module_path}",
                duration_ms=(time.time() - start) * 1000,
            )
        except ImportError as e:
            return CheckResult(
                name=f"Module: {description}",
                category=CheckCategory.DEPENDENCIES,
                status=CheckStatus.FAIL,
                message=f"Import failed: {module_path}",
                details=str(e),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return CheckResult(
                name=f"Module: {description}",
                category=CheckCategory.DEPENDENCIES,
                status=CheckStatus.WARN,
                message=f"Import warning: {module_path}",
                details=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# Runtime Checker
# =============================================================================

class RuntimeChecker:
    """Check runtime components."""

    @staticmethod
    async def check_anthropic_api() -> CheckResult:
        """Test Anthropic API connectivity."""
        start = time.time()
        api_key = os.getenv("ANTHROPIC_API_KEY", "")

        if not api_key:
            return CheckResult(
                name="Anthropic API",
                category=CheckCategory.CONNECTIVITY,
                status=CheckStatus.SKIP,
                message="API key not set, skipping connectivity test",
                duration_ms=(time.time() - start) * 1000,
            )

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    timeout=10.0,
                )

                if response.status_code == 200:
                    return CheckResult(
                        name="Anthropic API",
                        category=CheckCategory.CONNECTIVITY,
                        status=CheckStatus.PASS,
                        message="API connectivity OK",
                        duration_ms=(time.time() - start) * 1000,
                    )
                elif response.status_code == 401:
                    return CheckResult(
                        name="Anthropic API",
                        category=CheckCategory.CONNECTIVITY,
                        status=CheckStatus.FAIL,
                        message="Invalid API key",
                        details="The provided ANTHROPIC_API_KEY is not valid",
                        fix_command="export ANTHROPIC_API_KEY=<valid_key>",
                        duration_ms=(time.time() - start) * 1000,
                    )
                else:
                    return CheckResult(
                        name="Anthropic API",
                        category=CheckCategory.CONNECTIVITY,
                        status=CheckStatus.WARN,
                        message=f"Unexpected response: {response.status_code}",
                        duration_ms=(time.time() - start) * 1000,
                    )
        except ImportError:
            return CheckResult(
                name="Anthropic API",
                category=CheckCategory.CONNECTIVITY,
                status=CheckStatus.SKIP,
                message="httpx not available, skipping API test",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return CheckResult(
                name="Anthropic API",
                category=CheckCategory.CONNECTIVITY,
                status=CheckStatus.WARN,
                message="Cannot reach Anthropic API",
                details=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# Pre-Flight Checker (Main Class)
# =============================================================================

class PreFlightChecker:
    """
    Comprehensive pre-flight check system.

    Runs all diagnostic checks in parallel and generates a report.
    """

    def __init__(self):
        self.report = DiagnosticReport()

    async def run_all_checks(self) -> DiagnosticReport:
        """Run all diagnostic checks."""
        self.report = DiagnosticReport()
        self.report.start_time = time.time()

        # Collect system info
        self.report.system_info = self._collect_system_info()

        # Run checks in parallel where possible
        checks = []

        # Environment checks
        for name, config in REQUIRED_ENV_VARS.items():
            checks.append(EnvironmentChecker.check_env_var(name, config))
        for name, config in OPTIONAL_ENV_VARS.items():
            checks.append(EnvironmentChecker.check_env_var(name, config))

        # Port checks
        for name, port in REQUIRED_PORTS.items():
            checks.append(PortChecker.check_port(port, name))

        # Trinity checks
        for name, path in TRINITY_REPOS.items():
            checks.append(TrinityChecker.check_repo(name, path))
        checks.append(TrinityChecker.check_heartbeat_dir())

        # Module checks
        module_checks = [
            ("backend.core.coding_council.startup", "Startup Module"),
            ("backend.core.coding_council.orchestrator", "Orchestrator"),
            ("backend.core.coding_council.ide", "IDE Module"),
            ("backend.core.coding_council.adapters.anthropic_engine", "Anthropic Engine"),
            ("backend.core.coding_council.integration", "Integration Module"),
        ]
        for module_path, description in module_checks:
            checks.append(ModuleChecker.check_module(module_path, description))

        # Runtime checks
        checks.append(RuntimeChecker.check_anthropic_api())

        # Execute all checks in parallel
        results = await asyncio.gather(*checks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self.report.checks.append(CheckResult(
                    name="Unknown Check",
                    category=CheckCategory.RUNTIME,
                    status=CheckStatus.FAIL,
                    message=f"Check failed with exception",
                    details=str(result),
                ))
            else:
                self.report.checks.append(result)

        self.report.end_time = time.time()
        return self.report

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        import platform

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "pid": os.getpid(),
            "timestamp": time.time(),
        }

    def print_report(self, report: Optional[DiagnosticReport] = None) -> None:
        """Print a formatted report to console."""
        r = report or self.report

        print("\n" + "=" * 70)
        print("JARVIS v77.3 - Unified Coding Council Diagnostics")
        print("=" * 70)

        # Summary
        status_icon = "✅" if r.is_healthy else "❌"
        print(f"\n{status_icon} Summary: {r.passed}/{r.total} checks passed")
        if r.warnings > 0:
            print(f"   ⚠️  {r.warnings} warnings")
        if r.failed > 0:
            print(f"   ❌ {r.failed} failures")
        print(f"   ⏱️  Duration: {r.duration_ms:.1f}ms")

        # Group by category
        by_category: Dict[CheckCategory, List[CheckResult]] = {}
        for check in r.checks:
            if check.category not in by_category:
                by_category[check.category] = []
            by_category[check.category].append(check)

        # Print each category
        for category in CheckCategory:
            if category not in by_category:
                continue

            print(f"\n── {category.value.upper()} ──")

            for check in by_category[category]:
                icon = {
                    CheckStatus.PASS: "✓",
                    CheckStatus.WARN: "⚠",
                    CheckStatus.FAIL: "✗",
                    CheckStatus.SKIP: "○",
                }[check.status]

                color = {
                    CheckStatus.PASS: "\033[92m",  # Green
                    CheckStatus.WARN: "\033[93m",  # Yellow
                    CheckStatus.FAIL: "\033[91m",  # Red
                    CheckStatus.SKIP: "\033[90m",  # Gray
                }[check.status]
                reset = "\033[0m"

                print(f"  {color}{icon} {check.name}{reset}")
                print(f"    {check.message}")

                if check.details:
                    print(f"    Details: {check.details}")

                if check.fix_command and check.status in (CheckStatus.FAIL, CheckStatus.WARN):
                    print(f"    Fix: {check.fix_command}")

        # Recommendations
        failed_checks = [c for c in r.checks if c.status == CheckStatus.FAIL]
        if failed_checks:
            print("\n── RECOMMENDED ACTIONS ──")
            for i, check in enumerate(failed_checks, 1):
                print(f"  {i}. Fix: {check.name}")
                if check.fix_command:
                    print(f"     Run: {check.fix_command}")

        print("\n" + "=" * 70)


# =============================================================================
# Auto-Recovery System
# =============================================================================

class AutoRecovery:
    """
    Automatic recovery for common issues.

    Can attempt to fix issues found during diagnostics.
    """

    @staticmethod
    async def attempt_recovery(report: DiagnosticReport) -> List[str]:
        """
        Attempt to recover from failures.

        Returns list of recovery actions taken.
        """
        actions = []

        for check in report.checks:
            if check.status != CheckStatus.FAIL:
                continue

            # Port conflict recovery
            if check.category == CheckCategory.PORTS and check.fix_command:
                if await AutoRecovery._kill_port_user(check):
                    actions.append(f"Killed process using port (from: {check.name})")

            # Directory creation recovery
            elif check.category == CheckCategory.TRINITY:
                if "directory" in check.message.lower() and check.fix_command:
                    if await AutoRecovery._create_directory(check):
                        actions.append(f"Created directory (from: {check.name})")

        return actions

    @staticmethod
    async def _kill_port_user(check: CheckResult) -> bool:
        """Kill process using a port."""
        if not check.fix_command:
            return False

        try:
            result = subprocess.run(
                check.fix_command,
                shell=True,
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    async def _create_directory(check: CheckResult) -> bool:
        """Create a missing directory."""
        if not check.fix_command:
            return False

        try:
            result = subprocess.run(
                check.fix_command,
                shell=True,
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False


# =============================================================================
# Public API
# =============================================================================

async def run_preflight_checks(
    print_report: bool = True,
    auto_recover: bool = False,
) -> DiagnosticReport:
    """
    Run pre-flight checks and optionally print report.

    Args:
        print_report: Whether to print the report to console
        auto_recover: Whether to attempt auto-recovery

    Returns:
        DiagnosticReport with all check results
    """
    checker = PreFlightChecker()
    report = await checker.run_all_checks()

    if auto_recover and not report.is_healthy:
        actions = await AutoRecovery.attempt_recovery(report)
        if actions:
            # Re-run checks after recovery
            report = await checker.run_all_checks()

    if print_report:
        checker.print_report(report)

    return report


def run_preflight_checks_sync(
    print_report: bool = True,
    auto_recover: bool = False,
) -> DiagnosticReport:
    """Synchronous wrapper for pre-flight checks."""
    return asyncio.run(run_preflight_checks(print_report, auto_recover))


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for diagnostics."""
    import argparse

    parser = argparse.ArgumentParser(
        description="JARVIS Coding Council Diagnostics"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--auto-recover",
        action="store_true",
        help="Attempt automatic recovery",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show failures",
    )

    args = parser.parse_args()

    # Run checks
    report = run_preflight_checks_sync(
        print_report=not args.json and not args.quiet,
        auto_recover=args.auto_recover,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif args.quiet:
        for check in report.checks:
            if check.status == CheckStatus.FAIL:
                print(f"❌ {check.name}: {check.message}")

    # Exit with appropriate code
    sys.exit(0 if report.is_healthy else 1)


if __name__ == "__main__":
    main()
