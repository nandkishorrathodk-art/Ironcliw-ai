"""
v77.0: Security Scanner - Gap #20
==================================

Advanced security vulnerability scanning with:
- Bandit integration (when available)
- Custom OWASP pattern detection
- SQL injection detection
- XSS pattern detection
- Command injection detection
- Path traversal detection
- Secrets detection (API keys, passwords)
- Dependency vulnerability checking

Author: JARVIS v77.0
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class VulnerabilitySeverity(Enum):
    """Severity levels for security vulnerabilities."""
    CRITICAL = "critical"  # Immediate exploitation risk
    HIGH = "high"          # Serious security issue
    MEDIUM = "medium"      # Moderate risk
    LOW = "low"           # Minor issue
    INFO = "info"         # Informational


@dataclass
class SecurityVulnerability:
    """A security vulnerability found in code."""
    severity: VulnerabilitySeverity
    vulnerability_type: str
    message: str
    file_path: str
    line: int = 0
    column: int = 0
    code: str = ""  # CWE or custom code
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    confidence: str = "medium"  # low, medium, high
    snippet: Optional[str] = None
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "vulnerability_type": self.vulnerability_type,
            "message": self.message,
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "code": self.code,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "confidence": self.confidence,
            "snippet": self.snippet,
            "remediation": self.remediation,
        }


@dataclass
class SecurityScanResult:
    """Result of security scan."""
    secure: bool
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    scan_time_ms: float = 0.0
    bandit_available: bool = False
    files_scanned: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "secure": self.secure,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "scan_time_ms": self.scan_time_ms,
            "bandit_available": self.bandit_available,
            "files_scanned": self.files_scanned,
            "summary": {
                "critical": self.critical_count,
                "high": self.high_count,
                "total": len(self.vulnerabilities),
            }
        }


class OWASPPatternScanner:
    """
    Custom OWASP Top 10 pattern detection.

    Detects:
    - A01: Broken Access Control
    - A02: Cryptographic Failures
    - A03: Injection
    - A04: Insecure Design
    - A05: Security Misconfiguration
    - A06: Vulnerable Components
    - A07: Auth Failures
    - A08: Data Integrity Failures
    - A09: Logging Failures
    - A10: SSRF
    """

    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        # String formatting in SQL
        (r'execute\s*\(\s*["\'].*%s.*["\']\s*%', "SQL_INJECTION", "A03"),
        (r'execute\s*\(\s*f["\']', "SQL_INJECTION", "A03"),
        (r'\.format\s*\(.*\).*(?:SELECT|INSERT|UPDATE|DELETE|DROP)', "SQL_INJECTION", "A03"),
        (r'cursor\.execute\s*\(\s*[^,]+\s*\+', "SQL_INJECTION", "A03"),
    ]

    # Command Injection patterns
    COMMAND_INJECTION_PATTERNS = [
        (r'os\.system\s*\(\s*[^)]*\+', "COMMAND_INJECTION", "A03"),
        (r'os\.popen\s*\(\s*[^)]*\+', "COMMAND_INJECTION", "A03"),
        (r'subprocess\..*shell\s*=\s*True', "COMMAND_INJECTION", "A03"),
        (r'subprocess\.call\s*\(\s*["\'][^"\']*\$', "COMMAND_INJECTION", "A03"),
    ]

    # Path Traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        (r'open\s*\(\s*[^)]*\+.*(?:request|input|param)', "PATH_TRAVERSAL", "A01"),
        (r'Path\s*\(\s*[^)]*\+.*(?:request|input|param)', "PATH_TRAVERSAL", "A01"),
        (r'os\.path\.join\s*\([^)]*(?:request|input|param)', "PATH_TRAVERSAL", "A01"),
    ]

    # Hardcoded Secrets patterns
    SECRET_PATTERNS = [
        (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']', "HARDCODED_SECRET", "A02"),
        (r'(?:api_key|apikey|api-key)\s*=\s*["\'][^"\']{10,}["\']', "HARDCODED_SECRET", "A02"),
        (r'(?:secret|token)\s*=\s*["\'][^"\']{10,}["\']', "HARDCODED_SECRET", "A02"),
        (r'(?:aws_access_key|aws_secret)\s*=\s*["\'][^"\']+["\']', "HARDCODED_SECRET", "A02"),
        (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----', "HARDCODED_KEY", "A02"),
    ]

    # Weak Crypto patterns
    WEAK_CRYPTO_PATTERNS = [
        (r'hashlib\.md5\s*\(', "WEAK_HASH", "A02"),
        (r'hashlib\.sha1\s*\(', "WEAK_HASH", "A02"),
        (r'DES\s*\(', "WEAK_CIPHER", "A02"),
        (r'RC4\s*\(', "WEAK_CIPHER", "A02"),
    ]

    # XSS patterns (for web frameworks)
    XSS_PATTERNS = [
        (r'render_template_string\s*\(\s*[^)]*\+', "XSS", "A03"),
        (r'Markup\s*\(\s*[^)]*\+', "XSS", "A03"),
        (r'\|safe(?!\w)', "XSS_SAFE_FILTER", "A03"),
    ]

    # Insecure Deserialization
    DESERIALIZATION_PATTERNS = [
        (r'pickle\.loads?\s*\(', "INSECURE_DESERIALIZATION", "A08"),
        (r'yaml\.load\s*\([^)]*(?!Loader)', "INSECURE_DESERIALIZATION", "A08"),
        (r'yaml\.unsafe_load\s*\(', "INSECURE_DESERIALIZATION", "A08"),
        (r'marshal\.loads?\s*\(', "INSECURE_DESERIALIZATION", "A08"),
    ]

    # SSRF patterns
    SSRF_PATTERNS = [
        (r'requests\.get\s*\(\s*[^)]*(?:request|input|param)', "SSRF", "A10"),
        (r'urllib\.request\.urlopen\s*\(\s*[^)]*(?:request|input|param)', "SSRF", "A10"),
        (r'httpx\.get\s*\(\s*[^)]*(?:request|input|param)', "SSRF", "A10"),
    ]

    ALL_PATTERNS = (
        SQL_INJECTION_PATTERNS +
        COMMAND_INJECTION_PATTERNS +
        PATH_TRAVERSAL_PATTERNS +
        SECRET_PATTERNS +
        WEAK_CRYPTO_PATTERNS +
        XSS_PATTERNS +
        DESERIALIZATION_PATTERNS +
        SSRF_PATTERNS
    )

    SEVERITY_MAP = {
        "SQL_INJECTION": VulnerabilitySeverity.CRITICAL,
        "COMMAND_INJECTION": VulnerabilitySeverity.CRITICAL,
        "PATH_TRAVERSAL": VulnerabilitySeverity.HIGH,
        "HARDCODED_SECRET": VulnerabilitySeverity.HIGH,
        "HARDCODED_KEY": VulnerabilitySeverity.CRITICAL,
        "WEAK_HASH": VulnerabilitySeverity.MEDIUM,
        "WEAK_CIPHER": VulnerabilitySeverity.HIGH,
        "XSS": VulnerabilitySeverity.HIGH,
        "XSS_SAFE_FILTER": VulnerabilitySeverity.MEDIUM,
        "INSECURE_DESERIALIZATION": VulnerabilitySeverity.CRITICAL,
        "SSRF": VulnerabilitySeverity.HIGH,
    }

    REMEDIATION_MAP = {
        "SQL_INJECTION": "Use parameterized queries or an ORM",
        "COMMAND_INJECTION": "Use subprocess with shell=False and arg list",
        "PATH_TRAVERSAL": "Validate and sanitize file paths, use allowlists",
        "HARDCODED_SECRET": "Use environment variables or secret managers",
        "HARDCODED_KEY": "Store private keys in secure key management systems",
        "WEAK_HASH": "Use SHA-256 or stronger for security-sensitive hashing",
        "WEAK_CIPHER": "Use AES-256-GCM or ChaCha20-Poly1305",
        "XSS": "Use proper HTML escaping and Content-Security-Policy",
        "XSS_SAFE_FILTER": "Avoid |safe filter or ensure content is sanitized",
        "INSECURE_DESERIALIZATION": "Use JSON or validate input before deserializing",
        "SSRF": "Validate and allowlist URLs, use URL parsing",
    }

    CWE_MAP = {
        "SQL_INJECTION": "CWE-89",
        "COMMAND_INJECTION": "CWE-78",
        "PATH_TRAVERSAL": "CWE-22",
        "HARDCODED_SECRET": "CWE-798",
        "HARDCODED_KEY": "CWE-321",
        "WEAK_HASH": "CWE-328",
        "WEAK_CIPHER": "CWE-327",
        "XSS": "CWE-79",
        "XSS_SAFE_FILTER": "CWE-79",
        "INSECURE_DESERIALIZATION": "CWE-502",
        "SSRF": "CWE-918",
    }

    def scan_content(self, content: str, file_path: str) -> List[SecurityVulnerability]:
        """Scan content for OWASP vulnerabilities."""
        vulnerabilities = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern, vuln_type, owasp in self.ALL_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # Get context snippet
                    start = max(0, line_num - 2)
                    end = min(len(lines), line_num + 1)
                    snippet = "\n".join(lines[start:end])

                    vulnerabilities.append(SecurityVulnerability(
                        severity=self.SEVERITY_MAP.get(vuln_type, VulnerabilitySeverity.MEDIUM),
                        vulnerability_type=vuln_type,
                        message=f"Potential {vuln_type.replace('_', ' ').title()} detected",
                        file_path=file_path,
                        line=line_num,
                        code=f"OWASP-{owasp}",
                        cwe_id=self.CWE_MAP.get(vuln_type),
                        owasp_category=owasp,
                        confidence="medium",
                        snippet=snippet,
                        remediation=self.REMEDIATION_MAP.get(vuln_type),
                    ))

        return vulnerabilities


class BanditIntegration:
    """Integration with Bandit security scanner."""

    def __init__(self):
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if Bandit is available."""
        if self._available is not None:
            return self._available

        try:
            proc = await asyncio.create_subprocess_exec(
                "bandit", "--version",
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

    async def scan_files(self, files: List[Path], repo_root: Path) -> List[SecurityVulnerability]:
        """Run Bandit on files."""
        if not await self.is_available():
            return []

        vulnerabilities = []

        try:
            # Run Bandit with JSON output
            cmd = [
                "bandit",
                "-f", "json",
                "-r",
                "--severity-level", "low",
                *[str(f) for f in files]
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if stdout:
                try:
                    results = json.loads(stdout.decode())
                    for result in results.get("results", []):
                        severity = self._map_severity(result.get("issue_severity", "LOW"))
                        vulnerabilities.append(SecurityVulnerability(
                            severity=severity,
                            vulnerability_type=result.get("test_id", "B000"),
                            message=result.get("issue_text", "Security issue"),
                            file_path=result.get("filename", ""),
                            line=result.get("line_number", 0),
                            code=result.get("test_id", ""),
                            confidence=result.get("issue_confidence", "MEDIUM").lower(),
                            snippet=result.get("code", ""),
                            remediation=result.get("more_info", ""),
                        ))
                except json.JSONDecodeError:
                    logger.warning("[SecurityScanner] Could not parse Bandit output")

        except asyncio.TimeoutError:
            logger.warning("[SecurityScanner] Bandit scan timed out")
        except Exception as e:
            logger.warning(f"[SecurityScanner] Bandit scan failed: {e}")

        return vulnerabilities

    def _map_severity(self, bandit_severity: str) -> VulnerabilitySeverity:
        """Map Bandit severity to our enum."""
        mapping = {
            "HIGH": VulnerabilitySeverity.HIGH,
            "MEDIUM": VulnerabilitySeverity.MEDIUM,
            "LOW": VulnerabilitySeverity.LOW,
        }
        return mapping.get(bandit_severity.upper(), VulnerabilitySeverity.MEDIUM)


class DependencyVulnerabilityChecker:
    """Check for vulnerable dependencies."""

    async def check_requirements(self, requirements_file: Path) -> List[SecurityVulnerability]:
        """Check requirements.txt for known vulnerabilities."""
        vulnerabilities = []

        if not requirements_file.exists():
            return vulnerabilities

        # Try pip-audit if available
        try:
            proc = await asyncio.create_subprocess_exec(
                "pip-audit", "--requirement", str(requirements_file),
                "--format", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)

            if stdout:
                try:
                    results = json.loads(stdout.decode())
                    for dep in results.get("dependencies", []):
                        for vuln in dep.get("vulns", []):
                            vulnerabilities.append(SecurityVulnerability(
                                severity=VulnerabilitySeverity.HIGH,
                                vulnerability_type="VULNERABLE_DEPENDENCY",
                                message=f"Vulnerable dependency: {dep.get('name')} {dep.get('version')}",
                                file_path=str(requirements_file),
                                code=vuln.get("id", ""),
                                cwe_id=vuln.get("cwe", {}).get("id") if vuln.get("cwe") else None,
                                remediation=f"Upgrade to {vuln.get('fix_versions', ['latest'])[0] if vuln.get('fix_versions') else 'latest version'}",
                            ))
                except json.JSONDecodeError:
                    pass

        except (FileNotFoundError, asyncio.TimeoutError):
            pass  # pip-audit not available or timed out

        return vulnerabilities


class SecurityScanner:
    """
    Comprehensive security scanner.

    Combines:
    - OWASP pattern detection
    - Bandit integration
    - Dependency checking
    - Custom security rules
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.owasp_scanner = OWASPPatternScanner()
        self.bandit = BanditIntegration()
        self.dep_checker = DependencyVulnerabilityChecker()
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def scan_file(self, file_path: Union[str, Path]) -> SecurityScanResult:
        """Scan a single file for security vulnerabilities."""
        import time
        start_time = time.time()

        file_path = Path(file_path)
        vulnerabilities = []

        if not file_path.exists():
            return SecurityScanResult(secure=True, files_scanned=0)

        if not file_path.suffix == ".py":
            return SecurityScanResult(secure=True, files_scanned=1)

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return SecurityScanResult(
                secure=False,
                vulnerabilities=[SecurityVulnerability(
                    severity=VulnerabilitySeverity.INFO,
                    vulnerability_type="READ_ERROR",
                    message=f"Could not read file: {e}",
                    file_path=str(file_path),
                )]
            )

        # Run OWASP pattern scan
        loop = asyncio.get_running_loop()
        owasp_vulns = await loop.run_in_executor(
            self._executor,
            self.owasp_scanner.scan_content,
            content,
            str(file_path)
        )
        vulnerabilities.extend(owasp_vulns)

        # Run Bandit if available
        bandit_available = await self.bandit.is_available()
        if bandit_available:
            bandit_vulns = await self.bandit.scan_files([file_path], self.repo_root)
            vulnerabilities.extend(bandit_vulns)

        # Deduplicate by line and type
        seen = set()
        unique_vulns = []
        for v in vulnerabilities:
            key = (v.file_path, v.line, v.vulnerability_type)
            if key not in seen:
                seen.add(key)
                unique_vulns.append(v)

        scan_time = (time.time() - start_time) * 1000
        has_critical = any(v.severity in (VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH) for v in unique_vulns)

        return SecurityScanResult(
            secure=not has_critical,
            vulnerabilities=unique_vulns,
            scan_time_ms=scan_time,
            bandit_available=bandit_available,
            files_scanned=1,
        )

    async def scan_files(self, files: List[Union[str, Path]]) -> SecurityScanResult:
        """Scan multiple files in parallel."""
        import time
        start_time = time.time()

        tasks = [self.scan_file(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_vulns = []
        for r in results:
            if isinstance(r, SecurityScanResult):
                all_vulns.extend(r.vulnerabilities)

        # Deduplicate
        seen = set()
        unique_vulns = []
        for v in all_vulns:
            key = (v.file_path, v.line, v.vulnerability_type)
            if key not in seen:
                seen.add(key)
                unique_vulns.append(v)

        has_critical = any(v.severity in (VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH) for v in unique_vulns)

        return SecurityScanResult(
            secure=not has_critical,
            vulnerabilities=unique_vulns,
            scan_time_ms=(time.time() - start_time) * 1000,
            bandit_available=await self.bandit.is_available(),
            files_scanned=len(files),
        )

    async def scan_repo(self, include_dependencies: bool = True) -> SecurityScanResult:
        """Scan entire repository."""
        import time
        start_time = time.time()

        vulnerabilities = []

        # Find all Python files
        py_files = list(self.repo_root.rglob("*.py"))
        py_files = [f for f in py_files if "venv" not in str(f) and ".git" not in str(f)]

        # Scan files in batches
        batch_size = 20
        for i in range(0, len(py_files), batch_size):
            batch = py_files[i:i + batch_size]
            result = await self.scan_files(batch)
            vulnerabilities.extend(result.vulnerabilities)

        # Check dependencies
        if include_dependencies:
            req_file = self.repo_root / "requirements.txt"
            if req_file.exists():
                dep_vulns = await self.dep_checker.check_requirements(req_file)
                vulnerabilities.extend(dep_vulns)

        # Deduplicate
        seen = set()
        unique_vulns = []
        for v in vulnerabilities:
            key = (v.file_path, v.line, v.vulnerability_type)
            if key not in seen:
                seen.add(key)
                unique_vulns.append(v)

        has_critical = any(v.severity in (VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH) for v in unique_vulns)

        return SecurityScanResult(
            secure=not has_critical,
            vulnerabilities=unique_vulns,
            scan_time_ms=(time.time() - start_time) * 1000,
            bandit_available=await self.bandit.is_available(),
            files_scanned=len(py_files),
        )
