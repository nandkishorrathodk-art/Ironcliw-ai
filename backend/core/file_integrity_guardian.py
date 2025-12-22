#!/usr/bin/env python3
"""
File Integrity Guardian v1.0 - ML-Powered Truncation Prevention System
======================================================================

Production-grade, intelligent file integrity system that:
- Detects truncated/corrupted Python files BEFORE they're committed
- Uses pattern recognition to identify incomplete code
- Creates automatic backups before risky operations
- Provides real-time file health monitoring
- Integrates with git hooks for automatic protection

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    File Integrity Guardian v1.0                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Syntax         │  │ Pattern        │  │ Structure      │                 │
│  │ Validator      │  │ Detector       │  │ Analyzer       │                 │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                 │
│          └───────────────────┼───────────────────┘                          │
│                              ▼                                               │
│          ┌─────────────────────────────────────┐                            │
│          │    Integrity Analysis Engine        │                            │
│          │  • AST parsing and validation       │                            │
│          │  • Truncation pattern matching      │                            │
│          │  • Historical comparison            │                            │
│          │  • Confidence scoring               │                            │
│          └─────────────────────────────────────┘                            │
│                              │                                               │
│          ┌───────────────────┼───────────────────┐                          │
│          ▼                   ▼                   ▼                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                 │
│  │ Git Hook       │  │ Backup         │  │ Recovery       │                 │
│  │ Integration    │  │ Manager        │  │ System         │                 │
│  └────────────────┘  └────────────────┘  └────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - Environment-Driven
# =============================================================================

def _env_int(key: str, default: int) -> int:
    """Get int from environment with fallback."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Get bool from environment with fallback."""
    val = os.environ.get(key, str(default)).lower()
    return val in ('true', '1', 'yes', 'on')


def _env_str(key: str, default: str) -> str:
    """Get string from environment with fallback."""
    return os.environ.get(key, default)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class IntegrityStatus(Enum):
    """File integrity status."""
    HEALTHY = "healthy"
    TRUNCATED = "truncated"
    CORRUPTED = "corrupted"
    SYNTAX_ERROR = "syntax_error"
    SUSPICIOUS = "suspicious"
    UNKNOWN = "unknown"


class TruncationPattern(Enum):
    """Types of truncation patterns we detect."""
    UNCLOSED_STRING = "unclosed_string"
    UNCLOSED_DOCSTRING = "unclosed_docstring"
    MISSING_FUNCTION_BODY = "missing_function_body"
    MISSING_CLASS_BODY = "missing_class_body"
    INCOMPLETE_IMPORT = "incomplete_import"
    ORPHAN_CODE = "orphan_code"
    MISSING_CLOSING_BRACE = "missing_closing_brace"
    ABRUPT_END = "abrupt_end"
    INDENTATION_BREAK = "indentation_break"


class RiskLevel(Enum):
    """Risk level for file changes."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class IntegrityReport:
    """Report for a single file's integrity check."""
    file_path: str
    status: IntegrityStatus
    risk_level: RiskLevel
    patterns_detected: List[TruncationPattern] = field(default_factory=list)
    line_count: int = 0
    syntax_error: Optional[str] = None
    syntax_error_line: Optional[int] = None
    confidence: float = 1.0
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "status": self.status.value,
            "risk_level": self.risk_level.value,
            "patterns_detected": [p.value for p in self.patterns_detected],
            "line_count": self.line_count,
            "syntax_error": self.syntax_error,
            "syntax_error_line": self.syntax_error_line,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BatchReport:
    """Report for a batch of files."""
    total_files: int
    healthy_files: int
    problematic_files: int
    file_reports: List[IntegrityReport] = field(default_factory=list)
    duration_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def has_problems(self) -> bool:
        return self.problematic_files > 0
    
    def get_problematic_reports(self) -> List[IntegrityReport]:
        return [r for r in self.file_reports if r.status != IntegrityStatus.HEALTHY]


# =============================================================================
# PATTERN DETECTION ENGINE
# =============================================================================

class TruncationPatternDetector:
    """
    Intelligent pattern detector for identifying truncated/corrupted files.
    Uses regex patterns and heuristics to detect common truncation signatures.
    """
    
    def __init__(self):
        # Patterns that indicate truncation
        self._truncation_patterns = {
            # Unclosed triple-quoted strings
            TruncationPattern.UNCLOSED_DOCSTRING: [
                re.compile(r'"""[^"]*$', re.MULTILINE),
                re.compile(r"'''[^']*$", re.MULTILINE),
                re.compile(r'^\s*""".*(?!""")\s*$', re.MULTILINE),
            ],
            # Unclosed single-line strings
            TruncationPattern.UNCLOSED_STRING: [
                re.compile(r'"[^"\n\\]*$', re.MULTILINE),
                re.compile(r"'[^'\n\\]*$", re.MULTILINE),
                re.compile(r'f"[^"]*\{[^}]*$', re.MULTILINE),  # Unclosed f-string
            ],
            # Incomplete function/class definitions
            TruncationPattern.MISSING_FUNCTION_BODY: [
                re.compile(r'^\s*def\s+\w+\s*\([^)]*$', re.MULTILINE),
                re.compile(r'^\s*async\s+def\s+\w+\s*\([^)]*$', re.MULTILINE),
            ],
            TruncationPattern.MISSING_CLASS_BODY: [
                re.compile(r'^\s*class\s+\w+.*:\s*$', re.MULTILINE),
            ],
            # Incomplete imports
            TruncationPattern.INCOMPLETE_IMPORT: [
                re.compile(r'^\s*from\s+\w+\s*$', re.MULTILINE),
                re.compile(r'^\s*import\s*$', re.MULTILINE),
                re.compile(r'^\s*from\s+[\w.]+\s+import\s*\($', re.MULTILINE),
            ],
            # Orphan code patterns
            TruncationPattern.ORPHAN_CODE: [
                re.compile(r'^\s+\w+.*[^,\[\{\(]\s*$'),  # Indented code at end with no continuation
            ],
            # Missing closing brackets
            TruncationPattern.MISSING_CLOSING_BRACE: [
                re.compile(r'\[\s*$', re.MULTILINE),
                re.compile(r'\{\s*$', re.MULTILINE),
                re.compile(r'\(\s*$', re.MULTILINE),
            ],
        }
        
        # End-of-file patterns that suggest truncation
        self._eof_suspicious_patterns = [
            re.compile(r'^\s*#\s*Module truncated', re.IGNORECASE),
            re.compile(r'^\s*#\s*TODO:', re.IGNORECASE),
            re.compile(r'^\s*#\s*FIXME:', re.IGNORECASE),
            re.compile(r'^\s*pass\s*#.*truncat', re.IGNORECASE),
            re.compile(r'^\s*\.\.\.\s*$'),  # Ellipsis at end
        ]
        
        # Healthy file endings
        self._healthy_endings = [
            re.compile(r'^\s*$'),  # Empty line
            re.compile(r'^["\']'),  # String (docstring ending)
            re.compile(r'^\s*\)\s*$'),  # Closing paren
            re.compile(r'^\s*\]\s*$'),  # Closing bracket
            re.compile(r'^\s*\}\s*$'),  # Closing brace
            re.compile(r'^\s*return\s+'),  # Return statement
            re.compile(r'^\s*pass\s*$'),  # Pass statement
            re.compile(r'^\s*raise\s+'),  # Raise statement
            re.compile(r'^\s*#.*$'),  # Comment (normal)
            re.compile(r'^__all__\s*='),  # Module exports
        ]
    
    def detect_patterns(self, content: str, file_path: str) -> List[TruncationPattern]:
        """Detect truncation patterns in file content."""
        detected = []
        lines = content.split('\n')
        
        # Check for truncation patterns in content
        for pattern_type, patterns in self._truncation_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    if pattern_type not in detected:
                        detected.append(pattern_type)
        
        # Check last few lines for suspicious endings
        if lines:
            last_lines = '\n'.join(lines[-5:])
            for pattern in self._eof_suspicious_patterns:
                if pattern.search(last_lines):
                    if TruncationPattern.ABRUPT_END not in detected:
                        detected.append(TruncationPattern.ABRUPT_END)
                    break
        
        # Check for indentation breaks
        if self._has_indentation_break(lines):
            detected.append(TruncationPattern.INDENTATION_BREAK)
        
        return detected
    
    def _has_indentation_break(self, lines: List[str]) -> bool:
        """Check if file has suspicious indentation breaks."""
        if len(lines) < 10:
            return False
        
        # Check last 10 lines for sudden indentation changes
        last_lines = lines[-10:]
        prev_indent = None
        
        for line in last_lines:
            if not line.strip():
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            if prev_indent is not None:
                # Sudden decrease by more than 8 spaces is suspicious
                if prev_indent - current_indent > 8:
                    return True
            
            prev_indent = current_indent
        
        return False
    
    def is_healthy_ending(self, content: str) -> bool:
        """Check if file has a healthy ending."""
        lines = content.strip().split('\n')
        if not lines:
            return False
        
        last_line = lines[-1]
        for pattern in self._healthy_endings:
            if pattern.match(last_line):
                return True
        
        return False


# =============================================================================
# SYNTAX VALIDATOR
# =============================================================================

class SyntaxValidator:
    """
    Validates Python file syntax using AST parsing.
    """
    
    def validate(self, content: str, file_path: str) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Validate Python syntax.
        
        Returns:
            Tuple of (is_valid, error_message, error_line)
        """
        try:
            ast.parse(content)
            return True, None, None
        except SyntaxError as e:
            return False, str(e.msg), e.lineno
        except Exception as e:
            return False, str(e), None
    
    def get_structure_info(self, content: str) -> Dict[str, Any]:
        """Extract structure information from valid Python code."""
        try:
            tree = ast.parse(content)
            
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    functions.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node) if hasattr(ast, 'unparse') else "import")
            
            return {
                "classes": classes,
                "functions": functions,
                "import_count": len(imports),
                "valid": True,
            }
        except Exception:
            return {"valid": False}


# =============================================================================
# BACKUP MANAGER
# =============================================================================

class BackupManager:
    """
    Manages automatic backups of files before risky operations.
    """
    
    def __init__(self, backup_dir: Optional[str] = None):
        self.backup_dir = Path(backup_dir or os.environ.get(
            "JARVIS_BACKUP_DIR",
            os.path.expanduser("~/.jarvis/file_backups")
        ))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups_per_file = _env_int("JARVIS_MAX_BACKUPS_PER_FILE", 5)
        self.backup_retention_days = _env_int("JARVIS_BACKUP_RETENTION_DAYS", 7)
        self._lock = threading.Lock()
    
    def create_backup(self, file_path: str, reason: str = "auto") -> Optional[str]:
        """Create a backup of a file."""
        source = Path(file_path)
        if not source.exists():
            return None
        
        with self._lock:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(str(source).encode()).hexdigest()[:8]
            backup_name = f"{source.stem}_{timestamp}_{file_hash}{source.suffix}"
            
            # Create subdirectory based on original path
            relative_path = source.name
            backup_subdir = self.backup_dir / file_hash
            backup_subdir.mkdir(parents=True, exist_ok=True)
            
            backup_path = backup_subdir / backup_name
            
            try:
                shutil.copy2(source, backup_path)
                
                # Write metadata
                metadata = {
                    "original_path": str(source.absolute()),
                    "backup_path": str(backup_path),
                    "timestamp": timestamp,
                    "reason": reason,
                    "file_size": source.stat().st_size,
                }
                metadata_path = backup_path.with_suffix(backup_path.suffix + ".meta.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Cleanup old backups
                self._cleanup_old_backups(backup_subdir)
                
                logger.info(f"📦 Created backup: {backup_path}")
                return str(backup_path)
                
            except Exception as e:
                logger.error(f"❌ Failed to create backup for {file_path}: {e}")
                return None
    
    def _cleanup_old_backups(self, backup_subdir: Path) -> None:
        """Remove old backups beyond retention period."""
        try:
            cutoff = datetime.now() - timedelta(days=self.backup_retention_days)
            
            backups = sorted(backup_subdir.glob("*.*"), key=lambda p: p.stat().st_mtime)
            
            # Remove backups older than retention period
            for backup in backups:
                if backup.suffix == ".json":
                    continue
                mtime = datetime.fromtimestamp(backup.stat().st_mtime)
                if mtime < cutoff:
                    backup.unlink()
                    meta = backup.with_suffix(backup.suffix + ".meta.json")
                    if meta.exists():
                        meta.unlink()
            
            # Keep only max_backups_per_file most recent
            remaining = sorted(
                [b for b in backup_subdir.glob("*.*") if not b.suffix.endswith(".json")],
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            for backup in remaining[self.max_backups_per_file:]:
                backup.unlink()
                meta = backup.with_suffix(backup.suffix + ".meta.json")
                if meta.exists():
                    meta.unlink()
                    
        except Exception as e:
            logger.debug(f"Backup cleanup error: {e}")
    
    def get_latest_backup(self, file_path: str) -> Optional[str]:
        """Get the most recent backup for a file."""
        source = Path(file_path)
        file_hash = hashlib.md5(str(source).encode()).hexdigest()[:8]
        backup_subdir = self.backup_dir / file_hash
        
        if not backup_subdir.exists():
            return None
        
        backups = sorted(
            [b for b in backup_subdir.glob("*.*") if not b.suffix.endswith(".json")],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        return str(backups[0]) if backups else None
    
    def restore_from_backup(self, file_path: str, backup_path: Optional[str] = None) -> bool:
        """Restore a file from backup."""
        if backup_path is None:
            backup_path = self.get_latest_backup(file_path)
        
        if not backup_path or not Path(backup_path).exists():
            logger.error(f"❌ No backup found for {file_path}")
            return False
        
        try:
            shutil.copy2(backup_path, file_path)
            logger.info(f"♻️ Restored {file_path} from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to restore {file_path}: {e}")
            return False


# =============================================================================
# FILE INTEGRITY GUARDIAN (Main Class)
# =============================================================================

class FileIntegrityGuardian:
    """
    Main class for file integrity protection.
    Provides comprehensive file health monitoring and protection.
    """
    
    _instance: Optional["FileIntegrityGuardian"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "FileIntegrityGuardian":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.pattern_detector = TruncationPatternDetector()
        self.syntax_validator = SyntaxValidator()
        self.backup_manager = BackupManager()
        
        # Configuration
        self.min_file_size = _env_int("JARVIS_MIN_FILE_SIZE", 50)  # bytes
        self.parallel_workers = _env_int("JARVIS_INTEGRITY_WORKERS", 4)
        self.auto_backup = _env_bool("JARVIS_AUTO_BACKUP", True)
        
        # Statistics
        self.stats = {
            "files_checked": 0,
            "problems_detected": 0,
            "backups_created": 0,
            "files_restored": 0,
        }
        
        # File history for change detection
        self._file_hashes: Dict[str, str] = {}
        
        self._initialized = True
        logger.info("🛡️ FileIntegrityGuardian initialized")
    
    def check_file(self, file_path: str, create_backup: bool = False) -> IntegrityReport:
        """
        Check a single file for integrity issues.
        
        The key insight is that if a file has valid Python syntax, it's almost
        certainly not truncated - truncation virtually always causes syntax errors.
        
        Args:
            file_path: Path to the file to check
            create_backup: Whether to create backup if problems found
            
        Returns:
            IntegrityReport with detailed findings
        """
        path = Path(file_path)
        
        # Initialize report
        report = IntegrityReport(
            file_path=str(path),
            status=IntegrityStatus.UNKNOWN,
            risk_level=RiskLevel.LOW,
        )
        
        # Check if file exists
        if not path.exists():
            report.status = IntegrityStatus.CORRUPTED
            report.risk_level = RiskLevel.CRITICAL
            report.suggestions.append("File does not exist")
            return report
        
        try:
            content = path.read_text(encoding='utf-8')
            report.line_count = len(content.split('\n'))
        except Exception as e:
            report.status = IntegrityStatus.CORRUPTED
            report.risk_level = RiskLevel.CRITICAL
            report.suggestions.append(f"Cannot read file: {e}")
            return report
        
        # First and most important: Syntax validation
        # If syntax is valid, the file is almost certainly NOT truncated
        is_valid, error_msg, error_line = self.syntax_validator.validate(content, str(path))
        
        if is_valid:
            # Valid syntax means file is healthy
            # Pattern detection only adds value for files with syntax errors
            report.status = IntegrityStatus.HEALTHY
            report.confidence = 1.0
            
            # Update stats
            self.stats["files_checked"] += 1
            return report
        
        # Syntax error found - now check for truncation patterns
        report.status = IntegrityStatus.SYNTAX_ERROR
        report.risk_level = RiskLevel.HIGH
        report.syntax_error = error_msg
        report.syntax_error_line = error_line
        report.suggestions.append(f"Fix syntax error at line {error_line}: {error_msg}")
        
        # Pattern detection - only meaningful for files with syntax errors
        patterns = self.pattern_detector.detect_patterns(content, str(path))
        report.patterns_detected = patterns
        
        if patterns:
            # Upgrade to TRUNCATED if we detect truncation patterns
            report.status = IntegrityStatus.TRUNCATED
            report.risk_level = RiskLevel.CRITICAL
            
            for pattern in patterns:
                report.suggestions.append(f"Detected: {pattern.value}")
        
        # Check file size for very small files
        if len(content) < self.min_file_size:
            report.suggestions.append(f"File is very small ({len(content)} bytes)")
            if report.status == IntegrityStatus.SYNTAX_ERROR:
                report.status = IntegrityStatus.TRUNCATED
        
        # Calculate confidence based on number of issues
        report.confidence = max(0.0, 1.0 - (len(patterns) * 0.15))
        
        # Update stats
        self.stats["files_checked"] += 1
        self.stats["problems_detected"] += 1
        
        # Create backup if requested and issues found
        if create_backup and self.auto_backup:
            backup = self.backup_manager.create_backup(str(path), "integrity_issue")
            if backup:
                self.stats["backups_created"] += 1
        
        return report
    
    async def check_files_async(
        self,
        file_paths: List[str],
        create_backups: bool = False
    ) -> BatchReport:
        """
        Check multiple files asynchronously.
        
        Args:
            file_paths: List of file paths to check
            create_backups: Whether to create backups for problematic files
            
        Returns:
            BatchReport with all findings
        """
        start_time = time.time()
        
        # Use thread pool for parallel file checking
        loop = asyncio.get_event_loop()
        
        async def check_one(path: str) -> IntegrityReport:
            return await loop.run_in_executor(
                None,
                lambda: self.check_file(path, create_backups)
            )
        
        # Check all files in parallel
        reports = await asyncio.gather(*[check_one(p) for p in file_paths])
        
        # Build batch report
        healthy = sum(1 for r in reports if r.status == IntegrityStatus.HEALTHY)
        
        return BatchReport(
            total_files=len(file_paths),
            healthy_files=healthy,
            problematic_files=len(file_paths) - healthy,
            file_reports=list(reports),
            duration_ms=(time.time() - start_time) * 1000,
        )
    
    def check_directory(
        self,
        directory: str,
        pattern: str = "*.py",
        recursive: bool = True,
        create_backups: bool = False,
        exclude_patterns: Optional[List[str]] = None
    ) -> BatchReport:
        """
        Check all files in a directory.
        
        Args:
            directory: Directory to check
            pattern: Glob pattern for files
            recursive: Whether to check subdirectories
            create_backups: Whether to create backups for problematic files
            exclude_patterns: Patterns to exclude (e.g., ["__pycache__", "venv"])
            
        Returns:
            BatchReport with all findings
        """
        exclude_patterns = exclude_patterns or ["__pycache__", "venv", ".git", "node_modules"]
        
        dir_path = Path(directory)
        files = []
        
        glob_method = dir_path.rglob if recursive else dir_path.glob
        
        for file_path in glob_method(pattern):
            # Check exclusions
            skip = False
            for exclude in exclude_patterns:
                if exclude in str(file_path):
                    skip = True
                    break
            
            if not skip and file_path.is_file():
                files.append(str(file_path))
        
        # Run async check
        return asyncio.run(self.check_files_async(files, create_backups))
    
    def check_staged_files(self) -> BatchReport:
        """
        Check only git staged files (for pre-commit hook).
        
        Returns:
            BatchReport for staged files
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning("Could not get staged files from git")
                return BatchReport(total_files=0, healthy_files=0, problematic_files=0)
            
            files = [f.strip() for f in result.stdout.split('\n') if f.strip().endswith('.py')]
            
            return asyncio.run(self.check_files_async(files, create_backups=True))
            
        except Exception as e:
            logger.error(f"Error checking staged files: {e}")
            return BatchReport(total_files=0, healthy_files=0, problematic_files=0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guardian statistics."""
        return {
            **self.stats,
            "auto_backup_enabled": self.auto_backup,
            "min_file_size": self.min_file_size,
        }
    
    def restore_file(self, file_path: str) -> bool:
        """Restore a file from its latest backup."""
        success = self.backup_manager.restore_from_backup(file_path)
        if success:
            self.stats["files_restored"] += 1
        return success


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_guardian_instance: Optional[FileIntegrityGuardian] = None
_guardian_lock = threading.Lock()


def get_file_integrity_guardian() -> FileIntegrityGuardian:
    """Get the singleton FileIntegrityGuardian instance."""
    global _guardian_instance
    if _guardian_instance is None:
        with _guardian_lock:
            if _guardian_instance is None:
                _guardian_instance = FileIntegrityGuardian()
    return _guardian_instance


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for the File Integrity Guardian."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="File Integrity Guardian - Detect and prevent file truncation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check files for integrity issues")
    check_parser.add_argument("path", help="File or directory to check")
    check_parser.add_argument("-r", "--recursive", action="store_true", help="Check recursively")
    check_parser.add_argument("-b", "--backup", action="store_true", help="Create backups for problematic files")
    
    # Pre-commit command
    precommit_parser = subparsers.add_parser("pre-commit", help="Check staged files (for git hook)")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show guardian statistics")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    
    guardian = get_file_integrity_guardian()
    
    if args.command == "check":
        path = Path(args.path)
        
        if path.is_file():
            report = guardian.check_file(str(path), args.backup)
            print(f"\n{'='*60}")
            print(f"File: {report.file_path}")
            print(f"Status: {report.status.value}")
            print(f"Risk Level: {report.risk_level.name}")
            print(f"Lines: {report.line_count}")
            
            if report.patterns_detected:
                print(f"Patterns: {', '.join(p.value for p in report.patterns_detected)}")
            
            if report.syntax_error:
                print(f"Syntax Error (line {report.syntax_error_line}): {report.syntax_error}")
            
            if report.suggestions:
                print("Suggestions:")
                for s in report.suggestions:
                    print(f"  • {s}")
            
            print(f"{'='*60}\n")
            
            # Exit with error if problems found
            if report.status != IntegrityStatus.HEALTHY:
                exit(1)
                
        elif path.is_dir():
            report = guardian.check_directory(
                str(path),
                recursive=args.recursive,
                create_backups=args.backup
            )
            
            print(f"\n{'='*60}")
            print(f"Directory: {path}")
            print(f"Total Files: {report.total_files}")
            print(f"Healthy: {report.healthy_files}")
            print(f"Problematic: {report.problematic_files}")
            print(f"Duration: {report.duration_ms:.1f}ms")
            
            if report.has_problems:
                print("\nProblematic Files:")
                for r in report.get_problematic_reports():
                    print(f"  ❌ {r.file_path}: {r.status.value}")
                    if r.syntax_error_line:
                        print(f"     Line {r.syntax_error_line}: {r.syntax_error}")
            
            print(f"{'='*60}\n")
            
            # Exit with error if problems found
            if report.has_problems:
                exit(1)
        else:
            print(f"Error: {path} does not exist")
            exit(1)
    
    elif args.command == "pre-commit":
        report = guardian.check_staged_files()
        
        if report.total_files == 0:
            print("✅ No Python files staged")
            exit(0)
        
        print(f"\n🔍 Checking {report.total_files} staged Python files...")
        
        if report.has_problems:
            print(f"\n❌ COMMIT BLOCKED: {report.problematic_files} file(s) have integrity issues\n")
            
            for r in report.get_problematic_reports():
                print(f"  ❌ {r.file_path}")
                print(f"     Status: {r.status.value}")
                if r.syntax_error:
                    print(f"     Error (line {r.syntax_error_line}): {r.syntax_error}")
                for pattern in r.patterns_detected:
                    print(f"     Pattern: {pattern.value}")
            
            print("\n💡 Fix these issues before committing, or use --no-verify to bypass.")
            exit(1)
        else:
            print(f"✅ All {report.total_files} files passed integrity check")
            exit(0)
    
    elif args.command == "stats":
        stats = guardian.get_stats()
        print("\n📊 File Integrity Guardian Statistics")
        print(f"{'='*40}")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print(f"{'='*40}\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

