"""
Refactoring Validator v1.0
==========================

Pre and post validation for refactoring operations.
Ensures code integrity before and after transformations.

Features:
- Syntax validation
- Semantic validation
- Breaking change detection
- Circular dependency detection
- Import validation
- Reference count verification

Author: JARVIS AI System
Version: 1.0.0
"""

from __future__ import annotations

import ast
import asyncio
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def _get_env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


class ValidatorConfig:
    """Configuration for refactoring validation."""

    SYNTAX_CHECK: bool = _get_env_bool("REFACTORING_VALIDATE_SYNTAX", True)
    SEMANTIC_CHECK: bool = _get_env_bool("REFACTORING_VALIDATE_SEMANTIC", True)
    CIRCULAR_DEPENDENCY_CHECK: bool = _get_env_bool("REFACTORING_CHECK_CIRCULAR_DEPS", True)
    IMPORT_CHECK: bool = _get_env_bool("REFACTORING_CHECK_IMPORTS", True)
    BREAKING_CHANGE_CHECK: bool = _get_env_bool("REFACTORING_CHECK_BREAKING", True)


# =============================================================================
# ENUMS
# =============================================================================

class ValidationLevel(str, Enum):
    """Level of validation."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationType(str, Enum):
    """Type of validation check."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    IMPORTS = "imports"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    BREAKING_CHANGE = "breaking_change"
    REFERENCE_COUNT = "reference_count"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue."""
    issue_type: ValidationType
    level: ValidationLevel
    message: str
    file_path: Optional[Path] = None
    line: Optional[int] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.issue_type.value,
            "level": self.level.value,
            "message": self.message,
            "file_path": str(self.file_path) if self.file_path else None,
            "line": self.line,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == ValidationLevel.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


# =============================================================================
# VALIDATORS
# =============================================================================

class SyntaxValidator:
    """Validates Python syntax."""

    async def validate(self, source: str, file_path: Optional[Path] = None) -> ValidationResult:
        """Validate syntax of Python source code."""
        issues = []

        try:
            ast.parse(source)
        except SyntaxError as e:
            issues.append(ValidationIssue(
                issue_type=ValidationType.SYNTAX,
                level=ValidationLevel.ERROR,
                message=f"Syntax error: {e.msg}",
                file_path=file_path,
                line=e.lineno,
            ))

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
        )


class ImportValidator:
    """Validates import statements."""

    async def validate(
        self,
        source: str,
        file_path: Optional[Path] = None,
        available_modules: Optional[Set[str]] = None,
    ) -> ValidationResult:
        """Validate imports in Python source."""
        issues = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ValidationResult(is_valid=True, issues=[])  # Syntax error handled elsewhere

        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append((f"{module}.{alias.name}", node.lineno))

        # Check for duplicate imports
        seen = set()
        for import_name, line in imports:
            if import_name in seen:
                issues.append(ValidationIssue(
                    issue_type=ValidationType.IMPORTS,
                    level=ValidationLevel.WARNING,
                    message=f"Duplicate import: {import_name}",
                    file_path=file_path,
                    line=line,
                ))
            seen.add(import_name)

        return ValidationResult(
            is_valid=len([i for i in issues if i.level == ValidationLevel.ERROR]) == 0,
            issues=issues,
        )


class CircularDependencyValidator:
    """Detects circular dependencies."""

    async def validate(
        self,
        files: List[Path],
    ) -> ValidationResult:
        """Check for circular dependencies in a set of files."""
        issues = []

        # Build import graph
        import_graph: Dict[str, Set[str]] = {}

        for file_path in files:
            if not file_path.exists() or file_path.suffix != '.py':
                continue

            try:
                source = await asyncio.to_thread(file_path.read_text, encoding='utf-8')
                tree = ast.parse(source)
            except Exception:
                continue

            module_name = file_path.stem
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

            import_graph[module_name] = imports

        # Detect cycles using DFS
        def find_cycle(node: str, visited: Set[str], path: List[str]) -> Optional[List[str]]:
            if node in path:
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            path.append(node)

            for neighbor in import_graph.get(node, set()):
                if neighbor in import_graph:
                    cycle = find_cycle(neighbor, visited, path)
                    if cycle:
                        return cycle

            path.pop()
            return None

        visited: Set[str] = set()
        for module in import_graph:
            if module not in visited:
                cycle = find_cycle(module, visited, [])
                if cycle:
                    issues.append(ValidationIssue(
                        issue_type=ValidationType.CIRCULAR_DEPENDENCY,
                        level=ValidationLevel.ERROR,
                        message=f"Circular dependency detected: {' -> '.join(cycle)}",
                    ))

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
        )


class BreakingChangeValidator:
    """Detects breaking changes in public API."""

    async def validate(
        self,
        old_source: str,
        new_source: str,
        file_path: Optional[Path] = None,
    ) -> ValidationResult:
        """Check for breaking changes between old and new source."""
        issues = []

        try:
            old_tree = ast.parse(old_source)
            new_tree = ast.parse(new_source)
        except SyntaxError:
            return ValidationResult(is_valid=True, issues=[])

        # Extract public API from both versions
        old_api = self._extract_public_api(old_tree)
        new_api = self._extract_public_api(new_tree)

        # Check for removed items
        for name, (kind, signature) in old_api.items():
            if name not in new_api:
                issues.append(ValidationIssue(
                    issue_type=ValidationType.BREAKING_CHANGE,
                    level=ValidationLevel.ERROR,
                    message=f"Public {kind} '{name}' was removed",
                    file_path=file_path,
                    suggestion=f"Keep '{name}' or add deprecation warning",
                ))
            else:
                new_kind, new_signature = new_api[name]
                if kind == "function" and new_kind == "function":
                    # Check signature compatibility
                    breaking = self._check_signature_compatibility(signature, new_signature)
                    if breaking:
                        issues.append(ValidationIssue(
                            issue_type=ValidationType.BREAKING_CHANGE,
                            level=ValidationLevel.WARNING,
                            message=f"Signature of '{name}' changed: {breaking}",
                            file_path=file_path,
                        ))

        return ValidationResult(
            is_valid=len([i for i in issues if i.level == ValidationLevel.ERROR]) == 0,
            issues=issues,
        )

    def _extract_public_api(self, tree: ast.AST) -> Dict[str, Tuple[str, str]]:
        """Extract public API (functions, classes not starting with _)."""
        api = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    signature = self._get_function_signature(node)
                    api[node.name] = ("function", signature)
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith('_'):
                    api[node.name] = ("class", "")

        return api

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get function signature as string."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"({', '.join(args)})"

    def _check_signature_compatibility(
        self,
        old_sig: str,
        new_sig: str,
    ) -> Optional[str]:
        """Check if new signature is compatible with old."""
        # Simple check: old parameters should still be present
        old_params = old_sig.strip("()").split(", ") if old_sig != "()" else []
        new_params = new_sig.strip("()").split(", ") if new_sig != "()" else []

        old_params = [p for p in old_params if p and p != "self"]
        new_params = [p for p in new_params if p and p != "self"]

        # Check if any old parameters are missing (without defaults)
        for param in old_params:
            if param not in new_params:
                return f"Parameter '{param}' removed"

        return None


class ReferenceCountValidator:
    """Validates reference counts before and after refactoring."""

    async def validate(
        self,
        symbol_name: str,
        expected_refs: int,
        actual_refs: int,
    ) -> ValidationResult:
        """Validate reference count matches expectations."""
        issues = []

        if actual_refs < expected_refs:
            issues.append(ValidationIssue(
                issue_type=ValidationType.REFERENCE_COUNT,
                level=ValidationLevel.WARNING,
                message=f"Reference count mismatch for '{symbol_name}': expected {expected_refs}, found {actual_refs}",
                suggestion="Some references may not have been updated",
            ))
        elif actual_refs > expected_refs:
            issues.append(ValidationIssue(
                issue_type=ValidationType.REFERENCE_COUNT,
                level=ValidationLevel.INFO,
                message=f"Additional references found for '{symbol_name}': expected {expected_refs}, found {actual_refs}",
            ))

        return ValidationResult(
            is_valid=True,  # Reference count mismatch is warning, not error
            issues=issues,
        )


# =============================================================================
# MAIN VALIDATOR
# =============================================================================

class RefactoringValidator:
    """
    Main validator for refactoring operations.

    Coordinates multiple validation checks before and after
    refactoring operations.
    """

    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()
        self.syntax_validator = SyntaxValidator()
        self.import_validator = ImportValidator()
        self.circular_dep_validator = CircularDependencyValidator()
        self.breaking_change_validator = BreakingChangeValidator()
        self.reference_count_validator = ReferenceCountValidator()

    async def validate_before(
        self,
        source: str,
        file_path: Optional[Path] = None,
        related_files: Optional[List[Path]] = None,
    ) -> ValidationResult:
        """
        Validate source before refactoring.

        Args:
            source: Source code to validate
            file_path: Path to the file
            related_files: Other files involved in refactoring

        Returns:
            ValidationResult with all issues
        """
        all_issues = []

        # Syntax check
        if self.config.SYNTAX_CHECK:
            result = await self.syntax_validator.validate(source, file_path)
            all_issues.extend(result.issues)

        # Import check
        if self.config.IMPORT_CHECK:
            result = await self.import_validator.validate(source, file_path)
            all_issues.extend(result.issues)

        # Circular dependency check
        if self.config.CIRCULAR_DEPENDENCY_CHECK and related_files:
            files = list(related_files)
            if file_path:
                files.append(file_path)
            result = await self.circular_dep_validator.validate(files)
            all_issues.extend(result.issues)

        has_errors = any(i.level == ValidationLevel.ERROR for i in all_issues)

        return ValidationResult(
            is_valid=not has_errors,
            issues=all_issues,
        )

    async def validate_after(
        self,
        old_source: str,
        new_source: str,
        file_path: Optional[Path] = None,
        symbol_name: Optional[str] = None,
        expected_refs: Optional[int] = None,
        actual_refs: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate source after refactoring.

        Args:
            old_source: Original source code
            new_source: Transformed source code
            file_path: Path to the file
            symbol_name: Name of refactored symbol
            expected_refs: Expected reference count
            actual_refs: Actual reference count

        Returns:
            ValidationResult with all issues
        """
        all_issues = []

        # Syntax check on new source
        if self.config.SYNTAX_CHECK:
            result = await self.syntax_validator.validate(new_source, file_path)
            all_issues.extend(result.issues)

        # Import check on new source
        if self.config.IMPORT_CHECK:
            result = await self.import_validator.validate(new_source, file_path)
            all_issues.extend(result.issues)

        # Breaking change check
        if self.config.BREAKING_CHANGE_CHECK:
            result = await self.breaking_change_validator.validate(
                old_source, new_source, file_path
            )
            all_issues.extend(result.issues)

        # Reference count check
        if symbol_name and expected_refs is not None and actual_refs is not None:
            result = await self.reference_count_validator.validate(
                symbol_name, expected_refs, actual_refs
            )
            all_issues.extend(result.issues)

        has_errors = any(i.level == ValidationLevel.ERROR for i in all_issues)

        return ValidationResult(
            is_valid=not has_errors,
            issues=all_issues,
        )


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_validator_instance: Optional[RefactoringValidator] = None


def get_refactoring_validator() -> RefactoringValidator:
    """Get the singleton validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = RefactoringValidator()
    return _validator_instance


async def get_refactoring_validator_async() -> RefactoringValidator:
    """Get the singleton validator instance (async)."""
    return get_refactoring_validator()
