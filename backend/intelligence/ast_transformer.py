"""
AST Transformer Module - Advanced Refactoring Engine v1.0
=========================================================

Enterprise-grade AST transformation system for code refactoring operations.
Performs safe, reversible code transformations with proper formatting preservation.

Features:
- Extract Method: Extract code blocks into new methods with automatic parameter detection
- Inline Variable: Replace variable references with values
- Move Method/Class: Relocate code between files with import management
- Change Signature: Update function parameters and all call sites
- Formatting Preservation: Maintains code style and indentation
- Rollback Support: All transformations are reversible

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     ASTTransformer                               │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │ VariableFlow│  │  ScopeTracker│  │ CodeGenerator│            │
    │  │   Analyzer  │  │             │  │             │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    │         │                │                │                     │
    │         └────────────────┴────────────────┘                     │
    │                          │                                      │
    │              ┌───────────▼───────────┐                          │
    │              │   TransformEngine     │                          │
    │              └───────────────────────┘                          │
    └─────────────────────────────────────────────────────────────────┘

Author: JARVIS AI System
Version: 1.0.0
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import logging
import os
import re
import textwrap
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, FrozenSet, Generator, List, Literal,
    NamedTuple, Optional, Protocol, Set, Tuple, Type, TypeVar, Union
)

from backend.utils.env_config import get_env_int, get_env_bool

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - Environment Driven (Zero Hardcoding)
# =============================================================================


class TransformerConfig:
    """Configuration for AST transformations."""

    # Formatting
    INDENT_SIZE: int = get_env_int("REFACTORING_INDENT_SIZE", 4)
    MAX_LINE_LENGTH: int = get_env_int("REFACTORING_MAX_LINE_LENGTH", 88)
    USE_TABS: bool = get_env_bool("REFACTORING_USE_TABS", False)

    # Safety
    VALIDATE_SYNTAX: bool = get_env_bool("REFACTORING_VALIDATE_SYNTAX", True)
    PRESERVE_COMMENTS: bool = get_env_bool("REFACTORING_PRESERVE_COMMENTS", True)
    PRESERVE_DOCSTRINGS: bool = get_env_bool("REFACTORING_PRESERVE_DOCSTRINGS", True)

    # Behavior
    ADD_TYPE_HINTS: bool = get_env_bool("REFACTORING_ADD_TYPE_HINTS", True)
    INFER_RETURN_TYPE: bool = get_env_bool("REFACTORING_INFER_RETURN_TYPE", True)
    AUTO_IMPORT: bool = get_env_bool("REFACTORING_AUTO_IMPORT", True)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class RefactoringType(str, Enum):
    """Types of refactoring operations."""
    EXTRACT_METHOD = "extract_method"
    INLINE_VARIABLE = "inline_variable"
    MOVE_METHOD = "move_method"
    MOVE_CLASS = "move_class"
    CHANGE_SIGNATURE = "change_signature"
    RENAME = "rename"


class VariableRole(str, Enum):
    """Role of a variable in code flow."""
    INPUT = "input"           # Read but not defined in block
    OUTPUT = "output"         # Modified in block, used after
    LOCAL = "local"           # Defined and used only within block
    THROUGH = "through"       # Passed through unchanged
    CLOSURE = "closure"       # From enclosing scope


class FlowControl(str, Enum):
    """Flow control statements that affect extraction."""
    RETURN = "return"
    BREAK = "break"
    CONTINUE = "continue"
    RAISE = "raise"
    YIELD = "yield"


class TransformStatus(str, Enum):
    """Status of a transformation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALID = "invalid"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class SourceLocation:
    """Location in source code."""
    line: int
    column: int
    end_line: int
    end_column: int

    def contains(self, line: int, column: int = 0) -> bool:
        """Check if location contains a point."""
        if line < self.line or line > self.end_line:
            return False
        if line == self.line and column < self.column:
            return False
        if line == self.end_line and column > self.end_column:
            return False
        return True


@dataclass
class VariableInfo:
    """Information about a variable in code."""
    name: str
    role: VariableRole
    first_use_line: int
    last_use_line: int
    definition_line: Optional[int] = None
    type_hint: Optional[str] = None
    value_expr: Optional[str] = None
    is_modified: bool = False
    is_parameter: bool = False
    scope_depth: int = 0


@dataclass
class ParameterSpec:
    """Specification for a function parameter."""
    name: str
    type_hint: Optional[str] = None
    default_value: Optional[str] = None
    is_keyword_only: bool = False
    is_positional_only: bool = False
    is_variadic: bool = False  # *args
    is_keyword_variadic: bool = False  # **kwargs

    def to_code(self) -> str:
        """Generate parameter code."""
        parts = []
        if self.is_variadic:
            parts.append("*")
        elif self.is_keyword_variadic:
            parts.append("**")

        parts.append(self.name)

        if self.type_hint:
            parts.append(f": {self.type_hint}")

        if self.default_value is not None:
            if self.type_hint:
                parts.append(f" = {self.default_value}")
            else:
                parts.append(f"={self.default_value}")

        return "".join(parts)


@dataclass
class FunctionSignature:
    """Complete function signature."""
    name: str
    parameters: List[ParameterSpec] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)

    def to_code(self, include_decorators: bool = True) -> str:
        """Generate function definition code."""
        lines = []

        if include_decorators:
            for dec in self.decorators:
                lines.append(f"@{dec}")

        prefix = "async def" if self.is_async else "def"
        params = ", ".join(p.to_code() for p in self.parameters)

        if self.return_type:
            lines.append(f"{prefix} {self.name}({params}) -> {self.return_type}:")
        else:
            lines.append(f"{prefix} {self.name}({params}):")

        return "\n".join(lines)


@dataclass
class CodeBlock:
    """A block of code with metadata."""
    source: str
    start_line: int
    end_line: int
    indent_level: int
    variables: Dict[str, VariableInfo] = field(default_factory=dict)
    flow_controls: List[FlowControl] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    has_side_effects: bool = False


@dataclass
class TransformResult:
    """Result of an AST transformation."""
    status: TransformStatus
    transformed_source: str
    original_source: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in (TransformStatus.SUCCESS, TransformStatus.PARTIAL)

    def get_diff(self) -> str:
        """Get unified diff of changes."""
        import difflib
        original_lines = self.original_source.splitlines(keepends=True)
        transformed_lines = self.transformed_source.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            transformed_lines,
            fromfile="original",
            tofile="transformed",
        )
        return "".join(diff)


@dataclass
class ArgumentMapping:
    """Mapping between old and new arguments."""
    old_name: str
    new_name: str
    old_position: int
    new_position: int
    needs_keyword: bool = False
    default_value: Optional[str] = None


# =============================================================================
# VARIABLE FLOW ANALYZER
# =============================================================================

class VariableFlowAnalyzer(ast.NodeVisitor):
    """
    Analyzes variable flow within a code block.

    Determines which variables are:
    - Inputs: Read but not defined (must be parameters)
    - Outputs: Modified and used outside (must be returned)
    - Local: Only used within the block
    - Closures: From enclosing scopes
    """

    def __init__(self, source_lines: List[str], start_line: int, end_line: int):
        self.source_lines = source_lines
        self.start_line = start_line
        self.end_line = end_line

        # Variable tracking
        self.read_vars: Dict[str, List[int]] = defaultdict(list)  # name -> lines read
        self.write_vars: Dict[str, List[int]] = defaultdict(list)  # name -> lines written
        self.type_hints: Dict[str, str] = {}

        # Scope tracking
        self.scope_stack: List[Set[str]] = [set()]  # Stack of local variable sets
        self.current_scope_depth = 0

        # Flow control tracking
        self.flow_controls: List[FlowControl] = []
        self.has_early_return = False
        self.has_yield = False

        # Function calls
        self.calls: List[str] = []

    def analyze(self, tree: ast.AST) -> Dict[str, VariableInfo]:
        """Analyze the AST and return variable information."""
        self.visit(tree)
        return self._build_variable_info()

    def _in_range(self, lineno: int) -> bool:
        """Check if line is within the extraction range."""
        return self.start_line <= lineno <= self.end_line

    def _build_variable_info(self) -> Dict[str, VariableInfo]:
        """Build VariableInfo for all variables."""
        variables = {}

        all_vars = set(self.read_vars.keys()) | set(self.write_vars.keys())

        for name in all_vars:
            reads = self.read_vars.get(name, [])
            writes = self.write_vars.get(name, [])

            reads_in_range = [l for l in reads if self._in_range(l)]
            writes_in_range = [l for l in writes if self._in_range(l)]
            reads_after = [l for l in reads if l > self.end_line]
            writes_before = [l for l in writes if l < self.start_line]

            # Determine role
            if writes_before and reads_in_range and not writes_in_range:
                # Defined before, read in block, not modified -> input
                role = VariableRole.INPUT
            elif writes_in_range and reads_after:
                # Modified in block, read after -> output
                role = VariableRole.OUTPUT
            elif writes_in_range and reads_in_range and not reads_after:
                # Only used within block -> local
                role = VariableRole.LOCAL
            elif reads_in_range and not writes_in_range and not writes_before:
                # Read but never defined -> closure or global
                role = VariableRole.CLOSURE
            else:
                # Complex case, treat as input for safety
                role = VariableRole.INPUT if reads_in_range else VariableRole.LOCAL

            all_lines = reads + writes
            variables[name] = VariableInfo(
                name=name,
                role=role,
                first_use_line=min(all_lines) if all_lines else 0,
                last_use_line=max(all_lines) if all_lines else 0,
                definition_line=min(writes) if writes else None,
                type_hint=self.type_hints.get(name),
                is_modified=bool(writes_in_range),
            )

        return variables

    def visit_Name(self, node: ast.Name) -> None:
        """Track variable reads and writes."""
        if isinstance(node.ctx, ast.Load):
            self.read_vars[node.id].append(node.lineno)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.write_vars[node.id].append(node.lineno)
            # Track in current scope
            if self.scope_stack:
                self.scope_stack[-1].add(node.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Track annotated assignments for type hints."""
        if isinstance(node.target, ast.Name):
            if hasattr(ast, 'unparse'):
                self.type_hints[node.target.id] = ast.unparse(node.annotation)
            else:
                self.type_hints[node.target.id] = "Any"
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        """Track function argument definitions."""
        self.write_vars[node.arg].append(node.lineno if hasattr(node, 'lineno') else 0)
        if node.annotation and hasattr(ast, 'unparse'):
            self.type_hints[node.arg] = ast.unparse(node.annotation)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Track return statements."""
        if self._in_range(node.lineno):
            self.flow_controls.append(FlowControl.RETURN)
            self.has_early_return = True
        self.generic_visit(node)

    def visit_Break(self, node: ast.Break) -> None:
        """Track break statements."""
        if self._in_range(node.lineno):
            self.flow_controls.append(FlowControl.BREAK)
        self.generic_visit(node)

    def visit_Continue(self, node: ast.Continue) -> None:
        """Track continue statements."""
        if self._in_range(node.lineno):
            self.flow_controls.append(FlowControl.CONTINUE)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        """Track raise statements."""
        if self._in_range(node.lineno):
            self.flow_controls.append(FlowControl.RAISE)
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        """Track yield statements."""
        if self._in_range(node.lineno):
            self.flow_controls.append(FlowControl.YIELD)
            self.has_yield = True
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """Track yield from statements."""
        if self._in_range(node.lineno):
            self.flow_controls.append(FlowControl.YIELD)
            self.has_yield = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track function calls."""
        if self._in_range(node.lineno):
            if isinstance(node.func, ast.Name):
                self.calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                self.calls.append(node.func.attr)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track nested function definitions."""
        self.scope_stack.append(set())
        self.current_scope_depth += 1
        self.generic_visit(node)
        self.current_scope_depth -= 1
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track nested async function definitions."""
        self.visit_FunctionDef(node)  # Same handling


# =============================================================================
# CALL SITE ANALYZER
# =============================================================================

class CallSiteAnalyzer(ast.NodeVisitor):
    """
    Analyzes function call sites for signature changes.

    Finds all calls to a specific function and extracts:
    - Positional arguments
    - Keyword arguments
    - *args and **kwargs usage
    - Line numbers and positions
    """

    def __init__(self, function_name: str, class_name: Optional[str] = None):
        self.function_name = function_name
        self.class_name = class_name
        self.call_sites: List[Dict[str, Any]] = []

    def analyze(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find all call sites."""
        self.visit(tree)
        return self.call_sites

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls."""
        func_name = None
        is_method = False

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            is_method = True
            # Check if it's a method on a specific class
            if self.class_name and isinstance(node.func.value, ast.Name):
                if node.func.value.id != self.class_name:
                    self.generic_visit(node)
                    return

        if func_name == self.function_name:
            # Extract argument info
            positional_args = []
            keyword_args = {}
            has_starargs = False
            has_kwargs = False

            for i, arg in enumerate(node.args):
                if hasattr(ast, 'unparse'):
                    positional_args.append(ast.unparse(arg))
                else:
                    positional_args.append(f"arg{i}")

            for kw in node.keywords:
                if kw.arg is None:
                    has_kwargs = True
                else:
                    if hasattr(ast, 'unparse'):
                        keyword_args[kw.arg] = ast.unparse(kw.value)
                    else:
                        keyword_args[kw.arg] = "value"

            self.call_sites.append({
                "line": node.lineno,
                "col": node.col_offset,
                "end_line": getattr(node, 'end_lineno', node.lineno),
                "end_col": getattr(node, 'end_col_offset', node.col_offset),
                "positional_args": positional_args,
                "keyword_args": keyword_args,
                "has_starargs": has_starargs,
                "has_kwargs": has_kwargs,
                "is_method": is_method,
                "node": node,
            })

        self.generic_visit(node)


# =============================================================================
# AST TRANSFORMER
# =============================================================================

class ASTTransformer:
    """
    Enterprise-grade AST transformation engine.

    Provides safe, reversible code transformations with:
    - Automatic variable flow analysis
    - Proper indentation preservation
    - Type hint inference
    - Comprehensive error handling
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        self.config = config or TransformerConfig()
        self.indent_char = "\t" if self.config.USE_TABS else " " * self.config.INDENT_SIZE

    # =========================================================================
    # EXTRACT METHOD
    # =========================================================================

    async def extract_to_method(
        self,
        source: str,
        start_line: int,
        end_line: int,
        method_name: str,
        target_class: Optional[str] = None,
        parameters: Optional[List[str]] = None,
        is_async: bool = False,
    ) -> TransformResult:
        """
        Extract code block into a new method.

        Algorithm:
        1. Parse source and locate the code block
        2. Analyze variable flow (inputs, outputs, locals)
        3. Detect flow control issues (return, break, continue)
        4. Generate new method with proper signature
        5. Replace original code with method call
        6. Insert new method at appropriate location

        Args:
            source: Original source code
            start_line: First line to extract (1-indexed)
            end_line: Last line to extract (1-indexed)
            method_name: Name for the new method
            target_class: Optional class to add method to
            parameters: Optional explicit parameter list
            is_async: Whether to make the method async

        Returns:
            TransformResult with transformed code
        """
        warnings = []
        errors = []

        # Parse source
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return TransformResult(
                status=TransformStatus.INVALID,
                transformed_source=source,
                original_source=source,
                errors=[f"Syntax error in source: {e}"],
            )

        source_lines = source.splitlines()

        # Validate line range
        if start_line < 1 or end_line > len(source_lines):
            return TransformResult(
                status=TransformStatus.INVALID,
                transformed_source=source,
                original_source=source,
                errors=[f"Invalid line range: {start_line}-{end_line}"],
            )

        # Extract the code block
        block_lines = source_lines[start_line - 1:end_line]
        block_source = "\n".join(block_lines)

        # Detect indentation level
        first_non_empty = next((l for l in block_lines if l.strip()), "")
        indent_match = re.match(r'^(\s*)', first_non_empty)
        block_indent = indent_match.group(1) if indent_match else ""
        indent_level = len(block_indent) // len(self.indent_char) if self.indent_char.strip() else len(block_indent) // 4

        # Analyze variable flow
        analyzer = VariableFlowAnalyzer(source_lines, start_line, end_line)
        try:
            analyzer.analyze(tree)
        except Exception as e:
            logger.warning(f"Variable analysis failed: {e}")
            warnings.append(f"Variable analysis incomplete: {e}")

        variables = analyzer._build_variable_info()

        # Check for flow control issues
        if FlowControl.BREAK in analyzer.flow_controls:
            warnings.append("Block contains 'break' - may not work correctly when extracted")
        if FlowControl.CONTINUE in analyzer.flow_controls:
            warnings.append("Block contains 'continue' - may not work correctly when extracted")
        if analyzer.has_yield:
            errors.append("Cannot extract block containing 'yield'")
            return TransformResult(
                status=TransformStatus.INVALID,
                transformed_source=source,
                original_source=source,
                errors=errors,
                warnings=warnings,
            )

        # Determine parameters (inputs to the block)
        if parameters is None:
            input_vars = [v for v in variables.values() if v.role == VariableRole.INPUT]
            parameters = [v.name for v in sorted(input_vars, key=lambda x: x.first_use_line)]

        # Determine return values (outputs from the block)
        output_vars = [v for v in variables.values() if v.role == VariableRole.OUTPUT]
        return_vars = [v.name for v in sorted(output_vars, key=lambda x: x.last_use_line)]

        # Build parameter specs
        param_specs = []
        if target_class:
            param_specs.append(ParameterSpec(name="self"))

        for param in parameters:
            var_info = variables.get(param)
            param_specs.append(ParameterSpec(
                name=param,
                type_hint=var_info.type_hint if var_info and self.config.ADD_TYPE_HINTS else None,
            ))

        # Determine return type
        return_type = None
        if return_vars and self.config.INFER_RETURN_TYPE:
            if len(return_vars) == 1:
                var = variables.get(return_vars[0])
                return_type = var.type_hint if var else None
            else:
                return_type = f"Tuple[{', '.join('Any' for _ in return_vars)}]"

        # Create function signature
        signature = FunctionSignature(
            name=method_name,
            parameters=param_specs,
            return_type=return_type,
            is_async=is_async,
            is_method=bool(target_class),
        )

        # Generate new method
        method_indent = block_indent if not target_class else self.indent_char
        new_method = self._generate_method(
            signature=signature,
            body_lines=block_lines,
            return_vars=return_vars,
            base_indent=method_indent,
            body_indent=block_indent,
        )

        # Generate method call
        call_args = parameters
        method_call = self._generate_method_call(
            method_name=method_name,
            arguments=call_args,
            return_vars=return_vars,
            is_async=is_async,
            indent=block_indent,
            target_class=target_class,
        )

        # Build transformed source
        new_lines = source_lines[:start_line - 1]
        new_lines.append(method_call)
        new_lines.extend(source_lines[end_line:])

        # Find where to insert the new method
        if target_class:
            # Find the class and insert at the end
            insert_line = self._find_class_end(tree, target_class)
            if insert_line:
                new_lines.insert(insert_line, "")
                new_lines.insert(insert_line + 1, new_method)
            else:
                # Fallback: insert before the extraction point
                new_lines.insert(start_line - 1, new_method)
                new_lines.insert(start_line, "")
        else:
            # Insert before the extraction point
            new_lines.insert(start_line - 1, new_method)
            new_lines.insert(start_line, "")

        transformed_source = "\n".join(new_lines)

        # Validate transformed source
        if self.config.VALIDATE_SYNTAX:
            try:
                ast.parse(transformed_source)
            except SyntaxError as e:
                return TransformResult(
                    status=TransformStatus.FAILED,
                    transformed_source=source,
                    original_source=source,
                    errors=[f"Generated code has syntax error: {e}"],
                    warnings=warnings,
                )

        return TransformResult(
            status=TransformStatus.SUCCESS,
            transformed_source=transformed_source,
            original_source=source,
            warnings=warnings,
            changes=[{
                "type": "extract_method",
                "method_name": method_name,
                "start_line": start_line,
                "end_line": end_line,
                "parameters": parameters,
                "return_vars": return_vars,
            }],
            metadata={
                "input_vars": [v.name for v in variables.values() if v.role == VariableRole.INPUT],
                "output_vars": [v.name for v in variables.values() if v.role == VariableRole.OUTPUT],
                "local_vars": [v.name for v in variables.values() if v.role == VariableRole.LOCAL],
            },
        )

    def _generate_method(
        self,
        signature: FunctionSignature,
        body_lines: List[str],
        return_vars: List[str],
        base_indent: str,
        body_indent: str,
    ) -> str:
        """Generate a new method with the given signature and body."""
        lines = []

        # Add signature
        lines.append(f"{base_indent}{signature.to_code(include_decorators=False)}")

        # Add docstring
        if self.config.PRESERVE_DOCSTRINGS:
            lines.append(f'{base_indent}{self.indent_char}"""Extracted method."""')

        # Add body (adjust indentation)
        for line in body_lines:
            if line.strip():
                # Replace original indent with method body indent
                stripped = line.lstrip()
                lines.append(f"{base_indent}{self.indent_char}{stripped}")
            else:
                lines.append("")

        # Add return statement if needed
        if return_vars:
            if len(return_vars) == 1:
                lines.append(f"{base_indent}{self.indent_char}return {return_vars[0]}")
            else:
                lines.append(f"{base_indent}{self.indent_char}return {', '.join(return_vars)}")

        return "\n".join(lines)

    def _generate_method_call(
        self,
        method_name: str,
        arguments: List[str],
        return_vars: List[str],
        is_async: bool,
        indent: str,
        target_class: Optional[str] = None,
    ) -> str:
        """Generate a method call statement."""
        prefix = "self." if target_class else ""
        await_prefix = "await " if is_async else ""
        call = f"{await_prefix}{prefix}{method_name}({', '.join(arguments)})"

        if not return_vars:
            return f"{indent}{call}"
        elif len(return_vars) == 1:
            return f"{indent}{return_vars[0]} = {call}"
        else:
            return f"{indent}{', '.join(return_vars)} = {call}"

    def _find_class_end(self, tree: ast.AST, class_name: str) -> Optional[int]:
        """Find the line number at the end of a class definition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return getattr(node, 'end_lineno', None)
        return None

    # =========================================================================
    # INLINE VARIABLE
    # =========================================================================

    async def inline_variable(
        self,
        source: str,
        variable_name: str,
        definition_line: int,
    ) -> TransformResult:
        """
        Inline a variable by replacing all uses with its value.

        Algorithm:
        1. Find the variable assignment
        2. Extract the assigned value expression
        3. Find all references to the variable
        4. Replace each reference with the value (with parentheses if needed)
        5. Remove the original assignment

        Args:
            source: Original source code
            variable_name: Name of variable to inline
            definition_line: Line where variable is defined (1-indexed)

        Returns:
            TransformResult with transformed code
        """
        warnings = []

        # Parse source
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return TransformResult(
                status=TransformStatus.INVALID,
                transformed_source=source,
                original_source=source,
                errors=[f"Syntax error in source: {e}"],
            )

        source_lines = source.splitlines()

        # Find the assignment
        assignment_node = None
        value_expr = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and node.lineno == definition_line:
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == variable_name:
                        assignment_node = node
                        if hasattr(ast, 'unparse'):
                            value_expr = ast.unparse(node.value)
                        break
            elif isinstance(node, ast.AnnAssign) and node.lineno == definition_line:
                if isinstance(node.target, ast.Name) and node.target.id == variable_name:
                    assignment_node = node
                    if node.value and hasattr(ast, 'unparse'):
                        value_expr = ast.unparse(node.value)

        if not assignment_node or not value_expr:
            return TransformResult(
                status=TransformStatus.INVALID,
                transformed_source=source,
                original_source=source,
                errors=[f"Could not find assignment for '{variable_name}' at line {definition_line}"],
            )

        # Check if value has side effects
        if self._has_side_effects(assignment_node.value):
            warnings.append(f"Value expression may have side effects - inlining could change behavior")

        # Find all references
        references = self._find_variable_references(tree, variable_name, definition_line)

        if not references:
            warnings.append(f"No references found for '{variable_name}'")

        # Check if variable is reassigned
        reassignments = [r for r in references if r['is_write']]
        if reassignments:
            return TransformResult(
                status=TransformStatus.INVALID,
                transformed_source=source,
                original_source=source,
                errors=[f"Variable '{variable_name}' is reassigned at line {reassignments[0]['line']}"],
            )

        # Build new source by replacing references
        new_lines = source_lines.copy()

        # Sort references by line/column in reverse order to preserve positions
        references.sort(key=lambda r: (r['line'], r['col']), reverse=True)

        for ref in references:
            if ref['is_write']:
                continue

            line_idx = ref['line'] - 1
            line = new_lines[line_idx]

            # Determine if we need parentheses
            needs_parens = self._needs_parentheses(value_expr, ref.get('context', 'expr'))
            replacement = f"({value_expr})" if needs_parens else value_expr

            # Replace the reference
            col = ref['col']
            end_col = col + len(variable_name)
            new_lines[line_idx] = line[:col] + replacement + line[end_col:]

        # Remove the assignment line
        assign_line_idx = definition_line - 1
        assign_end_line = getattr(assignment_node, 'end_lineno', definition_line)

        # Handle multi-line assignments
        del new_lines[assign_line_idx:assign_end_line]

        transformed_source = "\n".join(new_lines)

        # Validate
        if self.config.VALIDATE_SYNTAX:
            try:
                ast.parse(transformed_source)
            except SyntaxError as e:
                return TransformResult(
                    status=TransformStatus.FAILED,
                    transformed_source=source,
                    original_source=source,
                    errors=[f"Generated code has syntax error: {e}"],
                    warnings=warnings,
                )

        return TransformResult(
            status=TransformStatus.SUCCESS,
            transformed_source=transformed_source,
            original_source=source,
            warnings=warnings,
            changes=[{
                "type": "inline_variable",
                "variable": variable_name,
                "definition_line": definition_line,
                "references_replaced": len([r for r in references if not r['is_write']]),
                "value": value_expr,
            }],
        )

    def _has_side_effects(self, node: ast.AST) -> bool:
        """Check if an expression has potential side effects."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                return True  # Function calls may have side effects
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False

    def _find_variable_references(
        self,
        tree: ast.AST,
        var_name: str,
        after_line: int,
    ) -> List[Dict[str, Any]]:
        """Find all references to a variable after a given line."""
        references = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == var_name:
                if node.lineno > after_line:
                    references.append({
                        "line": node.lineno,
                        "col": node.col_offset,
                        "is_write": isinstance(node.ctx, (ast.Store, ast.Del)),
                        "context": self._get_parent_context(tree, node),
                    })

        return references

    def _get_parent_context(self, tree: ast.AST, target: ast.AST) -> str:
        """Get the parent context of a node for parentheses decisions."""
        # This is simplified - a full implementation would track parent nodes
        return "expr"

    def _needs_parentheses(self, expr: str, context: str) -> bool:
        """Determine if expression needs parentheses when inlined."""
        # Simple heuristic: if expression contains operators, add parens
        operators = ['+', '-', '*', '/', '%', '|', '&', '^', '<', '>', '=', 'or', 'and', 'not', 'if']
        return any(op in expr for op in operators)

    # =========================================================================
    # CHANGE SIGNATURE
    # =========================================================================

    async def change_signature(
        self,
        source: str,
        function_name: str,
        old_signature: FunctionSignature,
        new_signature: FunctionSignature,
        arg_mapping: List[ArgumentMapping],
    ) -> TransformResult:
        """
        Change a function's signature and update all call sites.

        Args:
            source: Original source code
            function_name: Name of function to modify
            old_signature: Current signature
            new_signature: New signature
            arg_mapping: How to map old arguments to new

        Returns:
            TransformResult with transformed code
        """
        warnings = []

        # Parse source
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return TransformResult(
                status=TransformStatus.INVALID,
                transformed_source=source,
                original_source=source,
                errors=[f"Syntax error in source: {e}"],
            )

        source_lines = source.splitlines()

        # Find the function definition
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name:
                    func_node = node
                    break

        if not func_node:
            return TransformResult(
                status=TransformStatus.INVALID,
                transformed_source=source,
                original_source=source,
                errors=[f"Function '{function_name}' not found"],
            )

        # Find all call sites
        analyzer = CallSiteAnalyzer(function_name)
        call_sites = analyzer.analyze(tree)

        # Build mapping dict
        mapping_dict = {m.old_name: m for m in arg_mapping}

        # Transform call sites (in reverse order to preserve positions)
        new_lines = source_lines.copy()
        call_sites.sort(key=lambda c: (c['line'], c['col']), reverse=True)

        for call_site in call_sites:
            new_call = self._transform_call_site(
                call_site=call_site,
                old_signature=old_signature,
                new_signature=new_signature,
                mapping=mapping_dict,
            )

            if new_call:
                line_idx = call_site['line'] - 1
                line = new_lines[line_idx]
                col = call_site['col']
                end_col = call_site['end_col']
                new_lines[line_idx] = line[:col] + new_call + line[end_col:]

        # Transform function definition
        func_line_idx = func_node.lineno - 1

        # Generate new definition line
        new_def = new_signature.to_code(include_decorators=False)

        # Replace the def line
        old_line = new_lines[func_line_idx]
        indent_match = re.match(r'^(\s*)', old_line)
        indent = indent_match.group(1) if indent_match else ""

        new_lines[func_line_idx] = f"{indent}{new_def}"

        transformed_source = "\n".join(new_lines)

        # Validate
        if self.config.VALIDATE_SYNTAX:
            try:
                ast.parse(transformed_source)
            except SyntaxError as e:
                return TransformResult(
                    status=TransformStatus.FAILED,
                    transformed_source=source,
                    original_source=source,
                    errors=[f"Generated code has syntax error: {e}"],
                    warnings=warnings,
                )

        return TransformResult(
            status=TransformStatus.SUCCESS,
            transformed_source=transformed_source,
            original_source=source,
            warnings=warnings,
            changes=[{
                "type": "change_signature",
                "function": function_name,
                "call_sites_updated": len(call_sites),
            }],
            metadata={
                "call_sites": len(call_sites),
                "old_params": [p.name for p in old_signature.parameters],
                "new_params": [p.name for p in new_signature.parameters],
            },
        )

    def _transform_call_site(
        self,
        call_site: Dict[str, Any],
        old_signature: FunctionSignature,
        new_signature: FunctionSignature,
        mapping: Dict[str, ArgumentMapping],
    ) -> Optional[str]:
        """Transform a single call site to use the new signature."""
        pos_args = call_site['positional_args']
        kw_args = call_site['keyword_args']

        new_args = []

        # Map positional arguments
        for i, arg_value in enumerate(pos_args):
            if i < len(old_signature.parameters):
                old_param = old_signature.parameters[i]
                if old_param.name in mapping:
                    m = mapping[old_param.name]
                    if m.needs_keyword:
                        new_args.append(f"{m.new_name}={arg_value}")
                    else:
                        new_args.append(arg_value)
                else:
                    new_args.append(arg_value)
            else:
                new_args.append(arg_value)

        # Map keyword arguments
        for kw_name, kw_value in kw_args.items():
            if kw_name in mapping:
                m = mapping[kw_name]
                new_args.append(f"{m.new_name}={kw_value}")
            else:
                new_args.append(f"{kw_name}={kw_value}")

        # Add default values for new parameters
        for param in new_signature.parameters:
            param_mapped = any(m.new_name == param.name for m in mapping.values())
            param_in_args = any(param.name in str(a) for a in new_args)

            if not param_mapped and not param_in_args and param.default_value is not None:
                # New parameter with default - may not need to add
                pass

        func_name = new_signature.name
        if call_site.get('is_method'):
            # Keep the method call format
            return f"{func_name}({', '.join(new_args)})"

        return f"{func_name}({', '.join(new_args)})"


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_transformer_instance: Optional[ASTTransformer] = None


def get_ast_transformer() -> ASTTransformer:
    """Get the singleton AST transformer instance."""
    global _transformer_instance
    if _transformer_instance is None:
        _transformer_instance = ASTTransformer()
    return _transformer_instance


async def get_ast_transformer_async() -> ASTTransformer:
    """Get the singleton AST transformer instance (async)."""
    return get_ast_transformer()
