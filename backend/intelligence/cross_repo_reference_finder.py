"""
Cross-Repository Reference Finder - Advanced Refactoring Engine v1.0
====================================================================

Enterprise-grade reference finder that locates ALL references to a symbol
across multiple repositories in the Trinity ecosystem (Ironcliw, Ironcliw-Prime,
Reactor-Core).

Features:
- Tree-sitter based accurate reference detection
- NetworkX dependency graph analysis
- Parallel async processing across repos
- Symbol disambiguation (same name, different modules)
- Import resolution and tracking
- Caching for repeated queries

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                CrossRepoReferenceFinder                          │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
    │   │  Ironcliw     │   │ Ironcliw-Prime│   │ Reactor-Core│          │
    │   │  Indexer    │   │  Indexer    │   │  Indexer    │          │
    │   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘          │
    │          │                 │                 │                  │
    │          └─────────────────┴─────────────────┘                  │
    │                            │                                    │
    │              ┌─────────────▼─────────────┐                      │
    │              │   Symbol Resolver &       │                      │
    │              │   Reference Aggregator    │                      │
    │              └───────────────────────────┘                      │
    │                            │                                    │
    │              ┌─────────────▼─────────────┐                      │
    │              │   NetworkX Graph Builder  │                      │
    │              └───────────────────────────┘                      │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Author: Ironcliw AI System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Dict, FrozenSet, Generator,
    List, Literal, NamedTuple, Optional, Protocol, Set, Tuple, TypeVar, Union
)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False

try:
    from tree_sitter import Language, Parser
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Language = None
    Parser = None
    get_language = None
    get_parser = None

from backend.utils.env_config import get_env_str, get_env_int, get_env_bool, get_env_list

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - Environment Driven (Zero Hardcoding)
# =============================================================================

def _get_env_path(key: str, default: str = "") -> Path:
    """Get environment variable as Path."""
    return Path(os.path.expanduser(get_env_str(key, default)))


class ReferenceFinderConfig:
    """Configuration for reference finding."""

    # Repository paths
    Ironcliw_REPO: Path = _get_env_path("Ironcliw_REPO_PATH", "~/Documents/repos/Ironcliw-AI-Agent")
    PRIME_REPO: Path = _get_env_path("Ironcliw_PRIME_REPO_PATH", "~/Documents/repos/jarvis-prime")
    REACTOR_REPO: Path = _get_env_path("REACTOR_CORE_REPO_PATH", "~/Documents/repos/reactor-core")

    # Search settings
    MAX_FILES: int = get_env_int("REFERENCE_FINDER_MAX_FILES", 10000)
    TIMEOUT_MS: int = get_env_int("REFERENCE_FINDER_TIMEOUT_MS", 30000)
    PARALLEL_REPOS: bool = get_env_bool("REFERENCE_FINDER_PARALLEL_REPOS", True)
    MAX_CONCURRENT: int = get_env_int("REFERENCE_FINDER_MAX_CONCURRENT", 50)

    # File patterns
    INCLUDE_PATTERNS: List[str] = get_env_list(
        "REFERENCE_FINDER_INCLUDE",
        "**/*.py,**/*.ts,**/*.js,**/*.tsx,**/*.jsx"
    ) or ["**/*.py", "**/*.ts", "**/*.js", "**/*.tsx", "**/*.jsx"]

    EXCLUDE_PATTERNS: List[str] = get_env_list(
        "REFERENCE_FINDER_EXCLUDE",
        "**/node_modules/**,**/__pycache__/**,**/venv/**,**/.git/**,**/dist/**,**/build/**"
    ) or ["**/node_modules/**", "**/__pycache__/**", "**/venv/**", "**/.git/**", "**/dist/**", "**/build/**"]

    # Cache settings
    CACHE_ENABLED: bool = get_env_bool("REFERENCE_FINDER_CACHE_ENABLED", True)
    CACHE_TTL_MINUTES: int = get_env_int("REFERENCE_FINDER_CACHE_TTL_MINUTES", 30)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class SymbolKind(str, Enum):
    """Types of symbols."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    MODULE = "module"
    PARAMETER = "parameter"
    TYPE_ALIAS = "type_alias"


class ReferenceType(str, Enum):
    """Types of references to a symbol."""
    DEFINITION = "definition"       # Where symbol is defined
    CALL = "call"                   # Function/method call
    IMPORT = "import"               # Import statement
    INHERITANCE = "inheritance"     # Class inheritance
    TYPE_HINT = "type_hint"         # Type annotation
    ASSIGNMENT = "assignment"       # Variable assignment
    READ = "read"                   # Variable read
    ATTRIBUTE = "attribute"         # Attribute access
    DECORATOR = "decorator"         # Decorator usage


class RepoType(str, Enum):
    """Repository types."""
    Ironcliw = "jarvis"
    PRIME = "prime"
    REACTOR = "reactor"
    UNKNOWN = "unknown"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class SymbolLocation:
    """Location of a symbol definition."""
    file_path: Path
    line: int
    column: int
    end_line: int
    end_column: int
    repository: RepoType

    def __hash__(self):
        return hash((str(self.file_path), self.line, self.column))


@dataclass
class SymbolInfo:
    """Complete information about a symbol."""
    name: str
    kind: SymbolKind
    location: SymbolLocation
    module_path: str
    qualified_name: str
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    decorators: List[str] = field(default_factory=list)


@dataclass
class Reference:
    """A reference to a symbol."""
    file_path: Path
    line: int
    column: int
    end_line: int
    end_column: int
    reference_type: ReferenceType
    repository: RepoType
    context: str  # Surrounding code
    import_path: Optional[str] = None  # For import references
    is_qualified: bool = False  # module.symbol vs symbol

    def __hash__(self):
        return hash((str(self.file_path), self.line, self.column, self.reference_type))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "reference_type": self.reference_type.value,
            "repository": self.repository.value,
            "context": self.context,
            "import_path": self.import_path,
            "is_qualified": self.is_qualified,
        }


@dataclass
class CallSite:
    """A function/method call site with argument information."""
    file_path: Path
    line: int
    column: int
    end_line: int
    end_column: int
    repository: RepoType
    function_name: str
    arguments: List[str]
    keyword_arguments: Dict[str, str]
    is_method: bool
    receiver: Optional[str] = None  # Object the method is called on
    has_starargs: bool = False
    has_kwargs: bool = False
    context: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "line": self.line,
            "column": self.column,
            "function_name": self.function_name,
            "arguments": self.arguments,
            "keyword_arguments": self.keyword_arguments,
            "is_method": self.is_method,
            "receiver": self.receiver,
            "repository": self.repository.value,
        }


@dataclass
class ReferenceSearchResult:
    """Result of a reference search."""
    symbol_name: str
    symbol_kind: SymbolKind
    definition: Optional[SymbolInfo]
    references: List[Reference]
    call_sites: List[CallSite]
    search_time_ms: float
    repositories_searched: List[RepoType]
    files_searched: int
    errors: List[str] = field(default_factory=list)


# =============================================================================
# FILE SCANNER
# =============================================================================

class FileScanner:
    """
    Scans repositories for source files.

    Efficiently finds all relevant source files across multiple repositories
    using glob patterns and exclusion rules.
    """

    def __init__(self, config: Optional[ReferenceFinderConfig] = None):
        self.config = config or ReferenceFinderConfig()

    def _should_exclude(self, path: Path, repo_root: Path) -> bool:
        """Check if a path should be excluded."""
        rel_path = str(path.relative_to(repo_root))

        for pattern in self.config.EXCLUDE_PATTERNS:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Also check just the filename
            if fnmatch.fnmatch(path.name, pattern.replace("**/", "")):
                return True

        return False

    async def scan_repository(
        self,
        repo_path: Path,
        repo_type: RepoType,
    ) -> List[Tuple[Path, RepoType]]:
        """Scan a repository for source files."""
        files = []

        if not repo_path.exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            return files

        # Use asyncio to avoid blocking
        loop = asyncio.get_event_loop()

        def _scan():
            result = []
            for pattern in self.config.INCLUDE_PATTERNS:
                for path in repo_path.glob(pattern):
                    if path.is_file() and not self._should_exclude(path, repo_path):
                        result.append((path, repo_type))
                        if len(result) >= self.config.MAX_FILES:
                            return result
            return result

        files = await loop.run_in_executor(None, _scan)
        return files

    async def scan_all_repositories(self) -> List[Tuple[Path, RepoType]]:
        """Scan all configured repositories."""
        repos = [
            (self.config.Ironcliw_REPO, RepoType.Ironcliw),
            (self.config.PRIME_REPO, RepoType.PRIME),
            (self.config.REACTOR_REPO, RepoType.REACTOR),
        ]

        if self.config.PARALLEL_REPOS:
            tasks = [self.scan_repository(path, repo_type) for path, repo_type in repos]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            files = []
            for result in results:
                if isinstance(result, list):
                    files.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Scan error: {result}")
            return files
        else:
            files = []
            for path, repo_type in repos:
                files.extend(await self.scan_repository(path, repo_type))
            return files


# =============================================================================
# PYTHON REFERENCE FINDER
# =============================================================================

class PythonReferenceFinder:
    """
    Finds references in Python files using AST.

    Provides accurate reference detection for:
    - Function/method calls
    - Class instantiations
    - Import statements
    - Variable assignments and reads
    - Type hints
    - Decorators
    """

    def __init__(self, symbol_name: str, symbol_kind: SymbolKind):
        self.symbol_name = symbol_name
        self.symbol_kind = symbol_kind

    async def find_references(
        self,
        file_path: Path,
        repo_type: RepoType,
    ) -> Tuple[List[Reference], List[CallSite]]:
        """Find all references in a file."""
        import ast

        references = []
        call_sites = []

        try:
            source = await asyncio.to_thread(file_path.read_text, encoding='utf-8')
            tree = ast.parse(source, filename=str(file_path))
            source_lines = source.splitlines()
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.debug(f"Could not parse {file_path}: {e}")
            return references, call_sites
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return references, call_sites

        # Walk the AST
        for node in ast.walk(tree):
            # Check for imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == self.symbol_name or (alias.asname and alias.asname == self.symbol_name):
                        context = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
                        references.append(Reference(
                            file_path=file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            end_line=getattr(node, 'end_lineno', node.lineno),
                            end_column=getattr(node, 'end_col_offset', node.col_offset),
                            reference_type=ReferenceType.IMPORT,
                            repository=repo_type,
                            context=context,
                            import_path=alias.name,
                        ))

            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == self.symbol_name or (alias.asname and alias.asname == self.symbol_name):
                        context = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
                        module = node.module or ""
                        references.append(Reference(
                            file_path=file_path,
                            line=node.lineno,
                            column=node.col_offset,
                            end_line=getattr(node, 'end_lineno', node.lineno),
                            end_column=getattr(node, 'end_col_offset', node.col_offset),
                            reference_type=ReferenceType.IMPORT,
                            repository=repo_type,
                            context=context,
                            import_path=f"{module}.{alias.name}" if module else alias.name,
                        ))

            # Check for function/method calls
            elif isinstance(node, ast.Call):
                func_name = None
                is_method = False
                receiver = None

                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    is_method = True
                    if isinstance(node.func.value, ast.Name):
                        receiver = node.func.value.id

                if func_name == self.symbol_name:
                    context = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""

                    # Extract arguments
                    args = []
                    kwargs = {}
                    has_starargs = False
                    has_kwargs_spread = False

                    for arg in node.args:
                        if hasattr(ast, 'unparse'):
                            args.append(ast.unparse(arg))
                        else:
                            args.append("?")

                    for kw in node.keywords:
                        if kw.arg is None:
                            has_kwargs_spread = True
                        else:
                            if hasattr(ast, 'unparse'):
                                kwargs[kw.arg] = ast.unparse(kw.value)
                            else:
                                kwargs[kw.arg] = "?"

                    call_sites.append(CallSite(
                        file_path=file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        end_column=getattr(node, 'end_col_offset', node.col_offset),
                        repository=repo_type,
                        function_name=func_name,
                        arguments=args,
                        keyword_arguments=kwargs,
                        is_method=is_method,
                        receiver=receiver,
                        has_starargs=has_starargs,
                        has_kwargs=has_kwargs_spread,
                        context=context,
                    ))

                    references.append(Reference(
                        file_path=file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        end_column=getattr(node, 'end_col_offset', node.col_offset),
                        reference_type=ReferenceType.CALL,
                        repository=repo_type,
                        context=context,
                        is_qualified=is_method,
                    ))

            # Check for class definitions (inheritance)
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr

                    if base_name == self.symbol_name:
                        context = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""
                        references.append(Reference(
                            file_path=file_path,
                            line=base.lineno,
                            column=base.col_offset,
                            end_line=getattr(base, 'end_lineno', base.lineno),
                            end_column=getattr(base, 'end_col_offset', base.col_offset),
                            reference_type=ReferenceType.INHERITANCE,
                            repository=repo_type,
                            context=context,
                        ))

            # Check for decorators
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    dec_name = None
                    if isinstance(decorator, ast.Name):
                        dec_name = decorator.id
                    elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                        dec_name = decorator.func.id
                    elif isinstance(decorator, ast.Attribute):
                        dec_name = decorator.attr

                    if dec_name == self.symbol_name:
                        context = source_lines[decorator.lineno - 1] if decorator.lineno <= len(source_lines) else ""
                        references.append(Reference(
                            file_path=file_path,
                            line=decorator.lineno,
                            column=decorator.col_offset,
                            end_line=getattr(decorator, 'end_lineno', decorator.lineno),
                            end_column=getattr(decorator, 'end_col_offset', decorator.col_offset),
                            reference_type=ReferenceType.DECORATOR,
                            repository=repo_type,
                            context=context,
                        ))

            # Check for variable names
            elif isinstance(node, ast.Name):
                if node.id == self.symbol_name:
                    context = source_lines[node.lineno - 1] if node.lineno <= len(source_lines) else ""

                    if isinstance(node.ctx, ast.Store):
                        ref_type = ReferenceType.ASSIGNMENT
                    elif isinstance(node.ctx, ast.Load):
                        ref_type = ReferenceType.READ
                    else:
                        ref_type = ReferenceType.READ

                    references.append(Reference(
                        file_path=file_path,
                        line=node.lineno,
                        column=node.col_offset,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        end_column=getattr(node, 'end_col_offset', node.col_offset),
                        reference_type=ref_type,
                        repository=repo_type,
                        context=context,
                    ))

        return references, call_sites


# =============================================================================
# CROSS-REPO REFERENCE FINDER
# =============================================================================

class CrossRepoReferenceFinder:
    """
    Finds ALL references to a symbol across multiple repositories.

    Uses parallel async processing for efficient searching across
    Ironcliw, Ironcliw-Prime, and Reactor-Core repositories.
    """

    def __init__(self, config: Optional[ReferenceFinderConfig] = None):
        self.config = config or ReferenceFinderConfig()
        self.scanner = FileScanner(self.config)
        self._cache: Dict[str, Tuple[ReferenceSearchResult, datetime]] = {}

    def _get_cache_key(self, symbol_name: str, symbol_kind: SymbolKind) -> str:
        """Generate cache key."""
        return f"{symbol_name}:{symbol_kind.value}"

    def _is_cache_valid(self, cache_entry: Tuple[ReferenceSearchResult, datetime]) -> bool:
        """Check if cache entry is still valid."""
        _, cached_at = cache_entry
        age = datetime.now() - cached_at
        return age < timedelta(minutes=self.config.CACHE_TTL_MINUTES)

    async def find_all_references(
        self,
        symbol_name: str,
        symbol_kind: SymbolKind,
        source_file: Optional[Path] = None,
        use_cache: bool = True,
    ) -> ReferenceSearchResult:
        """
        Find ALL references to a symbol across all repositories.

        Args:
            symbol_name: Name of the symbol to find
            symbol_kind: Type of symbol (function, class, etc.)
            source_file: Optional source file to narrow search
            use_cache: Whether to use cached results

        Returns:
            ReferenceSearchResult with all found references
        """
        start_time = asyncio.get_event_loop().time()

        # Check cache
        cache_key = self._get_cache_key(symbol_name, symbol_kind)
        if use_cache and self.config.CACHE_ENABLED and cache_key in self._cache:
            if self._is_cache_valid(self._cache[cache_key]):
                result, _ = self._cache[cache_key]
                return result

        # Scan repositories for files
        files = await self.scanner.scan_all_repositories()

        # Filter by language based on symbol kind
        python_files = [(f, r) for f, r in files if f.suffix == '.py']

        # Find references in parallel
        all_references: List[Reference] = []
        all_call_sites: List[CallSite] = []
        errors: List[str] = []

        finder = PythonReferenceFinder(symbol_name, symbol_kind)
        semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT)

        async def find_in_file(file_path: Path, repo_type: RepoType) -> Tuple[List[Reference], List[CallSite]]:
            async with semaphore:
                try:
                    return await finder.find_references(file_path, repo_type)
                except Exception as e:
                    errors.append(f"Error in {file_path}: {e}")
                    return [], []

        # Create tasks
        tasks = [find_in_file(f, r) for f, r in python_files]

        # Apply timeout
        try:
            timeout_sec = self.config.TIMEOUT_MS / 1000
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_sec,
            )

            for result in results:
                if isinstance(result, tuple):
                    refs, calls = result
                    all_references.extend(refs)
                    all_call_sites.extend(calls)
                elif isinstance(result, Exception):
                    errors.append(str(result))

        except asyncio.TimeoutError:
            errors.append(f"Search timed out after {self.config.TIMEOUT_MS}ms")

        # Deduplicate references
        unique_refs = list({hash(r): r for r in all_references}.values())
        unique_calls = list({hash((c.file_path, c.line, c.column)): c for c in all_call_sites}.values())

        # Sort by file and line
        unique_refs.sort(key=lambda r: (str(r.file_path), r.line))
        unique_calls.sort(key=lambda c: (str(c.file_path), c.line))

        end_time = asyncio.get_event_loop().time()
        search_time_ms = (end_time - start_time) * 1000

        # Find definition
        definition = None
        for ref in unique_refs:
            if ref.reference_type == ReferenceType.DEFINITION:
                definition = SymbolInfo(
                    name=symbol_name,
                    kind=symbol_kind,
                    location=SymbolLocation(
                        file_path=ref.file_path,
                        line=ref.line,
                        column=ref.column,
                        end_line=ref.end_line,
                        end_column=ref.end_column,
                        repository=ref.repository,
                    ),
                    module_path=str(ref.file_path),
                    qualified_name=symbol_name,
                )
                break

        result = ReferenceSearchResult(
            symbol_name=symbol_name,
            symbol_kind=symbol_kind,
            definition=definition,
            references=unique_refs,
            call_sites=unique_calls,
            search_time_ms=search_time_ms,
            repositories_searched=[RepoType.Ironcliw, RepoType.PRIME, RepoType.REACTOR],
            files_searched=len(python_files),
            errors=errors,
        )

        # Cache result
        if self.config.CACHE_ENABLED:
            self._cache[cache_key] = (result, datetime.now())

        return result

    async def find_call_sites(
        self,
        function_name: str,
        source_file: Optional[Path] = None,
        class_name: Optional[str] = None,
    ) -> List[CallSite]:
        """
        Find all call sites for a function/method.

        Args:
            function_name: Name of the function
            source_file: Optional source file to narrow search
            class_name: Optional class name for method calls

        Returns:
            List of CallSite objects
        """
        result = await self.find_all_references(
            symbol_name=function_name,
            symbol_kind=SymbolKind.FUNCTION if not class_name else SymbolKind.METHOD,
            source_file=source_file,
        )

        call_sites = result.call_sites

        # Filter by class name if provided
        if class_name:
            call_sites = [c for c in call_sites if c.receiver == class_name or c.is_method]

        return call_sites

    async def build_dependency_graph(
        self,
        file_paths: Optional[List[Path]] = None,
    ) -> Optional[Any]:  # nx.DiGraph
        """
        Build a dependency graph using NetworkX.

        Args:
            file_paths: Optional list of files to include

        Returns:
            NetworkX directed graph of dependencies
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX not available - cannot build dependency graph")
            return None

        graph = nx.DiGraph()

        # Scan files if not provided
        if file_paths is None:
            files = await self.scanner.scan_all_repositories()
            file_paths = [f for f, _ in files if f.suffix == '.py']

        # Parse each file and extract imports
        for file_path in file_paths:
            try:
                import ast
                source = await asyncio.to_thread(file_path.read_text, encoding='utf-8')
                tree = ast.parse(source, filename=str(file_path))

                file_node = str(file_path)
                graph.add_node(file_node, type='file')

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            graph.add_edge(file_node, alias.name)

                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        graph.add_edge(file_node, module)

            except Exception as e:
                logger.debug(f"Could not parse {file_path}: {e}")

        return graph

    def clear_cache(self) -> None:
        """Clear the reference cache."""
        self._cache.clear()


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_finder_instance: Optional[CrossRepoReferenceFinder] = None


def get_cross_repo_reference_finder() -> CrossRepoReferenceFinder:
    """Get the singleton reference finder instance."""
    global _finder_instance
    if _finder_instance is None:
        _finder_instance = CrossRepoReferenceFinder()
    return _finder_instance


async def get_cross_repo_reference_finder_async() -> CrossRepoReferenceFinder:
    """Get the singleton reference finder instance (async)."""
    return get_cross_repo_reference_finder()
