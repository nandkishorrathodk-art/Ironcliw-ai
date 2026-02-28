"""
v77.1: Cross-Repo Dependency Tracker - Gap #43
===============================================

Tracks dependencies across repos to prevent breakage.

Problem:
    - Ironcliw imports from jarvis-prime/core/models.py
    - Evolution moves models.py → models_v2.py
    - Must update Ironcliw imports atomically

Solution:
    - Parse Python AST to find imports
    - Build cross-repo dependency graph
    - Validate changes don't break imports
    - Auto-update imports when files move

Features:
    - AST-based import analysis
    - Cross-repo dependency graph
    - Break detection before changes
    - Automatic import updates
    - Cycle detection

Author: Ironcliw v77.1
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies."""
    IMPORT = "import"  # import module
    FROM_IMPORT = "from_import"  # from module import X
    RELATIVE_IMPORT = "relative_import"  # from . import X
    DYNAMIC_IMPORT = "dynamic_import"  # __import__(), importlib
    TYPE_HINT = "type_hint"  # TYPE_CHECKING imports
    RUNTIME = "runtime"  # Used at runtime only


@dataclass(frozen=True)
class Dependency:
    """
    A dependency between two code locations.

    Immutable for use in sets and as dict keys.
    """
    source_repo: str
    source_file: str
    source_line: int
    target_repo: str
    target_module: str
    imported_names: FrozenSet[str]
    dependency_type: DependencyType
    is_optional: bool = False  # In try/except

    def __hash__(self):
        return hash((
            self.source_repo,
            self.source_file,
            self.target_repo,
            self.target_module,
            self.imported_names,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_repo": self.source_repo,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "target_repo": self.target_repo,
            "target_module": self.target_module,
            "imported_names": list(self.imported_names),
            "dependency_type": self.dependency_type.value,
            "is_optional": self.is_optional,
        }


@dataclass
class DependencyNode:
    """A node in the dependency graph."""
    repo: str
    module: str
    file_path: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)  # Modules this depends on
    dependents: Set[str] = field(default_factory=set)  # Modules that depend on this


class DependencyGraph:
    """
    Graph of dependencies across repos.

    Supports:
    - Adding/removing dependencies
    - Finding affected modules
    - Cycle detection
    - Impact analysis
    """

    def __init__(self):
        self._nodes: Dict[str, DependencyNode] = {}
        self._edges: Set[Dependency] = set()

    def add_dependency(self, dep: Dependency) -> None:
        """Add a dependency to the graph."""
        source_key = f"{dep.source_repo}:{dep.source_file}"
        target_key = f"{dep.target_repo}:{dep.target_module}"

        # Ensure nodes exist
        if source_key not in self._nodes:
            self._nodes[source_key] = DependencyNode(
                repo=dep.source_repo,
                module=dep.source_file,
                file_path=dep.source_file,
            )
        if target_key not in self._nodes:
            self._nodes[target_key] = DependencyNode(
                repo=dep.target_repo,
                module=dep.target_module,
            )

        # Add edges
        self._nodes[source_key].dependencies.add(target_key)
        self._nodes[target_key].dependents.add(source_key)
        self._edges.add(dep)

    def get_dependents(self, repo: str, module: str) -> Set[str]:
        """Get all modules that depend on the given module."""
        key = f"{repo}:{module}"
        node = self._nodes.get(key)
        return node.dependents.copy() if node else set()

    def get_dependencies(self, repo: str, module: str) -> Set[str]:
        """Get all modules the given module depends on."""
        key = f"{repo}:{module}"
        node = self._nodes.get(key)
        return node.dependencies.copy() if node else set()

    def get_transitive_dependents(self, repo: str, module: str) -> Set[str]:
        """Get all modules transitively affected by changes to this module."""
        affected = set()
        to_visit = {f"{repo}:{module}"}

        while to_visit:
            current = to_visit.pop()
            if current in affected:
                continue

            affected.add(current)
            node = self._nodes.get(current)
            if node:
                to_visit.update(node.dependents - affected)

        affected.discard(f"{repo}:{module}")  # Remove the original module
        return affected

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph."""
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(node: str, path: List[str]) -> None:
            if node in path:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)

            node_obj = self._nodes.get(node)
            if node_obj:
                for dep in node_obj.dependencies:
                    dfs(dep, path.copy())

        for node in self._nodes:
            if node not in visited:
                dfs(node, [])

        return cycles

    def get_cross_repo_edges(self) -> List[Dependency]:
        """Get all dependencies that cross repo boundaries."""
        return [
            dep for dep in self._edges
            if dep.source_repo != dep.target_repo
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        cross_repo = self.get_cross_repo_edges()
        cycles = self.detect_cycles()

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "cross_repo_edges": len(cross_repo),
            "cycle_count": len(cycles),
            "repos": list(set(dep.source_repo for dep in self._edges)),
        }


class ImportAnalyzer:
    """
    Analyzes Python files for imports.

    Uses AST parsing for accurate import detection.
    """

    def __init__(self, repo_name: str, repo_path: Path):
        self.repo_name = repo_name
        self.repo_path = repo_path
        self._known_repos: Dict[str, str] = {}  # module prefix -> repo name

    def register_repo_mapping(self, module_prefix: str, repo_name: str) -> None:
        """Register a module prefix -> repo mapping."""
        self._known_repos[module_prefix] = repo_name

    async def analyze_file(self, file_path: Path) -> List[Dependency]:
        """Analyze a Python file for imports."""
        if not file_path.suffix == ".py":
            return []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.debug(f"[ImportAnalyzer] Could not parse {file_path}: {e}")
            return []

        relative_path = str(file_path.relative_to(self.repo_path))
        dependencies = []

        # Track if we're inside TYPE_CHECKING block
        in_type_checking = False

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for TYPE_CHECKING block
                if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                    in_type_checking = True
                elif isinstance(node.test, ast.Attribute):
                    if getattr(node.test, "attr", "") == "TYPE_CHECKING":
                        in_type_checking = True

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    dep = self._create_dependency(
                        source_file=relative_path,
                        module=alias.name,
                        imported_names=frozenset([alias.asname or alias.name]),
                        line=node.lineno,
                        dep_type=DependencyType.IMPORT,
                        in_type_checking=in_type_checking,
                    )
                    if dep:
                        dependencies.append(dep)

            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    # Relative import
                    dep_type = DependencyType.RELATIVE_IMPORT
                    module = "." * node.level + (node.module or "")
                else:
                    dep_type = DependencyType.FROM_IMPORT
                    module = node.module or ""

                imported = frozenset(
                    alias.asname or alias.name for alias in node.names
                )

                dep = self._create_dependency(
                    source_file=relative_path,
                    module=module,
                    imported_names=imported,
                    line=node.lineno,
                    dep_type=dep_type,
                    in_type_checking=in_type_checking,
                )
                if dep:
                    dependencies.append(dep)

        return dependencies

    async def analyze_directory(
        self,
        directory: Optional[Path] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Dependency]:
        """Analyze all Python files in a directory."""
        directory = directory or self.repo_path
        exclude_patterns = exclude_patterns or [
            "__pycache__",
            ".git",
            "venv",
            "node_modules",
            ".tox",
            "build",
            "dist",
        ]

        dependencies = []

        for py_file in directory.rglob("*.py"):
            # Check exclusions
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue

            file_deps = await self.analyze_file(py_file)
            dependencies.extend(file_deps)

        return dependencies

    def _create_dependency(
        self,
        source_file: str,
        module: str,
        imported_names: FrozenSet[str],
        line: int,
        dep_type: DependencyType,
        in_type_checking: bool = False,
    ) -> Optional[Dependency]:
        """Create a dependency object, determining target repo."""
        # Determine target repo from module name
        target_repo = self.repo_name  # Default to same repo

        for prefix, repo in self._known_repos.items():
            if module.startswith(prefix):
                target_repo = repo
                break

        # Skip standard library and common packages
        stdlib_prefixes = {
            "os", "sys", "re", "json", "time", "datetime", "pathlib",
            "asyncio", "typing", "logging", "collections", "itertools",
            "functools", "dataclasses", "enum", "abc", "contextlib",
            "unittest", "pytest", "subprocess", "threading", "multiprocessing",
            "hashlib", "uuid", "random", "math", "copy", "pickle",
        }

        if module.split(".")[0] in stdlib_prefixes:
            return None

        # Skip common third-party packages (can be configured)
        third_party = {
            "aiohttp", "fastapi", "pydantic", "sqlalchemy", "redis",
            "openai", "anthropic", "langchain", "numpy", "pandas",
            "requests", "httpx", "websockets", "uvicorn", "starlette",
        }

        if module.split(".")[0] in third_party:
            return None

        return Dependency(
            source_repo=self.repo_name,
            source_file=source_file,
            source_line=line,
            target_repo=target_repo,
            target_module=module,
            imported_names=imported_names,
            dependency_type=DependencyType.TYPE_HINT if in_type_checking else dep_type,
            is_optional=False,
        )


class CrossRepoDependencyTracker:
    """
    Tracks and validates dependencies across repos.

    Features:
    - Scans repos for import statements
    - Builds cross-repo dependency graph
    - Validates changes don't break imports
    - Suggests import updates when files move

    Usage:
        tracker = CrossRepoDependencyTracker(repos={
            "jarvis": Path("/path/to/jarvis"),
            "jarvis_prime": Path("/path/to/jprime"),
            "reactor_core": Path("/path/to/reactor"),
        })

        await tracker.scan_all()

        # Check if a change would break imports
        breaks = await tracker.validate_change(
            repo="jarvis_prime",
            old_path="core/models.py",
            new_path="core/models_v2.py",
        )
    """

    def __init__(
        self,
        repos: Dict[str, Path],
        cache_dir: Optional[Path] = None,
    ):
        self.repos = repos
        self.cache_dir = cache_dir or Path.home() / ".jarvis" / "dependencies"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._graph = DependencyGraph()
        self._analyzers: Dict[str, ImportAnalyzer] = {}
        self._last_scan: Dict[str, float] = {}

        # Register repo module mappings
        self._module_mappings = {
            "backend": "jarvis",
            "jarvis_prime": "jarvis_prime",
            "reactor_core": "reactor_core",
        }

        # Initialize analyzers
        for repo_name, repo_path in repos.items():
            if repo_path.exists():
                analyzer = ImportAnalyzer(repo_name, repo_path)
                for prefix, target in self._module_mappings.items():
                    analyzer.register_repo_mapping(prefix, target)
                self._analyzers[repo_name] = analyzer

    async def scan_all(self, force: bool = False) -> Dict[str, int]:
        """
        Scan all repos for dependencies.

        Args:
            force: Force rescan even if cache is fresh

        Returns:
            Dict mapping repo -> dependency count
        """
        results = {}

        for repo_name, analyzer in self._analyzers.items():
            # Check cache freshness (1 hour)
            if not force and self._last_scan.get(repo_name, 0) > time.time() - 3600:
                continue

            logger.info(f"[DependencyTracker] Scanning {repo_name}...")
            deps = await analyzer.analyze_directory()

            for dep in deps:
                self._graph.add_dependency(dep)

            results[repo_name] = len(deps)
            self._last_scan[repo_name] = time.time()

        # Save to cache
        await self._save_cache()

        return results

    async def validate_change(
        self,
        repo: str,
        changes: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Validate that proposed changes don't break imports.

        Args:
            repo: The repo being changed
            changes: List of {"old_path": "...", "new_path": "..."} dicts

        Returns:
            List of potential breaks with details
        """
        breaks = []

        for change in changes:
            old_path = change.get("old_path")
            new_path = change.get("new_path")

            if not old_path:
                continue

            # Get old module name
            old_module = self._path_to_module(old_path)

            # Find all dependents
            dependents = self._graph.get_transitive_dependents(repo, old_module)

            if dependents:
                new_module = self._path_to_module(new_path) if new_path else None

                for dep_key in dependents:
                    dep_repo, dep_file = dep_key.split(":", 1)

                    breaks.append({
                        "broken_file": dep_file,
                        "broken_repo": dep_repo,
                        "old_import": old_module,
                        "new_import": new_module,
                        "change_type": "move" if new_path else "delete",
                        "can_auto_fix": new_path is not None,
                    })

        return breaks

    async def get_affected_by_change(
        self,
        repo: str,
        module: str,
    ) -> Dict[str, List[str]]:
        """
        Get all files affected by changes to a module.

        Args:
            repo: The repo containing the module
            module: The module being changed

        Returns:
            Dict mapping repo -> list of affected files
        """
        dependents = self._graph.get_transitive_dependents(repo, module)

        affected: Dict[str, List[str]] = {}
        for dep_key in dependents:
            dep_repo, dep_file = dep_key.split(":", 1)
            if dep_repo not in affected:
                affected[dep_repo] = []
            affected[dep_repo].append(dep_file)

        return affected

    async def generate_import_updates(
        self,
        repo: str,
        old_module: str,
        new_module: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate import update patches for affected files.

        Args:
            repo: The repo where module was renamed
            old_module: Old module path
            new_module: New module path

        Returns:
            List of update patches
        """
        updates = []
        dependents = self._graph.get_dependents(repo, old_module)

        for dep_key in dependents:
            dep_repo, dep_file = dep_key.split(":", 1)
            repo_path = self.repos.get(dep_repo)

            if not repo_path:
                continue

            file_path = repo_path / dep_file
            if not file_path.exists():
                continue

            content = file_path.read_text()
            new_content = content

            # Replace import statements
            # from old.module import X -> from new.module import X
            new_content = re.sub(
                rf"from\s+{re.escape(old_module)}\s+import",
                f"from {new_module} import",
                new_content,
            )

            # import old.module -> import new.module
            new_content = re.sub(
                rf"import\s+{re.escape(old_module)}(\s|$|,)",
                f"import {new_module}\\1",
                new_content,
            )

            if new_content != content:
                updates.append({
                    "repo": dep_repo,
                    "file": dep_file,
                    "old_content": content,
                    "new_content": new_content,
                })

        return updates

    async def get_cross_repo_dependencies(self) -> List[Dict[str, Any]]:
        """Get all cross-repo dependencies."""
        cross_repo = self._graph.get_cross_repo_edges()
        return [dep.to_dict() for dep in cross_repo]

    async def detect_cycles(self) -> List[List[str]]:
        """Detect dependency cycles."""
        return self._graph.detect_cycles()

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get dependency graph statistics."""
        return self._graph.get_stats()

    def _path_to_module(self, path: str) -> str:
        """Convert file path to module name."""
        if path.endswith(".py"):
            path = path[:-3]
        return path.replace("/", ".").replace("\\", ".")

    async def _save_cache(self) -> None:
        """Save dependency graph to cache."""
        cache_file = self.cache_dir / "graph.json"
        data = {
            "last_scan": self._last_scan,
            "edges": [dep.to_dict() for dep in self._graph._edges],
        }
        cache_file.write_text(json.dumps(data, indent=2))

    async def _load_cache(self) -> None:
        """Load dependency graph from cache."""
        cache_file = self.cache_dir / "graph.json"
        if not cache_file.exists():
            return

        try:
            data = json.loads(cache_file.read_text())
            self._last_scan = data.get("last_scan", {})

            for edge_data in data.get("edges", []):
                dep = Dependency(
                    source_repo=edge_data["source_repo"],
                    source_file=edge_data["source_file"],
                    source_line=edge_data["source_line"],
                    target_repo=edge_data["target_repo"],
                    target_module=edge_data["target_module"],
                    imported_names=frozenset(edge_data["imported_names"]),
                    dependency_type=DependencyType(edge_data["dependency_type"]),
                    is_optional=edge_data.get("is_optional", False),
                )
                self._graph.add_dependency(dep)

        except Exception as e:
            logger.warning(f"[DependencyTracker] Failed to load cache: {e}")


# Global instance
_tracker: Optional[CrossRepoDependencyTracker] = None


def get_dependency_tracker(
    repos: Optional[Dict[str, Path]] = None
) -> CrossRepoDependencyTracker:
    """Get or create global dependency tracker."""
    global _tracker

    if _tracker is None:
        if repos is None:
            repos = {
                "jarvis": Path(os.getenv("Ironcliw_REPO", str(Path.home() / "Documents/repos/Ironcliw-AI-Agent"))),
                "jarvis_prime": Path(os.getenv("Ironcliw_PRIME_REPO", str(Path.home() / "Documents/repos/jarvis-prime"))),
                "reactor_core": Path(os.getenv("REACTOR_CORE_REPO", str(Path.home() / "Documents/repos/reactor-core"))),
            }
        _tracker = CrossRepoDependencyTracker(repos=repos)

    return _tracker
