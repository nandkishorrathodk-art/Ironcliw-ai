"""
v77.0: RepoMaster Framework Adapter
===================================

Adapter for RepoMaster - intelligent codebase analysis.

RepoMaster provides:
- Dependency graph analysis
- Code complexity scoring
- Risk assessment
- File discovery

For codebase understanding before making changes.

Author: Ironcliw v77.0
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import AnalysisResult, CodingCouncilConfig

logger = logging.getLogger(__name__)


class RepoMasterAdapter:
    """
    Adapter for RepoMaster codebase analysis.

    If RepoMaster is not installed, falls back to basic
    AST-based analysis using Python stdlib.
    """

    def __init__(self, config: "CodingCouncilConfig"):
        self.config = config
        self.repo_root = config.repo_root
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._available: Optional[bool] = None

    async def is_available(self) -> bool:
        """Check if RepoMaster is available."""
        if self._available is not None:
            return self._available

        try:
            # Check for repomaster
            import importlib.util
            spec = importlib.util.find_spec("repomaster")
            self._available = spec is not None
        except Exception:
            self._available = False

        if not self._available:
            logger.info("[RepoMasterAdapter] Using fallback analysis")

        return True  # Always available due to fallback

    async def analyze(
        self,
        target_files: List[str],
        description: str
    ) -> "AnalysisResult":
        """
        Analyze codebase around target files.

        Returns:
            AnalysisResult with dependencies, structure, and insights
        """
        from ..types import AnalysisResult

        logger.info(f"[RepoMasterAdapter] Analyzing {len(target_files)} files")

        # Run analysis in parallel
        results = await asyncio.gather(
            self._analyze_dependencies(target_files),
            self._analyze_complexity(target_files),
            self._find_related_files(target_files, description),
            return_exceptions=True
        )

        dependencies = results[0] if isinstance(results[0], dict) else {}
        complexity = results[1] if isinstance(results[1], dict) else {}
        related = results[2] if isinstance(results[2], list) else []

        # Calculate risk score
        risk_score = self._calculate_risk(target_files, dependencies)

        # Generate insights
        insights = self._generate_insights(target_files, dependencies, complexity)

        return AnalysisResult(
            target_files=target_files + related[:5],  # Add up to 5 related files
            dependencies=dependencies.get("imports", {}),
            dependents=dependencies.get("imported_by", {}),
            structure=complexity,
            insights=insights,
            suggestions=[],
            complexity_score=complexity.get("average_complexity", 0.5),
            risk_score=risk_score,
        )

    async def _analyze_dependencies(
        self,
        files: List[str]
    ) -> Dict[str, Any]:
        """Analyze import dependencies."""
        loop = asyncio.get_running_loop()

        def _analyze():
            result = {
                "imports": {},      # file -> [files it imports]
                "imported_by": {},  # file -> [files that import it]
            }

            import ast

            for filepath in files:
                full_path = self.repo_root / filepath
                if not full_path.exists() or not filepath.endswith(".py"):
                    continue

                try:
                    with open(full_path) as f:
                        tree = ast.parse(f.read())

                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)

                    result["imports"][filepath] = imports

                except Exception as e:
                    logger.debug(f"Could not parse {filepath}: {e}")

            return result

        return await loop.run_in_executor(self._executor, _analyze)

    async def _analyze_complexity(
        self,
        files: List[str]
    ) -> Dict[str, Any]:
        """Analyze code complexity."""
        loop = asyncio.get_running_loop()

        def _analyze():
            result = {
                "files": {},
                "total_lines": 0,
                "total_functions": 0,
                "total_classes": 0,
                "average_complexity": 0.5,
            }

            import ast

            complexities = []

            for filepath in files:
                full_path = self.repo_root / filepath
                if not full_path.exists() or not filepath.endswith(".py"):
                    continue

                try:
                    with open(full_path) as f:
                        content = f.read()
                        tree = ast.parse(content)

                    lines = len(content.split("\n"))
                    functions = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
                    classes = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))

                    # Simple complexity heuristic
                    complexity = min(1.0, (lines + functions * 10 + classes * 20) / 1000)
                    complexities.append(complexity)

                    result["files"][filepath] = {
                        "lines": lines,
                        "functions": functions,
                        "classes": classes,
                        "complexity": complexity,
                    }

                    result["total_lines"] += lines
                    result["total_functions"] += functions
                    result["total_classes"] += classes

                except Exception as e:
                    logger.debug(f"Could not analyze {filepath}: {e}")

            if complexities:
                result["average_complexity"] = sum(complexities) / len(complexities)

            return result

        return await loop.run_in_executor(self._executor, _analyze)

    async def _find_related_files(
        self,
        files: List[str],
        description: str
    ) -> List[str]:
        """Find files related to the target files."""
        related = set()

        # Find files in same directories
        for filepath in files:
            parent = (self.repo_root / filepath).parent
            if parent.exists():
                for sibling in parent.glob("*.py"):
                    rel_path = str(sibling.relative_to(self.repo_root))
                    if rel_path not in files:
                        related.add(rel_path)

        # Find files matching keywords from description
        keywords = re.findall(r'\b[a-z_]+\b', description.lower())
        if keywords:
            for keyword in keywords[:3]:  # Limit to 3 keywords
                if len(keyword) > 3:
                    pattern = f"**/*{keyword}*.py"
                    for match in self.repo_root.glob(pattern):
                        rel_path = str(match.relative_to(self.repo_root))
                        if rel_path not in files:
                            related.add(rel_path)

        return list(related)[:10]  # Limit to 10 related files

    def _calculate_risk(
        self,
        files: List[str],
        dependencies: Dict[str, Any]
    ) -> float:
        """Calculate risk score for modifying files."""
        risk = 0.3  # Base risk

        # Higher risk for core files
        core_patterns = ["main.py", "__init__.py", "supervisor", "core", "config"]
        for filepath in files:
            for pattern in core_patterns:
                if pattern in filepath:
                    risk += 0.1

        # Higher risk if many files depend on these
        imported_by = dependencies.get("imported_by", {})
        for filepath in files:
            dependents = len(imported_by.get(filepath, []))
            if dependents > 10:
                risk += 0.2
            elif dependents > 5:
                risk += 0.1

        return min(1.0, risk)

    def _generate_insights(
        self,
        files: List[str],
        dependencies: Dict[str, Any],
        complexity: Dict[str, Any]
    ) -> List[str]:
        """Generate analysis insights."""
        insights = []

        # File count insight
        insights.append(f"Analyzing {len(files)} target file(s)")

        # Complexity insights
        avg_complexity = complexity.get("average_complexity", 0.5)
        if avg_complexity > 0.7:
            insights.append("High complexity detected - changes may have broad impact")
        elif avg_complexity < 0.3:
            insights.append("Low complexity - changes should be straightforward")

        # Line count insight
        total_lines = complexity.get("total_lines", 0)
        if total_lines > 1000:
            insights.append(f"Large codebase section ({total_lines} lines)")

        # Function/class insights
        total_functions = complexity.get("total_functions", 0)
        total_classes = complexity.get("total_classes", 0)
        if total_functions > 20:
            insights.append(f"Many functions ({total_functions}) - ensure changes don't break callers")
        if total_classes > 5:
            insights.append(f"Multiple classes ({total_classes}) - check inheritance")

        return insights
