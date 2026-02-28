#!/usr/bin/env python3
"""
Ironcliw Changelog Analyzer
==========================

AI-powered commit summarization and categorization for the Self-Updating Lifecycle Manager.
Parses git logs and generates user-friendly descriptions of changes.

Author: Ironcliw System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from .supervisor_config import SupervisorConfig, get_supervisor_config

logger = logging.getLogger(__name__)


class ChangeCategory(str, Enum):
    """Categories of changes."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    FEATURE = "feature"
    FIX = "fix"
    REFACTOR = "refactor"
    DOCS = "docs"
    CHORE = "chore"
    UNKNOWN = "unknown"


@dataclass
class CommitSummary:
    """Summarized commit information."""
    sha: str
    original_message: str
    summary: str
    category: ChangeCategory
    impact: str  # low, medium, high
    author: str
    date: datetime
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


@dataclass
class ChangelogSummary:
    """Summary of multiple commits for user presentation."""
    total_commits: int
    date_range: tuple[datetime, datetime]
    summary: str  # Human-friendly summary
    highlights: list[str]  # Key changes
    categories: dict[ChangeCategory, int]  # Count per category
    commits: list[CommitSummary]
    security_changes: bool = False
    breaking_changes: bool = False


class ChangelogAnalyzer:
    """
    AI-powered commit summarization.
    
    Features:
    - Git log parsing with configurable depth
    - Conventional Commit detection
    - Category classification
    - Impact assessment
    - User-friendly summary generation
    
    Example:
        >>> analyzer = ChangelogAnalyzer(config)
        >>> summary = await analyzer.analyze_since("HEAD~10")
        >>> print(summary.summary)
    """
    
    # Conventional Commit patterns
    CONVENTIONAL_PATTERNS = {
        ChangeCategory.FEATURE: [r"^feat(\(.+\))?:", r"^add:", r"^new:"],
        ChangeCategory.FIX: [r"^fix(\(.+\))?:", r"^bug:", r"^patch:"],
        ChangeCategory.SECURITY: [r"^security:", r"(?i)security", r"(?i)cve-", r"(?i)vulnerability"],
        ChangeCategory.PERFORMANCE: [r"^perf(\(.+\))?:", r"(?i)performance", r"(?i)optimize", r"(?i)speed"],
        ChangeCategory.REFACTOR: [r"^refactor(\(.+\))?:", r"^improve:", r"^clean:"],
        ChangeCategory.DOCS: [r"^docs(\(.+\))?:", r"^doc:"],
        ChangeCategory.CHORE: [r"^chore(\(.+\))?:", r"^build:", r"^ci:"],
    }
    
    # Keywords for impact assessment
    IMPACT_KEYWORDS = {
        "high": ["breaking", "critical", "security", "major", "important", "urgent"],
        "medium": ["improve", "enhance", "update", "add", "new", "change"],
        "low": ["minor", "small", "fix", "typo", "cleanup", "refactor"],
    }
    
    def __init__(
        self,
        config: Optional[SupervisorConfig] = None,
        repo_path: Optional[Path] = None,
    ):
        """
        Initialize the changelog analyzer.
        
        Args:
            config: Supervisor configuration
            repo_path: Path to git repository
        """
        self.config = config or get_supervisor_config()
        self.repo_path = repo_path or self._detect_repo_path()
        
        logger.info("🔧 Changelog analyzer initialized")
    
    def _detect_repo_path(self) -> Path:
        """Detect the git repository path."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists():
                return parent
        return Path.cwd()
    
    async def _run_git_command(
        self,
        command: str,
        timeout: int = 30,
    ) -> tuple[bool, str]:
        """Run a git command asynchronously."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            
            if process.returncode == 0:
                return True, stdout.decode().strip()
            else:
                return False, stderr.decode().strip()
                
        except asyncio.TimeoutError:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
    
    def _classify_category(self, message: str) -> ChangeCategory:
        """Classify a commit message into a category."""
        for category, patterns in self.CONVENTIONAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return category
        return ChangeCategory.UNKNOWN
    
    def _assess_impact(self, message: str) -> str:
        """Assess the impact level of a change."""
        message_lower = message.lower()
        
        for impact, keywords in self.IMPACT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return impact
        
        return "medium"  # Default
    
    def _extract_summary(self, message: str) -> str:
        """Extract a clean summary from a commit message."""
        # Remove conventional commit prefix
        summary = re.sub(r"^(feat|fix|docs|style|refactor|perf|test|chore)(\(.+\))?:\s*", "", message)
        
        # Take first line only
        summary = summary.split("\n")[0]
        
        # Capitalize and limit length
        summary = summary.strip()
        if summary:
            summary = summary[0].upper() + summary[1:]
        
        if len(summary) > 100:
            summary = summary[:97] + "..."
        
        return summary
    
    async def get_commits(
        self,
        since: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[CommitSummary]:
        """
        Get and analyze commits.
        
        Args:
            since: Git reference to start from (e.g., "HEAD~10", commit SHA)
            limit: Maximum number of commits
            
        Returns:
            List of analyzed commit summaries
        """
        limit = limit or self.config.changelog.max_commits
        
        # Build git log command
        if since:
            range_spec = f"{since}..HEAD"
        else:
            range_spec = f"-n {limit}"
        
        # Get commit info with stats
        success, output = await self._run_git_command(
            f'git log {range_spec} --format="COMMIT_START%n%H|%s|%an|%aI%nCOMMIT_END" --shortstat',
            timeout=60,
        )
        
        if not success:
            logger.warning(f"Git log failed: {output}")
            return []
        
        commits = []
        current_commit = None
        
        lines = output.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line == "COMMIT_START":
                i += 1
                if i < len(lines):
                    commit_line = lines[i]
                    parts = commit_line.split("|", 3)
                    if len(parts) >= 4:
                        message = parts[1]
                        current_commit = CommitSummary(
                            sha=parts[0],
                            original_message=message,
                            summary=self._extract_summary(message),
                            category=self._classify_category(message),
                            impact=self._assess_impact(message),
                            author=parts[2],
                            date=datetime.fromisoformat(parts[3]),
                        )
            
            elif line == "COMMIT_END" and current_commit:
                commits.append(current_commit)
                current_commit = None
            
            elif current_commit and "file" in line:
                # Parse stat line: "3 files changed, 10 insertions(+), 5 deletions(-)"
                match = re.search(r"(\d+) file", line)
                if match:
                    current_commit.files_changed = int(match.group(1))
                
                match = re.search(r"(\d+) insertion", line)
                if match:
                    current_commit.insertions = int(match.group(1))
                
                match = re.search(r"(\d+) deletion", line)
                if match:
                    current_commit.deletions = int(match.group(1))
            
            i += 1
        
        return commits[:limit]
    
    def _generate_highlights(self, commits: list[CommitSummary]) -> list[str]:
        """Generate highlight bullet points from commits."""
        highlights = []
        
        # Group by category
        by_category: dict[ChangeCategory, list[CommitSummary]] = {}
        for commit in commits:
            if commit.category not in by_category:
                by_category[commit.category] = []
            by_category[commit.category].append(commit)
        
        # Generate highlights per category (prioritized)
        priority = [
            ChangeCategory.SECURITY,
            ChangeCategory.FEATURE,
            ChangeCategory.FIX,
            ChangeCategory.PERFORMANCE,
        ]
        
        for category in priority:
            if category in by_category:
                for commit in by_category[category][:2]:  # Max 2 per category
                    prefix = {
                        ChangeCategory.SECURITY: "🔒",
                        ChangeCategory.FEATURE: "✨",
                        ChangeCategory.FIX: "🐛",
                        ChangeCategory.PERFORMANCE: "⚡",
                    }.get(category, "📝")
                    highlights.append(f"{prefix} {commit.summary}")
        
        return highlights[:5]  # Max 5 highlights
    
    def _generate_user_summary(self, commits: list[CommitSummary]) -> str:
        """Generate a user-friendly summary for voice announcement."""
        if not commits:
            return "No changes to report"
        
        # Count by category
        categories: dict[ChangeCategory, int] = {}
        for commit in commits:
            categories[commit.category] = categories.get(commit.category, 0) + 1
        
        # Build natural language summary
        parts = []
        
        # Security first
        if ChangeCategory.SECURITY in categories:
            count = categories[ChangeCategory.SECURITY]
            parts.append(f"{count} security update{'s' if count > 1 else ''}")
        
        # Features
        if ChangeCategory.FEATURE in categories:
            count = categories[ChangeCategory.FEATURE]
            parts.append(f"{count} new feature{'s' if count > 1 else ''}")
        
        # Fixes
        if ChangeCategory.FIX in categories:
            count = categories[ChangeCategory.FIX]
            parts.append(f"{count} bug fix{'es' if count > 1 else ''}")
        
        # Performance
        if ChangeCategory.PERFORMANCE in categories:
            count = categories[ChangeCategory.PERFORMANCE]
            parts.append(f"{count} performance improvement{'s' if count > 1 else ''}")
        
        # Other
        other_count = sum(
            count for cat, count in categories.items()
            if cat not in [ChangeCategory.SECURITY, ChangeCategory.FEATURE, 
                          ChangeCategory.FIX, ChangeCategory.PERFORMANCE]
        )
        if other_count > 0:
            parts.append(f"{other_count} other change{'s' if other_count > 1 else ''}")
        
        if not parts:
            return f"{len(commits)} commits with various changes"
        
        # Join with proper grammar
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return ", ".join(parts[:-1]) + f", and {parts[-1]}"
    
    async def analyze_since(
        self,
        since: str,
        limit: Optional[int] = None,
    ) -> ChangelogSummary:
        """
        Analyze commits since a reference point.
        
        Args:
            since: Git reference (commit SHA, tag, "HEAD~N")
            limit: Maximum commits to analyze
            
        Returns:
            ChangelogSummary with analyzed changes
        """
        commits = await self.get_commits(since=since, limit=limit)
        
        if not commits:
            return ChangelogSummary(
                total_commits=0,
                date_range=(datetime.now(), datetime.now()),
                summary="No changes found",
                highlights=[],
                categories={},
                commits=[],
            )
        
        # Aggregate categories
        categories: dict[ChangeCategory, int] = {}
        for commit in commits:
            categories[commit.category] = categories.get(commit.category, 0) + 1
        
        # Date range
        dates = [c.date for c in commits]
        date_range = (min(dates), max(dates))
        
        # Check for security/breaking changes
        security_changes = ChangeCategory.SECURITY in categories
        breaking_changes = any(
            "breaking" in c.original_message.lower()
            for c in commits
        )
        
        return ChangelogSummary(
            total_commits=len(commits),
            date_range=date_range,
            summary=self._generate_user_summary(commits),
            highlights=self._generate_highlights(commits),
            categories=categories,
            commits=commits,
            security_changes=security_changes,
            breaking_changes=breaking_changes,
        )
    
    async def get_voice_announcement(
        self,
        since: str,
        max_words: Optional[int] = None,
    ) -> str:
        """
        Generate a voice-friendly announcement.
        
        Args:
            since: Git reference to compare from
            max_words: Maximum words in announcement
            
        Returns:
            Voice-friendly announcement string
        """
        max_words = max_words or self.config.changelog.max_length_words
        
        summary = await self.analyze_since(since)
        
        if summary.total_commits == 0:
            return "You are already running the latest version"
        
        # Build announcement
        parts = [
            f"A system update is available with {summary.summary}.",
        ]
        
        if summary.security_changes:
            parts.append("This includes important security improvements.")
        
        if summary.highlights:
            parts.append(f"Key changes include: {summary.highlights[0]}.")
        
        announcement = " ".join(parts)
        
        # Truncate if needed
        words = announcement.split()
        if len(words) > max_words:
            announcement = " ".join(words[:max_words]) + "..."
        
        return announcement
