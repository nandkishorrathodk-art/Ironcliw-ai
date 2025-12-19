#!/usr/bin/env python3
"""
JARVIS Update Detector
=======================

Async GitHub polling and change detection for the Self-Updating Lifecycle Manager.
Checks for new commits, releases, and provides update information.

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiohttp

from .supervisor_config import SupervisorConfig, UpdateSource, get_supervisor_config

logger = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    """Information about a git commit."""
    sha: str
    message: str
    author: str
    date: datetime
    url: Optional[str] = None


@dataclass
class UpdateInfo:
    """Information about an available update."""
    available: bool = False
    current_version: Optional[str] = None
    remote_version: Optional[str] = None
    commits_behind: int = 0
    commits: list[CommitInfo] = field(default_factory=list)
    summary: str = ""
    is_release: bool = False
    release_tag: Optional[str] = None
    release_notes: Optional[str] = None
    checked_at: datetime = field(default_factory=datetime.now)


class UpdateDetector:
    """
    Async GitHub polling for update detection.
    
    Features:
    - Configurable polling intervals
    - GitHub API rate limit awareness
    - Local cache of last known commit
    - Semantic version comparison
    - Release and commit-based detection
    
    Example:
        >>> detector = UpdateDetector(config)
        >>> update_info = await detector.check_for_updates()
        >>> if update_info.available:
        ...     print(f"Update available: {update_info.summary}")
    """
    
    # GitHub API base
    GITHUB_API = "https://api.github.com"
    
    def __init__(
        self,
        config: Optional[SupervisorConfig] = None,
        repo_path: Optional[Path] = None,
    ):
        """
        Initialize the update detector.
        
        Args:
            config: Supervisor configuration
            repo_path: Path to local git repository
        """
        self.config = config or get_supervisor_config()
        self.repo_path = repo_path or self._detect_repo_path()
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_check: Optional[datetime] = None
        self._cached_update: Optional[UpdateInfo] = None
        self._rate_limit_reset: Optional[datetime] = None
        
        # Extract owner/repo from git remote
        self._owner: Optional[str] = None
        self._repo: Optional[str] = None
        
        logger.info("🔧 Update detector initialized")
    
    def _detect_repo_path(self) -> Path:
        """Detect the git repository path."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / ".git").exists():
                return parent
        return Path.cwd()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "JARVIS-Update-Detector/1.0",
            }
            
            # Add GitHub token if available
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                headers["Authorization"] = f"token {token}"
            
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        
        return self._session
    
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
    
    async def _get_remote_info(self) -> tuple[Optional[str], Optional[str]]:
        """Get owner and repo from git remote."""
        if self._owner and self._repo:
            return self._owner, self._repo
        
        remote = self.config.update.remote
        success, output = await self._run_git_command(
            f"git remote get-url {remote}"
        )
        
        if not success:
            return None, None
        
        # Parse GitHub URL
        # SSH: git@github.com:owner/repo.git
        # HTTPS: https://github.com/owner/repo.git
        patterns = [
            r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                self._owner = match.group(1)
                self._repo = match.group(2)
                return self._owner, self._repo
        
        return None, None
    
    async def get_current_commit(self) -> Optional[str]:
        """Get the current local commit."""
        success, output = await self._run_git_command("git rev-parse HEAD")
        return output if success else None
    
    async def get_local_commits(self, limit: int = 10) -> list[CommitInfo]:
        """Get recent local commits."""
        success, output = await self._run_git_command(
            f'git log -n {limit} --format="%H|%s|%an|%aI"'
        )
        
        if not success:
            return []
        
        commits = []
        for line in output.split("\n"):
            if line:
                parts = line.split("|", 3)
                if len(parts) >= 4:
                    commits.append(CommitInfo(
                        sha=parts[0],
                        message=parts[1],
                        author=parts[2],
                        date=datetime.fromisoformat(parts[3]),
                    ))
        
        return commits
    
    async def _check_rate_limit(self) -> bool:
        """Check if we're within GitHub rate limits."""
        if self._rate_limit_reset:
            if datetime.now() < self._rate_limit_reset:
                remaining_seconds = (
                    self._rate_limit_reset - datetime.now()
                ).total_seconds()
                logger.debug(f"Rate limited, {remaining_seconds:.0f}s remaining")
                return False
            else:
                self._rate_limit_reset = None
        return True
    
    async def _handle_rate_limit(
        self,
        response: aiohttp.ClientResponse,
    ) -> None:
        """Handle rate limit headers from GitHub."""
        remaining = response.headers.get("X-RateLimit-Remaining", "")
        reset = response.headers.get("X-RateLimit-Reset", "")
        
        if remaining and int(remaining) == 0 and reset:
            self._rate_limit_reset = datetime.fromtimestamp(int(reset))
            logger.warning(
                f"⚠️ GitHub rate limit reached, resets at {self._rate_limit_reset}"
            )
    
    async def check_github_api(self) -> Optional[UpdateInfo]:
        """Check for updates using GitHub API."""
        if not await self._check_rate_limit():
            return self._cached_update
        
        owner, repo = await self._get_remote_info()
        if not owner or not repo:
            logger.warning("Could not determine GitHub owner/repo")
            return None
        
        session = await self._get_session()
        branch = self.config.update.branch
        
        try:
            # Get latest commit from remote branch
            url = f"{self.GITHUB_API}/repos/{owner}/{repo}/branches/{branch}"
            
            async with session.get(url) as response:
                await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    remote_sha = data.get("commit", {}).get("sha", "")
                    
                    current_sha = await self.get_current_commit()
                    
                    if current_sha and remote_sha and current_sha != remote_sha:
                        # Get commits between current and remote
                        compare_url = (
                            f"{self.GITHUB_API}/repos/{owner}/{repo}"
                            f"/compare/{current_sha[:12]}...{remote_sha[:12]}"
                        )
                        
                        async with session.get(compare_url) as compare_response:
                            if compare_response.status == 200:
                                compare_data = await compare_response.json()
                                commits_data = compare_data.get("commits", [])
                                
                                commits = []
                                for c in commits_data[:10]:  # Limit to 10
                                    commits.append(CommitInfo(
                                        sha=c.get("sha", ""),
                                        message=c.get("commit", {}).get("message", "").split("\n")[0],
                                        author=c.get("commit", {}).get("author", {}).get("name", ""),
                                        date=datetime.fromisoformat(
                                            c.get("commit", {}).get("author", {}).get("date", "").replace("Z", "+00:00")
                                        ),
                                        url=c.get("html_url"),
                                    ))
                                
                                return UpdateInfo(
                                    available=True,
                                    current_version=current_sha[:12],
                                    remote_version=remote_sha[:12],
                                    commits_behind=compare_data.get("ahead_by", len(commits)),
                                    commits=commits,
                                    summary=self._generate_summary(commits),
                                    checked_at=datetime.now(),
                                )
                    
                    # No update available
                    return UpdateInfo(
                        available=False,
                        current_version=current_sha[:12] if current_sha else None,
                        remote_version=remote_sha[:12],
                        checked_at=datetime.now(),
                    )
                
                elif response.status == 404:
                    logger.warning(f"Repository or branch not found: {owner}/{repo}/{branch}")
                    return None
                else:
                    logger.warning(f"GitHub API error: {response.status}")
                    return None
                    
        except aiohttp.ClientError as e:
            logger.error(f"GitHub API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error checking GitHub: {e}")
            return None
    
    async def check_git_fetch(self) -> Optional[UpdateInfo]:
        """Check for updates using local git fetch."""
        remote = self.config.update.remote
        branch = self.config.update.branch
        
        # Fetch from remote
        success, _ = await self._run_git_command(
            f"git fetch {remote}",
            timeout=60,
        )
        
        if not success:
            logger.warning("Git fetch failed")
            return None
        
        # Check commits behind
        success, output = await self._run_git_command(
            f"git rev-list HEAD..{remote}/{branch} --count"
        )
        
        if not success:
            return None
        
        try:
            commits_behind = int(output.strip())
        except ValueError:
            commits_behind = 0
        
        if commits_behind == 0:
            current = await self.get_current_commit()
            return UpdateInfo(
                available=False,
                current_version=current[:12] if current else None,
                checked_at=datetime.now(),
            )
        
        # Get log of new commits
        success, log_output = await self._run_git_command(
            f'git log HEAD..{remote}/{branch} --format="%H|%s|%an|%aI" -n 10'
        )
        
        commits = []
        if success:
            for line in log_output.split("\n"):
                if line:
                    parts = line.split("|", 3)
                    if len(parts) >= 4:
                        commits.append(CommitInfo(
                            sha=parts[0],
                            message=parts[1],
                            author=parts[2],
                            date=datetime.fromisoformat(parts[3]),
                        ))
        
        current = await self.get_current_commit()
        success, remote_head = await self._run_git_command(
            f"git rev-parse {remote}/{branch}"
        )
        
        return UpdateInfo(
            available=True,
            current_version=current[:12] if current else None,
            remote_version=remote_head[:12] if success else None,
            commits_behind=commits_behind,
            commits=commits,
            summary=self._generate_summary(commits),
            checked_at=datetime.now(),
        )
    
    def _generate_summary(self, commits: list[CommitInfo]) -> str:
        """Generate a human-readable summary of commits."""
        if not commits:
            return "Updates available"
        
        if len(commits) == 1:
            return commits[0].message
        
        # Categorize commits
        features = []
        fixes = []
        other = []
        
        for commit in commits:
            msg = commit.message.lower()
            if any(kw in msg for kw in ["feat", "add", "new", "implement"]):
                features.append(commit.message)
            elif any(kw in msg for kw in ["fix", "bug", "patch", "resolve"]):
                fixes.append(commit.message)
            else:
                other.append(commit.message)
        
        parts = []
        if features:
            parts.append(f"{len(features)} new feature(s)")
        if fixes:
            parts.append(f"{len(fixes)} fix(es)")
        if other:
            parts.append(f"{len(other)} other change(s)")
        
        return ", ".join(parts) if parts else f"{len(commits)} commits"
    
    async def check_for_updates(
        self,
        force: bool = False,
    ) -> Optional[UpdateInfo]:
        """
        Check for available updates.
        
        Args:
            force: Force check even if recently checked
            
        Returns:
            UpdateInfo with update details
        """
        # Check cache
        if not force and self._cached_update:
            elapsed = (datetime.now() - self._cached_update.checked_at).total_seconds()
            if elapsed < self.config.update.check.interval_seconds:
                return self._cached_update
        
        logger.info("🔍 Checking for updates...")
        
        # Try GitHub API first if using GitHub
        if self.config.update.source_type == UpdateSource.GITHUB:
            update_info = await self.check_github_api()
            if update_info:
                self._cached_update = update_info
                
                if update_info.available:
                    logger.info(
                        f"📦 Update available: {update_info.commits_behind} commit(s) behind"
                    )
                else:
                    logger.info("✅ Already up to date")
                
                return update_info
        
        # Fallback to git fetch
        update_info = await self.check_git_fetch()
        if update_info:
            self._cached_update = update_info
            
            if update_info.available:
                logger.info(
                    f"📦 Update available: {update_info.commits_behind} commit(s) behind"
                )
            else:
                logger.info("✅ Already up to date")
        
        return update_info
    
    def get_cached_update(self) -> Optional[UpdateInfo]:
        """Get the cached update info (if any)."""
        return self._cached_update
    
    async def close(self) -> None:
        """Close resources."""
        if self._session and not self._session.closed:
            await self._session.close()
