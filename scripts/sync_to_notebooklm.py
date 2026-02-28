#!/usr/bin/env python3
"""
NotebookLM Sync Script

Pulls code/docs from GitHub repositories and prepares them for NotebookLM.
Supports both automated upload via MCP server and manual upload preparation.

Usage:
    python sync_to_notebooklm.py --mode auto    # Uses MCP browser automation
    python sync_to_notebooklm.py --mode manual  # Prepares files for manual upload
    python sync_to_notebooklm.py --repos owner/repo1,owner/repo2  # Custom repos
"""

import asyncio
import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Optional imports with graceful fallbacks
try:
    from github import Github, GithubException
    HAS_PYGITHUB = True
except ImportError:
    HAS_PYGITHUB = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class NotebookConfig:
    """Configuration for a NotebookLM notebook."""
    name: str
    repo: str
    description: str = ""
    include_patterns: list = field(default_factory=lambda: [
        "*.md", "*.txt", "*.py", "*.ts", "*.js", "*.json", "*.yaml", "*.yml",
        "*.sh", "*.sql", "*.html", "*.css", "*.rst", "*.toml"
    ])
    exclude_patterns: list = field(default_factory=lambda: [
        "node_modules/*", ".git/*", "__pycache__/*", "*.pyc", ".env*",
        "*.lock", "package-lock.json", "yarn.lock", "*.min.js", "*.min.css",
        "dist/*", "build/*", ".next/*", "coverage/*", "*.log", "*.bak",
        "venv/*", ".venv/*", "*.egg-info/*", ".pytest_cache/*"
    ])
    max_file_size_kb: int = 500  # Skip files larger than this
    priority_files: list = field(default_factory=lambda: [
        "README.md", "ARCHITECTURE.md", "DESIGN.md", "CONTRIBUTING.md",
        "docs/*", "documentation/*", "*.md"
    ])


@dataclass
class SyncReport:
    """Report of sync operation."""
    notebook_name: str
    repo: str
    files_processed: int = 0
    files_skipped: int = 0
    total_size_kb: float = 0
    errors: list = field(default_factory=list)
    files_included: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GitHubFetcher:
    """Fetches files from GitHub repositories."""

    def __init__(self, token: Optional[str] = None):
        if not HAS_PYGITHUB:
            raise ImportError("PyGithub not installed. Run: pip install PyGithub")

        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN env var or pass token.")

        from github import Auth
        self.github = Github(auth=Auth.Token(self.token))

    def get_repo_contents_fast(
        self,
        repo_name: str,
        config: NotebookConfig,
        output_dir: Path
    ) -> SyncReport:
        """
        Fast fetch using archive download (tarball).
        Much faster than API-based file-by-file fetching.
        """
        import tarfile
        import io
        import requests

        report = SyncReport(notebook_name=config.name, repo=repo_name)

        try:
            repo = self.github.get_repo(repo_name)
            archive_url = repo.get_archive_link("tarball")
        except GithubException as e:
            report.errors.append(f"Failed to access repo: {e}")
            return report

        # Download and extract archive
        try:
            headers = {"Authorization": f"token {self.token}"}
            response = requests.get(archive_url, headers=headers, stream=True, timeout=60)
            response.raise_for_status()

            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
                # GitHub adds a prefix directory to archives
                _ = tar.getnames()[0].split("/")[0]  # root_dir not needed

                for member in tar.getmembers():
                    if not member.isfile():
                        continue

                    # Get relative path (remove GitHub's prefix)
                    parts = member.name.split("/", 1)
                    if len(parts) < 2:
                        continue
                    rel_path = parts[1]

                    # Check exclusions
                    if self._should_exclude(rel_path, config.exclude_patterns):
                        report.files_skipped += 1
                        continue

                    # Check inclusions
                    if not self._should_include(rel_path, config.include_patterns):
                        report.files_skipped += 1
                        continue

                    # Check size
                    size_kb = member.size / 1024
                    if size_kb > config.max_file_size_kb:
                        report.files_skipped += 1
                        continue

                    # Extract and read content
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        content = f.read()
                        text_content = content.decode("utf-8")
                    except (UnicodeDecodeError, AttributeError):
                        report.files_skipped += 1
                        continue

                    # Save file
                    file_path = output_dir / rel_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(text_content, encoding="utf-8")

                    report.files_processed += 1
                    report.total_size_kb += size_kb
                    report.files_included.append(rel_path)

        except Exception as e:
            report.errors.append(f"Archive download failed: {e}")

        return report

    def get_repo_contents(
        self,
        repo_name: str,
        config: NotebookConfig,
        output_dir: Path
    ) -> SyncReport:
        """
        Fetch repository contents and save to output directory.

        Args:
            repo_name: Full repo name (owner/repo)
            config: Notebook configuration
            output_dir: Directory to save files

        Returns:
            SyncReport with operation details
        """
        report = SyncReport(notebook_name=config.name, repo=repo_name)

        try:
            repo = self.github.get_repo(repo_name)
        except GithubException as e:
            report.errors.append(f"Failed to access repo: {e}")
            return report

        # Get all files recursively
        self._fetch_directory(repo, "", config, output_dir, report)

        return report

    def _fetch_directory(
        self,
        repo,
        path: str,
        config: NotebookConfig,
        output_dir: Path,
        report: SyncReport
    ):
        """Recursively fetch directory contents."""
        try:
            contents = repo.get_contents(path)
        except GithubException as e:
            report.errors.append(f"Failed to get contents of {path}: {e}")
            return

        if not isinstance(contents, list):
            contents = [contents]

        for item in contents:
            if item.type == "dir":
                # Check if directory should be excluded
                if self._should_exclude(item.path, config.exclude_patterns):
                    continue
                self._fetch_directory(repo, item.path, config, output_dir, report)

            elif item.type == "file":
                self._process_file(repo, item, config, output_dir, report)

    def _process_file(
        self,
        _repo,  # Reserved for future use (e.g., fetching raw content)
        item,
        config: NotebookConfig,
        output_dir: Path,
        report: SyncReport
    ):
        """Process a single file."""
        # Check exclusions
        if self._should_exclude(item.path, config.exclude_patterns):
            report.files_skipped += 1
            return

        # Check inclusions
        if not self._should_include(item.path, config.include_patterns):
            report.files_skipped += 1
            return

        # Check file size
        size_kb = item.size / 1024
        if size_kb > config.max_file_size_kb:
            report.files_skipped += 1
            return

        # Fetch and save file
        try:
            # Handle binary vs text content
            try:
                content = item.decoded_content.decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                # Binary file, skip
                report.files_skipped += 1
                return

            # Create output path
            file_path = output_dir / item.path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            file_path.write_text(content, encoding="utf-8")

            report.files_processed += 1
            report.total_size_kb += size_kb
            report.files_included.append(item.path)

        except Exception as e:
            report.errors.append(f"Failed to process {item.path}: {e}")

    def _should_exclude(self, path: str, patterns: list) -> bool:
        """Check if path matches any exclusion pattern."""
        from fnmatch import fnmatch
        return any(fnmatch(path, p) or fnmatch(path.split("/")[-1], p) for p in patterns)

    def _should_include(self, path: str, patterns: list) -> bool:
        """Check if path matches any inclusion pattern."""
        from fnmatch import fnmatch
        filename = path.split("/")[-1]
        return any(fnmatch(path, p) or fnmatch(filename, p) for p in patterns)


class NotebookLMPreparer:
    """Prepares files for NotebookLM upload."""

    def __init__(self, output_base: Optional[Path] = None):
        self.output_base = output_base or Path.home() / "NotebookLM_Sources"
        self.output_base.mkdir(parents=True, exist_ok=True)

    def prepare_notebook_sources(
        self,
        notebook_name: str,
        source_dir: Path,
        max_sources: int = 50
    ) -> dict:
        """
        Prepare sources for NotebookLM upload.

        NotebookLM has limits:
        - Max 50 sources per notebook
        - Max 500,000 words per source
        - Supported: PDF, TXT, Markdown, Audio, Web URLs, Google Docs/Slides

        Strategy:
        1. Prioritize README, docs, architecture files
        2. Combine small files into larger documents
        3. Generate table of contents
        """
        notebook_dir = self.output_base / self._sanitize_name(notebook_name)
        notebook_dir.mkdir(parents=True, exist_ok=True)

        # Collect all files
        all_files = list(source_dir.rglob("*"))
        text_files = [f for f in all_files if f.is_file() and self._is_text_file(f)]

        # Prioritize files
        prioritized = self._prioritize_files(text_files)

        # Combine files into sources
        sources = self._create_sources(prioritized, notebook_dir, max_sources)

        # Generate manifest
        manifest = {
            "notebook_name": notebook_name,
            "created_at": datetime.now().isoformat(),
            "source_count": len(sources),
            "sources": sources,
            "upload_instructions": self._get_upload_instructions(notebook_dir)
        }

        manifest_path = notebook_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return manifest

    def _prioritize_files(self, files: list) -> list:
        """Sort files by priority."""
        priority_order = {
            "readme": 1,
            "architecture": 2,
            "design": 3,
            "contributing": 4,
            "changelog": 5,
            "api": 6,
            "index": 7,
        }

        def get_priority(path: Path) -> tuple:
            name_lower = path.stem.lower()
            # Check priority keywords
            for keyword, priority in priority_order.items():
                if keyword in name_lower:
                    return (priority, str(path))
            # Docs folder gets priority
            if "doc" in str(path).lower():
                return (10, str(path))
            # Code files last
            return (100, str(path))

        return sorted(files, key=get_priority)

    def _create_sources(
        self,
        files: list,
        output_dir: Path,
        max_sources: int
    ) -> list:
        """Create NotebookLM source files."""
        sources = []

        # Group 1: Individual important docs (README, ARCHITECTURE, etc.)
        important_docs = []
        code_files = []
        other_docs = []

        for f in files:
            name_lower = f.stem.lower()
            if any(k in name_lower for k in ["readme", "architecture", "design", "contributing"]):
                important_docs.append(f)
            elif f.suffix in [".py", ".js", ".ts", ".sh", ".sql"]:
                code_files.append(f)
            else:
                other_docs.append(f)

        # Add important docs as individual sources
        for doc in important_docs[:10]:  # Max 10 important docs
            source_name = f"{doc.stem}{doc.suffix}"
            dest_path = output_dir / source_name
            shutil.copy2(doc, dest_path)
            sources.append({
                "name": source_name,
                "path": str(dest_path),
                "type": "individual",
                "original": str(doc)
            })

        # Combine code files by directory
        remaining_slots = max_sources - len(sources)
        if code_files and remaining_slots > 0:
            code_sources = self._combine_files_by_directory(
                code_files, output_dir, "code", min(remaining_slots // 2, 20)
            )
            sources.extend(code_sources)

        # Combine other docs
        remaining_slots = max_sources - len(sources)
        if other_docs and remaining_slots > 0:
            doc_sources = self._combine_files_by_directory(
                other_docs, output_dir, "docs", min(remaining_slots, 15)
            )
            sources.extend(doc_sources)

        return sources

    def _combine_files_by_directory(
        self,
        files: list,
        output_dir: Path,
        prefix: str,
        max_combined: int
    ) -> list:
        """Combine files from same directory into single sources."""
        sources = []

        # Group by parent directory
        by_dir = {}
        for f in files:
            parent = f.parent.name or "root"
            if parent not in by_dir:
                by_dir[parent] = []
            by_dir[parent].append(f)

        # Create combined files
        for dir_name, dir_files in list(by_dir.items())[:max_combined]:
            combined_name = f"{prefix}_{self._sanitize_name(dir_name)}.md"
            combined_path = output_dir / combined_name

            content_parts = [f"# {prefix.title()} - {dir_name}\n\n"]
            content_parts.append("## Table of Contents\n\n")

            for f in dir_files:
                content_parts.append(f"- [{f.name}](#{self._sanitize_anchor(f.name)})\n")

            content_parts.append("\n---\n\n")

            for f in dir_files:
                content_parts.append(f"## {f.name} {{#{self._sanitize_anchor(f.name)}}}\n\n")
                content_parts.append(f"**Path:** `{f}`\n\n")
                content_parts.append("```" + self._get_language(f) + "\n")
                try:
                    content_parts.append(f.read_text(encoding="utf-8", errors="replace"))
                except Exception as e:
                    content_parts.append(f"[Error reading file: {e}]")
                content_parts.append("\n```\n\n---\n\n")

            combined_path.write_text("".join(content_parts), encoding="utf-8")

            sources.append({
                "name": combined_name,
                "path": str(combined_path),
                "type": "combined",
                "file_count": len(dir_files),
                "files": [str(f) for f in dir_files]
            })

        return sources

    def _sanitize_name(self, name: str) -> str:
        """Sanitize filename."""
        return re.sub(r'[^\w\-]', '_', name)

    def _sanitize_anchor(self, name: str) -> str:
        """Create markdown anchor from name."""
        return re.sub(r'[^\w\-]', '-', name.lower())

    def _get_language(self, path: Path) -> str:
        """Get language for syntax highlighting."""
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".sh": "bash",
            ".sql": "sql",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".html": "html",
            ".css": "css",
            ".md": "markdown",
            ".toml": "toml",
        }
        return lang_map.get(path.suffix, "")

    def _is_text_file(self, path: Path) -> bool:
        """Check if file is text-based."""
        text_extensions = {
            ".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
            ".sh", ".sql", ".html", ".css", ".rst", ".toml", ".ini", ".cfg",
            ".env.example", ".gitignore", ".dockerignore"
        }
        return path.suffix.lower() in text_extensions or path.name in {
            "Dockerfile", "Makefile", "LICENSE", "CODEOWNERS"
        }

    def _get_upload_instructions(self, notebook_dir: Path) -> str:
        """Generate upload instructions."""
        return f"""
NotebookLM Upload Instructions
==============================

1. Go to https://notebooklm.google.com/

2. Click "New Notebook" or open existing notebook

3. Click "Add source" button

4. For each file in: {notebook_dir}
   - Select "Upload file"
   - Choose the file
   - Wait for processing

5. Files are organized as:
   - Individual docs: README, ARCHITECTURE, etc.
   - Combined code files: code_*.md
   - Combined documentation: docs_*.md

6. After upload, you can:
   - Ask questions about the codebase
   - Generate summaries
   - Create study guides
   - Get audio overviews

Tips:
- Start with README and ARCHITECTURE files
- Add code files as needed for context
- NotebookLM works best with 10-20 focused sources
"""


class NotebookLMMCPClient:
    """
    Client for NotebookLM via MCP (Model Context Protocol).

    Uses browser automation to interact with NotebookLM since
    there's no official API.
    """

    def __init__(self, mcp_endpoint: Optional[str] = None):
        self.mcp_endpoint = mcp_endpoint or os.environ.get(
            "NOTEBOOKLM_MCP_ENDPOINT",
            "http://localhost:3000"
        )
        self.available = False

    async def check_availability(self) -> bool:
        """Check if MCP server is available."""
        if not HAS_AIOHTTP:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.mcp_endpoint}/health", timeout=5) as resp:
                    self.available = resp.status == 200
                    return self.available
        except Exception:
            return False

    async def create_notebook(self, name: str, description: str = "") -> Optional[str]:
        """Create a new notebook via MCP."""
        if not self.available:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                payload = {"name": name, "description": description}
                async with session.post(
                    f"{self.mcp_endpoint}/notebooks",
                    json=payload,
                    timeout=30
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("notebook_id")
        except Exception:
            pass
        return None

    async def upload_source(
        self,
        notebook_id: str,
        file_path: Path,
        source_name: str
    ) -> bool:
        """Upload a source file to notebook."""
        if not self.available:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                with open(file_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f, filename=source_name)
                    data.add_field('notebook_id', notebook_id)

                    async with session.post(
                        f"{self.mcp_endpoint}/sources",
                        data=data,
                        timeout=60
                    ) as resp:
                        return resp.status == 200
        except Exception:
            pass
        return False


async def sync_repos_to_notebooklm(
    repos: list[str],
    mode: str = "manual",
    github_token: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Main sync function.

    Args:
        repos: List of repo names (owner/repo format)
        mode: "auto" for MCP automation, "manual" for file preparation
        github_token: GitHub access token
        output_dir: Output directory for prepared files

    Returns:
        Summary of sync operation
    """
    results = {
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "repos": {},
        "notebooks": [],
        "errors": []
    }

    # Initialize components
    try:
        fetcher = GitHubFetcher(token=github_token)
    except (ImportError, ValueError) as e:
        results["errors"].append(str(e))
        return results

    preparer = NotebookLMPreparer(output_base=output_dir)
    mcp_client = NotebookLMMCPClient() if mode == "auto" else None

    # Check MCP availability for auto mode
    if mcp_client:
        mcp_available = await mcp_client.check_availability()
        if not mcp_available:
            print("MCP server not available, falling back to manual mode")
            mode = "manual"
            results["mode"] = "manual (fallback)"

    # Process each repo
    for repo_name in repos:
        print(f"\nProcessing: {repo_name}")

        # Create notebook config
        repo_short = repo_name.split("/")[-1]
        config = NotebookConfig(
            name=f"{repo_short.replace('-', ' ').title()}",
            repo=repo_name,
            description=f"Source code and documentation from {repo_name}"
        )

        # Create temp directory for repo files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Fetch from GitHub (fast archive method)
            print(f"  Fetching files from GitHub (archive download)...")
            report = fetcher.get_repo_contents_fast(repo_name, config, temp_path)

            if report.errors:
                results["errors"].extend(report.errors)

            print(f"  Processed: {report.files_processed} files ({report.total_size_kb:.1f} KB)")
            print(f"  Skipped: {report.files_skipped} files")

            # Prepare for NotebookLM
            print(f"  Preparing NotebookLM sources...")
            manifest = preparer.prepare_notebook_sources(config.name, temp_path)

            print(f"  Created: {manifest['source_count']} sources")

            # Store results
            results["repos"][repo_name] = {
                "files_processed": report.files_processed,
                "files_skipped": report.files_skipped,
                "size_kb": report.total_size_kb,
                "sources_created": manifest["source_count"],
                "errors": report.errors
            }

            results["notebooks"].append({
                "name": config.name,
                "repo": repo_name,
                "source_dir": str(preparer.output_base / preparer._sanitize_name(config.name)),
                "sources": manifest["sources"]
            })

            # Auto upload if MCP available
            if mode == "auto" and mcp_client and mcp_client.available:
                print(f"  Uploading to NotebookLM via MCP...")
                notebook_id = await mcp_client.create_notebook(
                    config.name,
                    config.description
                )

                if notebook_id:
                    for source in manifest["sources"]:
                        success = await mcp_client.upload_source(
                            notebook_id,
                            Path(source["path"]),
                            source["name"]
                        )
                        if not success:
                            results["errors"].append(
                                f"Failed to upload {source['name']} to {config.name}"
                            )

    # Print summary
    print("\n" + "=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)
    print(f"Mode: {results['mode']}")
    print(f"Repos processed: {len(results['repos'])}")

    for nb in results["notebooks"]:
        print(f"\nNotebook: {nb['name']}")
        print(f"  Sources: {len(nb['sources'])}")
        print(f"  Location: {nb['source_dir']}")

    if mode == "manual" or results["mode"] == "manual (fallback)":
        print("\n" + preparer._get_upload_instructions(preparer.output_base))

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"][:5]:
            print(f"  - {err}")
        if len(results["errors"]) > 5:
            print(f"  ... and {len(results['errors']) - 5} more")

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sync GitHub repos to NotebookLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Sync default Ironcliw repos (manual mode)
    python sync_to_notebooklm.py

    # Sync with auto upload via MCP
    python sync_to_notebooklm.py --mode auto

    # Sync custom repos
    python sync_to_notebooklm.py --repos owner/repo1,owner/repo2

    # Specify output directory
    python sync_to_notebooklm.py --output ~/my_notebooks
        """
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "manual"],
        default="manual",
        help="Upload mode: 'auto' uses MCP browser automation, 'manual' prepares files"
    )

    parser.add_argument(
        "--repos",
        type=str,
        default="drussell23/Ironcliw-AI-Agent,drussell23/jarvis-prime,drussell23/reactor-core",
        help="Comma-separated list of repos (owner/repo format)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for prepared files"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="GitHub token (or set GITHUB_TOKEN env var)"
    )

    args = parser.parse_args()

    # Parse repos
    repos = [r.strip() for r in args.repos.split(",")]

    # Check dependencies
    if not HAS_PYGITHUB:
        print("Error: PyGithub not installed")
        print("Run: pip install PyGithub")
        return 1

    # Run sync
    try:
        results = asyncio.run(
            sync_repos_to_notebooklm(
                repos=repos,
                mode=args.mode,
                github_token=args.token,
                output_dir=args.output
            )
        )

        # Save results
        output_base = args.output or Path.home() / "NotebookLM_Sources"
        results_path = output_base / "sync_results.json"
        results_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to: {results_path}")

        return 0 if not results["errors"] else 1

    except KeyboardInterrupt:
        print("\nSync cancelled")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
