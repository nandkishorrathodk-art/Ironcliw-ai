"""
Ironcliw MCP (Model Context Protocol) Client Integration v1.0
============================================================
Provides Ironcliw with access to MCP servers for:
- Filesystem access (read/write files)
- GitHub repo management
- Web fetching
- Persistent memory
- Browser automation (Puppeteer)
- Brave Search

Usage:
    from backend.mcp_integration.mcp_client import MCPClientManager
    mcp = MCPClientManager()
    await mcp.start()
"""
import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("jarvis.mcp_client")

# ─────────────────────────────────────────────────────────────
# MCP Server Configuration
# ─────────────────────────────────────────────────────────────
Ironcliw_ROOT = Path(__file__).parent.parent.parent
Ironcliw_DATA_DIR = Path.home() / ".jarvis"
Ironcliw_DATA_DIR.mkdir(parents=True, exist_ok=True)

MCP_SERVERS: Dict[str, Dict] = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(Ironcliw_ROOT),
            str(Ironcliw_DATA_DIR),
            str(Path.home() / "Documents"),
        ],
        "description": "Read/write files, list directories, search files",
        "enabled": True,
    },
    "fetch": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "description": "Fetch web pages and APIs",
        "enabled": True,
    },
    "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"],
        "description": "Persistent key-value memory across sessions",
        "enabled": True,
    },
    "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env_required": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
        "description": "Manage GitHub repos, PRs, issues",
        "enabled": True,
    },
    "puppeteer": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
        "description": "Browser automation for bug bounty and web tasks",
        "enabled": True,
    },
    "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "env_required": ["BRAVE_API_KEY"],
        "description": "Search the web with Brave Search",
        "enabled": bool(os.getenv("BRAVE_API_KEY")),
    },
}


class MCPServer:
    """Manages a single MCP server process."""

    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.ready = False

    def is_enabled(self) -> bool:
        if not self.config.get("enabled", True):
            return False
        for env_key in self.config.get("env_required", []):
            if not os.getenv(env_key):
                logger.warning(
                    f"[MCP:{self.name}] Skipped — missing env var: {env_key}"
                )
                return False
        return True

    def start(self) -> bool:
        if not self.is_enabled():
            return False
        try:
            env = os.environ.copy()
            env_overrides = self.config.get("env", {})
            for k, v in env_overrides.items():
                resolved = os.path.expandvars(v)
                env[k] = resolved

            cmd = [self.config["command"]] + self.config["args"]
            logger.info(f"[MCP:{self.name}] Starting: {' '.join(cmd)}")

            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.ready = True
            logger.info(f"[MCP:{self.name}] ✅ Started (PID: {self.process.pid})")
            return True
        except Exception as e:
            logger.error(f"[MCP:{self.name}] ❌ Failed to start: {e}")
            return False

    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info(f"[MCP:{self.name}] Stopped.")
            except Exception as e:
                logger.warning(f"[MCP:{self.name}] Force kill: {e}")
                self.process.kill()
        self.ready = False


class MCPClientManager:
    """
    Manages all Ironcliw MCP server connections.

    Use this singleton to get access to MCP tools from anywhere in Ironcliw.
    """

    _instance: Optional["MCPClientManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.servers: Dict[str, MCPServer] = {}
        self._started = False

    def start(self) -> Dict[str, bool]:
        """Start all enabled MCP servers."""
        if self._started:
            return {name: s.ready for name, s in self.servers.items()}

        results = {}
        for name, config in MCP_SERVERS.items():
            server = MCPServer(name, config)
            self.servers[name] = server
            results[name] = server.start()

        self._started = True
        ready_count = sum(1 for r in results.values() if r)
        logger.info(
            f"✅ MCP Client Manager started: {ready_count}/{len(MCP_SERVERS)} servers ready"
        )
        return results

    def stop(self):
        """Stop all MCP servers."""
        for server in self.servers.values():
            server.stop()
        self._started = False
        logger.info("🛑 MCP Client Manager stopped.")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all MCP servers."""
        return {
            name: {
                "ready": server.ready,
                "pid": server.process.pid if server.process else None,
                "description": server.config.get("description", ""),
            }
            for name, server in self.servers.items()
        }

    def print_status(self):
        """Print a human-readable status report."""
        print("\n" + "=" * 50)
        print("🔌 Ironcliw MCP Server Status")
        print("=" * 50)
        for name, server in self.servers.items():
            icon = "✅" if server.ready else "❌"
            desc = server.config.get("description", "")
            print(f"  {icon} {name:<15} — {desc}")
        print("=" * 50 + "\n")


# Module-level singleton
_mcp_manager: Optional[MCPClientManager] = None


def get_mcp_manager() -> MCPClientManager:
    """Get or create the global MCP manager singleton."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager()
    return _mcp_manager


def start_mcp_servers() -> Dict[str, bool]:
    """Convenience function to start all MCP servers."""
    manager = get_mcp_manager()
    return manager.start()
