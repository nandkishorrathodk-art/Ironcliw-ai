#!/usr/bin/env python3
"""
Ironcliw MCP Setup Script
=======================
Installs all required MCP server npm packages and deploys the
claude_desktop_config.json to the correct location.

Usage:
    python setup_mcp.py
"""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

Ironcliw_ROOT = Path(__file__).parent

MCP_PACKAGES = [
    "@modelcontextprotocol/server-filesystem",
    "@modelcontextprotocol/server-fetch",
    "@modelcontextprotocol/server-memory",
    "@modelcontextprotocol/server-github",
    "@modelcontextprotocol/server-puppeteer",
    "@modelcontextprotocol/server-brave-search",
]


def check_node():
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        print(f"✅ Node.js: {result.stdout.strip()}")
        result = subprocess.run(["npx", "--version"], capture_output=True, text=True)
        print(f"✅ npx: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Node.js not found. Install from https://nodejs.org")
        return False


def pre_install_packages():
    """Pre-install MCP packages so first startup is fast."""
    print("\n📦 Pre-installing MCP server packages...")
    for pkg in MCP_PACKAGES:
        print(f"   → Installing {pkg}...", end=" ", flush=True)
        result = subprocess.run(
            ["npm", "install", "-g", pkg],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅")
        else:
            print(f"⚠️ (will install on first use via npx)")


def deploy_claude_desktop_config():
    """Deploy the MCP config to Claude Desktop if installed."""
    config_src = Ironcliw_ROOT / "mcp_config" / "claude_desktop_config.json"

    if platform.system() == "Windows":
        config_dest = (
            Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
        )
    elif platform.system() == "Darwin":
        config_dest = (
            Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        )
    else:
        config_dest = Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

    if not config_dest.parent.exists():
        print(f"\n⚠️  Claude Desktop not installed (config dir not found).")
        print(f"   When you install Claude Desktop, copy this config:")
        print(f"   → Source: {config_src}")
        print(f"   → Destination: {config_dest}")
    else:
        config_dest.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing config if present
        if config_dest.exists():
            with open(config_dest) as f:
                existing = json.load(f)
            with open(config_src) as f:
                jarvis_config = json.load(f)

            if "mcpServers" not in existing:
                existing["mcpServers"] = {}
            existing["mcpServers"].update(jarvis_config.get("mcpServers", {}))
            with open(config_dest, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"✅ Merged MCP config into existing Claude Desktop config.")
        else:
            shutil.copy(config_src, config_dest)
            print(f"✅ Claude Desktop MCP config deployed to: {config_dest}")


def save_env_template():
    """Append MCP env vars to .env if not already present."""
    env_file = Ironcliw_ROOT / ".env"
    env_additions = """
# ═══════════════════════════════════════
# MCP Server Environment Variables
# ═══════════════════════════════════════
# GitHub MCP: create token at https://github.com/settings/tokens
# GITHUB_PERSONAL_ACCESS_TOKEN=your_token_here

# Brave Search MCP: get key at https://api.search.brave.com/
# BRAVE_API_KEY=your_key_here
"""
    if env_file.exists():
        content = env_file.read_text(encoding="utf-8")
        if "MCP Server Environment Variables" not in content:
            with open(env_file, "a", encoding="utf-8") as f:
                f.write(env_additions)
            print(f"✅ MCP env vars added to .env")
        else:
            print(f"ℹ️  MCP env vars already in .env")
    else:
        print(f"⚠️  .env not found, skipping env template.")


def main():
    print("=" * 50)
    print("🔌 Ironcliw MCP Setup v1.0")
    print("=" * 50)

    if not check_node():
        sys.exit(1)

    pre_install_packages()
    deploy_claude_desktop_config()
    save_env_template()

    print("\n✅ MCP setup complete!")
    print("\n📋 Next steps:")
    print("  1. Add your GITHUB_PERSONAL_ACCESS_TOKEN to .env")
    print("  2. Add your BRAVE_API_KEY to .env (optional)")
    print("  3. Ironcliw will auto-start MCP servers on launch")
    print("  4. If you install Claude Desktop, the config is already ready!")


if __name__ == "__main__":
    main()
