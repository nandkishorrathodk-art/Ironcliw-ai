#!/usr/bin/env python3
"""
Fast App Launcher for Ironcliw
Optimized for quick app launching without complex routing.

Cross-platform: supports macOS (open -a / osascript) and
Windows (os.startfile / subprocess).
"""

import asyncio
import logging
import os
import subprocess
import sys
from typing import Tuple, Dict, Optional, Set

logger = logging.getLogger(__name__)


class FastAppLauncher:
    """Direct app launcher for common applications — cross-platform."""

    def __init__(self):
        if sys.platform == "win32":
            # Windows app mappings (executable names / start-menu names)
            self.common_apps: Dict[str, str] = {
                "chrome": "chrome",
                "edge": "msedge",
                "firefox": "firefox",
                "whatsapp": "WhatsApp",
                "spotify": "Spotify",
                "slack": "slack",
                "zoom": "zoom",
                "mail": "outlook",
                "messages": "ms-chat:",        # Windows Chat URI
                "explorer": "explorer",
                "notepad": "notepad",
                "notes": "ms-actioncenter:",   # Sticky Notes via URI
                "calculator": "calc",
                "calendar": "outlookcal:",
                "photos": "ms-photos:",
                "settings": "ms-settings:",
                "terminal": "wt",              # Windows Terminal
                "cmd": "cmd",
                "powershell": "powershell",
                "vscode": "code",
            }
        else:
            # macOS app mappings
            self.common_apps: Dict[str, str] = {
                "safari": "Safari",
                "chrome": "Google Chrome",
                "whatsapp": "WhatsApp",
                "spotify": "Spotify",
                "slack": "Slack",
                "zoom": "zoom.us",
                "mail": "Mail",
                "messages": "Messages",
                "finder": "Finder",
                "notes": "Notes",
                "calendar": "Calendar",
                "music": "Music",
                "photos": "Photos",
            }

        # Background task tracking to prevent GC collection
        self._background_tasks: Set[asyncio.Task] = set()

    async def quick_open_app(self, app_name: str) -> Tuple[bool, str]:
        """Quickly open app using direct system call — cross-platform."""
        # Normalize app name
        app_lower = app_name.lower().strip()

        # Check common apps first
        resolved_name = self.common_apps.get(app_lower, app_name)

        if sys.platform == "win32":
            return await self._open_windows(resolved_name, app_name)
        else:
            return await self._open_macos(resolved_name, app_name)

    # ──────────────────────────────────────────────────────────────────
    # Windows
    # ──────────────────────────────────────────────────────────────────

    async def _open_windows(self, resolved: str, display_name: str) -> Tuple[bool, str]:
        """Launch app on Windows via multiple strategies."""
        # Strategy 1: URI-scheme launch (ms-settings:, ms-photos:, etc.)
        if ":" in resolved:
            try:
                os.startfile(resolved)
                return True, f"Opened {display_name}"
            except Exception as e:
                logger.debug(f"URI launch failed for {resolved}: {e}")

        # Strategy 2: Direct executable (chrome, notepad, calc, etc.)
        try:
            process = await asyncio.create_subprocess_exec(
                resolved,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            # Don't wait — fire and background
            _task = asyncio.create_task(
                self._wait_for_process(process),
                name=f"fast-launcher-{display_name}",
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
            return True, f"Opened {display_name}"
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"Direct exec failed for {resolved}: {e}")

        # Strategy 3: 'start' command (handles Start Menu names)
        try:
            process = await asyncio.create_subprocess_exec(
                "cmd", "/c", "start", "", resolved,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=3.0)
            if process.returncode == 0:
                return True, f"Opened {display_name}"
        except asyncio.TimeoutError:
            return True, f"Opening {display_name}"
        except Exception as e:
            logger.debug(f"Start command failed for {resolved}: {e}")

        # Strategy 4: os.startfile as last resort
        try:
            os.startfile(resolved)
            return True, f"Opened {display_name}"
        except Exception as e:
            logger.warning(f"All launch methods failed for {display_name}: {e}")
            return False, f"Failed to open {display_name}: {e}"

    # ──────────────────────────────────────────────────────────────────
    # macOS
    # ──────────────────────────────────────────────────────────────────

    async def _open_macos(self, resolved: str, display_name: str) -> Tuple[bool, str]:
        """Launch app on macOS via 'open -a' with AppleScript fallback."""
        try:
            process = await asyncio.create_subprocess_exec(
                "open", "-a", resolved,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=1.0,
                )
                if process.returncode == 0:
                    return True, f"Opened {display_name}"
                else:
                    return await self._applescript_open(resolved, display_name)
            except asyncio.TimeoutError:
                return True, f"Opening {display_name}"
        except Exception as e:
            logger.error(f"Fast launch error: {e}")
            return await self._applescript_open(resolved, display_name)

    async def _applescript_open(
        self, app_name: str, display_name: str
    ) -> Tuple[bool, str]:
        """Fallback to AppleScript — macOS only."""
        script = f'tell application "{app_name}" to activate'
        try:
            process = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _task = asyncio.create_task(
                self._wait_for_process(process),
                name="fast-launcher-wait-process",
            )
            self._background_tasks.add(_task)
            _task.add_done_callback(self._background_tasks.discard)
            return True, f"Opening {display_name}"
        except Exception as e:
            return False, f"Failed to open {display_name}: {str(e)}"

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    async def _wait_for_process(self, process):
        """Wait for process completion in background."""
        try:
            await asyncio.wait_for(process.communicate(), timeout=2.0)
        except Exception:
            pass

    def is_common_app(self, app_name: str) -> bool:
        """Check if app is in common apps list."""
        return app_name.lower().strip() in self.common_apps


# Singleton instance
_fast_launcher = None


def get_fast_app_launcher() -> FastAppLauncher:
    """Get or create fast app launcher instance."""
    global _fast_launcher
    if _fast_launcher is None:
        _fast_launcher = FastAppLauncher()
    return _fast_launcher