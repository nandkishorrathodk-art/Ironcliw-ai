#!/usr/bin/env python3
"""
JARVIS Update Intent Handler
==============================

Handles the "update system" voice command by speaking a confirmation
and triggering sys.exit(100) to signal the supervisor to update.

This module integrates with the JARVIS event system to register handlers
for update-related intents without modifying the main start_system.py.

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Exit codes for supervisor communication
EXIT_CODE_UPDATE_REQUEST = 100
EXIT_CODE_ROLLBACK_REQUEST = 101
EXIT_CODE_RESTART_REQUEST = 102


class UpdateIntentHandler:
    """
    Handler for update and rollback voice commands.
    
    Integrates with JARVIS event system to handle:
    - "update yourself" / "check for updates"
    - "rollback" / "revert update"
    
    When triggered, announces the action and exits with
    the appropriate code for the supervisor to handle.
    
    Example:
        >>> handler = UpdateIntentHandler(speak_func=my_tts_function)
        >>> await handler.handle_update_request()  # Speaks and exits with 100
    """
    
    def __init__(
        self,
        speak_func: Optional[Callable[[str], Any]] = None,
        confirmation_enabled: bool = True,
    ):
        """
        Initialize the update intent handler.
        
        Args:
            speak_func: Async function to speak text (e.g., TTS system)
            confirmation_enabled: Whether to require voice confirmation
        """
        self._speak_func = speak_func
        self._confirmation_enabled = confirmation_enabled
        self._update_pending = False
        self._registered = False
        
        logger.info("🔧 Update intent handler initialized")
    
    def set_speak_function(self, func: Callable[[str], Any]) -> None:
        """Set the TTS speak function."""
        self._speak_func = func
    
    async def _speak(self, text: str) -> None:
        """Speak text using the configured TTS function."""
        if self._speak_func:
            try:
                result = self._speak_func(text)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"TTS error: {e}")
        else:
            # Fallback to macOS say command
            logger.info(f"🔊 [ANNOUNCE]: {text}")
            try:
                proc = await asyncio.create_subprocess_exec(
                    "say", "-v", "Daniel", text,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            except Exception as e:
                logger.debug(f"Fallback TTS failed: {e}")
    
    async def handle_update_request(
        self,
        summary: Optional[str] = None,
        immediate: bool = False,
    ) -> None:
        """
        Handle the "update system" voice command.
        
        Speaks a confirmation and exits with code 100 to signal
        the supervisor to perform an update.
        
        Args:
            summary: Optional changelog summary to announce
            immediate: Skip confirmation if True
        """
        logger.info("🔄 Update request received")
        
        # Announce update
        if summary:
            await self._speak(
                f"I've found an update. {summary}. "
                "Initiating update sequence now. I'll be back shortly."
            )
        else:
            await self._speak(
                "Understood, Sir. Initiating update sequence. "
                "I'll be back online shortly."
            )
        
        # Give TTS time to finish
        await asyncio.sleep(0.5)
        
        # Log and exit with update code
        logger.info(f"📤 Exiting with code {EXIT_CODE_UPDATE_REQUEST} for supervisor")
        os._exit(EXIT_CODE_UPDATE_REQUEST)
    
    async def handle_rollback_request(self) -> None:
        """
        Handle the "rollback" voice command.
        
        Speaks a confirmation and exits with code 101 to signal
        the supervisor to perform a rollback.
        """
        logger.info("🔄 Rollback request received")
        
        await self._speak(
            "Understood. Rolling back to the previous stable version. "
            "Standby for system restart."
        )
        
        await asyncio.sleep(0.5)
        
        logger.info(f"📤 Exiting with code {EXIT_CODE_ROLLBACK_REQUEST} for supervisor")
        os._exit(EXIT_CODE_ROLLBACK_REQUEST)
    
    async def handle_restart_request(self) -> None:
        """
        Handle a restart request without update.
        """
        logger.info("🔄 Restart request received")
        
        await self._speak("Restarting core systems. Back in a moment.")
        
        await asyncio.sleep(0.5)
        
        logger.info(f"📤 Exiting with code {EXIT_CODE_RESTART_REQUEST} for supervisor")
        os._exit(EXIT_CODE_RESTART_REQUEST)
    
    async def check_for_updates(self) -> dict[str, Any]:
        """
        Check for available updates and return info.
        
        Returns:
            Dict with update information
        """
        try:
            from backend.core.supervisor.update_detector import UpdateDetector
            
            detector = UpdateDetector()
            update_info = await detector.check_for_updates()
            
            if update_info and update_info.available:
                return {
                    "available": True,
                    "commits_behind": update_info.commits_behind,
                    "summary": update_info.summary,
                    "current": update_info.current_version,
                    "latest": update_info.remote_version,
                }
            
            return {"available": False}
            
        except ImportError:
            logger.warning("Update detector not available")
            return {"available": False, "error": "Update detector not installed"}
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return {"available": False, "error": str(e)}
    
    async def announce_update_status(self) -> None:
        """Check for updates and announce if available."""
        status = await self.check_for_updates()
        
        if status.get("available"):
            await self._speak(
                f"An update is available with {status.get('summary', 'new changes')}. "
                "Would you like me to update now?"
            )
        else:
            await self._speak("You're running the latest version. No updates available.")
    
    def register_with_event_coordinator(
        self,
        coordinator: Any,
    ) -> None:
        """
        Register handlers with the JARVIS event coordinator.
        
        Args:
            coordinator: JARVISEventCoordinator instance
        """
        if self._registered:
            return
        
        try:
            # Subscribe to update intent events
            coordinator.event_bus.subscribe(
                "intent.update_system",
                lambda _: asyncio.create_task(self.handle_update_request()),
            )
            
            coordinator.event_bus.subscribe(
                "intent.rollback_system",
                lambda _: asyncio.create_task(self.handle_rollback_request()),
            )
            
            self._registered = True
            logger.info("✅ Update intent handlers registered with event coordinator")
            
        except Exception as e:
            logger.error(f"Failed to register event handlers: {e}")


# Singleton instance
_handler: Optional[UpdateIntentHandler] = None


def get_update_handler() -> UpdateIntentHandler:
    """Get singleton update intent handler."""
    global _handler
    if _handler is None:
        _handler = UpdateIntentHandler()
    return _handler


async def handle_update_command(speak_func: Optional[Callable] = None) -> None:
    """
    Convenience function to handle update command.
    
    Called from voice processing when "update yourself" is detected.
    """
    handler = get_update_handler()
    if speak_func:
        handler.set_speak_function(speak_func)
    await handler.handle_update_request()


async def handle_rollback_command(speak_func: Optional[Callable] = None) -> None:
    """
    Convenience function to handle rollback command.
    
    Called from voice processing when "rollback" is detected.
    """
    handler = get_update_handler()
    if speak_func:
        handler.set_speak_function(speak_func)
    await handler.handle_rollback_request()


def is_supervised() -> bool:
    """Check if JARVIS is running under the supervisor."""
    return os.environ.get("JARVIS_SUPERVISED") == "1"


def get_supervisor_pid() -> Optional[int]:
    """Get the supervisor's PID if running supervised."""
    pid_str = os.environ.get("JARVIS_SUPERVISOR_PID")
    if pid_str:
        try:
            return int(pid_str)
        except ValueError:
            pass
    return None
