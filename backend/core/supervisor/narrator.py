#!/usr/bin/env python3
"""
JARVIS Supervisor Voice Narrator
=================================

Lightweight TTS narrator for the supervisor to provide engaging voice
feedback during updates, restarts, and system events.

Uses macOS native 'say' command with Daniel (British) voice for immediate
playback without loading heavy TTS models.

Author: JARVIS System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class NarratorVoice(str, Enum):
    """Available voices for narration."""
    DANIEL = "Daniel"  # British English (default JARVIS voice)
    ALEX = "Alex"      # American English
    SAMANTHA = "Samantha"  # American English female
    KAREN = "Karen"    # Australian English
    MOIRA = "Moira"    # Irish English


class NarratorEvent(str, Enum):
    """Supervisor events that trigger narration."""
    SUPERVISOR_START = "supervisor_start"
    UPDATE_AVAILABLE = "update_available"
    UPDATE_STARTING = "update_starting"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    BUILDING = "building"
    VERIFYING = "verifying"
    UPDATE_COMPLETE = "update_complete"
    UPDATE_FAILED = "update_failed"
    ROLLBACK_STARTING = "rollback_starting"
    ROLLBACK_COMPLETE = "rollback_complete"
    RESTART_STARTING = "restart_starting"
    JARVIS_ONLINE = "jarvis_online"
    CRASH_DETECTED = "crash_detected"
    IDLE_UPDATE = "idle_update"


# Narration templates with variations for natural feel
NARRATION_TEMPLATES: dict[NarratorEvent, list[str]] = {
    NarratorEvent.SUPERVISOR_START: [
        "Lifecycle supervisor online. Initializing JARVIS core systems.",
        "Supervisor active. Bringing JARVIS systems online.",
    ],
    NarratorEvent.UPDATE_AVAILABLE: [
        "Sir, a system update is available. {summary}",
        "I've detected a new update. {summary}",
        "An update is ready for installation. {summary}",
    ],
    NarratorEvent.UPDATE_STARTING: [
        "Initiating update sequence. Please stand by.",
        "Beginning system update. This will only take a moment.",
        "Update sequence initiated. Standby for system refresh.",
    ],
    NarratorEvent.DOWNLOADING: [
        "Downloading updates from the repository.",
        "Fetching the latest changes now.",
        "Pulling updates. Almost there.",
    ],
    NarratorEvent.INSTALLING: [
        "Installing dependencies. This may take a moment.",
        "Updating system packages.",
        "Installing new components.",
    ],
    NarratorEvent.BUILDING: [
        "Rebuilding core systems.",
        "Compiling performance modules.",
        "Building optimized components.",
    ],
    NarratorEvent.VERIFYING: [
        "Verifying installation integrity.",
        "Running system verification checks.",
        "Confirming update success.",
    ],
    NarratorEvent.UPDATE_COMPLETE: [
        "Update complete. Systems nominal. {version}",
        "Successfully updated. All systems operational. {version}",
        "Update finished. Ready to assist. {version}",
    ],
    NarratorEvent.UPDATE_FAILED: [
        "Update encountered an error. Initiating recovery.",
        "The update failed. Reverting to stable version.",
        "I'm sorry, the update didn't complete. Rolling back now.",
    ],
    NarratorEvent.ROLLBACK_STARTING: [
        "Initiating rollback to previous stable version.",
        "Reverting to the last known good configuration.",
        "Rolling back. I'll have us back online shortly.",
    ],
    NarratorEvent.ROLLBACK_COMPLETE: [
        "Rollback complete. Previous version restored.",
        "Successfully reverted. Systems stable.",
        "Rollback finished. We're back to the stable version.",
    ],
    NarratorEvent.RESTART_STARTING: [
        "Restarting core systems. Back in a moment.",
        "System restart initiated. Please stand by.",
        "Restarting now. I'll be right back.",
    ],
    NarratorEvent.JARVIS_ONLINE: [
        "JARVIS online. All systems operational.",
        "Good to be back, Sir. How may I assist you?",
        "Systems restored. Ready when you are.",
    ],
    NarratorEvent.CRASH_DETECTED: [
        "I detected a system fault. Attempting recovery.",
        "An unexpected error occurred. Restarting now.",
        "Crash detected. Initiating recovery protocol.",
    ],
    NarratorEvent.IDLE_UPDATE: [
        "You've been away. I've updated myself while you were gone.",
        "I applied a system update during idle time. {summary}",
        "While you were away, I installed some improvements.",
    ],
}


@dataclass
class NarratorConfig:
    """Configuration for the narrator."""
    enabled: bool = True
    voice: NarratorVoice = NarratorVoice.DANIEL
    rate: int = 180  # Words per minute (default macOS is ~175)
    volume: float = 1.0  # 0.0 to 1.0
    async_playback: bool = True  # Don't block on speech


class SupervisorNarrator:
    """
    Voice narrator for supervisor events.
    
    Uses lightweight macOS 'say' command for immediate voice feedback
    without loading heavy TTS models. Falls back to silent logging
    on non-macOS systems.
    
    Example:
        >>> narrator = SupervisorNarrator()
        >>> await narrator.narrate(NarratorEvent.UPDATE_STARTING)
        >>> await narrator.narrate(NarratorEvent.UPDATE_COMPLETE, version="v1.2.3")
    """
    
    def __init__(self, config: Optional[NarratorConfig] = None):
        """
        Initialize the narrator.
        
        Args:
            config: Narrator configuration
        """
        self.config = config or NarratorConfig()
        self._is_macos = platform.system() == "Darwin"
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._speech_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        
        if self._is_macos:
            logger.info(f"🔊 Narrator initialized (voice: {self.config.voice.value})")
        else:
            logger.info("🔇 Narrator initialized (silent mode - non-macOS)")
    
    async def start(self) -> None:
        """Start the speech queue processor."""
        if self.config.async_playback and self._processor_task is None:
            self._processor_task = asyncio.create_task(self._process_queue())
    
    async def stop(self) -> None:
        """Stop the narrator and cancel pending speech."""
        if self._current_process:
            self._current_process.terminate()
            await self._current_process.wait()
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
    
    async def _process_queue(self) -> None:
        """Process queued speech in order."""
        while True:
            try:
                text = await self._speech_queue.get()
                await self._speak_sync(text)
                self._speech_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Speech queue error: {e}")
    
    async def _speak_sync(self, text: str) -> None:
        """Speak text synchronously using macOS say command."""
        if not self._is_macos or not self.config.enabled:
            logger.info(f"🔊 [WOULD SAY]: {text}")
            return
        
        try:
            # Build say command with voice and rate
            cmd = [
                "say",
                "-v", self.config.voice.value,
                "-r", str(self.config.rate),
                text,
            ]
            
            self._current_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            
            await self._current_process.wait()
            
        except Exception as e:
            logger.warning(f"TTS error: {e}")
        finally:
            self._current_process = None
    
    async def speak(self, text: str, wait: bool = False) -> None:
        """
        Speak arbitrary text.
        
        Args:
            text: Text to speak
            wait: If True, wait for speech to complete
        """
        if wait or not self.config.async_playback:
            await self._speak_sync(text)
        else:
            await self._speech_queue.put(text)
    
    async def narrate(
        self,
        event: NarratorEvent,
        wait: bool = False,
        **kwargs,
    ) -> None:
        """
        Narrate a supervisor event.
        
        Args:
            event: The event to narrate
            wait: If True, wait for speech to complete
            **kwargs: Template variables (e.g., summary, version)
        """
        templates = NARRATION_TEMPLATES.get(event, [])
        if not templates:
            logger.warning(f"No narration template for event: {event}")
            return
        
        # Pick a random template for variety
        template = random.choice(templates)
        
        # Format with provided variables
        try:
            text = template.format(**kwargs) if kwargs else template
            # Clean up any unfilled placeholders
            import re
            text = re.sub(r'\{[^}]+\}', '', text).strip()
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            text = template
        
        logger.info(f"🔊 Narrating: {text}")
        await self.speak(text, wait=wait)
    
    async def announce_update_progress(
        self,
        phase: str,
        detail: Optional[str] = None,
    ) -> None:
        """
        Announce update progress with optional detail.
        
        Args:
            phase: Current phase name
            detail: Optional detail message
        """
        # Map phase to event
        phase_map = {
            "fetching": NarratorEvent.DOWNLOADING,
            "downloading": NarratorEvent.DOWNLOADING,
            "installing": NarratorEvent.INSTALLING,
            "building": NarratorEvent.BUILDING,
            "verifying": NarratorEvent.VERIFYING,
        }
        
        event = phase_map.get(phase.lower())
        if event:
            await self.narrate(event)
        elif detail:
            await self.speak(detail)
    
    def set_voice(self, voice: NarratorVoice) -> None:
        """Change the narrator voice."""
        self.config.voice = voice
        logger.info(f"🔊 Voice changed to {voice.value}")
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable narration."""
        self.config.enabled = enabled


# Singleton instance
_narrator: Optional[SupervisorNarrator] = None


def get_narrator(config: Optional[NarratorConfig] = None) -> SupervisorNarrator:
    """Get singleton narrator instance."""
    global _narrator
    if _narrator is None:
        _narrator = SupervisorNarrator(config)
    return _narrator


async def narrate(event: NarratorEvent, **kwargs) -> None:
    """Quick utility to narrate an event."""
    narrator = get_narrator()
    await narrator.narrate(event, **kwargs)
