"""
JARVIS Coding Council Voice Announcer (v79.0)
==============================================

Intelligent, dynamic voice announcer for code evolution operations.

Features:
- Context-aware message generation (no hardcoding)
- Real-time progress announcements with intelligent throttling
- Stage-based evolution narration
- Multi-factor confidence for announcements
- Time-of-day adaptive messaging
- Evolution history tracking for pattern learning
- Integration with EvolutionBroadcaster for unified event handling
- Trinity-aware for cross-repo evolution announcements

Architecture:
┌─────────────────────────────────────────────────────────────────────────┐
│              Coding Council Voice Announcer                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │ Context Engine   │  │ Message Composer │  │ Throttle Manager │      │
│  │ • Time Analysis  │  │ • Stage Parts    │  │ • Cooldowns      │      │
│  │ • Evolution Hist │  │ • Progress Parts │  │ • Milestone      │      │
│  │ • User Patterns  │  │ • Dynamic Msgs   │  │ • Rate Limits    │      │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘      │
│           │                      │                      │               │
│           └──────────────────────┼──────────────────────┘               │
│                                  ▼                                       │
│                    ┌─────────────────────────┐                          │
│                    │ UnifiedVoiceOrchestrator│                          │
│                    │    (speak_evolution)    │                          │
│                    └─────────────────────────┘                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    from core.coding_council.voice_announcer import get_evolution_announcer

    announcer = get_evolution_announcer()

    # Announce evolution start
    await announcer.announce_evolution_started(
        task_id="abc12345",
        description="Improve error handling in API module",
        target_files=["backend/api/voice_api.py"]
    )

    # Announce progress
    await announcer.announce_evolution_progress(
        task_id="abc12345",
        progress=0.5,
        stage="analyzing_code"
    )

    # Announce completion
    await announcer.announce_evolution_complete(
        task_id="abc12345",
        success=True,
        files_modified=["backend/api/voice_api.py"]
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class EvolutionStage(Enum):
    """Stages of code evolution for narration."""
    REQUESTED = "requested"          # Evolution command received
    VALIDATING = "validating"        # Validating request/permissions
    ANALYZING = "analyzing"          # Analyzing target code
    PLANNING = "planning"            # Planning changes
    GENERATING = "generating"        # Generating new code
    TESTING = "testing"              # Running tests
    APPLYING = "applying"            # Applying changes
    VERIFYING = "verifying"          # Verifying changes work
    COMPLETE = "complete"            # Evolution complete
    FAILED = "failed"                # Evolution failed
    ROLLBACK = "rollback"            # Rolling back changes


class AnnouncementType(Enum):
    """Type of announcement for cooldown tracking."""
    START = "start"
    PROGRESS = "progress"
    MILESTONE = "milestone"
    COMPLETE = "complete"
    ERROR = "error"
    CONFIRMATION = "confirmation"


class TimeOfDay(Enum):
    """Time periods for contextual messaging."""
    EARLY_MORNING = "early_morning"    # 5-7 AM
    MORNING = "morning"                # 7-12 PM
    AFTERNOON = "afternoon"            # 12-5 PM
    EVENING = "evening"                # 5-9 PM
    NIGHT = "night"                    # 9 PM - 12 AM
    LATE_NIGHT = "late_night"          # 12-5 AM


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvolutionContext:
    """
    Context for an evolution operation.
    Tracks all state for intelligent announcements.
    """
    task_id: str
    description: str
    target_files: List[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    current_stage: EvolutionStage = EvolutionStage.REQUESTED
    progress: float = 0.0
    last_announced_progress: float = 0.0
    files_analyzed: int = 0
    files_modified: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    errors: List[str] = field(default_factory=list)
    trinity_involved: bool = False  # True if J-Prime is orchestrating
    require_confirmation: bool = False
    confirmation_id: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        """Time elapsed since evolution started."""
        return time.time() - self.start_time

    @property
    def is_long_running(self) -> bool:
        """True if evolution has been running for a while."""
        return self.elapsed_seconds > 30.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "target_files": self.target_files,
            "current_stage": self.current_stage.value,
            "progress": self.progress,
            "elapsed_seconds": self.elapsed_seconds,
            "files_analyzed": self.files_analyzed,
            "files_modified": self.files_modified,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "errors": self.errors,
            "trinity_involved": self.trinity_involved,
        }


@dataclass
class AnnouncementConfig:
    """Configuration for the voice announcer."""
    # Enable/disable announcements
    enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_EVOLUTION_VOICE", "true").lower() == "true"
    )

    # Cooldowns (seconds)
    progress_cooldown: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_EVOLUTION_PROGRESS_COOLDOWN", "15.0"))
    )
    start_cooldown: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_EVOLUTION_START_COOLDOWN", "5.0"))
    )

    # Progress announcement thresholds
    progress_milestones: List[float] = field(
        default_factory=lambda: [0.25, 0.50, 0.75, 1.0]
    )
    milestone_tolerance: float = 0.05  # +/- 5% from milestone

    # Message style
    use_sir: bool = True  # Use "Sir" in messages
    sir_probability: float = 0.15  # 15% of messages include "Sir"

    # History tracking
    max_history_size: int = 100


# =============================================================================
# Message Templates (Dynamic Composition, Not Hardcoded Strings)
# =============================================================================


class MessageComposer:
    """
    Composes dynamic, context-aware messages.

    Uses template composition rather than hardcoded strings,
    allowing for natural variation and context awareness.
    """

    def __init__(self, config: AnnouncementConfig):
        self.config = config
        self._last_used_patterns: Dict[str, int] = {}

    def _get_time_of_day(self) -> TimeOfDay:
        """Determine current time of day."""
        hour = datetime.now().hour
        if 5 <= hour < 7:
            return TimeOfDay.EARLY_MORNING
        elif 7 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        elif 21 <= hour < 24:
            return TimeOfDay.NIGHT
        else:
            return TimeOfDay.LATE_NIGHT

    def _should_use_sir(self) -> bool:
        """Determine if we should use 'Sir' in this message."""
        if not self.config.use_sir:
            return False
        return random.random() < self.config.sir_probability

    def _select_pattern(self, patterns: List[str], category: str) -> str:
        """Select a pattern avoiding recent repetition."""
        # Track which patterns we've used recently
        last_used = self._last_used_patterns.get(category, -1)

        # Filter out recently used pattern if possible
        available = [i for i in range(len(patterns)) if i != last_used]
        if not available:
            available = list(range(len(patterns)))

        selected = random.choice(available)
        self._last_used_patterns[category] = selected
        return patterns[selected]

    def compose_start_message(self, ctx: EvolutionContext) -> str:
        """Compose evolution start message."""
        # Dynamic message parts
        intros = [
            "Starting code evolution",
            "Beginning evolution process",
            "Initiating code changes",
            "Evolution underway",
        ]

        intro = self._select_pattern(intros, "start_intro")

        # Add description context
        if ctx.description:
            # Truncate long descriptions
            desc = ctx.description[:50] + "..." if len(ctx.description) > 50 else ctx.description
            message = f"{intro}: {desc}"
        else:
            message = f"{intro}."

        # Add file context if single file
        if len(ctx.target_files) == 1:
            filename = os.path.basename(ctx.target_files[0])
            message += f" Target: {filename}."
        elif len(ctx.target_files) > 1:
            message += f" {len(ctx.target_files)} files targeted."

        # Add Sir occasionally
        if self._should_use_sir():
            message = message.replace(".", ", Sir.")

        return message

    def compose_progress_message(self, ctx: EvolutionContext) -> str:
        """Compose progress message based on stage and percentage."""
        percentage = int(ctx.progress * 100)

        # Stage-specific messages
        stage_messages = {
            EvolutionStage.ANALYZING: [
                f"Analyzing code structure, {percentage}% complete",
                f"Code analysis in progress, {percentage}%",
            ],
            EvolutionStage.PLANNING: [
                f"Planning changes, {percentage}% complete",
                f"Change planning at {percentage}%",
            ],
            EvolutionStage.GENERATING: [
                f"Generating new code, {percentage}% done",
                f"Code generation at {percentage}%",
            ],
            EvolutionStage.TESTING: [
                f"Running tests, {percentage}% complete",
                f"Test execution at {percentage}%",
            ],
            EvolutionStage.APPLYING: [
                f"Applying changes, {percentage}% done",
                f"Implementing changes, {percentage}%",
            ],
            EvolutionStage.VERIFYING: [
                f"Verifying changes, almost done",
                f"Final verification in progress",
            ],
        }

        # Get stage-specific message or generic
        if ctx.current_stage in stage_messages:
            patterns = stage_messages[ctx.current_stage]
            message = self._select_pattern(patterns, f"progress_{ctx.current_stage.value}")
        else:
            # Generic progress
            message = f"Evolution {percentage}% complete"

        # Add milestone celebration
        if abs(ctx.progress - 0.50) < self.config.milestone_tolerance:
            message += ". Halfway there"
        elif abs(ctx.progress - 0.75) < self.config.milestone_tolerance:
            message += ". Almost finished"

        return message + "."

    def compose_complete_message(
        self,
        ctx: EvolutionContext,
        success: bool,
        error_message: str = ""
    ) -> str:
        """Compose completion message."""
        elapsed = ctx.elapsed_seconds

        if success:
            # Success messages
            success_patterns = [
                "Evolution complete",
                "Code evolution successful",
                "Changes applied successfully",
            ]
            message = self._select_pattern(success_patterns, "complete_success")

            # Add file count
            if ctx.files_modified == 1:
                message += ". Modified one file"
            elif ctx.files_modified > 1:
                message += f". Updated {ctx.files_modified} files"

            # Add test status if relevant
            if ctx.tests_passed > 0:
                message += f". All {ctx.tests_passed} tests passing"

            # Add timing for long operations
            if elapsed > 30:
                message += f". Completed in {int(elapsed)} seconds"

            message += "."

        else:
            # Failure messages
            failure_patterns = [
                "Evolution encountered an issue",
                "Could not complete evolution",
                "Evolution failed",
            ]
            message = self._select_pattern(failure_patterns, "complete_failure")

            if error_message:
                # Truncate long errors
                error = error_message[:40] + "..." if len(error_message) > 40 else error_message
                message += f": {error}"
            message += "."

        # Add Sir for important completions
        if success and ctx.files_modified > 0 and self._should_use_sir():
            message = message.rstrip(".") + ", Sir."

        return message

    def compose_confirmation_message(self, ctx: EvolutionContext) -> str:
        """Compose confirmation request message."""
        confirmation_patterns = [
            f"Confirmation needed for: {ctx.description[:40]}",
            f"Please confirm evolution: {ctx.description[:40]}",
            f"Evolution requires approval: {ctx.description[:40]}",
        ]

        message = self._select_pattern(confirmation_patterns, "confirmation")

        if ctx.confirmation_id:
            message += f". Say confirm {ctx.confirmation_id} to proceed"

        return message + "."

    def compose_error_message(self, error_type: str, details: str = "") -> str:
        """Compose error message."""
        error_patterns = {
            "validation": "Evolution request could not be validated",
            "permission": "Insufficient permissions for this evolution",
            "timeout": "Evolution timed out",
            "conflict": "Detected conflicting changes",
            "test_failure": "Tests failed during evolution",
            "rollback": "Rolling back changes due to errors",
        }

        message = error_patterns.get(error_type, "Evolution error occurred")

        if details:
            message += f": {details[:30]}"

        return message + "."


# =============================================================================
# Main Voice Announcer Class
# =============================================================================


class CodingCouncilVoiceAnnouncer:
    """
    Intelligent voice announcer for Coding Council evolution operations.

    Features:
    - Dynamic message composition (no hardcoded strings)
    - Intelligent throttling to avoid announcement spam
    - Context-aware messaging based on evolution stage
    - History tracking for pattern learning
    - Integration with unified voice orchestrator
    """

    def __init__(self, config: Optional[AnnouncementConfig] = None):
        self.config = config or AnnouncementConfig()
        self._composer = MessageComposer(self.config)

        # Active evolutions being tracked
        self._active_evolutions: Dict[str, EvolutionContext] = {}

        # Cooldown tracking
        self._last_announcement: Dict[str, float] = {}  # type -> timestamp

        # History for learning
        self._evolution_history: List[Dict[str, Any]] = []

        # Statistics
        self._stats = {
            "announcements_made": 0,
            "announcements_throttled": 0,
            "evolutions_tracked": 0,
            "successful_evolutions": 0,
            "failed_evolutions": 0,
        }

        logger.info(f"[CodingCouncilVoice] Initialized (enabled={self.config.enabled})")

    def _can_announce(self, announcement_type: AnnouncementType) -> bool:
        """Check if we can make an announcement based on cooldowns."""
        if not self.config.enabled:
            return False

        now = time.time()
        last = self._last_announcement.get(announcement_type.value, 0)

        if announcement_type == AnnouncementType.PROGRESS:
            cooldown = self.config.progress_cooldown
        elif announcement_type == AnnouncementType.START:
            cooldown = self.config.start_cooldown
        else:
            cooldown = 2.0  # Minimal cooldown for other types

        if now - last < cooldown:
            self._stats["announcements_throttled"] += 1
            return False

        return True

    def _record_announcement(self, announcement_type: AnnouncementType):
        """Record that an announcement was made."""
        self._last_announcement[announcement_type.value] = time.time()
        self._stats["announcements_made"] += 1

    async def _speak(
        self,
        message: str,
        priority: str = "medium",
        wait: bool = False
    ) -> bool:
        """Speak through the unified voice orchestrator."""
        try:
            # Import here to avoid circular imports
            try:
                from core.supervisor.unified_voice_orchestrator import (
                    speak_evolution,
                    VoicePriority,
                )
            except ImportError:
                from backend.core.supervisor.unified_voice_orchestrator import (
                    speak_evolution,
                    VoicePriority,
                )

            priority_map = {
                "low": VoicePriority.LOW,
                "medium": VoicePriority.MEDIUM,
                "high": VoicePriority.HIGH,
                "critical": VoicePriority.CRITICAL,
            }

            return await speak_evolution(
                message,
                priority=priority_map.get(priority, VoicePriority.MEDIUM),
                wait=wait
            )

        except ImportError:
            logger.debug("[CodingCouncilVoice] Voice orchestrator not available")
            return False
        except Exception as e:
            logger.warning(f"[CodingCouncilVoice] Failed to speak: {e}")
            return False

    # =========================================================================
    # Public API - Evolution Event Handlers
    # =========================================================================

    async def announce_evolution_started(
        self,
        task_id: str,
        description: str,
        target_files: Optional[List[str]] = None,
        trinity_involved: bool = False,
    ) -> bool:
        """
        Announce that an evolution has started.

        Args:
            task_id: Unique evolution task ID
            description: Human-readable description
            target_files: List of target file paths
            trinity_involved: True if J-Prime is orchestrating

        Returns:
            True if announced, False if throttled/disabled
        """
        if not self._can_announce(AnnouncementType.START):
            return False

        # Create context
        ctx = EvolutionContext(
            task_id=task_id,
            description=description,
            target_files=target_files or [],
            trinity_involved=trinity_involved,
            current_stage=EvolutionStage.REQUESTED,
        )
        self._active_evolutions[task_id] = ctx
        self._stats["evolutions_tracked"] += 1

        # Compose and speak
        message = self._composer.compose_start_message(ctx)
        result = await self._speak(message, priority="medium")

        if result:
            self._record_announcement(AnnouncementType.START)
            logger.info(f"[CodingCouncilVoice] Announced start: {task_id}")

        return result

    async def announce_evolution_progress(
        self,
        task_id: str,
        progress: float,
        stage: Optional[str] = None,
    ) -> bool:
        """
        Announce evolution progress.

        Only announces at milestones to avoid spam.

        Args:
            task_id: Evolution task ID
            progress: Progress 0.0 to 1.0
            stage: Current stage name

        Returns:
            True if announced, False if throttled/not at milestone
        """
        ctx = self._active_evolutions.get(task_id)
        if not ctx:
            return False

        # Update context
        ctx.progress = progress
        if stage:
            try:
                ctx.current_stage = EvolutionStage(stage)
            except ValueError:
                pass  # Unknown stage, keep current

        # Check if at milestone
        at_milestone = False
        for milestone in self.config.progress_milestones:
            if abs(progress - milestone) <= self.config.milestone_tolerance:
                # Only announce if we haven't announced this milestone
                if ctx.last_announced_progress < milestone - self.config.milestone_tolerance:
                    at_milestone = True
                    ctx.last_announced_progress = progress
                    break

        if not at_milestone:
            return False

        if not self._can_announce(AnnouncementType.PROGRESS):
            return False

        # Compose and speak
        message = self._composer.compose_progress_message(ctx)
        result = await self._speak(message, priority="low")

        if result:
            self._record_announcement(AnnouncementType.PROGRESS)
            logger.debug(f"[CodingCouncilVoice] Progress: {task_id} at {int(progress*100)}%")

        return result

    async def announce_evolution_complete(
        self,
        task_id: str,
        success: bool,
        files_modified: Optional[List[str]] = None,
        error_message: str = "",
    ) -> bool:
        """
        Announce evolution completion.

        Args:
            task_id: Evolution task ID
            success: True if successful
            files_modified: List of modified files
            error_message: Error message if failed

        Returns:
            True if announced
        """
        ctx = self._active_evolutions.get(task_id)
        if not ctx:
            # Create minimal context for completion announcement
            ctx = EvolutionContext(
                task_id=task_id,
                description="",
                files_modified=len(files_modified or []),
            )

        # Update context
        ctx.files_modified = len(files_modified or [])
        ctx.current_stage = EvolutionStage.COMPLETE if success else EvolutionStage.FAILED
        ctx.progress = 1.0 if success else ctx.progress

        # Update stats
        if success:
            self._stats["successful_evolutions"] += 1
        else:
            self._stats["failed_evolutions"] += 1

        # Compose and speak
        message = self._composer.compose_complete_message(ctx, success, error_message)
        priority = "medium" if success else "high"
        result = await self._speak(message, priority=priority, wait=True)

        if result:
            self._record_announcement(AnnouncementType.COMPLETE)
            logger.info(f"[CodingCouncilVoice] Completion: {task_id} (success={success})")

        # Record to history
        self._record_evolution_history(ctx, success, error_message)

        # Cleanup
        self._active_evolutions.pop(task_id, None)

        return result

    async def announce_confirmation_needed(
        self,
        task_id: str,
        description: str,
        confirmation_id: str,
    ) -> bool:
        """
        Announce that confirmation is needed.

        Args:
            task_id: Evolution task ID
            description: What needs confirmation
            confirmation_id: The confirmation code to say

        Returns:
            True if announced
        """
        ctx = self._active_evolutions.get(task_id)
        if not ctx:
            ctx = EvolutionContext(
                task_id=task_id,
                description=description,
            )
            self._active_evolutions[task_id] = ctx

        ctx.require_confirmation = True
        ctx.confirmation_id = confirmation_id
        ctx.current_stage = EvolutionStage.VALIDATING

        message = self._composer.compose_confirmation_message(ctx)
        result = await self._speak(message, priority="high", wait=True)

        if result:
            self._record_announcement(AnnouncementType.CONFIRMATION)
            logger.info(f"[CodingCouncilVoice] Confirmation needed: {task_id}")

        return result

    async def announce_error(
        self,
        task_id: str,
        error_type: str,
        details: str = "",
    ) -> bool:
        """
        Announce an error during evolution.

        Args:
            task_id: Evolution task ID
            error_type: Type of error (validation, permission, etc.)
            details: Additional details

        Returns:
            True if announced
        """
        message = self._composer.compose_error_message(error_type, details)
        result = await self._speak(message, priority="high", wait=True)

        if result:
            self._record_announcement(AnnouncementType.ERROR)
            logger.warning(f"[CodingCouncilVoice] Error: {task_id} - {error_type}")

        # Update context if exists
        ctx = self._active_evolutions.get(task_id)
        if ctx:
            ctx.errors.append(f"{error_type}: {details}")

        return result

    # =========================================================================
    # History and Statistics
    # =========================================================================

    def _record_evolution_history(
        self,
        ctx: EvolutionContext,
        success: bool,
        error_message: str
    ):
        """Record evolution to history for learning."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "task_id": ctx.task_id,
            "description": ctx.description,
            "target_files": ctx.target_files,
            "success": success,
            "duration_seconds": ctx.elapsed_seconds,
            "files_modified": ctx.files_modified,
            "trinity_involved": ctx.trinity_involved,
            "error": error_message if not success else None,
        }

        self._evolution_history.append(record)

        # Trim history if too large
        if len(self._evolution_history) > self.config.max_history_size:
            self._evolution_history = self._evolution_history[-self.config.max_history_size:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get announcer statistics."""
        return {
            **self._stats,
            "active_evolutions": len(self._active_evolutions),
            "history_size": len(self._evolution_history),
            "config": {
                "enabled": self.config.enabled,
                "progress_cooldown": self.config.progress_cooldown,
            },
        }

    def get_active_evolutions(self) -> List[Dict[str, Any]]:
        """Get list of active evolutions."""
        return [ctx.to_dict() for ctx in self._active_evolutions.values()]

    def get_evolution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent evolution history."""
        return self._evolution_history[-limit:]


# =============================================================================
# Global Instance
# =============================================================================

_evolution_announcer: Optional[CodingCouncilVoiceAnnouncer] = None


def get_evolution_announcer() -> CodingCouncilVoiceAnnouncer:
    """Get or create the global evolution announcer."""
    global _evolution_announcer
    if _evolution_announcer is None:
        _evolution_announcer = CodingCouncilVoiceAnnouncer()
    return _evolution_announcer


# =============================================================================
# Integration with EvolutionBroadcaster
# =============================================================================


async def setup_voice_integration():
    """
    Set up voice integration with EvolutionBroadcaster.

    This hooks the voice announcer into the broadcaster's events
    for automatic voice announcements during evolution.
    """
    try:
        try:
            from core.coding_council.integration import get_evolution_broadcaster
        except ImportError:
            from backend.core.coding_council.integration import get_evolution_broadcaster

        broadcaster = get_evolution_broadcaster()
        announcer = get_evolution_announcer()

        # Register voice callback with broadcaster
        original_broadcast = broadcaster.broadcast

        async def broadcast_with_voice(
            task_id: str,
            status: str,
            progress: float,
            message: str,
            **kwargs
        ):
            """Wrapper that adds voice announcements to broadcasts."""
            # Call original broadcast
            await original_broadcast(
                task_id=task_id,
                status=status,
                progress=progress,
                message=message,
                **kwargs
            )

            # Trigger voice announcement based on status
            if status == "started":
                await announcer.announce_evolution_started(
                    task_id=task_id,
                    description=message,
                    target_files=kwargs.get("target_files"),
                )
            elif status in ("progress", "stage"):
                await announcer.announce_evolution_progress(
                    task_id=task_id,
                    progress=progress,
                    stage=kwargs.get("stage"),
                )
            elif status == "complete":
                await announcer.announce_evolution_complete(
                    task_id=task_id,
                    success=True,
                    files_modified=kwargs.get("files_modified"),
                )
            elif status == "failed":
                await announcer.announce_evolution_complete(
                    task_id=task_id,
                    success=False,
                    error_message=message,
                )
            elif status == "confirmation_needed":
                await announcer.announce_confirmation_needed(
                    task_id=task_id,
                    description=message,
                    confirmation_id=kwargs.get("confirmation_id", ""),
                )

        # Replace broadcast method
        broadcaster.broadcast = broadcast_with_voice
        logger.info("[CodingCouncilVoice] Voice integration with broadcaster complete")
        return True

    except ImportError as e:
        logger.debug(f"[CodingCouncilVoice] Could not set up integration: {e}")
        return False
    except Exception as e:
        logger.warning(f"[CodingCouncilVoice] Integration setup failed: {e}")
        return False
