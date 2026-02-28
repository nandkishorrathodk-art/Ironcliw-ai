#!/usr/bin/env python3
"""
Ironcliw Supervisor Voice Narrator v4.0 - Intelligent Speech Edition
===================================================================

Intelligent, context-aware TTS narrator for the supervisor to provide engaging
voice feedback during updates, restarts, and system events.

v4.0 Features:
- Intelligent speech context tracking (prevents repetition)
- Topic-based cooldowns (avoid repeating same topic too soon)
- Semantic deduplication (skip similar messages)
- Natural pacing (pauses between rapid-fire messages)
- Message coalescing (batch similar messages into summaries)
- Adaptive verbosity (reduce chatter during long operations)

v3.0 Features:
- Zero-Touch autonomous update narration
- Context-aware dynamic message generation
- Update classification-based announcements
- Prime Directives violation alerts
- Dead Man's Switch status narration
- Intelligent message prioritization
- Parallel event processing
- Adaptive narration based on system state

v2.0 CHANGE: Now delegates to UnifiedVoiceOrchestrator instead of spawning
its own `say` processes. This prevents the "multiple voices" issue where
concurrent narrator systems would speak simultaneously.

Author: Ironcliw System
Version: 4.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Import unified voice orchestrator (single source of truth for all voice)
from .unified_voice_orchestrator import (
    get_voice_orchestrator,
    VoicePriority,
    VoiceSource,
    SpeechTopic,
)

logger = logging.getLogger(__name__)


class NarratorVoice(str, Enum):
    """Available voices for narration."""
    DANIEL = "Daniel"  # British English (default Ironcliw voice)
    ALEX = "Alex"      # American English
    SAMANTHA = "Samantha"  # American English female
    KAREN = "Karen"    # Australian English
    MOIRA = "Moira"    # Irish English


class NarratorEvent(str, Enum):
    """Supervisor events that trigger narration."""
    # Lifecycle events
    SUPERVISOR_START = "supervisor_start"
    Ironcliw_ONLINE = "jarvis_online"
    RESTART_STARTING = "restart_starting"
    CRASH_DETECTED = "crash_detected"
    
    # Update events
    UPDATE_AVAILABLE = "update_available"
    UPDATE_STARTING = "update_starting"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    BUILDING = "building"
    VERIFYING = "verifying"
    UPDATE_COMPLETE = "update_complete"
    UPDATE_FAILED = "update_failed"
    IDLE_UPDATE = "idle_update"
    
    # Rollback events
    ROLLBACK_STARTING = "rollback_starting"
    ROLLBACK_COMPLETE = "rollback_complete"
    
    # Startup phase events (v19.6.0)
    STARTUP_CLEANUP = "startup_cleanup"
    STARTUP_SPAWNING = "startup_spawning"
    STARTUP_BACKEND = "startup_backend"
    STARTUP_DATABASE = "startup_database"
    STARTUP_DOCKER = "startup_docker"
    STARTUP_DOCKER_SLOW = "startup_docker_slow"
    STARTUP_MODELS = "startup_models"
    STARTUP_MODELS_SLOW = "startup_models_slow"
    STARTUP_VOICE = "startup_voice"
    STARTUP_VISION = "startup_vision"
    STARTUP_FRONTEND = "startup_frontend"
    STARTUP_PROGRESS_25 = "startup_progress_25"
    STARTUP_PROGRESS_50 = "startup_progress_50"
    STARTUP_PROGRESS_75 = "startup_progress_75"
    STARTUP_SLOW = "startup_slow"
    STARTUP_ERROR = "startup_error"
    STARTUP_RECOVERY = "startup_recovery"

    # Local change awareness events (v2.0)
    LOCAL_COMMIT_DETECTED = "local_commit_detected"
    LOCAL_PUSH_DETECTED = "local_push_detected"
    RESTART_RECOMMENDED = "restart_recommended"
    CODE_CHANGES_DETECTED = "code_changes_detected"
    
    # =========================================================================
    # v3.0: Zero-Touch Autonomous Update Events
    # =========================================================================
    
    # Zero-Touch Pre-Flight
    ZERO_TOUCH_INITIATED = "zero_touch_initiated"
    ZERO_TOUCH_PRE_FLIGHT = "zero_touch_pre_flight"
    ZERO_TOUCH_PRE_FLIGHT_PASSED = "zero_touch_pre_flight_passed"
    ZERO_TOUCH_PRE_FLIGHT_FAILED = "zero_touch_pre_flight_failed"
    ZERO_TOUCH_BLOCKED = "zero_touch_blocked"
    
    # Zero-Touch Staging & Validation
    ZERO_TOUCH_STAGING = "zero_touch_staging"
    ZERO_TOUCH_VALIDATING = "zero_touch_validating"
    ZERO_TOUCH_VALIDATION_PASSED = "zero_touch_validation_passed"
    ZERO_TOUCH_VALIDATION_FAILED = "zero_touch_validation_failed"
    
    # Zero-Touch Update Classification
    ZERO_TOUCH_SECURITY_UPDATE = "zero_touch_security_update"
    ZERO_TOUCH_CRITICAL_UPDATE = "zero_touch_critical_update"
    ZERO_TOUCH_MINOR_UPDATE = "zero_touch_minor_update"
    ZERO_TOUCH_MAJOR_BLOCKED = "zero_touch_major_blocked"
    
    # Zero-Touch Update Execution
    ZERO_TOUCH_APPLYING = "zero_touch_applying"
    ZERO_TOUCH_COMPLETE = "zero_touch_complete"
    ZERO_TOUCH_FAILED = "zero_touch_failed"
    ZERO_TOUCH_COOLDOWN = "zero_touch_cooldown"
    
    # Dead Man's Switch Events
    DMS_PROBATION_START = "dms_probation_start"
    DMS_HEARTBEAT_OK = "dms_heartbeat_ok"
    DMS_HEARTBEAT_FAILED = "dms_heartbeat_failed"
    DMS_PROBATION_PASSED = "dms_probation_passed"
    DMS_ROLLBACK_TRIGGERED = "dms_rollback_triggered"
    DMS_STABLE_COMMITTED = "dms_stable_committed"
    
    # Prime Directives Events
    PRIME_DIRECTIVE_ENFORCED = "prime_directive_enforced"
    PROTECTED_FILE_BLOCKED = "protected_file_blocked"
    IMMUTABLE_CORE_VIOLATION = "immutable_core_violation"
    CONSENT_REQUIRED = "consent_required"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    
    # Busy State Events
    Ironcliw_BUSY_DETECTED = "jarvis_busy_detected"
    Ironcliw_IDLE_DETECTED = "jarvis_idle_detected"
    UPDATE_DEFERRED = "update_deferred"


# Narration templates with variations for natural feel
NARRATION_TEMPLATES: dict[NarratorEvent, list[str]] = {
    # Lifecycle events
    NarratorEvent.SUPERVISOR_START: [
        "Lifecycle supervisor online. Initializing Ironcliw core systems.",
        "Supervisor active. Bringing Ironcliw systems online.",
    ],
    NarratorEvent.Ironcliw_ONLINE: [
        "Ironcliw online. All systems operational.",
        "Good to be back, Sir. How may I assist you?",
        "Systems restored. Ready when you are.",
    ],
    NarratorEvent.RESTART_STARTING: [
        "Restarting core systems. Back in a moment.",
        "System restart initiated. Please stand by.",
        "Restarting now. I'll be right back.",
    ],
    NarratorEvent.CRASH_DETECTED: [
        "I detected a system fault. Attempting recovery.",
        "An unexpected error occurred. Restarting now.",
        "Crash detected. Initiating recovery protocol.",
    ],
    
    # Update events
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
    NarratorEvent.IDLE_UPDATE: [
        "You've been away. I've updated myself while you were gone.",
        "I applied a system update during idle time. {summary}",
        "While you were away, I installed some improvements.",
    ],
    
    # Rollback events
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
    
    # Startup phase events (v19.6.0)
    NarratorEvent.STARTUP_CLEANUP: [
        "Cleaning up previous sessions.",
        "Preparing a fresh workspace.",
    ],
    NarratorEvent.STARTUP_SPAWNING: [
        "Spawning Ironcliw core process.",
        "Launching main system.",
    ],
    NarratorEvent.STARTUP_BACKEND: [
        "Initializing backend services.",
        "Backend is coming online.",
    ],
    NarratorEvent.STARTUP_DATABASE: [
        "Connecting to databases.",
        "Establishing data connections.",
    ],
    NarratorEvent.STARTUP_DOCKER: [
        "Initializing Docker environment.",
        "Starting container services.",
    ],
    NarratorEvent.STARTUP_DOCKER_SLOW: [
        "Docker is taking a moment. Please stand by.",
        "Waiting for Docker daemon. This may take a minute.",
    ],
    NarratorEvent.STARTUP_MODELS: [
        "Loading machine learning models.",
        "Initializing neural networks.",
    ],
    NarratorEvent.STARTUP_MODELS_SLOW: [
        "Loading models. This is the heavy lifting.",
        "Neural networks are warming up.",
    ],
    NarratorEvent.STARTUP_VOICE: [
        "Initializing voice systems.",
        "Calibrating speech recognition.",
    ],
    NarratorEvent.STARTUP_VISION: [
        "Calibrating vision systems.",
        "Initializing visual processing.",
    ],
    NarratorEvent.STARTUP_FRONTEND: [
        "Connecting to user interface.",
        "Frontend is coming online.",
    ],
    NarratorEvent.STARTUP_PROGRESS_25: [
        "About a quarter of the way through.",
        "25 percent loaded.",
    ],
    NarratorEvent.STARTUP_PROGRESS_50: [
        "Halfway there.",
        "50 percent complete.",
    ],
    NarratorEvent.STARTUP_PROGRESS_75: [
        "Almost ready. Just a few more moments.",
        "75 percent. Nearly done.",
    ],
    NarratorEvent.STARTUP_SLOW: [
        "Taking a bit longer than usual. Everything is fine.",
        "Still working on it. Thank you for your patience.",
    ],
    NarratorEvent.STARTUP_ERROR: [
        "I've encountered a problem during startup.",
        "Something went wrong. Attempting recovery.",
    ],
    NarratorEvent.STARTUP_RECOVERY: [
        "Initiating recovery sequence.",
        "Attempting to recover from failure.",
    ],

    # Local change awareness events (v2.0)
    NarratorEvent.LOCAL_COMMIT_DETECTED: [
        "I notice you've made a new commit. {summary}",
        "A new commit has been detected. {summary}",
        "You've been busy! I see {summary}.",
    ],
    NarratorEvent.LOCAL_PUSH_DETECTED: [
        "I see you've pushed code to the repository. {summary}",
        "Nice! Your changes have been pushed. {summary}",
        "Your code is now on the remote. {summary}",
    ],
    NarratorEvent.RESTART_RECOMMENDED: [
        "Sir, I recommend a restart to apply your changes. {reason}",
        "A restart would pick up your recent modifications. {reason}",
        "Would you like me to restart? {reason}",
    ],
    NarratorEvent.CODE_CHANGES_DETECTED: [
        "I've detected code changes since I started. {summary}",
        "Some files have been modified. {summary}",
        "There are uncommitted changes in the repository. {summary}",
    ],
    
    # =========================================================================
    # v3.0: Zero-Touch Autonomous Update Narration Templates
    # =========================================================================
    
    # Zero-Touch Pre-Flight
    NarratorEvent.ZERO_TOUCH_INITIATED: [
        "Initiating autonomous update sequence.",
        "Zero-Touch update detected. Preparing for automatic update.",
        "I've detected an update and am preparing to apply it automatically.",
    ],
    NarratorEvent.ZERO_TOUCH_PRE_FLIGHT: [
        "Running pre-flight safety checks.",
        "Verifying it's safe to update.",
        "Checking system state before update.",
    ],
    NarratorEvent.ZERO_TOUCH_PRE_FLIGHT_PASSED: [
        "Pre-flight checks passed. Proceeding with update.",
        "All safety checks clear. Ready to update.",
        "System is ready for autonomous update.",
    ],
    NarratorEvent.ZERO_TOUCH_PRE_FLIGHT_FAILED: [
        "Pre-flight checks failed. Update postponed. {reason}",
        "Cannot proceed with update at this time. {reason}",
        "Autonomous update blocked by safety check. {reason}",
    ],
    NarratorEvent.ZERO_TOUCH_BLOCKED: [
        "Autonomous update blocked. {reason}",
        "I cannot update automatically right now. {reason}",
        "Update deferred for safety. {reason}",
    ],
    
    # Zero-Touch Staging & Validation
    NarratorEvent.ZERO_TOUCH_STAGING: [
        "Staging update for validation.",
        "Downloading and preparing update safely.",
        "Validating update in staging area.",
    ],
    NarratorEvent.ZERO_TOUCH_VALIDATING: [
        "Validating code integrity.",
        "Running syntax and dependency checks.",
        "Verifying update is safe to apply.",
    ],
    NarratorEvent.ZERO_TOUCH_VALIDATION_PASSED: [
        "Validation complete. Update is safe.",
        "All {count} files validated successfully.",
        "Code checks passed. Ready to merge.",
    ],
    NarratorEvent.ZERO_TOUCH_VALIDATION_FAILED: [
        "Validation failed. Found {count} issues.",
        "Update contains errors. Aborting for safety.",
        "I detected problems with this update. Rolling back staging.",
    ],
    
    # Zero-Touch Update Classification
    NarratorEvent.ZERO_TOUCH_SECURITY_UPDATE: [
        "Security update detected. Applying immediately.",
        "This is a security fix. Prioritizing now.",
        "Critical security patch. Updating with high priority.",
    ],
    NarratorEvent.ZERO_TOUCH_CRITICAL_UPDATE: [
        "Critical bug fix available. Applying automatically.",
        "This update contains important fixes.",
        "Urgent update detected. Proceeding now.",
    ],
    NarratorEvent.ZERO_TOUCH_MINOR_UPDATE: [
        "Minor update detected. Applying automatically.",
        "Routine update with improvements.",
        "Installing standard update package.",
    ],
    NarratorEvent.ZERO_TOUCH_MAJOR_BLOCKED: [
        "Major version update detected. This requires your confirmation.",
        "Breaking changes found. I'll wait for your approval.",
        "Significant update requires manual review. Please confirm when ready.",
    ],
    
    # Zero-Touch Update Execution
    NarratorEvent.ZERO_TOUCH_APPLYING: [
        "Applying autonomous update now.",
        "Merging validated changes.",
        "Installing verified update.",
    ],
    NarratorEvent.ZERO_TOUCH_COMPLETE: [
        "Autonomous update complete. Monitoring for stability.",
        "Zero-Touch update successful. All systems nominal.",
        "Update applied. I'm now verifying stability.",
    ],
    NarratorEvent.ZERO_TOUCH_FAILED: [
        "Autonomous update failed. {reason}",
        "I couldn't complete the update. {reason}",
        "Update aborted safely. {reason}",
    ],
    NarratorEvent.ZERO_TOUCH_COOLDOWN: [
        "Entering cooldown period. Next update check in {minutes} minutes.",
        "Update cooldown active.",
        "Waiting before checking for more updates.",
    ],
    
    # Dead Man's Switch Events
    NarratorEvent.DMS_PROBATION_START: [
        "Starting stability monitoring.",
        "Dead Man's Switch active. Monitoring for {seconds} seconds.",
        "Post-update probation period started.",
    ],
    NarratorEvent.DMS_HEARTBEAT_OK: [
        "System heartbeat normal.",
        "All components responding.",
    ],
    NarratorEvent.DMS_HEARTBEAT_FAILED: [
        "Warning: Heartbeat failure detected.",
        "Component health check failed. Monitoring closely.",
        "Stability check failed. Strike {count} of 3.",
    ],
    NarratorEvent.DMS_PROBATION_PASSED: [
        "Stability verified. Update confirmed successful.",
        "Probation passed. System is stable.",
        "All checks passed. This version is now marked as stable.",
    ],
    NarratorEvent.DMS_ROLLBACK_TRIGGERED: [
        "Stability check failed. Initiating automatic rollback.",
        "Update unstable. Reverting to previous version.",
        "Rolling back to last known good configuration.",
    ],
    NarratorEvent.DMS_STABLE_COMMITTED: [
        "Version committed as stable.",
        "This update is now the baseline version.",
        "Stability confirmed. Ready for the next update.",
    ],
    
    # Prime Directives Events
    NarratorEvent.PRIME_DIRECTIVE_ENFORCED: [
        "Prime Directive enforced. {action}",
        "Safety constraint activated. {action}",
        "Constitutional limit applied. {action}",
    ],
    NarratorEvent.PROTECTED_FILE_BLOCKED: [
        "Blocked modification of protected file. {file}",
        "Cannot modify protected system file. {file}",
        "Protected file access denied. {file}",
    ],
    NarratorEvent.IMMUTABLE_CORE_VIOLATION: [
        "Supervisor protection active. Cannot modify core files.",
        "The supervisor is read-only. This is a safety feature.",
        "Core system files are protected from modification.",
    ],
    NarratorEvent.CONSENT_REQUIRED: [
        "This action requires your confirmation. {action}",
        "I need your approval before proceeding. {action}",
        "Sir, may I have permission to {action}?",
    ],
    NarratorEvent.RESOURCE_LIMIT_EXCEEDED: [
        "Resource limit would be exceeded. {limit}",
        "Autonomous action capped. {limit}",
        "Safety limit reached. {limit}",
    ],
    
    # Busy State Events
    NarratorEvent.Ironcliw_BUSY_DETECTED: [
        "I'm currently processing tasks. Update will wait.",
        "Active tasks detected. Deferring update.",
        "I'm busy at the moment. Will update when idle.",
    ],
    NarratorEvent.Ironcliw_IDLE_DETECTED: [
        "System is idle. Ready for maintenance.",
        "No active tasks. Update window available.",
    ],
    NarratorEvent.UPDATE_DEFERRED: [
        "Update deferred until conditions are optimal.",
        "I'll apply this update when the time is right.",
        "Update queued for later application.",
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
    
    # v3.0: Zero-Touch narration settings
    zero_touch_verbose: bool = True      # Announce Zero-Touch events
    dms_verbose: bool = True             # Announce Dead Man's Switch events
    prime_directives_verbose: bool = True  # Announce safety violations
    skip_heartbeat_ok: bool = True       # Skip routine heartbeat success messages
    context_awareness: bool = True       # Adapt messages based on context
    intelligent_dedup: bool = True       # Skip redundant announcements


@dataclass
class NarrationContext:
    """
    Context for intelligent narration (v3.0).
    
    Tracks state to enable context-aware messaging.
    """
    # Current operation state
    is_zero_touch_active: bool = False
    is_dms_active: bool = False
    current_update_classification: Optional[str] = None
    
    # Counters for adaptive messaging
    heartbeat_count: int = 0
    heartbeat_failures: int = 0
    consecutive_successes: int = 0
    
    # Timing
    last_narration_time: Optional[datetime] = None
    operation_start_time: Optional[datetime] = None
    
    # Recent events for dedup
    recent_events: List[str] = field(default_factory=list)
    
    def reset(self) -> None:
        """Reset context for new operation."""
        self.is_zero_touch_active = False
        self.is_dms_active = False
        self.current_update_classification = None
        self.heartbeat_count = 0
        self.heartbeat_failures = 0
        self.consecutive_successes = 0
        self.operation_start_time = None
        self.recent_events.clear()


class SupervisorNarrator:
    """
    Voice narrator for supervisor events v3.0 - Zero-Touch Edition.

    v3.0 Features:
    - Zero-Touch autonomous update narration
    - Context-aware dynamic messaging
    - Intelligent deduplication
    - Update classification-based announcements
    - Dead Man's Switch status narration
    - Prime Directives violation alerts

    v2.0: Now delegates ALL voice output to UnifiedVoiceOrchestrator,
    ensuring only one voice speaks at a time across the entire system.

    Example:
        >>> narrator = SupervisorNarrator()
        >>> await narrator.narrate(NarratorEvent.UPDATE_STARTING)
        >>> await narrator.narrate_zero_touch_update(classification="security", commits=3)
        >>> await narrator.narrate_dms_status(state="monitoring", health_score=0.95)
    """

    def __init__(self, config: Optional[NarratorConfig] = None):
        """
        Initialize the narrator.

        Args:
            config: Narrator configuration
        """
        self.config = config or NarratorConfig()
        self._is_macos = platform.system() == "Darwin"

        # v2.0: Get unified voice orchestrator (single source of truth)
        self._orchestrator = get_voice_orchestrator()
        
        # v3.0: Context for intelligent narration
        self._context = NarrationContext()
        
        # Event callbacks for extensibility
        self._on_narrate: List[Callable[[NarratorEvent, Dict[str, Any]], None]] = []

        if self._is_macos:
            logger.info(f"🔊 Narrator v3.0 initialized (Zero-Touch Edition)")
        else:
            logger.info("🔇 Narrator initialized (silent mode - non-macOS)")

    async def start(self) -> None:
        """Start the narrator (starts unified orchestrator if needed)."""
        # v2.0: Delegate to unified orchestrator
        if not self._orchestrator._running:
            await self._orchestrator.start()

    async def stop(self) -> None:
        """Stop the narrator."""
        # v2.0: Don't stop orchestrator here - it's shared across components
        # The orchestrator will be stopped by the supervisor at shutdown
        pass

    async def speak(
        self,
        text: str,
        wait: bool = False,
        priority: VoicePriority = VoicePriority.MEDIUM,
        topic: Optional[SpeechTopic] = None,
    ) -> None:
        """
        Speak arbitrary text through unified orchestrator.

        Args:
            text: Text to speak
            wait: If True, wait for speech to complete
            priority: Message priority (v4.0)
            topic: Speech topic for intelligent grouping (v4.0)
        """
        if not self.config.enabled:
            logger.debug(f"🔇 Narrator disabled, skipping: {text[:50]}...")
            return

        # v4.0: Delegate to unified orchestrator with topic
        await self._orchestrator.speak(
            text=text,
            priority=priority,
            source=VoiceSource.SUPERVISOR,
            wait=wait,
            topic=topic,
        )
    
    def _event_to_topic(self, event: NarratorEvent) -> SpeechTopic:
        """v4.0: Map narrator event to speech topic for intelligent grouping."""
        event_value = event.value.lower()
        
        if 'startup' in event_value or 'online' in event_value:
            return SpeechTopic.STARTUP
        elif 'shutdown' in event_value:
            return SpeechTopic.SHUTDOWN
        elif 'update' in event_value or 'download' in event_value or 'install' in event_value:
            return SpeechTopic.UPDATE
        elif 'rollback' in event_value or 'revert' in event_value:
            return SpeechTopic.ROLLBACK
        elif 'error' in event_value or 'fail' in event_value or 'crash' in event_value:
            return SpeechTopic.ERROR
        elif 'health' in event_value or 'heartbeat' in event_value:
            return SpeechTopic.HEALTH
        elif 'zero_touch' in event_value or 'autonomous' in event_value:
            return SpeechTopic.ZERO_TOUCH
        elif 'dms' in event_value or 'probation' in event_value:
            return SpeechTopic.DMS
        elif 'progress' in event_value:
            return SpeechTopic.PROGRESS
        else:
            return SpeechTopic.GENERAL
    
    def _event_to_priority(self, event: NarratorEvent) -> VoicePriority:
        """v4.0: Map narrator event to appropriate priority."""
        event_value = event.value.lower()
        
        # Critical events
        if any(kw in event_value for kw in ['crash', 'fail', 'error', 'violation', 'rollback']):
            return VoicePriority.HIGH
        # Important milestones
        elif any(kw in event_value for kw in ['complete', 'online', 'passed', 'success']):
            return VoicePriority.HIGH
        # Standard updates
        elif any(kw in event_value for kw in ['start', 'init', 'begin']):
            return VoicePriority.MEDIUM
        # Progress updates
        elif 'progress' in event_value:
            return VoicePriority.LOW
        else:
            return VoicePriority.MEDIUM
    
    async def narrate(
        self,
        event: NarratorEvent,
        wait: bool = False,
        priority: Optional[VoicePriority] = None,
        **kwargs,
    ) -> None:
        """
        Narrate a supervisor event with intelligent topic handling.
        
        Args:
            event: The event to narrate
            wait: If True, wait for speech to complete
            priority: Override priority (auto-inferred if not provided)
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
        
        # v4.0: Infer topic and priority from event
        topic = self._event_to_topic(event)
        inferred_priority = priority or self._event_to_priority(event)
        
        logger.info(f"🔊 Narrating ({topic.value}, {inferred_priority.name}): {text}")
        await self.speak(text, wait=wait, priority=inferred_priority, topic=topic)
    
    async def announce_update_progress(
        self,
        phase: str,
        detail: Optional[str] = None,
    ) -> None:
        """
        Announce update progress with optional detail.
        
        v4.0: Uses topic-based cooldown to prevent rapid-fire announcements.
        
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
            # v4.0: Use LOW priority for progress updates to allow skipping
            await self.narrate(event, priority=VoicePriority.LOW)
        elif detail:
            await self.speak(detail, priority=VoicePriority.LOW, topic=SpeechTopic.UPDATE)
    
    def set_voice(self, voice: NarratorVoice) -> None:
        """Change the narrator voice."""
        self.config.voice = voice
        logger.info(f"🔊 Voice changed to {voice.value}")
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable narration."""
        self.config.enabled = enabled
    
    # =========================================================================
    # v3.0: Intelligent Zero-Touch Narration Methods
    # =========================================================================
    
    async def narrate_zero_touch_update(
        self,
        classification: str,
        commits: int = 0,
        files_changed: int = 0,
        validation_passed: bool = True,
        wait: bool = False,
    ) -> None:
        """
        Intelligently narrate a Zero-Touch autonomous update.
        
        Adapts the message based on update classification and context.
        
        Args:
            classification: Update type (security, critical, minor, major)
            commits: Number of commits in update
            files_changed: Number of files changed
            validation_passed: Whether validation passed
            wait: Wait for narration to complete
        """
        if not self.config.zero_touch_verbose:
            return
        
        # Update context
        self._context.is_zero_touch_active = True
        self._context.current_update_classification = classification
        self._context.operation_start_time = datetime.now()
        
        # Select event based on classification
        event_map = {
            "security": NarratorEvent.ZERO_TOUCH_SECURITY_UPDATE,
            "critical": NarratorEvent.ZERO_TOUCH_CRITICAL_UPDATE,
            "minor": NarratorEvent.ZERO_TOUCH_MINOR_UPDATE,
            "major": NarratorEvent.ZERO_TOUCH_MAJOR_BLOCKED,
        }
        
        event = event_map.get(classification, NarratorEvent.ZERO_TOUCH_MINOR_UPDATE)
        
        # Build context for template
        kwargs = {}
        if commits > 0:
            kwargs["commits"] = f"{commits} commits"
        if files_changed > 0:
            kwargs["files"] = f"{files_changed} files"
        
        await self.narrate(event, wait=wait, **kwargs)
    
    async def narrate_zero_touch_validation(
        self,
        files_checked: int,
        syntax_errors: int = 0,
        import_errors: int = 0,
        pip_conflicts: int = 0,
        wait: bool = False,
    ) -> None:
        """
        Narrate Zero-Touch validation results.
        
        Args:
            files_checked: Number of files validated
            syntax_errors: Number of syntax errors found
            import_errors: Number of import errors found
            pip_conflicts: Number of pip conflicts found
            wait: Wait for narration to complete
        """
        if not self.config.zero_touch_verbose:
            return
        
        total_issues = syntax_errors + import_errors + pip_conflicts
        
        if total_issues == 0:
            await self.narrate(
                NarratorEvent.ZERO_TOUCH_VALIDATION_PASSED,
                count=files_checked,
                wait=wait,
            )
        else:
            await self.narrate(
                NarratorEvent.ZERO_TOUCH_VALIDATION_FAILED,
                count=total_issues,
                wait=wait,
            )
    
    async def narrate_zero_touch_blocked(
        self,
        reason: str,
        wait: bool = False,
    ) -> None:
        """
        Narrate when Zero-Touch update is blocked.
        
        Args:
            reason: Why the update was blocked
            wait: Wait for narration to complete
        """
        if not self.config.zero_touch_verbose:
            return
        
        # Simplify reason for speech
        simplified = self._simplify_reason(reason)
        
        await self.narrate(
            NarratorEvent.ZERO_TOUCH_BLOCKED,
            reason=simplified,
            wait=wait,
        )
    
    async def narrate_zero_touch_complete(
        self,
        success: bool,
        new_version: Optional[str] = None,
        duration_seconds: float = 0.0,
        error: Optional[str] = None,
        wait: bool = True,
    ) -> None:
        """
        Narrate Zero-Touch update completion.
        
        Args:
            success: Whether update succeeded
            new_version: New version if successful
            duration_seconds: How long the update took
            error: Error message if failed
            wait: Wait for narration to complete
        """
        if not self.config.zero_touch_verbose:
            return
        
        if success:
            await self.narrate(NarratorEvent.ZERO_TOUCH_COMPLETE, wait=wait)
        else:
            reason = self._simplify_reason(error or "Unknown error")
            await self.narrate(NarratorEvent.ZERO_TOUCH_FAILED, reason=reason, wait=wait)
        
        # Reset context
        self._context.is_zero_touch_active = False
    
    # =========================================================================
    # v3.0: Dead Man's Switch Narration Methods
    # =========================================================================
    
    async def narrate_dms_status(
        self,
        state: str,
        health_score: float = 1.0,
        consecutive_failures: int = 0,
        probation_remaining: float = 0.0,
        wait: bool = False,
    ) -> None:
        """
        Narrate Dead Man's Switch status intelligently.
        
        Adapts verbosity based on state and history.
        
        Args:
            state: Current DMS state (monitoring, healthy, failing, etc.)
            health_score: Current health score (0.0 - 1.0)
            consecutive_failures: Number of consecutive heartbeat failures
            probation_remaining: Seconds remaining in probation
            wait: Wait for narration to complete
        """
        if not self.config.dms_verbose:
            return
        
        self._context.is_dms_active = True
        
        # State-based narration
        if state == "monitoring":
            if probation_remaining > 0:
                await self.narrate(
                    NarratorEvent.DMS_PROBATION_START,
                    seconds=int(probation_remaining),
                    wait=wait,
                )
        
        elif state == "heartbeat_ok":
            self._context.heartbeat_count += 1
            self._context.consecutive_successes += 1
            
            # Skip routine heartbeat messages unless it's first or after recovery
            if self.config.skip_heartbeat_ok and self._context.consecutive_successes > 1:
                return
            
            # Only announce first heartbeat or recovery
            if self._context.heartbeat_failures > 0:
                # Recovering from failures - use HIGH priority
                await self.speak(
                    "Health recovered. Resuming normal monitoring.",
                    wait=wait,
                    priority=VoicePriority.HIGH,
                    topic=SpeechTopic.DMS,
                )
                self._context.heartbeat_failures = 0
        
        elif state == "heartbeat_failed":
            self._context.heartbeat_failures = consecutive_failures
            self._context.consecutive_successes = 0
            
            await self.narrate(
                NarratorEvent.DMS_HEARTBEAT_FAILED,
                count=consecutive_failures,
                wait=wait,
            )
        
        elif state == "passed":
            await self.narrate(NarratorEvent.DMS_PROBATION_PASSED, wait=wait)
            self._context.is_dms_active = False
        
        elif state == "rollback":
            await self.narrate(NarratorEvent.DMS_ROLLBACK_TRIGGERED, wait=wait)
            self._context.is_dms_active = False
        
        elif state == "committed":
            await self.narrate(NarratorEvent.DMS_STABLE_COMMITTED, wait=wait)
            self._context.is_dms_active = False
    
    # =========================================================================
    # v3.0: Prime Directives Narration Methods
    # =========================================================================
    
    async def narrate_prime_directive_violation(
        self,
        directive_type: str,
        action: Optional[str] = None,
        file: Optional[str] = None,
        limit: Optional[str] = None,
        wait: bool = True,
    ) -> None:
        """
        Narrate a Prime Directive safety violation.
        
        These are always announced as they represent safety constraints.
        
        Args:
            directive_type: Type of violation (protected_file, immutable_core, consent, limit)
            action: The blocked action
            file: The protected file
            limit: The exceeded limit
            wait: Wait for narration to complete
        """
        if not self.config.prime_directives_verbose:
            return
        
        event_map = {
            "protected_file": NarratorEvent.PROTECTED_FILE_BLOCKED,
            "immutable_core": NarratorEvent.IMMUTABLE_CORE_VIOLATION,
            "consent": NarratorEvent.CONSENT_REQUIRED,
            "limit": NarratorEvent.RESOURCE_LIMIT_EXCEEDED,
        }
        
        event = event_map.get(directive_type, NarratorEvent.PRIME_DIRECTIVE_ENFORCED)
        
        kwargs = {}
        if action:
            kwargs["action"] = action
        if file:
            kwargs["file"] = file
        if limit:
            kwargs["limit"] = limit
        
        # Prime directive violations are HIGH priority
        await self._orchestrator.speak(
            text=self._get_template_text(event, **kwargs),
            priority=VoicePriority.HIGH,
            source=VoiceSource.PRIME_DIRECTIVES,
            wait=wait,
        )
    
    # =========================================================================
    # v3.0: Busy State Narration Methods
    # =========================================================================
    
    async def narrate_busy_state(
        self,
        is_busy: bool,
        active_tasks: int = 0,
        wait: bool = False,
    ) -> None:
        """
        Narrate Ironcliw busy state for update decisions.
        
        Args:
            is_busy: Whether Ironcliw is currently busy
            active_tasks: Number of active tasks
            wait: Wait for narration to complete
        """
        if is_busy:
            # Dedup: Don't repeat busy state
            if self._should_skip_event("busy_detected"):
                return
            
            await self.narrate(NarratorEvent.Ironcliw_BUSY_DETECTED, wait=wait)
        else:
            # Only announce idle if we were previously busy
            if "busy_detected" in self._context.recent_events:
                await self.narrate(NarratorEvent.Ironcliw_IDLE_DETECTED, wait=wait)
    
    # =========================================================================
    # v3.0: Helper Methods
    # =========================================================================
    
    def _simplify_reason(self, reason: str) -> str:
        """Simplify a technical reason for speech."""
        if not reason:
            return "Unknown reason"
        
        # Truncate long reasons
        if len(reason) > 100:
            reason = reason[:97] + "..."
        
        # Remove technical jargon for cleaner speech
        replacements = {
            "pre-flight": "safety check",
            "timeout": "took too long",
            "SIGTERM": "shutdown signal",
            "SIGKILL": "force stop",
            "OOM": "out of memory",
            "API": "service",
            "HTTP": "connection",
        }
        
        for old, new in replacements.items():
            reason = reason.replace(old, new)
        
        return reason
    
    def _get_template_text(self, event: NarratorEvent, **kwargs) -> str:
        """Get formatted template text for an event."""
        templates = NARRATION_TEMPLATES.get(event, [])
        if not templates:
            return str(event.value)
        
        template = random.choice(templates)
        
        try:
            text = template.format(**kwargs) if kwargs else template
            # Clean up unfilled placeholders
            import re
            text = re.sub(r'\{[^}]+\}', '', text).strip()
            return text
        except KeyError:
            return template
    
    def _should_skip_event(self, event_key: str) -> bool:
        """Check if an event should be skipped (intelligent dedup)."""
        if not self.config.intelligent_dedup:
            return False
        
        if event_key in self._context.recent_events:
            return True
        
        self._context.recent_events.append(event_key)
        
        # Keep last 10 events
        if len(self._context.recent_events) > 10:
            self._context.recent_events.pop(0)
        
        return False
    
    def reset_context(self) -> None:
        """Reset narration context for new operation."""
        self._context.reset()
    
    def on_narrate(self, callback: Callable[[NarratorEvent, Dict[str, Any]], None]) -> None:
        """Register a callback for narration events."""
        self._on_narrate.append(callback)


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
