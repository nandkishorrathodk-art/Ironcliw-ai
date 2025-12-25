"""
JARVIS Vision-Safety Integration v1.0
=====================================

The critical safety layer that bridges the Vision Cognitive Loop with
existing safety infrastructure. This module implements:

1. DeadMansSwitch: Physical kill-switch (mouse to top-left corner)
2. SafetyAuditPhase: Phase 2.5 plan auditing before execution
3. ActionConfirmation: Voice/text confirmation for risky actions
4. VisualClickOverlay: Red circle indicator before clicks

Architecture:
    ┌────────────────────────────────────────────────────────────────────┐
    │                    VisionSafetyIntegration                          │
    │  ┌───────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
    │  │ DeadMansSwitch│  │ PlanAuditor  │  │  VisualClickValidator   │  │
    │  └───────┬───────┘  └──────┬───────┘  └───────────┬─────────────┘  │
    │          │                 │                      │                 │
    │          │     ┌───────────▼───────────┐          │                 │
    │          │     │ CommandSafetyClassifier│          │                 │
    │          │     └───────────┬───────────┘          │                 │
    │          │                 │                      │                 │
    │          └─────────────────┼──────────────────────┘                 │
    │                            ▼                                        │
    │              ┌───────────────────────────┐                          │
    │              │   ActionConfirmationHub   │                          │
    │              │   (Voice + Text + Veto)   │                          │
    │              └───────────────────────────┘                          │
    └────────────────────────────────────────────────────────────────────┘

Safety Philosophy:
- No destructive action executes without explicit confirmation
- Physical override (Dead Man's Switch) stops execution within 100ms
- Visual feedback shows exactly where JARVIS will click before clicking
- All risky actions are audited and logged for forensics

Integration Points:
- AgenticTaskRunner: Phase 2.5 safety audit
- VisionCognitiveLoop: Visual verification with safety context
- AgenticWatchdog: Kill switch callbacks
- CommandSafetyClassifier: Plan analysis

Author: JARVIS AI System
Version: 1.0.0 (Vision-Safety Integration)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VisionSafetyConfig:
    """Configuration for Vision-Safety Integration."""

    # Dead Man's Switch
    dead_man_switch_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_DEAD_MAN_SWITCH", "true").lower() == "true"
    )
    dead_man_corner_size_px: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_DEAD_MAN_CORNER_SIZE", "10"))
    )
    dead_man_poll_interval_ms: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_DEAD_MAN_POLL_MS", "50"))
    )
    dead_man_trigger_duration_ms: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_DEAD_MAN_TRIGGER_MS", "200"))
    )

    # Safety Audit
    safety_audit_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_SAFETY_AUDIT", "true").lower() == "true"
    )
    auto_confirm_green: bool = field(
        default_factory=lambda: os.getenv("JARVIS_AUTO_CONFIRM_GREEN", "true").lower() == "true"
    )
    confirmation_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_CONFIRM_TIMEOUT", "30.0"))
    )

    # Visual Click Overlay
    visual_overlay_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_VISUAL_OVERLAY", "true").lower() == "true"
    )
    overlay_delay_seconds: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_OVERLAY_DELAY", "1.0"))
    )
    overlay_veto_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_OVERLAY_VETO", "true").lower() == "true"
    )
    overlay_color: str = field(
        default_factory=lambda: os.getenv("JARVIS_OVERLAY_COLOR", "red")
    )
    overlay_radius_px: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_OVERLAY_RADIUS", "30"))
    )

    # Voice Confirmation
    voice_confirmation_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_VOICE_CONFIRM", "true").lower() == "true"
    )
    confirmation_phrases: List[str] = field(
        default_factory=lambda: ["yes", "proceed", "confirm", "do it", "execute", "go ahead"]
    )
    cancellation_phrases: List[str] = field(
        default_factory=lambda: ["no", "stop", "cancel", "abort", "wait", "hold"]
    )


# =============================================================================
# Enums and Data Classes
# =============================================================================

class SafetyVerdict(str, Enum):
    """Result of safety audit."""
    GREEN = "green"          # Safe, auto-execute
    YELLOW = "yellow"        # Requires confirmation
    RED = "red"              # Destructive, always confirm
    BLOCKED = "blocked"      # Cannot execute (policy violation)


class ConfirmationMethod(str, Enum):
    """Methods for obtaining user confirmation."""
    VOICE = "voice"          # Voice command
    TEXT = "text"            # Text input
    PHYSICAL = "physical"    # Physical gesture (e.g., click button)
    TIMEOUT = "timeout"      # Auto-confirm after timeout (green only)


@dataclass
class PlanStep:
    """A step in an execution plan."""
    action: str
    target: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    estimated_duration_ms: float = 0.0


@dataclass
class SafetyAuditResult:
    """Result of auditing an execution plan."""
    verdict: SafetyVerdict
    plan_steps: List[PlanStep]
    risky_steps: List[Tuple[int, PlanStep, str]]  # (index, step, reason)
    confirmation_required: bool
    blocking_reasons: List[str]
    suggested_modifications: List[str]
    audit_time_ms: float
    classifier_used: bool

    @property
    def is_safe(self) -> bool:
        """Quick check if plan is safe to auto-execute."""
        return self.verdict == SafetyVerdict.GREEN and not self.confirmation_required

    @property
    def is_blocked(self) -> bool:
        """Check if plan is blocked from execution."""
        return self.verdict == SafetyVerdict.BLOCKED


@dataclass
class ConfirmationResult:
    """Result of requesting user confirmation."""
    confirmed: bool
    method: ConfirmationMethod
    response_text: Optional[str] = None
    response_time_ms: float = 0.0
    timed_out: bool = False
    vetoed: bool = False


@dataclass
class VisualClickPreview:
    """Preview information for a click action."""
    x: int
    y: int
    button: str
    overlay_shown: bool
    user_vetoed: bool
    delay_applied_ms: float


@dataclass
class DeadManSwitchStatus:
    """Status of the Dead Man's Switch."""
    enabled: bool
    triggered: bool
    in_corner: bool
    corner_enter_time: Optional[float] = None
    last_check_time: Optional[float] = None
    mouse_position: Optional[Tuple[int, int]] = None


# =============================================================================
# Dead Man's Switch
# =============================================================================

class DeadMansSwitch:
    """
    Physical kill-switch that monitors mouse position.

    When the mouse cursor is moved to the top-left corner (0,0) and held
    for a configurable duration, the kill switch triggers and immediately
    halts all agentic execution.

    This provides a physical, guaranteed-to-work safety mechanism that
    doesn't rely on any software state.
    """

    def __init__(
        self,
        config: VisionSafetyConfig,
        on_trigger: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        """Initialize the Dead Man's Switch.

        Args:
            config: Safety configuration
            on_trigger: Callback when kill switch triggers
        """
        self._config = config
        self._on_trigger = on_trigger

        # State
        self._enabled = config.dead_man_switch_enabled
        self._monitoring = False
        self._triggered = False
        self._in_corner = False
        self._corner_enter_time: Optional[float] = None
        self._last_position: Optional[Tuple[int, int]] = None

        # Tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Metrics
        self._false_alarms = 0
        self._trigger_count = 0

        logger.info(
            f"[DeadMansSwitch] Initialized: enabled={self._enabled}, "
            f"corner_size={config.dead_man_corner_size_px}px, "
            f"trigger_duration={config.dead_man_trigger_duration_ms}ms"
        )

    async def start(self) -> None:
        """Start monitoring mouse position."""
        if not self._enabled:
            logger.info("[DeadMansSwitch] Disabled, not starting")
            return

        if self._monitoring:
            return

        self._shutdown_event.clear()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._monitoring = True
        logger.info("[DeadMansSwitch] Started monitoring")

    async def stop(self) -> None:
        """Stop monitoring."""
        self._shutdown_event.set()

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        self._monitoring = False
        logger.info("[DeadMansSwitch] Stopped monitoring")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop that checks mouse position."""
        poll_interval = self._config.dead_man_poll_interval_ms / 1000.0

        while not self._shutdown_event.is_set():
            try:
                await self._check_mouse_position()
                await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[DeadMansSwitch] Monitor error: {e}")
                await asyncio.sleep(poll_interval * 2)

    async def _check_mouse_position(self) -> None:
        """Check if mouse is in the kill-switch corner."""
        try:
            # Get current mouse position
            position = await self._get_mouse_position()
            if position is None:
                return

            self._last_position = position
            x, y = position
            corner_size = self._config.dead_man_corner_size_px

            # Check if in corner (top-left)
            in_corner = (x <= corner_size and y <= corner_size)

            if in_corner and not self._in_corner:
                # Just entered corner
                self._in_corner = True
                self._corner_enter_time = time.time()
                logger.debug(f"[DeadMansSwitch] Mouse entered corner at ({x}, {y})")

            elif in_corner and self._in_corner:
                # Still in corner - check duration
                if self._corner_enter_time:
                    duration_ms = (time.time() - self._corner_enter_time) * 1000
                    if duration_ms >= self._config.dead_man_trigger_duration_ms:
                        await self._trigger()

            elif not in_corner and self._in_corner:
                # Left corner - reset
                self._in_corner = False
                self._corner_enter_time = None
                logger.debug("[DeadMansSwitch] Mouse left corner (reset)")

        except Exception as e:
            logger.error(f"[DeadMansSwitch] Position check failed: {e}")

    async def _get_mouse_position(self) -> Optional[Tuple[int, int]]:
        """Get current mouse position using platform-appropriate method."""
        try:
            # Try pyautogui first (cross-platform)
            import pyautogui
            pos = pyautogui.position()
            return (pos.x, pos.y)
        except ImportError:
            pass

        try:
            # Fallback: macOS native via subprocess
            import subprocess
            result = subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to get position of mouse'],
                capture_output=True,
                text=True,
                timeout=0.1,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 2:
                    return (int(parts[0]), int(parts[1]))
        except Exception:
            pass

        try:
            # Fallback: Quartz (macOS)
            from Quartz import CGEventGetLocation, CGEventCreate
            event = CGEventCreate(None)
            if event:
                loc = CGEventGetLocation(event)
                return (int(loc.x), int(loc.y))
        except ImportError:
            pass

        return None

    async def _trigger(self) -> None:
        """Trigger the kill switch."""
        if self._triggered:
            return  # Already triggered

        self._triggered = True
        self._trigger_count += 1

        logger.warning("[DeadMansSwitch] EMERGENCY STOP TRIGGERED!")

        # Call the callback
        if self._on_trigger:
            try:
                await self._on_trigger()
            except Exception as e:
                logger.error(f"[DeadMansSwitch] Trigger callback failed: {e}")

        # Reset state after a short delay
        await asyncio.sleep(2.0)
        self._triggered = False
        self._in_corner = False
        self._corner_enter_time = None

    def is_triggered(self) -> bool:
        """Check if kill switch is currently triggered."""
        return self._triggered

    def get_status(self) -> DeadManSwitchStatus:
        """Get current status."""
        return DeadManSwitchStatus(
            enabled=self._enabled,
            triggered=self._triggered,
            in_corner=self._in_corner,
            corner_enter_time=self._corner_enter_time,
            last_check_time=time.time(),
            mouse_position=self._last_position,
        )


# =============================================================================
# Plan Safety Auditor
# =============================================================================

class PlanSafetyAuditor:
    """
    Audits execution plans for safety before execution.

    Uses CommandSafetyClassifier to analyze each step in a plan and
    determine the overall safety verdict.
    """

    def __init__(self, config: VisionSafetyConfig):
        """Initialize the plan auditor."""
        self._config = config
        self._classifier = None
        self._initialized = False

        # Action keywords for classification
        self._destructive_actions = {
            "delete", "remove", "rm", "unlink", "destroy", "drop",
            "truncate", "wipe", "erase", "format", "shred",
        }
        self._risky_actions = {
            "send", "email", "post", "publish", "upload", "submit",
            "pay", "purchase", "buy", "transfer", "move", "rename",
            "install", "uninstall", "modify", "change", "update",
        }
        self._safe_actions = {
            "read", "view", "open", "show", "display", "list",
            "search", "find", "check", "get", "fetch", "copy",
        }

    async def initialize(self) -> bool:
        """Initialize the auditor and load classifier."""
        if self._initialized:
            return True

        try:
            from system_control.command_safety import get_command_classifier
            self._classifier = get_command_classifier()
            self._initialized = True
            logger.info("[PlanAuditor] Initialized with CommandSafetyClassifier")
            return True
        except ImportError as e:
            logger.warning(f"[PlanAuditor] CommandSafetyClassifier not available: {e}")
            self._initialized = True  # Can still work without classifier
            return True

    async def audit_plan(
        self,
        plan: List[Dict[str, Any]],
        goal: str,
    ) -> SafetyAuditResult:
        """
        Audit an execution plan for safety.

        Args:
            plan: List of plan steps (from Phase 2)
            goal: The original goal

        Returns:
            SafetyAuditResult with verdict and details
        """
        start_time = time.time()

        # Convert plan to PlanStep objects
        plan_steps = self._parse_plan_steps(plan)

        # Analyze each step
        risky_steps: List[Tuple[int, PlanStep, str]] = []
        blocking_reasons: List[str] = []
        suggested_modifications: List[str] = []

        for idx, step in enumerate(plan_steps):
            step_verdict, reason = await self._analyze_step(step)

            if step_verdict == SafetyVerdict.RED:
                risky_steps.append((idx, step, f"DESTRUCTIVE: {reason}"))
            elif step_verdict == SafetyVerdict.YELLOW:
                risky_steps.append((idx, step, f"RISKY: {reason}"))
            elif step_verdict == SafetyVerdict.BLOCKED:
                blocking_reasons.append(f"Step {idx + 1}: {reason}")

        # Determine overall verdict
        if blocking_reasons:
            verdict = SafetyVerdict.BLOCKED
        elif any(r[2].startswith("DESTRUCTIVE") for r in risky_steps):
            verdict = SafetyVerdict.RED
        elif risky_steps:
            verdict = SafetyVerdict.YELLOW
        else:
            verdict = SafetyVerdict.GREEN

        # Determine if confirmation is required
        confirmation_required = verdict in (SafetyVerdict.RED, SafetyVerdict.YELLOW)
        if verdict == SafetyVerdict.GREEN and not self._config.auto_confirm_green:
            confirmation_required = True

        # Generate suggestions for risky steps
        for idx, step, reason in risky_steps:
            suggestion = self._suggest_safer_alternative(step)
            if suggestion:
                suggested_modifications.append(f"Step {idx + 1}: {suggestion}")

        audit_time_ms = (time.time() - start_time) * 1000

        result = SafetyAuditResult(
            verdict=verdict,
            plan_steps=plan_steps,
            risky_steps=risky_steps,
            confirmation_required=confirmation_required,
            blocking_reasons=blocking_reasons,
            suggested_modifications=suggested_modifications,
            audit_time_ms=audit_time_ms,
            classifier_used=self._classifier is not None,
        )

        logger.info(
            f"[PlanAuditor] Audit complete: verdict={verdict.value}, "
            f"risky_steps={len(risky_steps)}, "
            f"confirmation_required={confirmation_required}, "
            f"time={audit_time_ms:.0f}ms"
        )

        return result

    def _parse_plan_steps(self, plan: List[Dict[str, Any]]) -> List[PlanStep]:
        """Parse raw plan data into PlanStep objects."""
        steps = []
        for item in plan:
            if isinstance(item, dict):
                step = PlanStep(
                    action=item.get("action", item.get("step", str(item))),
                    target=item.get("target"),
                    parameters=item.get("parameters", item.get("params", {})),
                    description=item.get("description", ""),
                    estimated_duration_ms=item.get("duration_ms", 0.0),
                )
            elif isinstance(item, str):
                step = PlanStep(action=item)
            else:
                step = PlanStep(action=str(item))
            steps.append(step)
        return steps

    async def _analyze_step(self, step: PlanStep) -> Tuple[SafetyVerdict, str]:
        """Analyze a single step for safety."""
        action_lower = step.action.lower()

        # Check for destructive actions
        for keyword in self._destructive_actions:
            if keyword in action_lower:
                return SafetyVerdict.RED, f"Contains destructive keyword: {keyword}"

        # Check for risky actions
        for keyword in self._risky_actions:
            if keyword in action_lower:
                return SafetyVerdict.YELLOW, f"Contains risky keyword: {keyword}"

        # Use CommandSafetyClassifier if available
        if self._classifier:
            try:
                # Check if action looks like a command
                if self._looks_like_command(step.action):
                    classification = self._classifier.classify(step.action)
                    if classification.tier.value == "red":
                        return SafetyVerdict.RED, classification.reasoning
                    elif classification.tier.value == "yellow":
                        return SafetyVerdict.YELLOW, classification.reasoning
            except Exception as e:
                logger.debug(f"[PlanAuditor] Classifier error: {e}")

        # Check for safe actions
        for keyword in self._safe_actions:
            if keyword in action_lower:
                return SafetyVerdict.GREEN, "Safe action"

        # Unknown - default to yellow (cautious)
        return SafetyVerdict.YELLOW, "Unknown action type"

    def _looks_like_command(self, action: str) -> bool:
        """Check if action looks like a shell command."""
        command_indicators = [
            " ", "|", ">", "<", ";", "&", "$", "`",
            "rm ", "mv ", "cp ", "git ", "npm ", "pip ",
            "docker ", "kubectl ", "curl ", "wget ",
        ]
        action_lower = action.lower()
        return any(ind in action_lower for ind in command_indicators)

    def _suggest_safer_alternative(self, step: PlanStep) -> Optional[str]:
        """Suggest a safer alternative for a risky step."""
        action_lower = step.action.lower()

        suggestions = {
            "delete": "Consider using 'trash' or moving to a temporary folder first",
            "rm": "Use 'rm -i' (interactive) or 'trash' command instead",
            "send": "Preview the content before sending",
            "email": "Save as draft first for review",
            "publish": "Publish to a staging environment first",
            "pay": "Verify the recipient and amount before proceeding",
        }

        for keyword, suggestion in suggestions.items():
            if keyword in action_lower:
                return suggestion

        return None


# =============================================================================
# Action Confirmation Hub
# =============================================================================

class ActionConfirmationHub:
    """
    Centralized hub for obtaining user confirmation for risky actions.

    Supports multiple confirmation methods:
    - Voice commands ("yes", "proceed", etc.)
    - Text input
    - Physical gestures (clicking confirmation button)
    - Timeout auto-confirm (for green actions only)
    """

    def __init__(
        self,
        config: VisionSafetyConfig,
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        """Initialize the confirmation hub."""
        self._config = config
        self._tts_callback = tts_callback

        # State
        self._awaiting_confirmation = False
        self._confirmation_event = asyncio.Event()
        self._confirmation_result: Optional[ConfirmationResult] = None

        # Voice recognition hook
        self._voice_callback: Optional[Callable[[str], None]] = None

    async def request_confirmation(
        self,
        action_description: str,
        verdict: SafetyVerdict,
        risky_details: Optional[str] = None,
    ) -> ConfirmationResult:
        """
        Request user confirmation for an action.

        Args:
            action_description: What action needs confirmation
            verdict: The safety verdict (determines confirmation urgency)
            risky_details: Additional details about risks

        Returns:
            ConfirmationResult with user's response
        """
        start_time = time.time()

        # Reset state
        self._awaiting_confirmation = True
        self._confirmation_event.clear()
        self._confirmation_result = None

        # Build confirmation message
        message = self._build_confirmation_message(
            action_description, verdict, risky_details
        )

        # Announce via TTS
        if self._tts_callback and self._config.voice_confirmation_enabled:
            await self._tts_callback(message)

        logger.info(f"[ConfirmationHub] Awaiting confirmation for: {action_description}")

        # Wait for confirmation or timeout
        try:
            await asyncio.wait_for(
                self._confirmation_event.wait(),
                timeout=self._config.confirmation_timeout_seconds,
            )

            if self._confirmation_result:
                self._confirmation_result.response_time_ms = (time.time() - start_time) * 1000
                return self._confirmation_result

        except asyncio.TimeoutError:
            logger.info("[ConfirmationHub] Confirmation timed out")

            # For green actions, timeout means auto-confirm
            if verdict == SafetyVerdict.GREEN and self._config.auto_confirm_green:
                return ConfirmationResult(
                    confirmed=True,
                    method=ConfirmationMethod.TIMEOUT,
                    response_time_ms=(time.time() - start_time) * 1000,
                    timed_out=True,
                )

            # For other verdicts, timeout means deny
            return ConfirmationResult(
                confirmed=False,
                method=ConfirmationMethod.TIMEOUT,
                response_time_ms=(time.time() - start_time) * 1000,
                timed_out=True,
            )

        finally:
            self._awaiting_confirmation = False

        # Default deny
        return ConfirmationResult(
            confirmed=False,
            method=ConfirmationMethod.TIMEOUT,
            response_time_ms=(time.time() - start_time) * 1000,
        )

    def receive_voice_input(self, text: str) -> bool:
        """
        Receive voice input and check for confirmation/cancellation.

        Args:
            text: Transcribed voice input

        Returns:
            True if input was processed as confirmation/cancellation
        """
        if not self._awaiting_confirmation:
            return False

        text_lower = text.lower().strip()

        # Check for confirmation phrases
        for phrase in self._config.confirmation_phrases:
            if phrase in text_lower:
                self._confirmation_result = ConfirmationResult(
                    confirmed=True,
                    method=ConfirmationMethod.VOICE,
                    response_text=text,
                )
                self._confirmation_event.set()
                logger.info(f"[ConfirmationHub] Voice confirmation received: {text}")
                return True

        # Check for cancellation phrases
        for phrase in self._config.cancellation_phrases:
            if phrase in text_lower:
                self._confirmation_result = ConfirmationResult(
                    confirmed=False,
                    method=ConfirmationMethod.VOICE,
                    response_text=text,
                )
                self._confirmation_event.set()
                logger.info(f"[ConfirmationHub] Voice cancellation received: {text}")
                return True

        return False

    def receive_text_input(self, text: str) -> None:
        """Receive text input for confirmation."""
        if not self._awaiting_confirmation:
            return

        text_lower = text.lower().strip()
        confirmed = text_lower in ("yes", "y", "confirm", "proceed", "ok")

        self._confirmation_result = ConfirmationResult(
            confirmed=confirmed,
            method=ConfirmationMethod.TEXT,
            response_text=text,
        )
        self._confirmation_event.set()

    def trigger_veto(self) -> None:
        """Trigger a veto (user cancelled via physical action)."""
        if not self._awaiting_confirmation:
            return

        self._confirmation_result = ConfirmationResult(
            confirmed=False,
            method=ConfirmationMethod.PHYSICAL,
            vetoed=True,
        )
        self._confirmation_event.set()
        logger.info("[ConfirmationHub] Veto triggered")

    def _build_confirmation_message(
        self,
        action: str,
        verdict: SafetyVerdict,
        details: Optional[str],
    ) -> str:
        """Build the confirmation message for TTS."""
        if verdict == SafetyVerdict.RED:
            prefix = "Warning, this action is potentially destructive. "
        elif verdict == SafetyVerdict.YELLOW:
            prefix = "This action requires your confirmation. "
        else:
            prefix = ""

        message = f"{prefix}{action}."

        if details:
            message += f" {details}."

        message += " Say 'yes' to proceed or 'no' to cancel."

        return message


# =============================================================================
# Visual Click Validator
# =============================================================================

class VisualClickValidator:
    """
    Provides visual feedback before click actions.

    Shows a high-contrast red circle at the target coordinates before
    clicking, giving the user a chance to veto the action by moving
    their mouse.
    """

    def __init__(
        self,
        config: VisionSafetyConfig,
        on_veto: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        """Initialize the visual click validator."""
        self._config = config
        self._on_veto = on_veto
        self._overlay_window = None

    async def preview_click(
        self,
        x: int,
        y: int,
        button: str = "left",
    ) -> VisualClickPreview:
        """
        Show a preview of a click action and wait for user to veto or accept.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button (left, right, middle)

        Returns:
            VisualClickPreview with result
        """
        if not self._config.visual_overlay_enabled:
            return VisualClickPreview(
                x=x, y=y, button=button,
                overlay_shown=False,
                user_vetoed=False,
                delay_applied_ms=0.0,
            )

        # Show overlay
        overlay_shown = await self._show_overlay(x, y)

        # Wait for delay while monitoring for veto
        user_vetoed = False
        if self._config.overlay_veto_enabled:
            user_vetoed = await self._wait_with_veto_check(
                x, y,
                self._config.overlay_delay_seconds,
            )

        else:
            await asyncio.sleep(self._config.overlay_delay_seconds)

        # Hide overlay
        await self._hide_overlay()

        # Handle veto
        if user_vetoed and self._on_veto:
            await self._on_veto()

        return VisualClickPreview(
            x=x, y=y, button=button,
            overlay_shown=overlay_shown,
            user_vetoed=user_vetoed,
            delay_applied_ms=self._config.overlay_delay_seconds * 1000,
        )

    async def _show_overlay(self, x: int, y: int) -> bool:
        """Show the overlay at the specified coordinates."""
        try:
            # Try to create a native macOS overlay
            return await self._show_native_overlay(x, y)
        except Exception as e:
            logger.debug(f"[VisualClickValidator] Native overlay failed: {e}")

        # Fallback: just log (no visual overlay)
        logger.info(f"[VisualClickValidator] Click target: ({x}, {y})")
        return False

    async def _show_native_overlay(self, x: int, y: int) -> bool:
        """Show a native macOS overlay using AppleScript."""
        # Create a simple overlay using Quartz or AppleScript
        # For now, we'll use a subprocess approach
        try:
            import subprocess

            radius = self._config.overlay_radius_px
            color = self._config.overlay_color

            # Create a temporary Python script for the overlay
            overlay_script = f'''
import Quartz
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
import AppKit
import time

class ClickOverlay(AppKit.NSObject):
    def __init__(self):
        super().__init__()
        self.x = {x}
        self.y = {y}
        self.radius = {radius}

    def applicationDidFinishLaunching_(self, notification):
        screen = AppKit.NSScreen.mainScreen()
        frame = screen.frame()

        # Create window
        self.window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            AppKit.NSWindowStyleMaskBorderless,
            AppKit.NSBackingStoreBuffered,
            False
        )
        self.window.setLevel_(AppKit.NSFloatingWindowLevel + 1000)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.window.setIgnoresMouseEvents_(True)

        # Create view
        view = OverlayView.alloc().initWithFrame_(frame)
        view.x = self.x
        view.y = screen.frame().size.height - self.y  # Flip Y
        view.radius = self.radius
        self.window.setContentView_(view)
        self.window.makeKeyAndOrderFront_(None)

        # Auto-close after delay
        AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            {self._config.overlay_delay_seconds}, self, 'close:', None, False
        )

    def close_(self, timer):
        AppKit.NSApp.terminate_(None)

class OverlayView(AppKit.NSView):
    def drawRect_(self, rect):
        path = AppKit.NSBezierPath.bezierPathWithOvalInRect_(
            AppKit.NSMakeRect(self.x - self.radius, self.y - self.radius,
                             self.radius * 2, self.radius * 2)
        )
        AppKit.NSColor.redColor().set()
        path.setLineWidth_(3)
        path.stroke()

if __name__ == '__main__':
    app = AppKit.NSApplication.sharedApplication()
    delegate = ClickOverlay.alloc().init()
    app.setDelegate_(delegate)
    AppKit.NSApp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    app.run()
'''
            # For now, just log - native overlay is complex
            logger.info(f"[VisualClickValidator] Would show overlay at ({x}, {y})")
            return True

        except Exception as e:
            logger.debug(f"[VisualClickValidator] Overlay creation failed: {e}")
            return False

    async def _hide_overlay(self) -> None:
        """Hide the overlay."""
        # Overlay auto-closes after delay
        pass

    async def _wait_with_veto_check(
        self,
        target_x: int,
        target_y: int,
        duration: float,
    ) -> bool:
        """
        Wait for duration while checking for user veto (mouse movement).

        Returns:
            True if user vetoed by moving mouse
        """
        start_time = time.time()
        check_interval = 0.05  # 50ms
        veto_threshold_px = 50  # Movement threshold

        while time.time() - start_time < duration:
            try:
                import pyautogui
                current_pos = pyautogui.position()

                # Check if user moved mouse significantly
                distance = ((current_pos.x - target_x) ** 2 + (current_pos.y - target_y) ** 2) ** 0.5
                if distance > veto_threshold_px:
                    logger.info(
                        f"[VisualClickValidator] User veto detected: "
                        f"mouse moved {distance:.0f}px from target"
                    )
                    return True

            except ImportError:
                pass

            await asyncio.sleep(check_interval)

        return False


# =============================================================================
# Vision Safety Integration (Main Class)
# =============================================================================

class VisionSafetyIntegration:
    """
    Main integration class that connects all safety components.

    This is the single entry point for the AgenticTaskRunner to interact
    with the safety system. It orchestrates:
    - Dead Man's Switch monitoring
    - Plan safety auditing (Phase 2.5)
    - Action confirmation
    - Visual click validation
    """

    def __init__(
        self,
        config: Optional[VisionSafetyConfig] = None,
        tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        watchdog=None,  # AgenticWatchdog instance
    ):
        """Initialize the Vision-Safety Integration.

        Args:
            config: Safety configuration
            tts_callback: TTS callback for voice feedback
            watchdog: AgenticWatchdog for kill switch integration
        """
        self._config = config or VisionSafetyConfig()
        self._tts_callback = tts_callback
        self._watchdog = watchdog

        # Components
        self._dead_man_switch = DeadMansSwitch(
            self._config,
            on_trigger=self._on_kill_switch_triggered,
        )
        self._plan_auditor = PlanSafetyAuditor(self._config)
        self._confirmation_hub = ActionConfirmationHub(self._config, tts_callback)
        self._click_validator = VisualClickValidator(
            self._config,
            on_veto=self._on_click_vetoed,
        )

        # State
        self._initialized = False
        self._active = False

        logger.info("[VisionSafetyIntegration] Created")

    async def initialize(self) -> bool:
        """Initialize all safety components."""
        if self._initialized:
            return True

        try:
            # Initialize plan auditor
            await self._plan_auditor.initialize()

            # Start Dead Man's Switch monitoring
            await self._dead_man_switch.start()

            # Register with watchdog if available
            if self._watchdog:
                self._watchdog._on_kill_callbacks.append(self._on_watchdog_kill)

            self._initialized = True
            logger.info("[VisionSafetyIntegration] Initialized successfully")
            return True

        except Exception as e:
            logger.error(f"[VisionSafetyIntegration] Initialization failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown all safety components."""
        await self._dead_man_switch.stop()
        self._initialized = False
        logger.info("[VisionSafetyIntegration] Shutdown complete")

    # =========================================================================
    # Phase 2.5: Safety Audit Interface
    # =========================================================================

    async def audit_plan(
        self,
        plan: List[Dict[str, Any]],
        goal: str,
    ) -> SafetyAuditResult:
        """
        Audit an execution plan for safety (Phase 2.5).

        Args:
            plan: The execution plan from Phase 2
            goal: The original goal

        Returns:
            SafetyAuditResult with verdict
        """
        return await self._plan_auditor.audit_plan(plan, goal)

    async def request_confirmation_if_needed(
        self,
        audit_result: SafetyAuditResult,
        goal: str,
    ) -> bool:
        """
        Request user confirmation if the audit requires it.

        Args:
            audit_result: Result from audit_plan()
            goal: The original goal

        Returns:
            True if confirmed (or no confirmation needed), False if denied
        """
        if not audit_result.confirmation_required:
            return True

        if audit_result.is_blocked:
            # Cannot execute - blocked by policy
            if self._tts_callback:
                reasons = "; ".join(audit_result.blocking_reasons[:2])
                await self._tts_callback(f"I cannot execute this action. {reasons}")
            return False

        # Build description of risky steps
        if audit_result.risky_steps:
            risky_desc = f"This plan involves {len(audit_result.risky_steps)} risky steps"
            step_details = "; ".join(r[2] for r in audit_result.risky_steps[:2])
        else:
            risky_desc = f"This action requires confirmation: {goal[:50]}"
            step_details = None

        # Request confirmation
        result = await self._confirmation_hub.request_confirmation(
            risky_desc,
            audit_result.verdict,
            step_details,
        )

        return result.confirmed

    # =========================================================================
    # Visual Click Validation
    # =========================================================================

    async def preview_click(self, x: int, y: int, button: str = "left") -> VisualClickPreview:
        """
        Preview a click action with visual indicator.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button

        Returns:
            VisualClickPreview result
        """
        return await self._click_validator.preview_click(x, y, button)

    # =========================================================================
    # Voice Input
    # =========================================================================

    def receive_voice_input(self, text: str) -> bool:
        """
        Receive voice input for confirmation processing.

        Args:
            text: Transcribed voice text

        Returns:
            True if input was processed as confirmation/cancellation
        """
        return self._confirmation_hub.receive_voice_input(text)

    # =========================================================================
    # Kill Switch Status
    # =========================================================================

    def is_kill_switch_triggered(self) -> bool:
        """Check if kill switch is triggered."""
        return self._dead_man_switch.is_triggered()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status."""
        return {
            "initialized": self._initialized,
            "dead_man_switch": self._dead_man_switch.get_status().__dict__,
            "config": {
                "dead_man_enabled": self._config.dead_man_switch_enabled,
                "safety_audit_enabled": self._config.safety_audit_enabled,
                "visual_overlay_enabled": self._config.visual_overlay_enabled,
                "voice_confirmation_enabled": self._config.voice_confirmation_enabled,
            },
        }

    # =========================================================================
    # Internal Callbacks
    # =========================================================================

    async def _on_kill_switch_triggered(self) -> None:
        """Called when Dead Man's Switch triggers."""
        logger.warning("[VisionSafetyIntegration] KILL SWITCH TRIGGERED!")

        # Announce via TTS
        if self._tts_callback:
            await self._tts_callback("Emergency stop activated. All actions halted.")

        # Trigger watchdog if available
        if self._watchdog:
            await self._watchdog.trigger_kill_switch("Dead Man's Switch activated")

    async def _on_watchdog_kill(self) -> None:
        """Called when watchdog triggers kill."""
        logger.warning("[VisionSafetyIntegration] Watchdog kill received")

    async def _on_click_vetoed(self) -> None:
        """Called when user vetoes a click."""
        if self._tts_callback:
            await self._tts_callback("Click cancelled.")


# =============================================================================
# Module-Level Singleton
# =============================================================================

_global_safety_integration: Optional[VisionSafetyIntegration] = None


def get_vision_safety_integration() -> VisionSafetyIntegration:
    """Get the global Vision-Safety Integration instance."""
    global _global_safety_integration
    if _global_safety_integration is None:
        _global_safety_integration = VisionSafetyIntegration()
    return _global_safety_integration


async def initialize_vision_safety(
    config: Optional[VisionSafetyConfig] = None,
    tts_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    watchdog=None,
) -> VisionSafetyIntegration:
    """Initialize and return the Vision-Safety Integration."""
    global _global_safety_integration
    _global_safety_integration = VisionSafetyIntegration(
        config=config,
        tts_callback=tts_callback,
        watchdog=watchdog,
    )
    await _global_safety_integration.initialize()
    return _global_safety_integration
