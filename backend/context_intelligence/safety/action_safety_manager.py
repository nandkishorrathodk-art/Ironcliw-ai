"""
Action Safety Manager for Ironcliw
=================================

Manages safety confirmations and risk assessment for actions

Features:
- Safety level evaluation
- User confirmation requests
- Automatic approval for safe actions
- Extra warnings for risky actions
- Trusted action allowlist
- Cross-repo event emission to Reactor Core (v10.3)
- Safety context for Ironcliw Prime routing (v10.3)

Author: Derek Russell
Date: 2025-10-19
Updated: 2025-12-25 (v10.3 - Cross-Repo Safety Integration)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, Set, Callable, Awaitable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from context_intelligence.analyzers.action_analyzer import ActionSafety
from context_intelligence.planners.action_planner import ExecutionPlan, ExecutionStep

# Cross-repo integration (v10.3)
if TYPE_CHECKING:
    from clients.reactor_core_client import ReactorCoreClient

logger = logging.getLogger(__name__)

# Cross-repo safety state directory
SAFETY_STATE_DIR = Path.home() / ".jarvis" / "cross_repo" / "safety"


# ============================================================================
# CONFIRMATION RESULTS
# ============================================================================

@dataclass
class ConfirmationResult:
    """Result of a confirmation request"""
    approved: bool
    confirmation_message: str
    user_response: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confirmation_method: str = "auto"  # auto, voice, text, timeout


@dataclass
class SafetyContext:
    """
    Current safety context for cross-repo sharing (v10.3).

    This context is written to disk for Ironcliw Prime to read,
    enabling safety-aware routing and reasoning.
    """
    kill_switch_active: bool = False
    current_risk_level: str = "low"  # low, medium, high, critical
    pending_confirmation: bool = False
    recent_blocks: int = 0
    recent_confirmations: int = 0
    recent_denials: int = 0
    user_trust_level: float = 1.0
    last_update: datetime = field(default_factory=datetime.now)
    session_start: datetime = field(default_factory=datetime.now)
    total_audits: int = 0
    total_blocks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kill_switch_active": self.kill_switch_active,
            "current_risk_level": self.current_risk_level,
            "pending_confirmation": self.pending_confirmation,
            "recent_blocks": self.recent_blocks,
            "recent_confirmations": self.recent_confirmations,
            "recent_denials": self.recent_denials,
            "user_trust_level": self.user_trust_level,
            "last_update": self.last_update.isoformat(),
            "session_start": self.session_start.isoformat(),
            "total_audits": self.total_audits,
            "total_blocks": self.total_blocks,
        }

    def to_prime_context_string(self) -> str:
        """Generate context string for Ironcliw Prime prompt injection."""
        lines = ["[Ironcliw SAFETY CONTEXT]"]

        if self.kill_switch_active:
            lines.append("- KILL SWITCH ACTIVE: All actions paused")

        if self.current_risk_level in ("high", "critical"):
            lines.append(f"- Risk Level: {self.current_risk_level.upper()}")

        if self.pending_confirmation:
            lines.append("- Awaiting user confirmation for risky action")

        if self.recent_blocks > 0:
            lines.append(f"- Recently blocked {self.recent_blocks} risky action(s)")

        if self.recent_denials > 0:
            lines.append(f"- User denied {self.recent_denials} action(s) recently")

        if self.user_trust_level < 0.7:
            lines.append("- User exercising caution with risky actions")

        if len(lines) == 1:
            lines.append("- All clear, normal operation")

        lines.append("[/Ironcliw SAFETY CONTEXT]")
        return "\n".join(lines)


# ============================================================================
# ACTION SAFETY MANAGER
# ============================================================================

class ActionSafetyManager:
    """
    Manages safety confirmations for actions

    Handles:
    - Automatic approval for safe actions
    - User confirmation for risky actions
    - Trusted action patterns
    - Safety level evaluation
    - Cross-repo event emission (v10.3)
    - Safety context for Ironcliw Prime (v10.3)
    """

    def __init__(
        self,
        auto_approve_safe: bool = True,
        enable_confirmations: bool = True,
        enable_cross_repo: bool = True,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the safety manager

        Args:
            auto_approve_safe: Automatically approve SAFE actions
            enable_confirmations: Enable user confirmations (False = auto-approve all)
            enable_cross_repo: Enable cross-repo safety event emission (v10.3)
            session_id: Session ID for cross-repo tracking
        """
        self.auto_approve_safe = auto_approve_safe
        self.enable_confirmations = enable_confirmations
        self.enable_cross_repo = enable_cross_repo
        self.session_id = session_id or f"jarvis-{int(datetime.now().timestamp())}"

        # Trusted actions that can be auto-approved
        self.trusted_actions = self._initialize_trusted_actions()

        # Confirmation callback (set by integration layer)
        self.confirmation_callback = None

        # Cross-repo integration (v10.3)
        self._safety_context = SafetyContext()
        self._reactor_client: Optional["ReactorCoreClient"] = None
        self._event_callbacks: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []

        # Initialize cross-repo state directory
        if self.enable_cross_repo:
            self._init_cross_repo_state()

        logger.info(f"[SAFETY-MANAGER] Initialized (auto_approve_safe={auto_approve_safe}, confirmations={enable_confirmations}, cross_repo={enable_cross_repo})")

    def _init_cross_repo_state(self) -> None:
        """Initialize cross-repo state directory and files."""
        try:
            SAFETY_STATE_DIR.mkdir(parents=True, exist_ok=True)
            self._write_safety_context()
        except Exception as e:
            logger.warning(f"[SAFETY-MANAGER] Failed to init cross-repo state: {e}")

    def _write_safety_context(self) -> None:
        """Write safety context to disk for Ironcliw Prime."""
        if not self.enable_cross_repo:
            return

        try:
            context_file = SAFETY_STATE_DIR / "context_for_prime.json"
            self._safety_context.last_update = datetime.now()
            context_file.write_text(json.dumps(self._safety_context.to_dict(), indent=2))
        except Exception as e:
            logger.warning(f"[SAFETY-MANAGER] Failed to write context: {e}")

    async def _emit_safety_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """Emit safety event to Reactor Core and callbacks (v10.3)."""
        if not self.enable_cross_repo:
            return

        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "payload": payload,
        }

        # Write event to file for Reactor Core to pick up
        try:
            events_dir = SAFETY_STATE_DIR / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            event_file = events_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{event_type}.json"
            event_file.write_text(json.dumps(event, indent=2))
        except Exception as e:
            logger.warning(f"[SAFETY-MANAGER] Failed to write event: {e}")

        # Call registered callbacks
        for callback in self._event_callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.warning(f"[SAFETY-MANAGER] Event callback error: {e}")

        # Send to Reactor Core client if connected
        if self._reactor_client:
            try:
                await self._reactor_client.emit_safety_event(event_type, payload)
            except Exception as e:
                logger.warning(f"[SAFETY-MANAGER] Failed to send to Reactor Core: {e}")

    def _initialize_trusted_actions(self) -> Set[str]:
        """Initialize set of trusted action patterns"""
        return {
            "yabai -m space --focus",  # Safe space switching
            "yabai -m window --focus",  # Safe window focusing
            "open http",  # Safe URL opening
        }

    async def request_confirmation(
        self,
        plan: ExecutionPlan,
        context: Optional[Dict[str, Any]] = None
    ) -> ConfirmationResult:
        """
        Request confirmation for an action plan

        Args:
            plan: The execution plan
            context: Additional context

        Returns:
            ConfirmationResult with approval decision
        """
        logger.info(f"[SAFETY-MANAGER] Requesting confirmation for plan: {plan.plan_id}")

        # Update safety context - we're auditing an action
        self._safety_context.total_audits += 1
        self._safety_context.current_risk_level = self._map_safety_to_risk(plan.safety_level)

        # Emit safety audit event (v10.3)
        await self._emit_safety_event("safety_audit", {
            "plan_id": plan.plan_id,
            "action_type": plan.action_intent.action_type.value if hasattr(plan, 'action_intent') else "unknown",
            "safety_level": plan.safety_level.value,
            "step_count": len(plan.steps),
            "steps": [{"command": s.command, "description": s.description} for s in plan.steps[:3]],  # First 3 steps
            "context": context or {},
        })

        # Check if confirmations are disabled
        if not self.enable_confirmations:
            logger.info("[SAFETY-MANAGER] Confirmations disabled - auto-approving")
            result = ConfirmationResult(
                approved=True,
                confirmation_message="Auto-approved (confirmations disabled)",
                metadata={"auto_approved": True}
            )
            await self._handle_confirmation_result(plan, result, "auto_disabled")
            return result

        # Auto-approve SAFE actions
        if plan.safety_level == ActionSafety.SAFE and self.auto_approve_safe:
            logger.info("[SAFETY-MANAGER] Auto-approving SAFE action")
            result = ConfirmationResult(
                approved=True,
                confirmation_message="Auto-approved (safe action)",
                metadata={"auto_approved": True, "safety_level": "SAFE"}
            )
            await self._handle_confirmation_result(plan, result, "auto_safe")
            return result

        # Check if all steps are trusted
        if self._all_steps_trusted(plan.steps):
            logger.info("[SAFETY-MANAGER] Auto-approving trusted actions")
            result = ConfirmationResult(
                approved=True,
                confirmation_message="Auto-approved (trusted actions)",
                metadata={"auto_approved": True, "trusted": True}
            )
            await self._handle_confirmation_result(plan, result, "auto_trusted")
            return result

        # Generate confirmation message
        message = self._generate_confirmation_message(plan)

        # Mark pending confirmation in context
        self._safety_context.pending_confirmation = True
        self._write_safety_context()

        # Request confirmation from user
        if self.confirmation_callback:
            approved = await self.confirmation_callback(message, plan)
        else:
            # Default: approve for now (in real system, would wait for user input)
            logger.warning("[SAFETY-MANAGER] No confirmation callback set - auto-approving")
            approved = True

        # Clear pending confirmation
        self._safety_context.pending_confirmation = False

        logger.info(f"[SAFETY-MANAGER] Confirmation result: approved={approved}")

        result = ConfirmationResult(
            approved=approved,
            confirmation_message=message,
            confirmation_method="callback" if self.confirmation_callback else "auto_fallback",
            metadata={
                "safety_level": plan.safety_level.value,
                "step_count": len(plan.steps)
            }
        )

        await self._handle_confirmation_result(plan, result, "user_confirmation")
        return result

    async def _handle_confirmation_result(
        self,
        plan: ExecutionPlan,
        result: ConfirmationResult,
        confirmation_type: str,
    ) -> None:
        """Handle confirmation result - emit events and update context (v10.3)."""
        if result.approved:
            self._safety_context.recent_confirmations += 1
            await self._emit_safety_event("safety_confirmed", {
                "plan_id": plan.plan_id,
                "safety_level": plan.safety_level.value,
                "confirmation_type": confirmation_type,
                "message": result.confirmation_message,
            })
        else:
            self._safety_context.recent_denials += 1
            self._safety_context.total_blocks += 1
            # Reduce trust level slightly on denial
            self._safety_context.user_trust_level = max(
                0.5, self._safety_context.user_trust_level - 0.05
            )
            await self._emit_safety_event("safety_denied", {
                "plan_id": plan.plan_id,
                "safety_level": plan.safety_level.value,
                "confirmation_type": confirmation_type,
                "message": result.confirmation_message,
            })

        self._write_safety_context()

    def _map_safety_to_risk(self, safety_level: ActionSafety) -> str:
        """Map ActionSafety to risk level string."""
        mapping = {
            ActionSafety.SAFE: "low",
            ActionSafety.NEEDS_CONFIRMATION: "medium",
            ActionSafety.RISKY: "high",
        }
        return mapping.get(safety_level, "medium")

    def _generate_confirmation_message(self, plan: ExecutionPlan) -> str:
        """Generate human-readable confirmation message"""
        action_name = plan.action_intent.action_type.value.replace("_", " ").title()

        msg = f"I'm about to {action_name.lower()}:\n\n"

        # List steps
        for i, step in enumerate(plan.steps, 1):
            msg += f"  {i}. {step.description}\n"

        # Add safety warning
        if plan.safety_level == ActionSafety.RISKY:
            msg += "\n⚠️  WARNING: This action may be irreversible!\n"
        elif plan.safety_level == ActionSafety.NEEDS_CONFIRMATION:
            msg += "\n⚠️  This action requires confirmation.\n"

        # Add resolution info if references were resolved
        if plan.resolved_references:
            msg += "\n📍 Resolved references:\n"
            for key, value in plan.resolved_references.items():
                if key in ["referent_entity", "app_name", "space_id"]:
                    msg += f"  - {key}: {value}\n"

        msg += "\nProceed? (yes/no)"

        return msg

    def _all_steps_trusted(self, steps: List[ExecutionStep]) -> bool:
        """Check if all steps are trusted"""
        for step in steps:
            # Check if command matches trusted patterns
            if not any(
                step.command.startswith(trusted)
                for trusted in self.trusted_actions
            ):
                return False

        return True

    def set_confirmation_callback(self, callback):
        """
        Set the confirmation callback function

        Args:
            callback: Async function(message, plan) -> bool
        """
        self.confirmation_callback = callback
        logger.info("[SAFETY-MANAGER] Confirmation callback set")

    def add_trusted_action(self, action_pattern: str):
        """Add a trusted action pattern"""
        self.trusted_actions.add(action_pattern)
        logger.info(f"[SAFETY-MANAGER] Added trusted action: {action_pattern}")

    def is_action_safe(self, plan: ExecutionPlan) -> bool:
        """Check if an action is safe to execute without confirmation"""
        return (
            plan.safety_level == ActionSafety.SAFE
            or self._all_steps_trusted(plan.steps)
        )

    # ========================================================================
    # CROSS-REPO INTEGRATION (v10.3)
    # ========================================================================

    def set_reactor_client(self, client: "ReactorCoreClient") -> None:
        """
        Set the Reactor Core client for cross-repo event emission.

        Args:
            client: ReactorCoreClient instance
        """
        self._reactor_client = client
        logger.info("[SAFETY-MANAGER] Reactor Core client connected")

    def register_event_callback(
        self, callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Register a callback for safety events.

        Args:
            callback: Async function(event_dict) to call on events
        """
        self._event_callbacks.append(callback)
        logger.info(f"[SAFETY-MANAGER] Event callback registered (total: {len(self._event_callbacks)})")

    def get_safety_context(self) -> SafetyContext:
        """Get current safety context for cross-repo sharing."""
        return self._safety_context

    def get_safety_context_for_prime(self) -> str:
        """
        Get safety context formatted for Ironcliw Prime prompt injection.

        Returns:
            Formatted string to inject into Prime's context
        """
        return self._safety_context.to_prime_context_string()

    def get_safety_context_dict(self) -> Dict[str, Any]:
        """Get safety context as dictionary for API responses."""
        return self._safety_context.to_dict()

    # ========================================================================
    # KILL SWITCH (v10.3)
    # ========================================================================

    async def activate_kill_switch(self, reason: str = "manual") -> None:
        """
        Activate the kill switch - blocks ALL actions.

        Args:
            reason: Reason for activation (manual, anomaly, threat, etc.)
        """
        self._safety_context.kill_switch_active = True
        self._safety_context.current_risk_level = "critical"
        self._write_safety_context()

        logger.warning(f"[SAFETY-MANAGER] 🛑 KILL SWITCH ACTIVATED - Reason: {reason}")

        await self._emit_safety_event("kill_switch_triggered", {
            "action": "activated",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

    async def deactivate_kill_switch(self, authorized_by: str = "user") -> None:
        """
        Deactivate the kill switch - resume normal operation.

        Args:
            authorized_by: Who authorized deactivation
        """
        self._safety_context.kill_switch_active = False
        self._safety_context.current_risk_level = "low"
        self._write_safety_context()

        logger.info(f"[SAFETY-MANAGER] ✅ Kill switch deactivated by: {authorized_by}")

        await self._emit_safety_event("kill_switch_triggered", {
            "action": "deactivated",
            "authorized_by": authorized_by,
            "timestamp": datetime.now().isoformat(),
        })

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self._safety_context.kill_switch_active

    async def check_kill_switch(self) -> bool:
        """
        Check kill switch before any action.

        Returns:
            True if action should be blocked (kill switch active)
        """
        if self._safety_context.kill_switch_active:
            self._safety_context.recent_blocks += 1
            self._safety_context.total_blocks += 1
            self._write_safety_context()

            await self._emit_safety_event("safety_blocked", {
                "reason": "kill_switch_active",
                "message": "All actions paused - kill switch is active",
            })
            return True
        return False

    # ========================================================================
    # VISUAL CLICK SAFETY (v10.3)
    # ========================================================================

    async def preview_visual_click(
        self,
        target_x: int,
        target_y: int,
        action_description: str,
        overlay_duration_ms: int = 2000,
    ) -> ConfirmationResult:
        """
        Show visual preview before click action (Red Circle Overlay).

        Args:
            target_x: X coordinate
            target_y: Y coordinate
            action_description: Description of the click action
            overlay_duration_ms: How long to show overlay

        Returns:
            ConfirmationResult with approval decision
        """
        # Emit preview event for UI to render overlay
        await self._emit_safety_event("visual_click_preview", {
            "x": target_x,
            "y": target_y,
            "description": action_description,
            "overlay_duration_ms": overlay_duration_ms,
        })

        # If we have a confirmation callback, use it
        if self.confirmation_callback:
            message = f"Click at ({target_x}, {target_y}): {action_description}"
            approved = await self.confirmation_callback(message, None)
        else:
            # Auto-approve with delay for visual preview
            await asyncio.sleep(overlay_duration_ms / 1000)
            approved = True

        if not approved:
            await self._emit_safety_event("visual_click_vetoed", {
                "x": target_x,
                "y": target_y,
                "description": action_description,
                "reason": "user_cancelled",
            })
            self._safety_context.recent_denials += 1

        self._write_safety_context()

        return ConfirmationResult(
            approved=approved,
            confirmation_message=f"Click at ({target_x}, {target_y})",
            confirmation_method="visual_preview",
            metadata={
                "x": target_x,
                "y": target_y,
                "action": action_description,
            }
        )

    # ========================================================================
    # RISK ASSESSMENT HELPERS (v10.3)
    # ========================================================================

    async def assess_command_risk(
        self,
        command: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assess risk of a raw command string.

        Args:
            command: The command to assess
            context: Optional context

        Returns:
            Risk assessment dictionary
        """
        # Check for dangerous patterns
        dangerous_patterns = [
            ("rm -rf", "critical", "Recursive force delete"),
            ("sudo rm", "high", "Privileged file deletion"),
            ("chmod 777", "high", "Insecure permissions"),
            ("curl | sh", "critical", "Piped remote script execution"),
            ("wget | bash", "critical", "Piped remote script execution"),
            ("> /dev/", "high", "Direct device write"),
            ("dd if=", "critical", "Disk imaging - potentially destructive"),
            ("mkfs", "critical", "Filesystem formatting"),
            ("shutdown", "high", "System shutdown"),
            ("reboot", "high", "System reboot"),
        ]

        risk_level = "low"
        risk_reason = None
        blocked = False

        for pattern, level, reason in dangerous_patterns:
            if pattern in command.lower():
                risk_level = level
                risk_reason = reason
                if level == "critical":
                    blocked = True
                break

        # Check kill switch
        if self._safety_context.kill_switch_active:
            blocked = True
            risk_reason = "Kill switch active - all commands blocked"

        assessment = {
            "command": command,
            "risk_level": risk_level,
            "risk_reason": risk_reason,
            "blocked": blocked,
            "requires_confirmation": risk_level in ("medium", "high", "critical"),
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Emit assessment event
        if risk_level in ("high", "critical") or blocked:
            await self._emit_safety_event("safety_audit", {
                "type": "command_risk_assessment",
                **assessment,
            })

        return assessment

    def reset_session_counters(self) -> None:
        """Reset session-based counters (call at session start)."""
        self._safety_context.recent_blocks = 0
        self._safety_context.recent_confirmations = 0
        self._safety_context.recent_denials = 0
        self._safety_context.session_start = datetime.now()
        self._write_safety_context()
        logger.info("[SAFETY-MANAGER] Session counters reset")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        session_duration = datetime.now() - self._safety_context.session_start
        return {
            "session_duration_seconds": session_duration.total_seconds(),
            "total_audits": self._safety_context.total_audits,
            "total_blocks": self._safety_context.total_blocks,
            "recent_blocks": self._safety_context.recent_blocks,
            "recent_confirmations": self._safety_context.recent_confirmations,
            "recent_denials": self._safety_context.recent_denials,
            "user_trust_level": self._safety_context.user_trust_level,
            "kill_switch_active": self._safety_context.kill_switch_active,
            "current_risk_level": self._safety_context.current_risk_level,
        }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_safety_manager: Optional[ActionSafetyManager] = None


def get_action_safety_manager() -> Optional[ActionSafetyManager]:
    """Get the global action safety manager instance"""
    return _global_safety_manager


def initialize_action_safety_manager(**kwargs) -> ActionSafetyManager:
    """Initialize the global action safety manager"""
    global _global_safety_manager
    _global_safety_manager = ActionSafetyManager(**kwargs)
    logger.info("[SAFETY-MANAGER] Global instance initialized")
    return _global_safety_manager
