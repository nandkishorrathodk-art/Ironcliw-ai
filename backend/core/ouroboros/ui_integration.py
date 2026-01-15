"""
Ouroboros UI Integration v1.0
=============================

Bridges the Native Self-Improvement Engine with JARVIS UI components:
- Menu bar status indicator
- WebSocket broadcasts
- Event bus integration
- Voice announcements

This makes self-improvement feel like a natural part of JARVIS,
with progress visible in the UI without disrupting the user.

Author: Trinity System
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import logging
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.ouroboros.native_integration import (
        ImprovementProgress,
        ImprovementPhase,
        ProgressBroadcaster,
    )

logger = logging.getLogger("Ouroboros.UIIntegration")


# =============================================================================
# UI STATE MAPPING
# =============================================================================

class UIActivityState(str, Enum):
    """Activity states for UI display."""
    IDLE = "idle"
    IMPROVING = "improving"
    TESTING = "testing"
    LEARNING = "learning"
    ERROR = "error"


# Phase to UI activity mapping
PHASE_TO_ACTIVITY = {
    "initializing": UIActivityState.IMPROVING,
    "analyzing": UIActivityState.IMPROVING,
    "generating": UIActivityState.IMPROVING,
    "validating": UIActivityState.TESTING,
    "testing": UIActivityState.TESTING,
    "applying": UIActivityState.IMPROVING,
    "committing": UIActivityState.IMPROVING,
    "learning": UIActivityState.LEARNING,
    "completed": UIActivityState.IDLE,
    "failed": UIActivityState.ERROR,
}


# Phase to icon mapping (for menu bar)
PHASE_ICONS = {
    "initializing": "🔧",
    "analyzing": "🔍",
    "generating": "⚡",
    "validating": "✅",
    "testing": "🧪",
    "applying": "📝",
    "committing": "💾",
    "learning": "🧠",
    "completed": "✨",
    "failed": "❌",
}


# =============================================================================
# UI INTEGRATION CONTROLLER
# =============================================================================

class OuroborosUIController:
    """
    Controller that bridges Ouroboros with JARVIS UI components.

    Handles:
    - Menu bar updates
    - WebSocket broadcasts
    - Event bus publishing
    - Voice announcements
    """

    def __init__(self):
        self.logger = logging.getLogger("Ouroboros.UIController")

        # UI component references (set during integration)
        self._menu_bar_indicator = None
        self._websocket_coordinator = None
        self._event_bus = None
        self._voice_integration = None

        # Progress tracking
        self._active_improvements: Dict[str, Dict[str, Any]] = {}
        self._listener_removal: Optional[Callable[[], None]] = None

        # Configuration
        self._announce_completions = True
        self._broadcast_progress = True

    async def connect(
        self,
        menu_bar=None,
        websocket_coordinator=None,
        event_bus=None,
        voice_integration=None,
    ) -> None:
        """
        Connect to UI components.

        Args:
            menu_bar: Menu bar status indicator instance
            websocket_coordinator: WebSocket coordinator for broadcasts
            event_bus: MacOS event bus for system events
            voice_integration: Voice integration for announcements
        """
        self._menu_bar_indicator = menu_bar
        self._websocket_coordinator = websocket_coordinator
        self._event_bus = event_bus
        self._voice_integration = voice_integration

        # Connect to native self-improvement progress broadcaster
        try:
            from backend.core.ouroboros.native_integration import get_native_self_improvement
            engine = get_native_self_improvement()

            # Add progress listener
            self._listener_removal = engine.progress_broadcaster.add_listener(
                self._on_progress_update
            )

            self.logger.info("Connected Ouroboros UI controller to progress broadcaster")

        except ImportError as e:
            self.logger.warning(f"Could not connect to native self-improvement: {e}")

    async def disconnect(self) -> None:
        """Disconnect from UI components."""
        if self._listener_removal:
            self._listener_removal()
            self._listener_removal = None

        self._menu_bar_indicator = None
        self._websocket_coordinator = None
        self._event_bus = None
        self._voice_integration = None

        self.logger.info("Disconnected Ouroboros UI controller")

    async def _on_progress_update(self, progress: "ImprovementProgress") -> None:
        """
        Handle progress updates from the native self-improvement engine.

        This is called automatically when improvement progress changes.
        """
        task_id = progress.task_id
        phase = progress.phase.value

        # Update local tracking
        self._active_improvements[task_id] = progress.to_dict()

        # Update menu bar
        await self._update_menu_bar(progress)

        # Broadcast via WebSocket
        if self._broadcast_progress:
            await self._broadcast_websocket(progress)

        # Publish to event bus
        await self._publish_event(progress)

        # Voice announcement for key phases
        await self._announce_progress(progress)

        # Clean up completed tasks
        if phase in ("completed", "failed"):
            await asyncio.sleep(5.0)  # Keep in list briefly
            self._active_improvements.pop(task_id, None)

    async def _update_menu_bar(self, progress: "ImprovementProgress") -> None:
        """Update the menu bar status indicator."""
        if not self._menu_bar_indicator:
            return

        try:
            phase = progress.phase.value
            icon = PHASE_ICONS.get(phase, "⏳")
            activity = PHASE_TO_ACTIVITY.get(phase, UIActivityState.IDLE)

            # Build status message
            if phase in ("completed", "failed"):
                message = None  # Clear the activity message
            else:
                message = f"{icon} {progress.message}"

            # Update menu bar (thread-safe method)
            if hasattr(self._menu_bar_indicator, 'set_activity_status'):
                # Main thread execution for PyObjC
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._menu_bar_indicator.set_activity_status(
                        activity=activity.value,
                        message=message,
                    )
                )

            # Also update stats if available
            if hasattr(self._menu_bar_indicator, 'update_stats'):
                stats = {
                    "ouroboros_active": len(self._active_improvements),
                    "ouroboros_phase": phase,
                    "ouroboros_progress": progress.progress_percent,
                }
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._menu_bar_indicator.update_stats(stats)
                )

        except Exception as e:
            self.logger.warning(f"Menu bar update failed: {e}")

    async def _broadcast_websocket(self, progress: "ImprovementProgress") -> None:
        """Broadcast progress via WebSocket."""
        if not self._websocket_coordinator:
            return

        try:
            message = {
                "type": "ouroboros_progress",
                "topic": "ouroboros_events",
                "timestamp": datetime.utcnow().isoformat(),
                "payload": progress.to_dict(),
            }

            if hasattr(self._websocket_coordinator, 'broadcast'):
                await self._websocket_coordinator.broadcast(
                    topic="ouroboros_events",
                    message=json.dumps(message),
                )
            elif hasattr(self._websocket_coordinator, 'publish'):
                await self._websocket_coordinator.publish(
                    topic="ouroboros_events",
                    message=message,
                )

        except Exception as e:
            self.logger.warning(f"WebSocket broadcast failed: {e}")

    async def _publish_event(self, progress: "ImprovementProgress") -> None:
        """Publish progress to the event bus."""
        if not self._event_bus:
            return

        try:
            event = {
                "type": "OUROBOROS_PROGRESS",
                "source": "ouroboros",
                "timestamp": datetime.utcnow().isoformat(),
                "data": progress.to_dict(),
            }

            if hasattr(self._event_bus, 'emit'):
                await self._event_bus.emit(event)
            elif hasattr(self._event_bus, 'publish'):
                await self._event_bus.publish("ouroboros", event)

        except Exception as e:
            self.logger.warning(f"Event bus publish failed: {e}")

    async def _announce_progress(self, progress: "ImprovementProgress") -> None:
        """Voice announcement for key progress phases."""
        if not self._voice_integration or not self._announce_completions:
            return

        phase = progress.phase.value

        # Only announce key phases
        if phase == "completed":
            message = f"I've finished improving {progress.target_file}. "
            if progress.iteration == 1:
                message += "Got it on the first try!"
            else:
                message += f"It took {progress.iteration} attempts."

        elif phase == "failed":
            message = (
                f"I wasn't able to improve {progress.target_file}. "
                f"The issue was: {progress.error or 'unknown error'}."
            )
        else:
            return  # Don't announce intermediate phases

        try:
            if hasattr(self._voice_integration, 'announce'):
                await self._voice_integration.announce(
                    message,
                    priority="low",
                    interruptible=True,
                )
            elif hasattr(self._voice_integration, 'speak'):
                await self._voice_integration.speak(message)

        except Exception as e:
            self.logger.warning(f"Voice announcement failed: {e}")

    def get_active_improvements(self) -> List[Dict[str, Any]]:
        """Get list of active improvements for UI display."""
        return list(self._active_improvements.values())

    def get_status(self) -> Dict[str, Any]:
        """Get UI integration status."""
        return {
            "connected_components": {
                "menu_bar": self._menu_bar_indicator is not None,
                "websocket": self._websocket_coordinator is not None,
                "event_bus": self._event_bus is not None,
                "voice": self._voice_integration is not None,
            },
            "active_improvements": len(self._active_improvements),
            "announce_completions": self._announce_completions,
            "broadcast_progress": self._broadcast_progress,
        }


# =============================================================================
# MENU BAR EXTENSION
# =============================================================================

class OuroborosMenuSection:
    """
    Menu section for Ouroboros in the JARVIS menu bar.

    Adds:
    - Active improvements list
    - Quick actions (pause, cancel)
    - Status summary
    """

    def __init__(self, ui_controller: OuroborosUIController):
        self.controller = ui_controller
        self.logger = logging.getLogger("Ouroboros.MenuSection")

    def build_menu_items(self) -> List[Dict[str, Any]]:
        """
        Build menu items for the Ouroboros section.

        Returns list of menu item configurations.
        """
        items = []

        # Header
        items.append({
            "type": "header",
            "title": "🐍 Self-Improvement",
        })

        # Active improvements
        active = self.controller.get_active_improvements()
        if active:
            for improvement in active[:5]:  # Limit to 5
                icon = PHASE_ICONS.get(improvement.get("phase", ""), "⏳")
                target = improvement.get("target_file", "unknown").split("/")[-1]
                progress = improvement.get("progress_percent", 0)

                items.append({
                    "type": "item",
                    "title": f"{icon} {target} ({progress:.0f}%)",
                    "enabled": False,  # Informational only
                    "indent": 1,
                })
        else:
            items.append({
                "type": "item",
                "title": "No active improvements",
                "enabled": False,
                "indent": 1,
            })

        # Separator
        items.append({"type": "separator"})

        # Quick actions
        items.append({
            "type": "item",
            "title": "Improve File...",
            "action": "ouroboros_improve_file",
            "shortcut": "Cmd+Shift+I",
        })

        return items

    def handle_action(self, action: str, context: Optional[Dict] = None) -> None:
        """Handle menu action."""
        if action == "ouroboros_improve_file":
            # This would trigger a file picker dialog
            self.logger.info("User requested file improvement via menu")
            # Integration with actual dialog would go here


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_ui_controller: Optional[OuroborosUIController] = None


def get_ouroboros_ui_controller() -> OuroborosUIController:
    """Get the global Ouroboros UI controller."""
    global _ui_controller
    if _ui_controller is None:
        _ui_controller = OuroborosUIController()
    return _ui_controller


async def connect_ouroboros_ui(
    menu_bar=None,
    websocket_coordinator=None,
    event_bus=None,
    voice_integration=None,
) -> OuroborosUIController:
    """
    Connect Ouroboros to UI components.

    Call this during JARVIS startup after UI components are initialized.
    """
    controller = get_ouroboros_ui_controller()
    await controller.connect(
        menu_bar=menu_bar,
        websocket_coordinator=websocket_coordinator,
        event_bus=event_bus,
        voice_integration=voice_integration,
    )
    return controller


async def disconnect_ouroboros_ui() -> None:
    """Disconnect Ouroboros from UI components."""
    global _ui_controller
    if _ui_controller:
        await _ui_controller.disconnect()
        _ui_controller = None
