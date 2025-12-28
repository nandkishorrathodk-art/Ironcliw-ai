"""
JARVIS Neural Mesh - Visual Monitor Agent v10.6
===============================================

The "Watcher" of Video Multi-Space Intelligence (VMSI).

This agent provides background visual surveillance capabilities:
- Watch background windows for specific events
- Monitor multiple windows in parallel
- Alert when visual events are detected
- Integrate with SpatialAwarenessAgent for window location
- Share state across repos (JARVIS ↔ JARVIS Prime ↔ Reactor Core)

Capabilities:
- watch_and_alert: Monitor a window for text/event and alert
- watch_multiple: Monitor multiple windows in parallel
- stop_watching: Cancel active watchers
- list_watchers: Get status of all active watchers

Usage from voice:
    "Watch the Terminal for 'Build Successful'"
    "Watch Chrome for 'Application Submitted' and Terminal for 'Error'"
    "Stop watching Terminal"

This is JARVIS's "second pair of eyes" - monitoring background activity
while you focus on your main work.

Author: JARVIS AI System
Version: 10.6 - Video Multi-Space Intelligence
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..base.base_neural_mesh_agent import BaseNeuralMeshAgent
from ..data_models import (
    AgentMessage,
    KnowledgeType,
    MessageType,
    MessagePriority,
)

logger = logging.getLogger(__name__)


@dataclass
class VisualMonitorConfig:
    """
    Configuration for Visual Monitor Agent.

    Inherits base agent configuration from BaseAgentConfig via composition.
    """
    # Base agent configuration
    heartbeat_interval_seconds: float = 10.0
    message_queue_size: int = 1000
    message_handler_timeout_seconds: float = 10.0
    enable_knowledge_access: bool = True
    knowledge_cache_size: int = 100
    log_messages: bool = True
    log_level: str = "INFO"

    # Visual monitoring specific
    default_fps: int = 5  # Default FPS for watchers
    default_timeout: float = 300.0  # Default timeout (5 minutes)
    max_parallel_watchers: int = 3  # Max simultaneous watchers
    enable_voice_alerts: bool = True  # Speak when events detected
    enable_notifications: bool = True  # macOS notifications
    enable_cross_repo_sync: bool = True  # Share state across repos

    # Cross-repo paths
    cross_repo_dir: str = "~/.jarvis/cross_repo"
    vmsi_state_file: str = "vmsi_state.json"
    sync_interval_seconds: float = 5.0  # How often to sync state


class VisualMonitorAgent(BaseNeuralMeshAgent):
    """
    Visual Monitor Agent - The "Watcher" of VMSI.

    Provides background visual surveillance for specific windows.

    Capabilities:
    - watch_and_alert: Monitor a window and alert on event
    - watch_multiple: Monitor multiple windows in parallel
    - stop_watching: Stop specific watcher
    - list_watchers: List all active watchers

    Example:
        result = await coordinator.request(
            to_agent="visual_monitor_agent",
            payload={
                "action": "watch_and_alert",
                "app_name": "Terminal",
                "trigger_text": "Build Successful"
            }
        )
    """

    def __init__(self, config: Optional[VisualMonitorConfig] = None) -> None:
        """Initialize the Visual Monitor Agent."""
        super().__init__(
            agent_name="visual_monitor_agent",
            agent_type="visual_monitor",
            capabilities={
                "watch_and_alert",
                "watch_multiple",
                "stop_watching",
                "list_watchers",
                "get_watcher_stats",
                "background_surveillance",  # Meta capability
            },
            version="10.6",
        )

        self.config = config or VisualMonitorConfig()

        # Lazy-loaded components
        self._watcher_manager = None
        self._detector = None
        self._spatial_agent = None

        # Active monitoring tasks
        self._watch_tasks: Dict[str, asyncio.Task] = {}
        self._active_watchers: Dict[str, Any] = {}  # watcher_id -> watcher_info

        # Stats
        self._total_watches_started = 0
        self._total_events_detected = 0
        self._total_alerts_sent = 0

        # Cross-repo state
        self._state_sync_task: Optional[asyncio.Task] = None

    async def on_initialize(self) -> None:
        """Initialize agent resources."""
        logger.info("Initializing VisualMonitorAgent v10.6 (VMSI Watcher)")

        # Initialize video watcher manager
        try:
            from backend.vision.macos_video_capture_advanced import get_watcher_manager
            self._watcher_manager = get_watcher_manager()
            logger.info("VideoWatcherManager initialized")
        except Exception as e:
            logger.warning(f"VideoWatcherManager init failed: {e}")

        # Initialize visual event detector
        try:
            from backend.vision.visual_event_detector import create_detector
            self._detector = create_detector()
            logger.info("VisualEventDetector initialized")
        except Exception as e:
            logger.warning(f"VisualEventDetector init failed: {e}")

        # Ensure cross-repo directory exists
        if self.config.enable_cross_repo_sync:
            cross_repo_path = Path(self.config.cross_repo_dir).expanduser()
            cross_repo_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cross-repo directory: {cross_repo_path}")

        # Subscribe to visual monitoring messages (only if connected)
        if self.message_bus:
            try:
                await self.subscribe(
                    MessageType.CUSTOM,
                    self._handle_visual_message,
                )
            except RuntimeError:
                logger.debug("Message bus not available for subscription")

        # Announce availability (only if connected)
        if self.message_bus:
            try:
                await self.broadcast(
                    message_type=MessageType.ANNOUNCEMENT,
                    payload={
                        "agent": self.agent_name,
                        "event": "agent_ready",
                        "capabilities": list(self.capabilities),
                        "watcher_available": self._watcher_manager is not None,
                        "detector_available": self._detector is not None,
                    },
                )
            except RuntimeError:
                logger.debug("Message bus not available for broadcast")

        logger.info(
            f"VisualMonitorAgent initialized - "
            f"Watcher: {'ACTIVE' if self._watcher_manager else 'INACTIVE'}, "
            f"Detector: {'ACTIVE' if self._detector else 'INACTIVE'}"
        )

    async def on_start(self) -> None:
        """Called when agent starts."""
        logger.info("VisualMonitorAgent started - ready for visual monitoring")

        # Start cross-repo state sync
        if self.config.enable_cross_repo_sync:
            self._state_sync_task = asyncio.create_task(self._sync_state_loop())
            logger.info("Cross-repo state sync started")

    async def on_stop(self) -> None:
        """Cleanup when agent stops."""
        logger.info(
            f"VisualMonitorAgent stopping - "
            f"Watches: {self._total_watches_started}, "
            f"Events: {self._total_events_detected}, "
            f"Alerts: {self._total_alerts_sent}"
        )

        # Stop all active watchers
        if self._watcher_manager:
            await self._watcher_manager.stop_all_watchers()

        # Cancel watch tasks
        for task_id, task in self._watch_tasks.items():
            task.cancel()
            logger.debug(f"Cancelled watch task: {task_id}")

        # Stop state sync
        if self._state_sync_task:
            self._state_sync_task.cancel()

        logger.info("All watchers stopped")

    async def execute_task(self, payload: Dict[str, Any]) -> Any:
        """
        Execute a visual monitoring task.

        Supported actions:
        - watch_and_alert: Monitor a window and alert on event
        - watch_multiple: Monitor multiple windows in parallel
        - stop_watching: Stop a specific watcher
        - list_watchers: List all active watchers
        - get_watcher_stats: Get detailed watcher statistics
        """
        action = payload.get("action", "")

        logger.debug(f"VisualMonitorAgent executing: {action}")

        if action == "watch_and_alert":
            app_name = payload.get("app_name", "")
            trigger_text = payload.get("trigger_text", "")
            space_id = payload.get("space_id")
            return await self.watch_and_alert(app_name, trigger_text, space_id)

        elif action == "watch_multiple":
            watch_specs = payload.get("watch_specs", [])
            return await self.watch_multiple(watch_specs)

        elif action == "stop_watching":
            watcher_id = payload.get("watcher_id", "")
            app_name = payload.get("app_name", "")
            return await self.stop_watching(watcher_id=watcher_id, app_name=app_name)

        elif action == "list_watchers":
            return await self.list_watchers()

        elif action == "get_watcher_stats":
            return self.get_stats()

        else:
            raise ValueError(f"Unknown visual monitoring action: {action}")

    async def watch_and_alert(
        self,
        app_name: str,
        trigger_text: str,
        space_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Watch an app for specific text/event and alert when found.

        This is the main capability - enables voice commands like:
        "Watch the Terminal for 'Build Successful'"

        Args:
            app_name: App to monitor (e.g., "Terminal", "Chrome")
            trigger_text: Text to wait for (e.g., "Build Successful")
            space_id: Optional specific space (auto-detect if None)

        Returns:
            Result with watcher_id and monitoring status
        """
        if not app_name or not trigger_text:
            return {
                "success": False,
                "error": "app_name and trigger_text are required"
            }

        if not self._watcher_manager or not self._detector:
            return {
                "success": False,
                "error": "Watcher or detector not available"
            }

        try:
            logger.info(f"Starting watch: {app_name} for '{trigger_text}'")
            self._total_watches_started += 1

            # Step 1: Find window using SpatialAwarenessAgent
            window_info = await self._find_window(app_name, space_id)

            if not window_info['found']:
                return {
                    "success": False,
                    "error": f"Could not find {app_name}",
                    "window_info": window_info
                }

            window_id = window_info['window_id']
            detected_space_id = window_info.get('space_id', 0)

            logger.info(
                f"Found {app_name}: Window {window_id}, Space {detected_space_id}"
            )

            # Step 2: Spawn watcher
            watcher = await self._watcher_manager.spawn_watcher(
                window_id=window_id,
                fps=self.config.default_fps,
                app_name=app_name,
                space_id=detected_space_id,
                priority="low",
                timeout=self.config.default_timeout
            )

            # Step 3: Start monitoring task
            task = asyncio.create_task(
                self._monitor_and_alert(
                    watcher=watcher,
                    trigger_text=trigger_text,
                    app_name=app_name,
                    space_id=detected_space_id
                )
            )

            self._watch_tasks[watcher.watcher_id] = task
            self._active_watchers[watcher.watcher_id] = {
                'watcher': watcher,
                'app_name': app_name,
                'trigger_text': trigger_text,
                'space_id': detected_space_id,
                'started_at': datetime.now().isoformat(),
            }

            return {
                "success": True,
                "watcher_id": watcher.watcher_id,
                "window_id": window_id,
                "app_name": app_name,
                "space_id": detected_space_id,
                "trigger_text": trigger_text,
                "message": f"Watching {app_name} on Space {detected_space_id}"
            }

        except Exception as e:
            logger.exception(f"Error in watch_and_alert: {e}")
            return {
                "success": False,
                "error": str(e),
                "app_name": app_name,
                "trigger_text": trigger_text
            }

    async def watch_multiple(
        self,
        watch_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Monitor multiple windows in parallel.

        Args:
            watch_specs: List of watch specifications
                [
                    {"app": "Terminal", "trigger": "Build Successful"},
                    {"app": "Chrome", "trigger": "Application Submitted"}
                ]

        Returns:
            Results for all watchers
        """
        if not watch_specs:
            return {"success": False, "error": "No watch specs provided"}

        logger.info(f"Starting {len(watch_specs)} parallel watchers")

        # Start all watchers in parallel
        tasks = []
        for spec in watch_specs:
            app_name = spec.get('app', '')
            trigger_text = spec.get('trigger', '')
            space_id = spec.get('space_id')

            task = self.watch_and_alert(app_name, trigger_text, space_id)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failed = len(results) - successful

        return {
            "success": successful > 0,
            "total_watchers": len(watch_specs),
            "successful": successful,
            "failed": failed,
            "results": results
        }

    async def stop_watching(
        self,
        watcher_id: Optional[str] = None,
        app_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stop a specific watcher.

        Args:
            watcher_id: Specific watcher ID (or)
            app_name: App name (stops all watchers for this app)

        Returns:
            Stop result
        """
        if not watcher_id and not app_name:
            return {"success": False, "error": "watcher_id or app_name required"}

        stopped_count = 0

        if watcher_id:
            # Stop specific watcher
            if watcher_id in self._active_watchers:
                await self._stop_watcher_by_id(watcher_id)
                stopped_count = 1
            else:
                return {
                    "success": False,
                    "error": f"Watcher {watcher_id} not found"
                }

        elif app_name:
            # Stop all watchers for this app
            watcher_ids = [
                wid for wid, info in self._active_watchers.items()
                if info['app_name'].lower() == app_name.lower()
            ]

            for wid in watcher_ids:
                await self._stop_watcher_by_id(wid)
                stopped_count += 1

            if stopped_count == 0:
                return {
                    "success": False,
                    "error": f"No watchers found for {app_name}"
                }

        return {
            "success": True,
            "stopped_count": stopped_count,
            "message": f"Stopped {stopped_count} watcher(s)"
        }

    async def list_watchers(self) -> Dict[str, Any]:
        """List all active watchers."""
        watchers = []

        for watcher_id, info in self._active_watchers.items():
            watcher = info['watcher']
            watchers.append({
                'watcher_id': watcher_id,
                'app_name': info['app_name'],
                'trigger_text': info['trigger_text'],
                'space_id': info['space_id'],
                'started_at': info['started_at'],
                'stats': watcher.get_stats() if hasattr(watcher, 'get_stats') else {}
            })

        return {
            "active_watchers": len(watchers),
            "watchers": watchers,
            "max_parallel": self.config.max_parallel_watchers
        }

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _find_window(
        self,
        app_name: str,
        space_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Find window using SpatialAwarenessAgent.

        Returns window_id and space_id.
        """
        # Try to get SpatialAwarenessAgent from coordinator
        if not self._spatial_agent and self.coordinator:
            try:
                # Request spatial awareness
                result = await self.coordinator.request(
                    to_agent="spatial_awareness_agent",
                    payload={
                        "action": "find_window",
                        "app_name": app_name
                    },
                    timeout=5.0
                )

                if result and result.get('found'):
                    # Get first space where app is found
                    spaces = result.get('spaces', [])
                    if spaces:
                        target_space = spaces[0] if not space_id else space_id

                        # Get window ID from yabai (simplified)
                        # In production, this would integrate with yabai properly
                        window_id = self._estimate_window_id(app_name)

                        return {
                            'found': True,
                            'window_id': window_id,
                            'space_id': target_space,
                            'app_name': app_name
                        }

            except Exception as e:
                logger.warning(f"Error finding window via SpatialAwarenessAgent: {e}")

        # Fallback: estimate window ID
        window_id = self._estimate_window_id(app_name)

        return {
            'found': window_id > 0,
            'window_id': window_id,
            'space_id': space_id or 1,
            'app_name': app_name,
            'fallback': True
        }

    def _estimate_window_id(self, app_name: str) -> int:
        """
        Estimate window ID for an app (fallback method).

        In production, this integrates with yabai or spatial awareness.
        For now, returns a placeholder ID.
        """
        # TODO: Integrate with yabai to get actual window ID
        # For now, return a placeholder
        import hashlib
        hash_val = int(hashlib.md5(app_name.encode()).hexdigest()[:8], 16)
        return hash_val % 10000 + 1000  # ID in range 1000-11000

    async def _monitor_and_alert(
        self,
        watcher: Any,
        trigger_text: str,
        app_name: str,
        space_id: int
    ):
        """
        Monitor watcher and send alert when event detected.

        This runs as a background task.
        """
        try:
            logger.info(
                f"[{watcher.watcher_id}] Monitoring {app_name} for '{trigger_text}'"
            )

            # Wait for visual event
            result = await self._watcher_manager.wait_for_visual_event(
                watcher=watcher,
                trigger=trigger_text,
                detector=self._detector,
                timeout=self.config.default_timeout
            )

            if result.detected:
                self._total_events_detected += 1

                logger.info(
                    f"[{watcher.watcher_id}] ✅ Event detected! "
                    f"Trigger: '{trigger_text}', Confidence: {result.confidence:.2f}"
                )

                # Send alert
                await self._send_alert(
                    app_name=app_name,
                    trigger_text=trigger_text,
                    space_id=space_id,
                    confidence=result.confidence,
                    detection_time=result.detection_time
                )

                # Store in knowledge graph
                if self.knowledge_graph:
                    await self.add_knowledge(
                        knowledge_type=KnowledgeType.OBSERVATION,
                        data={
                            "type": "visual_event_detected",
                            "app_name": app_name,
                            "trigger_text": trigger_text,
                            "space_id": space_id,
                            "confidence": result.confidence,
                            "detection_time": result.detection_time,
                            "timestamp": datetime.now().isoformat(),
                        },
                        confidence=result.confidence,
                    )
            else:
                logger.info(
                    f"[{watcher.watcher_id}] ⏱️ Timeout waiting for '{trigger_text}'"
                )

            # Cleanup
            await self._stop_watcher_by_id(watcher.watcher_id)

        except Exception as e:
            logger.exception(f"Error in monitor_and_alert: {e}")
            await self._stop_watcher_by_id(watcher.watcher_id)

    async def _send_alert(
        self,
        app_name: str,
        trigger_text: str,
        space_id: int,
        confidence: float,
        detection_time: float
    ):
        """Send voice alert and notification when event detected."""
        self._total_alerts_sent += 1

        # Voice alert
        if self.config.enable_voice_alerts:
            await self._send_voice_alert(app_name, trigger_text, space_id)

        # macOS notification
        if self.config.enable_notifications:
            await self._send_notification(app_name, trigger_text, space_id)

        # Broadcast event to other agents (only if connected)
        if self.message_bus:
            try:
                await self.broadcast(
                    message_type=MessageType.ANNOUNCEMENT,
                    payload={
                        "event": "visual_event_detected",
                        "app_name": app_name,
                        "trigger_text": trigger_text,
                        "space_id": space_id,
                        "confidence": confidence,
                        "detection_time": detection_time,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            except RuntimeError:
                logger.debug("Message bus not available for broadcast")

    async def _send_voice_alert(
        self,
        app_name: str,
        trigger_text: str,
        space_id: int
    ):
        """Send voice alert via JARVIS Voice API."""
        try:
            # TODO: Integrate with JARVIS Voice API
            # For now, log the alert
            narration = f"{trigger_text} detected on {app_name}, Space {space_id}"
            logger.info(f"🔊 Voice Alert: {narration}")

            # In production, would call:
            # await jarvis_voice_api.speak(narration)

        except Exception as e:
            logger.error(f"Error sending voice alert: {e}")

    async def _send_notification(
        self,
        app_name: str,
        trigger_text: str,
        space_id: int
    ):
        """Send macOS notification."""
        try:
            # Use osascript for macOS notification
            import subprocess

            title = f"JARVIS - {app_name}"
            message = f"{trigger_text} (Space {space_id})"

            subprocess.run([
                'osascript', '-e',
                f'display notification "{message}" with title "{title}"'
            ], check=False, capture_output=True)

            logger.info(f"📬 Notification sent: {message}")

        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    async def _stop_watcher_by_id(self, watcher_id: str):
        """Stop watcher by ID and cleanup."""
        if watcher_id in self._active_watchers:
            info = self._active_watchers[watcher_id]
            watcher = info['watcher']

            # Stop watcher
            if self._watcher_manager:
                await self._watcher_manager.stop_watcher(watcher_id)

            # Cancel task
            if watcher_id in self._watch_tasks:
                task = self._watch_tasks[watcher_id]
                if not task.done():
                    task.cancel()
                del self._watch_tasks[watcher_id]

            # Remove from active watchers
            del self._active_watchers[watcher_id]

            logger.info(f"Stopped watcher {watcher_id}")

    async def _sync_state_loop(self):
        """Sync state to cross-repo file periodically."""
        while True:
            try:
                await asyncio.sleep(self.config.sync_interval_seconds)
                await self._sync_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in state sync: {e}")

    async def _sync_state(self):
        """Write current state to cross-repo file."""
        try:
            state_file = Path(self.config.cross_repo_dir).expanduser() / self.config.vmsi_state_file

            state = {
                "active_watchers": [
                    {
                        "watcher_id": watcher_id,
                        "app_name": info['app_name'],
                        "trigger_text": info['trigger_text'],
                        "space_id": info['space_id'],
                        "started_at": info['started_at'],
                        "status": "watching",
                        "repo": "JARVIS-AI-Agent"
                    }
                    for watcher_id, info in self._active_watchers.items()
                ],
                "stats": {
                    "total_watches_started": self._total_watches_started,
                    "total_events_detected": self._total_events_detected,
                    "total_alerts_sent": self._total_alerts_sent,
                    "active_watchers": len(self._active_watchers),
                },
                "last_updated": datetime.now().isoformat(),
            }

            # Write to file
            state_file.write_text(json.dumps(state, indent=2))

        except Exception as e:
            logger.error(f"Error syncing state: {e}")

    async def _handle_visual_message(self, message: AgentMessage) -> None:
        """Handle incoming visual monitoring messages."""
        payload = message.payload

        if payload.get("type") != "visual_request":
            return

        action = payload.get("action")
        if not action:
            return

        logger.info(f"Received visual request from {message.from_agent}: {action}")

        try:
            result = await self.execute_task(payload)

            if self.message_bus:
                await self.message_bus.respond(
                    message,
                    payload={
                        "type": "visual_response",
                        "action": action,
                        "result": result,
                    },
                    from_agent=self.agent_name,
                )

        except Exception as e:
            logger.exception(f"Error handling visual message: {e}")
            if self.message_bus:
                await self.message_bus.respond(
                    message,
                    payload={
                        "type": "visual_response",
                        "action": action,
                        "error": str(e),
                    },
                    from_agent=self.agent_name,
                )

    # =========================================================================
    # Convenience methods
    # =========================================================================

    async def watch(
        self,
        app_name: str,
        trigger_text: str,
        space_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Quick method to start watching."""
        return await self.watch_and_alert(app_name, trigger_text, space_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_watches_started": self._total_watches_started,
            "total_events_detected": self._total_events_detected,
            "total_alerts_sent": self._total_alerts_sent,
            "active_watchers": len(self._active_watchers),
            "max_parallel": self.config.max_parallel_watchers,
            "capabilities": list(self.capabilities),
            "version": "10.6",
            "watcher_manager_available": self._watcher_manager is not None,
            "detector_available": self._detector is not None,
        }


# =============================================================================
# Factory function for agent registration
# =============================================================================

async def create_visual_monitor_agent(
    config: Optional[VisualMonitorConfig] = None,
) -> VisualMonitorAgent:
    """
    Create a Visual Monitor Agent.

    This is the factory function used by AgentInitializer.

    Args:
        config: Optional configuration

    Returns:
        Configured VisualMonitorAgent
    """
    return VisualMonitorAgent(config=config)
