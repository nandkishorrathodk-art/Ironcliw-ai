# backend/core/startup_summary.py
"""
StartupSummary - Human-readable startup completion reporting.

This module provides:
- StartupCompletionCriteria: Determines when startup is complete
- ComponentSummary: Summary of individual component startup
- StartupSummary: Formats and outputs startup summary

Usage:
    from backend.core.startup_summary import (
        StartupCompletionCriteria, StartupSummary
    )
    from backend.core.component_registry import get_component_registry

    # Create completion criteria
    criteria = StartupCompletionCriteria(
        start_time=time.time(),
        global_timeout=180.0
    )

    # Check if startup is complete
    is_complete, reason = criteria.is_complete()

    # Create summary
    registry = get_component_registry()
    summary = StartupSummary(registry, version="148.0")
    summary.start_time = datetime.now(timezone.utc)
    summary.end_time = datetime.now(timezone.utc)

    # Print formatted summary
    print(summary.format_summary())

    # Save to file
    summary.save_to_file()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.component_registry import ComponentRegistry

from backend.core.component_registry import (
    ComponentStatus,
    Criticality,
)

logger = logging.getLogger("jarvis.startup_summary")


class StartupCompletionCriteria:
    """
    Determines when startup sequence is complete.

    Tracks startup progress and determines completion based on:
    - Required component failure (immediate completion with failure)
    - All components resolved (success)
    - Global timeout (timeout completion)

    Attributes:
        start_time: Unix timestamp when startup began
        global_timeout: Maximum time allowed for startup in seconds
        all_components_resolved: True when all components have final status
        required_failure: True if any required component failed
    """

    def __init__(self, start_time: float, global_timeout: float = 180.0):
        """
        Initialize completion criteria.

        Args:
            start_time: Unix timestamp when startup began
            global_timeout: Maximum time allowed for startup in seconds
        """
        self.start_time = start_time
        self.global_timeout = global_timeout
        self.all_components_resolved: bool = False
        self.required_failure: bool = False

    def is_complete(self) -> tuple[bool, str]:
        """
        Check if startup is complete.

        Returns:
            Tuple of (is_complete, reason) where reason is one of:
            - "required_component_failed": A required component failed
            - "all_resolved": All components have final status
            - "global_timeout": Startup exceeded global timeout
            - "in_progress": Startup is still in progress
        """
        # Required failure takes priority - immediate failure
        if self.required_failure:
            return (True, "required_component_failed")

        # All components resolved - success
        if self.all_components_resolved:
            return (True, "all_resolved")

        # Global timeout - forced completion
        if time.time() > self.start_time + self.global_timeout:
            return (True, "global_timeout")

        # Still in progress
        return (False, "in_progress")


@dataclass
class ComponentSummary:
    """
    Summary of individual component startup.

    Attributes:
        name: Component name
        status: Current component status
        criticality: Component criticality level
        startup_time: Time taken to start (or time so far if starting)
        message: Optional status/error message
    """
    name: str
    status: ComponentStatus
    criticality: Criticality
    startup_time: float  # seconds
    message: Optional[str] = None


class StartupSummary:
    """
    Formats and outputs startup summary.

    Provides human-readable terminal output and JSON serialization
    for startup state tracking.

    Usage:
        registry = get_component_registry()
        summary = StartupSummary(registry, version="148.0")
        summary.start_time = datetime.now(timezone.utc)
        summary.end_time = datetime.now(timezone.utc)

        print(summary.format_summary())
        summary.save_to_file()
    """

    STATUS_ICONS: Dict[ComponentStatus, str] = {
        ComponentStatus.HEALTHY: "✓",
        ComponentStatus.DEGRADED: "⟳",
        ComponentStatus.FAILED: "✗",
        ComponentStatus.DISABLED: "○",
        ComponentStatus.STARTING: "⟳",
        ComponentStatus.PENDING: "·",
    }

    def __init__(
        self,
        registry: 'ComponentRegistry',
        version: str = "148.0"
    ):
        """
        Initialize startup summary.

        Args:
            registry: ComponentRegistry with component states
            version: Ironcliw version string for display
        """
        self.registry = registry
        self.version = version
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def _get_component_summaries(self) -> List[ComponentSummary]:
        """
        Get summaries for all registered components.

        Returns:
            List of ComponentSummary objects
        """
        summaries = []
        for state in self.registry.all_states():
            # Calculate startup time
            startup_time = 0.0
            if state.started_at is not None:
                if state.healthy_at is not None:
                    startup_time = (
                        state.healthy_at - state.started_at
                    ).total_seconds()
                elif state.failed_at is not None:
                    startup_time = (
                        state.failed_at - state.started_at
                    ).total_seconds()
                else:
                    # Still starting - time elapsed so far
                    startup_time = (
                        datetime.now(timezone.utc) - state.started_at
                    ).total_seconds()

            summaries.append(ComponentSummary(
                name=state.definition.name,
                status=state.status,
                criticality=state.definition.criticality,
                startup_time=startup_time,
                message=state.failure_reason,
            ))

        return summaries

    def _get_capabilities_info(self) -> Dict[str, Dict]:
        """
        Get capability availability information.

        Returns:
            Dict mapping capability name to info dict with:
            - available: bool
            - provider: component name
            - status: provider status
        """
        capabilities = {}
        for state in self.registry.all_states():
            for cap in state.definition.provides_capabilities:
                available = state.status in (
                    ComponentStatus.HEALTHY,
                    ComponentStatus.DEGRADED
                )
                capabilities[cap] = {
                    "available": available,
                    "provider": state.definition.name,
                    "status": state.status.value,
                }
        return capabilities

    def format_summary(self) -> str:
        """
        Format human-readable terminal output with status icons.

        Returns:
            Formatted string ready for terminal output.
        """
        lines = []
        separator = "━" * 52

        # Header
        lines.append(separator)
        lines.append(f"Ironcliw Startup Summary (v{self.version})")
        lines.append(separator)
        lines.append("")

        # Component lines
        summaries = self._get_component_summaries()
        for summary in summaries:
            icon = self.STATUS_ICONS.get(summary.status, "?")
            status_str = summary.status.value.upper()
            crit_str = f"[{summary.criticality.value}]"

            # Format startup time
            if summary.status == ComponentStatus.STARTING:
                time_str = f"{summary.startup_time:.0f}s..."
            elif summary.status == ComponentStatus.DISABLED:
                time_str = "--"
            else:
                time_str = f"{summary.startup_time:.1f}s"

            # Build line
            line = f"{icon} {summary.name:<20} {status_str:<10} {crit_str:<14} {time_str}"

            # Add message if present
            if summary.message:
                line += f'   "{summary.message}"'

            lines.append(line)

        # Capabilities section
        lines.append("")
        lines.append("Capabilities Available:")
        capabilities = self._get_capabilities_info()
        if capabilities:
            for cap_name, cap_info in capabilities.items():
                if cap_info["available"]:
                    icon = "✓"
                    status_note = ""
                    if cap_info["status"] == "starting":
                        status_note = ", starting"
                    lines.append(
                        f"  {icon} {cap_name} ({cap_info['provider']}{status_note})"
                    )
                else:
                    icon = "✗"
                    lines.append(
                        f"  {icon} {cap_name} ({cap_info['provider']} {cap_info['status']})"
                    )
        else:
            lines.append("  (none)")

        # Total time and status
        lines.append("")
        if self.start_time and self.end_time:
            total_time = (self.end_time - self.start_time).total_seconds()
            lines.append(f"Total startup time: {total_time:.1f}s")
        else:
            lines.append("Total startup time: N/A")

        overall_status = self.compute_overall_status()

        # Count starting and failed components
        starting_count = sum(
            1 for s in summaries if s.status == ComponentStatus.STARTING
        )
        failed_count = sum(
            1 for s in summaries if s.status == ComponentStatus.FAILED
        )

        status_notes = []
        if starting_count > 0:
            status_notes.append(f"{starting_count} starting")
        if failed_count > 0:
            status_notes.append(f"{failed_count} failed")

        if status_notes:
            lines.append(
                f"System status: {overall_status} ({', '.join(status_notes)})"
            )
        else:
            lines.append(f"System status: {overall_status}")

        lines.append(separator)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert to JSON-serializable dict for state file.

        Returns:
            Dict with startup summary data
        """
        components = {}
        for state in self.registry.all_states():
            # Calculate startup time
            startup_time = 0.0
            if state.started_at is not None:
                if state.healthy_at is not None:
                    startup_time = (
                        state.healthy_at - state.started_at
                    ).total_seconds()
                elif state.failed_at is not None:
                    startup_time = (
                        state.failed_at - state.started_at
                    ).total_seconds()
                else:
                    startup_time = (
                        datetime.now(timezone.utc) - state.started_at
                    ).total_seconds()

            components[state.definition.name] = {
                "status": state.status.value,
                "criticality": state.definition.criticality.value,
                "startup_time": startup_time,
                "message": state.failure_reason,
            }

        capabilities = {}
        for cap_name, cap_info in self._get_capabilities_info().items():
            capabilities[cap_name] = cap_info

        return {
            "version": self.version,
            "start_time": (
                self.start_time.isoformat() if self.start_time else None
            ),
            "end_time": (
                self.end_time.isoformat() if self.end_time else None
            ),
            "components": components,
            "capabilities": capabilities,
            "overall_status": self.compute_overall_status(),
        }

    def save_to_file(self, path: Optional[Path] = None):
        """
        Save to JSON file.

        Args:
            path: File path. Defaults to ~/.jarvis/state/startup_summary.json
        """
        if path is None:
            path = Path.home() / ".jarvis" / "state" / "startup_summary.json"

        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.debug(f"Saved startup summary to {path}")

    def compute_overall_status(self) -> str:
        """
        Compute overall system status based on component states.

        Returns:
            One of "HEALTHY", "DEGRADED", or "FAILED"
        """
        states = self.registry.all_states()

        if not states:
            return "HEALTHY"

        has_required_failure = False
        has_starting = False
        has_non_required_failure = False

        for state in states:
            # Skip disabled components
            if state.status == ComponentStatus.DISABLED:
                continue

            if state.status == ComponentStatus.FAILED:
                if state.definition.criticality == Criticality.REQUIRED:
                    has_required_failure = True
                else:
                    has_non_required_failure = True

            if state.status == ComponentStatus.STARTING:
                has_starting = True

        # Required failure = system failed
        if has_required_failure:
            return "FAILED"

        # Any optional failures or components still starting = degraded
        if has_non_required_failure or has_starting:
            return "DEGRADED"

        return "HEALTHY"
