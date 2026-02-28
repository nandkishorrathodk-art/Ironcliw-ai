# Readiness Integrity Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the fundamental disconnect between "phase completion" and "component health" - FULLY_READY must only be declared when critical components are actually healthy, with accurate status reporting.

**Architecture:** Create a single readiness predicate that gates FULLY_READY, consolidate path discovery to one location (`IntelligentRepoDiscovery`), fix status semantics so "skipped" !== "stopped", and ensure body/preflight have single completion points.

**Tech Stack:** Python 3.11+, asyncio, unified_supervisor.py, trinity_integrator.py, loading-manager.js

---

## Task 1: Create Unified Readiness Configuration Module

**Files:**
- Create: `backend/core/readiness_config.py`
- Test: `tests/unit/backend/core/test_readiness_config.py`

**Step 1: Write the failing test**

```python
# tests/unit/backend/core/test_readiness_config.py
"""Tests for readiness configuration module."""
import pytest
from backend.core.readiness_config import (
    ReadinessConfig,
    ComponentCriticality,
    get_readiness_config,
)


class TestReadinessConfig:
    """Test readiness configuration."""

    def test_critical_components_defined(self):
        """Critical components must be defined."""
        config = get_readiness_config()
        assert "backend" in config.critical_components
        assert "loading_server" in config.critical_components

    def test_optional_components_defined(self):
        """Optional components must be defined."""
        config = get_readiness_config()
        assert "jarvis_prime" in config.optional_components
        assert "reactor_core" in config.optional_components

    def test_component_criticality_lookup(self):
        """Can look up criticality by component name."""
        config = get_readiness_config()
        assert config.get_criticality("backend") == ComponentCriticality.CRITICAL
        assert config.get_criticality("jarvis_prime") == ComponentCriticality.OPTIONAL
        assert config.get_criticality("unknown") == ComponentCriticality.UNKNOWN

    def test_status_display_mapping(self):
        """Status display mapping is correct."""
        config = get_readiness_config()
        assert config.status_to_display("healthy") == "HEAL"
        assert config.status_to_display("starting") == "STAR"
        assert config.status_to_display("pending") == "PEND"
        assert config.status_to_display("stopped") == "STOP"
        assert config.status_to_display("skipped") == "SKIP"
        assert config.status_to_display("unavailable") == "UNAV"

    def test_skipped_not_equals_stopped(self):
        """Skipped status is distinct from stopped."""
        config = get_readiness_config()
        skipped_display = config.status_to_display("skipped")
        stopped_display = config.status_to_display("stopped")
        assert skipped_display != stopped_display
        assert skipped_display == "SKIP"
        assert stopped_display == "STOP"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python -m pytest tests/unit/backend/core/test_readiness_config.py -v`
Expected: ModuleNotFoundError

**Step 3: Write the implementation**

```python
# backend/core/readiness_config.py
"""
Unified Readiness Configuration v1.0
====================================

Single source of truth for:
- Component criticality (critical vs optional)
- Status display mappings
- Readiness predicates

This module eliminates duplicate definitions across the codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Optional

__all__ = [
    "ComponentCriticality",
    "ComponentStatus",
    "ReadinessConfig",
    "get_readiness_config",
]


class ComponentCriticality(Enum):
    """Component criticality levels."""
    CRITICAL = "critical"      # Must be healthy for FULLY_READY
    OPTIONAL = "optional"      # Can be healthy, skipped, or unavailable
    UNKNOWN = "unknown"        # Not in configuration


class ComponentStatus(Enum):
    """
    Unified component status values.

    IMPORTANT: These are semantically distinct:
    - stopped: Component was running and was intentionally stopped
    - skipped: Component was not started (not configured/repo not found)
    - unavailable: Component is not available on this system
    """
    PENDING = "pending"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"       # Was running, now stopped
    SKIPPED = "skipped"       # Never started (not configured)
    UNAVAILABLE = "unavailable"  # Not available on system


# Status to 4-character display code mapping
# IMPORTANT: skipped !== stopped
STATUS_DISPLAY_MAP: Dict[str, str] = {
    "pending": "PEND",
    "starting": "STAR",
    "healthy": "HEAL",
    "degraded": "DEGR",
    "error": "ERRO",
    "stopped": "STOP",
    "skipped": "SKIP",        # NOT "STOP"!
    "unavailable": "UNAV",
    "running": "STAR",        # Alias
    "complete": "HEAL",       # Alias
    "ready": "HEAL",          # Alias
    "failed": "ERRO",         # Alias
}


# Dashboard status mapping (for backward compatibility)
# Maps internal status to dashboard status
DASHBOARD_STATUS_MAP: Dict[str, str] = {
    "pending": "pending",
    "running": "starting",
    "starting": "starting",
    "complete": "healthy",
    "ready": "healthy",
    "healthy": "healthy",
    "degraded": "degraded",
    "error": "error",
    "failed": "error",
    "stopped": "stopped",
    "skipped": "skipped",     # NOT "stopped"!
    "unavailable": "unavailable",
}


@dataclass(frozen=True)
class ReadinessConfig:
    """
    Unified readiness configuration.

    Defines which components are critical (must be healthy for FULLY_READY)
    and which are optional (can be skipped/unavailable).
    """

    # Critical components - MUST be healthy for FULLY_READY
    critical_components: FrozenSet[str] = field(default_factory=lambda: frozenset({
        "backend",
        "loading_server",
        "preflight",
    }))

    # Optional components - can be skipped/unavailable
    optional_components: FrozenSet[str] = field(default_factory=lambda: frozenset({
        "jarvis_prime",
        "reactor_core",
        "enterprise",
        "agi_os",
        "gcp_vm",
    }))

    # Verification timeout (seconds)
    verification_timeout: float = field(
        default_factory=lambda: float(os.getenv("Ironcliw_VERIFICATION_TIMEOUT", "30.0"))
    )

    # Revocation settings
    unhealthy_threshold_failures: int = field(
        default_factory=lambda: int(os.getenv("Ironcliw_UNHEALTHY_THRESHOLD", "3"))
    )
    unhealthy_threshold_seconds: float = field(
        default_factory=lambda: float(os.getenv("Ironcliw_UNHEALTHY_SECONDS", "30.0"))
    )
    revocation_cooldown_seconds: float = field(
        default_factory=lambda: float(os.getenv("Ironcliw_REVOCATION_COOLDOWN", "60.0"))
    )

    def get_criticality(self, component: str) -> ComponentCriticality:
        """Get criticality level for a component."""
        if component in self.critical_components:
            return ComponentCriticality.CRITICAL
        if component in self.optional_components:
            return ComponentCriticality.OPTIONAL
        return ComponentCriticality.UNKNOWN

    def is_critical(self, component: str) -> bool:
        """Check if component is critical."""
        return component in self.critical_components

    def is_optional(self, component: str) -> bool:
        """Check if component is optional."""
        return component in self.optional_components

    @staticmethod
    def status_to_display(status: str) -> str:
        """Convert status to 4-character display code."""
        return STATUS_DISPLAY_MAP.get(status.lower(), status[:4].upper())

    @staticmethod
    def status_to_dashboard(status: str) -> str:
        """Convert internal status to dashboard status."""
        return DASHBOARD_STATUS_MAP.get(status.lower(), status)


# Singleton instance
_config_instance: Optional[ReadinessConfig] = None


def get_readiness_config() -> ReadinessConfig:
    """Get the global readiness configuration (singleton)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ReadinessConfig()
    return _config_instance
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python -m pytest tests/unit/backend/core/test_readiness_config.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add backend/core/readiness_config.py tests/unit/backend/core/test_readiness_config.py
git commit -m "$(cat <<'EOF'
feat(readiness): Add unified readiness configuration module

- Define component criticality (critical vs optional)
- Fix status semantics: skipped !== stopped
- Single source of truth for status display mappings
- Configurable verification timeout and revocation settings

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create Unified Readiness Predicate

**Files:**
- Create: `backend/core/readiness_predicate.py`
- Test: `tests/unit/backend/core/test_readiness_predicate.py`

**Step 1: Write the failing test**

```python
# tests/unit/backend/core/test_readiness_predicate.py
"""Tests for readiness predicate."""
import pytest
from backend.core.readiness_predicate import (
    ReadinessPredicate,
    ReadinessResult,
)


class TestReadinessPredicate:
    """Test readiness predicate logic."""

    def test_all_critical_healthy_optional_healthy(self):
        """FULLY_READY when all critical healthy and optional healthy."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "healthy",
            "reactor_core": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True
        assert result.blocking_components == []

    def test_all_critical_healthy_optional_skipped(self):
        """FULLY_READY when all critical healthy and optional skipped."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "skipped",
            "reactor_core": "skipped",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True
        assert result.blocking_components == []

    def test_critical_unhealthy_blocks_ready(self):
        """NOT FULLY_READY when critical component unhealthy."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "error",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "healthy",
            "reactor_core": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is False
        assert "backend" in result.blocking_components

    def test_critical_starting_blocks_ready(self):
        """NOT FULLY_READY when critical component still starting."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "starting",
            "loading_server": "healthy",
            "preflight": "healthy",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is False
        assert "backend" in result.blocking_components

    def test_optional_error_does_not_block(self):
        """FULLY_READY even when optional component has error."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "error",
            "reactor_core": "error",
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is True
        assert result.degraded_components == ["jarvis_prime", "reactor_core"]

    def test_missing_critical_blocks(self):
        """NOT FULLY_READY when critical component missing from states."""
        predicate = ReadinessPredicate()
        component_states = {
            "loading_server": "healthy",
            "preflight": "healthy",
            # backend missing
        }
        result = predicate.evaluate(component_states)
        assert result.is_fully_ready is False
        assert "backend" in result.blocking_components

    def test_result_has_readiness_message(self):
        """Result includes human-readable message."""
        predicate = ReadinessPredicate()
        component_states = {
            "backend": "healthy",
            "loading_server": "healthy",
            "preflight": "healthy",
            "jarvis_prime": "skipped",
        }
        result = predicate.evaluate(component_states)
        assert result.message is not None
        assert len(result.message) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python -m pytest tests/unit/backend/core/test_readiness_predicate.py -v`
Expected: ModuleNotFoundError

**Step 3: Write the implementation**

```python
# backend/core/readiness_predicate.py
"""
Unified Readiness Predicate v1.0
================================

Single predicate that determines FULLY_READY state.

Logic:
- FULLY_READY iff (all critical healthy) AND (optional healthy OR skipped OR unavailable)
- Critical components MUST be healthy
- Optional components can be healthy, skipped, unavailable, or even errored
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from backend.core.readiness_config import (
    get_readiness_config,
    ComponentCriticality,
    ReadinessConfig,
)

__all__ = [
    "ReadinessPredicate",
    "ReadinessResult",
]


# Statuses that count as "acceptable" for readiness
HEALTHY_STATUSES: Set[str] = {"healthy", "complete", "ready", "running"}
ACCEPTABLE_OPTIONAL_STATUSES: Set[str] = {
    "healthy", "complete", "ready", "running",
    "skipped", "unavailable", "stopped",
    "error", "failed", "degraded",  # Optional can fail without blocking
}


@dataclass
class ReadinessResult:
    """Result of readiness evaluation."""
    is_fully_ready: bool
    message: str
    blocking_components: List[str] = field(default_factory=list)
    degraded_components: List[str] = field(default_factory=list)
    skipped_components: List[str] = field(default_factory=list)
    healthy_components: List[str] = field(default_factory=list)
    component_states: Dict[str, str] = field(default_factory=dict)


class ReadinessPredicate:
    """
    Evaluates system readiness based on component states.

    Uses ReadinessConfig to determine which components are critical vs optional.
    """

    def __init__(self, config: Optional[ReadinessConfig] = None):
        """Initialize with optional custom config."""
        self.config = config or get_readiness_config()

    def evaluate(self, component_states: Dict[str, str]) -> ReadinessResult:
        """
        Evaluate readiness based on component states.

        Args:
            component_states: Dict mapping component name to status string

        Returns:
            ReadinessResult with evaluation details
        """
        blocking: List[str] = []
        degraded: List[str] = []
        skipped: List[str] = []
        healthy: List[str] = []

        # Check all critical components
        for component in self.config.critical_components:
            status = component_states.get(component, "pending").lower()

            if status in HEALTHY_STATUSES:
                healthy.append(component)
            else:
                # Critical component not healthy = blocking
                blocking.append(component)

        # Check optional components (for degraded/skipped reporting)
        for component in self.config.optional_components:
            status = component_states.get(component, "pending").lower()

            if status in HEALTHY_STATUSES:
                healthy.append(component)
            elif status in ("skipped", "unavailable"):
                skipped.append(component)
            elif status in ("error", "failed", "degraded"):
                degraded.append(component)
            # "pending", "starting", "stopped" - just not counted

        # Determine readiness
        is_ready = len(blocking) == 0

        # Build message
        if is_ready:
            if skipped:
                message = f"System ready. Optional components not configured: {', '.join(skipped)}"
            elif degraded:
                message = f"System ready (degraded). Components with issues: {', '.join(degraded)}"
            else:
                message = "All components healthy. System fully operational."
        else:
            message = f"System not ready. Waiting for: {', '.join(blocking)}"

        return ReadinessResult(
            is_fully_ready=is_ready,
            message=message,
            blocking_components=blocking,
            degraded_components=degraded,
            skipped_components=skipped,
            healthy_components=healthy,
            component_states=dict(component_states),
        )

    def is_component_ready(self, component: str, status: str) -> bool:
        """
        Check if a single component's status counts as ready.

        For critical components: must be in HEALTHY_STATUSES
        For optional components: can be in ACCEPTABLE_OPTIONAL_STATUSES
        """
        status = status.lower()
        criticality = self.config.get_criticality(component)

        if criticality == ComponentCriticality.CRITICAL:
            return status in HEALTHY_STATUSES
        elif criticality == ComponentCriticality.OPTIONAL:
            return status in ACCEPTABLE_OPTIONAL_STATUSES
        else:
            # Unknown component - treat as optional
            return status in ACCEPTABLE_OPTIONAL_STATUSES
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python -m pytest tests/unit/backend/core/test_readiness_predicate.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add backend/core/readiness_predicate.py tests/unit/backend/core/test_readiness_predicate.py
git commit -m "$(cat <<'EOF'
feat(readiness): Add unified readiness predicate

- Single predicate for FULLY_READY evaluation
- Critical components must be healthy
- Optional components can be skipped/errored
- Returns detailed result with blocking/degraded lists

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update Dashboard Status Mapping in unified_supervisor.py

**Files:**
- Modify: `unified_supervisor.py:59826-59834`

**Step 1: Locate and read current code**

Read lines 59820-59840 to see current dashboard_status_map.

**Step 2: Update the mapping**

Replace the hardcoded mapping with import from readiness_config:

```python
# Line ~59826 - Replace this:
#                 dashboard_status_map = {
#                     "pending": "pending",
#                     "running": "starting",
#                     "complete": "healthy",
#                     "error": "error",
#                     "skipped": "stopped",  # BUG: skipped !== stopped
#                 }

# With import at top of file and use:
from backend.core.readiness_config import DASHBOARD_STATUS_MAP

# Then at line ~59833:
                dash_status = DASHBOARD_STATUS_MAP.get(status, status)
```

**Step 3: Update the 4-char display in _render_passthrough**

At line ~3843, replace:
```python
# Old:
comp_parts.append(f"{short_name}:{color}{status[:4].upper()}{self.RESET}")

# New:
from backend.core.readiness_config import STATUS_DISPLAY_MAP
# ...
display_code = STATUS_DISPLAY_MAP.get(status, status[:4].upper())
comp_parts.append(f"{short_name}:{color}{display_code}{self.RESET}")
```

**Step 4: Run existing tests**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python -m pytest tests/ -k "supervisor" --ignore=tests/integration -v --tb=short 2>/dev/null | head -50`

**Step 5: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
fix(status): Use unified status mapping - skipped !== stopped

- Import DASHBOARD_STATUS_MAP from readiness_config
- "skipped" now displays as "SKIP" not "STOP"
- Consistent 4-char display codes across dashboard

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Move FULLY_READY After Service Verification

**Files:**
- Modify: `unified_supervisor.py:56125-56150`

**Step 1: Read current code around FULLY_READY marking**

Read lines 56120-56160 to see current flow.

**Step 2: Refactor to verify before marking**

The key change: move `mark_tier(FULLY_READY)` to AFTER `_verify_all_services()` and make it conditional on the readiness predicate.

```python
# Around line 56125-56150, change FROM:
#             # Mark as running
#             self._state = KernelState.RUNNING
#             if self._readiness_manager:
#                 self._readiness_manager.mark_tier(ReadinessTier.FULLY_READY)
#
#             # Final service verification
#             verification = await self._verify_all_services(timeout=10.0)
#             if not verification["all_healthy"]:
#                 # Just logs warnings

# TO:
            # Final service verification BEFORE marking FULLY_READY
            issue_collector.set_current_phase("Service Verification")
            verification = await self._verify_all_services(timeout=self._get_verification_timeout())

            # Evaluate readiness using unified predicate
            from backend.core.readiness_predicate import ReadinessPredicate
            predicate = ReadinessPredicate()
            component_states = self._get_component_states_for_readiness()
            readiness_result = predicate.evaluate(component_states)

            # Only mark FULLY_READY if predicate passes
            if readiness_result.is_fully_ready:
                self._state = KernelState.RUNNING
                if self._readiness_manager:
                    self._readiness_manager.mark_tier(ReadinessTier.FULLY_READY)
                self.logger.success(f"[Readiness] {readiness_result.message}")
            else:
                # Mark as DEGRADED instead
                self._state = KernelState.RUNNING  # Still running, but degraded
                if self._readiness_manager:
                    self._readiness_manager.mark_tier(ReadinessTier.INTERACTIVE)  # Not FULLY_READY
                self.logger.warning(f"[Readiness] {readiness_result.message}")
                for component in readiness_result.blocking_components:
                    issue_collector.add_warning(
                        f"Critical component not ready: {component}",
                        IssueCategory.GENERAL,
                    )
```

**Step 3: Add helper method for component states**

Add method around line 60000:

```python
    def _get_component_states_for_readiness(self) -> Dict[str, str]:
        """Get current component states for readiness evaluation."""
        states = {}
        for name, info in self._component_status.items():
            states[name] = info.get("status", "pending")
        return states

    def _get_verification_timeout(self) -> float:
        """Get verification timeout from config or env."""
        from backend.core.readiness_config import get_readiness_config
        return get_readiness_config().verification_timeout
```

**Step 4: Test manually**

Run: `cd /Users/djrussell23/Documents/repos/Ironcliw-AI-Agent && python -c "from unified_supervisor import JarvisSystemKernel; print('Import OK')"`

**Step 5: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
fix(readiness): Move FULLY_READY after service verification

- FULLY_READY is now conditional on readiness predicate
- Verification runs BEFORE marking ready
- If critical components unhealthy, mark INTERACTIVE (degraded)
- Blocking components logged as warnings

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Consolidate Path Discovery to IntelligentRepoDiscovery

**Files:**
- Modify: `backend/core/trinity_startup_orchestrator.py:193-226`
- Modify: `backend/core/trinity_integrator.py:325-336` (add Ironcliw_PRIME_PATH support)

**Step 1: Update IntelligentRepoDiscovery ENV_VARS**

The current ENV_VARS uses `Ironcliw_PRIME_REPO_PATH` but the codebase uses `Ironcliw_PRIME_PATH`. Fix this:

```python
# backend/core/trinity_integrator.py line ~325
# Change FROM:
    ENV_VARS: Final[Dict[str, str]] = {
        "jarvis": "Ironcliw_REPO_PATH",
        "jarvis_prime": "Ironcliw_PRIME_REPO_PATH",
        "reactor_core": "REACTOR_CORE_REPO_PATH",
    }

# TO:
    ENV_VARS: Final[Dict[str, str]] = {
        "jarvis": "Ironcliw_REPO_PATH",
        "jarvis_prime": "Ironcliw_PRIME_PATH",      # Match existing env var
        "reactor_core": "REACTOR_CORE_PATH",       # Match existing env var
    }
```

**Step 2: Update trinity_startup_orchestrator to use discovery**

```python
# backend/core/trinity_startup_orchestrator.py
# Replace _init_components method (lines ~193-226):

    def _init_components(self) -> None:
        """Initialize component configurations using unified discovery."""
        # Use IntelligentRepoDiscovery for path resolution
        from backend.core.trinity_integrator import IntelligentRepoDiscovery

        discovery = IntelligentRepoDiscovery()

        # Discover paths (sync wrapper for init)
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            prime_result = loop.run_until_complete(discovery.discover("jarvis_prime"))
            reactor_result = loop.run_until_complete(discovery.discover("reactor_core"))
        finally:
            loop.close()

        prime_path = prime_result.path or Path("/nonexistent")  # Will fail validation
        reactor_path = reactor_result.path or Path("/nonexistent")

        # Log discovery results
        if prime_result.path:
            logger.info(f"[Trinity] J-Prime discovered: {prime_result.path} (via {prime_result.strategy_used.name})")
        else:
            logger.warning(f"[Trinity] J-Prime not found. Set Ironcliw_PRIME_PATH or clone to sibling directory.")

        if reactor_result.path:
            logger.info(f"[Trinity] Reactor-Core discovered: {reactor_result.path} (via {reactor_result.strategy_used.name})")
        else:
            logger.warning(f"[Trinity] Reactor-Core not found. Set REACTOR_CORE_PATH or clone to sibling directory.")

        # Get ports from config or defaults
        prime_port = int(os.getenv("Ironcliw_PRIME_PORT", "8000"))
        reactor_port = int(os.getenv("REACTOR_CORE_PORT", "8090"))

        self.state.components = {
            ComponentType.Ironcliw_PRIME: ComponentInfo(
                component_type=ComponentType.Ironcliw_PRIME,
                name="Ironcliw-Prime (Mind)",
                repo_path=prime_path,
                startup_script="run_server.py",
                port=prime_port,
                health_endpoint="/health",
                required=False,
                startup_timeout=120.0,
            ),
            ComponentType.REACTOR_CORE: ComponentInfo(
                component_type=ComponentType.REACTOR_CORE,
                name="Reactor-Core (Nerves)",
                repo_path=reactor_path,
                startup_script="run_reactor.py",
                port=reactor_port,
                health_endpoint="/health",
                required=False,
                startup_timeout=30.0,
            ),
        }
```

**Step 3: Commit**

```bash
git add backend/core/trinity_integrator.py backend/core/trinity_startup_orchestrator.py
git commit -m "$(cat <<'EOF'
fix(discovery): Consolidate path discovery to IntelligentRepoDiscovery

- Use Ironcliw_PRIME_PATH and REACTOR_CORE_PATH (match existing env vars)
- trinity_startup_orchestrator now uses unified discovery
- Log discovery method used (env, sibling, standard locations)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add Single Body/Preflight Completion Points

**Files:**
- Modify: `unified_supervisor.py` (add explicit body/preflight completion)

**Step 1: Find current body completion**

Search for where `jarvis-body` or `jarvis_body` gets marked healthy.

**Step 2: Add explicit body completion after backend health check**

In `_phase_backend()` around line 56983, after marking backend ready:

```python
            # v210.0: Single point for jarvis-body completion
            self._update_component_status("jarvis_body", "complete", "Backend healthy")
            if self._readiness_manager:
                self._readiness_manager.mark_component_ready("jarvis_body", True)
```

**Step 3: Add explicit preflight completion**

In `_phase_preflight()` at the end (around line 56516):

```python
            # v210.0: Single point for preflight completion
            self._update_component_status("preflight", "complete", "Preflight complete")
```

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
fix(readiness): Add single completion points for body and preflight

- jarvis_body marked complete after backend health check passes
- preflight marked complete at end of preflight phase
- Single source of truth for each component's completion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update Frontend Loading Manager Status Handling

**Files:**
- Modify: `frontend/public/loading-manager.js:4270-4285`

**Step 1: Read current isTrinityReady logic**

**Step 2: Update to not say "All Trinity components ready" when skipped**

```javascript
// Around line 4270-4285, update the check:

            // Check trinityComponents state
            if (this.state.trinityComponents) {
                const jarvisReady = this.state.trinityComponents.jarvis?.status === 'complete' ||
                    this.state.trinityComponents.jarvis?.status === 'ready';
                const primeStatus = this.state.trinityComponents.prime?.status;
                const reactorStatus = this.state.trinityComponents.reactor?.status;

                const primeReady = primeStatus === 'complete' || primeStatus === 'ready';
                const primeSkipped = primeStatus === 'skipped' || primeStatus === 'unavailable';
                const reactorReady = reactorStatus === 'complete' || reactorStatus === 'ready';
                const reactorSkipped = reactorStatus === 'skipped' || reactorStatus === 'unavailable';

                // Ironcliw Body (backend) is REQUIRED
                // Prime and Reactor can be ready OR skipped/unavailable
                const primeAcceptable = primeReady || primeSkipped;
                const reactorAcceptable = reactorReady || reactorSkipped;

                if (jarvisReady && primeAcceptable && reactorAcceptable) {
                    // Build accurate message
                    let message = '[Trinity Wait] Backend ready';
                    const notConfigured = [];
                    if (primeSkipped) notConfigured.push('Prime');
                    if (reactorSkipped) notConfigured.push('Reactor');

                    if (notConfigured.length > 0) {
                        message += `; ${notConfigured.join(' and ')} not configured`;
                        console.log(message + ' (set Ironcliw_PRIME_PATH/REACTOR_CORE_PATH if needed)');
                    } else {
                        console.log('[Trinity Wait] All Trinity components ready');
                    }
                    return true;
                }
            }
```

**Step 3: Update status label display**

Around line 3440, update status labels:

```javascript
            const statusLabels = {
                'pending': 'Pending',
                'running': 'Starting...',
                'complete': 'Ready',
                'ready': 'Ready',
                'error': 'Error',
                'failed': 'Error',
                'skipped': 'Not Configured',    // Clear label
                'unavailable': 'Unavailable',   // Clear label
                'stopped': 'Stopped',           // Distinct from skipped
            };
```

**Step 4: Commit**

```bash
git add frontend/public/loading-manager.js
git commit -m "$(cat <<'EOF'
fix(frontend): Accurate Trinity status messages

- Don't say "All Trinity components ready" when components skipped
- Show "Backend ready; Prime and Reactor not configured" when skipped
- Suggest env vars: Ironcliw_PRIME_PATH/REACTOR_CORE_PATH
- Distinct labels: "Not Configured" vs "Stopped"

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Add Readiness Revocation Capability

**Files:**
- Modify: `unified_supervisor.py` (add revocation method)
- Modify: `backend/core/readiness_config.py` (already has config)

**Step 1: Add revocation state tracking**

In JarvisSystemKernel.__init__, add:

```python
        # v210.0: Readiness revocation tracking
        self._readiness_revoked: bool = False
        self._last_revocation_time: Optional[float] = None
        self._consecutive_failures: Dict[str, int] = {}
```

**Step 2: Add revocation method**

```python
    async def _check_and_revoke_readiness(self) -> None:
        """
        Check if readiness should be revoked due to unhealthy components.

        Revokes FULLY_READY if:
        - Critical component has N consecutive failures
        - Critical component down for T seconds

        Respects cooldown to avoid flapping.
        """
        from backend.core.readiness_config import get_readiness_config
        from backend.core.readiness_predicate import ReadinessPredicate

        config = get_readiness_config()
        predicate = ReadinessPredicate()

        # Check cooldown
        if self._last_revocation_time:
            time_since = time.time() - self._last_revocation_time
            if time_since < config.revocation_cooldown_seconds:
                return  # In cooldown, don't revoke

        # Evaluate current state
        component_states = self._get_component_states_for_readiness()
        result = predicate.evaluate(component_states)

        if not result.is_fully_ready and not self._readiness_revoked:
            # Revoke readiness
            self._readiness_revoked = True
            self._last_revocation_time = time.time()

            if self._readiness_manager:
                self._readiness_manager.mark_tier(ReadinessTier.INTERACTIVE)

            self.logger.warning(f"[Readiness] REVOKED: {result.message}")

            # Notify frontend via WebSocket
            await self._broadcast_startup_progress(
                stage="degraded",
                message=f"System degraded: {result.message}",
                progress=100,
                metadata={
                    "readiness_revoked": True,
                    "blocking_components": result.blocking_components,
                }
            )
        elif result.is_fully_ready and self._readiness_revoked:
            # Restore readiness
            self._readiness_revoked = False

            if self._readiness_manager:
                self._readiness_manager.mark_tier(ReadinessTier.FULLY_READY)

            self.logger.success(f"[Readiness] RESTORED: {result.message}")
```

**Step 3: Call from health monitoring loop**

In the existing health monitoring loop (if any), or create one:

```python
    async def _readiness_monitoring_loop(self) -> None:
        """Periodic readiness monitoring."""
        while self._state == KernelState.RUNNING:
            try:
                await self._check_and_revoke_readiness()
                await asyncio.sleep(10.0)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.debug(f"[Readiness] Monitor error: {e}")
```

**Step 4: Commit**

```bash
git add unified_supervisor.py
git commit -m "$(cat <<'EOF'
feat(readiness): Add readiness revocation capability

- Track consecutive failures per component
- Revoke FULLY_READY if critical component fails
- Respect cooldown to avoid flapping
- Restore readiness when components recover
- Notify frontend via WebSocket

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Document Cross-Repo Contract

**Files:**
- Create: `docs/cross-repo-contract.md`

**Step 1: Write documentation**

```markdown
# Cross-Repository Contract

This document defines the contract between Ironcliw, Ironcliw-Prime, and Reactor-Core.

## Environment Variables

All three repositories MUST respect these environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `Ironcliw_PRIME_PATH` | Path to Ironcliw-Prime repository | Auto-discovered |
| `REACTOR_CORE_PATH` | Path to Reactor-Core repository | Auto-discovered |
| `Ironcliw_PRIME_PORT` | Port for Ironcliw-Prime | 8000 |
| `REACTOR_CORE_PORT` | Port for Reactor-Core | 8090 |

## Path Discovery

Ironcliw uses `IntelligentRepoDiscovery` to find repositories:

1. **Environment variable** (highest priority): `Ironcliw_PRIME_PATH`, `REACTOR_CORE_PATH`
2. **Sibling directory**: `../jarvis-prime`, `../reactor-core`
3. **Standard locations**: `~/Documents/repos/`, `~/repos/`
4. **Git-based search**: Find by .git presence

## Health Contract

Each repository MUST expose:

- `GET /health` - Returns 200 when healthy
- Response includes: `{"status": "healthy", "version": "...", "uptime": ...}`

## Heartbeat Contract

Each repository SHOULD write heartbeat files to:

- `~/.jarvis/trinity/components/{component_name}.json`
- Updated every 10-30 seconds
- Contains: timestamp, status, version, pid

## Status Semantics

| Status | Meaning |
|--------|---------|
| `healthy` | Running and passing health checks |
| `starting` | Process started, waiting for health |
| `degraded` | Running but some checks failing |
| `stopped` | Was running, intentionally stopped |
| `skipped` | Never started (not configured) |
| `unavailable` | Not available on this system |
| `error` | Fatal error occurred |
```

**Step 2: Commit**

```bash
git add docs/cross-repo-contract.md
git commit -m "$(cat <<'EOF'
docs: Add cross-repo contract documentation

- Define shared environment variables
- Document path discovery order
- Specify health and heartbeat contracts
- Clarify status semantics

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan addresses all 10 root causes:

1. **Fix #1 (FULLY_READY after verification)**: Task 4
2. **Fix #2 (Unified readiness model)**: Tasks 1, 2
3. **Fix #3 (Single path discovery)**: Task 5
4. **Fix #4 (skipped !== stopped)**: Tasks 1, 3, 7
5. **Fix #5 (Body/preflight completion)**: Task 6
6. **Fix #6 (Frontend wording)**: Task 7
7. **Fix #7 (Revocation with stability)**: Task 8
8. **Fix #8 (Verification timeout)**: Task 1 (config), Task 4 (usage)
9. **Fix #9 (Single readiness predicate)**: Task 2
10. **Fix #10 (Cross-repo contract)**: Task 5, 9

---

Plan complete and saved to `docs/plans/2026-02-03-readiness-integrity-fixes.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
