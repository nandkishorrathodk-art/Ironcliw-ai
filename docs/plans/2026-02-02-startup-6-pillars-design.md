# Startup 6 Pillars — Root Cause Refactor Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix STARTUP TIMEOUT (900s), Duplicate Process Spawning, and macOS M1 16GB optimization through 6 comprehensive pillars.

**Architecture:** Bottom-up timeout calculus with per-phase budgets, single owner policy via spawn_processes flag, hardware-aware Hollow Client enforcement, deep async audit, idempotent tiered shutdown, and centralized env-overridable configuration.

**Tech Stack:** Python 3.11+, asyncio, psutil, dataclasses

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STARTUP FLOW WITH 6 PILLARS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STARTUP ENTRY: Timeout Calculus (Pillar 2)                          │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │                                                                     │   │
│  │ # All from startup_timeouts.py, env-overridable (Pillar 6)         │   │
│  │ # SAFETY_MARGIN is NOT in budgets dict (added separately)          │   │
│  │ budgets = {                                                         │   │
│  │   PRE_TRINITY:     env("JARVIS_PRE_TRINITY_BUDGET", 30),           │   │
│  │   TRINITY_PHASE:   env("JARVIS_TRINITY_PHASE_BUDGET", 300),        │   │
│  │   GCP_WAIT_BUFFER: env("JARVIS_GCP_WAIT_BUFFER", 120),             │   │
│  │   POST_TRINITY:    env("JARVIS_POST_TRINITY_BUDGET", 60),          │   │
│  │   DISCOVERY:       env("JARVIS_DISCOVERY_BUDGET", 45),             │   │
│  │   HEALTH_CHECK:    env("JARVIS_HEALTH_CHECK_BUDGET", 30),          │   │
│  │   CLEANUP:         env("JARVIS_CLEANUP_BUDGET", 30),               │   │
│  │ }                                                                   │   │
│  │ SAFETY_MARGIN = env("JARVIS_SAFETY_MARGIN", 30)                    │   │
│  │ HARD_CAP = env("JARVIS_STARTUP_HARD_CAP", 900)                     │   │
│  │ # Single HARD_CAP for all phases (per-phase caps can be added later)│  │
│  │                                                                     │   │
│  │ # Per-phase effective (so each phase can self-tune from history)   │   │
│  │ for phase in budgets:                                               │   │
│  │     p95_val = history.p95(phase) if history.has(phase) else None   │   │
│  │     if p95_val is not None:                                        │   │
│  │         effective[phase] = min(max(base[phase], p95_val × 1.2),    │   │
│  │                                HARD_CAP)                            │   │
│  │     else:                                                           │   │
│  │         effective[phase] = min(base[phase], HARD_CAP)  # No history│   │
│  │                                                                     │   │
│  │ # Trinity budget (Rule of Three) — only when trinity_enabled       │   │
│  │ if trinity_enabled:                                                 │   │
│  │     trinity_budget = effective[TRINITY_PHASE]                      │   │
│  │                      + (effective[GCP_WAIT_BUFFER] if gcp_enabled  │   │
│  │                         else 0)                                     │   │
│  │ else:                                                               │   │
│  │     trinity_budget = 0                                              │   │
│  │   └─► Used for: (1) global formula, (2) Trinity phase timeout,     │   │
│  │                 (3) DMS operational timeout                         │   │
│  │                                                                     │   │
│  │ # Global timeout — exclude phases that won't run                   │   │
│  │ included_phases = [k for k in budgets if                           │   │
│  │     (trinity_enabled or k not in (TRINITY_PHASE, GCP_WAIT_BUFFER)) │   │
│  │     and (gcp_enabled or k != GCP_WAIT_BUFFER)]                     │   │
│  │ global_timeout = sum(effective[k] for k in included_phases)        │   │
│  │                  + SAFETY_MARGIN                                    │   │
│  │                                                                     │   │
│  │ # Wrap entire startup:                                              │   │
│  │ await asyncio.wait_for(_startup_impl(), timeout=global_timeout)    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 0-4: Pre-Trinity (budget: effective[PRE_TRINITY])            │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ • Hollow Client Enforcer (Pillar 3) runs at module import          │   │
│  │ • RAM threshold: env("JARVIS_HOLLOW_RAM_THRESHOLD_GB", 32)         │   │
│  │ • Sets JARVIS_HOLLOW_CLIENT=true if RAM < threshold                │   │
│  │ • All async calls use asyncio.to_thread() (Pillar 4)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 5: Trinity Phase (budget: trinity_budget) — IF trinity_enabled│  │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │                                                                     │   │
│  │  # unified_supervisor: spawn_processes=False (Trinity is spawner)  │   │
│  │  # run_supervisor/legacy: spawn_processes=True (default)           │   │
│  │                                                                     │   │
│  │  async with orchestrator.startup_lock_context(                     │   │
│  │      spawn_processes=False  # ◄── unified_supervisor path          │   │
│  │  ) as ctx:                                                         │   │
│  │      │                                                             │   │
│  │      │  # __aenter__ does:                                         │   │
│  │      │  #   1. Acquire cross-repo lock (failure raises)            │   │
│  │      │  #   2. Re-enforce hardware env (Hollow Client check)       │   │
│  │      │  #   3. Start GCP pre-warm if enabled                       │   │
│  │      │  #   4. Init cross_repo state (no process spawn)            │   │
│  │      │                                                             │   │
│  │      │  integrator = None  # ◄── Init before try (avoids NameError)│   │
│  │      │                                                             │   │
│  │      │  ┌─────────────────────────────────────────────────────┐   │   │
│  │      │  │ try:                                                 │   │   │
│  │      │  │     async with asyncio.timeout(trinity_budget):     │   │   │
│  │      │  │         integrator = TrinityIntegrator(...)         │   │   │
│  │      │  │         await integrator.initialize()               │   │   │
│  │      │  │         await integrator.start_components()         │   │   │
│  │      │  │         # DMS timeout also set to trinity_budget    │   │   │
│  │      │  │                                                     │   │   │
│  │      │  │ except TimeoutError:                                │   │   │
│  │      │  │     log("Trinity phase timeout after {trinity_budget}s")│ │   │
│  │      │  │     set_state(TRINITY_TIMEOUT)                      │   │   │
│  │      │  │                                                     │   │   │
│  │      │  │ finally:  # ◄── CLEANUP RUNS HERE (Pillar 5)       │   │   │
│  │      │  │     if integrator is not None:  # Safe even if     │   │   │
│  │      │  │         await integrator.stop() # timeout before   │   │   │
│  │      │  │         # Tiered: graceful → force → kill          │   │   │
│  │      │  │         # Idempotent, never raises                 │   │   │
│  │      │  │         # Bounded time (CLEANUP budget)            │   │   │
│  │      │  └─────────────────────────────────────────────────────┘   │   │
│  │      │                                                             │   │
│  │  # __aexit__ guarantees lock release (even on exception/timeout)   │   │
│  │                                                                     │   │
│  │  Note: If Trinity disabled → skip entire block, no integrator.stop │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 6+: Post-Trinity (budgets: POST_TRINITY + DISCOVERY + ...)   │   │
│  │ ─────────────────────────────────────────────────────────────────── │   │
│  │ • Discovery, health checks, final initialization                   │   │
│  │ • Each sub-phase has own effective budget from startup_timeouts.py │   │
│  │ • Lock already released after Phase 5 __aexit__                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

PILLAR MAPPING:
  Pillar 1 (Single Owner)     → spawn_processes=False + lock context (Trinity sole spawner)
  Pillar 2 (Timeout Calculus) → Per-phase effective, conditional inclusion, global_timeout
  Pillar 3 (Hollow Client)    → Module-level, RAM < 32GB threshold (configurable)
  Pillar 4 (Async Audit)      → All blocking I/O wrapped in asyncio.to_thread()
  Pillar 5 (Idempotent Stop)  → finally block, integrator=None init, tiered, bounded
  Pillar 6 (Future-Proofing)  → All budgets in startup_timeouts.py, env-overridable

KEY DEFAULTS (all env-overridable):
  TRINITY_PHASE:      300s (cold start/GCP may need more)
  GCP_WAIT_BUFFER:    120s (added only when gcp_enabled AND trinity_enabled)
  RAM_THRESHOLD:       32GB (below → Hollow Client enforced)
  HARD_CAP:           900s (single cap for all phases; per-phase caps can be added)
  SAFETY_MARGIN:       30s (not in budgets dict; added once to global_timeout)

HISTORY FALLBACK:
  When no history data for a phase: effective[phase] = min(base[phase], HARD_CAP)
```

---

## Pillar 1: Single Owner Policy

**Goal:** Eliminate duplicate process spawning by ensuring only TrinityIntegrator spawns processes when unified_supervisor is the entry point.

**Files:**
- Modify: `backend/supervisor/cross_repo_startup_orchestrator.py`
- Modify: `unified_supervisor.py`

### Task 1.1: Add startup_lock_context() to CrossRepoStartupOrchestrator

**Step 1: Write test for startup_lock_context**

```python
# tests/unit/supervisor/test_cross_repo_startup_orchestrator.py

import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_startup_lock_context_acquires_lock():
    """Lock is acquired in __aenter__ and released in __aexit__."""
    from backend.supervisor.cross_repo_startup_orchestrator import CrossRepoStartupOrchestrator

    orchestrator = CrossRepoStartupOrchestrator()

    with patch.object(orchestrator, '_acquire_lock', new_callable=AsyncMock) as mock_acquire:
        with patch.object(orchestrator, '_release_lock', new_callable=AsyncMock) as mock_release:
            mock_acquire.return_value = True

            async with orchestrator.startup_lock_context(spawn_processes=False) as ctx:
                mock_acquire.assert_called_once()
                assert ctx is not None

            mock_release.assert_called_once()


@pytest.mark.asyncio
async def test_startup_lock_context_spawn_processes_false():
    """When spawn_processes=False, orchestrator does not spawn processes."""
    from backend.supervisor.cross_repo_startup_orchestrator import CrossRepoStartupOrchestrator

    orchestrator = CrossRepoStartupOrchestrator()

    with patch.object(orchestrator, '_acquire_lock', new_callable=AsyncMock, return_value=True):
        with patch.object(orchestrator, '_release_lock', new_callable=AsyncMock):
            with patch.object(orchestrator, '_initialize_cross_repo_state', new_callable=AsyncMock) as mock_init:
                async with orchestrator.startup_lock_context(spawn_processes=False):
                    pass

                mock_init.assert_called_once_with(spawn_processes=False)


@pytest.mark.asyncio
async def test_startup_lock_context_failure_raises():
    """Lock acquisition failure raises StartupLockError."""
    from backend.supervisor.cross_repo_startup_orchestrator import (
        CrossRepoStartupOrchestrator,
        StartupLockError,
    )

    orchestrator = CrossRepoStartupOrchestrator()

    with patch.object(orchestrator, '_acquire_lock', new_callable=AsyncMock, return_value=False):
        with pytest.raises(StartupLockError):
            async with orchestrator.startup_lock_context(spawn_processes=False):
                pass
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/supervisor/test_cross_repo_startup_orchestrator.py -v -k "startup_lock_context"`
Expected: FAIL (startup_lock_context not implemented)

**Step 3: Implement startup_lock_context**

```python
# backend/supervisor/cross_repo_startup_orchestrator.py

from contextlib import asynccontextmanager

class StartupLockError(Exception):
    """Raised when startup lock acquisition fails."""
    pass


class CrossRepoStartupOrchestrator:
    """Cross-Repo Startup Orchestrator v6.0 — Single Owner Policy"""

    @asynccontextmanager
    async def startup_lock_context(
        self,
        spawn_processes: bool = True,
    ):
        """
        Async context manager for startup with cross-repo lock.

        Args:
            spawn_processes: If False, orchestrator does NOT spawn any processes;
                           TrinityIntegrator is the sole spawner (unified_supervisor path).
                           If True, orchestrator spawns processes (legacy/run_supervisor path).
        """
        self._spawn_processes = spawn_processes

        # 1. Acquire lock
        acquired = await self._acquire_lock()
        if not acquired:
            raise StartupLockError("Failed to acquire cross-repo startup lock")

        try:
            # 2. Re-enforce hardware env
            await self._enforce_hardware_environment()

            # 3. GCP pre-warm (if enabled)
            if self._gcp_enabled:
                await self._start_gcp_prewarm()

            # 4. Init cross_repo state (respects spawn_processes flag)
            await self._initialize_cross_repo_state(spawn_processes=spawn_processes)

            yield self

        finally:
            await self._release_lock()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/supervisor/test_cross_repo_startup_orchestrator.py -v -k "startup_lock_context"`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/supervisor/cross_repo_startup_orchestrator.py tests/unit/supervisor/test_cross_repo_startup_orchestrator.py
git commit -m "feat(pillar1): add startup_lock_context() with spawn_processes flag"
```

### Task 1.2: Update unified_supervisor.py to use spawn_processes=False

**Step 1: Write test for unified_supervisor Trinity phase**

```python
# tests/unit/test_unified_supervisor_trinity.py

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_unified_supervisor_uses_spawn_processes_false():
    """unified_supervisor calls startup_lock_context with spawn_processes=False."""
    # Test implementation details...
```

**Step 2-5: Implement and commit**

(Similar TDD pattern)

---

## Pillar 2: Dynamic Timeout Calculus

**Goal:** Replace arbitrary 900s global timeout with bottom-up budgets.

**Files:**
- Modify: `backend/config/startup_timeouts.py`
- Modify: `unified_supervisor.py`

### Task 2.1: Implement StartupTimeoutCalculator

**Step 1: Write test**

```python
# tests/unit/config/test_startup_timeout_calculator.py

import pytest
from backend.config.startup_timeouts import StartupTimeoutCalculator, PhaseBudgets

def test_trinity_budget_with_gcp():
    """trinity_budget includes GCP_WAIT_BUFFER when gcp_enabled."""
    calc = StartupTimeoutCalculator(
        trinity_enabled=True,
        gcp_enabled=True,
    )
    # Default: TRINITY_PHASE=300 + GCP_WAIT_BUFFER=120 = 420
    assert calc.trinity_budget == 420.0


def test_trinity_budget_without_gcp():
    """trinity_budget excludes GCP_WAIT_BUFFER when not gcp_enabled."""
    calc = StartupTimeoutCalculator(
        trinity_enabled=True,
        gcp_enabled=False,
    )
    # Default: TRINITY_PHASE=300 only
    assert calc.trinity_budget == 300.0


def test_trinity_budget_disabled():
    """trinity_budget is 0 when trinity_enabled=False."""
    calc = StartupTimeoutCalculator(
        trinity_enabled=False,
        gcp_enabled=False,
    )
    assert calc.trinity_budget == 0.0


def test_global_timeout_excludes_disabled_phases():
    """global_timeout excludes TRINITY_PHASE and GCP_WAIT_BUFFER when disabled."""
    calc = StartupTimeoutCalculator(
        trinity_enabled=False,
        gcp_enabled=False,
    )
    # Should not include TRINITY_PHASE (300) or GCP_WAIT_BUFFER (120)
    # PRE=30 + POST=60 + DISCOVERY=45 + HEALTH=30 + CLEANUP=30 + SAFETY=30 = 225
    assert calc.global_timeout == 225.0


def test_effective_uses_history_p95():
    """effective budget uses max(base, p95*1.2) when history exists."""
    mock_history = MagicMock()
    mock_history.has.return_value = True
    mock_history.get_p95.return_value = 400.0  # p95 = 400s

    calc = StartupTimeoutCalculator(
        trinity_enabled=True,
        gcp_enabled=False,
        history=mock_history,
    )
    # TRINITY_PHASE: max(300, 400*1.2) = max(300, 480) = 480, capped at HARD_CAP=900
    assert calc.effective("TRINITY_PHASE") == 480.0


def test_effective_fallback_when_no_history():
    """effective budget = base when no history data."""
    calc = StartupTimeoutCalculator(
        trinity_enabled=True,
        gcp_enabled=False,
        history=None,
    )
    assert calc.effective("TRINITY_PHASE") == 300.0
```

**Step 2-5: Implement and commit**

(Full implementation as shown in Pillar 2 design section)

---

## Pillar 3: Hollow Client Enforcer

**Goal:** Automatically enforce Hollow Client mode on machines with insufficient RAM.

**Files:**
- Create: `backend/config/hardware_enforcer.py`
- Modify: `backend/supervisor/cross_repo_startup_orchestrator.py`

### Task 3.1: Implement hardware_enforcer.py

**Step 1: Write test**

```python
# tests/unit/config/test_hardware_enforcer.py

import pytest
import os
from unittest.mock import patch

def test_enforce_hollow_client_when_ram_below_threshold():
    """Sets JARVIS_HOLLOW_CLIENT=true when RAM < threshold."""
    from backend.config import hardware_enforcer

    with patch.object(hardware_enforcer, 'get_system_ram_gb', return_value=16.0):
        with patch.dict(os.environ, {}, clear=True):
            result = hardware_enforcer.enforce_hollow_client(source="test")
            assert result is True
            assert os.environ.get("JARVIS_HOLLOW_CLIENT") == "true"


def test_no_enforcement_when_ram_above_threshold():
    """Does not set JARVIS_HOLLOW_CLIENT when RAM >= threshold."""
    from backend.config import hardware_enforcer

    with patch.object(hardware_enforcer, 'get_system_ram_gb', return_value=64.0):
        with patch.dict(os.environ, {}, clear=True):
            result = hardware_enforcer.enforce_hollow_client(source="test")
            assert result is False
            assert os.environ.get("JARVIS_HOLLOW_CLIENT") is None


def test_threshold_is_configurable():
    """RAM threshold is configurable via JARVIS_HOLLOW_RAM_THRESHOLD_GB."""
    from backend.config import hardware_enforcer

    with patch.object(hardware_enforcer, 'get_system_ram_gb', return_value=48.0):
        with patch.dict(os.environ, {"JARVIS_HOLLOW_RAM_THRESHOLD_GB": "64"}, clear=True):
            # Reload to pick up new threshold
            import importlib
            importlib.reload(hardware_enforcer)

            result = hardware_enforcer.enforce_hollow_client(source="test")
            assert result is True  # 48 < 64
```

**Step 2-5: Implement and commit**

---

## Pillar 4: Deep Async Audit

**Goal:** Wrap all blocking I/O calls with asyncio.to_thread().

**Files:**
- Create: `backend/utils/async_io.py`
- Modify: Various files in startup path

### Task 4.1: Implement async_io.py utilities

**Step 1: Write test**

```python
# tests/unit/utils/test_async_io.py

import pytest
import asyncio
from backend.utils.async_io import run_sync, path_exists, run_subprocess

@pytest.mark.asyncio
async def test_run_sync_executes_in_executor():
    """run_sync executes blocking function without blocking event loop."""
    import time

    def blocking_func():
        time.sleep(0.1)
        return "done"

    result = await run_sync(blocking_func)
    assert result == "done"


@pytest.mark.asyncio
async def test_run_subprocess_returns_completed_process():
    """run_subprocess returns CompletedProcess with stdout/stderr."""
    result = await run_subprocess(["echo", "hello"])
    assert result.returncode == 0
    assert b"hello" in result.stdout


@pytest.mark.asyncio
async def test_run_subprocess_timeout():
    """run_subprocess raises TimeoutError on timeout."""
    with pytest.raises(asyncio.TimeoutError):
        await run_subprocess(["sleep", "10"], timeout=0.1)
```

**Step 2-5: Implement and commit**

### Task 4.2: Audit and fix blocking calls

**Files to audit:**
1. `unified_supervisor.py` — grep for subprocess, open(, time.sleep, requests.
2. `backend/core/trinity_integrator.py` — grep for same patterns
3. `backend/supervisor/cross_repo_startup_orchestrator.py` — grep for same patterns

**Pattern replacements:**
- `subprocess.run()` → `await run_subprocess()`
- `time.sleep()` → `await asyncio.sleep()`
- `open().read()` → `await read_file()`
- `psutil.*` → `await run_sync(psutil.*)`

---

## Pillar 5: Idempotent Cleanup

**Goal:** Ensure TrinityIntegrator.stop() is tiered, idempotent, never raises, bounded.

**Files:**
- Modify: `backend/core/trinity_integrator.py`

### Task 5.1: Implement tiered stop() method

**Step 1: Write test**

```python
# tests/unit/core/test_trinity_integrator_stop.py

import pytest
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_stop_is_idempotent():
    """Calling stop() multiple times is safe."""
    from backend.core.trinity_integrator import TrinityIntegrator

    integrator = TrinityIntegrator(...)

    result1 = await integrator.stop()
    result2 = await integrator.stop()

    # Second call returns empty (already stopped)
    assert result2 == {}


@pytest.mark.asyncio
async def test_stop_never_raises():
    """stop() catches all exceptions and returns results."""
    from backend.core.trinity_integrator import TrinityIntegrator

    integrator = TrinityIntegrator(...)
    integrator.register_pid(99999)  # Non-existent PID

    # Should not raise
    result = await integrator.stop()
    assert 99999 in result


@pytest.mark.asyncio
async def test_stop_bounded_by_timeout():
    """stop() completes within timeout even if processes hang."""
    from backend.core.trinity_integrator import TrinityIntegrator

    integrator = TrinityIntegrator(...)
    # Register a PID that won't respond to signals

    import time
    start = time.monotonic()
    result = await integrator.stop(timeout=2.0)
    elapsed = time.monotonic() - start

    assert elapsed < 3.0  # Should complete within timeout + buffer


@pytest.mark.asyncio
async def test_stop_tiered_graceful_then_force():
    """stop() tries SIGTERM first, then SIGKILL."""
    from backend.core.trinity_integrator import TrinityIntegrator

    integrator = TrinityIntegrator(...)
    # Test that SIGTERM is sent first, then SIGKILL if needed
```

**Step 2-5: Implement and commit**

---

## Pillar 6: Future-Proofing

**Goal:** Centralize all config in startup_timeouts.py with env-overridable values.

**Files:**
- Modify: `backend/config/startup_timeouts.py`

### Task 6.1: Implement StartupConfig dataclass

**Step 1: Write test**

```python
# tests/unit/config/test_startup_config.py

import pytest
import os
from unittest.mock import patch

def test_config_env_override():
    """Config values can be overridden via environment variables."""
    with patch.dict(os.environ, {"JARVIS_TRINITY_PHASE_BUDGET": "600"}):
        from backend.config.startup_timeouts import StartupConfig
        config = StartupConfig()
        assert config.TRINITY_PHASE_BUDGET == 600.0


def test_config_defaults():
    """Config has sensible defaults when no env vars set."""
    with patch.dict(os.environ, {}, clear=True):
        from backend.config.startup_timeouts import StartupConfig
        config = StartupConfig()
        assert config.TRINITY_PHASE_BUDGET == 300.0
        assert config.HOLLOW_RAM_THRESHOLD_GB == 32.0
        assert config.HARD_CAP == 900.0


def test_get_phase_budgets_returns_dict():
    """get_phase_budgets() returns all phase budgets as dict."""
    from backend.config.startup_timeouts import StartupConfig
    config = StartupConfig()
    budgets = config.get_phase_budgets()

    assert "PRE_TRINITY" in budgets
    assert "TRINITY_PHASE" in budgets
    assert "GCP_WAIT_BUFFER" in budgets
```

**Step 2-5: Implement and commit**

---

## Integration: unified_supervisor.py Changes

### Task 7.1: Wire all pillars into unified_supervisor.py

**Step 1: Write integration test**

```python
# tests/integration/test_unified_supervisor_pillars.py

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_startup_uses_calculated_timeouts():
    """Startup uses timeouts from StartupTimeoutCalculator."""
    # Integration test verifying all pillars work together
```

**Step 2: Implementation sketch**

```python
# unified_supervisor.py

import asyncio
from backend.config.startup_timeouts import config, StartupTimeoutCalculator, load_startup_metrics_history
import backend.config.hardware_enforcer  # noqa: F401 (module-level enforcement)

async def _startup_impl():
    """Main startup implementation."""

    # Pillar 2: Calculate timeouts
    calculator = StartupTimeoutCalculator(
        budgets=config.get_phase_budgets(),
        history=load_startup_metrics_history(),
        trinity_enabled=config.TRINITY_ENABLED,
        gcp_enabled=config.GCP_ENABLED,
    )

    trinity_budget = calculator.trinity_budget

    # Phase 0-4: Pre-Trinity
    async with asyncio.timeout(calculator.effective("PRE_TRINITY")):
        await _run_pre_trinity_phases()

    # Phase 5: Trinity (if enabled)
    if config.TRINITY_ENABLED:
        async with orchestrator.startup_lock_context(spawn_processes=False) as ctx:
            integrator = None
            try:
                async with asyncio.timeout(trinity_budget):
                    integrator = TrinityIntegrator(...)
                    await integrator.initialize()
                    await integrator.start_components()
            except asyncio.TimeoutError:
                logger.error(f"Trinity phase timeout after {trinity_budget}s")
            finally:
                if integrator is not None:
                    await integrator.stop(timeout=config.CLEANUP_BUDGET)

    # Phase 6+: Post-Trinity
    async with asyncio.timeout(calculator.effective("POST_TRINITY")):
        await _run_post_trinity_phases()


async def main():
    """Entry point with global timeout."""
    calculator = StartupTimeoutCalculator(...)
    config.log_config()

    try:
        await asyncio.wait_for(
            _startup_impl(),
            timeout=calculator.global_timeout,
        )
    except asyncio.TimeoutError:
        logger.critical(f"STARTUP TIMEOUT after {calculator.global_timeout}s")
        raise
```

**Step 3-5: Implement and commit**

---

## Summary

| Pillar | Files | Key Changes |
|--------|-------|-------------|
| 1. Single Owner | cross_repo_startup_orchestrator.py, unified_supervisor.py | startup_lock_context(spawn_processes=False) |
| 2. Timeout Calculus | startup_timeouts.py, unified_supervisor.py | StartupTimeoutCalculator, per-phase effective |
| 3. Hollow Client | hardware_enforcer.py (new), cross_repo_startup_orchestrator.py | Module-level + lock context enforcement |
| 4. Async Audit | async_io.py (new), various | run_sync(), run_subprocess(), audit fixes |
| 5. Idempotent Stop | trinity_integrator.py | Tiered stop(), idempotent, bounded |
| 6. Future-Proofing | startup_timeouts.py | StartupConfig dataclass, env-overridable |
| Integration | unified_supervisor.py | Wire all pillars together |
