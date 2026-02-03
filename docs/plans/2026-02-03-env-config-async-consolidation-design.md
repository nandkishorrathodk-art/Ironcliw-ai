# Environment Config & Async Consolidation Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the two minor issues from the 6 Pillars code review by consolidating duplicate env parsing functions and converting blocking calls to async.

**Architecture:** Create `backend/utils/env_config.py` as the single source of truth for environment variable parsing, extend `backend/utils/async_io.py` with psutil wrappers, then migrate all 12+ files to use these consolidated utilities.

**Tech Stack:** Python 3.11+, asyncio, psutil, dataclasses, typing

---

## Context

### Problem Statement

The 6 Pillars code review identified two minor issues:

1. **Duplicate `_get_env_float` implementations**: 12+ files each define their own env parsing functions with slight variations, creating maintenance burden and inconsistency.

2. **Blocking calls in sync helper functions**: Several functions contain blocking psutil/subprocess calls that could cause issues if called from async contexts.

### Phase Strategy

- **Phase 1 (This Plan)**: Fix issues in JARVIS-AI-Agent only
- **Phase 2 (Future)**: Extract to `jarvis-common` package consumed by JARVIS, JARVIS-Prime, and reactor-core

---

## Section 1: `backend/utils/env_config.py` Module

### Purpose

Generic layer for type-safe environment variable parsing. All config modules (startup_timeouts.py, hardware_enforcer.py, voice_unlock/config.py, etc.) import from here.

### API Surface

```python
"""
Environment Configuration Utilities
====================================

Generic layer for type-safe environment variable parsing.
Config modules like startup_timeouts.py use these primitives.

Behavior:
- On parse/validation error: log warning, return default (never raise)
- All parsing is case-insensitive where applicable
- Designed for jarvis-common extraction in Phase 2
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from typing import Any, ClassVar, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_env_str(key: str, default: str = "") -> str:
    """Get string environment variable."""
    return os.environ.get(key, default)


def get_env_optional_str(key: str) -> str | None:
    """
    Get optional string environment variable.

    Returns:
        None if key is not set (unset)
        "" if key is set but empty
        value if key is set with value

    Use this when unset != empty matters.
    """
    return os.environ.get(key)  # Returns None if not set


def get_env_int(
    key: str,
    default: int,
    *,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """
    Get integer environment variable with optional bounds validation.

    On parse error: logs warning, returns default.
    On bounds violation: logs warning, clamps to bounds.
    """
    ...


def get_env_float(
    key: str,
    default: float,
    *,
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    """
    Get float environment variable with optional bounds validation.

    On parse error: logs warning, returns default.
    On bounds violation: logs warning, clamps to bounds.
    """
    ...


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.

    Case-insensitive parsing:
        True:  "true", "1", "yes"
        False: "false", "0", "no", "" (empty), unset

    On unrecognized value: logs warning, returns default.
    """
    ...


def get_env_list(
    key: str,
    default: list[str] | None = None,
    *,
    separator: str = ",",
    strip: bool = True,
) -> list[str]:
    """
    Get list environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set (None becomes [])
        separator: Delimiter between elements (default: ",")
        strip: Strip whitespace from each element (default: True)

    Example:
        JARVIS_FEATURES="feat1, feat2, feat3"
        get_env_list("JARVIS_FEATURES") -> ["feat1", "feat2", "feat3"]
    """
    ...


@dataclass
class EnvConfig:
    """
    Base class for type-safe environment config sections.

    Convention:
        Field 'my_setting' -> env key '{prefix}MY_SETTING'
        Default prefix: "JARVIS_"
        Override via class attribute: _env_prefix = "CUSTOM_"

    Example:
        @dataclass
        class VoiceConfig(EnvConfig):
            sample_rate: int = 16000      # -> JARVIS_SAMPLE_RATE
            base_threshold: float = 0.85  # -> JARVIS_BASE_THRESHOLD
            enabled: bool = True          # -> JARVIS_ENABLED

        config = VoiceConfig.from_env()

    Behavior:
        - Uses get_env_* functions internally
        - Never raises on parse errors
        - Logs warnings for invalid values
    """
    _env_prefix: ClassVar[str] = "JARVIS_"

    @classmethod
    def from_env(cls) -> "EnvConfig":
        """Create instance from environment variables."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes class variables)."""
        ...
```

### Error Handling Examples

```python
# Parse error - logs warning, returns default:
# WARNING [EnvConfig] JARVIS_TIMEOUT: cannot parse 'abc' as float, using default 30.0

# Validation error - logs warning, clamps to bounds:
# WARNING [EnvConfig] JARVIS_TIMEOUT: value -5.0 below minimum 0.0, using 0.0

# Bool unrecognized - logs warning, returns default:
# WARNING [EnvConfig] JARVIS_ENABLED: unrecognized bool 'maybe', using default True
```

---

## Section 2: Async System Utilities

### Additions to `backend/utils/async_io.py`

```python
"""
Async I/O Utilities (Extended)
==============================

Existing:
- run_sync(func, *args, **kwargs) -> T
- path_exists(path) -> bool
- read_file(path, encoding) -> str
- run_subprocess(cmd, timeout, **kwargs) -> CompletedProcess

New (psutil wrappers):
- pid_exists(pid) -> bool
- get_process(pid) -> psutil.Process | None
- process_is_running(pid) -> bool
- iter_processes(attrs) -> AsyncIterator[dict]
- get_net_connections(kind) -> list[sconn]
- get_cpu_percent(interval) -> float
- get_virtual_memory() -> svmem
- get_disk_usage(path) -> sdiskusage
"""

import psutil
from typing import Any, AsyncIterator


# =============================================================================
# Process Utilities
# =============================================================================

async def pid_exists(pid: int) -> bool:
    """Check if process exists without blocking."""
    return await run_sync(psutil.pid_exists, pid)


async def get_process(pid: int) -> psutil.Process | None:
    """
    Get Process object for given PID.

    Returns:
        Process object if found
        None if process doesn't exist or access denied

    Never raises - returns None for expected error conditions.
    """
    try:
        return await run_sync(psutil.Process, pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


async def process_is_running(pid: int) -> bool:
    """
    Check if process is running.

    Returns False if process doesn't exist, access denied, or not running.
    Never raises.
    """
    proc = await get_process(pid)
    if proc is None:
        return False
    try:
        return await run_sync(proc.is_running)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


async def iter_processes(
    attrs: list[str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Iterate over running processes without blocking.

    Args:
        attrs: Process attributes to include (default: ['pid', 'name'])

    Yields:
        Process info dictionaries

    Silently skips processes that disappear or become inaccessible.
    """
    attrs = attrs or ['pid', 'name']
    procs = await run_sync(lambda: list(psutil.process_iter(attrs)))
    for proc in procs:
        try:
            yield proc.info
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


# =============================================================================
# Network Utilities
# =============================================================================

async def get_net_connections(kind: str = 'inet') -> list:
    """
    Get network connections without blocking.

    Args:
        kind: Connection type ('inet', 'inet4', 'inet6', 'tcp', 'udp', 'all')

    Returns:
        List of named tuples with connection info
    """
    return await run_sync(psutil.net_connections, kind)


# =============================================================================
# System Resource Utilities
# =============================================================================

async def get_cpu_percent(interval: float = 0.1) -> float:
    """
    Get CPU usage percentage.

    Args:
        interval: Sampling interval in seconds (blocks during measurement)

    Note: Use small intervals to minimize blocking time.
    """
    return await run_sync(psutil.cpu_percent, interval=interval)


async def get_virtual_memory():
    """Get virtual memory statistics without blocking."""
    return await run_sync(psutil.virtual_memory)


async def get_disk_usage(path: str = '/'):
    """Get disk usage statistics without blocking."""
    return await run_sync(psutil.disk_usage, path)
```

### Error Handling Philosophy

- **Never raise** for expected conditions (process not found, access denied)
- Return `None` or `False` for missing/inaccessible resources
- Log warnings only for unexpected errors
- Consistent with env_config's "never raise" principle

---

## Section 3: Migration Strategy

### Files to Migrate

| Priority | File | Functions to Remove |
|----------|------|---------------------|
| **P1** | `backend/config/startup_timeouts.py` | `_get_env_float`, `_get_env_bool` |
| **P1** | `backend/config/hardware_enforcer.py` | `_get_env_float` |
| **P2** | `backend/intelligence/git_intelligence.py` | `_get_env`, `_get_env_int`, `_get_env_float`, `_get_env_bool` |
| **P2** | `backend/intelligence/cloud_sql_connection_manager.py` | `_get_env_int`, `_get_env_float`, `_get_env_bool` |
| **P3** | `backend/intelligence/computer_use_refinements.py` | env parsing functions |
| **P3** | `backend/intelligence/unified_memory_system.py` | env parsing functions |
| **P3** | `backend/intelligence/repository_intelligence.py` | env parsing functions |
| **P3** | `backend/intelligence/cross_repo_refactoring.py` | env parsing functions |
| **P3** | `backend/intelligence/speaker_profile_store.py` | env parsing functions |

### Migration Pattern

```python
# BEFORE (duplicate in each file):
def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        return default

TIMEOUT = _get_env_float("JARVIS_TIMEOUT", 30.0)


# AFTER (consolidated):
from backend.utils.env_config import get_env_float

TIMEOUT = get_env_float("JARVIS_TIMEOUT", 30.0)
```

---

## Section 4: Blocking Call Conversion

### Category 1: Initialization-Only (Document, Don't Convert)

These run before the event loop starts:

| Function | Call | Action |
|----------|------|--------|
| `_calculate_memory_budget()` | `psutil.virtual_memory().total` | Add docstring note |
| `_find_poetry_python()` | `subprocess.run()` | Add docstring note |
| `_find_pipenv_python()` | `subprocess.run()` | Add docstring note |

```python
def _calculate_memory_budget() -> int:
    """
    Calculate memory budget for subprocess pools.

    Note: Called during module initialization before event loop.
    Blocking psutil call is acceptable here.
    """
    ...
```

### Category 2: Runtime (Must Convert)

These run inside async functions:

| Location | Blocking Call | Async Replacement |
|----------|---------------|-------------------|
| Process lifecycle | `psutil.pid_exists()` | `await pid_exists()` |
| Process lifecycle | `psutil.Process(pid)` | `await get_process()` |
| Process lifecycle | `proc.is_running()` | `await process_is_running()` |
| Port discovery | `psutil.process_iter()` | `async for p in iter_processes()` |
| Port discovery | `psutil.net_connections()` | `await get_net_connections()` |
| Resource monitoring | `psutil.cpu_percent()` | `await get_cpu_percent()` |
| Resource monitoring | `psutil.virtual_memory()` | `await get_virtual_memory()` |

---

## Section 5: Task Breakdown

### Task 1: Create backend/utils/env_config.py

**Files:**
- Create: `backend/utils/env_config.py`

**Step 1: Write the test file first**

Create `tests/unit/utils/test_env_config.py` with tests for:
- `get_env_str()` - default, set, empty
- `get_env_optional_str()` - unset vs empty distinction
- `get_env_int()` - valid, invalid, bounds
- `get_env_float()` - valid, invalid, bounds
- `get_env_bool()` - true/1/yes, false/0/no/empty, invalid
- `get_env_list()` - separator, strip, empty
- `EnvConfig.from_env()` - field to env key convention
- Warning logging on parse/validation errors

**Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/utils/test_env_config.py -v
```

**Step 3: Implement env_config.py**

Implement all functions per API specification.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/utils/test_env_config.py -v
```

**Step 5: Commit**

```bash
git add backend/utils/env_config.py tests/unit/utils/test_env_config.py
git commit -m "feat(env-config): add consolidated environment parsing utilities"
```

---

### Task 2: Extend async_io.py with psutil wrappers

**Files:**
- Modify: `backend/utils/async_io.py`
- Modify: `tests/unit/utils/test_async_io.py`

**Step 1: Add tests for psutil wrappers**

Add to existing test file:
- `test_pid_exists()` - valid pid, invalid pid
- `test_get_process()` - valid, invalid, returns None not raises
- `test_process_is_running()` - running, not running, not found
- `test_iter_processes()` - yields dicts, skips inaccessible
- `test_get_net_connections()` - returns list
- `test_get_cpu_percent()` - returns float
- `test_get_virtual_memory()` - returns svmem
- `test_get_disk_usage()` - returns sdiskusage

**Step 2: Run tests to verify new tests fail**

```bash
pytest tests/unit/utils/test_async_io.py -v -k "pid_exists or get_process or process_is_running"
```

**Step 3: Implement psutil wrappers**

Add all functions per Section 2 specification.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/utils/test_async_io.py -v
```

**Step 5: Commit**

```bash
git add backend/utils/async_io.py tests/unit/utils/test_async_io.py
git commit -m "feat(async-io): add psutil async wrappers"
```

---

### Task 3: Migrate startup_timeouts.py to env_config

**Files:**
- Modify: `backend/config/startup_timeouts.py`

**Step 1: Add import**

```python
from backend.utils.env_config import get_env_float, get_env_bool
```

**Step 2: Find and replace usages**

Replace `_get_env_float(...)` with `get_env_float(...)`.
Replace `_get_env_bool(...)` with `get_env_bool(...)`.

**Step 3: Remove local function definitions**

Delete the `def _get_env_float(...)` and `def _get_env_bool(...)` functions.

**Step 4: Run existing tests**

```bash
pytest tests/unit/config/test_startup_timeout_calculator.py tests/unit/config/test_startup_config.py -v
```

**Step 5: Commit**

```bash
git add backend/config/startup_timeouts.py
git commit -m "refactor(startup-timeouts): migrate to centralized env_config"
```

---

### Task 4: Migrate hardware_enforcer.py to env_config

**Files:**
- Modify: `backend/config/hardware_enforcer.py`

**Step 1: Add import**

```python
from backend.utils.env_config import get_env_float
```

**Step 2: Replace usage and remove local function**

**Step 3: Run existing tests**

```bash
pytest tests/unit/config/test_hardware_enforcer.py -v
```

**Step 4: Commit**

```bash
git add backend/config/hardware_enforcer.py
git commit -m "refactor(hardware-enforcer): migrate to centralized env_config"
```

---

### Task 5: Convert blocking calls in unified_supervisor.py

**Files:**
- Modify: `unified_supervisor.py`

**Step 1: Add import for async psutil wrappers**

```python
from backend.utils.async_io import (
    pid_exists,
    get_process,
    process_is_running,
    iter_processes,
    get_net_connections,
    get_cpu_percent,
    get_virtual_memory,
)
```

**Step 2: Find all runtime blocking calls**

Search for psutil usage in async functions.

**Step 3: Convert each to async wrapper**

Example:
```python
# Before
if psutil.pid_exists(pid):

# After
if await pid_exists(pid):
```

**Step 4: Document init-only sync functions**

Add docstring notes to `_calculate_memory_budget()`, `_find_poetry_python()`, `_find_pipenv_python()`.

**Step 5: Syntax check**

```bash
python3 -m py_compile unified_supervisor.py && echo "Syntax OK"
```

**Step 6: Commit**

```bash
git add unified_supervisor.py
git commit -m "refactor(supervisor): convert blocking psutil calls to async wrappers"
```

---

### Task 6: Migrate intelligence modules (parallelizable)

**Files:**
- `backend/intelligence/git_intelligence.py`
- `backend/intelligence/cloud_sql_connection_manager.py`
- `backend/intelligence/computer_use_refinements.py`
- `backend/intelligence/unified_memory_system.py`
- `backend/intelligence/repository_intelligence.py`
- `backend/intelligence/cross_repo_refactoring.py`
- `backend/intelligence/speaker_profile_store.py`

**Per file:**

1. Add import: `from backend.utils.env_config import get_env_str, get_env_int, get_env_float, get_env_bool`
2. Replace all `_get_env_*` calls with centralized versions
3. Remove local function definitions
4. Syntax check: `python3 -m py_compile <file>`
5. Commit

---

## Dependency Graph

```
Task 1 (env_config.py)
    │
    ├──► Task 3 (startup_timeouts.py)
    │
    ├──► Task 4 (hardware_enforcer.py)
    │
    └──► Task 6 (intelligence modules)

Task 2 (async_io.py extensions)
    │
    └──► Task 5 (unified_supervisor.py)
```

Tasks 1 and 2 can run in parallel.
Tasks 3, 4, 5, 6 depend on their respective prerequisites.
Task 6 subtasks (each module) can run in parallel.

---

## Success Criteria

1. **No duplicate env parsing**: All files use `backend/utils/env_config.py`
2. **No runtime blocking calls**: All psutil/subprocess in async functions use wrappers
3. **All tests pass**: Existing + new tests green
4. **Code review clean**: No minor issues remaining
5. **Phase 2 ready**: env_config.py designed for extraction to jarvis-common

---

## Out of Scope (Phase 2)

- jarvis-common package creation
- Cross-repo installation mechanism (pip install)
- JARVIS-Prime and reactor-core migration
- voice_unlock/config.py EnvConfig conversion (larger refactor)
