# Unified Monolith Refactor - v111.0

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform JARVIS from subprocess-spawning architecture to a unified in-process orchestration model where `run_supervisor.py` is the single entry point that imports and manages all components within one Python process.

**Architecture:** Single asyncio event loop orchestrating FastAPI (Uvicorn in-process), cross-repo services, and system management - no subprocess spawning, shared memory, graceful shutdown.

**Tech Stack:** Python 3.10+, asyncio, FastAPI, Uvicorn, aiohttp

---

## Problem Statement

**Current Architecture (Problematic):**
```
run_supervisor.py (process 1)
    └── subprocess.Popen(start_system.py)  # (process 2)
            └── subprocess.Popen(backend/main.py)  # (process 3)
    └── subprocess.Popen(jarvis-prime)  # (process 4)
    └── subprocess.Popen(reactor-core)  # (process 5)
```

**Issues:**
1. **Resource exhaustion**: Multiple Python interpreters competing for RAM/CPU
2. **Complex coordination**: PID files, fcntl locks, environment variables for IPC
3. **Fragile shutdown**: Ctrl+C doesn't propagate cleanly to all children
4. **No shared state**: Services can't share memory, causing redundant work
5. **Dual entry points**: Both `run_supervisor.py` and `start_system.py` try to be "main"

**Target Architecture (Unified):**
```
python3 run_supervisor.py  # Single process
    └── asyncio event loop
        ├── backend FastAPI (uvicorn.Server.serve())
        ├── cross-repo orchestrator (in-process)
        ├── jarvis-prime subprocess (external, must remain subprocess)
        ├── reactor-core subprocess (external, must remain subprocess)
        └── graceful shutdown handler
```

---

## Phase 1: Make start_system.py Import-Safe

### Task 1.1: Extract AsyncSystemManager to Standalone Module

**Files:**
- Create: `backend/core/async_system_manager.py`
- Modify: `start_system.py`

**Step 1: Read the existing AsyncSystemManager class**

The class exists at line 8317 in `start_system.py`. Read it to understand its structure.

**Step 2: Create the new module**

```python
# backend/core/async_system_manager.py
"""
v111.0: Unified Monolith - AsyncSystemManager
=============================================

Extracted from start_system.py to enable import without execution.
This class manages the JARVIS backend system lifecycle.

Author: JARVIS System
Version: 111.0.0 (January 2026)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SystemPhase(str, Enum):
    """System lifecycle phases."""
    INIT = "init"
    STARTING = "starting"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class SystemState:
    """Current system state."""
    phase: SystemPhase = SystemPhase.INIT
    started_at: Optional[datetime] = None
    services_healthy: Dict[str, bool] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None


class AsyncSystemManager:
    """
    v111.0: Import-safe async system manager.

    This class can be imported and instantiated without triggering
    any side effects. Call start() to begin system operations.

    Usage:
        manager = AsyncSystemManager(config)
        await manager.start()
        # ... run until shutdown
        await manager.stop()
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        uvicorn_host: str = "0.0.0.0",
        uvicorn_port: int = 8010,
        log_level: str = "info",
    ):
        """
        Initialize the system manager.

        Args:
            config: Optional configuration dictionary
            uvicorn_host: Host to bind Uvicorn server
            uvicorn_port: Port for the FastAPI backend
            log_level: Logging level
        """
        self.config = config or {}
        self.uvicorn_host = uvicorn_host
        self.uvicorn_port = uvicorn_port
        self.log_level = log_level

        self._state = SystemState()
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._uvicorn_server: Optional[Any] = None

        # Callbacks for lifecycle events
        self._on_start_callbacks: List[Callable[[], None]] = []
        self._on_stop_callbacks: List[Callable[[], None]] = []

    @property
    def state(self) -> SystemState:
        """Get current system state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if system is running."""
        return self._state.phase == SystemPhase.RUNNING

    def on_start(self, callback: Callable[[], None]) -> None:
        """Register a callback to run when system starts."""
        self._on_start_callbacks.append(callback)

    def on_stop(self, callback: Callable[[], None]) -> None:
        """Register a callback to run when system stops."""
        self._on_stop_callbacks.append(callback)

    async def start(self) -> None:
        """
        Start the system.

        This method:
        1. Initializes the FastAPI application
        2. Starts Uvicorn server in-process
        3. Runs startup callbacks
        4. Transitions to RUNNING phase
        """
        if self._state.phase not in (SystemPhase.INIT, SystemPhase.STOPPED):
            logger.warning(f"Cannot start from phase {self._state.phase}")
            return

        self._state.phase = SystemPhase.STARTING
        self._state.started_at = datetime.now()

        try:
            logger.info("[v111.0] AsyncSystemManager starting...")

            # Import FastAPI app (deferred to avoid import-time side effects)
            from backend.main import app

            # Configure and start Uvicorn in-process
            import uvicorn

            config = uvicorn.Config(
                app=app,
                host=self.uvicorn_host,
                port=self.uvicorn_port,
                log_level=self.log_level,
                access_log=True,
            )
            self._uvicorn_server = uvicorn.Server(config)

            # Start server as background task
            server_task = asyncio.create_task(
                self._uvicorn_server.serve(),
                name="uvicorn-server"
            )
            self._tasks.append(server_task)

            # Wait for server to be ready
            while not self._uvicorn_server.started:
                await asyncio.sleep(0.1)

            logger.info(f"[v111.0] ✅ Backend started on {self.uvicorn_host}:{self.uvicorn_port}")

            # Run start callbacks
            for callback in self._on_start_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Start callback error: {e}")

            self._state.phase = SystemPhase.RUNNING
            self._state.services_healthy["backend"] = True

            logger.info("[v111.0] ✅ AsyncSystemManager running")

        except Exception as e:
            logger.error(f"[v111.0] ❌ Start failed: {e}")
            self._state.phase = SystemPhase.FAILED
            raise

    async def stop(self, timeout: float = 10.0) -> None:
        """
        Stop the system gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if self._state.phase == SystemPhase.STOPPED:
            return

        self._state.phase = SystemPhase.SHUTTING_DOWN
        logger.info("[v111.0] AsyncSystemManager shutting down...")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop Uvicorn server
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True

        # Cancel all tasks with timeout
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            done, pending = await asyncio.wait(
                self._tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            for task in pending:
                logger.warning(f"Task {task.get_name()} didn't complete, forcing cancel")
                task.cancel()

        # Run stop callbacks
        for callback in self._on_stop_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Stop callback error: {e}")

        self._state.phase = SystemPhase.STOPPED
        logger.info("[v111.0] ✅ AsyncSystemManager stopped")

    async def wait_for_shutdown(self) -> None:
        """Wait until shutdown is signaled."""
        await self._shutdown_event.wait()

    def signal_shutdown(self) -> None:
        """Signal the system to shut down."""
        self._shutdown_event.set()


# Module-level factory
_manager_instance: Optional[AsyncSystemManager] = None


def get_system_manager(
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AsyncSystemManager:
    """
    Get or create the system manager singleton.

    Args:
        config: Configuration dictionary (only used on first call)
        **kwargs: Additional arguments for AsyncSystemManager

    Returns:
        The singleton AsyncSystemManager instance
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = AsyncSystemManager(config, **kwargs)
    return _manager_instance


def reset_system_manager() -> None:
    """Reset the singleton (for testing)."""
    global _manager_instance
    _manager_instance = None
```

**Step 3: Update start_system.py to use the extracted module**

At the end of start_system.py (around line 21950), update the main block:

```python
# start_system.py - End of file

if __name__ == "__main__":
    # v111.0: Support both direct execution and import

    # v110.0: SINGLETON ENFORCEMENT
    if _SINGLETON_AVAILABLE:
        is_running, existing_state = is_supervisor_running()
        if is_running and existing_state:
            print("\n" + "="*60)
            print("❌ JARVIS SUPERVISOR ALREADY RUNNING!")
            print("="*60)
            print(f"  Entry point: {existing_state.get('entry_point', 'unknown')}")
            print(f"  PID: {existing_state.get('pid', 'unknown')}")
            print(f"  Started: {existing_state.get('started_at', 'unknown')}")
            print(f"  Working dir: {existing_state.get('working_dir', 'unknown')}")
            print()
            print("To stop the existing supervisor:")
            print(f"  kill {existing_state.get('pid', 'PID')}")
            print("="*60 + "\n")
            sys.exit(1)

        if not acquire_supervisor_lock("start_system"):
            print("❌ Failed to acquire supervisor lock")
            sys.exit(1)

    try:
        # Use the extracted manager for actual execution
        from backend.core.async_system_manager import get_system_manager

        manager = get_system_manager()
        asyncio.run(manager.start())
        asyncio.run(manager.wait_for_shutdown())

    finally:
        if _SINGLETON_AVAILABLE:
            release_supervisor_lock()
```

**Step 4: Commit**

```bash
git add backend/core/async_system_manager.py start_system.py
git commit -m "refactor: extract AsyncSystemManager for import-safe usage (v111.0)

- Created backend/core/async_system_manager.py
- AsyncSystemManager can now be imported without side effects
- Supports both direct execution and programmatic use
- Phase 1.1 of Unified Monolith Refactor
"
```

---

### Task 1.2: Make backend/main.py Import-Safe

**Files:**
- Modify: `backend/main.py`

**Step 1: Read current main.py structure**

The file has the FastAPI app at line 4074 and likely calls `uvicorn.run()` at the bottom.

**Step 2: Wrap uvicorn.run in __name__ guard**

Ensure the file can be imported without starting the server:

```python
# backend/main.py - End of file (around line 4074+)

# v111.0: Import-safe guard
# The FastAPI `app` object is always available for import.
# uvicorn.run() only executes when this file is run directly.

if __name__ == "__main__":
    import uvicorn

    # Default standalone execution (for development/testing)
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8010,
        reload=False,
        log_level="info",
    )
```

**Step 3: Verify app is importable**

```bash
python3 -c "from backend.main import app; print(f'App imported: {app}')"
```

**Step 4: Commit**

```bash
git add backend/main.py
git commit -m "refactor: make backend/main.py import-safe (v111.0)

- Wrapped uvicorn.run() in __name__ == '__main__' guard
- FastAPI app can now be imported without starting server
- Phase 1.2 of Unified Monolith Refactor
"
```

---

## Phase 2: Unify run_supervisor.py as Single Entry Point

### Task 2.1: Refactor Supervisor to Import Components Directly

**Files:**
- Modify: `run_supervisor.py`

**Step 1: Read current supervisor structure**

The supervisor is around 22,000 lines with `SupervisorBootstrapper` at line 4163.

**Step 2: Add in-process backend management**

Add a new method to start the backend in-process instead of via subprocess:

```python
# run_supervisor.py - Add to SupervisorBootstrapper class (around line 4163)

class SupervisorBootstrapper:
    """
    v111.0: Unified supervisor that runs all components in-process.
    """

    def __init__(self):
        # ... existing init ...

        # v111.0: In-process components
        self._backend_server: Optional[uvicorn.Server] = None
        self._backend_task: Optional[asyncio.Task] = None
        self._system_manager: Optional[AsyncSystemManager] = None

    async def _start_backend_in_process(self) -> bool:
        """
        v111.0: Start the JARVIS backend in-process using Uvicorn.

        This replaces subprocess spawning with direct in-process execution,
        enabling shared memory and coordinated shutdown.

        Returns:
            True if backend started successfully
        """
        try:
            logger.info("[v111.0] Starting backend in-process...")

            # Import the FastAPI app
            from backend.main import app
            import uvicorn

            # Configure Uvicorn
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=8010,
                log_level="info",
                access_log=True,
                # Don't install signal handlers - we manage signals
                # in the supervisor
            )

            self._backend_server = uvicorn.Server(config)

            # Override signal handlers (supervisor manages these)
            self._backend_server.install_signal_handlers = lambda: None

            # Start as background task
            self._backend_task = asyncio.create_task(
                self._backend_server.serve(),
                name="backend-uvicorn"
            )

            # Wait for server to be ready
            max_wait = 30.0
            start_time = time.time()
            while not self._backend_server.started:
                if time.time() - start_time > max_wait:
                    raise TimeoutError("Backend didn't start within 30s")
                await asyncio.sleep(0.1)

            elapsed = time.time() - start_time
            logger.info(f"[v111.0] ✅ Backend started in-process in {elapsed:.1f}s")
            return True

        except Exception as e:
            logger.error(f"[v111.0] ❌ Failed to start backend: {e}")
            return False

    async def _stop_backend_in_process(self, timeout: float = 10.0) -> None:
        """
        v111.0: Stop the in-process backend gracefully.
        """
        if self._backend_server:
            logger.info("[v111.0] Stopping backend...")
            self._backend_server.should_exit = True

            if self._backend_task and not self._backend_task.done():
                try:
                    await asyncio.wait_for(
                        self._backend_task,
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("[v111.0] Backend didn't stop gracefully, cancelling")
                    self._backend_task.cancel()
                    try:
                        await self._backend_task
                    except asyncio.CancelledError:
                        pass

            logger.info("[v111.0] ✅ Backend stopped")
```

**Step 3: Update the main run loop**

Modify the `run()` method to use in-process backend:

```python
# run_supervisor.py - Modify run() method (around line 4890)

async def run(self) -> None:
    """
    v111.0: Unified run loop with in-process components.
    """
    try:
        # v110.0: Singleton enforcement
        if _SINGLETON_AVAILABLE:
            if not acquire_supervisor_lock("run_supervisor"):
                logger.error("[Singleton] Another supervisor is running!")
                return
            await start_supervisor_heartbeat()

        logger.info("="*60)
        logger.info("🚀 JARVIS Unified Supervisor v111.0")
        logger.info("="*60)

        # Phase 1: Start backend in-process
        backend_ok = await self._start_backend_in_process()
        if not backend_ok:
            raise RuntimeError("Backend failed to start")

        # Phase 2: Start cross-repo services (still subprocesses for now)
        # jarvis-prime and reactor-core are external repos
        await self._start_external_services()

        # Phase 3: Run until shutdown
        logger.info("[v111.0] ✅ All systems running - awaiting shutdown signal")
        await self._wait_for_shutdown()

    except asyncio.CancelledError:
        logger.info("[v111.0] Supervisor cancelled")
    except Exception as e:
        logger.error(f"[v111.0] Supervisor error: {e}")
        raise
    finally:
        # Graceful shutdown
        await self._shutdown_all()

        if _SINGLETON_AVAILABLE:
            release_supervisor_lock()

async def _shutdown_all(self) -> None:
    """
    v111.0: Coordinated shutdown of all components.
    """
    logger.info("[v111.0] Beginning coordinated shutdown...")

    # Stop external services first (they depend on backend)
    await self._stop_external_services()

    # Stop backend last
    await self._stop_backend_in_process()

    logger.info("[v111.0] ✅ All components stopped")
```

**Step 4: Commit**

```bash
git add run_supervisor.py
git commit -m "refactor: run backend in-process instead of subprocess (v111.0)

- Added _start_backend_in_process() using uvicorn.Server.serve()
- Added _stop_backend_in_process() for graceful shutdown
- Backend now shares memory with supervisor
- Ctrl+C propagates correctly to backend
- Phase 2.1 of Unified Monolith Refactor
"
```

---

### Task 2.2: Add Graceful Signal Handling

**Files:**
- Modify: `run_supervisor.py`

**Step 1: Add unified signal handler**

```python
# run_supervisor.py - Add to class or module level

class UnifiedSignalHandler:
    """
    v111.0: Unified signal handling for the monolith.

    Handles SIGINT (Ctrl+C) and SIGTERM gracefully, ensuring
    all components shut down in the correct order.
    """

    def __init__(self):
        self._shutdown_event = asyncio.Event()
        self._shutdown_requested = False
        self._shutdown_count = 0

    def install(self, loop: asyncio.AbstractEventLoop) -> None:
        """Install signal handlers on the event loop."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            )
        logger.info("[v111.0] Signal handlers installed")

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle incoming signal."""
        self._shutdown_count += 1
        sig_name = sig.name

        if self._shutdown_count == 1:
            logger.info(f"\n[v111.0] Received {sig_name} - initiating graceful shutdown...")
            self._shutdown_requested = True
            self._shutdown_event.set()
        elif self._shutdown_count == 2:
            logger.warning(f"[v111.0] Received second {sig_name} - forcing faster shutdown...")
        else:
            logger.error(f"[v111.0] Received third {sig_name} - forcing immediate exit!")
            sys.exit(128 + sig)

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown_requested
```

**Step 2: Integrate into main()**

```python
# run_supervisor.py - Update main() at bottom of file

async def main() -> int:
    """
    v111.0: Unified main entry point.
    """
    # Setup signal handling
    signal_handler = UnifiedSignalHandler()
    loop = asyncio.get_event_loop()
    signal_handler.install(loop)

    # Create and run supervisor
    supervisor = SupervisorBootstrapper()

    try:
        # Run supervisor with signal-aware shutdown
        supervisor_task = asyncio.create_task(supervisor.run())
        shutdown_task = asyncio.create_task(signal_handler.wait_for_shutdown())

        # Wait for either completion or shutdown signal
        done, pending = await asyncio.wait(
            [supervisor_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        if shutdown_task in done:
            # Shutdown signal received
            logger.info("[v111.0] Shutdown signal received, stopping supervisor...")
            supervisor_task.cancel()
            try:
                await supervisor_task
            except asyncio.CancelledError:
                pass

        return 0

    except Exception as e:
        logger.error(f"[v111.0] Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

**Step 3: Commit**

```bash
git add run_supervisor.py
git commit -m "feat: add unified signal handling for graceful shutdown (v111.0)

- UnifiedSignalHandler captures SIGINT/SIGTERM
- First signal: graceful shutdown
- Second signal: faster shutdown
- Third signal: immediate exit
- All components shut down in correct order
- Phase 2.2 of Unified Monolith Refactor
"
```

---

### Task 2.3: Remove Subprocess Spawning for Backend

**Files:**
- Modify: `run_supervisor.py`

**Step 1: Find and remove subprocess spawning**

Search for `subprocess.Popen` or similar calls that spawn `start_system.py` or `backend/main.py`:

```python
# REMOVE or comment out code like:
# process = subprocess.Popen(
#     [sys.executable, "start_system.py"],
#     ...
# )

# REPLACE with in-process call (already added in Task 2.1)
```

**Step 2: Update any process tracking**

```python
# If there's a dict tracking subprocess PIDs, remove backend entries:
# self._managed_processes.pop("backend", None)

# Instead, track in-process components differently:
self._in_process_components = {
    "backend": {
        "server": self._backend_server,
        "task": self._backend_task,
        "started_at": datetime.now(),
    }
}
```

**Step 3: Commit**

```bash
git add run_supervisor.py
git commit -m "refactor: remove subprocess spawning for backend (v111.0)

- Removed subprocess.Popen calls for start_system.py/backend
- Backend now managed via in-process Uvicorn
- Subprocess tracking updated for in-process components
- Phase 2.3 of Unified Monolith Refactor
"
```

---

## Phase 3: Integration and Testing

### Task 3.1: Create Integration Test

**Files:**
- Create: `tests/integration/test_unified_monolith.py`

**Step 1: Write the test**

```python
# tests/integration/test_unified_monolith.py
"""
v111.0: Integration tests for unified monolith architecture.

Tests that verify:
1. Single process runs all components
2. Graceful shutdown works correctly
3. No subprocess spawning for backend
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
import pytest
from unittest.mock import patch, MagicMock


class TestUnifiedMonolith:
    """Test the unified monolith architecture."""

    @pytest.mark.asyncio
    async def test_backend_starts_in_process(self):
        """Verify backend starts in-process, not as subprocess."""
        try:
            from backend.main import app
            import uvicorn
        except ImportError:
            pytest.skip("Required modules not available")

        # Verify app is importable
        assert app is not None
        assert hasattr(app, "routes")

        # Create server config
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=19999,  # Test port
            log_level="error",
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None

        # Start server
        task = asyncio.create_task(server.serve())

        try:
            # Wait for server to start
            max_wait = 10.0
            start = time.time()
            while not server.started:
                if time.time() - start > max_wait:
                    pytest.fail("Server didn't start in time")
                await asyncio.sleep(0.1)

            # Verify it's running in our process
            assert server.started

            # Make a test request
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:19999/health") as resp:
                    assert resp.status in (200, 404)  # 404 if no /health route

        finally:
            server.should_exit = True
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_async_system_manager_lifecycle(self):
        """Test AsyncSystemManager start/stop lifecycle."""
        try:
            from backend.core.async_system_manager import (
                AsyncSystemManager,
                SystemPhase,
                reset_system_manager,
            )
        except ImportError:
            pytest.skip("AsyncSystemManager not available")

        # Reset singleton
        reset_system_manager()

        manager = AsyncSystemManager(
            uvicorn_host="127.0.0.1",
            uvicorn_port=19998,
            log_level="error",
        )

        assert manager.state.phase == SystemPhase.INIT

        # Start
        await manager.start()
        assert manager.state.phase == SystemPhase.RUNNING
        assert manager.is_running

        # Stop
        await manager.stop()
        assert manager.state.phase == SystemPhase.STOPPED
        assert not manager.is_running

    def test_no_subprocess_in_supervisor(self):
        """Verify supervisor doesn't spawn backend as subprocess."""
        # Read supervisor source and check for subprocess calls
        import inspect
        try:
            # This would be the refactored version
            from run_supervisor import SupervisorBootstrapper

            source = inspect.getsource(SupervisorBootstrapper)

            # Should NOT contain subprocess spawning for backend
            # (May still have subprocess for external repos like jarvis-prime)
            assert "subprocess.Popen" not in source or \
                   "start_system.py" not in source, \
                   "Supervisor still spawns backend as subprocess"

        except ImportError:
            pytest.skip("Supervisor not importable")


class TestGracefulShutdown:
    """Test graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_signal_handler_first_signal(self):
        """Test that first SIGINT initiates graceful shutdown."""
        try:
            from run_supervisor import UnifiedSignalHandler
        except ImportError:
            pytest.skip("UnifiedSignalHandler not available")

        handler = UnifiedSignalHandler()

        assert not handler.shutdown_requested

        # Simulate first signal
        await handler._handle_signal(signal.SIGINT)

        assert handler.shutdown_requested
        assert handler._shutdown_count == 1

    @pytest.mark.asyncio
    async def test_signal_handler_multiple_signals(self):
        """Test escalating shutdown on multiple signals."""
        try:
            from run_supervisor import UnifiedSignalHandler
        except ImportError:
            pytest.skip("UnifiedSignalHandler not available")

        handler = UnifiedSignalHandler()

        # First signal
        await handler._handle_signal(signal.SIGINT)
        assert handler._shutdown_count == 1

        # Second signal
        await handler._handle_signal(signal.SIGINT)
        assert handler._shutdown_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: Run tests**

```bash
pytest tests/integration/test_unified_monolith.py -v
```

**Step 3: Commit**

```bash
git add tests/integration/test_unified_monolith.py
git commit -m "test: add unified monolith integration tests (v111.0)

- Test backend starts in-process
- Test AsyncSystemManager lifecycle
- Test graceful shutdown with signals
- Phase 3.1 of Unified Monolith Refactor
"
```

---

### Task 3.2: Manual Verification

**Step 1: Start the unified system**

```bash
python3 run_supervisor.py
```

**Expected output:**
```
==================================================
🚀 JARVIS Unified Supervisor v111.0
==================================================
[v111.0] Signal handlers installed
[v111.0] Starting backend in-process...
[v111.0] ✅ Backend started in-process in 1.2s
[v111.0] Starting external services...
[v111.0] ✅ jarvis-prime started
[v111.0] ✅ reactor-core started
[v111.0] ✅ All systems running - awaiting shutdown signal
```

**Step 2: Verify single process**

```bash
# Should show only ONE python3 process for supervisor
ps aux | grep -E "run_supervisor|start_system" | grep -v grep
```

**Step 3: Test graceful shutdown**

Press Ctrl+C and verify:
```
[v111.0] Received SIGINT - initiating graceful shutdown...
[v111.0] Beginning coordinated shutdown...
[v111.0] Stopping external services...
[v111.0] ✅ jarvis-prime stopped
[v111.0] ✅ reactor-core stopped
[v111.0] Stopping backend...
[v111.0] ✅ Backend stopped
[v111.0] ✅ All components stopped
```

**Step 4: Commit final verification**

```bash
git add -A
git commit -m "chore: unified monolith refactor complete (v111.0)

- Single entry point: python3 run_supervisor.py
- Backend runs in-process via Uvicorn
- Graceful Ctrl+C shutdown
- External services (jarvis-prime, reactor-core) remain subprocesses
- Phase 3.2 complete - Unified Monolith Refactor finished
"
```

---

## Summary

| Phase | Task | Description | Status |
|-------|------|-------------|--------|
| 1.1 | Extract AsyncSystemManager | Create import-safe module | TODO |
| 1.2 | Make main.py import-safe | Add __name__ guard | TODO |
| 2.1 | In-process backend | Use uvicorn.Server.serve() | TODO |
| 2.2 | Signal handling | UnifiedSignalHandler | TODO |
| 2.3 | Remove subprocess spawning | Clean up Popen calls | TODO |
| 3.1 | Integration tests | Verify architecture | TODO |
| 3.2 | Manual verification | End-to-end test | TODO |

## Success Criteria

- [ ] Single `python3 run_supervisor.py` command starts everything
- [ ] `ps aux` shows only one Python process for supervisor
- [ ] Ctrl+C cleanly stops all components in order
- [ ] Backend shares memory with supervisor (no IPC needed)
- [ ] No subprocess spawning for backend (jarvis-prime/reactor-core may remain subprocess)
- [ ] All tests pass

## Rollback Plan

If issues occur, revert to previous subprocess model:
```bash
git revert HEAD~N  # Revert N commits
```

The previous architecture with subprocess spawning remains in git history.
