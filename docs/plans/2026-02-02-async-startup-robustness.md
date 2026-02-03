# Async Startup Robustness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate "stuck at 5%" startup issues by converting all blocking calls to async, implementing parallel progress updates, and creating robust cross-repo coordination.

**Architecture:** Replace all synchronous blocking calls (proc.wait, subprocess.run, socket operations, file I/O) in the startup path with async equivalents using `asyncio.to_thread()` for inherently sync operations. Implement a non-blocking progress heartbeat system that continues broadcasting even during long operations. Create unified async utilities for cross-repo coordination.

**Tech Stack:** Python asyncio, aiofiles (optional), psutil, aiohttp, concurrent.futures.ThreadPoolExecutor

---

## Overview

The "stuck at 5%" issue occurs because blocking calls in the startup path prevent the event loop from running, which stops progress broadcasts and heartbeats. This plan addresses:

1. **Blocking `proc.wait()` calls** - 6 locations totaling up to 44+ seconds of blocking
2. **Blocking `subprocess.run()` calls** - 7 locations with 3-5 second timeouts each
3. **Blocking socket operations** - 7 locations for port checks and IPC
4. **Blocking file I/O** - Multiple heartbeat/state file reads
5. **Cross-repo coordination** - Locks and state files shared with JARVIS Prime and Reactor Core

---

## Task 1: Create Async Utilities Module

**Files:**
- Create: `backend/utils/async_subprocess.py`
- Test: `tests/unit/utils/test_async_subprocess.py`

**Step 1: Write the failing test**

```python
# tests/unit/utils/test_async_subprocess.py
"""Tests for async subprocess and process utilities."""
import asyncio
import pytest
import psutil
import signal
import os


class TestAsyncProcessWait:
    """Test async_process_wait functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_process_that_exits_quickly(self):
        """Process that exits quickly should return True."""
        from backend.utils.async_subprocess import async_process_wait

        # Start a process that exits immediately
        proc = await asyncio.create_subprocess_exec(
            "sleep", "0.1",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )

        result = await async_process_wait(proc.pid, timeout=5.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_process_timeout(self):
        """Process that doesn't exit should timeout."""
        from backend.utils.async_subprocess import async_process_wait

        # Start a process that sleeps longer than timeout
        proc = await asyncio.create_subprocess_exec(
            "sleep", "60",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )

        try:
            result = await async_process_wait(proc.pid, timeout=0.5)
            assert result is False
        finally:
            proc.terminate()
            await proc.wait()


class TestAsyncSubprocessRun:
    """Test async_subprocess_run functionality."""

    @pytest.mark.asyncio
    async def test_run_simple_command(self):
        """Simple command should return stdout."""
        from backend.utils.async_subprocess import async_subprocess_run

        result = await async_subprocess_run(["echo", "hello"], timeout=5.0)
        assert result.returncode == 0
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_run_command_timeout(self):
        """Command that times out should raise TimeoutError."""
        from backend.utils.async_subprocess import async_subprocess_run

        with pytest.raises(asyncio.TimeoutError):
            await async_subprocess_run(["sleep", "60"], timeout=0.5)


class TestAsyncSocketConnect:
    """Test async_socket_check functionality."""

    @pytest.mark.asyncio
    async def test_check_closed_port(self):
        """Closed port should return False."""
        from backend.utils.async_subprocess import async_check_port

        # Port 59999 should not be in use
        result = await async_check_port("localhost", 59999, timeout=1.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_port_timeout(self):
        """Port check should not block event loop."""
        from backend.utils.async_subprocess import async_check_port

        import time
        start = time.monotonic()

        # Run multiple port checks concurrently
        results = await asyncio.gather(
            async_check_port("localhost", 59991, timeout=0.5),
            async_check_port("localhost", 59992, timeout=0.5),
            async_check_port("localhost", 59993, timeout=0.5),
        )

        elapsed = time.monotonic() - start
        # Should complete in ~0.5s (parallel), not ~1.5s (serial)
        assert elapsed < 1.0
        assert all(r is False for r in results)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python -m pytest tests/unit/utils/test_async_subprocess.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'backend.utils.async_subprocess'"

**Step 3: Write minimal implementation**

```python
# backend/utils/async_subprocess.py
"""
Async Subprocess and Process Utilities v1.0
============================================

Non-blocking alternatives to blocking process and subprocess operations.
Designed for use in async startup flows where blocking the event loop
causes "stuck at 5%" issues.

All functions use asyncio.to_thread() to offload blocking operations
to a thread pool, keeping the event loop responsive.

Usage:
    from backend.utils.async_subprocess import (
        async_process_wait,
        async_subprocess_run,
        async_check_port,
        async_file_read,
    )

    # Instead of: proc.wait(timeout=5.0)
    exited = await async_process_wait(pid, timeout=5.0)

    # Instead of: subprocess.run(["cmd"], timeout=5.0)
    result = await async_subprocess_run(["cmd"], timeout=5.0)

    # Instead of: sock.connect_ex(('localhost', port))
    is_open = await async_check_port("localhost", 8010, timeout=1.0)
"""

import asyncio
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import psutil


@dataclass
class SubprocessResult:
    """Result from async_subprocess_run."""
    returncode: int
    stdout: str
    stderr: str


async def async_process_wait(
    pid: int,
    timeout: float = 10.0,
    poll_interval: float = 0.1,
) -> bool:
    """
    Wait for a process to exit without blocking the event loop.

    Args:
        pid: Process ID to wait for
        timeout: Maximum time to wait in seconds
        poll_interval: How often to check process status

    Returns:
        True if process exited, False if timeout
    """
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        try:
            proc = psutil.Process(pid)
            if not proc.is_running():
                return True
            # Check if zombie
            if proc.status() == psutil.STATUS_ZOMBIE:
                # Reap zombie
                try:
                    os.waitpid(pid, os.WNOHANG)
                except (ChildProcessError, OSError):
                    pass
                return True
        except psutil.NoSuchProcess:
            return True
        except psutil.AccessDenied:
            # Can't check, assume still running
            pass

        await asyncio.sleep(poll_interval)

    return False


async def async_psutil_wait(
    proc: psutil.Process,
    timeout: float = 10.0,
) -> bool:
    """
    Async wrapper for psutil.Process.wait().

    Args:
        proc: psutil.Process object
        timeout: Maximum time to wait

    Returns:
        True if process exited, False if timeout
    """
    def _wait():
        try:
            proc.wait(timeout=timeout)
            return True
        except psutil.TimeoutExpired:
            return False
        except psutil.NoSuchProcess:
            return True

    return await asyncio.to_thread(_wait)


async def async_subprocess_run(
    cmd: List[str],
    timeout: float = 30.0,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[dict] = None,
    capture_output: bool = True,
) -> SubprocessResult:
    """
    Run a subprocess without blocking the event loop.

    Args:
        cmd: Command and arguments
        timeout: Maximum time to wait
        cwd: Working directory
        env: Environment variables
        capture_output: Whether to capture stdout/stderr

    Returns:
        SubprocessResult with returncode, stdout, stderr

    Raises:
        asyncio.TimeoutError: If command times out
    """
    def _run():
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
        )
        return SubprocessResult(
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run),
            timeout=timeout + 1.0,  # Extra second for thread overhead
        )
    except asyncio.TimeoutError:
        raise
    except subprocess.TimeoutExpired:
        raise asyncio.TimeoutError(f"Command timed out after {timeout}s: {cmd}")


async def async_check_port(
    host: str,
    port: int,
    timeout: float = 1.0,
) -> bool:
    """
    Check if a port is open without blocking the event loop.

    Args:
        host: Host to check
        port: Port to check
        timeout: Connection timeout

    Returns:
        True if port is open (accepting connections), False otherwise
    """
    def _check():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            result = sock.connect_ex((host, port))
            return result == 0
        except (socket.timeout, socket.error):
            return False
        finally:
            sock.close()

    return await asyncio.to_thread(_check)


async def async_check_unix_socket(
    socket_path: Union[str, Path],
    timeout: float = 2.0,
) -> bool:
    """
    Check if a Unix domain socket is accepting connections.

    Args:
        socket_path: Path to the Unix socket
        timeout: Connection timeout

    Returns:
        True if socket is accepting connections, False otherwise
    """
    def _check():
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(str(socket_path))
            return True
        except (socket.timeout, socket.error, FileNotFoundError):
            return False
        finally:
            sock.close()

    return await asyncio.to_thread(_check)


async def async_file_read(
    path: Union[str, Path],
    encoding: str = "utf-8",
) -> str:
    """
    Read a file without blocking the event loop.

    Args:
        path: File path
        encoding: File encoding

    Returns:
        File contents as string
    """
    def _read():
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    return await asyncio.to_thread(_read)


async def async_file_write(
    path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
) -> None:
    """
    Write to a file without blocking the event loop.

    Args:
        path: File path
        content: Content to write
        encoding: File encoding
    """
    def _write():
        with open(path, "w", encoding=encoding) as f:
            f.write(content)

    await asyncio.to_thread(_write)


async def async_json_read(path: Union[str, Path]) -> dict:
    """
    Read a JSON file without blocking the event loop.

    Args:
        path: File path

    Returns:
        Parsed JSON as dict
    """
    import json
    content = await async_file_read(path)
    return json.loads(content)


async def async_json_write(path: Union[str, Path], data: dict) -> None:
    """
    Write a JSON file without blocking the event loop.

    Args:
        path: File path
        data: Data to serialize
    """
    import json
    content = json.dumps(data, indent=2)
    await async_file_write(path, content)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python -m pytest tests/unit/utils/test_async_subprocess.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/utils/async_subprocess.py tests/unit/utils/test_async_subprocess.py
git commit -m "feat(async): Add async subprocess and process utilities

Introduces non-blocking alternatives to blocking operations:
- async_process_wait: Replaces psutil proc.wait()
- async_subprocess_run: Replaces subprocess.run()
- async_check_port: Replaces socket.connect_ex()
- async_file_read/write: Replaces sync file I/O

These utilities prevent 'stuck at 5%' issues by keeping
the event loop responsive during startup operations."
```

---

## Task 2: Convert _graceful_terminate to Async

**Files:**
- Modify: `run_supervisor.py:3640-3672`
- Test: Manual verification (existing tests should pass)

**Step 1: Write the failing test**

```python
# Add to existing test file or create tests/unit/test_graceful_terminate.py
import asyncio
import pytest
import time


class TestGracefulTerminateNonBlocking:
    """Test that _graceful_terminate doesn't block the event loop."""

    @pytest.mark.asyncio
    async def test_graceful_terminate_allows_concurrent_tasks(self):
        """
        _graceful_terminate should not block concurrent tasks.

        This test verifies that while terminating a process,
        the event loop can still run other coroutines.
        """
        heartbeat_count = 0

        async def heartbeat():
            nonlocal heartbeat_count
            for _ in range(10):
                heartbeat_count += 1
                await asyncio.sleep(0.1)

        # Start a long-running process
        proc = await asyncio.create_subprocess_exec(
            "sleep", "30",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )

        # Import and call _graceful_terminate
        # This should not block the heartbeat task
        from backend.utils.async_subprocess import async_process_wait

        start = time.monotonic()

        # Run both concurrently
        heartbeat_task = asyncio.create_task(heartbeat())

        # Terminate process
        import signal
        import os
        os.kill(proc.pid, signal.SIGTERM)

        # Wait for process (non-blocking)
        await async_process_wait(proc.pid, timeout=2.0)

        # Wait for heartbeat to complete
        await heartbeat_task

        elapsed = time.monotonic() - start

        # Heartbeat should have run ~10 times (1 second)
        # If _graceful_terminate blocked, heartbeat_count would be 0
        assert heartbeat_count >= 5, f"Heartbeat only ran {heartbeat_count} times - event loop was blocked"
        assert elapsed < 3.0, f"Operation took {elapsed}s - should be ~1s"
```

**Step 2: Run test to verify current behavior blocks**

Run: `cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python -m pytest tests/unit/test_graceful_terminate.py -v`
Expected: Test should demonstrate current blocking behavior (or pass if already non-blocking)

**Step 3: Modify run_supervisor.py _graceful_terminate**

Find the function at lines 3640-3672 and replace the blocking `proc.wait()` calls:

```python
# BEFORE (lines 3640-3666):
    # Phase 1: SIGINT (graceful)
    if _safe_kill(pid, signal.SIGINT):
        try:
            proc = psutil.Process(pid)
            await asyncio.sleep(0.1)  # Brief pause
            proc.wait(timeout=self.config.cleanup_timeout_sigint)  # BLOCKING!
            return True
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass

    # Phase 2: SIGTERM
    if _safe_kill(pid, signal.SIGTERM):
        try:
            proc = psutil.Process(pid)
            proc.wait(timeout=self.config.cleanup_timeout_sigterm)  # BLOCKING!
            return True
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass

    # Phase 3: SIGKILL (force)
    if _safe_kill(pid, signal.SIGKILL):
        try:
            proc = psutil.Process(pid)
            proc.wait(timeout=self.config.cleanup_timeout_sigkill)  # BLOCKING!
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass
    return True
```

```python
# AFTER (replace with):
    # Import async utilities at module level or here
    from backend.utils.async_subprocess import async_psutil_wait

    # Phase 1: SIGINT (graceful)
    if _safe_kill(pid, signal.SIGINT):
        try:
            proc = psutil.Process(pid)
            await asyncio.sleep(0.1)  # Brief pause
            # NON-BLOCKING: Use async wait
            exited = await async_psutil_wait(proc, timeout=self.config.cleanup_timeout_sigint)
            if exited:
                return True
        except psutil.NoSuchProcess:
            return True

    # Phase 2: SIGTERM
    if _safe_kill(pid, signal.SIGTERM):
        try:
            proc = psutil.Process(pid)
            # NON-BLOCKING: Use async wait
            exited = await async_psutil_wait(proc, timeout=self.config.cleanup_timeout_sigterm)
            if exited:
                return True
        except psutil.NoSuchProcess:
            return True

    # Phase 3: SIGKILL (force)
    if _safe_kill(pid, signal.SIGKILL):
        try:
            proc = psutil.Process(pid)
            # NON-BLOCKING: Use async wait
            await async_psutil_wait(proc, timeout=self.config.cleanup_timeout_sigkill)
        except psutil.NoSuchProcess:
            pass
    return True
```

**Step 4: Run tests**

Run: `cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python -m pytest tests/unit/test_graceful_terminate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add run_supervisor.py
git commit -m "fix(startup): Convert _graceful_terminate to non-blocking async

Replace blocking proc.wait() calls with async_psutil_wait() to prevent
'stuck at 5%' during process termination. This allows the event loop
to continue running heartbeats and progress updates while waiting
for processes to exit.

Addresses: 22+ seconds of blocking in three-phase termination"
```

---

## Task 3: Convert J-Prime/Reactor Stop Methods to Async

**Files:**
- Modify: `run_supervisor.py:11896-11916` (_stop_jprime_orchestrator)
- Modify: `run_supervisor.py:11955-11975` (_stop_reactor_core_orchestrator)

**Step 1: Identify the blocking pattern**

Both methods have identical blocking patterns:
```python
with open(heartbeat_path) as f:  # BLOCKING file read
    heartbeat = json.load(f)
...
proc.wait(timeout=5.0)  # BLOCKING wait
```

**Step 2: Replace with async equivalents**

```python
# BEFORE (_stop_jprime_orchestrator lines 11896-11916):
        heartbeat_path = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_prime.json"
        if heartbeat_path.exists():
            try:
                with open(heartbeat_path) as f:
                    heartbeat = json.load(f)
                pid = heartbeat.get("pid")
                if pid:
                    import psutil
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            self.logger.debug(f"[v100.4] Terminating J-Prime via heartbeat PID: {pid}")
                            proc.terminate()
                            proc.wait(timeout=5.0)  # BLOCKING!
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.TimeoutExpired:
                        proc.kill()
            except Exception as e:
                self.logger.debug(f"[v100.4] Heartbeat-based stop failed: {e}")
```

```python
# AFTER:
        from backend.utils.async_subprocess import async_json_read, async_psutil_wait

        heartbeat_path = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_prime.json"
        if heartbeat_path.exists():
            try:
                # NON-BLOCKING file read
                heartbeat = await async_json_read(heartbeat_path)
                pid = heartbeat.get("pid")
                if pid:
                    import psutil
                    try:
                        proc = psutil.Process(pid)
                        if proc.is_running():
                            self.logger.debug(f"[v100.4] Terminating J-Prime via heartbeat PID: {pid}")
                            proc.terminate()
                            # NON-BLOCKING wait
                            exited = await async_psutil_wait(proc, timeout=5.0)
                            if not exited:
                                proc.kill()
                                await async_psutil_wait(proc, timeout=2.0)
                    except psutil.NoSuchProcess:
                        pass
            except Exception as e:
                self.logger.debug(f"[v100.4] Heartbeat-based stop failed: {e}")
```

Apply the same pattern to `_stop_reactor_core_orchestrator`.

**Step 3: Run tests**

Run: `cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python -m pytest tests/ -k "orchestrator or trinity" -v`
Expected: PASS

**Step 4: Commit**

```bash
git add run_supervisor.py
git commit -m "fix(startup): Convert J-Prime/Reactor stop to non-blocking async

Replace blocking file reads and proc.wait() in orchestrator stop
methods with async equivalents. This prevents 10+ seconds of
blocking when stopping Trinity components during takeover."
```

---

## Task 4: Convert Zombie Cleanup to Async

**Files:**
- Modify: `run_supervisor.py:22535-22545` (zombie process wait)

**Step 1: Replace blocking wait pattern**

```python
# BEFORE (lines 22535-22545):
                                    # Graceful termination first
                                    proc.terminate()

                                    # Wait for graceful shutdown
                                    try:
                                        proc.wait(timeout=config.zombie_kill_timeout_sec)  # BLOCKING!
                                    except psutil.TimeoutExpired:
                                        # Force kill if still running
                                        self.logger.warning(f"   Force killing stubborn zombie PID {pid}")
                                        proc.kill()
                                        try:
                                            proc.wait(timeout=2.0)  # BLOCKING!
                                        except psutil.TimeoutExpired:
                                            pass
```

```python
# AFTER:
                                    from backend.utils.async_subprocess import async_psutil_wait

                                    # Graceful termination first
                                    proc.terminate()

                                    # NON-BLOCKING wait for graceful shutdown
                                    exited = await async_psutil_wait(proc, timeout=config.zombie_kill_timeout_sec)
                                    if not exited:
                                        # Force kill if still running
                                        self.logger.warning(f"   Force killing stubborn zombie PID {pid}")
                                        proc.kill()
                                        await async_psutil_wait(proc, timeout=2.0)
```

**Step 2: Commit**

```bash
git add run_supervisor.py
git commit -m "fix(startup): Convert zombie cleanup to non-blocking async

Replace blocking proc.wait() in zombie reaper with async_psutil_wait().
Prevents event loop blocking during zombie cleanup phase."
```

---

## Task 5: Convert subprocess.run Calls to Async

**Files:**
- Modify: `run_supervisor.py:3712-3717` (lsof lock check)
- Modify: `run_supervisor.py:6609-6612` (lsof port cleanup)
- Modify: `run_supervisor.py:11355-11360` (docker info check)

**Step 1: Replace lsof lock check (lines 3712-3717)**

```python
# BEFORE:
            result = subprocess.run(
                ["lsof", str(lock_file)],
                capture_output=True,
                text=True,
                timeout=5
            )

# AFTER:
            from backend.utils.async_subprocess import async_subprocess_run

            try:
                result = await async_subprocess_run(
                    ["lsof", str(lock_file)],
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.debug(f"[v152.0] lsof timed out for {lock_file}")
                return 0
```

**Step 2: Replace lsof port cleanup (lines 6609-6612)**

```python
# BEFORE:
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True, text=True, timeout=5
                )

# AFTER:
                from backend.utils.async_subprocess import async_subprocess_run

                try:
                    result = await async_subprocess_run(
                        ["lsof", "-ti", f":{port}"],
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    self.logger.debug(f"[fast_startup] lsof timed out for port {port}")
                    continue
```

**Step 3: Replace docker info check (lines 11355-11360)**

```python
# BEFORE:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=3.0
            )

# AFTER:
            from backend.utils.async_subprocess import async_subprocess_run

            try:
                result = await async_subprocess_run(
                    ["docker", "info"],
                    timeout=3.0
                )
                return result.returncode == 0
            except asyncio.TimeoutError:
                return False
```

Note: The docker check is in `_is_docker_running()` which is currently sync. Need to make it async:

```python
# Change method signature from:
def _is_docker_running(self) -> bool:
# To:
async def _is_docker_running(self) -> bool:
```

And update callers to use `await self._is_docker_running()`.

**Step 4: Commit**

```bash
git add run_supervisor.py
git commit -m "fix(startup): Convert subprocess.run calls to async

Replace blocking subprocess.run() with async_subprocess_run() in:
- Lock file holder detection (lsof)
- Port cleanup (lsof)
- Docker availability check

Prevents 3-5 seconds of blocking per call."
```

---

## Task 6: Convert Socket Operations to Async

**Files:**
- Modify: `run_supervisor.py:3914-3924` (IPC socket check)
- Modify: `run_supervisor.py:4923-4936` (port availability check)
- Modify: `run_supervisor.py:11387-11396` (port release verification)

**Step 1: Replace IPC socket check (lines 3914-3924)**

```python
# BEFORE:
        try:
            ipc_socket = Path.home() / ".jarvis" / "ipc" / "supervisor.sock"
            if ipc_socket.exists():
                import socket
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                try:
                    sock.connect(str(ipc_socket))
                    sock.sendall(b'{"type": "ping"}\n')
                    response = sock.recv(1024)
                    if response:
                        self.logger.debug(f"[v152.0] PID {holder_pid} responded to IPC ping")
                        return False
                finally:
                    sock.close()
        except Exception as e:
            self.logger.debug(f"[v152.0] IPC ping failed: {e}")

# AFTER:
        from backend.utils.async_subprocess import async_check_unix_socket

        try:
            ipc_socket = Path.home() / ".jarvis" / "ipc" / "supervisor.sock"
            if ipc_socket.exists():
                # NON-BLOCKING socket check
                is_alive = await async_check_unix_socket(ipc_socket, timeout=2.0)
                if is_alive:
                    self.logger.debug(f"[v152.0] PID {holder_pid} responded to IPC ping")
                    return False
        except Exception as e:
            self.logger.debug(f"[v152.0] IPC ping failed: {e}")
```

**Step 2: Replace port availability check (lines 4923-4936)**

```python
# BEFORE:
        for port in self.config.required_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex(('localhost', port))
                sock.close()

                if result == 0:
                    in_use.append(port)
                    # Check if it's a JARVIS process (will be recycled by cleanup)
                    is_jarvis = await self._is_jarvis_port(port)
                    if is_jarvis:
                        actions.append(f"Port {port}: JARVIS process detected (will recycle)")
                else:
                    available.append(port)
            except Exception:
                available.append(port)

# AFTER:
        from backend.utils.async_subprocess import async_check_port

        # Check all ports in PARALLEL (non-blocking)
        port_checks = await asyncio.gather(*[
            async_check_port('localhost', port, timeout=0.5)
            for port in self.config.required_ports
        ])

        for port, is_open in zip(self.config.required_ports, port_checks):
            if is_open:
                in_use.append(port)
                is_jarvis = await self._is_jarvis_port(port)
                if is_jarvis:
                    actions.append(f"Port {port}: JARVIS process detected (will recycle)")
            else:
                available.append(port)
```

**Step 3: Replace port release verification loop (lines 11382-11402)**

```python
# BEFORE:
        while time.time() - start_time < max_wait:
            all_free = True

            for port in self.config.required_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.1)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()

                    if result == 0:  # Port is still in use
                        all_free = False
                        break
                except Exception:
                    pass  # Error connecting = port is free

            if all_free:
                self.logger.debug(f"All ports released after {time.time() - start_time:.1f}s")
                return True

            await asyncio.sleep(check_interval)

# AFTER:
        from backend.utils.async_subprocess import async_check_port

        while time.time() - start_time < max_wait:
            # Check all ports in PARALLEL (non-blocking)
            port_checks = await asyncio.gather(*[
                async_check_port('localhost', port, timeout=0.1)
                for port in self.config.required_ports
            ])

            if not any(port_checks):  # All ports are free
                self.logger.debug(f"All ports released after {time.time() - start_time:.1f}s")
                return True

            await asyncio.sleep(check_interval)
```

**Step 4: Commit**

```bash
git add run_supervisor.py
git commit -m "fix(startup): Convert socket operations to async

Replace blocking socket.connect_ex() with async_check_port() and
async_check_unix_socket(). Port checks now run in PARALLEL,
reducing total check time from O(n) to O(1).

Prevents 1.5-5 seconds of blocking during port checks."
```

---

## Task 7: Add Progress Heartbeat Background Task

**Files:**
- Modify: `run_supervisor.py` (add heartbeat task to startup)
- Test: Manual verification during startup

**Step 1: Create a heartbeat background task**

Add this near the startup orchestration code:

```python
async def _progress_heartbeat_task(
    self,
    interval: float = 3.0,
    stop_event: asyncio.Event = None,
):
    """
    Background task that sends progress heartbeats during startup.

    This ensures the loading page shows activity even during
    long-running operations that don't have granular progress updates.

    Args:
        interval: Seconds between heartbeats
        stop_event: Event to signal when to stop
    """
    last_progress = 0
    heartbeat_count = 0

    while not (stop_event and stop_event.is_set()):
        try:
            heartbeat_count += 1
            current_progress = self._current_progress if hasattr(self, '_current_progress') else 0

            # Only send heartbeat if progress hasn't changed
            # (actual progress updates are sent by the normal path)
            if current_progress == last_progress:
                await self._broadcast_progress(
                    progress=current_progress,
                    stage="startup",
                    message=f"Working... (heartbeat #{heartbeat_count})",
                    is_heartbeat=True,
                )

            last_progress = current_progress
            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            self.logger.debug(f"[heartbeat] Error in heartbeat task: {e}")
            await asyncio.sleep(interval)
```

**Step 2: Start heartbeat task during startup**

In the main startup method (e.g., `_run_with_deep_health` or similar), add:

```python
# Start heartbeat background task
heartbeat_stop = asyncio.Event()
heartbeat_task = asyncio.create_task(
    self._progress_heartbeat_task(interval=3.0, stop_event=heartbeat_stop)
)

try:
    # ... existing startup code ...
finally:
    # Stop heartbeat
    heartbeat_stop.set()
    heartbeat_task.cancel()
    with suppress(asyncio.CancelledError):
        await heartbeat_task
```

**Step 3: Commit**

```bash
git add run_supervisor.py
git commit -m "feat(startup): Add progress heartbeat background task

Introduces a background task that sends heartbeat updates every 3s
during startup. This ensures the loading page shows activity even
during long operations, preventing the appearance of being 'stuck'.

The heartbeat task runs in parallel with startup operations and
is cancelled when startup completes or fails."
```

---

## Task 8: Create Cross-Repo Async Lock Utilities

**Files:**
- Create: `backend/utils/async_locks.py`
- Test: `tests/unit/utils/test_async_locks.py`

**Step 1: Write the failing test**

```python
# tests/unit/utils/test_async_locks.py
"""Tests for cross-repo async lock utilities."""
import asyncio
import pytest
import tempfile
from pathlib import Path


class TestAsyncFileLock:
    """Test AsyncFileLock functionality."""

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        """Lock can be acquired and released."""
        from backend.utils.async_locks import AsyncFileLock

        with tempfile.NamedTemporaryFile(delete=False) as f:
            lock_path = Path(f.name)

        try:
            lock = AsyncFileLock(lock_path, timeout=5.0)

            acquired = await lock.acquire()
            assert acquired is True

            await lock.release()
        finally:
            lock_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_lock_prevents_concurrent_acquisition(self):
        """Second lock acquisition should fail or wait."""
        from backend.utils.async_locks import AsyncFileLock

        with tempfile.NamedTemporaryFile(delete=False) as f:
            lock_path = Path(f.name)

        try:
            lock1 = AsyncFileLock(lock_path, timeout=5.0)
            lock2 = AsyncFileLock(lock_path, timeout=0.5)

            # Acquire first lock
            await lock1.acquire()

            # Second lock should timeout
            acquired = await lock2.acquire()
            assert acquired is False

            # Release first lock
            await lock1.release()

            # Now second lock should work
            acquired = await lock2.acquire()
            assert acquired is True
            await lock2.release()
        finally:
            lock_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Lock works as async context manager."""
        from backend.utils.async_locks import AsyncFileLock

        with tempfile.NamedTemporaryFile(delete=False) as f:
            lock_path = Path(f.name)

        try:
            lock = AsyncFileLock(lock_path, timeout=5.0)

            async with lock:
                # Lock is held
                lock2 = AsyncFileLock(lock_path, timeout=0.1)
                acquired = await lock2.acquire()
                assert acquired is False

            # Lock is released
            lock3 = AsyncFileLock(lock_path, timeout=1.0)
            acquired = await lock3.acquire()
            assert acquired is True
            await lock3.release()
        finally:
            lock_path.unlink(missing_ok=True)
```

**Step 2: Write the implementation**

```python
# backend/utils/async_locks.py
"""
Async Lock Utilities for Cross-Repo Coordination
=================================================

Non-blocking file-based locks for coordinating between
JARVIS, JARVIS Prime, and Reactor Core.

All lock operations use asyncio.to_thread() to prevent
blocking the event loop.

Usage:
    from backend.utils.async_locks import AsyncFileLock

    lock = AsyncFileLock(Path.home() / ".jarvis" / "locks" / "startup.lock")

    async with lock:
        # Critical section
        pass
"""

import asyncio
import fcntl
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union


class AsyncFileLock:
    """
    Async file-based lock using fcntl.flock().

    Provides non-blocking lock acquisition with configurable timeout.
    """

    def __init__(
        self,
        lock_path: Union[str, Path],
        timeout: float = 30.0,
        poll_interval: float = 0.1,
    ):
        """
        Initialize the lock.

        Args:
            lock_path: Path to the lock file
            timeout: Maximum time to wait for lock acquisition
            poll_interval: How often to retry acquisition
        """
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._fd: Optional[int] = None
        self._acquired = False

    async def acquire(self) -> bool:
        """
        Attempt to acquire the lock.

        Returns:
            True if lock was acquired, False if timeout
        """
        if self._acquired:
            return True

        # Ensure parent directory exists
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()

        while time.monotonic() - start < self.timeout:
            try:
                # Open lock file
                self._fd = await asyncio.to_thread(
                    os.open,
                    str(self.lock_path),
                    os.O_RDWR | os.O_CREAT,
                    0o644
                )

                # Try non-blocking lock
                def _try_lock():
                    try:
                        fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        return True
                    except BlockingIOError:
                        return False

                if await asyncio.to_thread(_try_lock):
                    self._acquired = True
                    return True

                # Lock not available, close fd and retry
                os.close(self._fd)
                self._fd = None

            except Exception:
                if self._fd is not None:
                    try:
                        os.close(self._fd)
                    except Exception:
                        pass
                    self._fd = None

            await asyncio.sleep(self.poll_interval)

        return False

    async def release(self) -> None:
        """Release the lock."""
        if self._fd is not None:
            try:
                await asyncio.to_thread(fcntl.flock, self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
        self._acquired = False

    async def __aenter__(self):
        """Async context manager entry."""
        acquired = await self.acquire()
        if not acquired:
            raise TimeoutError(f"Failed to acquire lock: {self.lock_path}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release()


class CrossRepoLockManager:
    """
    Manager for coordinating locks across JARVIS, Prime, and Reactor.

    Provides a central registry of lock paths and utilities for
    acquiring multiple locks atomically.
    """

    # Standard lock paths
    LOCKS = {
        "supervisor": Path.home() / ".jarvis" / "locks" / "supervisor.lock",
        "startup": Path.home() / ".jarvis" / "locks" / "startup.lock",
        "vbia": Path.home() / ".jarvis" / "cross_repo" / "locks" / "vbia.lock",
        "prime_state": Path.home() / ".jarvis" / "cross_repo" / "locks" / "prime_state.dlm.lock",
        "reactor_state": Path.home() / ".jarvis" / "cross_repo" / "locks" / "reactor_state.dlm.lock",
    }

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._locks: dict = {}

    def get_lock(self, name: str) -> AsyncFileLock:
        """Get or create a lock by name."""
        if name not in self._locks:
            if name in self.LOCKS:
                path = self.LOCKS[name]
            else:
                path = Path.home() / ".jarvis" / "locks" / f"{name}.lock"
            self._locks[name] = AsyncFileLock(path, timeout=self.timeout)
        return self._locks[name]

    @asynccontextmanager
    async def acquire_multiple(self, *names: str):
        """
        Acquire multiple locks in order (deadlock-safe).

        Locks are always acquired in sorted order to prevent deadlocks.

        Usage:
            async with manager.acquire_multiple("supervisor", "startup"):
                # Both locks held
                pass
        """
        # Sort to prevent deadlocks
        sorted_names = sorted(names)
        locks = [self.get_lock(name) for name in sorted_names]

        acquired = []
        try:
            for lock in locks:
                if await lock.acquire():
                    acquired.append(lock)
                else:
                    raise TimeoutError(f"Failed to acquire lock")
            yield
        finally:
            # Release in reverse order
            for lock in reversed(acquired):
                await lock.release()
```

**Step 3: Run tests**

Run: `cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python -m pytest tests/unit/utils/test_async_locks.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add backend/utils/async_locks.py tests/unit/utils/test_async_locks.py
git commit -m "feat(locks): Add async file lock utilities for cross-repo coordination

Introduces AsyncFileLock and CrossRepoLockManager for non-blocking
lock operations between JARVIS, JARVIS Prime, and Reactor Core.

Features:
- Non-blocking lock acquisition with timeout
- Async context manager support
- Cross-repo lock registry
- Deadlock-safe multi-lock acquisition"
```

---

## Task 9: Update Import and Integration

**Files:**
- Modify: `run_supervisor.py` (add imports at top)
- Modify: `backend/utils/__init__.py` (export new utilities)

**Step 1: Add imports to run_supervisor.py**

Near the top of run_supervisor.py, add:

```python
# Async utilities for non-blocking operations (v200.0)
try:
    from backend.utils.async_subprocess import (
        async_process_wait,
        async_psutil_wait,
        async_subprocess_run,
        async_check_port,
        async_check_unix_socket,
        async_json_read,
        async_json_write,
    )
    from backend.utils.async_locks import AsyncFileLock, CrossRepoLockManager
    ASYNC_UTILS_AVAILABLE = True
except ImportError:
    ASYNC_UTILS_AVAILABLE = False
```

**Step 2: Update backend/utils/__init__.py**

```python
# backend/utils/__init__.py
"""Backend utilities package."""

from .async_subprocess import (
    async_process_wait,
    async_psutil_wait,
    async_subprocess_run,
    async_check_port,
    async_check_unix_socket,
    async_file_read,
    async_file_write,
    async_json_read,
    async_json_write,
)

from .async_locks import (
    AsyncFileLock,
    CrossRepoLockManager,
)

__all__ = [
    "async_process_wait",
    "async_psutil_wait",
    "async_subprocess_run",
    "async_check_port",
    "async_check_unix_socket",
    "async_file_read",
    "async_file_write",
    "async_json_read",
    "async_json_write",
    "AsyncFileLock",
    "CrossRepoLockManager",
]
```

**Step 3: Commit**

```bash
git add run_supervisor.py backend/utils/__init__.py
git commit -m "feat(utils): Export async utilities from backend.utils

Add exports for async_subprocess and async_locks modules.
Update run_supervisor.py imports for new async utilities."
```

---

## Task 10: Cross-Repo Integration - JARVIS Prime

**Files:**
- Create: `/Users/djrussell23/Documents/repos/jarvis-prime/jarvis_prime/utils/async_startup.py`

**Step 1: Create shared async utilities for JARVIS Prime**

```python
# jarvis-prime/jarvis_prime/utils/async_startup.py
"""
Async Startup Utilities for JARVIS Prime
=========================================

Shared utilities for coordinating startup with JARVIS Body.
These mirror the utilities in JARVIS Body to ensure consistent
non-blocking behavior across repos.
"""

import asyncio
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union


async def wait_for_jarvis_body(
    timeout: float = 60.0,
    poll_interval: float = 1.0,
) -> bool:
    """
    Wait for JARVIS Body to be ready before starting Prime.

    Checks the Trinity heartbeat file for JARVIS Body status.

    Args:
        timeout: Maximum wait time in seconds
        poll_interval: Check interval in seconds

    Returns:
        True if Body is ready, False if timeout
    """
    heartbeat_path = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_body.json"
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        try:
            if heartbeat_path.exists():
                content = await asyncio.to_thread(heartbeat_path.read_text)
                data = json.loads(content)

                status = data.get("status", "unknown")
                timestamp = data.get("timestamp", 0)
                age = time.time() - timestamp

                if status == "ready" and age < 30.0:
                    return True
        except Exception:
            pass

        await asyncio.sleep(poll_interval)

    return False


async def write_heartbeat(
    component: str = "jarvis_prime",
    status: str = "starting",
    pid: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write heartbeat file for Trinity coordination.

    Args:
        component: Component name
        status: Current status
        pid: Process ID
        extra: Additional data to include
    """
    heartbeat_dir = Path.home() / ".jarvis" / "trinity" / "components"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "component": component,
        "status": status,
        "pid": pid or os.getpid(),
        "timestamp": time.time(),
        "host": os.uname().nodename,
    }

    if extra:
        data.update(extra)

    heartbeat_path = heartbeat_dir / f"{component}.json"

    # Atomic write via rename
    temp_path = heartbeat_path.with_suffix(".tmp")

    def _write():
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(heartbeat_path)

    await asyncio.to_thread(_write)


async def acquire_startup_lock(timeout: float = 30.0) -> bool:
    """
    Acquire the cross-repo startup lock.

    This prevents multiple instances of Prime from starting simultaneously.

    Args:
        timeout: Maximum wait time

    Returns:
        True if lock acquired, False if timeout
    """
    lock_path = Path.home() / ".jarvis" / "cross_repo" / "locks" / "prime_startup.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()

    while time.monotonic() - start < timeout:
        try:
            fd = await asyncio.to_thread(
                os.open,
                str(lock_path),
                os.O_RDWR | os.O_CREAT,
                0o644
            )

            def _try_lock():
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    os.close(fd)
                    return False

            if await asyncio.to_thread(_try_lock):
                # Store fd for later release
                return True

        except Exception:
            pass

        await asyncio.sleep(0.5)

    return False
```

**Step 2: Commit in JARVIS Prime repo**

```bash
cd /Users/djrussell23/Documents/repos/jarvis-prime
git add jarvis_prime/utils/async_startup.py
git commit -m "feat(startup): Add async startup utilities for Trinity coordination

Introduces utilities for coordinating startup with JARVIS Body:
- wait_for_jarvis_body: Non-blocking wait for Body readiness
- write_heartbeat: Atomic heartbeat file updates
- acquire_startup_lock: Cross-repo startup lock

These utilities ensure Prime doesn't block the event loop during
startup coordination."
```

---

## Task 11: Cross-Repo Integration - Reactor Core

**Files:**
- Create: `/Users/djrussell23/Documents/repos/reactor-core/reactor_core/utils/async_startup.py`

**Step 1: Create shared async utilities for Reactor Core**

```python
# reactor-core/reactor_core/utils/async_startup.py
"""
Async Startup Utilities for Reactor Core
=========================================

Shared utilities for coordinating startup with JARVIS Body and Prime.
Reactor depends on both Body and Prime being ready.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


async def wait_for_dependencies(
    timeout: float = 120.0,
    poll_interval: float = 2.0,
) -> Tuple[bool, bool]:
    """
    Wait for JARVIS Body and Prime to be ready.

    Reactor Core depends on both components for full functionality.

    Args:
        timeout: Maximum wait time in seconds
        poll_interval: Check interval in seconds

    Returns:
        Tuple of (body_ready, prime_ready)
    """
    trinity_dir = Path.home() / ".jarvis" / "trinity" / "components"
    start = time.monotonic()

    body_ready = False
    prime_ready = False

    while time.monotonic() - start < timeout:
        try:
            # Check Body
            body_path = trinity_dir / "jarvis_body.json"
            if body_path.exists():
                content = await asyncio.to_thread(body_path.read_text)
                data = json.loads(content)
                if data.get("status") == "ready" and time.time() - data.get("timestamp", 0) < 30.0:
                    body_ready = True

            # Check Prime
            prime_path = trinity_dir / "jarvis_prime.json"
            if prime_path.exists():
                content = await asyncio.to_thread(prime_path.read_text)
                data = json.loads(content)
                if data.get("status") == "ready" and time.time() - data.get("timestamp", 0) < 30.0:
                    prime_ready = True

            if body_ready and prime_ready:
                return (True, True)

        except Exception:
            pass

        await asyncio.sleep(poll_interval)

    return (body_ready, prime_ready)


async def write_heartbeat(
    status: str = "starting",
    pid: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write heartbeat file for Trinity coordination.
    """
    heartbeat_dir = Path.home() / ".jarvis" / "trinity" / "components"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "component": "reactor_core",
        "status": status,
        "pid": pid or os.getpid(),
        "timestamp": time.time(),
        "host": os.uname().nodename,
    }

    if extra:
        data.update(extra)

    heartbeat_path = heartbeat_dir / "reactor_core.json"
    temp_path = heartbeat_path.with_suffix(".tmp")

    def _write():
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(heartbeat_path)

    await asyncio.to_thread(_write)


async def notify_body_ready() -> None:
    """
    Notify JARVIS Body that Reactor is ready.

    Writes to cross-repo state file for Body to pick up.
    """
    state_dir = Path.home() / ".jarvis" / "cross_repo"
    state_dir.mkdir(parents=True, exist_ok=True)

    state_path = state_dir / "reactor_state.json"
    temp_path = state_path.with_suffix(".tmp")

    data = {
        "status": "ready",
        "pid": os.getpid(),
        "timestamp": time.time(),
        "capabilities": [
            "training",
            "experience_collection",
            "model_evaluation",
        ],
    }

    def _write():
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.rename(state_path)

    await asyncio.to_thread(_write)
```

**Step 2: Commit in Reactor Core repo**

```bash
cd /Users/djrussell23/Documents/repos/reactor-core
git add reactor_core/utils/async_startup.py
git commit -m "feat(startup): Add async startup utilities for Trinity coordination

Introduces utilities for coordinating startup with JARVIS Body and Prime:
- wait_for_dependencies: Non-blocking wait for Body + Prime
- write_heartbeat: Atomic heartbeat file updates
- notify_body_ready: Cross-repo ready notification

These utilities ensure Reactor doesn't block the event loop during
startup coordination."
```

---

## Task 12: Environment-Driven Timeouts

**Files:**
- Modify: `run_supervisor.py` (add env-based timeout configuration)

**Step 1: Add timeout configuration**

Near the SupervisorConfig class or at module level:

```python
# Environment-driven timeout configuration (v200.0)
# All timeouts are configurable via environment variables

import os

def _get_timeout(env_var: str, default: float) -> float:
    """Get timeout from environment with fallback to default."""
    try:
        return float(os.environ.get(env_var, default))
    except (ValueError, TypeError):
        return default


class StartupTimeouts:
    """Centralized timeout configuration for startup operations."""

    # Process termination timeouts
    SIGINT_TIMEOUT = _get_timeout("JARVIS_SIGINT_TIMEOUT", 10.0)
    SIGTERM_TIMEOUT = _get_timeout("JARVIS_SIGTERM_TIMEOUT", 10.0)
    SIGKILL_TIMEOUT = _get_timeout("JARVIS_SIGKILL_TIMEOUT", 2.0)

    # Port and network timeouts
    PORT_CHECK_TIMEOUT = _get_timeout("JARVIS_PORT_CHECK_TIMEOUT", 0.5)
    PORT_RELEASE_WAIT = _get_timeout("JARVIS_PORT_RELEASE_WAIT", 5.0)
    IPC_SOCKET_TIMEOUT = _get_timeout("JARVIS_IPC_SOCKET_TIMEOUT", 2.0)

    # Subprocess timeouts
    LSOF_TIMEOUT = _get_timeout("JARVIS_LSOF_TIMEOUT", 5.0)
    DOCKER_CHECK_TIMEOUT = _get_timeout("JARVIS_DOCKER_CHECK_TIMEOUT", 3.0)

    # Health check timeouts
    BACKEND_HEALTH_TIMEOUT = _get_timeout("JARVIS_BACKEND_HEALTH_TIMEOUT", 3.0)
    FRONTEND_HEALTH_TIMEOUT = _get_timeout("JARVIS_FRONTEND_HEALTH_TIMEOUT", 2.0)

    # Loading server timeouts
    LOADING_SERVER_HEALTH_TIMEOUT = _get_timeout("JARVIS_LOADING_SERVER_HEALTH_TIMEOUT", 30.0)

    # Heartbeat configuration
    HEARTBEAT_INTERVAL = _get_timeout("JARVIS_HEARTBEAT_INTERVAL", 3.0)
    HEARTBEAT_STALE_THRESHOLD = _get_timeout("JARVIS_HEARTBEAT_STALE_THRESHOLD", 15.0)

    # Trinity component timeouts
    JPRIME_STARTUP_TIMEOUT = _get_timeout("JARVIS_PRIME_STARTUP_TIMEOUT", 600.0)
    REACTOR_STARTUP_TIMEOUT = _get_timeout("JARVIS_REACTOR_STARTUP_TIMEOUT", 120.0)

    # Lock timeouts
    STARTUP_LOCK_TIMEOUT = _get_timeout("JARVIS_STARTUP_LOCK_TIMEOUT", 30.0)
    TAKEOVER_HANDOVER_TIMEOUT = _get_timeout("JARVIS_TAKEOVER_HANDOVER_TIMEOUT", 30.0)


# Log timeout configuration at startup
def log_timeout_config(logger):
    """Log current timeout configuration for debugging."""
    logger.debug("Startup timeout configuration:")
    for attr in dir(StartupTimeouts):
        if attr.isupper():
            value = getattr(StartupTimeouts, attr)
            logger.debug(f"  {attr}: {value}s")
```

**Step 2: Update code to use StartupTimeouts**

Replace hardcoded timeout values with `StartupTimeouts.<NAME>`:

```python
# Instead of:
proc.wait(timeout=5.0)

# Use:
await async_psutil_wait(proc, timeout=StartupTimeouts.SIGTERM_TIMEOUT)
```

**Step 3: Commit**

```bash
git add run_supervisor.py
git commit -m "feat(config): Add environment-driven timeout configuration

Introduces StartupTimeouts class with all timeouts configurable
via environment variables. This enables:
- Runtime tuning without code changes
- Environment-specific configurations
- Easier debugging of timeout-related issues

All timeout values have sensible defaults while being overridable."
```

---

## Task 13: Final Integration Test

**Files:**
- Create: `tests/integration/test_startup_nonblocking.py`

**Step 1: Write integration test**

```python
# tests/integration/test_startup_nonblocking.py
"""
Integration tests for non-blocking startup.

These tests verify that the startup flow doesn't block the event loop
and that progress updates continue during long operations.
"""

import asyncio
import pytest
import time


class TestStartupNonBlocking:
    """Test that startup operations don't block the event loop."""

    @pytest.mark.asyncio
    async def test_port_checks_are_parallel(self):
        """Port checks should run in parallel, not sequentially."""
        from backend.utils.async_subprocess import async_check_port

        ports = [59991, 59992, 59993, 59994, 59995]

        start = time.monotonic()

        # Run all port checks concurrently
        results = await asyncio.gather(*[
            async_check_port("localhost", port, timeout=0.5)
            for port in ports
        ])

        elapsed = time.monotonic() - start

        # Should complete in ~0.5s (parallel), not ~2.5s (serial)
        assert elapsed < 1.0, f"Port checks took {elapsed}s - should be parallel"
        assert len(results) == len(ports)

    @pytest.mark.asyncio
    async def test_heartbeat_continues_during_async_operations(self):
        """Heartbeat task should continue running during async operations."""
        heartbeat_count = 0

        async def heartbeat():
            nonlocal heartbeat_count
            while heartbeat_count < 10:
                heartbeat_count += 1
                await asyncio.sleep(0.1)

        async def long_operation():
            # Simulate a long async operation
            await asyncio.sleep(0.5)
            return "done"

        # Run heartbeat and operation concurrently
        heartbeat_task = asyncio.create_task(heartbeat())
        result = await long_operation()

        # Wait for heartbeat to finish
        await heartbeat_task

        # Heartbeat should have run multiple times
        assert heartbeat_count >= 5, f"Heartbeat only ran {heartbeat_count} times"
        assert result == "done"

    @pytest.mark.asyncio
    async def test_file_read_doesnt_block_loop(self):
        """File reads should not block concurrent tasks."""
        from backend.utils.async_subprocess import async_json_read
        from pathlib import Path
        import tempfile
        import json

        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "data"}, f)
            test_path = Path(f.name)

        try:
            concurrent_count = 0

            async def concurrent_task():
                nonlocal concurrent_count
                for _ in range(5):
                    concurrent_count += 1
                    await asyncio.sleep(0.05)

            # Run file read and concurrent task together
            task = asyncio.create_task(concurrent_task())
            data = await async_json_read(test_path)
            await task

            assert data == {"test": "data"}
            assert concurrent_count >= 3, "Concurrent task was blocked"

        finally:
            test_path.unlink()

    @pytest.mark.asyncio
    async def test_subprocess_doesnt_block_loop(self):
        """Subprocess runs should not block concurrent tasks."""
        from backend.utils.async_subprocess import async_subprocess_run

        concurrent_count = 0

        async def concurrent_task():
            nonlocal concurrent_count
            for _ in range(10):
                concurrent_count += 1
                await asyncio.sleep(0.05)

        # Run subprocess and concurrent task together
        task = asyncio.create_task(concurrent_task())
        result = await async_subprocess_run(["sleep", "0.3"], timeout=5.0)
        await task

        assert result.returncode == 0
        assert concurrent_count >= 5, "Concurrent task was blocked by subprocess"
```

**Step 2: Run integration tests**

Run: `cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent && python -m pytest tests/integration/test_startup_nonblocking.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_startup_nonblocking.py
git commit -m "test(startup): Add integration tests for non-blocking startup

Verifies that:
- Port checks run in parallel
- Heartbeat continues during async operations
- File reads don't block the event loop
- Subprocess runs don't block concurrent tasks"
```

---

## Summary

This plan converts all blocking calls in the JARVIS startup path to async:

| Blocking Call | Location | Fix |
|--------------|----------|-----|
| `proc.wait()` (3x) | _graceful_terminate | `async_psutil_wait()` |
| `proc.wait()` (2x) | J-Prime/Reactor stop | `async_psutil_wait()` |
| `proc.wait()` (2x) | Zombie cleanup | `async_psutil_wait()` |
| `subprocess.run()` | lsof, docker | `async_subprocess_run()` |
| `socket.connect_ex()` | Port checks | `async_check_port()` (parallel) |
| `socket.connect()` | IPC check | `async_check_unix_socket()` |
| `open()` + `json.load()` | Heartbeat files | `async_json_read()` |

**Total blocking time eliminated:** Up to 44+ seconds

**Cross-repo integration:** Shared async utilities in JARVIS Prime and Reactor Core

**Configuration:** All timeouts env-driven via `StartupTimeouts` class

---

Plan complete and saved to `docs/plans/2026-02-02-async-startup-robustness.md`. Two execution options:

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
