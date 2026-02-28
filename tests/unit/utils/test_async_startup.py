"""
Tests for backend.utils.async_startup module.

These tests verify that:
1. Blocking operations (process wait, subprocess, socket checks, file I/O) don't block the event loop
2. Operations run in parallel when independent
3. The dedicated startup executor is bounded and properly managed

Following 35-point checklist items: 1-2 (event loop non-blocking), 9 (bounded executor), 32 (startup executor)
"""

import asyncio
import json
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

# Import after the module is created
from backend.utils.async_startup import (
    async_process_wait,
    async_psutil_wait,
    async_subprocess_run,
    async_check_port,
    async_check_unix_socket,
    async_file_read,
    async_file_write,
    async_json_read,
    async_json_write,
    shutdown_startup_executor,
    SubprocessResult,
    _STARTUP_EXECUTOR,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_file(tmp_path: Path) -> Path:
    """Create a temporary file with some content."""
    file = tmp_path / "test_file.txt"
    file.write_text("Hello, World!\nLine 2\n")
    return file


@pytest.fixture
def temp_json_file(tmp_path: Path) -> Path:
    """Create a temporary JSON file."""
    file = tmp_path / "test_data.json"
    data = {"name": "Ironcliw", "version": 1, "features": ["voice", "vision"]}
    file.write_text(json.dumps(data))
    return file


@pytest.fixture
def available_port() -> int:
    """Find an available port for testing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def temp_unix_socket() -> Path:
    """Create a path for a temporary Unix socket (short path to avoid AF_UNIX limit)."""
    # Unix sockets have a max path length (~104 chars on macOS)
    # Use /tmp directly with a short name
    short_id = uuid.uuid4().hex[:8]
    return Path(f"/tmp/jarvis_test_{short_id}.sock")


# =============================================================================
# Test: Process Wait Doesn't Block Concurrent Tasks
# =============================================================================


class TestAsyncProcessWait:
    """Tests for async_process_wait - ensures process waits don't block event loop."""

    @pytest.mark.asyncio
    async def test_process_wait_non_blocking(self) -> None:
        """
        Verify that waiting for a process doesn't block other async tasks.

        We start a slow process and simultaneously run a counter task.
        If the wait was blocking, the counter would not increment during the wait.
        """
        counter = {"value": 0}

        async def increment_counter():
            """Increment counter every 50ms to prove event loop is running."""
            for _ in range(10):
                counter["value"] += 1
                await asyncio.sleep(0.05)

        # Start a process that sleeps for 300ms
        process = await asyncio.create_subprocess_exec(
            "sleep", "0.3",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Run both tasks concurrently
        wait_task = asyncio.create_task(async_process_wait(process.pid, timeout=5.0))
        counter_task = asyncio.create_task(increment_counter())

        start = time.monotonic()
        results = await asyncio.gather(wait_task, counter_task)
        elapsed = time.monotonic() - start

        # The process should have completed (returned True)
        assert results[0] is True

        # The counter should have incremented multiple times during the wait
        # If blocking, counter would be 0 or 1
        assert counter["value"] >= 5, f"Counter only reached {counter['value']}, likely blocked"

        # Total time should be ~500ms (the longer task), not 300ms + 500ms (sequential = 800ms+)
        # Use generous threshold (1.0s) to avoid flaky CI failures due to timing variance
        assert elapsed < 1.0, f"Took {elapsed}s, looks like sequential execution"

    @pytest.mark.asyncio
    async def test_process_wait_timeout(self) -> None:
        """Test that process wait respects timeout."""
        # Start a long-running process
        process = await asyncio.create_subprocess_exec(
            "sleep", "10",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        start = time.monotonic()
        result = await async_process_wait(process.pid, timeout=0.2)
        elapsed = time.monotonic() - start

        # Should return False (timed out) quickly
        assert result is False
        # Use generous threshold to avoid flaky CI failures
        assert elapsed < 1.0, f"Timeout took {elapsed}s, expected ~0.2s"

        # Clean up the process
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass

    @pytest.mark.asyncio
    async def test_process_wait_nonexistent_pid(self) -> None:
        """Test waiting for a non-existent PID returns True immediately (process already gone)."""
        # Use a PID that almost certainly doesn't exist
        nonexistent_pid = 999999

        start = time.monotonic()
        result = await async_process_wait(nonexistent_pid, timeout=5.0)
        elapsed = time.monotonic() - start

        # Should return True immediately - the process is "done" (doesn't exist)
        # This is the correct behavior: if PID doesn't exist, it has already exited
        assert result is True
        # Use generous threshold to avoid flaky CI failures
        assert elapsed < 1.0, f"Non-existent PID check took {elapsed}s"


class TestAsyncPsutilWait:
    """Tests for async_psutil_wait - psutil.Process.wait() wrapper."""

    @pytest.mark.asyncio
    async def test_psutil_wait_non_blocking(self) -> None:
        """
        Verify psutil process wait doesn't block event loop.
        """
        counter = {"value": 0}

        async def increment_counter():
            for _ in range(10):
                counter["value"] += 1
                await asyncio.sleep(0.05)

        # Start a short-lived process
        process = await asyncio.create_subprocess_exec(
            "sleep", "0.3",
            stdout=asyncio.subprocess.DEVNULL,
        )

        import psutil
        psutil_proc = psutil.Process(process.pid)

        wait_task = asyncio.create_task(async_psutil_wait(psutil_proc, timeout=5.0))
        counter_task = asyncio.create_task(increment_counter())

        await asyncio.gather(wait_task, counter_task)

        # Counter should have incremented during the wait
        assert counter["value"] >= 5, f"Counter only reached {counter['value']}, likely blocked"


# =============================================================================
# Test: Port Checks Run in Parallel
# =============================================================================


class TestAsyncPortChecks:
    """Tests for async_check_port and parallel execution."""

    @pytest.mark.asyncio
    async def test_port_check_listening(self, available_port: int) -> None:
        """Test port check returns True when something is listening."""
        # Start a server on the available port
        server = await asyncio.start_server(
            lambda r, w: None,  # Dummy handler
            "127.0.0.1",
            available_port,
        )

        try:
            result = await async_check_port("127.0.0.1", available_port, timeout=1.0)
            assert result is True
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_port_check_not_listening(self) -> None:
        """Test port check returns False when nothing is listening."""
        # Use a port that's very unlikely to be in use (high ephemeral range)
        result = await async_check_port("127.0.0.1", 59999, timeout=0.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_port_checks_parallel_execution(self) -> None:
        """
        Verify multiple port checks run in parallel, not sequentially.

        If 5 checks each take 500ms timeout and run sequentially, total = 2.5s
        If parallel, total should be ~500ms
        """
        # Use ports that won't be listening (will all timeout)
        ports = [59991, 59992, 59993, 59994, 59995]
        timeout = 0.5

        start = time.monotonic()

        # Run all checks in parallel
        tasks = [async_check_port("127.0.0.1", port, timeout=timeout) for port in ports]
        results = await asyncio.gather(*tasks)

        elapsed = time.monotonic() - start

        # All should return False (not listening)
        assert all(r is False for r in results)

        # Total time should be ~500ms (parallel), not ~2500ms (sequential)
        # Use generous threshold (1.5s) to avoid flaky CI failures
        assert elapsed < 1.5, f"Parallel checks took {elapsed}s, expected ~{timeout}s"


class TestAsyncUnixSocketCheck:
    """Tests for async_check_unix_socket."""

    @pytest.mark.asyncio
    async def test_unix_socket_listening(self, temp_unix_socket: Path) -> None:
        """Test Unix socket check when server is listening."""
        # Create a Unix socket server
        server = await asyncio.start_unix_server(
            lambda r, w: None,
            path=str(temp_unix_socket),
        )

        try:
            result = await async_check_unix_socket(str(temp_unix_socket), timeout=1.0)
            assert result is True
        finally:
            server.close()
            await server.wait_closed()
            # Clean up socket file
            if temp_unix_socket.exists():
                temp_unix_socket.unlink()

    @pytest.mark.asyncio
    async def test_unix_socket_not_listening(self, temp_unix_socket: Path) -> None:
        """Test Unix socket check when nothing is listening."""
        # Socket file doesn't exist
        result = await async_check_unix_socket(str(temp_unix_socket), timeout=0.5)
        assert result is False


# =============================================================================
# Test: File Reads Don't Block Event Loop
# =============================================================================


class TestAsyncFileIO:
    """Tests for async file I/O operations."""

    @pytest.mark.asyncio
    async def test_file_read_non_blocking(self, temp_file: Path) -> None:
        """
        Verify file read doesn't block the event loop.
        """
        counter = {"value": 0}

        async def increment_counter():
            for _ in range(5):
                counter["value"] += 1
                await asyncio.sleep(0.01)

        # Run file read and counter increment concurrently
        read_task = asyncio.create_task(async_file_read(str(temp_file)))
        counter_task = asyncio.create_task(increment_counter())

        content, _ = await asyncio.gather(read_task, counter_task)

        # Verify file was read correctly
        assert "Hello, World!" in content

        # Counter should have incremented (not blocked)
        assert counter["value"] >= 3

    @pytest.mark.asyncio
    async def test_file_read_content(self, temp_file: Path) -> None:
        """Test file read returns correct content."""
        content = await async_file_read(str(temp_file))
        assert content == "Hello, World!\nLine 2\n"

    @pytest.mark.asyncio
    async def test_file_read_encoding(self, tmp_path: Path) -> None:
        """Test file read with different encoding."""
        file = tmp_path / "utf8.txt"
        file.write_text("Hello unicode!", encoding="utf-8")

        content = await async_file_read(str(file), encoding="utf-8")
        assert content == "Hello unicode!"

    @pytest.mark.asyncio
    async def test_file_read_not_found(self) -> None:
        """Test file read raises appropriate error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            await async_file_read("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_file_write_and_read(self, tmp_path: Path) -> None:
        """Test writing and reading a file."""
        file = tmp_path / "written.txt"
        test_content = "Written by async_file_write"

        await async_file_write(str(file), test_content)

        # Verify with sync read
        assert file.read_text() == test_content

        # Verify with async read
        content = await async_file_read(str(file))
        assert content == test_content


class TestAsyncJsonIO:
    """Tests for async JSON I/O operations."""

    @pytest.mark.asyncio
    async def test_json_read(self, temp_json_file: Path) -> None:
        """Test JSON file read."""
        data = await async_json_read(str(temp_json_file))

        assert data["name"] == "Ironcliw"
        assert data["version"] == 1
        assert "voice" in data["features"]

    @pytest.mark.asyncio
    async def test_json_read_not_found(self) -> None:
        """Test JSON read raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            await async_json_read("/nonexistent/data.json")

    @pytest.mark.asyncio
    async def test_json_write_and_read(self, tmp_path: Path) -> None:
        """Test writing and reading JSON."""
        file = tmp_path / "output.json"
        test_data = {"key": "value", "numbers": [1, 2, 3]}

        await async_json_write(str(file), test_data)

        # Verify with async read
        loaded = await async_json_read(str(file))
        assert loaded == test_data

    @pytest.mark.asyncio
    async def test_json_read_invalid(self, tmp_path: Path) -> None:
        """Test JSON read raises error for invalid JSON."""
        file = tmp_path / "invalid.json"
        file.write_text("not valid json {")

        with pytest.raises(json.JSONDecodeError):
            await async_json_read(str(file))


# =============================================================================
# Test: Subprocess Execution
# =============================================================================


class TestAsyncSubprocessRun:
    """Tests for async_subprocess_run."""

    @pytest.mark.asyncio
    async def test_subprocess_success(self) -> None:
        """Test successful subprocess execution."""
        result = await async_subprocess_run(["echo", "hello"], timeout=5.0)

        assert isinstance(result, SubprocessResult)
        assert result.returncode == 0
        assert result.success is True
        assert b"hello" in result.stdout

    @pytest.mark.asyncio
    async def test_subprocess_failure(self) -> None:
        """Test subprocess that exits with non-zero code."""
        result = await async_subprocess_run(["false"], timeout=5.0)

        assert result.returncode != 0
        assert result.success is False

    @pytest.mark.asyncio
    async def test_subprocess_timeout(self) -> None:
        """Test subprocess timeout."""
        start = time.monotonic()
        result = await async_subprocess_run(["sleep", "10"], timeout=0.3)
        elapsed = time.monotonic() - start

        assert result.timed_out is True
        assert result.success is False
        # Use generous threshold to avoid flaky CI failures
        assert elapsed < 1.0, f"Timeout took {elapsed}s"

    @pytest.mark.asyncio
    async def test_subprocess_with_cwd(self, tmp_path: Path) -> None:
        """Test subprocess with custom working directory."""
        result = await async_subprocess_run(["pwd"], timeout=5.0, cwd=str(tmp_path))

        assert result.returncode == 0
        # pwd output should contain the temp path
        assert str(tmp_path) in result.stdout.decode().strip()

    @pytest.mark.asyncio
    async def test_subprocess_with_env(self) -> None:
        """Test subprocess with custom environment."""
        custom_env = os.environ.copy()
        custom_env["TEST_VAR"] = "test_value_12345"

        result = await async_subprocess_run(
            ["env"],
            timeout=5.0,
            env=custom_env,
        )

        assert result.returncode == 0
        assert b"TEST_VAR=test_value_12345" in result.stdout

    @pytest.mark.asyncio
    async def test_subprocess_non_blocking(self) -> None:
        """Verify subprocess execution doesn't block event loop."""
        counter = {"value": 0}

        async def increment_counter():
            for _ in range(10):
                counter["value"] += 1
                await asyncio.sleep(0.05)

        subprocess_task = asyncio.create_task(
            async_subprocess_run(["sleep", "0.3"], timeout=5.0)
        )
        counter_task = asyncio.create_task(increment_counter())

        await asyncio.gather(subprocess_task, counter_task)

        # Counter should have incremented during subprocess wait
        assert counter["value"] >= 5, f"Counter only reached {counter['value']}"


# =============================================================================
# Test: Executor Management
# =============================================================================


class TestExecutorManagement:
    """Tests for startup executor management."""

    def test_executor_is_bounded(self) -> None:
        """Verify the startup executor has bounded workers."""
        assert _STARTUP_EXECUTOR._max_workers == 4

    def test_executor_thread_naming(self) -> None:
        """Verify executor threads have proper naming prefix."""
        # Submit a task that returns the thread name
        import threading

        future = _STARTUP_EXECUTOR.submit(lambda: threading.current_thread().name)
        thread_name = future.result(timeout=5.0)

        assert thread_name.startswith("startup_async_")

    @pytest.mark.asyncio
    async def test_multiple_operations_use_executor(self) -> None:
        """Verify multiple operations can run concurrently in executor."""
        # Run multiple blocking file reads concurrently
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create multiple files
            files = []
            for i in range(4):
                f = Path(tmp_dir) / f"file_{i}.txt"
                f.write_text(f"Content {i}")
                files.append(f)

            start = time.monotonic()

            # Read all files concurrently
            tasks = [async_file_read(str(f)) for f in files]
            results = await asyncio.gather(*tasks)

            elapsed = time.monotonic() - start

            # Verify all files were read
            assert len(results) == 4
            for i, content in enumerate(results):
                assert f"Content {i}" in content

            # Should complete quickly (parallel), not sequentially
            # Use generous threshold to avoid flaky CI failures
            assert elapsed < 1.0


# =============================================================================
# Test: Cleanup
# =============================================================================


class TestShutdownExecutor:
    """Tests for executor shutdown."""

    def test_shutdown_executor_callable(self) -> None:
        """Verify shutdown_startup_executor is callable without error."""
        # This should not raise even if called multiple times
        # (In production, only call once at shutdown)
        # We don't actually call it here to avoid affecting other tests
        assert callable(shutdown_startup_executor)
