"""
Tests for backend.utils.async_io module.

This module tests the async-safe wrapper utilities for blocking I/O operations.
These utilities ensure the event loop is never blocked during startup or other
async operations.

Test coverage:
- run_sync: Generic async wrapper for blocking functions
- path_exists: Async-safe file existence check
- read_file: Async-safe file reading
- run_subprocess: Async subprocess execution with timeout support

Following TDD: Tests written first, then implementation.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest


# =============================================================================
# Test: run_sync - Generic async wrapper for blocking functions
# =============================================================================


class TestRunSync:
    """Tests for run_sync - async wrapper for blocking operations."""

    @pytest.mark.asyncio
    async def test_run_sync_executes_in_executor(self) -> None:
        """
        Verify that run_sync executes blocking function without blocking the event loop.

        We run a blocking sleep alongside an async counter. If run_sync blocked
        the event loop, the counter wouldn't increment during the sleep.
        """
        from backend.utils.async_io import run_sync

        counter = {"value": 0}

        async def increment_counter() -> None:
            """Increment counter every 50ms to prove event loop is running."""
            for _ in range(10):
                counter["value"] += 1
                await asyncio.sleep(0.05)

        def blocking_sleep(duration: float) -> str:
            """A blocking function that sleeps."""
            time.sleep(duration)
            return "completed"

        # Run both tasks concurrently
        sleep_task = asyncio.create_task(run_sync(blocking_sleep, 0.3))
        counter_task = asyncio.create_task(increment_counter())

        start = time.monotonic()
        result, _ = await asyncio.gather(sleep_task, counter_task)
        elapsed = time.monotonic() - start

        # The blocking function should return its result
        assert result == "completed"

        # The counter should have incremented multiple times during the sleep
        # If blocking, counter would be 0 or 1
        assert counter["value"] >= 5, f"Counter only reached {counter['value']}, likely blocked"

        # Total time should be ~500ms (the longer task), not additive
        assert elapsed < 1.0, f"Took {elapsed}s, looks like sequential execution"

    @pytest.mark.asyncio
    async def test_run_sync_passes_args_and_kwargs(self) -> None:
        """Verify that run_sync correctly passes positional and keyword arguments."""
        from backend.utils.async_io import run_sync

        def func_with_args(a: int, b: int, *, multiplier: int = 1) -> int:
            """Function with both positional and keyword arguments."""
            return (a + b) * multiplier

        # Test positional args
        result = await run_sync(func_with_args, 2, 3)
        assert result == 5

        # Test keyword args
        result = await run_sync(func_with_args, 2, 3, multiplier=10)
        assert result == 50

    @pytest.mark.asyncio
    async def test_run_sync_propagates_exceptions(self) -> None:
        """Verify that exceptions from the blocking function are propagated."""
        from backend.utils.async_io import run_sync

        def raising_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await run_sync(raising_function)

    @pytest.mark.asyncio
    async def test_run_sync_returns_correct_type(self) -> None:
        """Verify run_sync preserves the return type."""
        from backend.utils.async_io import run_sync

        def return_dict() -> dict[str, int]:
            return {"a": 1, "b": 2}

        def return_list() -> list[str]:
            return ["x", "y", "z"]

        def return_none() -> None:
            pass

        # Test various return types
        dict_result = await run_sync(return_dict)
        assert dict_result == {"a": 1, "b": 2}

        list_result = await run_sync(return_list)
        assert list_result == ["x", "y", "z"]

        none_result = await run_sync(return_none)
        assert none_result is None


# =============================================================================
# Test: path_exists - Async-safe file existence check
# =============================================================================


class TestPathExists:
    """Tests for path_exists - async file existence check."""

    @pytest.mark.asyncio
    async def test_path_exists_returns_bool(self, tmp_path: Path) -> None:
        """Verify path_exists returns True for existing file, False for non-existent."""
        from backend.utils.async_io import path_exists

        # Create a test file
        existing_file = tmp_path / "existing.txt"
        existing_file.write_text("test content")

        # Test existing file
        assert await path_exists(str(existing_file)) is True
        assert await path_exists(existing_file) is True  # Also test with Path object

        # Test non-existent file
        nonexistent = tmp_path / "nonexistent.txt"
        assert await path_exists(str(nonexistent)) is False
        assert await path_exists(nonexistent) is False

    @pytest.mark.asyncio
    async def test_path_exists_directory(self, tmp_path: Path) -> None:
        """Verify path_exists works for directories."""
        from backend.utils.async_io import path_exists

        # Test existing directory
        assert await path_exists(str(tmp_path)) is True
        assert await path_exists(tmp_path) is True

        # Test non-existent directory
        nonexistent_dir = tmp_path / "nonexistent_dir"
        assert await path_exists(str(nonexistent_dir)) is False

    @pytest.mark.asyncio
    async def test_path_exists_non_blocking(self, tmp_path: Path) -> None:
        """Verify path_exists doesn't block the event loop."""
        from backend.utils.async_io import path_exists

        counter = {"value": 0}

        async def increment_counter() -> None:
            for _ in range(5):
                counter["value"] += 1
                await asyncio.sleep(0.01)

        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Run concurrently
        exists_task = asyncio.create_task(path_exists(str(test_file)))
        counter_task = asyncio.create_task(increment_counter())

        result, _ = await asyncio.gather(exists_task, counter_task)

        assert result is True
        assert counter["value"] >= 3  # Counter should have incremented


# =============================================================================
# Test: read_file - Async-safe file reading
# =============================================================================


class TestReadFile:
    """Tests for read_file - async file reading."""

    @pytest.mark.asyncio
    async def test_read_file_returns_contents(self, tmp_path: Path) -> None:
        """Verify file contents are read correctly."""
        from backend.utils.async_io import read_file

        # Create a test file with specific content
        test_content = "Hello, World!\nLine 2\nLine 3"
        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)

        # Read the file
        result = await read_file(str(test_file))
        assert result == test_content

        # Also test with Path object
        result = await read_file(test_file)
        assert result == test_content

    @pytest.mark.asyncio
    async def test_read_file_with_encoding(self, tmp_path: Path) -> None:
        """Verify read_file respects the encoding parameter."""
        from backend.utils.async_io import read_file

        # Create a file with UTF-8 content
        test_file = tmp_path / "unicode.txt"
        test_file.write_text("Hello!")

        # Read with default encoding
        result = await read_file(str(test_file))
        assert result == "Hello!"

        # Read with explicit encoding
        result = await read_file(str(test_file), encoding="utf-8")
        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_read_file_not_found(self) -> None:
        """Verify FileNotFoundError is raised for non-existent file."""
        from backend.utils.async_io import read_file

        with pytest.raises(FileNotFoundError):
            await read_file("/nonexistent/path/file.txt")

    @pytest.mark.asyncio
    async def test_read_file_non_blocking(self, tmp_path: Path) -> None:
        """Verify read_file doesn't block the event loop."""
        from backend.utils.async_io import read_file

        counter = {"value": 0}

        async def increment_counter() -> None:
            for _ in range(5):
                counter["value"] += 1
                await asyncio.sleep(0.01)

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Run concurrently
        read_task = asyncio.create_task(read_file(str(test_file)))
        counter_task = asyncio.create_task(increment_counter())

        content, _ = await asyncio.gather(read_task, counter_task)

        assert content == "test content"
        assert counter["value"] >= 3


# =============================================================================
# Test: run_subprocess - Async subprocess execution
# =============================================================================


class TestRunSubprocess:
    """Tests for run_subprocess - async subprocess execution."""

    @pytest.mark.asyncio
    async def test_run_subprocess_returns_completed_process(self) -> None:
        """Verify returncode, stdout, stderr are captured correctly."""
        from backend.utils.async_io import run_subprocess

        result = await run_subprocess(["echo", "hello"])

        # Check return type
        assert isinstance(result, subprocess.CompletedProcess)

        # Check returncode
        assert result.returncode == 0

        # Check stdout (should contain "hello")
        assert b"hello" in result.stdout

        # Check stderr (should be empty or bytes)
        assert isinstance(result.stderr, bytes)

    @pytest.mark.asyncio
    async def test_run_subprocess_captures_stderr(self) -> None:
        """Verify stderr is captured correctly."""
        from backend.utils.async_io import run_subprocess

        # Use bash to write to stderr
        result = await run_subprocess(["bash", "-c", "echo error >&2"])

        assert result.returncode == 0
        assert b"error" in result.stderr

    @pytest.mark.asyncio
    async def test_run_subprocess_returns_exit_code(self) -> None:
        """Verify non-zero exit codes are captured."""
        from backend.utils.async_io import run_subprocess

        # 'false' command exits with code 1
        result = await run_subprocess(["false"])
        assert result.returncode == 1

        # Use bash to return specific exit code
        result = await run_subprocess(["bash", "-c", "exit 42"])
        assert result.returncode == 42

    @pytest.mark.asyncio
    async def test_run_subprocess_timeout(self) -> None:
        """Verify TimeoutError is raised when command exceeds timeout."""
        from backend.utils.async_io import run_subprocess

        start = time.monotonic()

        with pytest.raises(asyncio.TimeoutError):
            await run_subprocess(["sleep", "10"], timeout=0.2)

        elapsed = time.monotonic() - start

        # Should have timed out quickly
        assert elapsed < 1.0, f"Timeout took {elapsed}s, expected ~0.2s"

    @pytest.mark.asyncio
    async def test_run_subprocess_no_timeout(self) -> None:
        """Verify command completes when timeout is None."""
        from backend.utils.async_io import run_subprocess

        result = await run_subprocess(["echo", "test"], timeout=None)
        assert result.returncode == 0
        assert b"test" in result.stdout

    @pytest.mark.asyncio
    async def test_run_subprocess_with_kwargs(self, tmp_path: Path) -> None:
        """Verify additional kwargs are passed to subprocess."""
        from backend.utils.async_io import run_subprocess

        # Test cwd kwarg
        result = await run_subprocess(["pwd"], cwd=str(tmp_path))
        assert result.returncode == 0
        # The output should contain the temp path
        assert str(tmp_path) in result.stdout.decode().strip()

    @pytest.mark.asyncio
    async def test_run_subprocess_with_env(self) -> None:
        """Verify environment variables are passed correctly."""
        from backend.utils.async_io import run_subprocess

        custom_env = os.environ.copy()
        custom_env["TEST_VAR"] = "test_value_xyz"

        result = await run_subprocess(["env"], env=custom_env)
        assert result.returncode == 0
        assert b"TEST_VAR=test_value_xyz" in result.stdout

    @pytest.mark.asyncio
    async def test_run_subprocess_non_blocking(self) -> None:
        """Verify run_subprocess doesn't block the event loop."""
        from backend.utils.async_io import run_subprocess

        counter = {"value": 0}

        async def increment_counter() -> None:
            for _ in range(10):
                counter["value"] += 1
                await asyncio.sleep(0.05)

        # Run a subprocess that takes ~300ms
        subprocess_task = asyncio.create_task(run_subprocess(["sleep", "0.3"]))
        counter_task = asyncio.create_task(increment_counter())

        start = time.monotonic()
        result, _ = await asyncio.gather(subprocess_task, counter_task)
        elapsed = time.monotonic() - start

        # Subprocess should complete successfully
        assert result.returncode == 0

        # Counter should have incremented multiple times
        assert counter["value"] >= 5, f"Counter only reached {counter['value']}, likely blocked"

        # Total time should be ~500ms (parallel), not ~800ms (sequential)
        assert elapsed < 1.0, f"Took {elapsed}s, looks like sequential execution"


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_run_subprocess_command_not_found(self) -> None:
        """Verify appropriate error for non-existent command."""
        from backend.utils.async_io import run_subprocess

        with pytest.raises(FileNotFoundError):
            await run_subprocess(["nonexistent_command_xyz"])

    @pytest.mark.asyncio
    async def test_path_exists_empty_path(self) -> None:
        """Verify path_exists handles empty string."""
        from backend.utils.async_io import path_exists

        # Empty string should return False (current directory behavior on some systems)
        # or could be True depending on implementation
        result = await path_exists("")
        # Just verify it doesn't crash and returns a boolean
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_read_file_empty_file(self, tmp_path: Path) -> None:
        """Verify read_file handles empty files correctly."""
        from backend.utils.async_io import read_file

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = await read_file(str(empty_file))
        assert result == ""

    @pytest.mark.asyncio
    async def test_run_sync_with_no_args(self) -> None:
        """Verify run_sync works with no-argument functions."""
        from backend.utils.async_io import run_sync

        def no_args() -> int:
            return 42

        result = await run_sync(no_args)
        assert result == 42


# =============================================================================
# Test: psutil Process Utilities
# =============================================================================


class TestPidExists:
    """Tests for pid_exists - async check if process exists."""

    @pytest.mark.asyncio
    async def test_pid_exists_current_process(self) -> None:
        """Verify pid_exists returns True for current process."""
        from backend.utils.async_io import pid_exists

        current_pid = os.getpid()
        result = await pid_exists(current_pid)
        assert result is True

    @pytest.mark.asyncio
    async def test_pid_exists_invalid_pid(self) -> None:
        """Verify pid_exists returns False for invalid/non-existent PID."""
        from backend.utils.async_io import pid_exists

        # Use an extremely large PID that's unlikely to exist
        invalid_pid = 999999999
        result = await pid_exists(invalid_pid)
        assert result is False

    @pytest.mark.asyncio
    async def test_pid_exists_zero_pid(self) -> None:
        """Verify pid_exists handles PID 0 (kernel scheduler on Unix)."""
        from backend.utils.async_io import pid_exists

        # PID 0 is the kernel scheduler on Unix, may or may not be "accessible"
        result = await pid_exists(0)
        # Just verify it returns a boolean without crashing
        assert isinstance(result, bool)


class TestGetProcess:
    """Tests for get_process - async get Process object."""

    @pytest.mark.asyncio
    async def test_get_process_current_process(self) -> None:
        """Verify get_process returns Process object for current process."""
        import psutil

        from backend.utils.async_io import get_process

        current_pid = os.getpid()
        proc = await get_process(current_pid)

        assert proc is not None
        assert isinstance(proc, psutil.Process)
        assert proc.pid == current_pid

    @pytest.mark.asyncio
    async def test_get_process_invalid_pid_returns_none(self) -> None:
        """Verify get_process returns None (not raises) for invalid PID."""
        from backend.utils.async_io import get_process

        invalid_pid = 999999999
        proc = await get_process(invalid_pid)

        # Should return None, not raise an exception
        assert proc is None

    @pytest.mark.asyncio
    async def test_get_process_negative_pid(self) -> None:
        """Verify get_process handles negative PID gracefully."""
        from backend.utils.async_io import get_process

        # Negative PIDs are invalid
        proc = await get_process(-1)
        assert proc is None


class TestProcessIsRunning:
    """Tests for process_is_running - async check if process is running."""

    @pytest.mark.asyncio
    async def test_process_is_running_current_process(self) -> None:
        """Verify process_is_running returns True for current running process."""
        from backend.utils.async_io import process_is_running

        current_pid = os.getpid()
        result = await process_is_running(current_pid)
        assert result is True

    @pytest.mark.asyncio
    async def test_process_is_running_invalid_pid(self) -> None:
        """Verify process_is_running returns False for invalid PID."""
        from backend.utils.async_io import process_is_running

        invalid_pid = 999999999
        result = await process_is_running(invalid_pid)
        assert result is False

    @pytest.mark.asyncio
    async def test_process_is_running_never_raises(self) -> None:
        """Verify process_is_running never raises for any PID."""
        from backend.utils.async_io import process_is_running

        # Test various edge cases that might raise exceptions
        test_pids = [-1, 0, 1, 999999999]

        for pid in test_pids:
            # Should never raise, always return bool
            result = await process_is_running(pid)
            assert isinstance(result, bool)


class TestIterProcesses:
    """Tests for iter_processes - async iterator over processes."""

    @pytest.mark.asyncio
    async def test_iter_processes_yields_dicts(self) -> None:
        """Verify iter_processes yields dicts with pid and name."""
        from backend.utils.async_io import iter_processes

        processes = []
        async for proc_info in iter_processes():
            processes.append(proc_info)
            # Collect a few to verify structure, don't need all
            if len(processes) >= 5:
                break

        assert len(processes) >= 1  # At least our own process

        # Verify structure of yielded dicts
        for proc_info in processes:
            assert isinstance(proc_info, dict)
            assert "pid" in proc_info
            assert "name" in proc_info
            assert isinstance(proc_info["pid"], int)

    @pytest.mark.asyncio
    async def test_iter_processes_with_custom_attrs(self) -> None:
        """Verify iter_processes respects custom attrs parameter."""
        from backend.utils.async_io import iter_processes

        processes = []
        async for proc_info in iter_processes(attrs=["pid", "status"]):
            processes.append(proc_info)
            if len(processes) >= 3:
                break

        # Verify custom attributes are present
        for proc_info in processes:
            assert "pid" in proc_info
            assert "status" in proc_info

    @pytest.mark.asyncio
    async def test_iter_processes_includes_current_process(self) -> None:
        """Verify iter_processes includes the current process."""
        from backend.utils.async_io import iter_processes

        current_pid = os.getpid()
        found_current = False

        async for proc_info in iter_processes():
            if proc_info.get("pid") == current_pid:
                found_current = True
                break

        assert found_current, "Current process not found in iter_processes"


# =============================================================================
# Test: psutil Network Utilities
# =============================================================================


class TestGetNetConnections:
    """Tests for get_net_connections - async network connections."""

    @pytest.mark.asyncio
    async def test_get_net_connections_returns_list(self) -> None:
        """Verify get_net_connections returns a list or raises AccessDenied.

        Note: On macOS, psutil.net_connections() requires elevated privileges.
        This test accepts either a list result or AccessDenied exception.
        """
        import psutil

        from backend.utils.async_io import get_net_connections

        try:
            connections = await get_net_connections()
            assert isinstance(connections, list)
            # May be empty on some systems, that's okay
        except psutil.AccessDenied:
            # On macOS without elevated privileges, this is expected
            pytest.skip("net_connections requires elevated privileges on this platform")

    @pytest.mark.asyncio
    async def test_get_net_connections_with_kind(self) -> None:
        """Verify get_net_connections accepts kind parameter.

        Note: On macOS, psutil.net_connections() requires elevated privileges.
        This test accepts either a list result or AccessDenied exception.
        """
        import psutil

        from backend.utils.async_io import get_net_connections

        # Test different connection types
        for kind in ["inet", "inet4", "inet6", "tcp", "udp"]:
            try:
                connections = await get_net_connections(kind=kind)
                assert isinstance(connections, list)
            except psutil.AccessDenied:
                # On macOS without elevated privileges, this is expected
                pytest.skip("net_connections requires elevated privileges on this platform")


# =============================================================================
# Test: psutil System Resource Utilities
# =============================================================================


class TestGetCpuPercent:
    """Tests for get_cpu_percent - async CPU usage."""

    @pytest.mark.asyncio
    async def test_get_cpu_percent_returns_float(self) -> None:
        """Verify get_cpu_percent returns a float between 0 and 100."""
        from backend.utils.async_io import get_cpu_percent

        cpu = await get_cpu_percent()

        assert isinstance(cpu, float)
        assert 0.0 <= cpu <= 100.0

    @pytest.mark.asyncio
    async def test_get_cpu_percent_with_interval(self) -> None:
        """Verify get_cpu_percent respects interval parameter."""
        import time

        from backend.utils.async_io import get_cpu_percent

        start = time.monotonic()
        cpu = await get_cpu_percent(interval=0.2)
        elapsed = time.monotonic() - start

        assert isinstance(cpu, float)
        # Should take approximately the interval time
        assert elapsed >= 0.15  # Allow some margin


class TestGetVirtualMemory:
    """Tests for get_virtual_memory - async memory stats."""

    @pytest.mark.asyncio
    async def test_get_virtual_memory_returns_svmem(self) -> None:
        """Verify get_virtual_memory returns svmem with expected fields."""
        from backend.utils.async_io import get_virtual_memory

        mem = await get_virtual_memory()

        # svmem is a named tuple with these fields
        assert hasattr(mem, "total")
        assert hasattr(mem, "available")
        assert hasattr(mem, "percent")

        # Verify types and reasonable values
        assert isinstance(mem.total, int)
        assert isinstance(mem.available, int)
        assert isinstance(mem.percent, float)

        assert mem.total > 0
        assert mem.available > 0
        assert 0.0 <= mem.percent <= 100.0


class TestGetDiskUsage:
    """Tests for get_disk_usage - async disk usage stats."""

    @pytest.mark.asyncio
    async def test_get_disk_usage_returns_sdiskusage(self) -> None:
        """Verify get_disk_usage returns sdiskusage with expected fields."""
        from backend.utils.async_io import get_disk_usage

        disk = await get_disk_usage("/")

        # sdiskusage is a named tuple with these fields
        assert hasattr(disk, "total")
        assert hasattr(disk, "used")
        assert hasattr(disk, "free")
        assert hasattr(disk, "percent")

        # Verify types and reasonable values
        assert isinstance(disk.total, int)
        assert isinstance(disk.used, int)
        assert isinstance(disk.free, int)
        assert isinstance(disk.percent, float)

        assert disk.total > 0
        assert disk.used >= 0
        assert disk.free >= 0
        assert 0.0 <= disk.percent <= 100.0

    @pytest.mark.asyncio
    async def test_get_disk_usage_default_path(self) -> None:
        """Verify get_disk_usage uses root as default path."""
        from backend.utils.async_io import get_disk_usage

        # Default should be root
        disk = await get_disk_usage()
        assert disk.total > 0

    @pytest.mark.asyncio
    async def test_get_disk_usage_custom_path(self, tmp_path: Path) -> None:
        """Verify get_disk_usage works with custom paths."""
        from backend.utils.async_io import get_disk_usage

        disk = await get_disk_usage(str(tmp_path))
        assert disk.total > 0
