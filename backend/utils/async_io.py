"""
Async I/O Utilities for Non-Blocking Operations
================================================

This module provides async-safe wrapper utilities for blocking I/O operations
to ensure the event loop is never blocked during startup or other async contexts.

These utilities are designed to be generic, type-safe, and composable.

Key Functions:
    Generic:
        - run_sync: Generic wrapper for any blocking function

    File System:
        - path_exists: Async-safe file/directory existence check
        - read_file: Async-safe file reading

    Subprocess:
        - run_subprocess: Async subprocess execution with timeout support

    Process Utilities (psutil wrappers):
        - pid_exists: Check if a process exists
        - get_process: Get a Process object (returns None if not found)
        - process_is_running: Check if a process is running (never raises)
        - iter_processes: Async iterator over running processes

    Network Utilities:
        - get_net_connections: Get network connections

    System Resources:
        - get_cpu_percent: Get CPU usage percentage
        - get_virtual_memory: Get memory statistics
        - get_disk_usage: Get disk usage statistics

Usage:
    from backend.utils.async_io import (
        run_sync,
        path_exists,
        read_file,
        run_subprocess,
        pid_exists,
        process_is_running,
        iter_processes,
        get_cpu_percent,
        get_virtual_memory,
    )

    # Wrap any blocking function
    result = await run_sync(os.listdir, "/some/path")

    # Check if path exists
    if await path_exists("/some/file.txt"):
        content = await read_file("/some/file.txt")

    # Run subprocess with timeout
    result = await run_subprocess(["ls", "-la"], timeout=5.0)

    # Check if process is running (never raises)
    if await process_is_running(1234):
        print("Process is active")

    # Iterate over processes
    async for proc in iter_processes(attrs=['pid', 'name', 'status']):
        print(f"PID {proc['pid']}: {proc['name']}")

    # Get system resources
    cpu = await get_cpu_percent()
    mem = await get_virtual_memory()
    disk = await get_disk_usage("/")

Design Notes:
    - run_sync uses asyncio.to_thread() for Python 3.9+ compatibility
    - run_subprocess uses asyncio.create_subprocess_exec() for true async I/O
    - All functions are type-safe with proper generics
    - Process utilities follow "never raise" principle - return None/False for errors
    - Network utilities may raise on platforms without elevated privileges
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator, Callable, TypeVar, Union

import psutil

# Type variable for generic return type
T = TypeVar("T")


# =============================================================================
# Generic Async Wrapper
# =============================================================================


async def run_sync(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Execute a blocking synchronous function without blocking the event loop.

    This function wraps any blocking operation and runs it in a thread pool
    executor, allowing the event loop to continue processing other tasks.

    Args:
        func: The blocking function to execute
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value of the function

    Raises:
        Any exception raised by the function is propagated to the caller

    Example:
        # Wrap a blocking file system operation
        files = await run_sync(os.listdir, "/some/path")

        # Wrap a blocking function with kwargs
        result = await run_sync(my_func, arg1, arg2, kwarg1="value")

        # Wrap psutil blocking calls
        cpu_percent = await run_sync(psutil.cpu_percent, interval=0.1)
    """
    # asyncio.to_thread handles both args and kwargs properly
    # and runs the function in the default executor
    return await asyncio.to_thread(func, *args, **kwargs)


# =============================================================================
# File System Utilities
# =============================================================================


async def path_exists(path: Union[str, Path]) -> bool:
    """
    Check if a path exists without blocking the event loop.

    This is an async-safe wrapper around os.path.exists().

    Args:
        path: The file or directory path to check (str or Path)

    Returns:
        True if the path exists, False otherwise

    Example:
        if await path_exists("/var/log/jarvis.log"):
            print("Log file exists")

        # Also works with Path objects
        from pathlib import Path
        if await path_exists(Path.home() / ".config"):
            print("Config directory exists")
    """
    # Convert Path to string for os.path.exists compatibility
    path_str = str(path) if isinstance(path, Path) else path
    return await run_sync(os.path.exists, path_str)


async def read_file(
    path: Union[str, Path],
    encoding: str = "utf-8",
) -> str:
    """
    Read a file's contents without blocking the event loop.

    This is an async-safe wrapper for reading text files.

    Args:
        path: The file path to read (str or Path)
        encoding: The file encoding (default: utf-8)

    Returns:
        The file contents as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file cannot be read due to permissions
        UnicodeDecodeError: If the file cannot be decoded with the specified encoding

    Example:
        content = await read_file("/etc/hostname")

        # With custom encoding
        content = await read_file("data.txt", encoding="latin-1")

        # With Path object
        content = await read_file(Path.home() / ".bashrc")
    """
    path_obj = Path(path) if isinstance(path, str) else path

    def _read() -> str:
        return path_obj.read_text(encoding=encoding)

    return await run_sync(_read)


# =============================================================================
# Subprocess Utilities
# =============================================================================


async def run_subprocess(
    cmd: list[str],
    timeout: float | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess asynchronously without blocking the event loop.

    This function uses asyncio.create_subprocess_exec() for true async I/O,
    which is more efficient than running subprocess.run() in a thread.

    Args:
        cmd: Command and arguments as a list of strings
        timeout: Maximum execution time in seconds. If exceeded, raises
                 asyncio.TimeoutError. If None, no timeout is applied.
        **kwargs: Additional arguments passed to asyncio.create_subprocess_exec().
                  Common kwargs include:
                  - cwd: Working directory for the subprocess
                  - env: Environment variables dict
                  - stdin: Standard input (default: None)

    Returns:
        subprocess.CompletedProcess with:
        - args: The command that was executed
        - returncode: Exit code of the subprocess
        - stdout: Captured stdout as bytes
        - stderr: Captured stderr as bytes

    Raises:
        asyncio.TimeoutError: If the command exceeds the specified timeout
        FileNotFoundError: If the command executable is not found
        PermissionError: If the command cannot be executed due to permissions

    Example:
        # Basic usage
        result = await run_subprocess(["ls", "-la"])
        print(result.stdout.decode())

        # With timeout
        try:
            result = await run_subprocess(["sleep", "10"], timeout=2.0)
        except asyncio.TimeoutError:
            print("Command timed out")

        # With custom working directory
        result = await run_subprocess(["pwd"], cwd="/tmp")
    """
    # Always capture stdout and stderr
    kwargs.setdefault("stdout", asyncio.subprocess.PIPE)
    kwargs.setdefault("stderr", asyncio.subprocess.PIPE)

    # Create the subprocess
    process = await asyncio.create_subprocess_exec(*cmd, **kwargs)

    try:
        # Wait for completion with optional timeout
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        # Kill the process on timeout
        try:
            process.kill()
            await process.wait()
        except ProcessLookupError:
            pass  # Process already terminated
        raise

    # Return a CompletedProcess for compatibility with subprocess.run()
    # Note: After communicate() completes, returncode is guaranteed to be set
    returncode = process.returncode if process.returncode is not None else -1
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# =============================================================================
# Process Utilities
# =============================================================================


async def pid_exists(pid: int) -> bool:
    """
    Check if a process exists without blocking the event loop.

    This is an async-safe wrapper around psutil.pid_exists().

    Args:
        pid: The process ID to check

    Returns:
        True if the process exists, False otherwise

    Example:
        if await pid_exists(1234):
            print("Process is running")
    """
    return await run_sync(psutil.pid_exists, pid)


async def get_process(pid: int) -> psutil.Process | None:
    """
    Get a Process object for the given PID without blocking.

    This function never raises exceptions for common error conditions.
    Returns None if the process doesn't exist or is inaccessible.

    Args:
        pid: The process ID to get

    Returns:
        psutil.Process object if process exists and is accessible, None otherwise

    Example:
        proc = await get_process(1234)
        if proc is not None:
            print(f"Process name: {proc.name()}")
    """
    try:
        return await run_sync(psutil.Process, pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        # ValueError can occur for invalid PIDs (e.g., negative)
        return None


async def process_is_running(pid: int) -> bool:
    """
    Check if a process is running without blocking the event loop.

    This function never raises exceptions - it returns False for any
    error condition (process doesn't exist, access denied, etc.).

    Args:
        pid: The process ID to check

    Returns:
        True if the process exists and is running, False otherwise

    Example:
        if await process_is_running(1234):
            print("Process is still active")
        else:
            print("Process has stopped")
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
    Iterate over running processes without blocking the event loop.

    This function silently skips processes that disappear or become
    inaccessible during iteration.

    Args:
        attrs: List of process attributes to include in each dict.
               Default is ['pid', 'name'].

    Yields:
        dict: Process information dict with the requested attributes

    Example:
        async for proc in iter_processes(attrs=['pid', 'name', 'status']):
            print(f"PID {proc['pid']}: {proc['name']} ({proc['status']})")
    """
    attrs = attrs or ["pid", "name"]
    # Get all processes in one blocking call to minimize event loop blocking
    procs = await run_sync(lambda: list(psutil.process_iter(attrs)))
    for proc in procs:
        try:
            yield proc.info
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process disappeared or became inaccessible, skip it
            continue


# =============================================================================
# Network Utilities
# =============================================================================


async def get_net_connections(kind: str = "inet") -> list[Any]:
    """
    Get network connections without blocking the event loop.

    This is an async-safe wrapper around psutil.net_connections().

    Args:
        kind: The type of connections to return. Common values:
              - 'inet': IPv4 and IPv6
              - 'inet4': IPv4 only
              - 'inet6': IPv6 only
              - 'tcp': TCP connections
              - 'udp': UDP connections
              - 'all': All types

    Returns:
        List of connection named tuples

    Example:
        connections = await get_net_connections(kind='tcp')
        for conn in connections:
            print(f"{conn.laddr} -> {conn.raddr}")
    """
    return await run_sync(psutil.net_connections, kind)


# =============================================================================
# System Resource Utilities
# =============================================================================


async def get_cpu_percent(interval: float = 0.1) -> float:
    """
    Get CPU usage percentage without blocking the event loop.

    This is an async-safe wrapper around psutil.cpu_percent().
    Uses a small default interval to minimize blocking time.

    Args:
        interval: Seconds to wait between CPU measurements.
                  Smaller values are less accurate but faster.
                  Default is 0.1 seconds.

    Returns:
        CPU usage percentage as a float (0.0 to 100.0)

    Example:
        cpu = await get_cpu_percent()
        print(f"CPU usage: {cpu}%")
    """
    return await run_sync(psutil.cpu_percent, interval=interval)


async def get_virtual_memory() -> Any:
    """
    Get virtual memory statistics without blocking the event loop.

    This is an async-safe wrapper around psutil.virtual_memory().

    Returns:
        Named tuple with memory statistics:
        - total: Total physical memory
        - available: Memory available without swapping
        - percent: Percentage of memory used
        - used: Memory used
        - free: Memory not being used at all

    Example:
        mem = await get_virtual_memory()
        print(f"Memory usage: {mem.percent}%")
        print(f"Available: {mem.available / (1024**3):.1f} GB")
    """
    return await run_sync(psutil.virtual_memory)


async def get_disk_usage(path: str = "/") -> Any:
    """
    Get disk usage statistics without blocking the event loop.

    This is an async-safe wrapper around psutil.disk_usage().

    Args:
        path: The path to check disk usage for. Default is root '/'.

    Returns:
        Named tuple with disk statistics:
        - total: Total disk space in bytes
        - used: Used disk space in bytes
        - free: Free disk space in bytes
        - percent: Percentage of disk used

    Example:
        disk = await get_disk_usage("/")
        print(f"Disk usage: {disk.percent}%")
        print(f"Free: {disk.free / (1024**3):.1f} GB")
    """
    return await run_sync(psutil.disk_usage, path)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Generic wrapper
    "run_sync",
    # File system utilities
    "path_exists",
    "read_file",
    # Subprocess utilities
    "run_subprocess",
    # Process utilities
    "pid_exists",
    "get_process",
    "process_is_running",
    "iter_processes",
    # Network utilities
    "get_net_connections",
    # System resource utilities
    "get_cpu_percent",
    "get_virtual_memory",
    "get_disk_usage",
]
