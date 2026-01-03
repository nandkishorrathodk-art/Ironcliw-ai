"""
Fork-Safe Subprocess Execution for macOS Multi-Threaded Environments
======================================================================

This module solves the critical macOS fork-safety crash:
  "multi-threaded process forked"
  "BUG IN CLIENT OF LIBDISPATCH: trying to lock recursively"

ROOT CAUSE:
- subprocess.Popen() uses fork() + exec() internally
- In a multi-threaded process, fork() only copies the calling thread
- Locks held by other threads are copied in a locked state
- The child process tries to acquire these orphaned locks → deadlock → crash

SOLUTION:
- asyncio.create_subprocess_exec() uses posix_spawn() on macOS (Python 3.8+)
- posix_spawn() is fork-safe and doesn't inherit the parent's thread locks
- This module wraps all subprocess operations to use posix_spawn under the hood

USAGE:
    # Async context (preferred)
    from backend.utils.fork_safe_subprocess import run_subprocess_async, popen_async

    result = await run_subprocess_async(['echo', 'hello'])

    async with popen_async(['ffmpeg', '-i', 'input.mp3', '-f', 'wav', 'pipe:1'],
                           stdin=asyncio.subprocess.PIPE,
                           stdout=asyncio.subprocess.PIPE) as proc:
        output, _ = await proc.communicate(input_data)

    # Synchronous context (creates event loop internally)
    from backend.utils.fork_safe_subprocess import run_subprocess_sync

    result = run_subprocess_sync(['echo', 'hello'])

ARCHITECTURE:
- All subprocess operations go through asyncio.create_subprocess_exec()
- Uses posix_spawn() on macOS (fork-safe)
- Thread-aware: detects active threads and warns/logs appropriately
- Process lifecycle tracking and cleanup
- Comprehensive error handling with detailed diagnostics
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default timeout for subprocess operations
DEFAULT_TIMEOUT = 30.0

# Maximum number of concurrent subprocess operations
MAX_CONCURRENT_SUBPROCESSES = int(os.getenv("JARVIS_MAX_SUBPROCESSES", "10"))

# Enable verbose fork-safety logging
FORK_SAFETY_DEBUG = os.getenv("JARVIS_FORK_SAFETY_DEBUG", "false").lower() == "true"

# Semaphore for limiting concurrent subprocess operations
_subprocess_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the subprocess semaphore (lazy initialization)."""
    global _subprocess_semaphore
    if _subprocess_semaphore is None:
        _subprocess_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUBPROCESSES)
    return _subprocess_semaphore


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class SubprocessResult:
    """
    Result of a subprocess execution.

    Attributes:
        returncode: Exit code of the process (None if still running or terminated)
        stdout: Captured stdout bytes (or None if not captured)
        stderr: Captured stderr bytes (or None if not captured)
        command: The command that was executed
        duration_seconds: How long the subprocess took to complete
        timed_out: Whether the process was killed due to timeout
        error: Exception message if an error occurred
    """
    returncode: Optional[int] = None
    stdout: Optional[bytes] = None
    stderr: Optional[bytes] = None
    command: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timed_out: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if the subprocess completed successfully."""
        return self.returncode == 0 and not self.timed_out and self.error is None

    def check_returncode(self) -> None:
        """Raise an exception if the subprocess failed."""
        if self.error:
            raise SubprocessError(self.error, self)
        if self.timed_out:
            raise SubprocessTimeoutError(f"Command timed out: {' '.join(self.command)}", self)
        if self.returncode != 0:
            stderr_msg = self.stderr.decode('utf-8', errors='replace')[:500] if self.stderr else ""
            raise SubprocessError(
                f"Command failed with exit code {self.returncode}: {stderr_msg}",
                self
            )


class SubprocessError(Exception):
    """Exception raised when a subprocess fails."""
    def __init__(self, message: str, result: SubprocessResult):
        super().__init__(message)
        self.result = result


class SubprocessTimeoutError(SubprocessError):
    """Exception raised when a subprocess times out."""
    pass


# =============================================================================
# THREAD SAFETY DETECTION
# =============================================================================


def _count_active_threads() -> int:
    """Count the number of active non-daemon threads."""
    return sum(1 for t in threading.enumerate() if t.is_alive() and not t.daemon)


def _log_thread_context(operation: str) -> None:
    """Log current thread context for debugging fork-safety issues."""
    if not FORK_SAFETY_DEBUG:
        return

    active = _count_active_threads()
    thread_names = [t.name for t in threading.enumerate() if t.is_alive()]

    logger.debug(
        f"[FORK-SAFE] {operation}: {active} active threads: {thread_names[:5]}"
        + (f"... and {len(thread_names) - 5} more" if len(thread_names) > 5 else "")
    )


def _warn_if_threaded(operation: str) -> None:
    """
    Warn if subprocess is being called from a multi-threaded context.

    This is informational - the fork-safe implementation handles it correctly,
    but it's useful for debugging and understanding the execution context.
    """
    active = _count_active_threads()
    if active > 2:  # Main thread + possibly one other (e.g., asyncio)
        logger.debug(
            f"[FORK-SAFE] {operation} with {active} active threads - "
            f"using posix_spawn for fork-safety"
        )


# =============================================================================
# CORE ASYNC SUBPROCESS FUNCTIONS
# =============================================================================


async def run_subprocess_async(
    command: Sequence[str],
    *,
    input: Optional[bytes] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    check: bool = False,
) -> SubprocessResult:
    """
    Run a subprocess asynchronously using fork-safe posix_spawn.

    This is the recommended way to run subprocesses in async code.
    It uses asyncio.create_subprocess_exec() which uses posix_spawn()
    on macOS, avoiding fork-safety issues.

    Args:
        command: Command and arguments to execute
        input: Optional bytes to send to stdin
        timeout: Maximum execution time in seconds (None for no timeout)
        cwd: Working directory for the subprocess
        env: Environment variables (None to inherit)
        capture_output: Whether to capture stdout/stderr
        check: If True, raise SubprocessError on non-zero exit code

    Returns:
        SubprocessResult with returncode, stdout, stderr, etc.

    Example:
        result = await run_subprocess_async(['echo', 'hello'])
        print(result.stdout.decode())  # "hello\n"
    """
    _warn_if_threaded("run_subprocess_async")
    _log_thread_context("run_subprocess_async")

    cmd_list = list(command)
    start_time = time.monotonic()

    result = SubprocessResult(command=cmd_list)

    try:
        async with _get_semaphore():
            # Prepare stdin/stdout/stderr based on options
            stdin = asyncio.subprocess.PIPE if input is not None else None
            stdout = asyncio.subprocess.PIPE if capture_output else None
            stderr = asyncio.subprocess.PIPE if capture_output else None

            # Create subprocess using posix_spawn (fork-safe on macOS)
            proc = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd,
                env=env,
            )

            try:
                if timeout is not None:
                    stdout_data, stderr_data = await asyncio.wait_for(
                        proc.communicate(input=input),
                        timeout=timeout
                    )
                else:
                    stdout_data, stderr_data = await proc.communicate(input=input)

                result.stdout = stdout_data
                result.stderr = stderr_data
                result.returncode = proc.returncode

            except asyncio.TimeoutError:
                # Kill the process on timeout
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                result.timed_out = True
                result.error = f"Command timed out after {timeout}s"
                logger.warning(f"[FORK-SAFE] Command timed out: {' '.join(cmd_list[:3])}")

    except Exception as e:
        result.error = str(e)
        logger.error(f"[FORK-SAFE] Subprocess error: {e}")

    result.duration_seconds = time.monotonic() - start_time

    if check:
        result.check_returncode()

    return result


@asynccontextmanager
async def popen_async(
    command: Sequence[str],
    *,
    stdin: Optional[int] = None,
    stdout: Optional[int] = None,
    stderr: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
):
    """
    Async context manager for subprocess.Popen-like functionality.

    This is fork-safe and uses posix_spawn() on macOS.

    Args:
        command: Command and arguments to execute
        stdin: Stdin pipe option (e.g., asyncio.subprocess.PIPE)
        stdout: Stdout pipe option
        stderr: Stderr pipe option
        cwd: Working directory
        env: Environment variables

    Yields:
        asyncio.subprocess.Process object

    Example:
        async with popen_async(
            ['ffmpeg', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE
        ) as proc:
            output, _ = await proc.communicate(input_data)
    """
    _warn_if_threaded("popen_async")
    _log_thread_context("popen_async")

    cmd_list = list(command)
    proc = None

    try:
        async with _get_semaphore():
            proc = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd,
                env=env,
            )
            yield proc

    finally:
        if proc is not None:
            # Ensure process is terminated and cleaned up
            if proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except Exception as e:
                    logger.warning(f"[FORK-SAFE] Failed to cleanup process: {e}")


async def run_ffmpeg_async(
    input_data: bytes,
    *,
    input_format: Optional[str] = None,
    output_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
    timeout: float = 10.0,
    extra_input_args: Optional[List[str]] = None,
    extra_output_args: Optional[List[str]] = None,
    error_handling: str = "ignore_err",
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Run FFmpeg audio conversion in a fork-safe manner.

    This is a specialized helper for the common use case of converting
    audio formats using FFmpeg piped I/O.

    Args:
        input_data: Raw input audio bytes
        input_format: FFmpeg input format hint (e.g., 'webm', 'opus')
        output_format: Output format (default: 'wav')
        sample_rate: Output sample rate (default: 16000)
        channels: Output channels (default: 1 mono)
        codec: Output codec (default: 'pcm_s16le')
        timeout: Maximum execution time
        extra_input_args: Additional FFmpeg input arguments
        extra_output_args: Additional FFmpeg output arguments
        error_handling: FFmpeg -err_detect value (default: 'ignore_err')

    Returns:
        Tuple of (output_bytes, error_message)
        - On success: (bytes, None)
        - On failure: (None, error_string)

    Example:
        wav_data, error = await run_ffmpeg_async(
            webm_bytes,
            input_format='webm',
            output_format='wav',
            sample_rate=16000
        )
    """
    _log_thread_context("run_ffmpeg_async")

    # Build FFmpeg command
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning']

    # Error handling
    if error_handling:
        cmd.extend(['-err_detect', error_handling])

    # Input format hint
    if input_format:
        cmd.extend(['-f', input_format])

    # Extra input args
    if extra_input_args:
        cmd.extend(extra_input_args)

    # Input from pipe
    cmd.extend(['-i', 'pipe:0'])

    # Output settings
    cmd.extend([
        '-f', output_format,
        '-ar', str(sample_rate),
        '-ac', str(channels),
        '-acodec', codec,
    ])

    # Extra output args
    if extra_output_args:
        cmd.extend(extra_output_args)

    # Output to pipe
    cmd.append('pipe:1')

    try:
        async with popen_async(
            cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        ) as proc:
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(input=input_data),
                    timeout=timeout
                )

                if proc.returncode == 0 and len(stdout_data) > 44:  # WAV header is 44 bytes
                    return stdout_data, None
                else:
                    error_msg = stderr_data.decode('utf-8', errors='ignore') if stderr_data else "Unknown error"
                    return None, f"FFmpeg failed (rc={proc.returncode}): {error_msg[:200]}"

            except asyncio.TimeoutError:
                return None, f"FFmpeg timed out after {timeout}s"

    except Exception as e:
        return None, f"FFmpeg error: {e}"


# =============================================================================
# SYNCHRONOUS WRAPPERS (for non-async code)
# =============================================================================


def run_subprocess_sync(
    command: Sequence[str],
    *,
    input: Optional[bytes] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    check: bool = False,
) -> SubprocessResult:
    """
    Run a subprocess synchronously using fork-safe posix_spawn.

    This creates a temporary event loop to run the async implementation.
    For async code, use run_subprocess_async() directly.

    Args:
        command: Command and arguments to execute
        input: Optional bytes to send to stdin
        timeout: Maximum execution time in seconds
        cwd: Working directory for the subprocess
        env: Environment variables
        capture_output: Whether to capture stdout/stderr
        check: If True, raise SubprocessError on non-zero exit code

    Returns:
        SubprocessResult with returncode, stdout, stderr, etc.

    Example:
        result = run_subprocess_sync(['echo', 'hello'])
        print(result.stdout.decode())  # "hello\n"
    """
    _warn_if_threaded("run_subprocess_sync")

    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - this is a code smell, but handle it gracefully
        logger.warning(
            "[FORK-SAFE] run_subprocess_sync called from async context. "
            "Consider using run_subprocess_async instead."
        )
        # Use run_coroutine_threadsafe to avoid nested event loop issues
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            run_subprocess_async(
                command,
                input=input,
                timeout=timeout,
                cwd=cwd,
                env=env,
                capture_output=capture_output,
                check=check,
            ),
            loop
        )
        return future.result(timeout=timeout + 5 if timeout else None)
    except RuntimeError:
        # No event loop running - create one
        pass

    # Create a new event loop for this call
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            run_subprocess_async(
                command,
                input=input,
                timeout=timeout,
                cwd=cwd,
                env=env,
                capture_output=capture_output,
                check=check,
            )
        )
    finally:
        loop.close()


def run_ffmpeg_sync(
    input_data: bytes,
    *,
    input_format: Optional[str] = None,
    output_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
    timeout: float = 10.0,
    extra_input_args: Optional[List[str]] = None,
    extra_output_args: Optional[List[str]] = None,
    error_handling: str = "ignore_err",
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Run FFmpeg synchronously using fork-safe posix_spawn.

    This is a synchronous wrapper around run_ffmpeg_async().
    For async code, use run_ffmpeg_async() directly.

    Args:
        See run_ffmpeg_async() for argument documentation.

    Returns:
        Tuple of (output_bytes, error_message)
    """
    _warn_if_threaded("run_ffmpeg_sync")

    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        logger.warning(
            "[FORK-SAFE] run_ffmpeg_sync called from async context. "
            "Consider using run_ffmpeg_async instead."
        )
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            run_ffmpeg_async(
                input_data,
                input_format=input_format,
                output_format=output_format,
                sample_rate=sample_rate,
                channels=channels,
                codec=codec,
                timeout=timeout,
                extra_input_args=extra_input_args,
                extra_output_args=extra_output_args,
                error_handling=error_handling,
            ),
            loop
        )
        return future.result(timeout=timeout + 5)
    except RuntimeError:
        pass

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            run_ffmpeg_async(
                input_data,
                input_format=input_format,
                output_format=output_format,
                sample_rate=sample_rate,
                channels=channels,
                codec=codec,
                timeout=timeout,
                extra_input_args=extra_input_args,
                extra_output_args=extra_output_args,
                error_handling=error_handling,
            )
        )
    finally:
        loop.close()


# =============================================================================
# EXECUTOR-SAFE SUBPROCESS (for run_in_executor contexts)
# =============================================================================


class ForkSafeExecutor:
    """
    A wrapper for running subprocess operations from within run_in_executor.

    When you need to call subprocess from a function that's being run via
    loop.run_in_executor(), you can't use asyncio directly. This class
    provides methods that are safe to call from executor threads.

    The key insight is that we create a NEW event loop in the executor thread,
    which allows us to use asyncio.create_subprocess_exec() (posix_spawn)
    without conflicting with the main event loop.

    Example:
        executor = ForkSafeExecutor()

        def sync_function():
            # This runs in a thread via run_in_executor
            result = executor.run(['echo', 'hello'])
            return result.stdout

        # In async code
        output = await loop.run_in_executor(None, sync_function)
    """

    def __init__(self, default_timeout: float = DEFAULT_TIMEOUT):
        self.default_timeout = default_timeout
        self._thread_loops: Dict[int, asyncio.AbstractEventLoop] = {}
        self._lock = threading.Lock()

    def _get_thread_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for the current thread."""
        thread_id = threading.current_thread().ident

        with self._lock:
            if thread_id not in self._thread_loops:
                loop = asyncio.new_event_loop()
                self._thread_loops[thread_id] = loop
            return self._thread_loops[thread_id]

    def run(
        self,
        command: Sequence[str],
        *,
        input: Optional[bytes] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True,
        check: bool = False,
    ) -> SubprocessResult:
        """
        Run a subprocess from an executor thread.

        This is safe to call from functions running via run_in_executor().
        """
        if timeout is None:
            timeout = self.default_timeout

        loop = self._get_thread_loop()

        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                run_subprocess_async(
                    command,
                    input=input,
                    timeout=timeout,
                    cwd=cwd,
                    env=env,
                    capture_output=capture_output,
                    check=check,
                )
            )
        finally:
            # Don't close the loop - we reuse it for the same thread
            pass

    def run_ffmpeg(
        self,
        input_data: bytes,
        *,
        input_format: Optional[str] = None,
        output_format: str = "wav",
        sample_rate: int = 16000,
        channels: int = 1,
        codec: str = "pcm_s16le",
        timeout: float = 10.0,
        extra_input_args: Optional[List[str]] = None,
        extra_output_args: Optional[List[str]] = None,
        error_handling: str = "ignore_err",
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """Run FFmpeg from an executor thread."""
        loop = self._get_thread_loop()

        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                run_ffmpeg_async(
                    input_data,
                    input_format=input_format,
                    output_format=output_format,
                    sample_rate=sample_rate,
                    channels=channels,
                    codec=codec,
                    timeout=timeout,
                    extra_input_args=extra_input_args,
                    extra_output_args=extra_output_args,
                    error_handling=error_handling,
                )
            )
        finally:
            pass

    def cleanup(self) -> None:
        """Clean up thread event loops. Call this on shutdown."""
        with self._lock:
            for loop in self._thread_loops.values():
                try:
                    loop.close()
                except Exception:
                    pass
            self._thread_loops.clear()


# Global executor instance for convenience
_global_executor: Optional[ForkSafeExecutor] = None


def get_fork_safe_executor() -> ForkSafeExecutor:
    """Get the global ForkSafeExecutor instance."""
    global _global_executor
    if _global_executor is None:
        _global_executor = ForkSafeExecutor()
    return _global_executor


# =============================================================================
# COMPATIBILITY LAYER (drop-in replacement for subprocess module)
# =============================================================================


class ForkSafeSubprocess:
    """
    Drop-in replacement for subprocess module with fork-safety.

    This provides a compatibility layer that mimics the subprocess module
    API but uses fork-safe posix_spawn under the hood.

    Usage:
        from backend.utils.fork_safe_subprocess import safe_subprocess

        # Instead of: subprocess.run(['echo', 'hello'])
        result = safe_subprocess.run(['echo', 'hello'])

        # Instead of: subprocess.Popen(...)
        proc = safe_subprocess.Popen(['echo', 'hello'], stdout=PIPE)
    """

    PIPE = asyncio.subprocess.PIPE
    STDOUT = asyncio.subprocess.STDOUT
    DEVNULL = asyncio.subprocess.DEVNULL

    @staticmethod
    def run(
        args: Sequence[str],
        *,
        capture_output: bool = False,
        timeout: Optional[float] = None,
        check: bool = False,
        input: Optional[bytes] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> SubprocessResult:
        """
        Fork-safe replacement for subprocess.run().

        Note: Some subprocess.run() options are not supported.
        Unsupported kwargs are logged and ignored.
        """
        if kwargs:
            logger.debug(f"[FORK-SAFE] Ignoring unsupported subprocess.run kwargs: {list(kwargs.keys())}")

        return run_subprocess_sync(
            args,
            input=input,
            timeout=timeout or DEFAULT_TIMEOUT,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            check=check,
        )

    @staticmethod
    async def run_async(
        args: Sequence[str],
        *,
        capture_output: bool = False,
        timeout: Optional[float] = None,
        check: bool = False,
        input: Optional[bytes] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> SubprocessResult:
        """Async version of run()."""
        if kwargs:
            logger.debug(f"[FORK-SAFE] Ignoring unsupported kwargs: {list(kwargs.keys())}")

        return await run_subprocess_async(
            args,
            input=input,
            timeout=timeout or DEFAULT_TIMEOUT,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            check=check,
        )

    @staticmethod
    def Popen(
        args: Sequence[str],
        **kwargs
    ):
        """
        Fork-safe Popen is NOT directly supported for sync code.

        This raises an error directing users to use the async version
        or the executor-based approach.
        """
        raise NotImplementedError(
            "ForkSafeSubprocess.Popen() is not available for synchronous code. "
            "Use one of these alternatives:\n"
            "1. Async: async with popen_async(cmd, ...) as proc: ...\n"
            "2. Sync with result: safe_subprocess.run(cmd, capture_output=True)\n"
            "3. In executor: ForkSafeExecutor().run(cmd, ...)"
        )


# Global instance for convenience
safe_subprocess = ForkSafeSubprocess()


# =============================================================================
# INITIALIZATION AND CLEANUP
# =============================================================================


def _cleanup_on_exit():
    """Cleanup function registered with atexit."""
    global _global_executor
    if _global_executor is not None:
        _global_executor.cleanup()
        _global_executor = None


# Register cleanup
import atexit
atexit.register(_cleanup_on_exit)


# Log initialization
if FORK_SAFETY_DEBUG:
    logger.info(
        f"[FORK-SAFE] Fork-safe subprocess module initialized. "
        f"Platform: {platform.system()}, "
        f"Max concurrent: {MAX_CONCURRENT_SUBPROCESSES}"
    )
