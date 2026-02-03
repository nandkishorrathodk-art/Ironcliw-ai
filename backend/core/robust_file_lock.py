"""
RobustFileLock - POSIX Cross-Process File Locking using fcntl.flock().

Guarantees:
- ATOMIC: Lock acquisition is atomic at the kernel level
- EPHEMERAL: Lock automatically released on process death
- NON-BLOCKING EVENT LOOP: All blocking I/O runs in executor
- CROSS-PROCESS: Works across all processes on same machine

Limitations:
- POSIX-ONLY: Does not work on Windows
- LOCAL FILESYSTEM: LOCK_DIR must be on a local filesystem (not NFS)
- NOT REENTRANT: Same process must not acquire same lock twice
- NO FORK: Do not fork while holding the lock
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

# =============================================================================
# Platform check
# =============================================================================

if sys.platform == "win32":
    raise RuntimeError(
        "RobustFileLock is POSIX-only (Linux, macOS). "
        "Windows requires a different implementation using msvcrt.locking."
    )

# =============================================================================
# Configuration (expansion deferred to runtime)
# =============================================================================

LOCK_DIR_RAW = os.environ.get("JARVIS_LOCK_DIR", "~/.jarvis/cross_repo/locks")
LOCK_ACQUIRE_TIMEOUT_S = float(os.environ.get("LOCK_ACQUIRE_TIMEOUT_S", "5.0"))
LOCK_POLL_INTERVAL_S = float(os.environ.get("LOCK_POLL_INTERVAL_S", "0.05"))
LOCK_STALE_WARNING_S = float(os.environ.get("LOCK_STALE_WARNING_S", "30.0"))

# =============================================================================
# Process-local reentrancy guard
# =============================================================================

_held_locks: Set[str] = set()
_held_locks_lock: Optional[asyncio.Lock] = None


def _get_held_locks_lock() -> asyncio.Lock:
    """Lazy initialization of held locks lock (Python 3.9 compatibility)."""
    global _held_locks_lock
    if _held_locks_lock is None:
        _held_locks_lock = asyncio.Lock()
    return _held_locks_lock


class RobustFileLock:
    """
    OS-level file lock using fcntl.flock().
    All blocking I/O runs in executor to avoid blocking the event loop.
    """

    def __init__(self, lock_name: str, source: str = "jarvis"):
        """
        Initialize lock.

        Args:
            lock_name: Unique name for this lock (e.g., "vbia_state")
            source: Identifier for the process holding the lock (for debugging)
        """
        self._lock_name = lock_name
        self._source = source

        # Expand path at runtime (handles both ~ and $VAR)
        lock_dir_raw = os.environ.get("JARVIS_LOCK_DIR", "~/.jarvis/cross_repo/locks")
        self._lock_dir = Path(os.path.expanduser(os.path.expandvars(lock_dir_raw)))
        self._lock_file = self._lock_dir / f"{lock_name}.lock"

        self._fd: Optional[int] = None
        self._acquired = False

    async def acquire(self, timeout_s: Optional[float] = None) -> bool:
        """
        Acquire the lock with timeout.

        Returns:
            True if acquired, False if timeout or error.

        Raises:
            RuntimeError: If same process already holds this lock (reentrancy)
        """
        timeout_s = timeout_s or LOCK_ACQUIRE_TIMEOUT_S

        # Reentrancy check
        held_lock = _get_held_locks_lock()
        async with held_lock:
            if self._lock_name in _held_locks:
                raise RuntimeError(
                    f"Lock '{self._lock_name}' already held by this process. "
                    f"RobustFileLock is NOT reentrant."
                )

        deadline = time.monotonic() + timeout_s
        loop = asyncio.get_running_loop()

        # Ensure directory exists
        await self._ensure_lock_dir()

        # Open lock file (retry once on ENOENT)
        try:
            self._fd = await loop.run_in_executor(None, self._open_lock_file)
        except FileNotFoundError:
            await self._ensure_lock_dir()
            try:
                self._fd = await loop.run_in_executor(None, self._open_lock_file)
            except FileNotFoundError as e:
                logger.error(f"[Lock] Lock dir keeps disappearing: {e}")
                return False
        except OSError as e:
            logger.error(f"[Lock] Failed to open {self._lock_file}: {e}")
            return False

        # Poll for lock acquisition
        while time.monotonic() < deadline:
            try:
                acquired = await loop.run_in_executor(None, self._try_flock)

                if acquired:
                    self._acquired = True

                    # Register in held locks
                    async with held_lock:
                        _held_locks.add(self._lock_name)

                    # Write metadata
                    await loop.run_in_executor(None, self._write_metadata_sync)

                    logger.debug(f"[Lock] Acquired: {self._lock_name}")
                    return True

            except OSError as e:
                logger.error(f"[Lock] flock() error on {self._lock_name}: {e}")
                await self._close_fd_async()
                return False

            await asyncio.sleep(LOCK_POLL_INTERVAL_S)

        # Timeout
        await self._log_stale_warning()
        logger.warning(f"[Lock] Timeout acquiring {self._lock_name} after {timeout_s}s")
        await self._close_fd_async()
        return False

    async def release(self) -> None:
        """Release the lock. Safe to call multiple times."""
        if self._fd is not None and self._acquired:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self._release_sync)
                logger.debug(f"[Lock] Released: {self._lock_name}")
            except OSError as e:
                logger.warning(f"[Lock] Error releasing {self._lock_name}: {e}")
            finally:
                self._acquired = False

                # Remove from held locks
                held_lock = _get_held_locks_lock()
                async with held_lock:
                    _held_locks.discard(self._lock_name)

                await self._close_fd_async()

    # =========================================================================
    # Sync methods (run in executor)
    # =========================================================================

    def _open_lock_file(self) -> int:
        """Open lock file (blocking, run in executor)."""
        return os.open(str(self._lock_file), os.O_RDWR | os.O_CREAT, 0o644)

    def _try_flock(self) -> bool:
        """Try to acquire flock. Returns True if acquired, False if would block."""
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _release_sync(self) -> None:
        """Release flock (blocking, run in executor)."""
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def _write_metadata_sync(self) -> None:
        """Write debugging metadata to lock file."""
        if self._fd is None:
            return

        try:
            metadata = {
                "owner_pid": os.getpid(),
                "owner_host": socket.gethostname(),
                "acquired_at": time.time(),
                "source": self._source,
            }
            content = json.dumps(metadata, indent=2).encode("utf-8")

            os.ftruncate(self._fd, 0)
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, content)
            os.fsync(self._fd)
        except OSError as e:
            logger.debug(f"[Lock] Metadata write failed (non-fatal): {e}")

    def _read_metadata_sync(self) -> Optional[dict]:
        """Read metadata from lock file (for stale warning)."""
        try:
            with open(self._lock_file, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    # =========================================================================
    # Async helpers
    # =========================================================================

    async def _ensure_lock_dir(self) -> None:
        """Ensure lock directory exists."""
        loop = asyncio.get_running_loop()

        def _mkdir():
            self._lock_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self._lock_dir, 0o700)
            except OSError:
                pass

        try:
            await loop.run_in_executor(None, _mkdir)
        except FileExistsError:
            pass
        except OSError as e:
            logger.error(f"[Lock] Failed to create lock dir {self._lock_dir}: {e}")
            raise

    async def _close_fd_async(self) -> None:
        """Close fd in executor."""
        if self._fd is not None:
            loop = asyncio.get_running_loop()
            fd = self._fd
            self._fd = None

            def _close():
                try:
                    os.close(fd)
                except OSError:
                    pass

            await loop.run_in_executor(None, _close)

    async def _log_stale_warning(self) -> None:
        """Log warning if lock appears stale."""
        loop = asyncio.get_running_loop()
        metadata = await loop.run_in_executor(None, self._read_metadata_sync)

        if metadata and "acquired_at" in metadata:
            held_for = time.time() - metadata["acquired_at"]
            if held_for > LOCK_STALE_WARNING_S:
                logger.warning(
                    f"[Lock] {self._lock_name} held for {held_for:.1f}s by "
                    f"PID {metadata.get('owner_pid')} ({metadata.get('source')}) - "
                    f"may be stale"
                )

    # =========================================================================
    # Context manager
    # =========================================================================

    async def __aenter__(self) -> bool:
        """Context manager entry. Returns True if acquired, False if timeout."""
        return await self.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - always releases lock."""
        await self.release()
