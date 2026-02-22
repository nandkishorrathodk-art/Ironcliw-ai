"""
RobustFileLock - Windows Implementation using msvcrt
====================================================

Windows-compatible file locking using msvcrt.locking().
Drop-in replacement for robust_file_lock.py on Windows.

Guarantees:
- ATOMIC: Lock acquisition is atomic
- EPHEMERAL: Lock automatically released on process death
- NON-BLOCKING EVENT LOOP: All blocking I/O runs in executor
- CROSS-PROCESS: Works across all processes on Windows
- REENTRANT: Same async task can re-acquire same lock safely

Implementation Notes:
- Uses msvcrt.locking() for Windows file locking
- Compatible with the POSIX version API
- Automatically handles .lock file creation in %USERPROFILE%\\.jarvis\\locks
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import time
import msvcrt
from pathlib import Path
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Platform check
# =============================================================================

if sys.platform != "win32":
    raise RuntimeError(
        "RobustFileLockWindows is Windows-only. "
        "Use RobustFileLock for POSIX systems (Linux, macOS)."
    )

# =============================================================================
# Configuration
# =============================================================================

# Default to %USERPROFILE%\.jarvis\locks on Windows
default_lock_dir = os.path.join(
    os.environ.get("USERPROFILE", "C:\\"),
    ".jarvis", "cross_repo", "locks"
)

LOCK_DIR_RAW = os.environ.get("JARVIS_LOCK_DIR", default_lock_dir)
LOCK_ACQUIRE_TIMEOUT_S = float(os.environ.get("LOCK_ACQUIRE_TIMEOUT_S", "10.0"))
LOCK_POLL_INTERVAL_S = float(os.environ.get("LOCK_POLL_INTERVAL_S", "0.05"))
LOCK_STALE_WARNING_S = float(os.environ.get("LOCK_STALE_WARNING_S", "30.0"))

# =============================================================================
# Reentrant lock tracking
# =============================================================================

# Maps lock_name -> (task_id, reentrance_count, file_handle)
_held_locks: Dict[str, Tuple[int, int, Optional[object]]] = {}
_held_locks_lock: Optional[asyncio.Lock] = None


def _get_held_locks_lock() -> asyncio.Lock:
    """Lazy initialization of held locks lock."""
    global _held_locks_lock
    if _held_locks_lock is None:
        _held_locks_lock = asyncio.Lock()
    return _held_locks_lock


def _get_task_id() -> int:
    """Get unique ID for current async task."""
    try:
        task = asyncio.current_task()
        return id(task) if task else id(asyncio.get_running_loop())
    except RuntimeError:
        import threading
        return threading.current_thread().ident or 0


class RobustFileLock:
    """
    OS-level file lock using msvcrt.locking() on Windows.
    All blocking I/O runs in executor to avoid blocking the event loop.
    
    Supports reentrancy within the same async task.
    """

    def __init__(self, lock_name: str, source: str = "jarvis"):
        """
        Initialize lock.

        Args:
            lock_name: Unique name for this lock (e.g., "vbia_state")
            source: Identifier for the process holding the lock (for debugging)
        """
        self.lock_name = lock_name
        self.source = source
        self.lock_dir: Optional[Path] = None
        self.lock_path: Optional[Path] = None
        self._file_handle: Optional[object] = None
        self._is_acquired = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_lock_path(self) -> Path:
        """Ensure lock directory exists and return lock file path."""
        if self.lock_path is not None:
            return self.lock_path
        
        # Expand home directory
        lock_dir_expanded = os.path.expanduser(LOCK_DIR_RAW)
        self.lock_dir = Path(lock_dir_expanded)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        
        # Create lock file path
        self.lock_path = self.lock_dir / f"{self.lock_name}.lock"
        
        logger.debug(f"Lock path: {self.lock_path}")
        return self.lock_path

    def _write_lock_info(self, fh) -> None:
        """Write lock metadata to file."""
        try:
            lock_info = {
                "lock_name": self.lock_name,
                "source": self.source,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "acquired_at": time.time(),
            }
            fh.seek(0)
            fh.write(json.dumps(lock_info, indent=2))
            fh.flush()
        except Exception as e:
            logger.warning(f"Failed to write lock info: {e}")

    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire the lock.

        Args:
            timeout: Maximum time to wait in seconds (default: LOCK_ACQUIRE_TIMEOUT_S)

        Returns:
            bool: True if lock was acquired

        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        if timeout is None:
            timeout = LOCK_ACQUIRE_TIMEOUT_S

        self._loop = asyncio.get_running_loop()
        lock_path = self._ensure_lock_path()
        task_id = _get_task_id()

        # Check for reentrancy
        async with _get_held_locks_lock():
            if self.lock_name in _held_locks:
                holder_task_id, count, fh = _held_locks[self.lock_name]
                if holder_task_id == task_id:
                    # Reentrant acquisition - increment count
                    _held_locks[self.lock_name] = (task_id, count + 1, fh)
                    logger.debug(
                        f"Reentrant lock '{self.lock_name}' (count={count + 1})"
                    )
                    self._is_acquired = True
                    self._file_handle = fh
                    return True

        # Non-reentrant path - acquire new lock
        start_time = time.time()
        
        while True:
            try:
                # Try to acquire lock
                fh = await self._loop.run_in_executor(
                    None, self._try_acquire_lock, lock_path
                )
                
                if fh is not None:
                    # Success - record lock
                    async with _get_held_locks_lock():
                        _held_locks[self.lock_name] = (task_id, 1, fh)
                    
                    self._file_handle = fh
                    self._is_acquired = True
                    
                    logger.debug(
                        f"Acquired lock '{self.lock_name}' (pid={os.getpid()})"
                    )
                    return True
                
                # Lock is held by another process - check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Could not acquire lock '{self.lock_name}' "
                        f"within {timeout:.1f}s"
                    )
                
                # Wait and retry
                await asyncio.sleep(LOCK_POLL_INTERVAL_S)
                
            except TimeoutError:
                raise
            except Exception as e:
                logger.error(f"Error acquiring lock '{self.lock_name}': {e}")
                raise

    def _try_acquire_lock(self, lock_path: Path):
        """
        Try to acquire lock (blocking call, run in executor).
        
        Returns:
            file handle if successful, None if lock is held
        """
        try:
            # Open/create lock file
            fh = open(lock_path, "w+")
            
            # Try to lock it (non-blocking)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except IOError:
                # Lock is held by another process
                fh.close()
                return None
            
            # Write lock info
            self._write_lock_info(fh)
            return fh
            
        except Exception as e:
            logger.debug(f"Lock attempt failed: {e}")
            return None

    async def release(self) -> None:
        """Release the lock."""
        if not self._is_acquired:
            logger.warning(f"Attempted to release unacquired lock '{self.lock_name}'")
            return

        task_id = _get_task_id()

        async with _get_held_locks_lock():
            if self.lock_name not in _held_locks:
                logger.warning(f"Lock '{self.lock_name}' not in held locks")
                self._is_acquired = False
                return

            holder_task_id, count, fh = _held_locks[self.lock_name]

            if holder_task_id != task_id:
                logger.error(
                    f"Task {task_id} tried to release lock '{self.lock_name}' "
                    f"held by task {holder_task_id}"
                )
                return

            # Decrement reentrance count
            if count > 1:
                _held_locks[self.lock_name] = (task_id, count - 1, fh)
                logger.debug(
                    f"Decremented reentrant lock '{self.lock_name}' (count={count - 1})"
                )
                return

            # Count is 1 - truly release the lock
            del _held_locks[self.lock_name]

        # Release the actual file lock
        if self._file_handle is not None:
            await self._loop.run_in_executor(
                None, self._release_lock, self._file_handle
            )

        self._file_handle = None
        self._is_acquired = False
        
        logger.debug(f"Released lock '{self.lock_name}'")

    def _release_lock(self, fh) -> None:
        """Release lock file (blocking call, run in executor)."""
        try:
            # Unlock
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            fh.close()
        except Exception as e:
            logger.warning(f"Error releasing lock: {e}")

    async def __aenter__(self):
        """Context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.release()
        return False

    @property
    def is_acquired(self) -> bool:
        """Check if lock is currently held by this instance."""
        return self._is_acquired


# Compatibility exports
__all__ = ["RobustFileLock"]
