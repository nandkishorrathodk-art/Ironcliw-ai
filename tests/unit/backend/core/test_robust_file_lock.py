# tests/unit/backend/core/test_robust_file_lock.py
"""Tests for RobustFileLock - OS-level file locking."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

# Will fail until we create the module
from backend.core.robust_file_lock import RobustFileLock


@pytest.fixture
def temp_lock_dir():
    """Create a temporary lock directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.asyncio
async def test_acquire_and_release(temp_lock_dir, monkeypatch):
    """Test basic lock acquire and release."""
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(temp_lock_dir))

    lock = RobustFileLock("test_lock", source="test")

    # Should acquire successfully
    async with lock as acquired:
        assert acquired is True
        # Lock file should exist
        assert (temp_lock_dir / "test_lock.lock").exists()

    # After release, should be able to acquire again
    async with lock as acquired:
        assert acquired is True


@pytest.mark.asyncio
async def test_lock_is_exclusive(temp_lock_dir, monkeypatch):
    """Test that lock is exclusive - second acquire of same name raises reentrancy error."""
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(temp_lock_dir))

    lock1 = RobustFileLock("exclusive_test", source="test1")
    lock2 = RobustFileLock("exclusive_test", source="test2")

    async with lock1 as acquired1:
        assert acquired1 is True

        # Second lock with same name should raise reentrancy error
        # (reentrancy guard protects against deadlock in same process)
        with pytest.raises(RuntimeError, match="already held"):
            await lock2.acquire(timeout_s=0.1)

    # After lock1 released, lock2 should succeed
    async with lock2 as acquired2:
        assert acquired2 is True


@pytest.mark.asyncio
async def test_different_locks_independent(temp_lock_dir, monkeypatch):
    """Test that differently named locks are independent."""
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(temp_lock_dir))

    lock1 = RobustFileLock("lock_a", source="test1")
    lock2 = RobustFileLock("lock_b", source="test2")

    async with lock1 as acquired1:
        assert acquired1 is True

        # Different named lock should succeed
        async with lock2 as acquired2:
            assert acquired2 is True
            # Both should be held simultaneously
            assert (temp_lock_dir / "lock_a.lock").exists()
            assert (temp_lock_dir / "lock_b.lock").exists()


@pytest.mark.asyncio
async def test_reentrancy_raises_error(temp_lock_dir, monkeypatch):
    """Test that re-acquiring same lock raises RuntimeError."""
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(temp_lock_dir))

    lock = RobustFileLock("reentrant_test", source="test")

    async with lock as acquired:
        assert acquired is True

        # Attempting to acquire again should raise
        with pytest.raises(RuntimeError, match="already held"):
            await lock.acquire()


@pytest.mark.asyncio
async def test_lock_creates_directory(temp_lock_dir, monkeypatch):
    """Test that lock creates directory if missing."""
    nested_dir = temp_lock_dir / "nested" / "locks"
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(nested_dir))

    lock = RobustFileLock("nested_test", source="test")

    assert not nested_dir.exists()

    async with lock as acquired:
        assert acquired is True
        assert nested_dir.exists()


@pytest.mark.asyncio
async def test_metadata_written(temp_lock_dir, monkeypatch):
    """Test that lock metadata is written for debugging."""
    import json

    monkeypatch.setenv("JARVIS_LOCK_DIR", str(temp_lock_dir))

    lock = RobustFileLock("metadata_test", source="jarvis")

    async with lock as acquired:
        assert acquired is True

        lock_file = temp_lock_dir / "metadata_test.lock"
        with open(lock_file) as f:
            metadata = json.load(f)

        assert metadata["owner_pid"] == os.getpid()
        assert metadata["source"] == "jarvis"
        assert "acquired_at" in metadata


@pytest.mark.asyncio
async def test_cross_process_exclusivity(temp_lock_dir, monkeypatch):
    """Test that lock is exclusive across processes using subprocess."""
    import subprocess
    import sys

    monkeypatch.setenv("JARVIS_LOCK_DIR", str(temp_lock_dir))

    lock = RobustFileLock("cross_process_test", source="parent")

    async with lock as acquired:
        assert acquired is True

        # Try to acquire the same lock from a subprocess
        # This should fail because we hold it
        subprocess_code = f'''
import asyncio
import sys
import os
os.environ["JARVIS_LOCK_DIR"] = "{temp_lock_dir}"
sys.path.insert(0, "{os.getcwd()}")

from backend.core.robust_file_lock import RobustFileLock

async def main():
    lock = RobustFileLock("cross_process_test", source="child")
    acquired = await lock.acquire(timeout_s=0.2)
    # Exit 0 if NOT acquired (expected), exit 1 if acquired (unexpected)
    sys.exit(0 if not acquired else 1)

asyncio.run(main())
'''
        result = subprocess.run(
            [sys.executable, "-c", subprocess_code],
            capture_output=True,
            timeout=5
        )

        # Subprocess should NOT have acquired the lock (exit code 0)
        assert result.returncode == 0, f"Subprocess acquired lock when it shouldn't have: {result.stderr.decode()}"

    # After we release, subprocess should be able to acquire
    subprocess_code_after = f'''
import asyncio
import sys
import os
os.environ["JARVIS_LOCK_DIR"] = "{temp_lock_dir}"
sys.path.insert(0, "{os.getcwd()}")

from backend.core.robust_file_lock import RobustFileLock

async def main():
    lock = RobustFileLock("cross_process_test", source="child_after")
    acquired = await lock.acquire(timeout_s=1.0)
    await lock.release()
    # Exit 0 if acquired (expected), exit 1 if not (unexpected)
    sys.exit(0 if acquired else 1)

asyncio.run(main())
'''
    result_after = subprocess.run(
        [sys.executable, "-c", subprocess_code_after],
        capture_output=True,
        timeout=5
    )

    assert result_after.returncode == 0, f"Subprocess failed to acquire lock after release: {result_after.stderr.decode()}"


@pytest.mark.asyncio
async def test_lock_auto_releases_on_process_death(temp_lock_dir, monkeypatch):
    """Test that lock is released when holding process dies."""
    import subprocess
    import sys
    import signal

    monkeypatch.setenv("JARVIS_LOCK_DIR", str(temp_lock_dir))

    # Start a subprocess that holds the lock and waits to be killed
    subprocess_code = f'''
import asyncio
import sys
import os
import time
os.environ["JARVIS_LOCK_DIR"] = "{temp_lock_dir}"
sys.path.insert(0, "{os.getcwd()}")

from backend.core.robust_file_lock import RobustFileLock

async def main():
    lock = RobustFileLock("auto_release_test", source="holder")
    acquired = await lock.acquire(timeout_s=1.0)
    if not acquired:
        sys.exit(1)
    # Hold the lock and wait to be killed
    print("ACQUIRED", flush=True)
    time.sleep(60)

asyncio.run(main())
'''
    proc = subprocess.Popen(
        [sys.executable, "-c", subprocess_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for the subprocess to acquire the lock
    try:
        line = proc.stdout.readline().decode().strip()
        assert line == "ACQUIRED", f"Subprocess did not acquire lock: {proc.stderr.read().decode()}"

        # Verify we can't acquire while subprocess holds it
        lock = RobustFileLock("auto_release_test", source="parent")
        acquired = await lock.acquire(timeout_s=0.2)
        assert not acquired, "Should not acquire lock held by subprocess"

        # Kill the subprocess
        proc.terminate()
        proc.wait(timeout=2)

        # Now we should be able to acquire (lock auto-released)
        acquired = await lock.acquire(timeout_s=1.0)
        assert acquired, "Should acquire lock after subprocess death"
        await lock.release()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.mark.asyncio
async def test_multiple_release_safe(temp_lock_dir, monkeypatch):
    """Test that calling release multiple times is safe."""
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(temp_lock_dir))

    lock = RobustFileLock("multi_release_test", source="test")

    async with lock as acquired:
        assert acquired is True

    # Additional releases should not raise
    await lock.release()
    await lock.release()
    await lock.release()


@pytest.mark.asyncio
async def test_release_without_acquire_safe(temp_lock_dir, monkeypatch):
    """Test that release without acquire is safe."""
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(temp_lock_dir))

    lock = RobustFileLock("no_acquire_test", source="test")

    # Should not raise
    await lock.release()
    await lock.release()
