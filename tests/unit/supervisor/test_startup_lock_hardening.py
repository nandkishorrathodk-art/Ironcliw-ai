from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest


@pytest.fixture
def supervisor_module(monkeypatch, tmp_path: Path):
    import unified_supervisor as us

    isolated_locks = tmp_path / "locks"
    isolated_locks.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "LOCKS_DIR", isolated_locks)
    return us


def test_allocate_ephemeral_port_returns_bindable_port(supervisor_module):
    us = supervisor_module

    # Ephemeral port allocation can race with unrelated processes, so retry a
    # few times before failing the test.
    for _ in range(5):
        port = us._allocate_ephemeral_port("127.0.0.1")
        assert isinstance(port, int) and port > 0
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", port))
            return
        except OSError:
            continue

    pytest.fail("Could not bind any allocated ephemeral port")


def test_startup_lock_reclaims_stale_lock_without_force(supervisor_module):
    us = supervisor_module
    lock = us.StartupLock(lock_name="stale-reclaim")

    stale_payload = {
        "pid": 999999,
        "acquired_at": "1970-01-01T00:00:00",
        "kernel_version": "test",
        "hostname": "test-host",
    }
    lock.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock.lock_path.write_text(json.dumps(stale_payload))

    assert lock.acquire(force=False) is True
    acquired_payload = json.loads(lock.lock_path.read_text())
    assert acquired_payload.get("pid") == os.getpid()

    lock.release()


def test_startup_lock_rejects_live_lock_steal_by_default(supervisor_module, monkeypatch):
    us = supervisor_module
    owner = us.StartupLock(lock_name="live-holder")
    contender = us.StartupLock(lock_name="live-holder")
    contender.pid = owner.pid + 1

    # Force the lock liveness predicate to treat owner's PID as alive for this test.
    monkeypatch.setattr(
        us.StartupLock,
        "_is_process_alive",
        staticmethod(lambda pid: pid == owner.pid),
    )

    assert owner.acquire(force=False) is True
    try:
        assert contender.acquire(force=False) is False
        assert contender.acquire(force=True) is False
    finally:
        owner.release()
