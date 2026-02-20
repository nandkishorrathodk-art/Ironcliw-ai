"""Path-resolution tests for supervisor singleton lock runtime directories."""

import importlib
import os
import sys
from pathlib import Path


def _import_fresh_supervisor_singleton():
    """Import supervisor_singleton with a clean module state."""
    sys.modules.pop("backend.core.supervisor_singleton", None)
    return importlib.import_module("backend.core.supervisor_singleton")


def test_lock_dir_uses_explicit_env_when_writable(monkeypatch, tmp_path):
    """Writable JARVIS_LOCK_DIR should be selected without fallback."""
    lock_dir = tmp_path / "locks"
    monkeypatch.setenv("JARVIS_LOCK_DIR", str(lock_dir))
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "home"))

    mod = _import_fresh_supervisor_singleton()

    assert mod.LOCK_DIR == lock_dir
    assert mod.LOCK_DIR.exists()
    assert os.environ["JARVIS_LOCK_DIR"] == str(lock_dir)


def test_lock_dir_falls_back_when_explicit_env_is_not_writable(monkeypatch, tmp_path):
    """Non-writable or invalid lock path should trigger deterministic fallback."""
    invalid_lock_dir = "/dev/null/not-a-dir"
    monkeypatch.setenv("JARVIS_LOCK_DIR", invalid_lock_dir)
    monkeypatch.setenv("JARVIS_HOME", str(tmp_path / "home"))

    mod = _import_fresh_supervisor_singleton()

    assert str(mod.LOCK_DIR) != invalid_lock_dir
    assert mod.LOCK_DIR.exists()
    assert os.environ["JARVIS_LOCK_DIR"] == str(mod.LOCK_DIR)

