"""Constructor contract tests for resilience DistributedLock."""

import pytest

from backend.core.resilience.distributed_lock import DistributedLock


def test_distributed_lock_accepts_canonical_name() -> None:
    lock = DistributedLock(name="vm_provisioning")
    assert lock.name == "vm_provisioning"
    assert lock._lock_key == "jarvis:lock:vm_provisioning"


def test_distributed_lock_accepts_legacy_lock_name_alias() -> None:
    lock = DistributedLock(lock_name="vm_provisioning")
    assert lock.name == "vm_provisioning"
    assert lock._lock_key == "jarvis:lock:vm_provisioning"


def test_distributed_lock_rejects_conflicting_identifiers() -> None:
    with pytest.raises(ValueError, match="conflicting identifiers"):
        DistributedLock(name="one", lock_name="two")


def test_distributed_lock_requires_identifier() -> None:
    with pytest.raises(TypeError, match="requires `name`"):
        DistributedLock()
