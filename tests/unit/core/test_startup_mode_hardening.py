from unified_supervisor import (
    _is_cloud_probe_candidate,
    _resolve_local_startup_mode_on_cloud_unavailable,
)


def test_resolve_cloud_unavailable_low_ram_forces_sequential():
    mode = _resolve_local_startup_mode_on_cloud_unavailable(
        current_mode="local_full",
        available_gb=2.6,
    )
    assert mode == "sequential"


def test_cloud_probe_candidate_uses_desired_mode_when_effective_degraded():
    assert _is_cloud_probe_candidate(
        desired_mode="cloud_first",
        effective_mode="sequential",
        cloud_recovery_candidate=False,
    ) is True


def test_cloud_probe_candidate_flag_preserves_recovery_path():
    assert _is_cloud_probe_candidate(
        desired_mode="local_full",
        effective_mode="sequential",
        cloud_recovery_candidate=True,
    ) is True
