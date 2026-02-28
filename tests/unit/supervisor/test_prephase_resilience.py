from __future__ import annotations


def test_cloud_only_fallback_never_escalates_to_local_full(monkeypatch):
    import unified_supervisor as us

    monkeypatch.setenv("Ironcliw_CRITICAL_THRESHOLD_GB", "2.0")
    monkeypatch.setenv("Ironcliw_OPTIMIZE_THRESHOLD_GB", "4.0")
    monkeypatch.setenv("Ironcliw_PLANNED_ML_GB", "4.6")
    monkeypatch.setenv("Ironcliw_MINIMAL_THRESHOLD_GB", "1.0")

    assert us._resolve_local_startup_mode_on_cloud_unavailable("cloud_only", 16.0) == "local_optimized"
    assert us._resolve_local_startup_mode_on_cloud_unavailable("cloud_only", 0.4) == "minimal"
    assert us._resolve_local_startup_mode_on_cloud_unavailable("cloud_only", None) == "sequential"


def test_cloud_first_fallback_is_memory_aware(monkeypatch):
    import unified_supervisor as us

    monkeypatch.setenv("Ironcliw_CRITICAL_THRESHOLD_GB", "2.0")
    monkeypatch.setenv("Ironcliw_OPTIMIZE_THRESHOLD_GB", "4.0")
    monkeypatch.setenv("Ironcliw_PLANNED_ML_GB", "4.6")
    monkeypatch.setenv("Ironcliw_MINIMAL_THRESHOLD_GB", "1.0")

    assert us._resolve_local_startup_mode_on_cloud_unavailable("cloud_first", 16.0) == "local_full"
    assert us._resolve_local_startup_mode_on_cloud_unavailable("cloud_first", 6.0) == "sequential"
    assert us._resolve_local_startup_mode_on_cloud_unavailable("cloud_first", None) == "local_optimized"


def test_startup_watchdog_invalid_env_values_fall_back_to_defaults(monkeypatch):
    import unified_supervisor as us

    class _Logger:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

        def debug(self, *_args, **_kwargs):
            pass

    monkeypatch.setenv("Ironcliw_DMS_ENABLED", "not-a-bool")
    monkeypatch.setenv("Ironcliw_DMS_STALL_THRESHOLD", "bad-float")
    monkeypatch.setenv("Ironcliw_DMS_CHECK_INTERVAL", "bad-float")
    monkeypatch.setenv("Ironcliw_DMS_RECOVERY_MODE", "unknown-mode")
    monkeypatch.setenv("Ironcliw_DMS_ESCALATION_COOLDOWN", "NaN???")
    monkeypatch.setenv("Ironcliw_DMS_STALL_ESCALATION_COOLDOWN", "NaN???")
    monkeypatch.setenv("Ironcliw_DMS_PROGRESS_WINDOW", "NaN???")
    monkeypatch.setenv("GCP_VM_STARTUP_TIMEOUT", "NaN???")

    watchdog = us.StartupWatchdog(logger=_Logger())

    assert watchdog.enabled is True
    assert watchdog._stall_threshold == 60.0
    assert watchdog._check_interval == 5.0
    assert watchdog._recovery_mode == "graduated"
    assert watchdog._escalation_cooldown == 60.0
    assert watchdog._stall_escalation_cooldown == 15.0
    assert watchdog._progress_advancing_window == 120.0
