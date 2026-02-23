from __future__ import annotations

from backend.agi_os.agi_os_coordinator import AGIOSCoordinator, AGIOSState, ComponentStatus


def test_determine_health_state_ignores_legacy_phase_component_entries():
    coordinator = AGIOSCoordinator()

    coordinator._component_status = {
        "voice": ComponentStatus(name="voice", available=True),
        "approval": ComponentStatus(name="approval", available=True),
        "events": ComponentStatus(name="events", available=True),
        "orchestrator": ComponentStatus(name="orchestrator", available=True),
        # Legacy/stale startup phase marker should not impact runtime health.
        "phase_components_connected": ComponentStatus(
            name="phase_components_connected",
            available=False,
            healthy=False,
            error="timeout (10.0s)",
        ),
    }

    assert coordinator._determine_health_state() == AGIOSState.ONLINE


async def test_start_resets_status_maps_and_tracks_phase_results(monkeypatch):
    coordinator = AGIOSCoordinator()
    coordinator._component_status["stale_component"] = ComponentStatus(
        name="stale_component",
        available=False,
        healthy=False,
        error="stale",
    )
    coordinator._phase_status["phase_old"] = ComponentStatus(
        name="phase_old",
        available=False,
        healthy=False,
        error="stale",
    )

    async def _init_components() -> None:
        coordinator._component_status["voice"] = ComponentStatus(name="voice", available=True)
        coordinator._component_status["approval"] = ComponentStatus(name="approval", available=True)
        coordinator._component_status["events"] = ComponentStatus(name="events", available=True)
        coordinator._component_status["orchestrator"] = ComponentStatus(
            name="orchestrator", available=True
        )

    async def _noop() -> None:
        return None

    monkeypatch.setattr(coordinator, "_init_agi_os_components", _init_components)
    monkeypatch.setattr(coordinator, "_init_intelligence_systems", _noop)
    monkeypatch.setattr(coordinator, "_init_neural_mesh", _noop)
    monkeypatch.setattr(coordinator, "_init_hybrid_orchestrator", _noop)
    monkeypatch.setattr(coordinator, "_init_screen_analyzer", _noop)
    monkeypatch.setattr(coordinator, "_connect_components", _noop)

    await coordinator.start()

    assert "stale_component" not in coordinator._component_status
    assert "phase_old" not in coordinator._phase_status
    assert "phase_components_connected" in coordinator._phase_status
    assert coordinator._phase_status["phase_components_connected"].available is True
    assert coordinator.state == AGIOSState.ONLINE


def test_get_status_includes_phases_separately():
    coordinator = AGIOSCoordinator()
    coordinator._component_status["voice"] = ComponentStatus(name="voice", available=True)
    coordinator._phase_status["phase_components_connected"] = ComponentStatus(
        name="phase_components_connected",
        available=False,
        healthy=False,
        error="timeout",
    )

    status = coordinator.get_status()
    assert "phases" in status
    assert "phase_components_connected" in status["phases"]
    assert "phase_components_connected" not in status["components"]
