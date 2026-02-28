# Voice Sidecar Control Plane (Go Observer)

## Scope
- Go sidecar is **observer-only**.
- Python (`unified_supervisor.py` + backend runtime) remains source of truth for startup, recovery, and inference behavior.
- Go sidecar publishes observability + advisory safety signals; it does not run model/business logic and does not own worker lifecycle.

## Contract
Read-only endpoints exposed by sidecar:
- `GET /healthz`
- `GET /metrics`
- `GET /v1/health`
- `GET /v1/observer/state`
- `GET /v1/gates/heavy-load` (advisory gate signal)

No control endpoints (`start/stop/restart`) are exposed.

## Python Integration Contract
Sidecar polls Python every 1s (configurable):
- Unix IPC (`command=status`) against supervisor socket, or
- HTTP status endpoint.

Signals consumed from Python status payload:
- `startup_modes.desired_mode`
- `startup_modes.effective_mode`
- `memory_pressure_signal.status`
- `memory_pressure_signal.recovery_state`
- `memory_pressure_signal.local_circuit_state`

Advisories emitted by sidecar:
- `recovery_stuck`
- `mode_oscillation_risk`

## Deterministic Startup Contract with `unified_supervisor.py`
Phase 2 (resources):
- Start sidecar if configured.
- Wait for sidecar health.
- Read heavy-load advisory gate and set `Ironcliw_BACKEND_MINIMAL=true` when closed.

Phase 3 (backend):
- Re-check heavy-load advisory gate.
- Keep Python-owned startup/recovery behavior unchanged.

Shutdown:
- Supervisor only terminates sidecar process if it spawned it.
- No worker stop/start calls are made via sidecar.

## Configuration (No Hardcoded Runtime Paths)
Primary env vars:
- `Ironcliw_VOICE_SIDECAR_ENABLED`
- `Ironcliw_VOICE_SIDECAR_REQUIRED`
- `Ironcliw_VOICE_SIDECAR_COMMAND`
- `Ironcliw_VOICE_SIDECAR_TRANSPORT` (`http|unix`)
- `Ironcliw_VOICE_SIDECAR_BASE_URL`
- `Ironcliw_VOICE_SIDECAR_SOCKET`
- `Ironcliw_VOICE_SIDECAR_START_TIMEOUT`
- `Ironcliw_VOICE_SIDECAR_HEALTH_TIMEOUT`
- `Ironcliw_VOICE_SIDECAR_CONTROL_TIMEOUT`
- `Ironcliw_VOICE_SIDECAR_POLL_INTERVAL_MS`

Reference config:
- `config/voice_sidecar.example.yaml`

## Migration Plan
1. Build and run sidecar observer in standalone mode.
2. Validate `GET /v1/observer/state`, `GET /healthz`, and `GET /metrics`.
3. Enable supervisor integration in advisory-only mode:
   - `Ironcliw_VOICE_SIDECAR_ENABLED=true`
   - `Ironcliw_VOICE_SIDECAR_REQUIRED=false`
4. Validate startup with `python3 unified_supervisor.py` and verify mode/recovery signals are visible.
5. Enable required mode only after burn-in:
   - `Ironcliw_VOICE_SIDECAR_REQUIRED=true`

## Rollback Plan
1. Disable sidecar integration:
   - `Ironcliw_VOICE_SIDECAR_ENABLED=false`
2. Restart supervisor (`python3 unified_supervisor.py`).
3. Confirm native Python startup/recovery path remains healthy.
4. Keep sidecar binary/config for rapid re-enable.

## Test Coverage
- Go tests (`tools/voice_sidecar/main_test.go`):
  - status extraction
  - fail-closed gate behavior
  - oscillation advisory detection
  - unix IPC status polling
