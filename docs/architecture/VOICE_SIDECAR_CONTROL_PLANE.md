# Voice Sidecar Control Plane (Go)

## Scope
- Go service supervises the Python voice worker lifecycle.
- Python keeps all model loading and inference logic.
- Contract surface is explicit and versioned under `/v1/*`.

## Contract
- `GET /v1/health`: sidecar + worker + gate state.
- `GET /v1/metrics`: Prometheus-format counters/gauges.
- `GET /v1/gates/heavy-load`: pressure admission gate.
- `POST /v1/control/start`: start worker (singleflight).
- `POST /v1/control/stop`: stop worker.
- `POST /v1/control/restart`: restart worker.
- `GET /v1/control/status`: worker + pressure state snapshot.

Transport options:
- `tcp` listener (default)
- `unix` listener (HTTP over Unix socket)

## Deterministic Startup Contract with `unified_supervisor.py`
Phase 2 (resources):
- Start sidecar if configured.
- Wait for sidecar health.
- Query heavy-load gate and set `JARVIS_BACKEND_MINIMAL=true` when closed.

Phase 3 (backend):
- Re-check heavy-load gate.
- Start worker via sidecar control endpoint when allowed.
- If sidecar is required and worker cannot start, fail startup deterministically.

Shutdown:
- Supervisor stops worker through sidecar contract.
- If supervisor spawned sidecar, it terminates sidecar process.

## Configuration (No Hardcoded Paths)
Primary env vars:
- `JARVIS_VOICE_SIDECAR_ENABLED`
- `JARVIS_VOICE_SIDECAR_REQUIRED`
- `JARVIS_VOICE_SIDECAR_MANAGE_WORKER`
- `JARVIS_VOICE_SIDECAR_COMMAND`
- `JARVIS_VOICE_SIDECAR_TRANSPORT` (`http|unix`)
- `JARVIS_VOICE_SIDECAR_BASE_URL`
- `JARVIS_VOICE_SIDECAR_SOCKET`
- `JARVIS_VOICE_SIDECAR_START_TIMEOUT`
- `JARVIS_VOICE_SIDECAR_HEALTH_TIMEOUT`
- `JARVIS_VOICE_SIDECAR_CONTROL_TIMEOUT`

Sidecar config file example:
- `config/voice_sidecar.example.yaml`

## Migration Plan
1. Build sidecar binary (`tools/voice_sidecar`) and deploy config file.
2. Run sidecar standalone; validate `/v1/health`, `/v1/metrics`, and gate behavior.
3. Enable supervisor contract in non-required mode:
   - `JARVIS_VOICE_SIDECAR_ENABLED=true`
   - `JARVIS_VOICE_SIDECAR_REQUIRED=false`
4. Validate startup ordering and worker lifecycle via `python3 unified_supervisor.py`.
5. Enable required mode after burn-in:
   - `JARVIS_VOICE_SIDECAR_REQUIRED=true`
6. Monitor pressure-gate closures and crash recoveries in metrics.

## Rollback Plan
1. Disable sidecar integration:
   - `JARVIS_VOICE_SIDECAR_ENABLED=false`
2. Restart supervisor (`python3 unified_supervisor.py`).
3. Verify legacy startup path and voice initialization remain operational.
4. Keep sidecar artifacts and config unchanged for fast re-enable.

## Test Coverage
Go tests in `tools/voice_sidecar/main_test.go` cover:
- crash recovery with exponential backoff
- memory-pressure fail-closed startup blocking
