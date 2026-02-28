# Ironcliw Voice Sidecar (Go Observer)

Read-only control-plane observer for Python health, startup mode signals, and safety advisories.

It does **not** start/stop/restart Python workers and does not own model/runtime logic.

## Build
```bash
cd tools/voice_sidecar
go mod tidy
go build -o voice-sidecar .
```

## Run
```bash
./voice-sidecar --config ../../config/voice_sidecar.example.yaml
```

## Endpoints
- `GET /healthz`
- `GET /metrics`
- `GET /v1/health`
- `GET /v1/observer/state`
- `GET /v1/gates/heavy-load`

## Contracts
- Polls Python status over either:
  - Supervisor Unix IPC socket (`command=status`), or
  - HTTP status endpoint (configurable)
- Publishes advisory-only signals:
  - `recovery_stuck`
  - `mode_oscillation_risk`

## Tests
```bash
cd tools/voice_sidecar
go test ./...
```
