# JARVIS Voice Sidecar (Go)

Control-plane supervisor for Python voice worker reliability.

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

Or env-only:
```bash
JARVIS_VOICE_SIDECAR_WORKER_COMMAND="python3 -m backend.voice.voice_worker_service" \
JARVIS_VOICE_SIDECAR_WORKER_AUTOSTART=true \
./voice-sidecar
```

## Endpoints
- `GET /v1/health`
- `GET /v1/metrics`
- `GET /v1/gates/heavy-load`
- `POST /v1/control/start`
- `POST /v1/control/stop`
- `POST /v1/control/restart`
- `GET /v1/control/status`

## Tests
```bash
cd tools/voice_sidecar
go test ./...
```
