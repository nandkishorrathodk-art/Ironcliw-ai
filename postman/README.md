# JARVIS Voice Biometric Authentication - Postman Collections

[![Postman Tests](https://github.com/YOUR_USERNAME/JARVIS-AI-Agent/actions/workflows/postman-api-tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/JARVIS-AI-Agent/actions/workflows/postman-api-tests.yml)

Complete Postman collections for testing the JARVIS Voice Biometric Authentication system with advanced ML-based confidence scoring, anti-spoofing detection, and multi-factor fusion.

## Collections Overview

| Collection | Description | Endpoints |
|------------|-------------|-----------|
| **JARVIS Voice Auth Intelligence (Advanced ML)** | Advanced voice authentication with AAM-Softmax, Platt/Isotonic calibration, and adaptive thresholds | 25+ |
| **JARVIS Voice Unlock Flow (Sequential)** | End-to-end voice unlock authentication pipeline with PRD v2.0 features | 12 |
| **JARVIS AI Agent API** | Complete JARVIS system API including health, screen control, vision, and more | 50+ |

## Quick Start

### Prerequisites

1. **JARVIS Backend Running**
   ```bash
   cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent
   ./venv/bin/python -m backend.main
   # or
   ./start_system.py
   ```

2. **Postman Desktop App** (recommended) or **Newman CLI**
   ```bash
   npm install -g newman newman-reporter-htmlextra
   ```

### Import Collections

#### Option 1: Import in Postman Desktop

1. Open Postman
2. Click **Import** in the sidebar
3. Drag and drop all files from `postman/collections/`
4. Import the environment from `postman/environments/`

#### Option 2: Run with Newman CLI

```bash
# Run all collections
npm run postman:test

# Run specific collection
newman run postman/collections/JARVIS_Voice_Auth_Intelligence_Collection.postman_collection.json \
  -e postman/environments/JARVIS_Environment.postman_environment.json

# Run with HTML report
newman run postman/collections/JARVIS_Voice_Unlock_Flow_Collection.postman_collection.json \
  -e postman/environments/JARVIS_Environment.postman_environment.json \
  --reporters cli,htmlextra \
  --reporter-htmlextra-export ./reports/voice-unlock-report.html
```

## Directory Structure

```
postman/
├── collections/
│   ├── JARVIS_Voice_Auth_Intelligence_Collection.postman_collection.json
│   ├── JARVIS_Voice_Unlock_Flow_Collection.postman_collection.json
│   └── JARVIS_API_Collection.postman_collection.json
├── environments/
│   └── JARVIS_Environment.postman_environment.json
├── flows/
│   └── README.md                    # Flow documentation
├── newman.config.json               # Newman CLI configuration
└── README.md                        # This file
```

## Collections Detail

### 1. JARVIS Voice Auth Intelligence (Advanced ML)

Advanced ML-powered voice authentication with comprehensive anti-spoofing.

**Key Features:**
- AAM-Softmax + Center Loss + Triplet Loss fine-tuning
- Platt Scaling & Isotonic Regression calibration
- Adaptive thresholds: base=0.90, high=0.95, critical=0.98
- Comprehensive anti-spoofing (replay, synthesis, voice conversion)
- LangGraph adaptive reasoning
- Langfuse audit trail
- Helicone-style caching

**Folders:**
| Folder | Purpose |
|--------|---------|
| 1. System Status | Health checks, component status |
| 2. Calibration System | Score calibration, adaptive thresholds |
| 3. Fine-Tuning System | AAM-Softmax training endpoints |
| 4. Anti-Spoofing | Replay, deepfake, voice conversion detection |
| 5. Authentication Simulation | Test various scenarios |
| 6. Audit Trail (Langfuse) | Session management, traces |
| 7. Multi-Factor Fusion | Behavioral + voice + context fusion |
| 8. Full Pipeline Tests | End-to-end calibrated authentication |

### 2. JARVIS Voice Unlock Flow (Sequential)

Complete voice unlock authentication pipeline designed for Collection Runner.

**Flow Steps:**
```
0. Backend Check
   ↓
1. System Health Check
   ↓
2. Start Audit Session (Langfuse)
   ↓
3. Comprehensive Anti-Spoofing Check
   ↓ (if clean)
4. Calibrated Voice Authentication
   ↓
5. Add Calibration Sample (Training)
   ↓ (if borderline)
6. Multi-Factor Fusion
   ↓
7. Unlock Screen
   ↓
8. JARVIS Success Feedback
   ↓
9. End Audit Session

Error Handlers:
- System Unavailable
- Security Alert (spoofing detected)
- Authentication Failed
- Fusion Failed - Challenge Required
```

### 3. JARVIS AI Agent API

Complete API coverage for the JARVIS system.

**Categories:**
- Health & Status
- Startup
- Screen Control
- Voice Biometric Authentication
- Wake Word Detection
- Vision Intelligence
- Display Monitoring
- Memory Management
- Self-Healing System
- Hybrid Cloud Integration

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `base_url` | JARVIS backend URL | `http://localhost:8010` |
| `user_id` | User identifier | `derek` |
| `user_name` | User display name | `Derek Russell` |
| `speaker_name` | Speaker for voice auth | `Derek` |
| `confidence_threshold` | Base confidence threshold | `0.85` |
| `security_level` | Security level (base/high/critical) | `base` |
| `websocket_url` | WebSocket URL | `ws://localhost:8000` |
| `voice_unlock_ws` | Voice unlock WebSocket | `ws://localhost:8000/api/voice-unlock/ws/authenticate` |

## Running Tests

### Local Development

```bash
# Start backend first
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent
./venv/bin/python start_system.py

# In another terminal, run tests
newman run postman/collections/JARVIS_Voice_Auth_Intelligence_Collection.postman_collection.json \
  --env-var "base_url=http://localhost:8010"
```

### CI/CD (GitHub Actions)

Tests run automatically on:
- Push to `main` or `develop` branches (if postman/ or backend/ changed)
- Pull requests to `main`
- Manual trigger via workflow dispatch

View results: **Actions** tab → **JARVIS Postman API Tests**

### Running with Docker

```bash
# Run tests in Docker container
docker run -t postman/newman:alpine \
  run "https://raw.githubusercontent.com/YOUR_USERNAME/JARVIS-AI-Agent/main/postman/collections/JARVIS_Voice_Auth_Intelligence_Collection.postman_collection.json" \
  --env-var "base_url=http://host.docker.internal:8010"
```

## Test Scenarios

### Anti-Spoofing Scenarios

| Scenario | Expected Result |
|----------|-----------------|
| `legitimate` | ALLOW - authentic voice |
| `replay_attack` | DENY - replay detected |
| `deepfake` | DENY - synthesis detected |
| `voice_conversion` | DENY - conversion detected |
| `mixed_attack` | DENY - multiple threats |

### Authentication Scenarios

| Scenario | Confidence | Result |
|----------|------------|--------|
| `success` | 90%+ | Direct unlock |
| `borderline` | 75-85% | Multi-factor fusion required |
| `sick_voice` | 68% | Challenge question |
| `noisy_environment` | Variable | Adaptive retry |
| `unknown_speaker` | <50% | Denied |

## API Endpoints Reference

### Voice Auth Intelligence

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/voice-auth-intelligence/health` | GET | System health |
| `/api/voice-auth-intelligence/status` | GET | Component status |
| `/api/voice-auth-intelligence/calibration/status` | GET | Calibration progress |
| `/api/voice-auth-intelligence/calibration/thresholds` | GET | Current thresholds |
| `/api/voice-auth-intelligence/calibration/authenticate` | POST | Calibrated auth |
| `/api/voice-auth-intelligence/calibration/add-sample` | POST | Add training sample |
| `/api/voice-auth-intelligence/anti-spoofing/comprehensive` | POST | Full anti-spoofing |
| `/api/voice-auth-intelligence/anti-spoofing/detect-synthesis` | POST | Deepfake detection |
| `/api/voice-auth-intelligence/fusion/calculate` | POST | Multi-factor fusion |
| `/api/voice-auth-intelligence/audit/session/start` | POST | Start audit trail |
| `/api/voice-auth-intelligence/audit/session/end` | POST | End audit trail |
| `/api/voice-auth-intelligence/fine-tuning/summary` | GET | Training progress |
| `/api/voice-auth-intelligence/fine-tuning/train-step` | POST | Train step |

### Screen Control

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/screen/unlock` | POST | Unlock screen |
| `/api/screen/lock` | POST | Lock screen |
| `/api/screen/status` | GET | Screen state |

### Voice Feedback

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/voice/jarvis/speak` | POST | TTS output |

## Contributing

### Adding New Requests

1. Open collection in Postman
2. Add request to appropriate folder
3. Add pre-request scripts if needed
4. Add test scripts for validation
5. Export collection (Collection v2.1 format)
6. Commit changes

### Best Practices

- Use collection variables instead of hardcoded values
- Add descriptions to all requests
- Include test scripts for validation
- Document expected responses
- Never commit sensitive values (API keys, passwords)

## Syncing with Postman Cloud

If using Postman cloud sync:

1. Create a Postman team workspace: "JARVIS Voice Auth"
2. Import collections from this repo
3. Enable GitHub integration in Postman
4. Set up two-way sync (optional)

## Troubleshooting

### Backend Not Running

```
Error: connect ECONNREFUSED 127.0.0.1:8010
```

**Solution:** Start the JARVIS backend:
```bash
cd /Users/djrussell23/Documents/repos/JARVIS-AI-Agent
./venv/bin/python start_system.py
```

### Collection Variable Not Found

```
Error: Could not find variable {{base_url}}
```

**Solution:** Import the environment file or set variables manually:
```bash
newman run collection.json --env-var "base_url=http://localhost:8010"
```

### Timeout Errors

**Solution:** Increase timeout in Newman:
```bash
newman run collection.json --timeout-request 30000
```

## Security Notes

- Environment files in this repo contain NO secrets
- API keys and passwords should be stored in Postman Vault
- Use `{{vault:secret_name}}` for sensitive values
- GitHub Actions uses repository secrets for sensitive data

## License

Part of the JARVIS AI Agent project.
