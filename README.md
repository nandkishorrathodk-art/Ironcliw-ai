<div align="center">

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

# ⚡ IRONCLIW-AI · JARVIS
### *Just A Rather Very Intelligent System*

**The world's most advanced personal AI agent — now fully on Windows.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://python.org)
[![Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-0078D4?logo=windows)](https://github.com/nandkishorrathodk-art/Ironcliw-ai)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React%2018-61DAFB?logo=react)](https://reactjs.org)
[![Claude AI](https://img.shields.io/badge/AI-Claude%20%7C%20Fireworks-FF6B00?logo=anthropic)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Phase](https://img.shields.io/badge/Port%20Phase-11%20Complete-success)](WINDOWS_PORT_BLUEPRINT.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>

---

## 🤖 What Is This?

**Ironcliw-AI** is a Windows port of the [drussell23/JARVIS](https://github.com/drussell23/JARVIS) personal AI agent — a self-hosted, voice-activated autonomous assistant inspired by Iron Man's J.A.R.V.I.S.

It combines:
- 🧠 **Large Language Models** (Claude 3.5, Fireworks AI) for reasoning
- 🎤 **Voice control** (Whisper STT + Microsoft Neural TTS `en-GB-RyanNeural`)
- 👁️ **Vision** (screen capture + Claude Vision) for seeing your desktop
- 🤖 **Autonomous automation** (Ghost Hands browser/keyboard control)
- ☁️ **Hybrid cloud** (GCP auto-routing when RAM is high)
- 🔐 **Voice biometric unlock** (ECAPA-TDNN speaker verification)

---

## 🖥️ Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| **Windows 10/11** | ✅ **Fully Supported** | Primary development target |
| macOS | ⚠️ Upstream | See [drussell23/JARVIS](https://github.com/drussell23/JARVIS) |
| Linux | 🔧 Partial | PAL layer compatible |

---

## ✨ Features

### Core Intelligence
- 🧠 **Multi-LLM routing** — Claude 3.5 Sonnet + Fireworks AI (`accounts/fireworks/models/llama-v3p1-70b-instruct`)
- 💬 **Natural conversation** with long-term memory (SQLite + ChromaDB)
- 🎯 **Goal inference** — JARVIS figures out what you want before you finish asking
- 🔮 **Situational Awareness Intelligence (SAI)** — understands context (emergency, routine, suspicious)

### Voice System
- 🎤 **Wake word**: "Hey JARVIS" — instant activation
- 🗣️ **Neural TTS**: Microsoft `en-GB-RyanNeural` via edge-tts (sounds human, not robotic)
- 👂 **Hybrid STT**: Whisper (local) + Cloud fallback, 12 model circuit-breaker
- 🔐 **Voice biometrics**: ECAPA-TDNN speaker verification (159ms unlock)

### Vision & Automation
- 👁️ **Real-time screen understanding** (30 FPS capture via mss)
- 🤖 **Ghost Hands**: autonomous browser + keyboard + mouse control
- 📋 **Context Intelligence**: tracks what app is open, what you're doing
- 🔍 **Semantic cache**: remembers what it's seen (ChromaDB, 24h TTL)

### System Integration
- 📊 **RAM monitoring**: auto-offloads to GCP when memory > 80%
- 💰 **Cost optimizer**: Spot VM auto-create ($0.029/hr), scale-to-zero after 15min idle
- 🔒 **Security**: CWE-117/532 log injection prevention, atomic writes (0o600)
- 🛡️ **Self-healing**: circuit breakers, ML-powered recovery, auto-reload

---

## 🚀 Quick Start (Windows)

### Prerequisites
```powershell
# Python 3.12+
python --version   # Must be 3.12+

# Node.js 18+ (for frontend)
node --version

# Git
git --version
```

### 1. Clone
```powershell
git clone https://github.com/nandkishorrathodk-art/Ironcliw-ai.git
cd Ironcliw-ai
```

### 2. Install Python Dependencies
```powershell
pip install -r requirements.txt
pip install edge-tts mss pyautogui pywin32 pyttsx3
```

### 3. Install Frontend
```powershell
cd frontend
npm install
cd ..
```

### 4. Configure Environment
```powershell
# Copy the Windows config template
copy .env.windows .env

# Edit .env and add your API keys:
# ANTHROPIC_API_KEY=sk-ant-xxxx
# FIREWORKS_API_KEY=fw-xxxxxxxx
```

### 5. Run
```powershell
python start_system.py
```

Open **http://localhost:3000** — JARVIS is ready.

> **First time?** It may take 2–3 minutes to initialize all models.  
> Say **"Hey JARVIS"** to activate voice control.

---

## 🔧 Configuration

### Key `.env` Settings
```env
# LLM
JARVIS_LLM_PROVIDER=fireworks          # or "claude"
FIREWORKS_API_KEY=fw-xxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx

# Voice
WHISPER_MODEL_SIZE=base                # tiny/base/small/medium
JARVIS_VOICE_BIOMETRIC_ENABLED=false   # true = needs speechbrain GPU

# Performance
JARVIS_ML_DEVICE=cpu                   # cpu (Windows default)
JARVIS_DYNAMIC_PORTS=false             # keep backend on port 8010
JARVIS_LAZY_LOAD_MODELS=true           # load models on demand
JARVIS_SKIP_GCP=true                   # disable GCP if not needed
JARVIS_SKIP_DOCKER=true

# Windows specifics
JARVIS_AUTO_BYPASS_WINDOWS=true        # bypass voice auth on Windows
JARVIS_DISABLE_SWIFT_EXTENSIONS=true
JARVIS_DISABLE_RUST_EXTENSIONS=true
JARVIS_DISABLE_COREML=true
```

---

## 📁 Project Structure

```
Ironcliw-ai/
├── backend/                    # FastAPI Python backend
│   ├── main.py                 # Entry point (UTF-8 + bootstrap)
│   ├── api/                    # REST endpoints
│   ├── agi_os/                 # AGI operating system layer
│   │   ├── realtime_voice_communicator.py   # edge-tts Neural TTS
│   │   └── notification_bridge.py           # Windows toast notifications
│   ├── voice/                  # STT/TTS/speaker verification
│   │   ├── hybrid_stt_router.py             # Whisper + cloud STT
│   │   └── speaker_verification_service.py  # ECAPA-TDNN biometrics
│   ├── vision/                 # Screen capture + Claude Vision
│   ├── ghost_hands/            # Autonomous automation (pyautogui)
│   ├── intelligence/           # Learning database + SAI
│   ├── core/                   # Pipeline, orchestrator, RAM monitor
│   ├── autonomy/               # Hardware + system control
│   ├── platform_adapter/       # Cross-platform abstraction layer
│   │   ├── windows_platform.py             # Windows-native impl
│   │   └── macos_platform.py               # macOS impl
│   └── windows_native/         # C# native extensions (P/Invoke)
│       ├── AudioEngine/        # Windows WASAPI audio
│       ├── ScreenCapture/      # GDI+ capture
│       └── SystemControl/      # Win32 system APIs
├── frontend/                   # React 18 UI
│   └── src/
│       └── components/
│           └── JarvisVoice.js  # Voice UI + edge-tts Neural voice
├── start_system.py             # Main launcher
├── unified_supervisor.py       # Process lifecycle manager
├── .env.windows                # Windows config template
├── WINDOWS_PORT_BLUEPRINT.md   # Full macOS→Windows conversion guide
├── SECURITY.md                 # Security policy
└── LICENSE                     # MIT
```

---

## 🗣️ Voice Commands

| Say This | JARVIS Does |
|----------|------------|
| "Hey JARVIS" | Activate |
| "What can you do?" | List capabilities |
| "Can you see my screen?" | Vision test |
| "Open Chrome and go to Google" | Browser control |
| "Search for AI news" | Web search |
| "What's my RAM usage?" | System status |
| "Start monitoring my screen" | Begin 30 FPS capture |
| "Set volume to 50%" | Volume control |
| "Lock my screen" | Windows lock (LockWorkStation) |
| "JARVIS, learn my voice" | Enroll voice biometrics |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Frontend (React)                    │
│              http://localhost:3000                       │
└─────────────────────┬───────────────────────────────────┘
                      │ WebSocket / REST
┌─────────────────────▼───────────────────────────────────┐
│                  FastAPI Backend                         │
│              http://localhost:8010                       │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  Voice   │  │  Vision  │  │  Ghost   │  │  SAI   │  │
│  │  System  │  │  System  │  │  Hands   │  │ Aware  │  │
│  │ Whisper  │  │   mss +  │  │pyautogui │  │  +CAI  │  │
│  │edge-tts  │  │  Claude  │  │  pywin32 │  │        │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬───┘  │
│       │             │              │              │      │
│  ┌────▼─────────────▼──────────────▼──────────────▼───┐ │
│  │              Intelligence Core                       │ │
│  │  Claude API │ Fireworks AI │ SQLite │ ChromaDB       │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                      │ GCP Auto-routing (RAM >80%)
┌─────────────────────▼───────────────────────────────────┐
│              GCP Cloud (Optional)                        │
│   e2-highmem-4 Spot VM ($0.029/hr) — 32GB RAM           │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 What's New (Phase 11 — Windows Port)

### All fixes applied and working ✅

| Fix | File | Status |
|-----|------|--------|
| Neural TTS voice (`en-GB-RyanNeural`) | `realtime_voice_communicator.py` | ✅ |
| Windows toast notifications | `notification_bridge.py` | ✅ |
| Ghost Hands: `cliclick` → `pyautogui` | `ghost_hands/background_actuator.py` | ✅ |
| Upstream sync (130 commits from drussell23) | merge commit `3ce7237a` | ✅ |
| ECAPA 25s timeout bypass | `ml_engine_registry.py` | ✅ |
| UNIQUE constraint spam eliminated | `learning_database.py` | ✅ |
| `os.uname()` crash fixed | `infrastructure_orchestrator.py` | ✅ |
| `NoneType` traceback fixed | `speaker_verification_service.py` | ✅ |
| Keychain `WinError 2` silenced | `start_system.py` | ✅ |
| `fcntl` Windows guard | `intelligent_gcp_optimizer.py` | ✅ |
| UTF-8 stdout/stderr (emoji safe) | `main.py` | ✅ |
| WebSocket npm.cmd path fix | `websocket_router.py` | ✅ |
| Secure logging (CWE-117/532) | `secure_logging.py` | ✅ |
| Hardware control: `caffeinate` → `SetThreadExecutionState` | `hardware_control.py` | ✅ |
| Vision: `screencapture` → `mss` | `claude_vision_chatbot.py` | ✅ |

### Remaining (Phase 12+)
See [WINDOWS_PORT_BLUEPRINT.md](WINDOWS_PORT_BLUEPRINT.md) for the complete 700-line guide.

---

## 📦 Key Dependencies

### Backend
```
fastapi, uvicorn, websockets      # Server
anthropic, fireworks-ai           # LLM APIs
openai-whisper                    # Local STT
edge-tts                          # Neural TTS (Windows)
mss, Pillow                       # Screen capture
pyautogui, pywin32                # Automation (Windows)
chromadb                          # Vector memory
pyttsx3                           # Fallback TTS
psutil                            # System monitoring
```

### Frontend
```
react@18                          # UI framework
socket.io-client                  # WebSocket
```

### Optional (Cloud)
```
google-cloud-compute              # GCP Spot VMs
speechbrain                       # ECAPA voice biometrics (GPU)
torchaudio                        # Audio ML (GPU)
```

---

## 🔐 Security

See [SECURITY.md](SECURITY.md) for full security policy.

**Quick notes:**
- API keys go in `.env` only — **never in code**
- JARVIS runs on `localhost` only by default
- Auth is bypassed on Windows MVP (`JARVIS_AUTO_BYPASS_WINDOWS=true`)
- Voice biometric auth requires `speechbrain` + GPU

---

## 🤝 Contributing

This is an active Windows port. Contributions welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/windows-audio`
3. Commit: `git commit -m "feat: add Windows audio engine"`
4. Push: `git push origin feat/windows-audio`
5. Open a PR

**Priority areas:**
- Windows notification system (plyer integration)
- ECAPA fast-fail on Windows (skip 25s timeout)
- Volume/brightness control (pycaw integration)
- Window management (pygetwindow / win32gui)

---

## 📜 Credits & Attribution

| | |
|---|---|
| **Original Author** | [drussell23](https://github.com/drussell23) — [JARVIS](https://github.com/drussell23/JARVIS) |
| **Windows Port** | [Nandkishor Rathod](https://github.com/nandkishorrathodk-art) |
| **Voice** | Microsoft Azure Neural TTS — `en-GB-RyanNeural` via [edge-tts](https://github.com/rany2/edge-tts) |
| **LLM** | [Anthropic Claude](https://anthropic.com) + [Fireworks AI](https://fireworks.ai) |
| **STT** | [OpenAI Whisper](https://github.com/openai/whisper) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

Original JARVIS project by drussell23. Windows port and modifications by Nandkishor Rathod (2026).

---

<div align="center">

**⚡ Ironcliw-AI · JARVIS Windows Port**

*"Sometimes you gotta run before you can walk."* — Tony Stark

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

</div>
