# Ironcliw Fresh Windows Install Blueprint
## Delete Everything → Clone → Run → Test
**Repo:** `nandkishorrathodk-art/Ironcliw-ai`
**Target OS:** Windows 10 / Windows 11
**Time needed:** ~30 minutes

---

## TABLE OF CONTENTS
1. [Before You Start — Checklist](#1-before-you-start--checklist)
2. [Step 1 — Clean Delete Old Ironcliw](#2-step-1--clean-delete-old-jarvis)
3. [Step 2 — Install Prerequisites](#3-step-2--install-prerequisites)
4. [Step 3 — Fresh Clone from GitHub](#4-step-3--fresh-clone-from-github)
5. [Step 4 — Install All Python Dependencies](#5-step-4--install-all-python-dependencies)
6. [Step 5 — Install Frontend Dependencies](#6-step-5--install-frontend-dependencies)
7. [Step 6 — Configure Environment (.env)](#7-step-6--configure-environment-env)
8. [Step 7 — Run Ironcliw](#8-step-7--run-jarvis)
9. [Step 8 — Integration Tests (Pass All 14)](#9-step-8--integration-tests-pass-all-14)
10. [Step 9 — Voice Test](#10-step-9--voice-test)
11. [Troubleshooting Guide](#11-troubleshooting-guide)
12. [Expected Output / What Working Looks Like](#12-expected-output--what-working-looks-like)

---

## 1. Before You Start — Checklist

Check these before anything else:

```powershell
# Open PowerShell as Administrator and run:

# ✅ Python 3.12+
python --version
# Must show: Python 3.12.x

# ✅ pip
pip --version

# ✅ Node.js 18+
node --version
# Must show: v18.x.x or higher

# ✅ npm 9+
npm --version

# ✅ Git
git --version

# ✅ Internet connection (for pip install, npm install, edge-tts)
ping 8.8.8.8
```

If anything is missing, install it first:
- **Python 3.12**: https://www.python.org/downloads/ (check "Add to PATH" during install)
- **Node.js 18+**: https://nodejs.org/en/download/
- **Git**: https://git-scm.com/download/win

---

## 2. Step 1 — Clean Delete Old Ironcliw

### Option A — Delete via PowerShell (Safe)
```powershell
# ⚠️ WARNING: This permanently deletes the folder
# Make sure you have committed everything to GitHub first!

# Go UP from the Ironcliw folder (don't be inside it)
cd C:\Users\nandk

# Verify git is clean before deleting
cd Ironcliw
git status
git log --oneline -3

# If everything is committed and pushed:
cd C:\Users\nandk
Remove-Item -Recurse -Force "C:\Users\nandk\Ironcliw"

# Verify it's gone
Test-Path "C:\Users\nandk\Ironcliw"
# Should print: False
```

### Option B — Delete via Windows Explorer
1. Open `C:\Users\nandk\`
2. Right-click `Ironcliw` folder → Delete
3. Empty Recycle Bin

### Also Clean Ironcliw Data (Optional — keeps your voice profiles etc.)
```powershell
# ⚠️ OPTIONAL: Only run if you want a completely fresh start
# This deletes your learning database, voice profiles, cached data

# Check what's there first:
ls $env:USERPROFILE\.jarvis\

# Delete only if you want a 100% fresh start:
# Remove-Item -Recurse -Force "$env:USERPROFILE\.jarvis"
```

---

## 3. Step 2 — Install Prerequisites

### Python Packages (system-level — do this once)
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install build tools (needed for some packages)
pip install wheel setuptools
```

### Windows SDK Tools (if not installed)
```powershell
# Check if Visual C++ Build Tools are installed
python -c "import ctypes; print('ctypes OK')"

# If anything fails with "Microsoft Visual C++ required", download:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Install: "C++ build tools" workload
```

---

## 4. Step 3 — Fresh Clone from GitHub

```powershell
# Go to where you want Ironcliw installed
cd C:\Users\nandk

# Clone your Windows fork
git clone https://github.com/nandkishorrathodk-art/Ironcliw-ai.git Ironcliw

# Enter the folder
cd Ironcliw

# Verify you're on main branch with latest commit
git log --oneline -5
git branch
```

**Expected output:**
```
69f39635 fix: resolve 3 integration test failures
5dbd1853 docs: rewrite README and SECURITY for Phase 11 Windows port
e2726392 docs(windows): Add native build dependencies to Quick Start
...
* main
```

---

## 5. Step 4 — Install All Python Dependencies

Run these in order from inside `C:\Users\nandk\Ironcliw`:

### Part A — Core requirements
```powershell
pip install -r requirements.txt
```
> This will take 5–15 minutes. Some packages are large (torch, transformers, etc.)

### Part B — Windows-specific packages (MUST install these)
```powershell
pip install edge-tts mss pyautogui pywin32 pyttsx3 plyer
```

### Part C — Audio & voice
```powershell
pip install soundfile sounddevice openai-whisper numba
```

### Part D — Database & memory
```powershell
pip install chromadb asyncpg aiohttp aiohttp-cors
```

### Part E — LLM APIs
```powershell
pip install anthropic fireworks-ai openai
```

### Part F — Verify critical ones
```powershell
python -c "import edge_tts; print('edge-tts OK')"
python -c "import mss; print('mss OK')"
python -c "import pyautogui; print('pyautogui OK')"
python -c "import whisper; print('whisper OK')"
python -c "import chromadb; print('chromadb OK')"
python -c "import anthropic; print('anthropic OK')"
```

All should print `OK`. If any fails, install it individually:
```powershell
pip install <package-name>
```

### Optional (for GPU / advanced features — skip on normal laptops)
```powershell
# Only if you have NVIDIA GPU + CUDA:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install speechbrain  # For voice biometrics

# For volume control (pycaw):
pip install pycaw comtypes

# For window management:
pip install pyvda pygetwindow
```

---

## 6. Step 5 — Install Frontend Dependencies

```powershell
# From C:\Users\nandk\Ironcliw
cd frontend
npm install
cd ..
```

**This will take 2–5 minutes.** Expected: `added XXX packages` — warnings are OK, errors are not.

Verify:
```powershell
# Should list react, socket.io-client etc.
ls frontend\node_modules\.bin\ | Select-Object -First 10
```

---

## 7. Step 6 — Configure Environment (.env)

```powershell
# Copy the Windows template
copy .env.windows .env

# Open .env in Notepad and fill in your API keys
notepad .env
```

### Required Settings in `.env`
```env
# ─── LLM API Keys (get from their websites) ─────────────────────
ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE
FIREWORKS_API_KEY=fw-YOUR_KEY_HERE

# ─── Windows Settings (these are already set correctly) ──────────
Ironcliw_AUTO_BYPASS_WINDOWS=true
Ironcliw_VOICE_BIOMETRIC_ENABLED=false
Ironcliw_ML_DEVICE=cpu
Ironcliw_DYNAMIC_PORTS=false
Ironcliw_SKIP_GCP=true
Ironcliw_SKIP_DOCKER=true
Ironcliw_DISABLE_SWIFT_EXTENSIONS=true
Ironcliw_DISABLE_RUST_EXTENSIONS=true
Ironcliw_DISABLE_COREML=true

# ─── Voice ───────────────────────────────────────────────────────
WHISPER_MODEL_SIZE=base

# ─── Ports ───────────────────────────────────────────────────────
Ironcliw_PORT=8010
FRONTEND_PORT=3000
```

### Where to get API keys:
- **Anthropic (Claude)**: https://console.anthropic.com/ → API Keys
- **Fireworks AI**: https://fireworks.ai/ → Account → API Keys

### Secure the .env file
```powershell
# Restrict to your user only
icacls .env /inheritance:r /grant:r "%USERNAME%:R"

# Verify it's gitignored (should print the .gitignore path)
git check-ignore -v .env
```

---

## 8. Step 7 — Run Ironcliw

```powershell
# From C:\Users\nandk\Ironcliw
python start_system.py
```

### What you'll see (normal):
```
✅ Cache cleared in XXXXms - using fresh code!
✓ Goal Inference Automation: ENABLED
...
✓ Starting in autonomous mode...
...
Starting backend with main.py (auto-reload enabled)...
...
FULL MODE reached!
✓ Optimized backend started (PID: XXXXX)
...
✓ Frontend verified at port 3000

============================================================
🎯 Ironcliw is ready!
============================================================
  • Frontend: http://localhost:3000/
  • Backend API: http://localhost:8010/docs
```

### Browser opens automatically to http://localhost:3000

**First launch takes 2–4 minutes.** Subsequent launches take ~30 seconds.

---

## 9. Step 8 — Integration Tests (Pass All 14)

Run this to verify all Windows port features are working:

```powershell
# From C:\Users\nandk\Ironcliw
python test_integration.py
```

### Expected: 14/14 PASS
```
=== Ironcliw Windows Integration Test — Phases 12-20 ===

  PASS  Phase 12 — notification_bridge import
  PASS  Phase 12 — ml_engine_registry import
  PASS  Phase 13 — ghost_hands.background_actuator import
  PASS  Phase 14 — claude_vision_chatbot import
  PASS  Phase 15 — hardware_control import
  PASS  Phase 16 — platform_adapter.get_platform()
  PASS  Phase 17 — action_executors import
  PASS  Phase 18 — fcntl absent, msvcrt present
  PASS  Phase 19 — edge_tts import
  PASS  Phase 20 — screen lock detection
  PASS  Win32 — GetForegroundWindow()
  PASS  mss — screen capture available
  PASS  psutil — virtual_memory()
  PASS  pyautogui — mouse position()

==================================================
PASSED: 14/14
==================================================
```

If any FAIL — see [Troubleshooting Guide](#11-troubleshooting-guide) below.

---

## 10. Step 9 — Voice Test

Once Ironcliw is running in the browser:

### Test 1 — TTS (Ironcliw speaks to you)
1. Open http://localhost:3000
2. Type in the chat: `Hello Ironcliw`
3. You should hear **"Hello! How can I assist you today?"** in a British male voice (`en-GB-RyanNeural`)

### Test 2 — Wake Word
1. Make sure microphone is allowed in your browser
2. Say **"Hey Ironcliw"** out loud
3. You should see the waveform activate and Ironcliw respond

### Test 3 — Vision
1. Type: `Can you see my screen?`
2. Ironcliw should describe what's on your screen

### Test 4 — System Info
1. Type: `What's my RAM usage?`
2. Ironcliw should report your current memory stats

### Test 5 — Voice Command
1. Say: **"What time is it?"**
2. Ironcliw should respond with the current time

---

## 11. Troubleshooting Guide

### Problem: `pip install -r requirements.txt` fails on a package

```powershell
# Try installing problematic package alone
pip install <package> --no-deps

# Or skip it and continue
pip install -r requirements.txt --ignore-requires-python

# For packages that need Visual C++:
# Install Microsoft C++ Build Tools from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Problem: `edge-tts` fails / no sound

```powershell
# Test edge-tts directly
python -c "import asyncio; import edge_tts; async def t(): c=edge_tts.Communicate('Hello Ironcliw', voice='en-GB-RyanNeural'); await c.save('test.mp3'); import os; print(os.path.getsize('test.mp3'), 'bytes'); os.unlink('test.mp3'); asyncio.run(t())"
```

If it fails with DNS error:
```powershell
# aiodns bug — this is handled in realtime_voice_communicator.py automatically
# Manual workaround: pip install aiohttp[speedups]
pip install aiohttp --upgrade
```

### Problem: `No module named 'fcntl'`

```powershell
# This was fixed in commit 69f39635
# Make sure you have the latest code:
git pull origin main

# Verify the fix is there:
python -c "import sys; sys.path.insert(0,'backend'); import core.resilience.atomic_file_ops; print('OK')"
```

### Problem: Backend starts on wrong port (not 8010)

```powershell
# Check .env has:
Ironcliw_DYNAMIC_PORTS=false
Ironcliw_PORT=8010

# Kill any stale processes
netstat -ano | findstr :8010
# Note the PID then:
taskkill /PID <PID> /F
```

### Problem: Frontend shows "CONNECTING..." forever

```powershell
# 1. Check backend is actually running
curl http://localhost:8010/health/ping

# 2. Check WebSocket port
curl http://localhost:8010/health/startup

# 3. Rebuild frontend
cd frontend
npm run build
cd ..
python start_system.py
```

### Problem: No voice / TTS not working

```powershell
# Test pyttsx3 fallback:
python -c "import pyttsx3; e=pyttsx3.init(); e.say('Hello'); e.runAndWait()"

# Test edge-tts save:
python -c "import asyncio, edge_tts; asyncio.run(edge_tts.Communicate('test', voice='en-GB-RyanNeural').save('t.mp3')); import os; print(os.path.exists('t.mp3')); os.unlink('t.mp3')"
```

### Problem: `pywin32` not working / ImportError

```powershell
# Reinstall pywin32 with post-install script
pip uninstall pywin32 -y
pip install pywin32
python C:\Users\nandk\AppData\Local\Programs\Python\Python312\Scripts\pywin32_postinstall.py -install
```

### Problem: `No module named 'speechbrain'` (25-second delay on startup)

This is expected on Windows without GPU. The fix is in `ml_engine_registry.py` (fast-fail). But if you still see a 25s delay:

```powershell
# Verify the ECAPA fast-fail is in place
python -c "import sys; sys.path.insert(0,'backend'); import voice_unlock.ml_engine_registry; print('OK')"
```

### Problem: `WinError 2` on startup (Keychain)

Already fixed in the code. If you see it:
```powershell
git pull origin main
python start_system.py
```

### Problem: ChromaDB fails to initialize

```powershell
pip install chromadb --upgrade
# Clear old ChromaDB data if corrupted:
Remove-Item -Recurse -Force "$env:USERPROFILE\.jarvis\chroma_voice_patterns"
```

### Problem: UNIQUE constraint error spam in logs

Already fixed. If you see it:
```powershell
git pull origin main
```

### Problem: `No module named 'resource'` warning

This is a known warning on Windows — it's non-fatal and can be ignored. The system uses `psutil` as fallback.

---

## 12. Expected Output / What Working Looks Like

### Startup Log (healthy)
```
✅ Cache cleared — using fresh code!
✓ PID lock acquired
✓ Rate Limiting: ML Forecasting + Adaptive Throttling
✓ Intelligent ECAPA Backend Orchestrator:
   ❌ speechbrain not installed (expected on Windows CPU)
   → Using cloud_run fallback
✓ Semantic Voice Cache initialized
ℹ️  Keychain: Skipped (Windows — not available)   ← GOOD
✓ Password Typer: FUNCTIONAL
✓ Server running on port 8010
✓ Frontend process started successfully
✓ Backend API ready (0.6s)
✓ Frontend ready (0.3s)

🎯 Ironcliw is ready!
  • Frontend: http://localhost:3000/
  • Backend API: http://localhost:8010/docs
```

### Backend Health Check
```powershell
curl http://localhost:8010/health/ping
# Expected: {"status":"ok","timestamp":"..."}

curl http://localhost:8010/health/startup
# Expected: {"progress":100,"status":"FULL MODE",...}
```

### Integration Test (all green)
```
PASSED: 14/14
```

### Frontend
- URL: http://localhost:3000
- Should show Ironcliw interface
- Status: "SYSTEM READY" (not "CONNECTING")
- Voice waveform visible in top area

---

## Summary — Command Sequence (Quick Reference)

```powershell
# === FRESH INSTALL IN ONE BLOCK ===

# 1. Delete old (from parent folder)
Remove-Item -Recurse -Force "C:\Users\nandk\Ironcliw"

# 2. Clone fresh
cd C:\Users\nandk
git clone https://github.com/nandkishorrathodk-art/Ironcliw-ai.git Ironcliw
cd Ironcliw

# 3. Install Python deps
pip install -r requirements.txt
pip install edge-tts mss pyautogui pywin32 pyttsx3 plyer soundfile sounddevice openai-whisper chromadb aiohttp aiohttp-cors anthropic fireworks-ai

# 4. Install frontend
cd frontend && npm install && cd ..

# 5. Set up .env
copy .env.windows .env
notepad .env   # Add your API keys

# 6. Run integration tests
python test_integration.py
# Must show: PASSED: 14/14

# 7. Start Ironcliw
python start_system.py
# Opens http://localhost:3000 automatically
```

---

## Version Reference

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12+ | Required |
| Node.js | 18+ | For React frontend |
| edge-tts | 7.2.7+ | Neural TTS (en-GB-RyanNeural) |
| Whisper | latest | CPU mode (base model) |
| FastAPI | latest | Backend server |
| React | 18 | Frontend |
| chromadb | latest | Vector memory |
| pywin32 | latest | Windows APIs |
| pyautogui | latest | Mouse/keyboard |
| mss | latest | Screen capture |
| Port Phase | **11 Complete** | All 14 integration tests pass |

---

*Fresh Install Blueprint v1.0 — Phase 11 Complete — 2026-02-25*
*Repo: nandkishorrathodk-art/Ironcliw-ai*
