# JARVIS for Windows - Quick Start Guide

## Overview

JARVIS AI Assistant has been successfully adapted to run on Windows 10/11. This guide will help you get started on your Windows machine.

**Platform**: Windows 10/11 (64-bit)  
**Python**: 3.9+ (3.12+ recommended)  
**RAM**: 4GB minimum, 16GB recommended  
**Status**: ✅ Core implementation tested and working

---

## Quick Start (5 Minutes)

### 1. Prerequisites

- **Python 3.9+**: Download from [python.org](https://www.python.org/downloads/) or Microsoft Store
- **Git**: Download from [git-scm.com](https://git-scm.com/download/win)
- **.NET 8 SDK**: Required for Windows Native C# DLLs. Download from [dotnet.microsoft.com](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
- **Rust (Cargo)**: Required for high-performance ML layers. Download from [rustup.rs](https://rustup.rs/)

Verify installation:
```cmd
python --version
git --version
dotnet --version
cargo --version
```

### 2. Clone Repository

```cmd
git clone https://github.com/drussell23/JARVIS
cd JARVIS
```

### 3. Install Dependencies and Build Native Extensions

First, install the Python requirements:
```cmd
pip install -r requirements.txt
pip install -r requirements-windows.txt
# For high-quality Piper TTS (Recommended over pyttsx3)
pip install piper-tts soundfile
```

Next, build the C# Windows Native DLLs:
```cmd
dotnet build backend/windows_native/JarvisWindowsNative.sln -c Release
```

Finally, verify the Rust high-performance libraries compile:
```cmd
cd backend/native_extensions/rust_processor
cargo check
cd ../../rust_performance
cargo check
cd ../../..
```

### 4. Configure Environment

```cmd
copy .env.windows .env
```

(Optional) Edit `.env` to add your Claude API key:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

### 5. Test Platform Abstraction

```cmd
python test_platform.py
```

Expected output:
```
[OK] Platform loaded: WindowsPlatform
[OK] System Info: <your-hostname>
[OK] Idle Time: <seconds>
[OK] Monitors: <count> detected
[OK] Windows: <count> open windows
[OK] Battery: <percentage>% 
```

### 6. Test Authentication Bypass

```cmd
python test_auth_bypass.py
```

Expected output:
```
[SUCCESS] Authentication bypassed successfully!
User ID: bypass_user_windows_<username>
Method: bypass_auto
```

### 7. Start JARVIS

```cmd
python unified_supervisor.py
```

The system will start:
- Backend on `http://localhost:8010`
- Frontend on `http://localhost:3000`

Open your browser to `http://localhost:3000` to access the UI.

---

## What Works on Windows

### ✅ Fully Functional
- **Platform Abstraction**: Automatic Windows detection
- **Window Management**: List, focus, and manipulate windows
- **Screen Capture**: Multi-monitor screenshot support (mss)
- **Mouse & Keyboard**: Full automation with pyautogui
- **System Information**: CPU, memory, disk, battery status
- **Idle Time Detection**: Accurate idle time tracking
- **Authentication**: Automatic bypass (voice biometric unavailable)
- **File Locking**: Cross-process locks with msvcrt

### ⚠️ Partially Working (Optional Packages)
- **Audio Devices**: Requires `sounddevice` installation
- **Notifications**: Requires `win10toast` installation
- **UI Automation**: Requires `pywinauto` installation

### ❌ Not Available (macOS-Only)
- Voice biometric authentication (replaced with auto-bypass)
- Swift native extensions
- Rust Metal GPU acceleration
- CoreML model acceleration
- AppleScript integration

---

## Configuration

### Environment Variables

The `.env.windows` file contains all Windows-specific settings:

#### Authentication
```bash
# Auto-bypass authentication on Windows
JARVIS_AUTO_BYPASS_WINDOWS=true

# Or require a password for bypass
# JARVIS_BYPASS_PASSWORD=your_secure_password
```

#### Performance
```bash
# Skip cloud features for faster startup
JARVIS_SKIP_GCP=true
JARVIS_SKIP_DOCKER=true

# Use CPU for ML inference
JARVIS_ML_DEVICE=cpu

# Lazy load models (load on demand)
JARVIS_LAZY_LOAD_MODELS=true
```

#### Platform Settings
```bash
# Screen capture
JARVIS_SCREEN_CAPTURE_METHOD=mss

# Text-to-speech
JARVIS_TTS_ENGINE=piper

# Window management
JARVIS_WINDOW_MANAGER=win32gui

# Automation
JARVIS_AUTOMATION_PROVIDER=pyautogui
```

---

## Install Optional Packages

For full functionality, install these optional packages:

```cmd
pip install sounddevice win10toast pywinauto wmi
```

- **sounddevice**: Audio device enumeration
- **win10toast**: Windows 10/11 toast notifications
- **pywinauto**: Advanced UI automation
- **wmi**: Windows Management Instrumentation

---

## Troubleshooting

### Python Not Found

If `python` command doesn't work, try:
```cmd
python3 --version
py --version
```

Or add Python to PATH:
1. Search "Environment Variables" in Start Menu
2. Edit "Path" in System Variables
3. Add Python installation directory

### ModuleNotFoundError

Install missing dependencies:
```cmd
pip install -r requirements.txt
pip install -r requirements-windows.txt
```

### Permission Errors

Run Command Prompt as Administrator:
1. Right-click Command Prompt
2. Select "Run as administrator"

### Port Already in Use

If port 8010 or 3000 is in use, change in `.env`:
```bash
JARVIS_PORT=8011
FRONTEND_PORT=3001
```

### Authentication Bypass Not Working

Verify `.env` contains:
```bash
JARVIS_AUTO_BYPASS_WINDOWS=true
```

Or set explicitly:
```bash
JARVIS_BYPASS_AUTH=true
```

---

## Next Steps

### 1. Customize Configuration

Edit `.env` to:
- Add API keys (Claude, OpenAI, ElevenLabs)
- Adjust performance settings
- Enable/disable features

### 2. Explore Features

Try these commands in the UI (http://localhost:3000):
- "What can you see on my screen?"
- "List all open windows"
- "Take a screenshot"
- "What's my battery status?"

### 3. Enable Additional Features

Install optional dependencies:
```cmd
pip install sounddevice win10toast pywinauto
```

Test audio:
```cmd
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### 4. Advanced Setup

- **CUDA GPU**: Install PyTorch with CUDA for GPU acceleration
- **Cloud Integration**: Configure GCP for cloud features
- **Trinity Ecosystem**: Clone JARVIS-Prime and Reactor-Core repos

---

## Known Limitations

1. **Voice Features**
   - Voice biometric authentication not available (auto-bypassed)
   - Wake word detection untested
   - TTS works but uses pyttsx3 (not as natural as macOS)

2. **ML Models**
   - CPU-only inference (slower than GPU)
   - CUDA GPU works if you have NVIDIA GPU + drivers
   - CoreML models not available (macOS-only)

3. **Native Extensions**
   - Swift extensions not available
   - Rust Metal GPU not available
   - Some advanced features may be limited

---

## Getting Help

### Documentation
- [Main README](README.md) - Full JARVIS documentation
- [Technical Spec](.zenflow/tasks/ironclaw-6377/spec.md) - Cross-platform migration details
- [Implementation Report](.zenflow/tasks/ironclaw-6377/report.md) - What was implemented

### Testing
- `test_platform.py` - Test platform abstraction
- `test_auth_bypass.py` - Test authentication bypass

### Logs
Check logs in `./logs/jarvis.log` for errors and warnings.

### Issues
Report Windows-specific issues with:
- Windows version (run `winver`)
- Python version (`python --version`)
- Error message from logs
- Steps to reproduce

---

## Development

### Virtual Environment (Recommended)

Create isolated Python environment:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-windows.txt
```

Deactivate when done:
```cmd
deactivate
```

### Running Tests

```cmd
python test_platform.py
python test_auth_bypass.py
```

### Hot Reload (Development Mode)

Enable auto-restart on code changes:
```bash
# In .env
JARVIS_DEV_MODE=true
JARVIS_HOT_RELOAD=true
```

---

## Platform Comparison

| Feature | Windows | macOS | Linux |
|---------|---------|-------|-------|
| Platform Abstraction | ✅ | ✅ | ✅ |
| Window Management | ✅ win32gui | ✅ AppleScript | ✅ wmctrl |
| Screen Capture | ✅ mss | ✅ Swift | ✅ mss |
| Automation | ✅ pyautogui | ✅ cliclick | ✅ xdotool |
| Voice Biometric | ❌ Bypassed | ✅ Native | ❌ Bypassed |
| TTS | ✅ piper (Recommended) / pyttsx3 | ✅ Native | ✅ espeak |
| Notifications | ✅ win10toast | ✅ Native | ✅ notify-send |
| GPU Acceleration | ⚠️ CUDA | ✅ Metal | ⚠️ CUDA/ROCm |
| File Locking | ✅ msvcrt | ✅ fcntl | ✅ fcntl |

---

## Changelog

### v1.0 (February 2026) - Initial Windows Support

**Added**:
- ✅ Platform Abstraction Layer (PAL)
- ✅ Windows-specific implementations (window, screen, audio, system)
- ✅ Authentication bypass system
- ✅ Windows file locking (msvcrt)
- ✅ Configuration template (.env.windows)
- ✅ Test scripts for verification

**Tested**:
- Windows 11 Build 26200
- Python 3.12.10
- Acer Swift Neo (16GB RAM, AMD64)

**Known Issues**:
- None critical - all core features working

---

## Credits

**Original JARVIS**: macOS implementation by drussell23  
**Windows Port**: Cross-platform migration (February 2026)  
**Platform Abstraction**: Already existed in codebase, verified working  
**Authentication Bypass**: New implementation for Windows/Linux  
**File Locking**: New Windows-compatible implementation  

---

## License

Same as main JARVIS project (see LICENSE file in root directory).

---

**Quick Links**:
- [Main README](README.md)
- [Technical Specification](.zenflow/tasks/ironclaw-6377/spec.md)
- [Implementation Report](.zenflow/tasks/ironclaw-6377/report.md)
- [GitHub Repository](https://github.com/drussell23/JARVIS)

**Windows Status**: ✅ **Working** (Core features tested and verified)
