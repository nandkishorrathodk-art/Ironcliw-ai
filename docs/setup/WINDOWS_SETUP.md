# JARVIS Windows Setup Guide

**Complete installation guide for JARVIS on Windows 10/11**

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Configuration](#configuration)
- [First Run](#first-run)
- [Troubleshooting](#troubleshooting)
- [Common Issues](#common-issues)

---

## Prerequisites

### Required Software

1. **Python 3.9 or later** (3.11+ recommended)
   - Download from: https://www.python.org/downloads/
   - **Important**: Check "Add Python to PATH" during installation

2. **Git for Windows**
   - Download from: https://git-scm.com/download/win
   - Use default settings during installation

3. **Visual Studio Build Tools** (for some dependencies)
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++" workload

4. **Node.js 16+** (for frontend)
   - Download from: https://nodejs.org/
   - Choose LTS version

### Optional Software

- **Docker Desktop for Windows** (for containerized ML models)
  - Download from: https://www.docker.com/products/docker-desktop
  - Requires WSL 2 backend

- **CUDA Toolkit** (for NVIDIA GPU acceleration)
  - Download from: https://developer.nvidia.com/cuda-downloads
  - Only if you have NVIDIA GPU

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10 (64-bit) | Windows 11 (64-bit) |
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 10 GB free | 20 GB+ free (SSD preferred) |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | None (CPU fallback) | NVIDIA GPU with CUDA support |
| **Network** | Internet connection | High-speed internet for cloud features |

---

## Installation Steps

### 1. Clone the Repository

Open PowerShell or Command Prompt:

```powershell
# Navigate to your projects folder
cd %USERPROFILE%\Documents

# Clone JARVIS
git clone https://github.com/drussell23/JARVIS-AI-Agent.git
cd JARVIS-AI-Agent
```

### 2. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

**Tip**: You'll see `(venv)` in your prompt when the virtual environment is active.

### 3. Install Dependencies

```powershell
# Install Python dependencies
pip install -r requirements.txt

# If you encounter errors, try:
pip install -r requirements.txt --use-pep517
```

**Common Installation Issues**:
- If `torch` fails to install, try: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- If `pywin32` fails, install via: `pip install pywin32==306`

### 4. Install Frontend Dependencies

```powershell
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Return to root directory
cd ..
```

### 5. Verify Installation

Run the dependency verification script:

```powershell
python verify_dependencies.py
```

You should see:
```
✅ Platform detected: Windows
✅ 27/27 dependencies verified successfully
```

---

## Configuration

### 1. Create Environment File

Copy the platform-specific environment template:

```powershell
copy .env.platform.example .env
```

### 2. Edit Configuration

Open `.env` in your favorite text editor (e.g., Notepad++, VS Code) and configure:

```bash
# Platform Configuration
JARVIS_PLATFORM=windows

# Authentication (bypass enabled for Windows by default)
JARVIS_AUTH_BYPASS=true

# TTS Engine (Windows uses SAPI)
JARVIS_TTS_ENGINE=pyttsx3

# Screen Capture Method
JARVIS_CAPTURE_METHOD=mss

# GPU Backend (auto-detect or specify)
JARVIS_GPU_BACKEND=auto  # Options: cpu, cuda, directx

# ML Inference Backend
JARVIS_ML_BACKEND=cpu  # Change to 'cuda' if you have NVIDIA GPU

# Data Directories (Windows-style paths)
JARVIS_CONFIG_DIR=%APPDATA%\JARVIS\config
JARVIS_LOG_DIR=%LOCALAPPDATA%\JARVIS\logs
JARVIS_DATA_DIR=%LOCALAPPDATA%\JARVIS\data
JARVIS_CACHE_DIR=%LOCALAPPDATA%\JARVIS\cache
```

### 3. Configure Windows-Specific Settings

Edit `backend/config/windows_config.yaml`:

```yaml
# Text-to-Speech Configuration
tts:
  engine: sapi  # Windows Speech API
  voice: default  # Or specify: "Microsoft David Desktop"
  rate: 200  # Speech rate (words per minute)

# Screen Capture
screen_capture:
  method: mss  # Fast screen capture library
  fps: 30
  
# GPU Configuration
gpu:
  backend: directx  # Or 'cuda' for NVIDIA
  device_id: 0
```

---

## First Run

### 1. Start JARVIS

**Important**: Always run from activated virtual environment!

```powershell
# Activate virtual environment (if not already active)
.\venv\Scripts\activate

# Start JARVIS supervisor
python unified_supervisor.py
```

### 2. What to Expect

First startup will:
1. ✅ Detect Windows platform
2. ✅ Initialize cross-platform abstractions
3. ✅ Start backend server (port 8010)
4. ✅ Start frontend UI (port 3000)
5. ✅ Open browser to http://localhost:3000

**Startup time**: 30-60 seconds on first run

### 3. Verify Everything Works

Once the UI loads, you should see:
- ✅ "JARVIS READY" status
- ✅ System status indicators (all green)
- ✅ No error messages in the console

### 4. Test Basic Features

Try these commands in the UI:
- "What's my screen resolution?" (tests screen capture)
- "List all windows" (tests window management)
- "Say hello" (tests text-to-speech)

---

## Troubleshooting

### Backend Won't Start

**Symptom**: Port 8010 already in use

**Solution**:
```powershell
# Find process using port 8010
netstat -ano | findstr :8010

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Frontend Won't Load

**Symptom**: Port 3000 already in use

**Solution**:
```powershell
# Find process using port 3000
netstat -ano | findstr :3000

# Kill the process
taskkill /PID <PID> /F
```

### Python Not Found

**Symptom**: `'python' is not recognized as an internal or external command`

**Solution**:
1. Verify Python installation: Open Command Prompt and run `py --version`
2. If Python is installed but not in PATH:
   - Search for "Environment Variables" in Windows Start menu
   - Edit "Path" variable
   - Add Python installation directory (e.g., `C:\Users\YourName\AppData\Local\Programs\Python\Python311`)
   - Add Scripts directory (e.g., `C:\Users\YourName\AppData\Local\Programs\Python\Python311\Scripts`)
3. Restart Command Prompt/PowerShell

### Dependencies Won't Install

**Symptom**: `pip install` fails with compilation errors

**Solution**:
1. Install Visual Studio Build Tools (see Prerequisites)
2. Try installing problematic packages individually:
   ```powershell
   pip install <package-name> --no-cache-dir
   ```

### Screen Capture Not Working

**Symptom**: Black screen or capture errors

**Solution**:
1. Ensure `mss` is installed: `pip install mss`
2. Check if you have permission to capture screen:
   - Windows Settings > Privacy > Screen capture
   - Allow apps to capture screen
3. Try running as Administrator (right-click PowerShell > Run as Administrator)

### TTS Not Speaking

**Symptom**: No voice output

**Solution**:
1. Verify TTS engine: `python -c "import pyttsx3; pyttsx3.speak('test')"`
2. Check Windows sound settings:
   - Sound Control Panel > Playback devices
   - Ensure default device is set
3. Try different voice:
   ```python
   import pyttsx3
   engine = pyttsx3.init()
   voices = engine.getProperty('voices')
   for voice in voices:
       print(voice.name)
   ```

### GPU Not Detected

**Symptom**: JARVIS falls back to CPU despite having GPU

**Solution**:
1. For NVIDIA GPUs:
   ```powershell
   # Verify CUDA is installed
   nvidia-smi
   
   # Install CUDA-enabled PyTorch
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
2. Update GPU drivers to latest version
3. Set GPU backend in `.env`:
   ```bash
   JARVIS_GPU_BACKEND=cuda
   ```

---

## Common Issues

### Issue: Virtual Environment Not Activating

**Symptom**: `.\venv\Scripts\activate` doesn't work

**Solution**:
```powershell
# PowerShell execution policy issue
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activating again
.\venv\Scripts\activate
```

### Issue: Import Errors After Installation

**Symptom**: `ModuleNotFoundError` despite installing dependencies

**Solution**:
```powershell
# Ensure you're in virtual environment
.\venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Slow Performance

**Symptom**: JARVIS is slow or unresponsive

**Solutions**:
1. **Enable GPU acceleration** (if available):
   - Edit `.env`: `JARVIS_GPU_BACKEND=cuda`
   - Edit `.env`: `JARVIS_ML_BACKEND=cuda`

2. **Reduce screen capture FPS**:
   - Edit `backend/config/windows_config.yaml`
   - Set `screen_capture.fps: 15` (lower = faster)

3. **Close unnecessary applications**:
   - JARVIS benefits from available RAM and CPU

4. **Use SSD instead of HDD**:
   - Move JARVIS to SSD if possible

### Issue: Firewall Blocks JARVIS

**Symptom**: Cannot access UI at http://localhost:3000

**Solution**:
1. Windows Firewall > Allow an app
2. Add Python and Node.js to allowed apps
3. Or temporarily disable firewall for testing

### Issue: Antivirus False Positives

**Symptom**: Antivirus blocks JARVIS files

**Solution**:
1. Add JARVIS directory to antivirus exclusions:
   - Windows Security > Virus & threat protection
   - Manage settings > Exclusions
   - Add folder: `C:\Users\YourName\Documents\JARVIS-AI-Agent`
2. This is safe: JARVIS source code is open and transparent

---

## Next Steps

✅ **Setup complete!** JARVIS is now running on Windows.

**Learn more**:
- [Main README](../../README.md) - Overview and features
- [API Documentation](../API.md) - REST and WebSocket APIs
- [Architecture Guide](../architecture/) - System architecture
- [Linux Setup](LINUX_SETUP.md) - Install on Linux

**Join the community**:
- GitHub Issues: Report bugs or request features
- Discussions: Ask questions and share experiences

---

## Advanced Configuration

### Multi-Monitor Setup

Edit `backend/config/windows_config.yaml`:

```yaml
screen_capture:
  monitor: 0  # Primary monitor
  # Or specify monitor by index: 1, 2, etc.
  # Use -1 for all monitors
```

### Custom Data Directories

Create custom directories for logs and data:

```powershell
# Create custom directories
mkdir D:\JARVIS_Data
mkdir D:\JARVIS_Logs

# Update .env
# JARVIS_DATA_DIR=D:\JARVIS_Data
# JARVIS_LOG_DIR=D:\JARVIS_Logs
```

### Performance Tuning

For high-performance systems:

```yaml
# backend/config/windows_config.yaml
screen_capture:
  fps: 60  # Higher FPS
  quality: 95  # Higher quality
  
gpu:
  backend: cuda
  enable_fp16: true  # Faster inference (requires NVIDIA GPU)
```

---

## Uninstallation

To completely remove JARVIS:

```powershell
# 1. Deactivate virtual environment
deactivate

# 2. Delete JARVIS directory
cd ..
rmdir /s /q JARVIS-AI-Agent

# 3. Delete user data (optional)
rmdir /s /q "%LOCALAPPDATA%\JARVIS"
rmdir /s /q "%APPDATA%\JARVIS"
```

---

**Last updated**: February 2026  
**Version**: 1.0.0 (Cross-Platform Release)  
**Platform**: Windows 10/11
