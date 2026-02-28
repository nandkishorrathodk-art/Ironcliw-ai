# Ironcliw Windows Installation Guide

## Overview

This guide will walk you through installing Ironcliw AI Assistant on Windows 10/11. The Windows port maintains feature parity with the macOS version while using Windows-native APIs and technologies.

**Platform Support:**
- ✅ Windows 10 (version 1809+)
- ✅ Windows 11
- ✅ Windows Server 2019/2022

**Hardware Requirements:**
- **CPU**: Intel Core i5/AMD Ryzen 5 or better (8 cores recommended)
- **RAM**: 16GB minimum (32GB recommended for ML workloads)
- **Storage**: 50GB available space (SSD recommended)
- **GPU** (Optional): DirectML-compatible GPU for hardware acceleration
  - NVIDIA GTX 1060 / RTX 20xx or newer
  - AMD RX 580 or newer
  - Intel Arc A-series

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Automated Installation](#automated-installation)
3. [Manual Installation](#manual-installation)
4. [Post-Installation Setup](#post-installation-setup)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## Prerequisites

### 1. Install Python 3.11+

Ironcliw requires Python 3.11 or later (3.12 recommended).

**Option A: Winget (Recommended)**
```powershell
winget install Python.Python.3.12
```

**Option B: Manual Download**
1. Download from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ **CHECK**: "Add Python to PATH"
4. Click "Install Now"

**Verify Installation:**
```powershell
python --version  # Should show Python 3.11+ or 3.12+
pip --version
```

### 2. Install Visual Studio Build Tools

Required for compiling native Python extensions (numpy, scipy, etc.).

**Option A: Winget**
```powershell
winget install Microsoft.VisualStudio.2022.BuildTools
```

**Option B: Manual Download**
1. Download [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
2. Run installer
3. Select workload: **"Desktop development with C++"**
4. Install (requires ~7GB disk space)

### 3. Install .NET SDK 8.0+

Required for Windows native layer (C# DLLs).

```powershell
winget install Microsoft.DotNet.SDK.8
```

**Verify Installation:**
```powershell
dotnet --version  # Should show 8.0.x
```

### 4. Install Rust (Optional - for performance extensions)

```powershell
winget install Rustlang.Rust.MSVC
```

After installation, restart your terminal and verify:
```powershell
rustc --version
cargo --version
```

### 5. Install Git

```powershell
winget install Git.Git
```

**Verify:**
```powershell
git --version
```

---

## Automated Installation

The fastest way to get Ironcliw running on Windows.

### Step 1: Clone Repository

```powershell
# Clone Ironcliw repository
git clone https://github.com/drussell23/Ironcliw.git
cd Ironcliw
```

### Step 2: Run Installation Script

```powershell
# Run automated installer (PowerShell as Administrator recommended)
.\scripts\windows\install_windows.ps1
```

**What the script does:**
1. Checks system prerequisites
2. Creates Python virtual environment (`.venv`)
3. Installs Python dependencies (Windows-optimized)
4. Builds C# native layer
5. Sets up configuration files
6. Creates required directories
7. Runs verification tests

**Installation Time:** ~15-20 minutes (depending on internet speed and hardware)

### Step 3: Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If you get a script execution error, allow PowerShell scripts:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: First Run

```powershell
python unified_supervisor.py --version
python unified_supervisor.py --help
```

---

## Manual Installation

If you prefer step-by-step manual installation or the automated script failed.

### Step 1: Clone Repository

```powershell
git clone https://github.com/drussell23/Ironcliw.git
cd Ironcliw
```

### Step 2: Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Python Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Install base requirements
pip install -r requirements.txt

# Install Windows-specific packages
pip install -r scripts\windows\requirements-windows.txt
```

**Windows-Specific Packages:**
- `pyaudiowpatch` - Windows audio capture (WASAPI)
- `pythonnet` - Python.NET interop for C# DLLs
- `pywin32` - Windows API bindings
- `watchdog` - File system monitoring (ReadDirectoryChangesW)
- `psutil` - Process and system utilities

### Step 4: Build C# Native Layer

```powershell
cd backend\windows_native
.\build.ps1
cd ..\..
```

**Verify DLLs were built:**
```powershell
dir backend\windows_native\bin\Release\*.dll
```

You should see:
- `SystemControl.dll`
- `ScreenCapture.dll`
- `AudioEngine.dll`

### Step 5: Set Up Configuration

```powershell
# Copy Windows environment template
copy .env.windows .env

# Copy Windows config
copy backend\config\windows_config.yaml backend\config\jarvis_config.yaml
```

**Edit `.env` file** and customize:
```bash
# Required: Set your API keys
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # Optional

# Platform detection (auto-detected, but can override)
Ironcliw_PLATFORM=windows

# Windows-specific settings
WINDOWS_AUDIO_DEVICE=default  # Or specific device name
WINDOWS_STARTUP_MODE=normal   # Options: normal, minimal, full

# Authentication (bypassed for MVP)
Ironcliw_AUTH_BYPASS=true
```

### Step 6: Create Required Directories

```powershell
mkdir $env:USERPROFILE\.jarvis
mkdir $env:USERPROFILE\.jarvis\logs
mkdir $env:USERPROFILE\.jarvis\cache
mkdir $env:USERPROFILE\.jarvis\data
mkdir $env:USERPROFILE\.jarvis\intelligence
mkdir $env:USERPROFILE\.jarvis\voice_memory
mkdir $env:USERPROFILE\.jarvis\cross_repo
```

### Step 7: Build Rust Extensions (Optional)

If you installed Rust and want performance optimizations:

```powershell
# Build main Rust extensions
cd backend\rust_extensions
cargo build --release
cd ..\..

# Build vision core
cd backend\vision\jarvis-rust-core
cargo build --release
cd ..\..\..
```

**Note:** Rust extensions have some pre-existing code issues (not Windows-specific). They're optional for core functionality.

---

## Post-Installation Setup

### 1. Configure Windows Firewall

Ironcliw needs to accept connections on:
- Port 8010 (Backend API)
- Port 3000 (Frontend UI)
- Port 8000 (Ironcliw-Prime - if using Trinity)
- Port 8090 (Reactor-Core - if using Trinity)

**Option A: Allow via Windows Defender (Recommended)**

Run as Administrator:
```powershell
New-NetFirewallRule -DisplayName "Ironcliw Backend" -Direction Inbound -LocalPort 8010 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Ironcliw Frontend" -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow
```

**Option B: Manual Configuration**
1. Open Windows Defender Firewall
2. Advanced Settings → Inbound Rules → New Rule
3. Port → TCP → Specific local ports: 8010,3000,8000,8090
4. Allow the connection → Apply to all profiles → Name: "Ironcliw"

### 2. Set Up Task Scheduler (Auto-Start)

To make Ironcliw start on boot:

```powershell
# Run as Administrator
python unified_supervisor.py --install-watchdog
```

This creates a Windows Task Scheduler task named `Ironcliw\Supervisor` that:
- Runs on system startup
- Runs on user logon
- Restarts on failure
- Uses your user account (not SYSTEM)

**Verify:**
```powershell
schtasks /Query /TN "Ironcliw\Supervisor"
```

### 3. Configure UAC (Optional)

Ironcliw may request UAC elevation for certain operations (system-level control). You can:

**Option A:** Run Ironcliw elevated (Run as Administrator)
```powershell
# Right-click PowerShell → Run as Administrator
python unified_supervisor.py
```

**Option B:** Lower UAC to "Never notify" (less secure)
- Control Panel → User Accounts → Change User Account Control settings
- Move slider to bottom → Restart required

**Option C:** Keep UAC, approve prompts as needed (recommended)

### 4. Install Frontend Dependencies (Optional)

If you want to use the web UI:

```powershell
cd frontend
npm install
npm run build  # Or 'npm run dev' for development
cd ..
```

### 5. Configure GCP (Optional - for cloud ML)

If using Google Cloud Platform for ML inference:

1. Install Google Cloud SDK:
   ```powershell
   winget install Google.CloudSDK
   ```

2. Authenticate:
   ```powershell
   gcloud auth login
   gcloud auth application-default login
   ```

3. Set project:
   ```powershell
   gcloud config set project YOUR_PROJECT_ID
   ```

4. Enable required APIs:
   ```powershell
   gcloud services enable compute.googleapis.com
   gcloud services enable sql-component.googleapis.com
   ```

---

## Verification

### Quick Test

```powershell
# Activate venv if not already
.\.venv\Scripts\Activate.ps1

# Check version
python unified_supervisor.py --version
# Output: Ironcliw Unified Supervisor v19.6.0+ (Windows x64)

# Run platform test
python unified_supervisor.py --test
# Output: Platform: windows, Python 3.12.x, All checks passed ✓

# Check health
python unified_supervisor.py --status
# Output: Shows component status
```

### Full Startup Test

```powershell
# Start Ironcliw supervisor
python unified_supervisor.py
```

**Expected Output:**
```
🧠 Ironcliw Unified Supervisor v19.6.0+
📍 Platform: Windows 11 (x64)
🐍 Python: 3.12.1
📁 Data Directory: C:\Users\<you>\.jarvis

[Phase 0: Early Protection]
✓ UTF-8 console encoding enabled
✓ Signal handlers registered (Windows-compatible)
✓ Platform detection: windows

[Phase 1: Foundation]
✓ Configuration loaded: backend\config\jarvis_config.yaml
✓ Environment variables loaded: .env
✓ Platform abstraction layer initialized

[Phase 2: Core Utilities]
✓ Unified logger initialized
✓ Startup lock acquired
...

[Backend Starting]
✓ FastAPI server initialized
✓ Backend listening on http://localhost:8010

[System Ready]
✅ Ironcliw is online!
```

### Test Backend API

In a separate terminal:

```powershell
# Test health endpoint
curl http://localhost:8010/health

# Expected: {"status":"healthy","platform":"windows","version":"19.6.0+"}

# Test command endpoint (echo test)
curl -X POST http://localhost:8010/api/command `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"test\"}"
```

### Test Frontend (if installed)

1. Start frontend dev server:
   ```powershell
   cd frontend
   npm run dev
   ```

2. Open browser: http://localhost:3000

3. You should see Ironcliw loading screen → System Ready → UI

### Test C# Native Layer

```powershell
python backend\windows_native\test_csharp_bindings.py
```

**Expected Output:**
```
Testing SystemControl...
✓ Window enumeration works
✓ Volume control works
✓ Notification works

Testing ScreenCapture...
✓ Screen capture works (1920x1080)
✓ Multi-monitor detection works (2 monitors)

Testing AudioEngine...
✓ Audio device enumeration works
✓ Recording works
✓ Playback works

All tests passed! ✓
```

---

## Troubleshooting

### Common Issues

#### 1. "Python was not found"

**Solution:** Add Python to PATH manually:
1. Find Python installation path (e.g., `C:\Users\<you>\AppData\Local\Programs\Python\Python312`)
2. Add to PATH:
   - Search "Environment Variables" in Windows
   - Edit "Path" variable
   - Add `C:\Users\<you>\AppData\Local\Programs\Python\Python312`
   - Add `C:\Users\<you>\AppData\Local\Programs\Python\Python312\Scripts`
3. Restart terminal

#### 2. "Module not found" errors

**Solution:**
```powershell
# Make sure venv is activated
.\.venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
pip install -r scripts\windows\requirements-windows.txt
```

#### 3. C# DLLs fail to load

**Error:** `clr.AddReference('SystemControl')` fails

**Solution:**
```powershell
# Check .NET SDK installed
dotnet --version

# Rebuild C# projects
cd backend\windows_native
.\build.ps1
cd ..\..

# Check pythonnet installed
pip install pythonnet
```

#### 4. "Access Denied" / Permission errors

**Solution:** Run PowerShell as Administrator or adjust UAC settings (see Post-Installation Setup #3)

#### 5. Port 8010 already in use

**Solution:**
```powershell
# Find process using port 8010
netstat -ano | findstr :8010

# Kill the process (replace <PID> with actual PID from above)
taskkill /PID <PID> /F

# Or change Ironcliw port in .env
# Ironcliw_BACKEND_PORT=8011
```

#### 6. Emoji/Unicode characters show as � or ???

**Solution:** Set UTF-8 encoding:
```powershell
# PowerShell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Or set environment variable
$env:PYTHONIOENCODING="utf-8"

# Or run Ironcliw with UTF-8
$env:PYTHONIOENCODING="utf-8"; python unified_supervisor.py
```

#### 7. Rust build fails

**Solution:** Rust extensions are optional. Skip them if you encounter build errors:
```powershell
# Ironcliw works without Rust extensions
# Just skip the Rust build step

# If you really need them, see:
# docs/windows_porting/troubleshooting.md#rust-build-issues
```

For more troubleshooting, see: [Troubleshooting Guide](./troubleshooting.md)

---

## Next Steps

### Basic Usage

```powershell
# Start Ironcliw
python unified_supervisor.py

# With frontend
# Terminal 1:
python unified_supervisor.py

# Terminal 2:
cd frontend && npm run dev
```

### Configuration

Edit `.env` and `backend\config\jarvis_config.yaml` to customize:
- API keys (Anthropic, OpenAI, GCP)
- Port numbers
- Logging levels
- Feature flags

See: [Configuration Examples](./configuration_examples.md)

### Advanced Features

- **GCP Cloud ML:** See `docs\gcp_setup.md`
- **Trinity Mode:** See `docs\trinity_setup.md`
- **Voice Authentication:** Currently bypassed on Windows (see [Known Limitations](./known_limitations.md))
- **Custom Commands:** See `docs\backend\custom_commands.md`

### Development

```powershell
# Run in dev mode with hot reload
python unified_supervisor.py --dev-mode

# Run tests
pytest tests\platform\test_windows_platform.py -v

# Build documentation
cd docs
mkdocs serve
```

### Community

- **GitHub Issues:** https://github.com/drussell23/Ironcliw/issues
- **Discussions:** https://github.com/drussell23/Ironcliw/discussions
- **Discord:** [Coming Soon]

---

## Security Notes

### Authentication Bypass (MVP)

**⚠️ Important:** The Windows port currently **bypasses voice authentication** for MVP. This means:
- Voice unlock is **disabled**
- Biometric authentication is **disabled**
- All commands are **accepted without authentication**

**Why:** ECAPA-TDNN speaker verification requires Metal (macOS) or CUDA (Linux) acceleration. Windows DirectML support is planned for a future release.

**Implications:**
- ✅ Full functionality without authentication complexity
- ⚠️ **Not suitable for production security-sensitive deployments**
- ✅ Suitable for personal use, development, testing

**Future:** Full voice biometric authentication with Windows Hello integration planned for v2.0.

### API Keys

Store API keys securely:
```powershell
# Don't commit .env to git
echo .env >> .gitignore

# Use environment variables instead of hardcoding
$env:ANTHROPIC_API_KEY="your_key"
python unified_supervisor.py
```

### Firewall

Keep Windows Firewall enabled. Only allow Ironcliw ports (8010, 3000) and only from trusted sources (localhost or your LAN).

---

## Updating Ironcliw

### Automated Update

```powershell
# Ironcliw has built-in auto-update (Zero-Touch Update System)
# Just leave it running, it will detect and apply updates automatically
```

### Manual Update

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade
pip install -r scripts\windows\requirements-windows.txt --upgrade

# Rebuild C# layer (if needed)
cd backend\windows_native
.\build.ps1
cd ..\..

# Restart Ironcliw
python unified_supervisor.py --restart
```

---

## Uninstalling Ironcliw

### Remove Software

```powershell
# Stop Ironcliw
python unified_supervisor.py --shutdown

# Remove scheduled task
schtasks /Delete /TN "Ironcliw\Supervisor" /F

# Remove virtual environment
Remove-Item -Recurse -Force .venv

# Remove user data (optional - includes logs, cache, learning data)
Remove-Item -Recurse -Force $env:USERPROFILE\.jarvis

# Remove firewall rules (optional)
Remove-NetFirewallRule -DisplayName "Ironcliw Backend"
Remove-NetFirewallRule -DisplayName "Ironcliw Frontend"

# Uninstall repository (optional)
cd ..
Remove-Item -Recurse -Force Ironcliw
```

### Keep Configuration

If you want to reinstall later, keep:
- `.env` (API keys and configuration)
- `$env:USERPROFILE\.jarvis\voice_memory\` (voice profiles - if using auth)
- `$env:USERPROFILE\.jarvis\data\` (learning data)

---

## Support

If you encounter issues not covered in this guide:

1. Check [Troubleshooting Guide](./troubleshooting.md)
2. Check [Known Limitations](./known_limitations.md)
3. Search [GitHub Issues](https://github.com/drussell23/Ironcliw/issues)
4. Open a new issue with:
   - Windows version (`winver`)
   - Python version (`python --version`)
   - Error logs (`.\.zenflow\worktrees\iron-cliw-0081\logs\supervisor.log`)
   - Steps to reproduce

---

## Congratulations!

You now have Ironcliw running on Windows! 🎉

Next steps:
1. Explore the frontend UI (http://localhost:3000)
2. Try voice commands (if you have a microphone)
3. Read the [User Guide](../guides/user_guide.md)
4. Configure advanced features

**Welcome to the Ironcliw ecosystem!** 🤖
