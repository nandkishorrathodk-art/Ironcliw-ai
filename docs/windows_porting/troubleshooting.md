# JARVIS Windows Troubleshooting Guide

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Performance Problems](#performance-problems)
4. [C# Native Layer Issues](#c-native-layer-issues)
5. [Python Environment Issues](#python-environment-issues)
6. [Network & API Issues](#network--api-issues)
7. [Frontend Issues](#frontend-issues)
8. [Platform-Specific Issues](#platform-specific-issues)
9. [Logging & Diagnostics](#logging--diagnostics)
10. [Known Issues & Workarounds](#known-issues--workarounds)

---

## Installation Issues

### Issue: Python not found

**Symptoms:**
```powershell
python : The term 'python' is not recognized...
```

**Solutions:**

1. **Check if Python is installed:**
   ```powershell
   where python
   # Or try:
   python3 --version
   py --version
   ```

2. **Add Python to PATH manually:**
   ```powershell
   # Find Python installation
   where.exe python.exe  # If installed but not in PATH

   # Add to PATH (replace with your actual path):
   $env:Path += ";C:\Users\<username>\AppData\Local\Programs\Python\Python312"
   $env:Path += ";C:\Users\<username>\AppData\Local\Programs\Python\Python312\Scripts"

   # Make permanent:
   [Environment]::SetEnvironmentVariable("Path", $env:Path, [EnvironmentVariableTarget]::User)
   ```

3. **Reinstall Python with PATH option:**
   - Download from python.org
   - **Check "Add Python to PATH"** during installation
   - Restart terminal

---

### Issue: `pip install` fails with "error: Microsoft Visual C++ 14.0 or greater is required"

**Symptoms:**
```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"
```

**Solution:**

Install Visual Studio Build Tools:
```powershell
# Via winget (recommended):
winget install Microsoft.VisualStudio.2022.BuildTools

# During installation, select:
# ✓ Desktop development with C++

# Restart terminal after installation
```

**Verify:**
```powershell
# Check if cl.exe (C++ compiler) is available
where cl.exe
```

---

### Issue: `.env` file not loading

**Symptoms:**
- API key errors: "ANTHROPIC_API_KEY not set"
- Platform defaults to "unknown"

**Solutions:**

1. **Check .env file exists:**
   ```powershell
   dir .env
   # If not found:
   copy .env.windows .env
   ```

2. **Check .env encoding (must be UTF-8, no BOM):**
   ```powershell
   # Open in notepad
   notepad .env

   # Save As → Encoding: UTF-8 (not UTF-8 BOM)
   ```

3. **Check for spaces around `=`:**
   ```bash
   # ❌ Wrong:
   ANTHROPIC_API_KEY = your_key

   # ✅ Correct:
   ANTHROPIC_API_KEY=your_key
   ```

4. **Load manually in PowerShell:**
   ```powershell
   Get-Content .env | ForEach-Object {
       if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.+?)\s*$') {
           [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
       }
   }
   ```

---

### Issue: Virtual environment activation fails

**Symptoms:**
```powershell
.\.venv\Scripts\Activate.ps1 : File cannot be loaded because running scripts is disabled...
```

**Solution:**

Allow PowerShell scripts for current user:
```powershell
# Run as Administrator or change for CurrentUser only
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verify:
Get-ExecutionPolicy -Scope CurrentUser
# Output should be: RemoteSigned

# Now retry activation:
.\.venv\Scripts\Activate.ps1
```

**Alternative (without changing execution policy):**
```powershell
# Bypass for single command:
PowerShell -ExecutionPolicy Bypass -File .\.venv\Scripts\Activate.ps1
```

---

## Runtime Errors

### Issue: "Port 8010 is already in use"

**Symptoms:**
```
OSError: [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted
```

**Solutions:**

1. **Find and kill process using port 8010:**
   ```powershell
   # Find PID using port 8010
   netstat -ano | findstr :8010
   # Output example: TCP 0.0.0.0:8010 0.0.0.0:0 LISTENING 12345

   # Kill process (replace 12345 with actual PID)
   taskkill /PID 12345 /F
   ```

2. **Change JARVIS port:**
   ```bash
   # Edit .env file
   JARVIS_BACKEND_PORT=8011

   # Restart JARVIS
   python unified_supervisor.py --restart
   ```

3. **Check for zombie JARVIS processes:**
   ```powershell
   Get-Process | Where-Object {$_.ProcessName -like "*python*"}

   # Kill all Python processes (use with caution!)
   Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force
   ```

---

### Issue: ModuleNotFoundError: No module named 'backend'

**Symptoms:**
```
ModuleNotFoundError: No module named 'backend'
ModuleNotFoundError: No module named 'backend.platform'
```

**Solutions:**

1. **Ensure you're in the JARVIS root directory:**
   ```powershell
   cd C:\path\to\JARVIS
   # Should contain: unified_supervisor.py, backend/, frontend/, etc.
   ```

2. **Check PYTHONPATH:**
   ```powershell
   # Add current directory to PYTHONPATH
   $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"

   # Or set permanently:
   [Environment]::SetEnvironmentVariable("PYTHONPATH", "$PWD", 'User')
   ```

3. **Reinstall in editable mode:**
   ```powershell
   pip install -e .
   ```

4. **Check virtual environment activated:**
   ```powershell
   # You should see (.venv) prefix in prompt
   .\.venv\Scripts\Activate.ps1
   ```

---

### Issue: UnicodeEncodeError on Windows console

**Symptoms:**
```
UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f4a1' in position 45: character maps to <undefined>
```

**Solutions:**

1. **Set UTF-8 encoding globally:**
   ```powershell
   # PowerShell
   $OutputEncoding = [System.Text.Encoding]::UTF8
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8

   # CMD
   chcp 65001
   ```

2. **Set PYTHONIOENCODING:**
   ```powershell
   $env:PYTHONIOENCODING="utf-8"
   python unified_supervisor.py
   ```

3. **Use Windows Terminal (recommended):**
   - Install Windows Terminal from Microsoft Store
   - Supports UTF-8 by default
   - Better emoji rendering

4. **Disable emoji logging:**
   ```yaml
   # backend/config/jarvis_config.yaml
   logging:
     use_emoji: false
   ```

---

### Issue: "Access is denied" / PermissionError

**Symptoms:**
```
PermissionError: [WinError 5] Access is denied
```

**Solutions:**

1. **Run as Administrator:**
   - Right-click PowerShell → "Run as administrator"
   - Run JARVIS

2. **Check UAC settings:**
   - Control Panel → User Accounts → Change User Account Control settings
   - Lower to "Notify me only when apps try to make changes..."

3. **Check file permissions:**
   ```powershell
   # Give full control to current user
   icacls "C:\Users\<username>\.jarvis" /grant:r "$env:USERNAME:(OI)(CI)F" /T
   ```

4. **Disable Windows Defender real-time protection temporarily:**
   - Windows Security → Virus & threat protection → Manage settings
   - Turn off "Real-time protection"
   - Try installation again
   - Re-enable after installation

---

## Performance Problems

### Issue: High CPU usage (>80%)

**Symptoms:**
- JARVIS uses 4+ CPU cores at 100%
- System becomes slow/unresponsive

**Diagnosis:**
```powershell
# Check CPU usage
Get-Process python | Select-Object ProcessName, CPU, Handles

# Check which Python threads are active
python -c "from backend.platform import get_platform_info; print(get_platform_info())"
```

**Solutions:**

1. **Reduce concurrent workers:**
   ```yaml
   # backend/config/jarvis_config.yaml
   performance:
     max_workers: 2  # Down from 4
     vision_fps: 10  # Down from 15
   ```

2. **Disable CPU-intensive features:**
   ```bash
   # .env
   JARVIS_VISION_ENABLED=false  # Disable vision if not needed
   JARVIS_CONTINUOUS_LEARNING=false
   ```

3. **Set CPU affinity:**
   ```powershell
   # Limit JARVIS to specific CPU cores (e.g., cores 0-3)
   $process = Get-Process python | Where-Object {$_.MainWindowTitle -like "*JARVIS*"}
   $process.ProcessorAffinity = 0x0F  # Binary: 0000 1111 (cores 0-3)
   ```

---

### Issue: High memory usage (>8GB)

**Symptoms:**
- JARVIS uses 8+ GB RAM
- System swapping to disk

**Solutions:**

1. **Enable memory-aware mode:**
   ```bash
   # .env
   JARVIS_MEMORY_MODE=efficient  # Options: normal, efficient, minimal
   ```

2. **Reduce cache sizes:**
   ```yaml
   # backend/config/jarvis_config.yaml
   memory:
     max_cache_size_mb: 512  # Down from 2048
     vision_buffer_frames: 30  # Down from 60
   ```

3. **Disable ML models (use cloud inference):**
   ```bash
   # .env
   JARVIS_LOCAL_ML_ENABLED=false
   JARVIS_USE_CLOUD_ML=true
   ```

4. **Monitor memory:**
   ```powershell
   # Watch memory usage
   while ($true) {
       Get-Process python | Select-Object ProcessName, @{Name='MemoryMB';Expression={[math]::Round($_.WS / 1MB, 2)}}
       Start-Sleep -Seconds 5
   }
   ```

---

### Issue: Slow startup (>2 minutes)

**Symptoms:**
- JARVIS takes >120 seconds to start
- Loading page stuck at certain percentage

**Solutions:**

1. **Check disk I/O:**
   ```powershell
   # Use Task Manager → Performance → Disk
   # Or use Resource Monitor (resmon.exe)
   ```

2. **Enable fast startup mode:**
   ```bash
   # .env
   FAST_START=true
   JARVIS_SKIP_HEAVY_INIT=true
   ```

3. **Disable slow components:**
   ```bash
   # .env
   JARVIS_NEURAL_MESH_ENABLED=false
   JARVIS_AGI_OS_ENABLED=false
   ```

4. **Use SSD instead of HDD:**
   - Move `.jarvis` directory to SSD
   - Set `JARVIS_DATA_DIR` in .env

---

## C# Native Layer Issues

### Issue: C# DLLs not found

**Symptoms:**
```python
clr.AddReference('SystemControl')
System.IO.FileNotFoundException: Unable to find assembly 'SystemControl'
```

**Solutions:**

1. **Verify DLLs exist:**
   ```powershell
   dir backend\windows_native\bin\Release\*.dll

   # Expected output:
   # SystemControl.dll
   # ScreenCapture.dll
   # AudioEngine.dll
   ```

2. **Rebuild C# projects:**
   ```powershell
   cd backend\windows_native
   .\build.ps1
   cd ..\..
   ```

3. **Check .NET SDK version:**
   ```powershell
   dotnet --version
   # Should be 8.0.x or higher

   # If not installed:
   winget install Microsoft.DotNet.SDK.8
   ```

4. **Manual build:**
   ```powershell
   cd backend\windows_native
   dotnet build -c Release
   cd ..\..
   ```

---

### Issue: pythonnet import fails

**Symptoms:**
```python
ModuleNotFoundError: No module named 'clr'
```

**Solutions:**

1. **Install pythonnet:**
   ```powershell
   pip install pythonnet
   ```

2. **Check Python version compatibility:**
   ```powershell
   python --version
   # pythonnet requires Python 3.7-3.12
   ```

3. **Reinstall with specific version:**
   ```powershell
   pip uninstall pythonnet
   pip install pythonnet==3.0.3
   ```

---

### Issue: C# methods fail silently

**Symptoms:**
- `SystemControl.SetVolume()` does nothing
- No errors, but operations don't work

**Solutions:**

1. **Check Windows API permissions:**
   - Some APIs require elevated privileges (Run as Administrator)

2. **Enable debug logging:**
   ```bash
   # .env
   WINDOWS_NATIVE_DEBUG=true
   ```

3. **Test C# layer directly:**
   ```powershell
   python backend\windows_native\test_csharp_bindings.py
   ```

4. **Check Windows version compatibility:**
   ```powershell
   winver
   # Should be Windows 10 1809+ or Windows 11
   ```

---

## Python Environment Issues

### Issue: Multiple Python versions conflict

**Symptoms:**
- `python` runs Python 2.7
- `python3` command not found
- Packages installed to wrong Python version

**Solutions:**

1. **Use Python Launcher (`py`):**
   ```powershell
   # List installed Python versions
   py -0

   # Use specific version
   py -3.12 unified_supervisor.py

   # Create venv with specific version
   py -3.12 -m venv .venv
   ```

2. **Set specific Python in PATH:**
   ```powershell
   # Remove old Python from PATH, add new one first
   # System → Advanced → Environment Variables
   # Edit Path, move Python 3.12 to top
   ```

3. **Create alias:**
   ```powershell
   # PowerShell profile
   Set-Alias python "C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe"
   ```

---

### Issue: Package installation fails

**Symptoms:**
```
ERROR: Could not build wheels for <package> which is required to install pyproject.toml-based projects
```

**Solutions:**

1. **Install build dependencies:**
   ```powershell
   pip install --upgrade pip setuptools wheel
   pip install --upgrade build
   ```

2. **Use pre-built wheels:**
   ```powershell
   # For packages like numpy, scipy, etc.
   pip install --only-binary :all: numpy scipy
   ```

3. **Install from conda-forge (if using Anaconda):**
   ```powershell
   conda install -c conda-forge <package>
   ```

4. **Check for Windows-specific package names:**
   ```powershell
   # Example: pyaudio vs pyaudiowpatch
   pip install pyaudiowpatch  # Windows WASAPI version
   ```

---

## Network & API Issues

### Issue: API key not recognized

**Symptoms:**
```
anthropic.AuthenticationError: invalid x-api-key
```

**Solutions:**

1. **Check API key format:**
   ```bash
   # .env
   ANTHROPIC_API_KEY=sk-ant-api03-...  # Should start with sk-ant-
   OPENAI_API_KEY=sk-proj-...  # Should start with sk-proj- or sk-
   ```

2. **Check for extra spaces/newlines:**
   ```powershell
   # Trim whitespace
   $key = (Get-Content .env | Select-String "ANTHROPIC_API_KEY").Line.Split('=')[1].Trim()
   echo "Key length: $($key.Length)"  # Should be ~100+ chars
   ```

3. **Test API key directly:**
   ```powershell
   curl https://api.anthropic.com/v1/messages `
     -H "x-api-key: YOUR_KEY" `
     -H "Content-Type: application/json" `
     -d "{\"model\":\"claude-3-5-sonnet-20241022\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":10}"
   ```

---

### Issue: GCP authentication fails

**Symptoms:**
```
google.auth.exceptions.DefaultCredentialsError: Could not automatically determine credentials
```

**Solutions:**

1. **Install Google Cloud SDK:**
   ```powershell
   winget install Google.CloudSDK
   ```

2. **Authenticate:**
   ```powershell
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Set project:**
   ```powershell
   gcloud config set project YOUR_PROJECT_ID
   ```

4. **Set credentials path manually:**
   ```bash
   # .env
   GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account-key.json
   ```

---

## Frontend Issues

### Issue: Frontend won't build

**Symptoms:**
```
npm ERR! missing script: build
```

**Solutions:**

1. **Install Node.js (if not installed):**
   ```powershell
   winget install OpenJS.NodeJS.LTS
   ```

2. **Install dependencies:**
   ```powershell
   cd frontend
   npm install
   ```

3. **Clear npm cache:**
   ```powershell
   npm cache clean --force
   rm -r node_modules
   rm package-lock.json
   npm install
   ```

4. **Use specific Node version (if version conflicts):**
   ```powershell
   # Install nvm-windows first
   nvm install 20
   nvm use 20
   npm install
   ```

---

### Issue: Frontend can't connect to backend

**Symptoms:**
- Frontend shows "Connection Error" or "Not connected to JARVIS"
- Browser console: `WebSocket connection failed`

**Solutions:**

1. **Check backend is running:**
   ```powershell
   curl http://localhost:8010/health
   # Should return: {"status":"healthy",...}
   ```

2. **Check CORS settings:**
   ```yaml
   # backend/config/jarvis_config.yaml
   api:
     cors_enabled: true
     cors_origins:
       - "http://localhost:3000"
       - "http://127.0.0.1:3000"
   ```

3. **Check frontend config:**
   ```javascript
   // frontend/src/config.js
   const config = {
     apiUrl: 'http://localhost:8010',  // Match backend port
     wsUrl: 'ws://localhost:8010'
   };
   ```

4. **Check Windows Firewall:**
   ```powershell
   # Allow backend port
   New-NetFirewallRule -DisplayName "JARVIS Backend" -Direction Inbound -LocalPort 8010 -Protocol TCP -Action Allow
   ```

---

## Platform-Specific Issues

### Issue: Hash-based vs file watcher hot reload

**Symptoms:**
- Hot reload doesn't detect file changes immediately
- Delays of 10+ seconds

**Solutions:**

The Windows implementation uses **hash-based detection** (every 10s) instead of file watchers (inotify/FSEvents).

1. **Reduce check interval:**
   ```yaml
   # backend/config/jarvis_config.yaml
   hot_reload:
     check_interval_seconds: 5  # Down from 10
   ```

2. **Use watchdog for file events (alternative):**
   ```bash
   # .env
   JARVIS_HOT_RELOAD_METHOD=watchdog  # Uses ReadDirectoryChangesW
   ```

3. **Disable hot reload in production:**
   ```bash
   # .env
   JARVIS_DEV_MODE=false
   ```

---

### Issue: Task Scheduler watchdog not working

**Symptoms:**
- JARVIS doesn't auto-restart after crash
- Scheduled task not running

**Solutions:**

1. **Verify task exists:**
   ```powershell
   schtasks /Query /TN "JARVIS\Supervisor"
   ```

2. **Check task status:**
   ```powershell
   Get-ScheduledTask -TaskPath "\JARVIS\" -TaskName "Supervisor" | Get-ScheduledTaskInfo
   ```

3. **Reinstall watchdog:**
   ```powershell
   # Run as Administrator
   python unified_supervisor.py --uninstall-watchdog
   python unified_supervisor.py --install-watchdog
   ```

4. **Run task manually:**
   ```powershell
   schtasks /Run /TN "JARVIS\Supervisor"
   ```

5. **Check task XML:**
   ```powershell
   schtasks /Query /TN "JARVIS\Supervisor" /XML
   ```

---

## Logging & Diagnostics

### Enable Debug Logging

```bash
# .env
LOG_LEVEL=DEBUG
JARVIS_VERBOSE=true
```

### Check Logs

```powershell
# Supervisor logs
Get-Content logs\supervisor.log -Tail 50 -Wait

# Backend logs
Get-Content $env:USERPROFILE\.jarvis\logs\backend.log -Tail 50 -Wait

# Error logs only
Get-Content logs\supervisor.log | Select-String "ERROR"
```

### Generate Diagnostic Report

```powershell
python unified_supervisor.py --diagnose

# Output: Creates diagnostics.json with:
# - Platform info
# - Python version
# - Installed packages
# - Running processes
# - Port status
# - Log excerpts
```

### Test Individual Components

```powershell
# Test platform detection
python -c "from backend.platform import get_platform, get_platform_info; print(f'Platform: {get_platform()}'); print(get_platform_info())"

# Test C# bindings
python backend\windows_native\test_csharp_bindings.py

# Test audio
python -c "from backend.platform.windows.audio import WindowsAudioEngine; print('Audio OK')"

# Test vision
python -c "from backend.platform.windows.vision import WindowsVisionCapture; print('Vision OK')"

# Test backend imports
python -c "from backend.platform.windows import *; print('All imports OK')"
```

---

## Known Issues & Workarounds

### 1. Voice Authentication Bypassed

**Issue:** Windows port disables ECAPA-TDNN voice biometric authentication.

**Reason:** Speaker verification requires GPU acceleration (Metal on macOS, CUDA on Linux). DirectML support planned for future.

**Workaround:** Use Windows Hello or password authentication at OS level.

**Status:** Won't Fix (for MVP). Full support in v2.0.

---

### 2. Rust Extensions Build Fails

**Issue:** Rust extensions have pre-existing code issues (not Windows-specific):
- `sysinfo` v0.30 API changes
- `memmap2` missing dependency
- `rayon` API changes
- PyO3 binding issues

**Workaround:** Rust extensions are **optional**. Core functionality works without them.

**Status:** Can be fixed (15-30 min), but outside Windows porting scope.

---

### 3. Emoji Rendering in Logs

**Issue:** Windows console (cmd.exe) doesn't render emojis correctly.

**Workarounds:**
- Use Windows Terminal (recommended)
- Set `PYTHONIOENCODING=utf-8`
- Disable emojis: `logging.use_emoji: false` in config

**Status:** Won't Fix (Windows Terminal solves this).

---

### 4. Slower Hot Reload Than macOS

**Issue:** File change detection uses hash comparison (every 10s) instead of FSEvents (instant).

**Reason:** Windows ReadDirectoryChangesW is less reliable than macOS FSEvents for large codebases.

**Workaround:** Reduce `check_interval_seconds` to 5s (more CPU, faster detection).

**Status:** Working as intended. Performance trade-off.

---

### 5. GCP Features Untested on Windows

**Issue:** GCP VM manager, Cloud SQL proxy, Spot VM orchestration not fully tested on Windows.

**Status:** Should work (Python code is cross-platform), but needs testing.

**Help Wanted:** Community testing and feedback appreciated!

---

## Getting More Help

### Escalation Path

1. ✅ Check this troubleshooting guide
2. ✅ Check [Known Limitations](./known_limitations.md)
3. ✅ Check [Setup Guide](./setup_guide.md)
4. ✅ Search [GitHub Issues](https://github.com/drussell23/JARVIS/issues)
5. ✅ Ask in [GitHub Discussions](https://github.com/drussell23/JARVIS/discussions)
6. ✅ Open a [new GitHub Issue](https://github.com/drussell23/JARVIS/issues/new)

### Issue Template

When reporting issues, include:

```
**Environment:**
- Windows Version: (run: winver)
- Python Version: (run: python --version)
- JARVIS Version: (run: python unified_supervisor.py --version)

**Error Message:**
(paste full error traceback)

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. ...

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Logs:**
(attach relevant log files from logs/ or %USERPROFILE%\.jarvis\logs\)

**Diagnostic Info:**
(run: python unified_supervisor.py --diagnose)
(attach diagnostics.json)
```

---

## Community Contributions

Found a fix not listed here? **Contribute it!**

1. Fork the repository
2. Add your fix to this troubleshooting guide
3. Open a Pull Request
4. Help others facing the same issue!

---

**Last Updated:** February 2026  
**Windows Port Version:** 1.0.0-MVP
