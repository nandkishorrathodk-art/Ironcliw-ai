# Installation Guide - Windows Native Layer

## Prerequisites

Before building the C# Windows Native Layer, you need to install the following:

### 1. .NET SDK 8.0 or later

#### Option A: Using winget (Recommended)
```powershell
winget install Microsoft.DotNet.SDK.8
```

#### Option B: Using Chocolatey
```powershell
choco install dotnet-sdk
```

#### Option C: Manual Download
1. Visit https://dotnet.microsoft.com/download
2. Download .NET 8.0 SDK (or later)
3. Run the installer
4. Restart your terminal

#### Verify Installation
```bash
dotnet --version
# Should output: 8.0.x or higher
```

### 2. Python 3.11+

Already required by JARVIS. Verify:
```bash
python --version
# Should output: 3.11.x or higher
```

### 3. pythonnet (Python.NET)

Install using pip:
```bash
pip install pythonnet
```

This package allows Python to call .NET/C# assemblies.

#### Verify Installation
```bash
python -c "import clr; print('pythonnet OK')"
# Should output: pythonnet OK
```

## Quick Start

### Step 1: Install .NET SDK
```powershell
# Using winget (recommended)
winget install Microsoft.DotNet.SDK.8

# Verify
dotnet --version
```

### Step 2: Install pythonnet
```bash
pip install pythonnet
```

### Step 3: Build C# Projects
```powershell
cd backend\windows_native
.\build.ps1
```

Or manually:
```bash
cd backend\windows_native
dotnet restore
dotnet build -c Release
```

### Step 4: Test
```bash
python test_csharp_bindings.py
```

Expected output:
```
============================================================
C# Windows Native Layer - Python Bindings Test
============================================================
✅ pythonnet is installed

=== Testing SystemControl ===
✅ SystemControl tests passed!

=== Testing ScreenCapture ===
✅ ScreenCapture tests passed!

=== Testing AudioEngine ===
✅ AudioEngine tests passed!

🎉 All tests passed!
```

## Detailed Installation Steps

### Install .NET SDK (Detailed)

#### Windows 11/10

1. **Download**:
   - Visit: https://dotnet.microsoft.com/download/dotnet/8.0
   - Click "Download .NET SDK x64" (or ARM64 for ARM devices)

2. **Install**:
   - Run the downloaded `.exe` installer
   - Follow the installation wizard
   - Choose "Install for all users" (recommended)

3. **Verify**:
   ```bash
   # Open new terminal (important - to refresh PATH)
   dotnet --version
   dotnet --list-sdks
   ```

4. **Troubleshooting**:
   - If `dotnet` command not found, restart your terminal/computer
   - Check that `C:\Program Files\dotnet` is in your PATH
   - Run: `$env:Path` in PowerShell to verify

#### Using Package Managers

**winget** (Windows 10 1809+):
```powershell
winget search dotnet-sdk
winget install Microsoft.DotNet.SDK.8
```

**Chocolatey**:
```powershell
# Install Chocolatey first if needed: https://chocolatey.org/install
choco install dotnet-sdk
```

**Scoop**:
```powershell
scoop bucket add main
scoop install dotnet-sdk
```

### Install pythonnet (Detailed)

1. **Using pip** (recommended):
   ```bash
   pip install pythonnet
   ```

2. **Using conda** (if using Anaconda):
   ```bash
   conda install -c conda-forge pythonnet
   ```

3. **From source** (advanced):
   ```bash
   git clone https://github.com/pythonnet/pythonnet
   cd pythonnet
   pip install .
   ```

4. **Verify**:
   ```bash
   python -c "import clr; print(clr.__version__)"
   ```

## Build Options

### Option 1: PowerShell Build Script (Recommended)
```powershell
cd backend\windows_native

# Basic build
.\build.ps1

# Clean build
.\build.ps1 -Clean

# Build and test
.\build.ps1 -Test

# Verbose output
.\build.ps1 -Verbose

# All options
.\build.ps1 -Clean -Test -Verbose
```

### Option 2: Manual Build
```bash
cd backend\windows_native

# Restore dependencies
dotnet restore

# Build all projects
dotnet build -c Release

# Build individual project
dotnet build SystemControl\SystemControl.csproj -c Release
dotnet build ScreenCapture\ScreenCapture.csproj -c Release
dotnet build AudioEngine\AudioEngine.csproj -c Release

# Clean previous builds
dotnet clean -c Release
```

### Option 3: Visual Studio (if installed)
1. Open `JarvisWindowsNative.sln` in Visual Studio
2. Select "Release" configuration
3. Build → Build Solution (Ctrl+Shift+B)

## Troubleshooting

### .NET SDK Issues

**Problem**: `dotnet: command not found`
- **Solution**: .NET SDK not installed or not in PATH
  - Install .NET SDK (see above)
  - Restart terminal/computer
  - Verify PATH: `echo $env:Path` (PowerShell) or `echo %PATH%` (CMD)

**Problem**: `It was not possible to find any installed .NET SDKs`
- **Solution**: Wrong SDK version
  - Check version: `dotnet --list-sdks`
  - Need SDK 8.0.x or higher
  - Reinstall if necessary

**Problem**: `The SDK 'Microsoft.NET.Sdk' specified could not be found`
- **Solution**: Corrupted installation
  - Uninstall .NET SDK
  - Delete `C:\Program Files\dotnet`
  - Reinstall .NET SDK

### pythonnet Issues

**Problem**: `No module named 'clr'`
- **Solution**: pythonnet not installed
  ```bash
  pip install pythonnet
  ```

**Problem**: `ImportError: DLL load failed`
- **Solution**: Missing .NET runtime
  - Install .NET SDK (includes runtime)
  - Or install .NET Runtime: https://dotnet.microsoft.com/download/dotnet/8.0

**Problem**: `FileNotFoundException: Could not load assembly`
- **Solution**: DLL path incorrect
  - Check DLL exists: `ls SystemControl\bin\Release\net8.0-windows\SystemControl.dll`
  - Verify path in Python code
  - Rebuild: `dotnet build -c Release`

### Build Issues

**Problem**: `error NU1101: Unable to find package`
- **Solution**: NuGet package not found
  ```bash
  dotnet restore --force
  dotnet nuget locals all --clear
  dotnet restore
  ```

**Problem**: `error CS0006: Metadata file could not be found`
- **Solution**: Dependency build failed
  ```bash
  dotnet clean
  dotnet restore
  dotnet build -c Release
  ```

**Problem**: `error CS0234: The type or namespace name does not exist`
- **Solution**: Missing using directive or NuGet package
  - Check .csproj has correct PackageReference
  - Run `dotnet restore`

### Runtime Issues

**Problem**: `System.PlatformNotSupportedException`
- **Solution**: Running on non-Windows platform
  - This code only works on Windows
  - Check target framework in .csproj

**Problem**: `System.UnauthorizedAccessException`
- **Solution**: Insufficient permissions
  - Run as Administrator (some operations require elevated privileges)
  - Check file/folder permissions

**Problem**: `System.DllNotFoundException: Unable to load DLL`
- **Solution**: Missing native dependency
  - Install Visual C++ Redistributable
  - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

## System Requirements

- **OS**: Windows 10 version 1809 or later, Windows 11 (recommended)
- **RAM**: 4GB minimum, 8GB+ recommended
- **.NET**: SDK 8.0 or later
- **Python**: 3.11 or later
- **Disk Space**: ~2GB for .NET SDK, ~100MB for NuGet packages

## Next Steps

After successful installation and build:

1. ✅ **Phase 2 Complete** - Windows Native Layer built
2. ➡️ **Phase 3** - Implement Python wrappers in `backend/platform/windows/`
3. **Phase 4** - Port Rust extensions
4. **Phase 5** - Port unified_supervisor.py

See `plan.md` for the complete porting roadmap.

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review build logs for specific errors
3. Check README.md for API documentation
4. Verify all prerequisites are installed
5. Try a clean build: `.\build.ps1 -Clean`

## Additional Resources

- [.NET Download](https://dotnet.microsoft.com/download)
- [.NET SDK Documentation](https://docs.microsoft.com/en-us/dotnet/)
- [pythonnet Documentation](https://pythonnet.github.io/)
- [NAudio Documentation](https://github.com/naudio/NAudio)
- [Windows API Documentation](https://docs.microsoft.com/en-us/windows/win32/)
