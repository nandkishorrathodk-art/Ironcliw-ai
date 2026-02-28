# Ironcliw Windows Installation Script
# ═══════════════════════════════════════════════════════════════════════════════
# Automated setup for Ironcliw AI Assistant on Windows 10/11
# 
# This script:
#   1. Checks system requirements (Python 3.11+, RAM, disk space)
#   2. Creates Python virtual environment
#   3. Installs dependencies (Python packages, .NET SDK, Rust)
#   4. Configures environment variables
#   5. Sets up directories and permissions
#   6. Runs initial platform detection test
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File install_windows.ps1
#
# Author: Ironcliw System
# Version: 1.0.0 (Windows Port Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════

# Requires PowerShell 5.1 or higher
#Requires -Version 5.1

# Enable strict mode
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor White
    Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Text)
    Write-Host "✓ $Text" -ForegroundColor Green
}

function Write-Error {
    param([string]$Text)
    Write-Host "✗ $Text" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Text)
    Write-Host "⚠ $Text" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Text)
    Write-Host "ℹ $Text" -ForegroundColor Cyan
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-PythonVersion {
    param([string]$MinVersion = "3.11.0")
    
    try {
        $pythonVersion = & python --version 2>&1 | Select-String -Pattern "Python (\d+\.\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
        
        if (-not $pythonVersion) {
            return $false
        }
        
        $current = [Version]$pythonVersion
        $required = [Version]$MinVersion
        
        return $current -ge $required
    }
    catch {
        return $false
    }
}

function Get-SystemInfo {
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $gpu = Get-CimInstance Win32_VideoController | Select-Object -First 1
    
    return @{
        OSName = $os.Caption
        OSVersion = $os.Version
        OSArchitecture = $os.OSArchitecture
        TotalRAMGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
        FreeRAMGB = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        CPUName = $cpu.Name
        CPUCores = $cpu.NumberOfCores
        GPUName = $gpu.Name
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

Write-Header "Ironcliw Windows Installation"

# Get project root (where this script is located)
$scriptPath = $PSScriptRoot
$projectRoot = Split-Path (Split-Path $scriptPath -Parent) -Parent
Set-Location $projectRoot

Write-Info "Project root: $projectRoot"

# Check administrator rights
if (Test-Administrator) {
    Write-Warning "Running as Administrator. This is not required for installation."
    Write-Warning "For security, consider running without elevation."
}

# ───────────────────────────────────────────────────────────────────────────────
# STEP 1: System Requirements Check
# ───────────────────────────────────────────────────────────────────────────────

Write-Header "Step 1: System Requirements Check"

$sysInfo = Get-SystemInfo

Write-Info "System Information:"
Write-Host "  OS: $($sysInfo.OSName)"
Write-Host "  Version: $($sysInfo.OSVersion)"
Write-Host "  Architecture: $($sysInfo.OSArchitecture)"
Write-Host "  CPU: $($sysInfo.CPUName) ($($sysInfo.CPUCores) cores)"
Write-Host "  GPU: $($sysInfo.GPUName)"
Write-Host "  RAM: $($sysInfo.TotalRAMGB) GB (Free: $($sysInfo.FreeRAMGB) GB)"
Write-Host ""

# Check RAM (minimum 16GB)
if ($sysInfo.TotalRAMGB -lt 15) {
    Write-Error "Insufficient RAM: $($sysInfo.TotalRAMGB) GB (minimum 16GB required)"
    Write-Warning "Ironcliw may not run optimally with less than 16GB RAM"
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne 'y') {
        exit 1
    }
}
else {
    Write-Success "RAM check passed: $($sysInfo.TotalRAMGB) GB"
}

# Check disk space (minimum 20GB free)
$drive = Get-PSDrive -Name C
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
if ($freeSpaceGB -lt 20) {
    Write-Error "Insufficient disk space: $freeSpaceGB GB (minimum 20GB required)"
    exit 1
}
else {
    Write-Success "Disk space check passed: $freeSpaceGB GB free"
}

# Check Python version
Write-Info "Checking Python installation..."
if (Test-PythonVersion -MinVersion "3.11.0") {
    $pythonVersion = & python --version 2>&1
    Write-Success "Python check passed: $pythonVersion"
}
else {
    Write-Error "Python 3.11+ not found or version too old"
    Write-Info "Please install Python 3.11 or newer from https://www.python.org/"
    Write-Info "Make sure to check 'Add Python to PATH' during installation"
    exit 1
}

# Check Git
Write-Info "Checking Git installation..."
try {
    $gitVersion = & git --version 2>&1
    Write-Success "Git check passed: $gitVersion"
}
catch {
    Write-Warning "Git not found. You may need it for updates."
    Write-Info "Install from https://git-scm.com/download/win"
}

# ───────────────────────────────────────────────────────────────────────────────
# STEP 2: Python Virtual Environment
# ───────────────────────────────────────────────────────────────────────────────

Write-Header "Step 2: Python Virtual Environment"

$venvPath = Join-Path $projectRoot "venv"

if (Test-Path $venvPath) {
    Write-Warning "Virtual environment already exists at: $venvPath"
    $recreate = Read-Host "Recreate venv? This will delete existing packages. (y/N)"
    if ($recreate -eq 'y') {
        Write-Info "Removing existing venv..."
        Remove-Item -Recurse -Force $venvPath
        Write-Success "Existing venv removed"
    }
    else {
        Write-Info "Keeping existing venv"
    }
}

if (-not (Test-Path $venvPath)) {
    Write-Info "Creating virtual environment..."
    & python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment"
        exit 1
    }
    Write-Success "Virtual environment created"
}

# Activate venv
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
Write-Info "Activating virtual environment..."
& $activateScript

# Upgrade pip
Write-Info "Upgrading pip..."
& python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Failed to upgrade pip (non-fatal)"
}
else {
    Write-Success "pip upgraded"
}

# ───────────────────────────────────────────────────────────────────────────────
# STEP 3: Python Dependencies
# ───────────────────────────────────────────────────────────────────────────────

Write-Header "Step 3: Python Dependencies"

# Check if requirements-windows.txt exists
$reqWindowsPath = Join-Path $scriptPath "requirements-windows.txt"
$reqPath = Join-Path $projectRoot "requirements.txt"

if (Test-Path $reqWindowsPath) {
    Write-Info "Installing Windows-specific dependencies..."
    & python -m pip install -r $reqWindowsPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install Windows dependencies"
        exit 1
    }
    Write-Success "Windows dependencies installed"
}

if (Test-Path $reqPath) {
    Write-Info "Installing general dependencies (this may take 5-10 minutes)..."
    & python -m pip install -r $reqPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install dependencies"
        exit 1
    }
    Write-Success "General dependencies installed"
}
else {
    Write-Warning "requirements.txt not found, skipping general dependencies"
}

# ───────────────────────────────────────────────────────────────────────────────
# STEP 4: Environment Configuration
# ───────────────────────────────────────────────────────────────────────────────

Write-Header "Step 4: Environment Configuration"

$envPath = Join-Path $projectRoot ".env"
$envWindowsPath = Join-Path $projectRoot ".env.windows"

if (-not (Test-Path $envPath)) {
    if (Test-Path $envWindowsPath) {
        Write-Info "Copying .env.windows to .env..."
        Copy-Item $envWindowsPath $envPath
        Write-Success ".env file created from .env.windows template"
        Write-Warning "Please edit .env and add your API keys (Claude, OpenAI, GCP)"
    }
    else {
        Write-Warning ".env.windows template not found"
        Write-Info "Creating minimal .env file..."
        
        $minimalEnv = @"
# Ironcliw Windows Environment
Ironcliw_PLATFORM=windows
Ironcliw_AUTH_MODE=BYPASS
Ironcliw_DEV_MODE=true
Ironcliw_VERBOSE_PLATFORM=true
PYTHONPATH=$projectRoot
"@
        $minimalEnv | Out-File -FilePath $envPath -Encoding UTF8
        Write-Success "Minimal .env file created"
    }
}
else {
    Write-Info ".env file already exists"
}

# ───────────────────────────────────────────────────────────────────────────────
# STEP 5: Directory Structure
# ───────────────────────────────────────────────────────────────────────────────

Write-Header "Step 5: Directory Structure"

$directories = @(
    "$env:LOCALAPPDATA\Ironcliw",
    "$env:LOCALAPPDATA\Ironcliw\logs",
    "$env:LOCALAPPDATA\Ironcliw\models",
    "$env:LOCALAPPDATA\Ironcliw\state",
    "$env:LOCALAPPDATA\Ironcliw\state\trinity",
    "$env:LOCALAPPDATA\Ironcliw\state\cross_repo",
    "$env:LOCALAPPDATA\Ironcliw\state\signals",
    "$env:APPDATA\Ironcliw"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Success "Created: $dir"
    }
    else {
        Write-Info "Exists: $dir"
    }
}

# ───────────────────────────────────────────────────────────────────────────────
# STEP 6: Optional Components
# ───────────────────────────────────────────────────────────────────────────────

Write-Header "Step 6: Optional Components"

# .NET SDK (for C# native layer - Phase 2)
Write-Info "Checking .NET SDK..."
try {
    $dotnetVersion = & dotnet --version 2>&1
    Write-Success ".NET SDK found: $dotnetVersion"
}
catch {
    Write-Warning ".NET SDK not found (required for Phase 2 - Windows native layer)"
    Write-Info "Install from: https://dotnet.microsoft.com/download"
}

# Rust (for Rust extensions)
Write-Info "Checking Rust..."
try {
    $rustVersion = & rustc --version 2>&1
    Write-Success "Rust found: $rustVersion"
}
catch {
    Write-Warning "Rust not found (required for Rust extensions)"
    Write-Info "Install from: https://www.rust-lang.org/tools/install"
}

# Visual Studio Build Tools (for native compilation)
Write-Info "Checking Visual Studio Build Tools..."
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    Write-Success "Visual Studio Build Tools found"
}
else {
    Write-Warning "Visual Studio Build Tools not found (may be needed for some packages)"
    Write-Info "Install from: https://visualstudio.microsoft.com/downloads/"
    Write-Info "Select 'Desktop development with C++' workload"
}

# ───────────────────────────────────────────────────────────────────────────────
# STEP 7: Platform Detection Test
# ───────────────────────────────────────────────────────────────────────────────

Write-Header "Step 7: Platform Detection Test"

Write-Info "Testing platform detection..."
$testScript = @"
from backend.platform import get_platform, get_platform_info

platform = get_platform()
info = get_platform_info()

print(f'Platform: {platform}')
print(f'OS: {info.os_release}')
print(f'Architecture: {info.architecture}')
print(f'Python: {info.python_version}')
print(f'GPU: {info.has_gpu}')
print(f'NPU: {info.has_npu}')
print(f'DirectML: {info.has_directml}')

assert platform == 'windows', f'Expected windows, got {platform}'
print('✓ Platform detection test passed!')
"@

try {
    $testScript | & python -c -
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Platform detection test passed!"
    }
    else {
        Write-Error "Platform detection test failed"
        exit 1
    }
}
catch {
    Write-Error "Platform detection test failed: $_"
    exit 1
}

# ───────────────────────────────────────────────────────────────────────────────
# COMPLETION
# ───────────────────────────────────────────────────────────────────────────────

Write-Header "Installation Complete!"

Write-Success "Ironcliw Platform Abstraction Layer installed successfully!"
Write-Host ""
Write-Info "Next steps:"
Write-Host "  1. Edit .env file and add your API keys (Claude, OpenAI, GCP)"
Write-Host "  2. Run: python unified_supervisor.py --test"
Write-Host "  3. If test passes, run: python unified_supervisor.py"
Write-Host ""
Write-Info "Current status: Phase 1 (Foundation & Platform Abstraction) complete"
Write-Info "Next phase: Windows Native Layer (C# DLLs)"
Write-Host ""
Write-Warning "Note: Full Ironcliw functionality requires completion of all 11 phases."
Write-Warning "See .zenflow/tasks/iron-cliw-0081/plan.md for the complete roadmap."
Write-Host ""

# Open .env in default editor if it's new
if (-not (Test-Path "$envPath.bak")) {
    $edit = Read-Host "Open .env file for editing? (y/N)"
    if ($edit -eq 'y') {
        Start-Process notepad $envPath
    }
}

Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
