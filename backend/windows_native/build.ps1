# Ironcliw Windows Native Layer Build Script
# This script builds all C# projects in Release mode

param(
    [switch]$Clean,
    [switch]$Test,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Ironcliw Windows Native Layer - Build Script" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check .NET SDK
Write-Host "Checking .NET SDK..." -ForegroundColor Yellow
try {
    $dotnetVersion = dotnet --version
    Write-Host "  ✅ .NET SDK $dotnetVersion found" -ForegroundColor Green
}
catch {
    Write-Host "  ❌ .NET SDK not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install .NET SDK 8.0+ from:" -ForegroundColor Red
    Write-Host "  https://dotnet.microsoft.com/download" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or use winget:" -ForegroundColor Yellow
    Write-Host "  winget install Microsoft.DotNet.SDK.8" -ForegroundColor Cyan
    exit 1
}

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Clean if requested
if ($Clean) {
    Write-Host ""
    Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
    dotnet clean -c Release
    Write-Host "  ✅ Clean complete" -ForegroundColor Green
}

# Restore NuGet packages
Write-Host ""
Write-Host "Restoring NuGet packages..." -ForegroundColor Yellow
dotnet restore
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Failed to restore packages" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ Packages restored" -ForegroundColor Green

# Build projects
Write-Host ""
Write-Host "Building C# projects..." -ForegroundColor Yellow

$projects = @(
    @{ Name = "SystemControl"; Path = "SystemControl\SystemControl.csproj"; Framework = "net8.0-windows" },
    @{ Name = "ScreenCapture"; Path = "ScreenCapture\ScreenCapture.csproj"; Framework = "net8.0-windows10.0.19041.0" },
    @{ Name = "AudioEngine"; Path = "AudioEngine\AudioEngine.csproj"; Framework = "net8.0-windows" }
)

$buildSuccess = $true
$buildResults = @()

foreach ($project in $projects) {
    Write-Host ""
    Write-Host "  Building $($project.Name)..." -ForegroundColor Cyan
    
    $buildArgs = @("build", $project.Path, "-c", "Release")
    if ($Verbose) {
        $buildArgs += "-v", "detailed"
    }
    
    & dotnet $buildArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✅ $($project.Name) built successfully" -ForegroundColor Green
        
        # Find DLL
        $dllPath = Join-Path $scriptDir "$($project.Name)\bin\Release\$($project.Framework)\$($project.Name).dll"
        if (Test-Path $dllPath) {
            $dllSize = (Get-Item $dllPath).Length
            Write-Host "    📦 DLL: $dllPath ($([Math]::Round($dllSize / 1KB, 2)) KB)" -ForegroundColor Gray
        }
        
        $buildResults += @{ Name = $project.Name; Success = $true }
    }
    else {
        Write-Host "    ❌ $($project.Name) build failed" -ForegroundColor Red
        $buildSuccess = $false
        $buildResults += @{ Name = $project.Name; Success = $false }
    }
}

# Summary
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Build Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

foreach ($result in $buildResults) {
    $status = if ($result.Success) { "✅ PASSED" } else { "❌ FAILED" }
    $color = if ($result.Success) { "Green" } else { "Red" }
    Write-Host ("  {0,-20} {1}" -f $result.Name, $status) -ForegroundColor $color
}

Write-Host ""

if ($buildSuccess) {
    Write-Host "🎉 All projects built successfully!" -ForegroundColor Green
    
    # Test if requested
    if ($Test) {
        Write-Host ""
        Write-Host "================================================" -ForegroundColor Cyan
        Write-Host " Running Python Integration Tests" -ForegroundColor Cyan
        Write-Host "================================================" -ForegroundColor Cyan
        Write-Host ""
        
        # Check pythonnet
        $pythonnetCheck = python -c "import clr; print('OK')" 2>$null
        if ($pythonnetCheck -eq "OK") {
            Write-Host "  ✅ pythonnet is installed" -ForegroundColor Green
            Write-Host ""
            
            # Run test
            python test_csharp_bindings.py
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host ""
                Write-Host "🎉 All Python tests passed!" -ForegroundColor Green
            }
            else {
                Write-Host ""
                Write-Host "⚠️ Some Python tests failed" -ForegroundColor Red
            }
        }
        else {
            Write-Host "  ⚠️ pythonnet not found. Install with:" -ForegroundColor Yellow
            Write-Host "      pip install pythonnet" -ForegroundColor Cyan
        }
    }
    else {
        Write-Host ""
        Write-Host "To test Python integration, run:" -ForegroundColor Yellow
        Write-Host "  .\build.ps1 -Test" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Or manually:" -ForegroundColor Yellow
        Write-Host "  pip install pythonnet" -ForegroundColor Cyan
        Write-Host "  python test_csharp_bindings.py" -ForegroundColor Cyan
    }
}
else {
    Write-Host "❌ Some projects failed to build" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " Next Steps" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Test Python bindings:" -ForegroundColor Yellow
Write-Host "   python test_csharp_bindings.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Move to Phase 3 (Platform Implementations):" -ForegroundColor Yellow
Write-Host "   Implement backend/platform/windows/*.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "See README.md for full documentation" -ForegroundColor Gray
Write-Host ""
