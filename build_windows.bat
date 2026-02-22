@echo off
REM JARVIS Windows Build Script
REM Version: 1.0.0
REM Platform: Windows 10/11

setlocal enabledelayedexpansion

echo ============================================
echo   JARVIS Windows Build Script
echo   Version 1.0.0
echo ============================================
echo.

REM Check if running in elevated mode
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: Not running as Administrator.
    echo Some features may require elevated privileges.
    echo.
)

REM Detect Python
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python not found in PATH!
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    exit /b 1
)

echo [1/7] Checking Python version...
python --version
echo.

REM Check Python version (must be 3.9+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% lss 3 (
    echo ERROR: Python 3.9+ required, found %PYTHON_VERSION%
    exit /b 1
)
if %PYTHON_MAJOR% equ 3 if %PYTHON_MINOR% lss 9 (
    echo ERROR: Python 3.9+ required, found %PYTHON_VERSION%
    exit /b 1
)

echo OK: Python %PYTHON_VERSION% detected
echo.

REM Check if virtual environment exists
echo [2/7] Checking virtual environment...
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorLevel% neq 0 (
        echo ERROR: Failed to create virtual environment!
        exit /b 1
    )
    echo OK: Virtual environment created
) else (
    echo OK: Virtual environment exists
)
echo.

REM Activate virtual environment
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorLevel% neq 0 (
    echo ERROR: Failed to activate virtual environment!
    exit /b 1
)
echo OK: Virtual environment activated
echo.

REM Upgrade pip
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
if %errorLevel% neq 0 (
    echo WARNING: Failed to upgrade pip (non-critical)
) else (
    echo OK: pip upgraded
)
echo.

REM Install Python dependencies
echo [5/7] Installing Python dependencies...
echo This may take several minutes...
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo ERROR: Failed to install Python dependencies!
    echo Try running: pip install -r requirements.txt --use-pep517
    exit /b 1
)
echo OK: Python dependencies installed
echo.

REM Check if Node.js is installed
echo [6/7] Checking Node.js...
node --version >nul 2>&1
if %errorLevel% neq 0 (
    echo WARNING: Node.js not found!
    echo Frontend will not be built.
    echo Install Node.js from https://nodejs.org/
    set SKIP_FRONTEND=1
) else (
    node --version
    echo OK: Node.js detected
    set SKIP_FRONTEND=0
)
echo.

REM Install frontend dependencies
if "%SKIP_FRONTEND%"=="0" (
    echo Installing frontend dependencies...
    cd frontend
    npm install
    if %errorLevel% neq 0 (
        echo ERROR: Failed to install frontend dependencies!
        cd ..
        exit /b 1
    )
    cd ..
    echo OK: Frontend dependencies installed
    echo.
)

REM Create .env file if it doesn't exist
echo [7/7] Checking configuration...
if not exist ".env" (
    echo Creating .env from template...
    copy .env.platform.example .env
    echo.
    echo IMPORTANT: .env file created with default settings.
    echo Please review and customize .env for your system.
    echo.
) else (
    echo OK: .env file exists
)
echo.

REM Verify installation
echo ============================================
echo   Running verification...
echo ============================================
echo.
python verify_dependencies.py
if %errorLevel% neq 0 (
    echo.
    echo WARNING: Some dependencies failed verification.
    echo JARVIS may not work correctly.
    echo Check the output above for details.
    echo.
) else (
    echo.
    echo ============================================
    echo   Build Successful!
    echo ============================================
    echo.
    echo JARVIS is ready to run on Windows.
    echo.
    echo To start JARVIS:
    echo   1. Activate virtual environment: venv\Scripts\activate
    echo   2. Run supervisor: python unified_supervisor.py
    echo.
    echo See docs\setup\WINDOWS_SETUP.md for detailed instructions.
    echo.
)

endlocal
pause
