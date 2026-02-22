#!/bin/bash
# JARVIS Linux Build Script
# Version: 1.0.0
# Platform: Ubuntu, Debian, Fedora, Arch Linux

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================"
echo "  JARVIS Linux Build Script"
echo "  Version 1.0.0"
echo "============================================"
echo ""

# Detect distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        DISTRO_VERSION=$VERSION_ID
        echo -e "${GREEN}[INFO]${NC} Detected: $PRETTY_NAME"
    else
        echo -e "${YELLOW}[WARNING]${NC} Cannot detect distribution"
        DISTRO="unknown"
    fi
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}[WARNING]${NC} Running as root is not recommended."
    echo "Please run as a regular user (sudo will be used when needed)."
    echo ""
fi

detect_distro
echo ""

# Check Python version
echo -e "${BLUE}[1/8]${NC} Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python 3 not found!"
    echo "Install Python 3.9+ for your distribution:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  Fedora:        sudo dnf install python3 python3-pip"
    echo "  Arch:          sudo pacman -S python python-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}[ERROR]${NC} Python 3.9+ required, found $PYTHON_VERSION"
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Python $PYTHON_VERSION detected"
echo ""

# Check system dependencies
echo -e "${BLUE}[2/8]${NC} Checking system dependencies..."

check_package() {
    local pkg=$1
    if command -v $pkg &> /dev/null || dpkg -s $pkg &> /dev/null 2>&1 || rpm -q $pkg &> /dev/null 2>&1 || pacman -Q $pkg &> /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $pkg"
        return 0
    else
        echo -e "  ${RED}✗${NC} $pkg (missing)"
        return 1
    fi
}

MISSING_DEPS=0

# Essential tools
check_package git || MISSING_DEPS=$((MISSING_DEPS + 1))
check_package gcc || check_package build-essential || MISSING_DEPS=$((MISSING_DEPS + 1))

# Audio dependencies
check_package espeak-ng || check_package espeak || MISSING_DEPS=$((MISSING_DEPS + 1))

# Window management (optional but recommended)
check_package wmctrl || echo -e "  ${YELLOW}○${NC} wmctrl (optional)"
check_package xdotool || echo -e "  ${YELLOW}○${NC} xdotool (optional)"

if [ $MISSING_DEPS -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}[WARNING]${NC} $MISSING_DEPS required dependencies are missing."
    echo ""
    echo "Install missing dependencies:"
    case "$DISTRO" in
        ubuntu|debian|linuxmint|pop)
            echo "  sudo apt install -y python3-pip python3-venv build-essential espeak-ng wmctrl xdotool"
            ;;
        fedora|rhel|centos)
            echo "  sudo dnf install -y python3-pip gcc gcc-c++ make espeak-ng wmctrl xdotool"
            ;;
        arch|manjaro)
            echo "  sudo pacman -S --needed python-pip base-devel espeak-ng wmctrl xdotool"
            ;;
        *)
            echo "  (Install commands vary by distribution)"
            ;;
    esac
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Create virtual environment
echo -e "${BLUE}[3/8]${NC} Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}[OK]${NC} Virtual environment created"
else
    echo -e "${GREEN}[OK]${NC} Virtual environment exists"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}[4/8]${NC} Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Failed to activate virtual environment!"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Virtual environment activated"
echo ""

# Upgrade pip
echo -e "${BLUE}[5/8]${NC} Upgrading pip..."
python -m pip install --upgrade pip --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[OK]${NC} pip upgraded"
else
    echo -e "${YELLOW}[WARNING]${NC} Failed to upgrade pip (non-critical)"
fi
echo ""

# Install Python dependencies
echo -e "${BLUE}[6/8]${NC} Installing Python dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${NC} Failed to install Python dependencies!"
    echo "Try running manually: pip install -r requirements.txt --verbose"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Python dependencies installed"
echo ""

# Check Node.js
echo -e "${BLUE}[7/8]${NC} Checking Node.js..."
SKIP_FRONTEND=0
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}[WARNING]${NC} Node.js not found!"
    echo "Frontend will not be built."
    echo "Install Node.js: https://nodejs.org/ or use your package manager"
    SKIP_FRONTEND=1
else
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}[OK]${NC} Node.js $NODE_VERSION detected"
fi
echo ""

# Install frontend dependencies
if [ $SKIP_FRONTEND -eq 0 ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${NC} Failed to install frontend dependencies!"
        cd ..
        exit 1
    fi
    cd ..
    echo -e "${GREEN}[OK]${NC} Frontend dependencies installed"
    echo ""
fi

# Create .env file
echo -e "${BLUE}[8/8]${NC} Checking configuration..."
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.platform.example .env
    echo ""
    echo -e "${YELLOW}[IMPORTANT]${NC} .env file created with default settings."
    echo "Please review and customize .env for your system."
    echo ""
else
    echo -e "${GREEN}[OK]${NC} .env file exists"
fi
echo ""

# Set execute permissions on scripts
chmod +x build_linux.sh
chmod +x verify_dependencies.py 2>/dev/null || true

# Verify installation
echo "============================================"
echo "  Running verification..."
echo "============================================"
echo ""
python verify_dependencies.py
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}[WARNING]${NC} Some dependencies failed verification."
    echo "JARVIS may not work correctly."
    echo "Check the output above for details."
    echo ""
else
    echo ""
    echo "============================================"
    echo "  Build Successful!"
    echo "============================================"
    echo ""
    echo -e "${GREEN}JARVIS is ready to run on Linux.${NC}"
    echo ""
    echo "To start JARVIS:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Run supervisor: python3 unified_supervisor.py"
    echo ""
    echo "See docs/setup/LINUX_SETUP.md for detailed instructions."
    echo ""
fi
