#!/bin/bash
###############################################################################
# Ironcliw Environment Setup Script
# ================================
# 
# This script sets up the complete Python environment for Ironcliw with all
# required dependencies for voice recognition, ML models, and async operations.
#
# Usage:
#   ./setup_environment.sh           # Full setup
#   ./setup_environment.sh --quick   # Quick install (skip optional deps)
#   ./setup_environment.sh --check   # Just check environment status
#
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_DIR="$SCRIPT_DIR/venv"

echo -e "${BOLD}${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                 🤖 Ironcliw Environment Setup                      ║"
echo "║           Voice Recognition & AI Assistant System                ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parse arguments
QUICK_MODE=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick   Skip optional dependencies for faster setup"
            echo "  --check   Only check environment status"
            echo "  --help    Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python module
python_has_module() {
    "$1" -c "import $2" 2>/dev/null
}

# Check environment status
check_environment() {
    echo -e "${CYAN}📋 Checking Environment Status...${NC}"
    echo ""
    
    local all_ok=true
    
    # Check Python
    if command_exists python3; then
        local py_version=$(python3 --version 2>&1)
        echo -e "  ${GREEN}✓${NC} Python: $py_version"
    else
        echo -e "  ${RED}✗${NC} Python 3 not found"
        all_ok=false
    fi
    
    # Check virtual environment
    if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python" ]; then
        echo -e "  ${GREEN}✓${NC} Virtual environment: $VENV_DIR"
        PYTHON_CMD="$VENV_DIR/bin/python"
    else
        echo -e "  ${RED}✗${NC} Virtual environment not found"
        all_ok=false
        PYTHON_CMD="python3"
    fi
    
    # Check critical dependencies
    echo ""
    echo -e "${CYAN}📦 Core Dependencies:${NC}"
    
    local deps=("numpy" "aiosqlite" "torch" "transformers" "speechbrain" "fastapi" "uvicorn")
    
    for dep in "${deps[@]}"; do
        if python_has_module "$PYTHON_CMD" "$dep"; then
            echo -e "  ${GREEN}✓${NC} $dep"
        else
            echo -e "  ${RED}✗${NC} $dep (not installed)"
            all_ok=false
        fi
    done
    
    # Check voice profile database
    echo ""
    echo -e "${CYAN}🔐 Voice Profile Database:${NC}"
    
    local db_path="$HOME/.jarvis/learning/jarvis_learning.db"
    if [ -f "$db_path" ]; then
        local db_size=$(du -h "$db_path" | cut -f1)
        echo -e "  ${GREEN}✓${NC} Database exists: $db_path ($db_size)"
        
        # Check for profiles
        local profile_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM speaker_profiles WHERE voiceprint_embedding IS NOT NULL" 2>/dev/null || echo "0")
        if [ "$profile_count" -gt 0 ]; then
            echo -e "  ${GREEN}✓${NC} Voice profiles: $profile_count"
        else
            echo -e "  ${YELLOW}⚠${NC} No voice profiles enrolled"
        fi
    else
        echo -e "  ${RED}✗${NC} Database not found: $db_path"
        all_ok=false
    fi
    
    echo ""
    if $all_ok; then
        echo -e "${GREEN}${BOLD}✅ Environment is ready!${NC}"
        return 0
    else
        echo -e "${YELLOW}${BOLD}⚠️  Environment needs setup. Run: ./setup_environment.sh${NC}"
        return 1
    fi
}

# If check only, just show status and exit
if $CHECK_ONLY; then
    check_environment
    exit $?
fi

# Step 1: Check Python 3
echo -e "${BLUE}Step 1: Checking Python installation...${NC}"

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    echo ""
    echo "Please install Python 3.9+ first:"
    echo "  brew install python@3.11"
    echo "  or visit: https://www.python.org/downloads/"
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓ Python $PY_VERSION found${NC}"

# Check minimum version
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    echo -e "${YELLOW}⚠️  Python 3.9+ is recommended. Current: $PY_VERSION${NC}"
fi

# Step 2: Create virtual environment
echo ""
echo -e "${BLUE}Step 2: Setting up virtual environment...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment exists. Recreating for clean install...${NC}"
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
echo -e "${GREEN}✓ Virtual environment created at $VENV_DIR${NC}"

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Step 3: Upgrade pip
echo ""
echo -e "${BLUE}Step 3: Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools
echo -e "${GREEN}✓ pip upgraded${NC}"

# Step 4: Install dependencies
echo ""
echo -e "${BLUE}Step 4: Installing dependencies...${NC}"
echo -e "${YELLOW}⏳ This may take 5-10 minutes for ML packages...${NC}"
echo ""

# Install core dependencies first
echo "Installing core dependencies..."
pip install numpy==1.24.3 aiosqlite==0.19.0

# Install PyTorch (CPU version for compatibility, or MPS for Apple Silicon)
echo "Installing PyTorch..."
if [[ $(uname -m) == 'arm64' ]]; then
    # Apple Silicon - use MPS-enabled PyTorch
    pip install torch==2.1.2 torchaudio==2.1.2
else
    # Intel Mac or other - CPU version
    pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
fi

# Install main requirements
if [ -f "$BACKEND_DIR/requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r "$BACKEND_DIR/requirements.txt" || {
        echo -e "${YELLOW}⚠️  Some packages failed. Trying essential packages only...${NC}"
        pip install fastapi uvicorn aiohttp websockets pydantic
        pip install transformers speechbrain
        pip install sounddevice soundfile pyaudio
        pip install faiss-cpu chromadb
    }
fi

# Install additional voice dependencies
if ! $QUICK_MODE; then
    echo "Installing voice/ML dependencies..."
    pip install openai-whisper 2>/dev/null || echo "Whisper install skipped"
    pip install speechbrain 2>/dev/null || echo "SpeechBrain install skipped"
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 5: Download models
echo ""
echo -e "${BLUE}Step 5: Downloading required models...${NC}"

# Download spaCy model
python -m spacy download en_core_web_sm 2>/dev/null || echo "spaCy model download skipped"

# Download NLTK data
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)
" 2>/dev/null || echo "NLTK data download skipped"

echo -e "${GREEN}✓ Models downloaded${NC}"

# Step 6: Create activation scripts
echo ""
echo -e "${BLUE}Step 6: Creating helper scripts...${NC}"

# Create activation script
cat > "$SCRIPT_DIR/activate_jarvis.sh" << 'EOF'
#!/bin/bash
# Activate Ironcliw virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"
echo "🤖 Ironcliw environment activated"
echo "   Python: $(which python)"
echo "   To start: python start_system.py"
EOF
chmod +x "$SCRIPT_DIR/activate_jarvis.sh"

# Create run script that activates venv first
cat > "$SCRIPT_DIR/run_jarvis.sh" << 'EOF'
#!/bin/bash
# Run Ironcliw with proper environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"

# Start Ironcliw
cd "$SCRIPT_DIR"
python start_system.py "$@"
EOF
chmod +x "$SCRIPT_DIR/run_jarvis.sh"

echo -e "${GREEN}✓ Helper scripts created${NC}"

# Step 7: Verify installation
echo ""
echo -e "${BLUE}Step 7: Verifying installation...${NC}"

check_environment

# Final instructions
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                    ✅ Setup Complete!                            ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}To start Ironcliw:${NC}"
echo ""
echo -e "  ${BOLD}Option 1:${NC} Use the run script (recommended)"
echo -e "    ${YELLOW}./run_jarvis.sh${NC}"
echo ""
echo -e "  ${BOLD}Option 2:${NC} Activate environment manually"
echo -e "    ${YELLOW}source activate_jarvis.sh${NC}"
echo -e "    ${YELLOW}python start_system.py${NC}"
echo ""
echo -e "${CYAN}To test speaker verification:${NC}"
echo -e "    ${YELLOW}source activate_jarvis.sh${NC}"
echo -e "    ${YELLOW}cd backend/scripts && python test_speaker_verification.py${NC}"
echo ""
