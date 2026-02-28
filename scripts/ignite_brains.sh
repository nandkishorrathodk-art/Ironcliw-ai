#!/bin/bash
#===============================================================================
# Ironcliw Brain Ignition Sequence
#===============================================================================
#
# The "Defibrillator" script that ensures all LLM infrastructure is running
# before Ouroboros attempts any self-improvement operations.
#
# This script:
# 1. Checks if Ollama is running, starts it if not
# 2. Verifies required models are pulled
# 3. Checks if Ironcliw Prime is running
# 4. Waits for all endpoints to be healthy
# 5. Reports final status
#
# Usage:
#   ./scripts/ignite_brains.sh [--wait-only] [--timeout SECONDS]
#
# Author: Trinity System
# Version: 2.0.0
#===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
PRIME_PORT="${PRIME_PORT:-8000}"
TIMEOUT="${TIMEOUT:-120}"
POLL_INTERVAL=2
REQUIRED_MODELS=("deepseek-coder-v2" "codellama")

# Parse arguments
WAIT_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --wait-only)
            WAIT_ONLY=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

#===============================================================================
# UTILITY FUNCTIONS
#===============================================================================

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    command -v "$1" &> /dev/null
}

check_port() {
    local port=$1
    local host="${2:-localhost}"
    nc -z "$host" "$port" 2>/dev/null
}

http_check() {
    local url=$1
    local timeout=${2:-5}
    curl -s --connect-timeout "$timeout" -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000"
}

#===============================================================================
# OLLAMA MANAGEMENT
#===============================================================================

check_ollama_running() {
    local status=$(http_check "http://localhost:${OLLAMA_PORT}/api/tags")
    [[ "$status" == "200" ]]
}

start_ollama() {
    log_info "Starting Ollama service..."

    if ! check_command ollama; then
        log_error "Ollama is not installed. Please install it first:"
        log_info "  brew install ollama  # macOS"
        log_info "  curl -fsSL https://ollama.ai/install.sh | sh  # Linux"
        return 1
    fi

    # Start Ollama in background
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!

    # Wait for Ollama to start
    local attempts=0
    local max_attempts=30
    while ! check_ollama_running && [[ $attempts -lt $max_attempts ]]; do
        sleep 1
        ((attempts++))
        echo -n "."
    done
    echo ""

    if check_ollama_running; then
        log_success "Ollama started (PID: $OLLAMA_PID)"
        return 0
    else
        log_error "Failed to start Ollama after $max_attempts seconds"
        return 1
    fi
}

ensure_model_pulled() {
    local model=$1
    log_info "Checking model: $model"

    # Check if model exists
    local models=$(curl -s "http://localhost:${OLLAMA_PORT}/api/tags" 2>/dev/null | grep -o "\"name\":\"[^\"]*\"" | grep -c "$model" || echo "0")

    if [[ "$models" == "0" ]]; then
        log_warning "Model $model not found, pulling..."
        ollama pull "$model"
        if [[ $? -eq 0 ]]; then
            log_success "Model $model pulled successfully"
        else
            log_warning "Failed to pull $model (optional)"
        fi
    else
        log_success "Model $model is available"
    fi
}

#===============================================================================
# Ironcliw PRIME MANAGEMENT
#===============================================================================

check_prime_running() {
    local status=$(http_check "http://localhost:${PRIME_PORT}/v1/models")
    [[ "$status" == "200" ]]
}

start_prime() {
    log_info "Checking Ironcliw Prime..."

    # Look for Ironcliw Prime server script
    local prime_script=""
    local search_paths=(
        "$HOME/Documents/repos/Ironcliw-AI-Agent/backend/ai/prime_server.py"
        "$HOME/Documents/repos/Ironcliw-Prime/server.py"
        "$HOME/Documents/repos/Ironcliw-AI-Agent/prime_server.py"
    )

    for path in "${search_paths[@]}"; do
        if [[ -f "$path" ]]; then
            prime_script="$path"
            break
        fi
    done

    if [[ -z "$prime_script" ]]; then
        log_warning "Ironcliw Prime server script not found"
        log_info "Falling back to Ollama-only mode"
        return 1
    fi

    log_info "Starting Ironcliw Prime from: $prime_script"
    nohup python3 "$prime_script" > /tmp/jarvis_prime.log 2>&1 &
    PRIME_PID=$!

    # Wait for Prime to start
    local attempts=0
    local max_attempts=60
    while ! check_prime_running && [[ $attempts -lt $max_attempts ]]; do
        sleep 1
        ((attempts++))
        echo -n "."
    done
    echo ""

    if check_prime_running; then
        log_success "Ironcliw Prime started (PID: $PRIME_PID)"
        return 0
    else
        log_warning "Ironcliw Prime did not start - using Ollama fallback"
        return 1
    fi
}

#===============================================================================
# HEALTH CHECK LOOP
#===============================================================================

wait_for_brains() {
    local start_time=$(date +%s)
    local deadline=$((start_time + TIMEOUT))

    log_info "Waiting for intelligence infrastructure (timeout: ${TIMEOUT}s)..."

    local ollama_ready=false
    local prime_ready=false

    while [[ $(date +%s) -lt $deadline ]]; do
        # Check Ollama
        if check_ollama_running; then
            if ! $ollama_ready; then
                log_success "Ollama is ONLINE (port $OLLAMA_PORT)"
                ollama_ready=true
            fi
        fi

        # Check Prime
        if check_prime_running; then
            if ! $prime_ready; then
                log_success "Ironcliw Prime is ONLINE (port $PRIME_PORT)"
                prime_ready=true
            fi
        fi

        # At least one brain is needed
        if $ollama_ready || $prime_ready; then
            return 0
        fi

        sleep $POLL_INTERVAL
        echo -n "."
    done

    echo ""
    log_error "Timeout waiting for intelligence infrastructure"
    return 1
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================

main() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║         Ironcliw BRAIN IGNITION SEQUENCE v2.0                    ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Phase 1: Check/Start Ollama
    echo -e "${CYAN}━━━ Phase 1: Ollama Backend ━━━${NC}"
    if check_ollama_running; then
        log_success "Ollama already running"
    elif ! $WAIT_ONLY; then
        start_ollama || log_warning "Ollama startup failed"
    else
        log_warning "Ollama not running (wait-only mode)"
    fi

    # Phase 2: Ensure models are available
    if check_ollama_running && ! $WAIT_ONLY; then
        echo ""
        echo -e "${CYAN}━━━ Phase 2: Model Verification ━━━${NC}"
        for model in "${REQUIRED_MODELS[@]}"; do
            ensure_model_pulled "$model" || true
        done
    fi

    # Phase 3: Check/Start Ironcliw Prime
    echo ""
    echo -e "${CYAN}━━━ Phase 3: Ironcliw Prime ━━━${NC}"
    if check_prime_running; then
        log_success "Ironcliw Prime already running"
    elif ! $WAIT_ONLY; then
        start_prime || log_warning "Prime startup skipped"
    else
        log_warning "Prime not running (wait-only mode)"
    fi

    # Phase 4: Final Health Check
    echo ""
    echo -e "${CYAN}━━━ Phase 4: Health Verification ━━━${NC}"
    if wait_for_brains; then
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║              🧠 INTELLIGENCE INFRASTRUCTURE ONLINE 🧠          ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""

        # Print status summary
        echo -e "${CYAN}Status Summary:${NC}"
        if check_ollama_running; then
            echo -e "  ${GREEN}●${NC} Ollama:       http://localhost:${OLLAMA_PORT}"
        else
            echo -e "  ${RED}●${NC} Ollama:       OFFLINE"
        fi

        if check_prime_running; then
            echo -e "  ${GREEN}●${NC} Ironcliw Prime: http://localhost:${PRIME_PORT}"
        else
            echo -e "  ${YELLOW}●${NC} Ironcliw Prime: OFFLINE (using Ollama fallback)"
        fi
        echo ""

        return 0
    else
        echo ""
        echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║              ⚠️  INTELLIGENCE INFRASTRUCTURE FAILED ⚠️         ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        return 1
    fi
}

# Run main function
main "$@"
