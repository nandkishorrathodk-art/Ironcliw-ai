#!/bin/bash
# Run Ironcliw with proper environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/backend:$PYTHONPATH"

# Start Ironcliw
cd "$SCRIPT_DIR"
python start_system.py "$@"
